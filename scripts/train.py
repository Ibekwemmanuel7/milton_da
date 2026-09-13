"""
Train the U-Net proxy and the diffusion prior on the scene archive, then sample the prior.

    python -m milton_da.scripts.train --archive data\\archive --out artifacts --preset small --downscale 2 \\
        --val-seasons 2023 --unet-steps 3000 --score-steps 20000 [--skip-unet] [--skip-score]

Split: scenes whose analysis year is in --val-seasons are validation; everything else is training.
The normaliser is fitted on the training split only and saved as <out>/norm_stats.json.
Presets: small (CPU / small GPU: U-Net 32 base, score 48 base), full (paper config).
--downscale 2 trains at 128 x 128 (fine) / 16 x 16 (coarse); the models are fully convolutional,
so checkpoints trained at 128 can be fine-tuned at 256 later.
Checkpoints: <out>/checkpoints/unet.pt and score.pt (EMA weights under "ema").
Prior samples: <out>/prior_samples.png, produced after score training with no observations at all;
you want to see eyewall rings and an upper-level warm core here before running guided retrievals.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch

from ..config import PipelineConfig, ScoreNetConfig, UNetConfig
from ..data.dataset import HurricaneSceneDataset, Normalizer
from ..models.sde import VPSDE
from ..physics.rtm import AnalyticRTM
from ..train.common import get_device
from ..train.train_score import train_score
from ..train.train_unet import train_unet

log = logging.getLogger("milton_da")


def split_scenes(archive: str, val_seasons):
    paths = sorted(glob.glob(os.path.join(archive, "scenes", "*.nc")))
    val = [p for p in paths if int(os.path.basename(p).split("_")[-1][:4]) in set(val_seasons)]
    train = [p for p in paths if p not in set(val)]
    return train, val


def apply_preset(cfg: PipelineConfig, preset: str) -> None:
    if preset == "small":
        cfg.unet = UNetConfig(base_channels=32, channel_mults=(1, 2, 4, 8), n_res_blocks=1, attn_heads=4, cross_attn_levels=(2, 3))
        cfg.score = ScoreNetConfig(base_channels=48, channel_mults=(1, 2, 4, 8), n_res_blocks=2, attn_heads=4, time_embed_dim=192, self_attn_levels=(3,))
        cfg.train.batch_size = 4


@torch.no_grad()
def sample_prior(score_net, cfg: PipelineConfig, norm: Normalizer, shape, out_png: str, n: int = 4, steps: int = 200, device=None):
    """Unconditional samples from the prior (no observations) as a sanity check.

    Always writes <out_png minus .png>.npz with the physical fields; the PNG is drawn only if
    matplotlib is installed, so a missing plotting library never kills a finished training run."""
    sde = VPSDE(cfg.sde)
    score_net = score_net.to(device).eval()
    x = torch.randn((n,) + tuple(shape), device=device)
    ts = torch.linspace(1.0, cfg.guidance.final_denoise_t, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(n)
        dt = float(ts[i + 1] - ts[i])
        score = sde.score_from_eps(score_net(x, t), t)
        drift, diffusion = sde.drift_diffusion(x, t)
        x = x + (drift - diffusion**2 * score) * dt + diffusion * (-dt) ** 0.5 * torch.randn_like(x)
    t = ts[-1].expand(n)
    x0 = sde.tweedie_x0(x, sde.score_from_eps(score_net(x, t), t), t)
    temp, precip = norm.state_to_physical(x0.cpu(), cfg.data.precip_log_transform)
    L = temp.shape[1]
    npz = os.path.splitext(out_png)[0] + ".npz"
    np.savez_compressed(npz, temperature=temp.numpy(), precip=precip[:, 0].numpy(), levels_hpa=np.asarray(cfg.data.levels_hpa))
    log.info(f"prior samples -> {npz}  (T range {temp.min():.1f}..{temp.max():.1f} K, rain max {precip.max():.1f} mm/h)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping the PNG (pip install matplotlib, then rerun with --sample-only to redraw)")
        return
    fig, ax = plt.subplots(3, n, figsize=(3.2 * n, 9))
    for j in range(n):
        im = ax[0, j].imshow(temp[j, L // 3].numpy(), cmap="RdYlBu_r"); ax[0, j].set_title(f"T {cfg.data.levels_hpa[L // 3]} hPa (K)"); plt.colorbar(im, ax=ax[0, j], fraction=0.046)
        im = ax[1, j].imshow(temp[j, -1].numpy(), cmap="RdYlBu_r"); ax[1, j].set_title("T 1000 hPa (K)"); plt.colorbar(im, ax=ax[1, j], fraction=0.046)
        im = ax[2, j].imshow(precip[j, 0].numpy(), cmap="Blues", vmax=30); ax[2, j].set_title("rain (mm/h)"); plt.colorbar(im, ax=ax[2, j], fraction=0.046)
        for a in ax[:, j]:
            a.set_xticks([]), a.set_yticks([])
    plt.suptitle("Unconditional samples from the diffusion prior (no observations)")
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()
    log.info(f"prior samples -> {out_png}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--preset", choices=["small", "full"], default="small")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--val-seasons", nargs="*", type=int, default=[2023])
    ap.add_argument("--unet-steps", type=int, default=3000)
    ap.add_argument("--score-steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--skip-unet", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--max-train-scenes", type=int, default=None, help="subsample the training split (quick experiments)")
    ap.add_argument("--lambda-rtm", type=float, default=None, help="U-Net RTM-consistency weight (default 0.1 from config; use 0.01 or 0 on real data, where the analytic RTM's representativeness error otherwise dominates the loss)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", action="store_true", help="continue the prior from <out>/checkpoints/score.pt (saved every 500 steps)")
    ap.add_argument("--sample-only", action="store_true", help="only draw prior samples from the existing score checkpoint")
    ap.add_argument("--unet-pos-embed", action="store_true", help="experiment: learned 2D positional embedding on the U-Net input features")
    ap.add_argument("--unet-self-attn-levels", nargs="*", type=int, default=None, help="experiment: U-Net levels given global self-attention over IR tokens, e.g. 1 2")
    ap.add_argument("--unet-ckpt-name", default="unet.pt", help="checkpoint file name for the U-Net (use a new name for an experiment so unet.pt is not overwritten)")
    ap.add_argument("--mw-encoder", choices=["grid", "graph"], default=None, help="experiment: microwave context encoder, convolutional (grid) or kNN message passing (graph)")
    ap.add_argument("--graph-k", type=int, default=None)
    ap.add_argument("--graph-rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="torch/numpy seed for a reproducible run")
    ap.add_argument("--ice", choices=["none", "iwp", "profile"], default="none", help="cloud ice in the state (scenes need 'iwp' / 'ciwc'; see scripts/add_cloud_ice.py). Changes the state size: train both networks and use a new --out or new checkpoint names")
    ap.add_argument("--score-ckpt-name", default="score.pt", help="checkpoint file name for the diffusion prior")
    args = ap.parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = PipelineConfig()
    apply_preset(cfg, args.preset)
    cfg.train.num_workers = args.num_workers
    cfg.train.ckpt_dir = os.path.join(args.out, "checkpoints")
    cfg.train.log_every = 20
    if args.batch:
        cfg.train.batch_size = args.batch
    if args.lambda_rtm is not None:
        cfg.train.lambda_rtm_consistency = args.lambda_rtm
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.unet_pos_embed:
        cfg.unet.pos_embed = True
    if args.unet_self_attn_levels is not None:
        cfg.unet.self_attn_levels = tuple(args.unet_self_attn_levels)
    if args.mw_encoder is not None:
        cfg.unet.mw_encoder = args.mw_encoder
    if args.graph_k is not None:
        cfg.unet.graph_k = args.graph_k
    if args.graph_rounds is not None:
        cfg.unet.graph_rounds = args.graph_rounds
    cfg.data.ice = args.ice
    if args.ice != "none":
        log.info(f"state carries cloud ice ({args.ice}); state channels = {cfg.data.state_channels}")
    log.info(f"U-Net variant: pos_embed={cfg.unet.pos_embed} self_attn_levels={tuple(cfg.unet.self_attn_levels)} mw_encoder={cfg.unet.mw_encoder} (k={cfg.unet.graph_k}, rounds={cfg.unet.graph_rounds}) -> checkpoints/{args.unet_ckpt_name}")
    device = get_device()
    log.info(f"device {device}; preset {args.preset}; downscale {args.downscale}")

    train_paths, val_paths = split_scenes(args.archive, args.val_seasons)
    if args.max_train_scenes:
        train_paths = list(np.random.default_rng(0).choice(train_paths, size=min(args.max_train_scenes, len(train_paths)), replace=False))
    log.info(f"scenes: {len(train_paths)} train, {len(val_paths)} val (seasons {args.val_seasons})")
    if not train_paths:
        log.error("no training scenes found"); return 1

    stats_path = os.path.join(args.out, "norm_stats.json")
    if os.path.exists(stats_path):
        norm = Normalizer.load(stats_path)
    else:
        log.info("fitting normaliser on the training split ...")
        norm = Normalizer.fit_paths(train_paths)
        norm.save(stats_path)
    log.info(f"normaliser -> {stats_path}")
    if args.ice != "none" and {"iwp": "iwp", "profile": "ciwc"}[args.ice] not in norm.stats:
        log.error(f"--ice {args.ice} needs '{ {'iwp': 'iwp', 'profile': 'ciwc'}[args.ice] }' statistics in {stats_path}; refit the normaliser on scenes that carry cloud ice (delete the file or use a new --out)"); return 1

    d = cfg.data
    train_ds = HurricaneSceneDataset(train_paths, d, norm, augment=True, downscale=args.downscale)
    val_ds = HurricaneSceneDataset(val_paths, d, norm, augment=False, downscale=args.downscale) if val_paths else None
    rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)

    sample_shape = train_ds[0]["state"].shape
    if args.sample_only:
        from ..models.score_net import ScoreUNet
        ckpt = torch.load(os.path.join(cfg.train.ckpt_dir, args.score_ckpt_name), map_location="cpu")
        net = ScoreUNet(cfg.data, cfg.score)
        net.load_state_dict(ckpt.get("ema", ckpt["model"]))
        log.info(f"sampling the prior from step {ckpt.get('step', '?')} checkpoint")
        sample_prior(net, cfg, norm, sample_shape, os.path.join(args.out, "prior_samples.png"), device=device)
        return 0
    if not args.skip_unet:
        log.info(f"stage 1: U-Net proxy for {args.unet_steps} steps")
        train_unet(cfg, train_ds, val_ds, norm, rtm, max_steps=args.unet_steps, device=device, ckpt_name=args.unet_ckpt_name)
    if not args.skip_score:
        log.info(f"stage 2: diffusion prior for {args.score_steps} steps")
        model, ema = train_score(cfg, train_ds, max_steps=args.score_steps, device=device, resume=args.resume, ckpt_name=args.score_ckpt_name)
        sample_prior(ema.shadow, cfg, norm, sample_shape, os.path.join(args.out, "prior_samples.png"), device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
