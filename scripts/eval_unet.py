"""
Score one or more U-Net proxy checkpoints directly (no sampler): the honest baseline comparison.

    python -m milton_da.scripts.eval_unet --stats artifacts/norm_stats.json --preset small --downscale 2 \
        --archive data/archive --val-seasons 2023 --milton data/milton/scenes \
        --ckpt baseline=artifacts/checkpoints/unet.pt --ckpt attn=artifacts/checkpoints/unet_attn.pt

For each checkpoint it reports
  * validation season: temperature RMSE (K, all levels) and precipitation RMSE (mm/h), the same
    numbers train_unet logs at the end of training;
  * Milton, per scene: 300 hPa domain RMSE, 300 hPa inner-core RMSE (central 16 x 16 pixels of the
    128 grid, the box used on the dashboard), warm-core anomaly at 300 hPa, and whether the scene
    had an ATMS overpass; plus the means over the ATMS scenes and over all scored scenes.
Each checkpoint's own U-Net config (pos_embed, self_attn_levels) is read from the checkpoint, so a
baseline and a variant can be scored in one call. Runs on CPU in a few minutes; faster on a GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import PipelineConfig
from ..data.dataset import HurricaneSceneDataset, Normalizer, collate
from ..models.unet_xattn import CrossAttentionUNet
from ..train.common import get_device, load_checkpoint
from .train import apply_preset, split_scenes

log = logging.getLogger("milton_da")
CORE_HALF = 8   # central 16 x 16 pixels at the 128 grid (about 70 km at 4.4 km pixels); matches the dashboard


def build(cfg: PipelineConfig, path: str) -> CrossAttentionUNet:
    ck = torch.load(path, map_location="cpu")
    uc = ck.get("extra", {}).get("unet_cfg") or {}
    cfg.unet.pos_embed = bool(uc.get("pos_embed", False))
    cfg.unet.self_attn_levels = tuple(uc.get("self_attn_levels", ()))
    net = CrossAttentionUNet(cfg.data, cfg.unet)
    load_checkpoint(path, net)
    n = sum(p.numel() for p in net.parameters())
    log.info(f"{path}: step {ck.get('step', '?')}, pos_embed={cfg.unet.pos_embed}, self_attn_levels={cfg.unet.self_attn_levels}, {n/1e6:.2f} M params")
    return net.eval()


@torch.no_grad()
def predict(net, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    return net(batch["ir"], batch["ir_mask"], batch["mw"], batch["mw_mask"], batch.get("mw_zen")), batch


def warm_core(t: np.ndarray, radius_frac: float = 0.375) -> np.ndarray:
    L, H, W = t.shape
    yy, xx = np.mgrid[:H, :W]
    env = np.hypot(yy - H / 2, xx - W / 2) > radius_frac * H
    core = t[:, H // 2 - 2 : H // 2 + 3, W // 2 - 2 : W // 2 + 3].mean((-2, -1))
    return core - t[:, env].mean(-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--preset", choices=["small", "full"], default="small")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--archive", default=None, help="archive root; validation scenes are picked by --val-seasons")
    ap.add_argument("--val-seasons", nargs="*", type=int, default=[2023])
    ap.add_argument("--milton", default=None, help="folder of MILTON_*.nc scenes")
    ap.add_argument("--ckpt", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=None, help="optional JSON file for the numbers")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = get_device()
    norm = Normalizer.load(args.stats)
    results = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        cfg = PipelineConfig()
        apply_preset(cfg, args.preset)
        net = build(cfg, path).to(device)
        d = cfg.data
        lev = list(d.levels_hpa)
        k300 = lev.index(300)
        res = {}

        if args.archive:
            _, val_paths = split_scenes(args.archive, args.val_seasons)
            ds = HurricaneSceneDataset(val_paths, d, norm, augment=False, downscale=args.downscale)
            loader = DataLoader(ds, batch_size=args.batch, collate_fn=collate)
            se_t = se_p = n_t = n_p = 0.0
            for batch in loader:
                x_det, batch = predict(net, batch, device)
                t_hat, p_hat = norm.state_to_physical(x_det, d.precip_log_transform)
                t_true, p_true = norm.state_to_physical(batch["state"], d.precip_log_transform)
                se_t += float(((t_hat - t_true) ** 2).sum()); n_t += t_true.numel()
                se_p += float(((p_hat - p_true) ** 2).sum()); n_p += p_true.numel()
            res["val"] = {"scenes": len(ds), "temp_rmse_K": round((se_t / n_t) ** 0.5, 3), "precip_rmse_mmh": round((se_p / n_p) ** 0.5, 3)}
            log.info(f"[{name}] validation {args.val_seasons}: {res['val']}")

        if args.milton:
            paths = sorted(glob.glob(os.path.join(args.milton, "MILTON_*.nc")))
            ds = HurricaneSceneDataset(paths, d, norm, augment=False, downscale=args.downscale)
            rows = []
            for i, p in enumerate(paths):
                x_det, batch = predict(net, collate([ds[i]]), device)
                t_hat, _ = norm.state_to_physical(x_det, d.precip_log_transform)
                t_true, _ = norm.state_to_physical(batch["state"], d.precip_log_transform)
                th, tt = t_hat[0, k300].cpu().numpy(), t_true[0, k300].cpu().numpy()
                H = th.shape[0]; c = H // 2; s = slice(c - CORE_HALF, c + CORE_HALF)
                # ERA5 labels have gaps on two scenes: the domain score needs a complete label, the
                # inner-core score is taken over the finite part of the box (as on the dashboard)
                dom_ok = bool(np.isfinite(tt).all())
                core_ok = bool(np.isfinite(tt[s, s]).mean() > 0.5)
                dom = float(np.sqrt(np.mean((th - tt) ** 2))) if dom_ok else None
                core = float(np.sqrt(np.nanmean((th[s, s] - tt[s, s]) ** 2))) if core_ok else None
                has_mw = bool(float(batch["mw_mask"].sum()) > 0)
                wc = float(warm_core(t_hat[0].cpu().numpy())[k300])
                rows.append({"scene": os.path.basename(p)[7:20], "atms": has_mw, "labelled": dom_ok,
                             "domain_rmse_300_K": round(dom, 3) if dom_ok else None, "core_rmse_300_K": round(core, 3) if core_ok else None,
                             "warm_core_300_K": round(wc, 2)})
                log.info(f"[{name}] {rows[-1]}")
            sc = [r for r in rows if r["core_rmse_300_K"] is not None]
            atms = [r for r in sc if r["atms"]]
            dm = [r for r in rows if r["domain_rmse_300_K"] is not None]
            summ = {"n_core_scored": len(sc), "n_atms": len(atms), "n_domain_scored": len(dm),
                    "core_rmse_300_K_atms_mean": round(float(np.mean([r["core_rmse_300_K"] for r in atms])), 3) if atms else None,
                    "core_rmse_300_K_all_mean": round(float(np.mean([r["core_rmse_300_K"] for r in sc])), 3) if sc else None,
                    "domain_rmse_300_K_all_mean": round(float(np.mean([r["domain_rmse_300_K"] for r in dm])), 3) if dm else None}
            res["milton"] = {"scenes": rows, "summary": summ}
            log.info(f"[{name}] Milton summary: {summ}")
        results[name] = res

    if len(results) > 1:
        names = list(results)
        log.info("---- comparison (inner-core 300 hPa RMSE, ATMS scenes) ----")
        for n in names:
            m = results[n].get("milton", {}).get("summary", {})
            v = results[n].get("val", {})
            log.info(f"  {n:>12s}: val T {v.get('temp_rmse_K')} K, val P {v.get('precip_rmse_mmh')} mm/h | core(ATMS) {m.get('core_rmse_300_K_atms_mean')} K, core(all) {m.get('core_rmse_300_K_all_mean')} K, domain {m.get('domain_rmse_300_K_all_mean')} K")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        log.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
