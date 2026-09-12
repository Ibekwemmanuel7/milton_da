"""
End-to-end inference on Hurricane Milton: observations -> U-Net proxy -> physics-guided score-based
posterior sampling -> CF NetCDF analysis with ensemble mean, spread, and physics diagnostics.

Example
-------
python -m milton_da.inference.run_milton \
    --scenes  artifacts/scenes/MILTON_2024-10-08T1200.nc artifacts/scenes/MILTON_2024-10-09T0000.nc \
    --stats   artifacts/norm_stats.json \
    --unet    artifacts/checkpoints/unet.pt --score artifacts/checkpoints/score.pt \
    --out     artifacts/analysis/

Scenes are produced with `data.dataset.build_scene_cache()` from GOES-16 ABI CMIP files,
NOAA-20/SNPP ATMS L1B granules, ERA5 (labels for verification only) and IMERG, storm-centred with
`data.best_track.BestTrack`. At inference ERA5/IMERG are only used for verification metrics.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
import xarray as xr

from ..assimilation.guidance import JointLikelihood, Observations
from ..assimilation.sampler import GuidedScoreSampler, SamplerOutput
from ..config import PipelineConfig
from ..data.dataset import HurricaneSceneDataset, Normalizer, collate
from ..models.score_net import ScoreUNet
from ..models.sde import VPSDE
from ..models.unet_xattn import CrossAttentionUNet
from ..physics.constraints import hydrostatic_thickness, warm_core_anomaly
from ..physics.rtm import AnalyticRTM
from ..train.common import get_device, load_checkpoint

log = logging.getLogger("milton_da")


def _block_mean(a: np.ndarray, target_shape) -> np.ndarray:
    """Coarsen a 2D coordinate array to target_shape by block averaging (identity if shapes match)."""
    f = a.shape[0] // target_shape[0]
    if f <= 1:
        return a
    H, W = a.shape
    return a[: H - H % f, : W - W % f].reshape(H // f, f, W // f, f).mean((1, 3))


class RetrievalEngine:
    """Holds all trained components and turns a batch of co-registered observations into an analysis."""

    def __init__(self, cfg: PipelineConfig, norm: Normalizer, unet: CrossAttentionUNet, score_net: ScoreUNet, rtm: Optional[torch.nn.Module] = None, device=None):
        self.cfg, self.norm = cfg, norm
        self.device = device or get_device()
        self.unet = unet.to(self.device).eval()
        self.score_net = score_net.to(self.device).eval()
        d = cfg.data
        self.rtm = (rtm or AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)).to(self.device)
        self.likelihood = JointLikelihood(self.rtm, norm, d, cfg.guidance).to(self.device)
        self.sampler = GuidedScoreSampler(self.score_net, VPSDE(cfg.sde), self.likelihood, cfg.guidance)

    @classmethod
    def from_checkpoints(cls, cfg: PipelineConfig, stats_path: str, unet_ckpt: str, score_ckpt: str, device=None) -> "RetrievalEngine":
        norm = Normalizer.load(stats_path)
        # a checkpoint trained as an attention-encoder variant records its U-Net config; honour it
        uc = torch.load(unet_ckpt, map_location="cpu").get("extra", {}).get("unet_cfg")
        if uc:
            cfg.unet.pos_embed = bool(uc.get("pos_embed", False))
            cfg.unet.self_attn_levels = tuple(uc.get("self_attn_levels", ()))
            cfg.unet.mw_encoder = uc.get("mw_encoder", "grid")
            cfg.unet.graph_k = int(uc.get("graph_k", 16))
            cfg.unet.graph_rounds = int(uc.get("graph_rounds", 3))
        unet = CrossAttentionUNet(cfg.data, cfg.unet)
        load_checkpoint(unet_ckpt, unet)
        score = ScoreUNet(cfg.data, cfg.score)
        ckpt = torch.load(score_ckpt, map_location="cpu")
        score.load_state_dict(ckpt.get("ema", ckpt["model"]))     # prefer EMA weights
        return cls(cfg, norm, unet, score, device=device)

    @torch.no_grad()
    def proxy(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.unet(batch["ir"], batch["ir_mask"], batch["mw"], batch["mw_mask"], batch.get("mw_zen"))

    def analyse(self, batch: Dict[str, torch.Tensor], ensemble_size: Optional[int] = None, log_every: int = 50, proxy_no_mw: bool = False) -> tuple[SamplerOutput, Observations]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if proxy_no_mw:      # experiment: the proxy sees only the infrared; ATMS reaches the state through the physics likelihood alone
            pb = dict(batch); pb["mw"] = torch.zeros_like(batch["mw"]); pb["mw_mask"] = torch.zeros_like(batch["mw_mask"]); pb["mw_zen"] = torch.zeros_like(batch["mw_zen"])
            x_det = self.proxy(pb)
        else:
            x_det = self.proxy(batch)                                                                      # [B, L+1, H, W]
        obs = Observations(x_det, batch["ir_raw"], batch["ir_mask"], batch["mw_raw"], batch["mw_mask"])
        out = self.sampler.sample(obs, ensemble_size=ensemble_size, log_every=log_every, callback=lambda d: log.info(f"  step {d['step']:4d} ir_rmse={d['ir_rmse_K']:.2f}K mw_rmse={d['mw_rmse_K']:.2f}K"))
        return out, obs

    def to_dataset(self, out: SamplerOutput, obs: Observations, scene: xr.Dataset, truth: Optional[torch.Tensor] = None) -> xr.Dataset:
        """Package member 0 of the batch as an analysis Dataset in physical units."""
        d = self.cfg.data
        with torch.no_grad():
            t_mem, p_mem = self.norm.state_to_physical(out.samples[:, 0], d.precip_log_transform)         # [N, L, H, W], [N, 1, H, W]
            t_mean, p_mean = t_mem.mean(0), p_mem.mean(0)
            t_std, p_std = t_mem.std(0), p_mem.std(0)
            t_det, p_det = self.norm.state_to_physical(obs.x_det[:1], d.precip_log_transform)
            ir_sim, mw_sim = self.rtm(t_mean[None], p_mean[None])
            thick = hydrostatic_thickness(t_mean[None], d.levels_hpa)[0]
            wc = warm_core_anomaly(t_mean[None])[0]
        ds = xr.Dataset(
            {
                "temperature": (("level", "y", "x"), t_mean.cpu().numpy(), {"units": "K", "long_name": "analysis ensemble mean temperature"}),
                "temperature_spread": (("level", "y", "x"), t_std.cpu().numpy(), {"units": "K"}),
                "precip": (("y", "x"), p_mean[0].cpu().numpy(), {"units": "mm h-1", "long_name": "analysis ensemble mean precipitation rate"}),
                "precip_spread": (("y", "x"), p_std[0].cpu().numpy(), {"units": "mm h-1"}),
                "temperature_unet": (("level", "y", "x"), t_det[0].cpu().numpy(), {"units": "K", "long_name": "deterministic U-Net proxy"}),
                "precip_unet": (("y", "x"), p_det[0, 0].cpu().numpy(), {"units": "mm h-1"}),
                "layer_thickness": (("layer", "y", "x"), thick.cpu().numpy(), {"units": "m", "long_name": "hypsometric thickness between consecutive levels"}),
                "warm_core_anomaly": (("level",), wc.cpu().numpy(), {"units": "K"}),
                "ir_tb_simulated": (("ir_channel", "y", "x"), ir_sim[0].cpu().numpy(), {"units": "K"}),
                "ir_tb_observed": (("ir_channel", "y", "x"), obs.ir_tb[0].cpu().numpy(), {"units": "K"}),
                "mw_tb_simulated": (("mw_channel", "yc", "xc"), mw_sim[0].cpu().numpy(), {"units": "K"}),
                "mw_tb_observed": (("mw_channel", "yc", "xc"), obs.mw_tb[0].cpu().numpy(), {"units": "K"}),
                "lat": (("y", "x"), _block_mean(scene["lat"].values, t_mean.shape[-2:])), "lon": (("y", "x"), _block_mean(scene["lon"].values, t_mean.shape[-2:])),
            },
            coords={"level": list(d.levels_hpa), "layer": np.arange(d.n_levels - 1), "ir_channel": list(d.ir_channels), "mw_channel": list(d.mw_channels)},
            attrs={**scene.attrs, "ensemble_size": int(out.samples.shape[0]), "n_steps": self.cfg.guidance.n_steps},
        )
        if truth is not None:
            t_true, p_true = self.norm.state_to_physical(truth[:1].to(self.device), d.precip_log_transform)
            ds["temperature_rmse_vs_era5"] = (("level",), torch.sqrt(((t_mean - t_true[0]) ** 2).mean((-2, -1))).cpu().numpy())
            ds["precip_rmse_vs_imerg"] = float(torch.sqrt(((p_mean - p_true[0]) ** 2).mean()))
        return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--unet", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ensemble", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--downscale", type=int, default=1, help="2 = run at 128 x 128 (use with checkpoints trained at --downscale 2)")
    ap.add_argument("--preset", choices=["small", "full"], default="full", help="must match the preset the checkpoints were trained with")
    ap.add_argument("--no-mw", action="store_true", help="ablation: zero the ATMS mask so the retrieval is infrared-only")
    ap.add_argument("--no-rtm", action="store_true", help="ablation: drop the radiative-transfer terms from the likelihood")
    ap.add_argument("--proxy-no-mw", action="store_true", help="experiment: hide ATMS from the U-Net proxy; the calibrated physics likelihood alone carries the microwave")
    ap.add_argument("--sigma-unet", type=float, default=None, help="trust in the U-Net proxy (normalised units, default 0.5); raise it to let the physics term compete")
    ap.add_argument("--rtm-audit", default=None, help="rtm_audit.json from colab/audit_rtm_cell.py: per-channel bias correction and error sigma")
    ap.add_argument("--audit-table", default="archive_2023", help="which table in the audit file to calibrate from")
    ap.add_argument("--ir-channels", nargs="*", default=[], help="IR channels kept in the physical likelihood (default none: the U-Net carries the IR)")
    ap.add_argument("--mw-channels", nargs="*", default=["5", "6", "7", "8", "9"], help="ATMS channels kept in the physical likelihood (default: O2 sounding channels)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = PipelineConfig()
    if args.preset == "small":
        from ..scripts.train import apply_preset
        apply_preset(cfg, "small")
    if args.steps:
        cfg.guidance.n_steps = args.steps
    if args.sigma_unet is not None:
        cfg.guidance.sigma_unet = args.sigma_unet
    if args.no_rtm:
        cfg.guidance.sigma_ir_K, cfg.guidance.sigma_mw_K = 1e6, 1e6
    engine = RetrievalEngine.from_checkpoints(cfg, args.stats, args.unet, args.score)
    if args.rtm_audit and not args.no_rtm:
        import json
        table = json.load(open(args.rtm_audit))[args.audit_table]
        engine.likelihood.calibrate_from_audit(table, use_ir=args.ir_channels, use_mw=args.mw_channels)
        lk = engine.likelihood
        log.info("calibrated RTM likelihood from %s [%s]: IR channels %s, MW channels %s (bias K %s, sigma K %s)",
                 args.rtm_audit, args.audit_table, args.ir_channels or "none", args.mw_channels,
                 [round(float(b), 1) for b, w in zip(lk.mw_bias, lk.mw_w) if w > 0],
                 [round(float(w) ** -0.5, 2) for w in lk.mw_w if w > 0])
    ds_obj = HurricaneSceneDataset(args.scenes, cfg.data, engine.norm, downscale=args.downscale)
    os.makedirs(args.out, exist_ok=True)
    for i in range(len(ds_obj)):
        scene = xr.load_dataset(args.scenes[i])
        batch = collate([ds_obj[i]])
        if args.no_mw:
            batch["mw_mask"].zero_(), batch["mw"].zero_(), batch["mw_zen"].zero_()
        out, obs = engine.analyse(batch, ensemble_size=args.ensemble, proxy_no_mw=args.proxy_no_mw)
        result = engine.to_dataset(out, obs, scene, truth=batch["state"])
        path = os.path.join(args.out, os.path.basename(args.scenes[i]).replace(".nc", "_analysis.nc"))
        result.to_netcdf(path)
        log.info(f"wrote {path}  warm core (K by level): {np.round(result['warm_core_anomaly'].values, 1)}")


if __name__ == "__main__":
    main()
