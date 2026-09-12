"""
Stage 1: supervised training of the Cross-Attention U-Net deterministic proxy.

Loss = masked MSE on the normalised state (temperature levels + log-precip, precip channel
up-weighted) + lambda * RTM consistency (simulated TB from the prediction vs the observed TB),
which teaches the proxy to respect the radiative constraint before the generative stage.

Usage (library):
    train_unet(cfg, train_ds, val_ds, normalizer, rtm)
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import PipelineConfig
from ..data.dataset import Normalizer, collate
from ..models.unet_xattn import CrossAttentionUNet
from ..physics.rtm import AnalyticRTM
from .common import MeterLogger, get_device, save_checkpoint

log = logging.getLogger("milton_da")


def unet_loss(model: CrossAttentionUNet, batch: Dict[str, torch.Tensor], norm: Normalizer, rtm: Optional[AnalyticRTM], cfg: PipelineConfig) -> Dict[str, torch.Tensor]:
    """Returns dict with 'loss' and its components. All state tensors are normalised units."""
    x_det = model(batch["ir"], batch["ir_mask"], batch["mw"], batch["mw_mask"], batch.get("mw_zen"))       # [B, L+1, H, W]
    target = batch["state"]
    L = model.n_levels
    w = torch.ones(L + 1, device=x_det.device)
    w[L] = 3.0                                                                          # precip channel weight
    mse = (((x_det - target) ** 2).mean(dim=(0, 2, 3)) * w).sum() / w.sum()
    out = {"mse": mse}
    if rtm is not None and cfg.train.lambda_rtm_consistency > 0:
        temp, precip = norm.state_to_physical(x_det, cfg.data.precip_log_transform)
        ir_sim, mw_sim = rtm(temp, precip)
        ir_err = (((ir_sim - batch["ir_raw"]) ** 2) * batch["ir_mask"]).sum() / (batch["ir_mask"].sum() * ir_sim.shape[1]).clamp(min=1)
        mw_err = (((mw_sim - batch["mw_raw"]) ** 2) * batch["mw_mask"]).sum() / (batch["mw_mask"].sum() * mw_sim.shape[1]).clamp(min=1)
        rtm_term = (ir_err / cfg.guidance.sigma_ir_K**2 + mw_err / cfg.guidance.sigma_mw_K**2) * 0.5
        out["rtm"] = rtm_term
        out["loss"] = mse + cfg.train.lambda_rtm_consistency * rtm_term
    else:
        out["loss"] = mse
    return out


def train_unet(cfg: PipelineConfig, train_ds: Dataset, val_ds: Optional[Dataset], norm: Normalizer, rtm: Optional[AnalyticRTM] = None, max_steps: Optional[int] = None, device=None, ckpt_name: str = "unet.pt") -> CrossAttentionUNet:
    device = device or get_device()
    tc = cfg.train
    model = CrossAttentionUNet(cfg.data, cfg.unet).to(device)
    rtm = rtm.to(device) if rtm is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, num_workers=tc.num_workers, collate_fn=collate, drop_last=True, pin_memory=device.type == "cuda")
    total = max_steps or tc.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=tc.amp and device.type == "cuda")
    meter, step = MeterLogger(tc.log_every), 0
    model.train()
    while step < total:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=tc.amp and device.type == "cuda"):
                out = unet_loss(model, batch, norm, rtm, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt), scaler.update(), sched.step()
            meter.add({k: v.item() for k, v in out.items()})
            step += 1
            meter.maybe_log(step)
            if step % 1000 == 0 or step == total:
                save_checkpoint(os.path.join(tc.ckpt_dir, ckpt_name), model, opt, step=step, extra={"unet_cfg": asdict(cfg.unet)})
                if val_ds is not None:
                    log.info(f"val {validate_unet(model, val_ds, norm, rtm, cfg, device)}")
            if step >= total:
                break
    return model


@torch.no_grad()
def validate_unet(model, val_ds, norm, rtm, cfg, device) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, collate_fn=collate)
    acc: Dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = unet_loss(model, batch, norm, rtm, cfg)
        x_det = model(batch["ir"], batch["ir_mask"], batch["mw"], batch["mw_mask"], batch.get("mw_zen"))
        t_hat, p_hat = norm.state_to_physical(x_det, cfg.data.precip_log_transform)
        t_true, p_true = norm.state_to_physical(batch["state"], cfg.data.precip_log_transform)
        out["temp_rmse_K"] = torch.sqrt(((t_hat - t_true) ** 2).mean())
        out["precip_rmse_mmh"] = torch.sqrt(((p_hat - p_true) ** 2).mean())
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}
