"""
Stage 2: train the unconditional prior score model s_theta(x, t) on ERA5/IMERG states only.

Observations are *not* used here: the model learns what physically realised 3D temperature and
precipitation structures look like (warm cores, eyewall rings, stable stratification), so that
at inference the observations can only move the sample within that manifold via the likelihood.

Training data should include many tropical cyclone scenes (e.g. every named Atlantic / East
Pacific storm 2017-2023 at 3-hourly resolution, storm-centred from IBTrACS), with Milton held out.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import PipelineConfig
from ..data.dataset import collate
from ..models.score_net import ScoreUNet
from ..models.sde import VPSDE
from .common import EMA, MeterLogger, get_device, load_checkpoint, save_checkpoint

log = logging.getLogger("milton_da")


def train_score(cfg: PipelineConfig, train_ds: Dataset, max_steps: Optional[int] = None, device=None, model: Optional[ScoreUNet] = None, resume: bool = False, ckpt_every: int = 500) -> tuple[ScoreUNet, EMA]:
    """Train the prior. With resume=True an existing <ckpt_dir>/score.pt (model, optimiser, EMA, step)
    is loaded and training continues from its step, so a long CPU run can be interrupted and restarted."""
    device = device or get_device()
    tc = cfg.train
    model = (model or ScoreUNet(cfg.data, cfg.score)).to(device)
    sde = VPSDE(cfg.sde)
    ema = EMA(model, tc.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay, betas=(0.9, 0.99))
    loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, num_workers=tc.num_workers, collate_fn=collate, drop_last=True, pin_memory=device.type == "cuda")
    total = max_steps or tc.epochs * len(loader)
    warmup = max(1, min(1000, total // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    scaler = torch.amp.GradScaler(enabled=tc.amp and device.type == "cuda")
    meter, step = MeterLogger(tc.log_every), 0
    ckpt_path = os.path.join(tc.ckpt_dir, "score.pt")
    if resume and os.path.exists(ckpt_path):
        step = load_checkpoint(ckpt_path, model, opt, ema, map_location=device)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # fast-forwarding the LR schedule, not a real misuse
            for _ in range(step):
                sched.step()
        log.info(f"resumed score prior from {ckpt_path} at step {step}")
        if step >= total:
            return model, ema
    model.train()
    while step < total:
        for batch in loader:
            x0 = batch["state"].to(device, non_blocking=True)                  # [B, L+1, H, W]
            with torch.autocast(device_type=device.type, enabled=tc.amp and device.type == "cuda"):
                loss = sde.dsm_loss(model, x0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt), scaler.update(), sched.step()
            ema.update(model)
            meter.add({"dsm": loss.item()})
            step += 1
            meter.maybe_log(step)
            if step % ckpt_every == 0 or step == total:
                save_checkpoint(ckpt_path, model, opt, ema, step=step)
            if step >= total:
                break
    return model, ema


def train_rtm_residual(cfg: PipelineConfig, train_ds: Dataset, hybrid_rtm, norm, max_steps: int = 5000, device=None):
    """Optional Stage 0: fit the NeuralRTMResidual so HybridRTM matches observed TB on collocated
    ERA5/IMERG-ABI/ATMS samples. Only the residual's parameters are trained."""
    device = device or get_device()
    hybrid_rtm = hybrid_rtm.to(device)
    opt = torch.optim.AdamW(hybrid_rtm.residual.parameters(), lr=1e-3)
    loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    step, meter = 0, MeterLogger(cfg.train.log_every)
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            temp, precip = norm.state_to_physical(batch["state"], cfg.data.precip_log_transform)
            ir_sim, mw_sim = hybrid_rtm(temp, precip)
            l_ir = (((ir_sim - batch["ir_raw"]) ** 2) * batch["ir_mask"]).sum() / (batch["ir_mask"].sum() * ir_sim.shape[1]).clamp(min=1)
            l_mw = (((mw_sim - batch["mw_raw"]) ** 2) * batch["mw_mask"]).sum() / (batch["mw_mask"].sum() * mw_sim.shape[1]).clamp(min=1)
            loss = l_ir + l_mw
            opt.zero_grad(set_to_none=True), loss.backward(), opt.step()
            meter.add({"rtm_ir_mse": l_ir.item(), "rtm_mw_mse": l_mw.item()})
            step += 1
            meter.maybe_log(step)
            if step >= max_steps:
                break
    return hybrid_rtm
