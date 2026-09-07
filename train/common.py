"""Shared training utilities: EMA, checkpointing, device/AMP handling, logging."""
from __future__ import annotations

import copy
import logging
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn

log = logging.getLogger("milton_da")


class EMA:
    """Exponential moving average of model weights (used for the score network at inference)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(self.decay).add_(pm.detach(), alpha=1.0 - self.decay)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, ema: Optional[EMA] = None, step: int = 0, extra: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {"model": model.state_dict(), "step": step, "extra": extra or {}}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.shadow.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, ema: Optional[EMA] = None, map_location="cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.shadow.load_state_dict(ckpt["ema"])
    return int(ckpt.get("step", 0))


class MeterLogger:
    def __init__(self, every: int):
        self.every, self.t0, self.acc, self.n = every, time.time(), {}, 0

    def add(self, d: Dict[str, float]) -> None:
        for k, v in d.items():
            self.acc[k] = self.acc.get(k, 0.0) + float(v)
        self.n += 1

    def maybe_log(self, step: int) -> None:
        if self.n and step % self.every == 0:
            msg = " ".join(f"{k}={v / self.n:.4f}" for k, v in self.acc.items())
            log.info(f"step {step} {msg} ({(time.time() - self.t0) / self.n * 1000:.0f} ms/it)")
            self.acc, self.n, self.t0 = {}, 0, time.time()
