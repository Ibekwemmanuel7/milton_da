"""
Joint likelihood log p(y | x) used to guide the reverse diffusion (the data-assimilation cost).

For a candidate clean state x0_hat (normalised units) we evaluate

    log p(y | x0) = - ||x0 - x_det||^2 / (2 sigma_u^2)                                    (1) U-Net proxy
                    - sum_ir  m_ir  ||TB_ir_obs - H_ir(x0)||^2 / (2 sigma_ir^2)              (2) IR radiative constraint
                    - sum_mw  m_mw  ||TB_mw_obs - H_mw(x0)||^2 / (2 sigma_mw^2)              (2) MW radiative constraint
                    - lambda_s * static_stability_penalty(T(x0))                              (3) thermodynamics
                    - lambda_p * precip_nonneg_penalty(P(x0))

Term (1) is the "background" in DA language (except it comes from the learned multi-modal proxy,
not a forecast). Term (2) is the classical observation term J_o with H the differentiable RTM.
Term (3) plays the role of weak physical constraints (J_c). Every term is a *sum* over valid pixels
and channels (a proper log-likelihood of independent observations), which is what makes the
gradient commensurate with the prior score in the reverse SDE. Use `diagnostics()` for
per-observation (mean) values when logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import DataConfig, GuidanceConfig
from ..data.dataset import Normalizer
from ..physics.constraints import precip_nonneg_penalty, static_stability_penalty


@dataclass
class Observations:
    """Everything the likelihood needs for one batch. TB fields are in Kelvin (raw)."""
    x_det: torch.Tensor                    # [B, L+1, H, W]  U-Net proxy, normalised state units
    ir_tb: torch.Tensor                    # [B, C_ir, H, W] K
    ir_mask: torch.Tensor                  # [B, 1, H, W]
    mw_tb: torch.Tensor                    # [B, C_mw, h, w] K
    mw_mask: torch.Tensor                  # [B, 1, h, w]

    def to(self, device) -> "Observations":
        return Observations(*[t.to(device) for t in (self.x_det, self.ir_tb, self.ir_mask, self.mw_tb, self.mw_mask)])


class JointLikelihood(nn.Module):
    """Per-channel calibration (optional): the analytic RTM has large, nearly constant biases against
    real GOES/ATMS radiances (tens of K on the ATMS sounding channels, with only 1-3 K of scatter).
    `calibrate_from_audit` loads the bias and error std measured by colab/audit_rtm_cell.py, subtracts
    the bias, sets sigma per channel and drops channels the operator cannot represent (weight 0), so
    the physics term only speaks where the physics is trustworthy and the U-Net proxy carries the rest."""

    def __init__(self, rtm: nn.Module, normalizer: Normalizer, data_cfg: DataConfig, cfg: GuidanceConfig):
        super().__init__()
        self.rtm, self.norm, self.data_cfg, self.cfg = rtm, normalizer, data_cfg, cfg
        n_ir, n_mw = len(data_cfg.ir_channels), len(data_cfg.mw_channels)
        self.register_buffer("ir_bias", torch.zeros(n_ir))
        self.register_buffer("mw_bias", torch.zeros(n_mw))
        self.register_buffer("ir_w", torch.full((n_ir,), 1.0 / cfg.sigma_ir_K**2))     # 1/sigma^2 per channel
        self.register_buffer("mw_w", torch.full((n_mw,), 1.0 / cfg.sigma_mw_K**2))

    def calibrate_from_audit(self, audit: dict, use_ir=None, use_mw=None, min_sigma_K: float = 1.0) -> None:
        """audit: one table from rtm_audit.json ({channel: {bias_K, std_K, verdict}}). Channels not in
        use_ir / use_mw (None = keep the audit's 'usable' ones) get weight 0."""
        def pick(chs, use, bias, w):
            for i, ch in enumerate(chs):
                a = audit.get(str(ch))
                keep = (str(ch) in {str(u) for u in use}) if use is not None else (a is not None and a["verdict"] == "usable")
                if a is None or not keep:
                    w[i] = 0.0; continue
                bias[i] = a["bias_K"]
                w[i] = 1.0 / max(a["std_K"], min_sigma_K) ** 2
        pick(self.data_cfg.ir_channels, use_ir, self.ir_bias, self.ir_w)
        pick(self.data_cfg.mw_channels, use_mw, self.mw_bias, self.mw_w)

    def _residuals(self, ir_sim, mw_sim, obs: Observations):
        return ir_sim - self.ir_bias[None, :, None, None] - obs.ir_tb, mw_sim - self.mw_bias[None, :, None, None] - obs.mw_tb

    def terms(self, x0_hat: torch.Tensor, obs: Observations, r2: float = 0.0) -> Dict[str, torch.Tensor]:
        """x0_hat [B, L+1, H, W] (normalised) -> dict of per-sample *negative* log-likelihood terms, each [B].

        r2 is the variance of the clean state given the current noisy sample, sigma_t^2 / alpha_t^2 in
        normalised units (0 = x0_hat is exact). Following pseudo-inverse guidance (Song et al. 2023),
        every observation error is inflated by the part of that uncertainty the observation sees:
        sigma_unet^2 + r2 for the proxy, sigma_c^2 + r2 * s_T^2 (s_T = normaliser temperature std, K)
        for a radiance channel, and the penalties are scaled by 1 / (1 + r2). Early in the reverse
        chain (r2 ~ 1e4) the guidance therefore vanishes instead of acting on a meaningless Tweedie
        estimate, which is what made the original sampler diverge on real data."""
        c = self.cfg
        B = x0_hat.shape[0]
        temp, precip = self.norm.state_to_physical(x0_hat, self.data_cfg.precip_log_transform)      # K, mm/h
        ir_sim, mw_sim = self.rtm(temp, precip)                                                       # [B,C_ir,H,W], [B,C_mw,h,w]
        r_ir, r_mw = self._residuals(ir_sim, mw_sim, obs)
        sT2 = self._temp_var_K2()
        ir_w = 1.0 / (1.0 / self.ir_w.clamp(min=1e-12) + r2 * sT2) * (self.ir_w > 0)
        mw_w = 1.0 / (1.0 / self.mw_w.clamp(min=1e-12) + r2 * sT2) * (self.mw_w > 0)

        n_pix = float(x0_hat.shape[-2] * x0_hat.shape[-1])
        j_unet = 0.5 * ((x0_hat - obs.x_det) ** 2).sum(dim=(1, 2, 3)) / (c.sigma_unet**2 + r2)
        j_ir = 0.5 * ((r_ir**2) * ir_w[None, :, None, None] * obs.ir_mask).sum(dim=(1, 2, 3))
        j_mw = 0.5 * ((r_mw**2) * mw_w[None, :, None, None] * obs.mw_mask).sum(dim=(1, 2, 3))
        # Penalties are means per pixel; scale by pixel count so lambda is grid-size independent.
        pen = n_pix / (1.0 + r2)
        j_stab = torch.stack([static_stability_penalty(temp[i : i + 1], self.data_cfg.levels_hpa) for i in range(B)]) * c.lambda_stability * pen
        j_pnn = torch.stack([precip_nonneg_penalty(precip[i : i + 1]) for i in range(B)]) * c.lambda_precip_nonneg * pen
        return {"unet": j_unet, "ir": j_ir, "mw": j_mw, "stability": j_stab, "precip_nonneg": j_pnn}

    def _temp_var_K2(self) -> float:
        """Mean variance (K^2) of one normalised temperature unit, from the normaliser."""
        if not hasattr(self, "_sT2"):
            std = self.norm.stats.get("temp", {}).get("std")
            self._sT2 = float(sum(s**2 for s in std) / len(std)) if std else 9.0
        return self._sT2

    def neg_log_likelihood(self, x0_hat: torch.Tensor, obs: Observations, r2: float = 0.0) -> torch.Tensor:
        """Scalar sum over batch of -log p(y | x0_hat)."""
        return sum(v.sum() for v in self.terms(x0_hat, obs, r2).values())

    @torch.no_grad()
    def diagnostics(self, x0_hat: torch.Tensor, obs: Observations) -> Dict[str, float]:
        """Observation-space RMSE (K) and cost terms; for logging / validation."""
        temp, precip = self.norm.state_to_physical(x0_hat, self.data_cfg.precip_log_transform)
        ir_sim, mw_sim = self.rtm(temp, precip)
        r_ir, r_mw = self._residuals(ir_sim, mw_sim, obs)
        # RMSE over the *active* (weight > 0) channels after bias correction, i.e. what the guidance sees.
        a_ir, a_mw = (self.ir_w > 0).float(), (self.mw_w > 0).float()
        ir_rmse = torch.sqrt(((r_ir**2) * a_ir[None, :, None, None] * obs.ir_mask).sum() / (obs.ir_mask.sum() * a_ir.sum()).clamp(min=1))
        mw_rmse = torch.sqrt(((r_mw**2) * a_mw[None, :, None, None] * obs.mw_mask).sum() / (obs.mw_mask.sum() * a_mw.sum()).clamp(min=1))
        n_pix = float(x0_hat.shape[-2] * x0_hat.shape[-1])
        out = {k: float(v.mean()) / n_pix for k, v in self.terms(x0_hat, obs).items()}   # cost per pixel
        out.update({"ir_rmse_K": float(ir_rmse), "mw_rmse_K": float(mw_rmse)})
        return out
