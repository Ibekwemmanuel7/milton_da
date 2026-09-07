"""
Differentiable forward operators H(x): 3D atmospheric state -> top-of-atmosphere brightness temperatures.

Two operators are provided and can be combined:

AnalyticRTM
    A fast, fully differentiable, weighting-function radiative-transfer proxy.
    * Microwave O2-band sounding channels (ATMS 5-9): TB = sum_l W_k(l) T_l with Gaussian
      weighting functions in ln(p) whose peaks follow the well-known ATMS/AMSU-A clear-sky
      weighting-function heights. Cloud/precip scattering is added as a channel-dependent
      depression proportional to log1p(rain rate), strongest at 165/183 GHz.
    * Infrared channels (ABI C08/C10/C13): a clear-sky weighting-function term blended with an
      opaque cloud-top term. The cloud-top pressure is a smooth monotonic function of the surface
      rain rate (deep convection -> high, cold tops), and the cloud-top temperature is read from
      the temperature profile with a differentiable soft interpolation in ln(p).
    This captures the *structure* of the radiative constraint (upper-level warm core in the O2
    channels, cold overshooting tops in the IR window, scattering depression in the eyewall) and is
    intended as the physics prior in the score-guided loop. For operational fidelity replace or
    augment it with CRTM / RTTOV (both expose tangent-linear/adjoint operators) or with the learned
    residual below, trained on collocated ERA5 / IMERG / ABI / ATMS samples.

NeuralRTMResidual
    A small CNN emulator that learns the residual between AnalyticRTM and observed TB. HybridRTM =
    AnalyticRTM + NeuralRTMResidual is the recommended "differentiable proxy function" once trained.

All operators take *physical* units and return brightness temperatures in Kelvin.

Shapes
------
temp    [B, L, H, W]   K
precip  [B, 1, H, W]   mm h-1
ir_tb   [B, C_ir, H, W]  K     (GOES grid = target grid)
mw_tb   [B, C_mw, h, w]  K     (ATMS grid, h = H / mw_downscale)
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Approximate clear-sky weighting-function peak pressures (hPa). Sources: ATMS/AMSU-A channel
# characteristics (Weng et al. 2012; Goldberg et al. 2001). ATMS 9 (55.5 GHz) peaks near 150 hPa,
# above our top level, so it is clipped to 200 hPa and mostly constrains the tropopause layer.
MW_WF_PEAK_HPA: Dict[int, float] = {5: 950.0, 6: 700.0, 7: 400.0, 8: 250.0, 9: 200.0,
                                    16: 1000.0, 17: 900.0, 18: 700.0, 22: 400.0}
# Scattering depression coefficient a_k (K per unit log1p(mm/h)). Larger at higher frequency.
MW_SCATTER_K: Dict[int, float] = {5: 0.5, 6: 0.3, 7: 0.1, 8: 0.0, 9: 0.0, 16: 6.0, 17: 14.0, 18: 20.0, 22: 9.0}
# ABI clear-sky peaks: water-vapour channels sense the mid/upper troposphere; C13 is a window channel.
IR_WF_PEAK_HPA: Dict[str, float] = {"C08": 350.0, "C10": 550.0, "C13": 1000.0}
IR_WF_WIDTH_LNP: Dict[str, float] = {"C08": 0.45, "C10": 0.45, "C13": 0.08}


def _gaussian_wf(lnp: torch.Tensor, peak_hpa: float, width: float) -> torch.Tensor:
    """Normalised Gaussian weighting function over levels: lnp [L] -> w [L], sum(w) = 1."""
    w = torch.exp(-0.5 * ((lnp - math.log(peak_hpa)) / width) ** 2)
    return w / w.sum()


class AnalyticRTM(nn.Module):
    def __init__(
        self,
        levels_hpa: Sequence[int],
        ir_channels: Sequence[str],
        mw_channels: Sequence[int],
        mw_downscale: int,
        mw_wf_width: float = 0.45,
        cloud_p0_mmh: float = 0.5,      # rain rate at which the pixel is ~63 % cloud-covered (IR)
        cloud_p1_mmh: float = 4.0,      # rain rate scale over which cloud tops rise to the tropopause
        cloud_top_max_hpa: float = 950.0,
        cloud_top_min_hpa: float = 150.0,
        soft_interp_width: float = 0.12,
    ):
        super().__init__()
        self.mw_downscale = mw_downscale
        self.ir_channels, self.mw_channels = list(ir_channels), list(mw_channels)
        lnp = torch.log(torch.tensor(list(levels_hpa), dtype=torch.float32))   # [L]
        self.register_buffer("lnp", lnp)
        self.register_buffer("mw_wf", torch.stack([_gaussian_wf(lnp, MW_WF_PEAK_HPA[c], mw_wf_width) for c in mw_channels]))   # [C_mw, L]
        self.register_buffer("mw_scatter", torch.tensor([MW_SCATTER_K[c] for c in mw_channels]))                                  # [C_mw]
        self.register_buffer("ir_wf", torch.stack([_gaussian_wf(lnp, IR_WF_PEAK_HPA[c], IR_WF_WIDTH_LNP[c]) for c in ir_channels]))   # [C_ir, L]
        self.cloud_p0, self.cloud_p1 = cloud_p0_mmh, cloud_p1_mmh
        self.ln_ct_max, self.ln_ct_min = math.log(cloud_top_max_hpa), math.log(cloud_top_min_hpa)
        self.soft_w = soft_interp_width

    # -- helpers -------------------------------------------------------------------------------
    def cloud_top_lnp(self, precip: torch.Tensor) -> torch.Tensor:
        """precip [B,1,H,W] -> ln(cloud-top pressure) [B,1,H,W]; monotone decreasing in rain rate."""
        frac = 1.0 - torch.exp(-precip / self.cloud_p1)
        return self.ln_ct_max - (self.ln_ct_max - self.ln_ct_min) * frac

    def soft_profile_sample(self, temp: torch.Tensor, ln_p_target: torch.Tensor) -> torch.Tensor:
        """Differentiable interpolation of temp [B,L,H,W] at ln p = ln_p_target [B,1,H,W] -> [B,1,H,W]."""
        d = (self.lnp.view(1, -1, 1, 1) - ln_p_target) / self.soft_w        # [B, L, H, W]
        w = torch.softmax(-0.5 * d * d, dim=1)
        return (w * temp).sum(1, keepdim=True)

    # -- forward ------------------------------------------------------------------------------
    def forward_ir(self, temp: torch.Tensor, precip: torch.Tensor) -> torch.Tensor:
        """[B,L,H,W], [B,1,H,W] -> ir_tb [B,C_ir,H,W]."""
        tb_clear = torch.einsum("cl,blhw->bchw", self.ir_wf, temp)               # clear-sky WF radiance
        cloud_frac = 1.0 - torch.exp(-precip / self.cloud_p0)                   # [B,1,H,W]
        t_ct = self.soft_profile_sample(temp, self.cloud_top_lnp(precip))       # cloud-top temperature
        return cloud_frac * t_ct + (1.0 - cloud_frac) * tb_clear

    def forward_mw(self, temp: torch.Tensor, precip: torch.Tensor) -> torch.Tensor:
        """[B,L,H,W], [B,1,H,W] -> mw_tb [B,C_mw,h,w]; fields are footprint-averaged first."""
        f = self.mw_downscale
        temp_c = F.avg_pool2d(temp, f)                                           # [B, L, h, w]
        precip_c = F.avg_pool2d(precip, f)                                       # [B, 1, h, w]
        tb_clear = torch.einsum("cl,blhw->bchw", self.mw_wf, temp_c)
        depression = self.mw_scatter.view(1, -1, 1, 1) * torch.log1p(precip_c)
        return tb_clear - depression

    def forward(self, temp: torch.Tensor, precip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_ir(temp, precip), self.forward_mw(temp, precip)


class NeuralRTMResidual(nn.Module):
    """CNN emulator of the residual TB_obs - AnalyticRTM(x). Input: [temp/300, log1p(precip)] concatenated."""

    def __init__(self, n_levels: int, n_ir: int, n_mw: int, mw_downscale: int, width: int = 64):
        super().__init__()
        cin = n_levels + 1
        self.mw_downscale = mw_downscale

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GELU(), nn.Conv2d(co, co, 3, padding=1), nn.GELU())

        self.trunk = block(cin, width)
        self.ir_head = nn.Conv2d(width, n_ir, 1)
        self.mw_head = nn.Sequential(nn.AvgPool2d(mw_downscale), block(width, width), nn.Conv2d(width, n_mw, 1))
        nn.init.zeros_(self.ir_head.weight), nn.init.zeros_(self.ir_head.bias)
        nn.init.zeros_(self.mw_head[-1].weight), nn.init.zeros_(self.mw_head[-1].bias)

    def forward(self, temp: torch.Tensor, precip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(torch.cat([temp / 300.0, torch.log1p(precip)], 1))
        return self.ir_head(z), self.mw_head(z)


class HybridRTM(nn.Module):
    """H(x) = AnalyticRTM(x) + NeuralRTMResidual(x). Residual starts at zero, so untrained == analytic."""

    def __init__(self, analytic: AnalyticRTM, residual: NeuralRTMResidual):
        super().__init__()
        self.analytic, self.residual = analytic, residual

    def forward(self, temp: torch.Tensor, precip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ir_a, mw_a = self.analytic(temp, precip)
        ir_r, mw_r = self.residual(temp, precip)
        return ir_a + ir_r, mw_a + mw_r
