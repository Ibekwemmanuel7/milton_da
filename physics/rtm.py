"""
Differentiable forward operators H(x): 3D atmospheric state -> top-of-atmosphere brightness temperatures.

Two operators are provided and can be combined:

AnalyticRTM
    A fast, fully differentiable, weighting-function radiative-transfer proxy.
    * Microwave O2-band sounding channels (ATMS 5-9): TB = sum_l W_k(l) T_l with Gaussian
      weighting functions in ln(p) whose peaks follow the well-known ATMS/AMSU-A clear-sky
      weighting-function heights. Cloud/precip scattering is added as a channel-dependent
      depression. Without ice in the state it is proportional to log1p(rain rate) (the original
      proxy); with ice water path in the state it is a saturating function of IWP,
      a_k (1 - exp(-IWP / I0_k)), whose per-channel a_k, I0_k are fitted by the operator audit
      (physics.audit.fit_scatter_depression). Ice, not rain, is what scatters at 89 to 183 GHz.
    * Infrared channels (ABI C08/C10/C13): a clear-sky weighting-function term blended with an
      opaque cloud-top term. Without ice in the state the cloud fraction and cloud-top pressure are
      smooth monotonic functions of the surface rain rate; with ice, the cloud emissivity is
      1 - exp(-k_ir IWP / mu) (k_ir = 3 Q_ext / (4 rho_ice r_eff), r_eff = 30 um by default) and the
      cloud top is the level where the ice optical depth from the top reaches one (profile state) or
      a monotonic function of IWP (IWP state). The cloud-top temperature is read from the temperature
      profile with a differentiable soft interpolation in ln(p).
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
from typing import Dict, Optional, Sequence, Tuple

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
# IWP-driven scattering depression, TB_clear - a_k (1 - exp(-IWP / I0_k)). Priors (K, kg m-2) before the
# audit fit: the 183 GHz and 165 GHz channels saturate at tens of K over deep ice, 89 GHz less, the O2
# sounding channels barely (5, 6 see a little through the low-peaking weighting functions).
MW_ICE_A_K: Dict[int, float] = {5: 3.0, 6: 1.5, 7: 0.5, 8: 0.0, 9: 0.0, 16: 40.0, 17: 70.0, 18: 60.0, 22: 45.0}
MW_ICE_I0: Dict[int, float] = {5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 16: 1.0, 17: 0.5, 18: 0.4, 22: 0.5}
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
        ir_ice_k_m2kg: float = 50.0,    # IR mass extinction of cloud ice: 3 Q_ext / (4 rho_ice r_eff), Q_ext 2, r_eff 30 um
        ir_mu: float = 0.85,            # cosine of the ABI view angle over the Gulf / Caribbean
        iwp_top_scale: float = 0.5,     # kg m-2 over which the cloud top rises to the tropopause (IWP state only)
    ):
        super().__init__()
        self.ir_ice_k, self.ir_mu, self.iwp_top_scale = ir_ice_k_m2kg, ir_mu, iwp_top_scale
        self.register_buffer("mw_ice_a", torch.tensor([MW_ICE_A_K[c] for c in mw_channels]))        # [C_mw] K
        self.register_buffer("mw_ice_i0", torch.tensor([MW_ICE_I0[c] for c in mw_channels]))        # [C_mw] kg m-2
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

    def set_scatter_fit(self, fit: Dict) -> None:
        """Load per-channel (a_K, I0) from the audit ({channel: {"a_K":..., "I0":...}}); channels absent keep the prior."""
        for i, ch in enumerate(self.mw_channels):
            f = fit.get(str(ch))
            if f:
                self.mw_ice_a[i] = float(f["a_K"]); self.mw_ice_i0[i] = float(f["I0"])

    def ice_cloud_top_lnp(self, ice: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ln(cloud-top pressure) [B,1,H,W] from ice. Profile: the level where the ice optical depth accumulated
        from the top reaches one (soft, differentiable). IWP only: monotone in IWP, like the rain version."""
        if "ciwc" in ice:
            ciwc = ice["ciwc"]                                                     # [B, L, H, W] kg kg-1
            p = torch.exp(self.lnp) * 100.0                                        # Pa
            dp = torch.cat([p[1:] - p[:-1], (p[-1] - p[-2]).view(1)])              # per-level layer thickness (Pa)
            tau = self.ir_ice_k * ciwc * dp.view(1, -1, 1, 1) / 9.80665 / self.ir_mu
            cum = torch.cumsum(tau, dim=1)                                         # top -> bottom
            # weight each level by how much of the optical depth interval [0, 1] it covers
            w = (cum.clamp(max=1.0) - torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], 1).clamp(max=1.0)).clamp(min=0)
            covered = w.sum(1, keepdim=True)                                       # 0 (no ice) .. 1 (opaque)
            lnp_mean = (w * self.lnp.view(1, -1, 1, 1)).sum(1, keepdim=True) / covered.clamp(min=1e-6)
            return covered * lnp_mean + (1.0 - covered) * self.ln_ct_max
        frac = 1.0 - torch.exp(-ice["iwp"] / self.iwp_top_scale)
        return self.ln_ct_max - (self.ln_ct_max - self.ln_ct_min) * frac

    # -- forward ------------------------------------------------------------------------------
    def forward_ir(self, temp: torch.Tensor, precip: torch.Tensor, ice: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """[B,L,H,W], [B,1,H,W] (, ice) -> ir_tb [B,C_ir,H,W]."""
        tb_clear = torch.einsum("cl,blhw->bchw", self.ir_wf, temp)               # clear-sky WF radiance
        if ice is None:
            cloud_frac = 1.0 - torch.exp(-precip / self.cloud_p0)               # [B,1,H,W]
            t_ct = self.soft_profile_sample(temp, self.cloud_top_lnp(precip))   # cloud-top temperature
        else:
            cloud_frac = 1.0 - torch.exp(-self.ir_ice_k * ice["iwp"] / self.ir_mu)   # ice-cloud emissivity
            t_ct = self.soft_profile_sample(temp, self.ice_cloud_top_lnp(ice))
        return cloud_frac * t_ct + (1.0 - cloud_frac) * tb_clear

    def mw_depression(self, precip_c: torch.Tensor, ice_c: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Scattering depression [B,C_mw,h,w] on the coarse grid, from ice water path when available."""
        if ice_c is None:
            return self.mw_scatter.view(1, -1, 1, 1) * torch.log1p(precip_c)
        return self.mw_ice_a.view(1, -1, 1, 1) * (1.0 - torch.exp(-ice_c["iwp"] / self.mw_ice_i0.view(1, -1, 1, 1)))

    def coarsen_ice(self, ice: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
        return None if ice is None else {k: F.avg_pool2d(v, self.mw_downscale) for k, v in ice.items()}

    def forward_mw(self, temp: torch.Tensor, precip: torch.Tensor, ice: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """[B,L,H,W], [B,1,H,W] (, ice) -> mw_tb [B,C_mw,h,w]; fields are footprint-averaged first."""
        f = self.mw_downscale
        temp_c = F.avg_pool2d(temp, f)                                           # [B, L, h, w]
        precip_c = F.avg_pool2d(precip, f)                                       # [B, 1, h, w]
        tb_clear = torch.einsum("cl,blhw->bchw", self.mw_wf, temp_c)
        return tb_clear - self.mw_depression(precip_c, self.coarsen_ice(ice))

    def clear_sky(self, temp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cloud-free simulation (no ice, no rain): the reference the all-sky cloud predictor is measured against."""
        zeros = torch.zeros_like(temp[:, :1])
        return self.forward_ir(temp, zeros, None), self.forward_mw(temp, zeros, None)

    def forward(self, temp: torch.Tensor, precip: torch.Tensor, ice: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_ir(temp, precip, ice), self.forward_mw(temp, precip, ice)


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

    def forward(self, temp: torch.Tensor, precip: torch.Tensor, ice: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ir_a, mw_a = self.analytic(temp, precip, ice)
        ir_r, mw_r = self.residual(temp, precip)
        return ir_a + ir_r, mw_a + mw_r

    def clear_sky(self, temp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.analytic.clear_sky(temp)
