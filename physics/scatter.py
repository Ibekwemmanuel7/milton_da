"""
Level two of the cloud-ice work: a differentiable two-stream ice-scattering operator for the microwave
channels, driven by a cloud-ice profile (or an ice water path spread over a fixed vertical shape).

The clear-sky part is exactly the weighting-function operator of AnalyticRTM (same audited biases). Each
layer between two state levels is given a gas transmittance t_l chosen so that, with no ice, the layer
contributions reproduce the channel's weighting function W_l exactly:

    contribution of layer l = (transmittance above l) x (1 - t_l) = W_l      =>  t_l = 1 - W_l / (1 - sum_{j<l} W_j)

Ice adds an optical depth tau_l = k_ext(f, IWC) q_l dp_l / g with single-scattering albedo omega and
asymmetry g from a bulk lookup computed at construction by Mie theory for soft ice spheres (Maxwell
Garnett ice-in-air mixing, density rho_s) over an exponential size distribution N(D) = N0 exp(-lambda D)
whose slope follows the ice water content. That is the classic first-order treatment (soft spheres are
known to misplace the 165 / 183 GHz depression against non-spherical habits by tens of percent); the
lookup is one table, so a DDA habit database (ARTS, Liu 2008) drops in by replacing `IceOptics`.

Each layer is solved with the two-stream (hemispheric-mean, delta-Eddington scaled) reflectance and
transmittance of Meador and Weaver (1980) with the thermal source of an isothermal layer (Kirchhoff:
emission = 1 - R - T times the layer's brightness temperature), and the layers are combined by the
adding method from the surface upward, which keeps everything element-wise and differentiable
(no linear solve). Rayleigh-Jeans: radiances are brightness temperatures throughout.

Shapes follow AnalyticRTM: temp [B, L, H, W], ciwc [B, L, H, W], output mw_tb [B, C_mw, h, w].
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ATMS_FREQ_GHZ
from .rtm import AnalyticRTM

G0 = 9.80665
RHO_ICE = 917.0


# ==============================================================================================
# Mie theory for one sphere (Bohren and Huffman 1983, numpy)
# ==============================================================================================
def _mie_q(m: complex, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extinction / scattering efficiencies and asymmetry parameter for size parameters x [N] and refractive index m."""
    x = np.asarray(x, np.float64)
    qext = np.zeros_like(x); qsca = np.zeros_like(x); gsca = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < 1e-6:
            continue
        nmax = int(xi + 4 * xi ** (1 / 3) + 2)
        mx = m * xi
        # logarithmic derivative D_n(mx) by downward recurrence
        nstart = nmax + 15
        D = np.zeros(nstart + 1, dtype=complex)
        for n in range(nstart, 0, -1):
            D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)
        # Riccati-Bessel psi, xi by upward recurrence
        psi0, psi1 = math.cos(xi), math.sin(xi)
        chi0, chi1 = -math.sin(xi), math.cos(xi)
        xi1 = complex(psi1, -chi1)
        an_prev = bn_prev = 0j
        qe = qs = 0.0; gsum = 0.0
        for n in range(1, nmax + 1):
            psi = (2 * n - 1) * psi1 / xi - psi0
            chi = (2 * n - 1) * chi1 / xi - chi0
            xin = complex(psi, -chi)
            dn = D[n]
            an = ((dn / m + n / xi) * psi - psi1) / ((dn / m + n / xi) * xin - xi1)
            bn = ((m * dn + n / xi) * psi - psi1) / ((m * dn + n / xi) * xin - xi1)
            qe += (2 * n + 1) * (an.real + bn.real)
            qs += (2 * n + 1) * (abs(an) ** 2 + abs(bn) ** 2)
            if n > 1:
                gsum += (n - 1) * (n + 1) / n * (an_prev * an.conjugate() + bn_prev * bn.conjugate()).real + (2 * n - 1) / ((n - 1) * n) * (an_prev * bn_prev.conjugate()).real
            an_prev, bn_prev = an, bn
            psi0, psi1, chi0, chi1, xi1 = psi1, psi, chi1, chi, xin
        qext[i] = 2.0 / xi**2 * qe
        qsca[i] = 2.0 / xi**2 * qs
        gsca[i] = 4.0 / (xi**2 * max(qsca[i], 1e-30)) * gsum
    return qext, qsca, np.clip(gsca, 0.0, 1.0)


def ice_permittivity(freq_ghz: float, temp_k: float = 250.0) -> complex:
    """Pure ice at microwave frequencies (Matzler 2006, as used in ARTS / RTTOV)."""
    eps_r = 3.1884 + 9.1e-4 * (temp_k - 273.0)
    theta = 300.0 / temp_k - 1.0
    alpha = (0.00504 + 0.0062 * theta) * math.exp(-22.1 * theta)
    beta = 0.0207 / temp_k * math.exp(335.0 / temp_k) / (math.exp(335.0 / temp_k) - 1.0) ** 2 + 1.16e-11 * freq_ghz**2 \
        + math.exp(-9.963 + 0.0372 * (temp_k - 273.16))
    eps_i = alpha / freq_ghz + beta * freq_ghz
    return complex(eps_r, eps_i)


def soft_sphere_index(freq_ghz: float, density: float, temp_k: float = 250.0) -> complex:
    """Maxwell Garnett mixing of ice inclusions in air for a snow particle of bulk density `density` (kg m-3)."""
    f = density / RHO_ICE
    e_i = ice_permittivity(freq_ghz, temp_k); e_a = 1.0 + 0j
    e = e_a * (1 + 2 * f * (e_i - e_a) / (e_i + 2 * e_a)) / (1 - f * (e_i - e_a) / (e_i + 2 * e_a))
    return np.sqrt(e)


class IceOptics:
    """Bulk optical properties per kg of ice, as a function of frequency and ice water content.

    Exponential PSD N(D) = N0 exp(-lam D) with mass m(D) = pi/6 rho_s D^3, so IWC = pi rho_s N0 / lam^4.
    Returns k_ext (m2 kg-1), single-scattering albedo, asymmetry on a log-spaced IWC grid."""

    def __init__(self, freqs_ghz: Sequence[float], density: float = 200.0, n0: float = 2.0e7, temp_k: float = 250.0,
                 iwc_grid: Optional[np.ndarray] = None, n_d: int = 60):
        self.freqs = list(freqs_ghz)
        self.iwc_grid = np.logspace(-6, -2, 41) if iwc_grid is None else np.asarray(iwc_grid)      # kg m-3 (1e-3 .. 10 g m-3)
        D = np.logspace(math.log10(5e-5), math.log10(1.5e-2), n_d)                                  # 50 um .. 15 mm
        dD = np.gradient(D)
        mass = math.pi / 6 * density * D**3
        kext = np.zeros((len(self.freqs), len(self.iwc_grid))); ssa = np.zeros_like(kext); asy = np.zeros_like(kext)
        for i, fq in enumerate(self.freqs):
            lam_m = 0.299792458 / fq                                            # wavelength (m)
            m = soft_sphere_index(fq, density, temp_k)
            qe, qs, g = _mie_q(m, math.pi * D / lam_m)
            area = math.pi / 4 * D**2
            for j, iwc in enumerate(self.iwc_grid):
                lam = (math.pi * density * n0 / iwc) ** 0.25
                N = n0 * np.exp(-lam * D) * dD
                iwc_chk = (mass * N).sum()
                ext = (qe * area * N).sum(); sca = (qs * area * N).sum()
                kext[i, j] = ext / max(iwc_chk, 1e-30)
                ssa[i, j] = sca / max(ext, 1e-30)
                asy[i, j] = (g * qs * area * N).sum() / max(sca, 1e-30)
        self.kext, self.ssa, self.asy = kext, np.clip(ssa, 0, 0.999), np.clip(asy, 0, 0.99)

    def tables(self) -> Dict[str, torch.Tensor]:
        return {"log_iwc": torch.tensor(np.log(self.iwc_grid), dtype=torch.float32),
                "kext": torch.tensor(self.kext, dtype=torch.float32), "ssa": torch.tensor(self.ssa, dtype=torch.float32),
                "asy": torch.tensor(self.asy, dtype=torch.float32)}


def _interp_rows(table: torch.Tensor, grid: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of table [C, G] at x [B, L, h, w] (same for every channel) -> [B, C, L, h, w]."""
    xc = x.clamp(grid[0], grid[-1])
    idx = torch.searchsorted(grid, xc.reshape(-1).contiguous()).clamp(1, grid.numel() - 1)
    x0, x1 = grid[idx - 1], grid[idx]
    w = ((xc.reshape(-1) - x0) / (x1 - x0)).clamp(0, 1)                     # [N]
    t0, t1 = table[:, idx - 1], table[:, idx]                                # [C, N]
    out = t0 + (t1 - t0) * w[None]
    return out.view(table.shape[0], *x.shape).permute(1, 0, 2, 3, 4)


# ==============================================================================================
# Two-stream layers and the adding method
# ==============================================================================================
def two_stream_layer(tau: torch.Tensor, omega: torch.Tensor, g: torch.Tensor, D: float = 1.66) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diffuse reflectance and transmittance of a homogeneous layer (Meador and Weaver 1980, hemispheric-mean
    coefficients with diffusivity D, after delta-Eddington scaling). All tensors broadcast."""
    f = g * g
    tau_s = (1.0 - omega * f) * tau
    omega_s = (1.0 - f) * omega / (1.0 - omega * f).clamp(min=1e-6)
    g_s = (g - f) / (1.0 - f).clamp(min=1e-6)
    gamma1 = D * (1.0 - 0.5 * omega_s * (1.0 + g_s))
    gamma2 = D * 0.5 * omega_s * (1.0 - g_s)
    k = torch.sqrt((gamma1**2 - gamma2**2).clamp(min=1e-10))
    e = torch.exp(-k * tau_s)
    e2 = e * e
    den = (k + gamma1) + (k - gamma1) * e2
    R = gamma2 * (1.0 - e2) / den
    T = 2.0 * k * e / den
    return R.clamp(0, 1), T.clamp(0, 1)


def adding_upwelling(R: torch.Tensor, T: torch.Tensor, B: torch.Tensor, surf_tb: torch.Tensor, surf_refl: float = 0.0) -> torch.Tensor:
    """Combine layers [.., L, ...] (index 0 = top) from the surface upward. R, T, B share a shape [B, C, L, h, w];
    surf_tb [B, C, h, w] (or broadcastable). Returns the TOA upwelling brightness temperature [B, C, h, w]."""
    E = (1.0 - R - T).clamp(min=0) * B                                       # isothermal-layer emission, both directions
    u = (1.0 - surf_refl) * surf_tb                                          # upwelling at the top of the stack below
    r = torch.full_like(u, surf_refl)                                        # reflectance of the stack below
    L = R.shape[2]
    for l in range(L - 1, -1, -1):
        Rl, Tl, El = R[:, :, l], T[:, :, l], E[:, :, l]
        den = (1.0 - Rl * r).clamp(min=1e-6)
        d = (El + Rl * u) / den                                              # downward flux into the stack
        u_int = u + r * d                                                    # upward flux at the interface
        u = El + Tl * u_int
        r = Rl + Tl * Tl * r / den
    return u


# ==============================================================================================
# The operator
# ==============================================================================================
class ScatteringRTM(AnalyticRTM):
    """AnalyticRTM whose microwave channels are computed by two-stream ice scattering (adding method).

    With ice=None (no ice in the state) or a state that carries only IWP, the ice is spread vertically with a
    fixed normalised profile peaking near `iwp_profile_peak_hpa` (a stated assumption; the profile state
    removes it). With no ice at all the result equals AnalyticRTM's clear-sky weighting-function TB to
    floating-point precision (unit test)."""

    def __init__(self, levels_hpa: Sequence[int], ir_channels: Sequence[str], mw_channels: Sequence[int], mw_downscale: int,
                 density: float = 200.0, n0: float = 2.0e7, diffusivity: float = 1.66, iwp_profile_peak_hpa: float = 300.0,
                 iwp_profile_width: float = 0.35, freqs_ghz: Optional[Dict[int, float]] = None, **kw):
        super().__init__(levels_hpa, ir_channels, mw_channels, mw_downscale, **kw)
        freqs = freqs_ghz or ATMS_FREQ_GHZ
        optics = IceOptics([freqs[c] for c in mw_channels], density=density, n0=n0)
        for k, v in optics.tables().items():
            self.register_buffer("ice_" + k, v)
        self.D = diffusivity
        # gas layer transmittances from the weighting functions (exact clear-sky equivalence)
        W = self.mw_wf                                                      # [C, L], sums to 1
        above = torch.cat([torch.zeros_like(W[:, :1]), torch.cumsum(W, 1)[:, :-1]], 1)
        t_gas = (1.0 - W / (1.0 - above).clamp(min=1e-6)).clamp(1e-6, 1.0)
        self.register_buffer("gas_tau", -torch.log(t_gas) / self.D)          # [C, L], so that exp(-D tau) = t_gas
        p = torch.exp(self.lnp) * 100.0
        dp = torch.cat([(p[1] - p[0]).view(1), p[1:] - p[:-1]])            # Pa, layer above each level (top layer = first gap)
        self.register_buffer("dp", dp)
        prof = torch.exp(-0.5 * ((self.lnp - math.log(iwp_profile_peak_hpa)) / iwp_profile_width) ** 2)
        self.register_buffer("iwp_profile", prof / prof.sum())               # [L], fraction of IWP per level

    def ice_path_per_level(self, ice_c: Optional[Dict[str, torch.Tensor]], shape) -> torch.Tensor:
        """Ice water path per level [B, L, h, w] (kg m-2) on the coarse grid, from ciwc or from IWP and the fixed profile."""
        B, L, h, w = shape
        if ice_c is None:
            return torch.zeros(B, L, h, w, device=self.lnp.device)
        if "ciwc" in ice_c:
            return ice_c["ciwc"].clamp(min=0) * self.dp.view(1, -1, 1, 1) / G0
        return ice_c["iwp"].clamp(min=0) * self.iwp_profile.view(1, -1, 1, 1)

    def forward_mw(self, temp: torch.Tensor, precip: torch.Tensor, ice: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        f = self.mw_downscale
        temp_c = F.avg_pool2d(temp, f)                                       # [B, L, h, w]
        ice_c = self.coarsen_ice(ice)
        B, L, h, w = temp_c.shape
        path = self.ice_path_per_level(ice_c, temp_c.shape)                  # [B, L, h, w] kg m-2
        # layer ice water content for the optics lookup: path / layer depth, depth from hydrostatics
        p = torch.exp(self.lnp) * 100.0
        rho = p.view(1, -1, 1, 1) / (287.05 * temp_c.clamp(min=150.0))       # kg m-3
        dz = self.dp.view(1, -1, 1, 1) / (rho * G0)
        iwc = path / dz.clamp(min=1.0)
        log_iwc = torch.log(iwc.clamp(min=1e-9))
        kext = _interp_rows(self.ice_kext, self.ice_log_iwc, log_iwc)        # [B, C, L, h, w]
        ssa = _interp_rows(self.ice_ssa, self.ice_log_iwc, log_iwc)
        asy = _interp_rows(self.ice_asy, self.ice_log_iwc, log_iwc)
        tau_ice = kext * path[:, None]                                       # [B, C, L, h, w]
        tau_gas = self.gas_tau.view(1, -1, L, 1, 1).expand(B, -1, -1, h, w)
        tau = tau_gas + tau_ice
        omega = ssa * tau_ice / tau.clamp(min=1e-12)
        g = asy
        R, T = two_stream_layer(tau, omega, g, self.D)
        Bt = temp_c[:, None].expand(-1, self.gas_tau.shape[0], -1, -1, -1)  # layer brightness temperature = level temperature
        surf = temp_c[:, -1][:, None].expand(-1, self.gas_tau.shape[0], -1, -1)
        return adding_upwelling(R, T, Bt, surf)

    def clear_sky(self, temp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(temp[:, :1])
        return self.forward_ir(temp, zeros, None), AnalyticRTM.forward_mw(self, temp, zeros, None)
