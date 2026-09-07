"""
Thermodynamic consistency penalties applied to generated 3D states (physical units).

Because the state holds temperature on fixed pressure levels, hydrostatic balance is not a
constraint on T alone: geopotential thickness follows *from* T through the hypsometric equation
(`hydrostatic_thickness`, provided as a diagnostic/derived product). What T on p-levels *can*
violate is static stability: a layer whose temperature falls with height faster than the dry
adiabatic rate is convectively unstable and does not persist in a resolved reanalysis state.
`static_stability_penalty` enforces  dT/d(ln p) <= (R_d / c_p) * T  =  kappa * T  in every layer.
"""
from __future__ import annotations

from typing import Sequence

import torch

R_D = 287.05       # J kg-1 K-1
C_P = 1004.6       # J kg-1 K-1
G0 = 9.80665       # m s-2
KAPPA = R_D / C_P  # 0.2857


def static_stability_penalty(temp: torch.Tensor, levels_hpa: Sequence[int], margin_K: float = 0.0) -> torch.Tensor:
    """Mean squared super-adiabatic excess. temp [B, L, H, W] K (levels top -> bottom). Returns scalar.

    lapse = (T_lower - T_upper) / (ln p_lower - ln p_upper)          [K per unit ln p]
    dry-adiabatic limit = kappa * T_mean of the layer
    penalty = mean( relu(lapse - kappa*T_mean - margin)^2 )
    """
    lnp = torch.log(torch.tensor(list(levels_hpa), dtype=temp.dtype, device=temp.device))
    dlnp = (lnp[1:] - lnp[:-1]).view(1, -1, 1, 1)                     # > 0
    dT = temp[:, 1:] - temp[:, :-1]                                    # [B, L-1, H, W]
    lapse = dT / dlnp
    limit = KAPPA * 0.5 * (temp[:, 1:] + temp[:, :-1]) + margin_K
    return torch.relu(lapse - limit).pow(2).mean()


def precip_nonneg_penalty(precip: torch.Tensor) -> torch.Tensor:
    """Quadratic penalty on negative rain rates. precip [B,1,H,W]."""
    return torch.relu(-precip).pow(2).mean()


def hydrostatic_thickness(temp: torch.Tensor, levels_hpa: Sequence[int]) -> torch.Tensor:
    """Hypsometric layer thickness (m) between consecutive levels: dz = R_d/g * T_mean * d(ln p).
    temp [B, L, H, W] -> [B, L-1, H, W]. Used to derive geopotential height / warm-core anomaly."""
    lnp = torch.log(torch.tensor(list(levels_hpa), dtype=temp.dtype, device=temp.device))
    dlnp = (lnp[1:] - lnp[:-1]).view(1, -1, 1, 1)
    tmean = 0.5 * (temp[:, 1:] + temp[:, :-1])
    return R_D / G0 * tmean * dlnp


def warm_core_anomaly(temp: torch.Tensor, radius_px: int = 48) -> torch.Tensor:
    """Diagnostic: T(centre) - environmental mean at radius. temp [B,L,H,W] -> [B,L]."""
    B, L, H, W = temp.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=temp.device), torch.arange(W, device=temp.device), indexing="ij")
    r = torch.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
    env = (r > radius_px).float()
    core = temp[:, :, H // 2 - 2 : H // 2 + 3, W // 2 - 2 : W // 2 + 3].mean((-2, -1))
    env_mean = (temp * env).sum((-2, -1)) / env.sum().clamp(min=1)
    return core - env_mean
