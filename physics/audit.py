"""
Operator audit helpers: fit the ice-scattering depression of each microwave channel from data.

The analytic operator writes the all-sky brightness temperature as TB_clear(T) - a_k (1 - exp(-IWP / I0_k)).
Given collocated (IWP, depression) pairs, where depression = TB_clear(ERA5 T) - TB_observed, the two
parameters per channel are fitted by least squares: for a fixed I0 the amplitude a has the closed form
a = sum(d f) / sum(f^2) with f = 1 - exp(-IWP / I0), so I0 is found on a log grid and a follows.
A bias offset b (the clear-sky bias of the channel, the number the existing audit measures) is fitted
jointly, so the depression fit does not absorb it: d = b + a f.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def fit_scatter_depression(iwp: np.ndarray, depression: np.ndarray, i0_grid: Optional[np.ndarray] = None,
                           min_pairs: int = 200) -> Dict[str, float]:
    """iwp [N] kg m-2, depression [N] K (simulated clear-sky minus observed) -> {a_K, I0, bias_K, rmse_K,
    rmse_clear_K, n}. rmse_clear_K is the residual with no ice term (bias only), so the gain of the ice term
    is rmse_clear_K - rmse_K."""
    x = np.asarray(iwp, np.float64).ravel(); d = np.asarray(depression, np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(d)
    x, d = x[ok], d[ok]
    n = int(x.size)
    if n < min_pairs:
        return {"a_K": 0.0, "I0": 1.0, "bias_K": float(d.mean()) if n else 0.0, "rmse_K": float(d.std()) if n else float("nan"),
                "rmse_clear_K": float(d.std()) if n else float("nan"), "n": n, "fitted": False}
    if i0_grid is None:
        i0_grid = np.logspace(-2, 1, 61)                       # 0.01 .. 10 kg m-2
    best = None
    for i0 in i0_grid:
        f = 1.0 - np.exp(-x / i0)
        # joint least squares for [bias, a]
        A = np.stack([np.ones_like(f), f], 1)
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        r = d - A @ coef
        rmse = float(np.sqrt((r * r).mean()))
        if best is None or rmse < best[0]:
            best = (rmse, float(coef[1]), float(i0), float(coef[0]))
    rmse, a, i0, b = best
    return {"a_K": max(a, 0.0), "I0": i0, "bias_K": b, "rmse_K": rmse, "rmse_clear_K": float(d.std()), "n": n, "fitted": True}


def fit_all_channels(pairs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    """pairs: {channel: {"iwp": [N], "depression": [N]}} -> {channel: fit}."""
    return {str(ch): fit_scatter_depression(v["iwp"], v["depression"]) for ch, v in pairs.items()}
