"""
First-order limb adjustment for cross-track microwave sounders.

Brightness temperature in the O2 and H2O sounding channels falls with satellite zenith angle because
the slant path lengthens and the weighting function rises into colder air (the "limb effect"),
typically 5 to 15 K between nadir and the swath edge for ATMS 52 to 57 GHz. A storm near the swath
edge is therefore seen through a large, geometry-driven gradient that can swamp the warm-core signal.

`limb_adjust` removes the scene-dependent part empirically, granule by granule: for each channel it
fits a low-order polynomial of TB against x = sec(zenith) - 1 using a robust binned-median estimate
over all valid footprints, then subtracts f(x) - f(0). This is a simplified version of the classical
regression-based limb adjustment (Goldberg et al. 2001). It removes the bulk of the effect without an
external coefficient file; the residual angle dependence is passed to the network as an input channel
(the zenith angle itself), so the model can learn the remainder.
"""
from __future__ import annotations

import numpy as np


def limb_adjust(tb: np.ndarray, zen_deg: np.ndarray, order: int = 2, n_bins: int = 12, min_per_bin: int = 20):
    """tb [C, ns, nf] (NaN = invalid), zen_deg [ns, nf] -> (tb_adjusted [C, ns, nf], coefficients [C, order+1]).

    Channels with too few valid footprints or too little angular range are returned unchanged.
    """
    x = 1.0 / np.cos(np.deg2rad(np.clip(zen_deg, 0.0, 70.0))) - 1.0        # 0 at nadir, ~1.9 at 70 deg
    out = tb.copy()
    coefs = np.zeros((tb.shape[0], order + 1), np.float64)
    for c in range(tb.shape[0]):
        v = np.isfinite(tb[c]) & np.isfinite(x)
        if v.sum() < 200 or np.nanmax(x[v]) - np.nanmin(x[v]) < 0.2:
            continue
        xs, ys = x[v], tb[c][v]
        edges = np.linspace(xs.min(), xs.max(), n_bins + 1)
        bx, by = [], []
        for i in range(n_bins):
            m = (xs >= edges[i]) & (xs < edges[i + 1]) if i < n_bins - 1 else (xs >= edges[i]) & (xs <= edges[i + 1])
            if m.sum() >= min_per_bin:
                bx.append(np.median(xs[m])), by.append(np.median(ys[m]))      # median: robust to rain-affected footprints
        if len(bx) <= order:
            continue
        p = np.polyfit(np.array(bx), np.array(by), order)
        coefs[c] = p
        out[c] = tb[c] - (np.polyval(p, x) - np.polyval(p, 0.0))
    return out, coefs
