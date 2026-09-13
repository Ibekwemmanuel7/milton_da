"""
Synthetic hurricane scenes for unit tests, CI, and pipeline dry-runs without any downloads.

An idealised tropical cyclone is generated analytically (warm core aloft, eyewall/rainband precip),
observations are produced by running the AnalyticRTM forward operator and adding sensor noise, and
the result is packaged exactly like `build_scene()` output so the rest of the pipeline is unaware.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import xarray as xr

from ..config import DataConfig
from ..physics.rtm import AnalyticRTM
from .coregistration import TargetGrid


def _standard_profile(levels_hpa) -> np.ndarray:
    """Tropical reference temperature (K) on the levels: ~ -6.5 K/km below 100 hPa."""
    p = np.asarray(levels_hpa, dtype=np.float64)
    z = 44330.0 * (1.0 - (p / 1013.25) ** 0.1903) / 1000.0   # km, ISA-like
    t = 300.0 - 6.5 * z
    return np.maximum(t, 200.0).astype(np.float32)


def make_synthetic_scene(cfg: DataConfig, seed: int = 0, center_lat: float = 22.0, center_lon: float = -90.0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    g = cfg.grid
    target = TargetGrid.storm_centred(center_lat, center_lon, g)
    coarse = target.coarsen(g.mw_downscale)
    H, W = target.shape
    L = cfg.n_levels

    # --- geometry (pixels) -----------------------------------------------------------------
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2 + rng.normal(0, 2), W / 2 + rng.normal(0, 2)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    theta = np.arctan2(yy - cy, xx - cx)
    rmw = rng.uniform(0.06, 0.12) * H                       # radius of max wind (pixels)

    # --- 3D temperature: reference + warm core peaking at 300-400 hPa ---------------------
    prof = _standard_profile(cfg.levels_hpa)                  # [L]
    lnp = np.log(np.asarray(cfg.levels_hpa, np.float32))
    core_shape = np.exp(-0.5 * ((lnp - np.log(350.0)) / 0.5) ** 2)         # vertical structure [L]
    amp = rng.uniform(8.0, 16.0)                                            # K, intense TC
    horiz = np.exp(-0.5 * (r / (1.5 * rmw)) ** 2)                           # [H, W]
    temp = prof[:, None, None] + amp * core_shape[:, None, None] * horiz[None]
    temp = (temp + rng.normal(0, 0.3, size=temp.shape)).astype(np.float32)  # small env. noise
    # eye: slightly cooler low levels (subsidence-dry, but keep simple) -> nothing extra

    # --- precipitation: eyewall ring + two spiral bands ----------------------------------
    eyewall = 30.0 * np.exp(-0.5 * ((r - rmw) / (0.25 * rmw)) ** 2)
    bands = 0.0
    for k in range(2):
        spiral = theta + 0.05 * r - k * np.pi
        bands = bands + 12.0 * np.exp(-0.5 * (np.mod(spiral, 2 * np.pi) - np.pi) ** 2 / 0.15) * (r > 1.3 * rmw) * np.exp(-r / (2.5 * H / 4))
    precip = (eyewall + bands) * (1.0 + 0.3 * rng.normal(size=(H, W))).clip(0, None)
    precip = precip.astype(np.float32).clip(0, 80)

    # --- cloud ice: an anvil over the eyewall and bands, peaking near 300 hPa -------------
    ice_shape = np.exp(-0.5 * ((lnp - np.log(300.0)) / 0.35) ** 2); ice_shape /= ice_shape.sum()      # [L], fraction of IWP per level
    iwp = (0.08 * precip * (1.0 + 0.5 * np.exp(-0.5 * (r / (2.0 * rmw)) ** 2))).astype(np.float32)     # kg m-2, ~2.4 at 30 mm/h
    p_pa = np.asarray(cfg.levels_hpa, np.float64) * 100.0
    dp = np.concatenate([[p_pa[1] - p_pa[0]], p_pa[1:] - p_pa[:-1]])
    ciwc = (iwp[None] * ice_shape[:, None, None] * 9.80665 / dp[:, None, None]).astype(np.float32)   # kg kg-1

    # --- observations from the forward operator + noise ---------------------------------
    rtm = AnalyticRTM(cfg.levels_hpa, cfg.ir_channels, cfg.mw_channels, g.mw_downscale)
    with torch.no_grad():
        ir_tb, mw_tb = rtm(torch.from_numpy(temp)[None], torch.from_numpy(precip)[None, None], {"iwp": torch.from_numpy(iwp)[None, None]})
    ir = ir_tb[0].numpy() + rng.normal(0, 1.0, ir_tb[0].shape).astype(np.float32)
    mw = mw_tb[0].numpy() + rng.normal(0, 0.7, mw_tb[0].shape).astype(np.float32)

    # partial ATMS swath coverage: a random straight swath edge
    h, w = coarse.shape
    yc, xc = np.mgrid[0:h, 0:w]
    ang = rng.uniform(0, np.pi)
    edge = (yc - h / 2) * np.cos(ang) + (xc - w / 2) * np.sin(ang)
    mw_mask = edge < rng.uniform(0.2, 1.0) * max(h, w)

    return xr.Dataset(
        {
            "ir": (("ir_channel", "y", "x"), ir.astype(np.float32)),
            "ir_mask": (("y", "x"), np.ones((H, W), bool)),
            "mw": (("mw_channel", "yc", "xc"), (mw * mw_mask).astype(np.float32)),
            "mw_mask": (("yc", "xc"), mw_mask),
            "temp": (("level", "y", "x"), temp.astype(np.float32)),
            "precip": (("y", "x"), precip),
            "ciwc": (("level", "y", "x"), ciwc), "iwp": (("y", "x"), iwp),
            "lat": (("y", "x"), target.lat2d), "lon": (("y", "x"), target.lon2d),
            "latc": (("yc", "xc"), coarse.lat2d), "lonc": (("yc", "xc"), coarse.lon2d),
        },
        coords={"ir_channel": list(cfg.ir_channels), "mw_channel": list(cfg.mw_channels), "level": list(cfg.levels_hpa)},
        attrs={"time": "synthetic", "storm_lat": center_lat, "storm_lon": center_lon, "storm_name": "SYNTHETIC"},
    )


def make_synthetic_scenes(cfg: DataConfig, n: int, seed: int = 0) -> List[xr.Dataset]:
    return [make_synthetic_scene(cfg, seed=seed + i) for i in range(n)]
