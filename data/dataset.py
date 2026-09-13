"""
Xarray-backed PyTorch dataset for co-registered hurricane scenes.

Pipeline
--------
raw files (GOES nc, ATMS h5/nc, ERA5 nc, IMERG h5) --build_scene()--> one CF-style xarray.Dataset
per analysis time, persisted as NetCDF ("scene cache") --HurricaneSceneDataset--> normalised tensors.

Every scene Dataset has:
    ir      (ir_channel, y, x)      K          GOES brightness temperature on the target grid
    ir_mask (y, x)                  bool
    mw      (mw_channel, yc, xc)    K          ATMS brightness temperature on the coarse grid
    mw_mask (yc, xc)                bool       False outside the swath
    mw_zenith (yc, xc)              degree     satellite zenith angle (limb geometry), mw_landfrac (yc, xc)
    temp    (level, y, x)           K          ERA5 temperature on pressure levels (target labels)
    precip  (y, x)                  mm h-1     IMERG surface precipitation rate (target labels)
    lat/lon (y, x), latc/lonc (yc, xc)
    attrs: time, storm_lat, storm_lon, storm_name

Batch tensor format produced by the Dataset (all float32, normalised):
    ir       [C_ir, H, W]     ir_mask  [1, H, W]
    mw       [C_mw, h, w]     mw_mask  [1, h, w]     mw_zen [1, h, w] (zenith/60, 0 where invalid)
    state    [L+1, H, W]      (temperature levels then log1p precip)
    ir_raw   [C_ir, H, W]     mw_raw [C_mw, h, w]   (physical units, K; used by the RTM likelihood)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from ..config import DataConfig
from ..physics.limb import limb_adjust
from .coregistration import TargetGrid, block_mean, regrid_goes_to_target, regrid_latlon_to_target, regrid_swath_to_target


# ----------------------------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------------------------
class Normalizer:
    """Per-channel affine normalisation, fitted once on the training split and stored as JSON.

    Keys: "ir" [C_ir], "mw" [C_mw], "temp" [L], "precip" [1] (statistics of log1p(mm/h) if enabled).
    """

    def __init__(self, stats: Dict[str, Dict[str, List[float]]]):
        self.stats = stats
        self._t = {k: (torch.tensor(v["mean"], dtype=torch.float32), torch.tensor(v["std"], dtype=torch.float32)) for k, v in stats.items()}

    # -- persistence ---------------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def fit(cls, scenes: Sequence[xr.Dataset], log_precip: bool = True) -> "Normalizer":
        """Compute masked per-channel mean/std over a list of scene Datasets."""
        acc: Dict[str, List[np.ndarray]] = {"ir": [], "mw": [], "temp": [], "precip": []}
        for s in scenes:
            irm = s["ir_mask"].values
            acc["ir"].append(s["ir"].values[:, irm].reshape(s.sizes["ir_channel"], -1))
            mwm = s["mw_mask"].values
            if mwm.any():
                acc["mw"].append(s["mw"].values[:, mwm].reshape(s.sizes["mw_channel"], -1))
            acc["temp"].append(s["temp"].values.reshape(s.sizes["level"], -1))
            p = s["precip"].values.reshape(1, -1)
            acc["precip"].append(np.log1p(np.clip(p, 0, None)) if log_precip else p)
        stats = {}
        for k, chunks in acc.items():
            a = np.concatenate(chunks, axis=1)
            stats[k] = {"mean": a.mean(1).tolist(), "std": (a.std(1) + 1e-6).tolist()}
        return cls(stats)

    @classmethod
    def fit_paths(cls, paths: Sequence[str], log_precip: bool = True, max_scenes: int = 400, seed: int = 0) -> "Normalizer":
        """Fit on a random subset of scene files without holding them all in memory."""
        rng = np.random.default_rng(seed)
        sel = list(paths) if len(paths) <= max_scenes else list(rng.choice(list(paths), size=max_scenes, replace=False))
        sums: Dict[str, np.ndarray] = {}
        sqs: Dict[str, np.ndarray] = {}
        cnt: Dict[str, float] = {}
        for pth in sel:
            with xr.open_dataset(pth) as s:
                irm = s["ir_mask"].values
                chunks = {"ir": s["ir"].values[:, irm].reshape(s.sizes["ir_channel"], -1), "temp": s["temp"].values.reshape(s.sizes["level"], -1)}
                mwm = s["mw_mask"].values
                if mwm.any():
                    chunks["mw"] = s["mw"].values[:, mwm].reshape(s.sizes["mw_channel"], -1)
                pr = s["precip"].values.reshape(1, -1)
                chunks["precip"] = np.log1p(np.clip(pr, 0, None)) if log_precip else pr
            for k, a in chunks.items():
                a = a.astype(np.float64)
                sums[k] = sums.get(k, 0) + a.sum(1)
                sqs[k] = sqs.get(k, 0) + (a * a).sum(1)
                cnt[k] = cnt.get(k, 0) + a.shape[1]
        stats = {}
        for k in sums:
            mean = sums[k] / cnt[k]
            var = np.maximum(sqs[k] / cnt[k] - mean**2, 0)
            stats[k] = {"mean": mean.tolist(), "std": (np.sqrt(var) + 1e-6).tolist()}
        return cls(stats)

    # -- application (torch, broadcast over trailing spatial dims) -----------------------------
    def _mv(self, key: str, x: torch.Tensor):
        m, s = self._t[key]
        shape = (-1,) + (1,) * (x.ndim - 1) if x.ndim == 3 else (1, -1) + (1,) * (x.ndim - 2)
        return m.to(x.device).view(shape), s.to(x.device).view(shape)

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        m, s = self._mv(key, x)
        return (x - m) / s

    def denormalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        m, s = self._mv(key, x)
        return x * s + m

    # -- state vector helpers ------------------------------------------------------------------
    def state_to_physical(self, state: torch.Tensor, log_precip: bool = True):
        """state [B, L+1, H, W] (normalised) -> temp [B, L, H, W] K, precip [B, 1, H, W] mm/h.
        Differentiable; used inside the RTM likelihood."""
        L = state.shape[1] - 1
        temp = self.denormalize("temp", state[:, :L])
        p = self.denormalize("precip", state[:, L:])
        if log_precip:
            p = torch.expm1(p.clamp(max=12.0))
        return temp, p.clamp(min=0.0)

    def physical_to_state(self, temp: torch.Tensor, precip: torch.Tensor, log_precip: bool = True) -> torch.Tensor:
        p = torch.log1p(precip.clamp(min=0)) if log_precip else precip
        return torch.cat([self.normalize("temp", temp), self.normalize("precip", p)], dim=1)


# ----------------------------------------------------------------------------------------------
# Scene construction from raw sources
# ----------------------------------------------------------------------------------------------
@dataclass
class RawScenePaths:
    """File locations for a single analysis time. Any of the observation sources may be missing."""
    time: np.datetime64
    storm_lat: float
    storm_lon: float
    goes_files: Dict[str, str]              # {"C13": ".../OR_ABI-L2-CMIPF-M6C13_G16_....nc", ...}
    atms_files: List[str]                   # consecutive 6-min ATMS L1B granules covering the storm (may be empty)
    era5_file: str                          # pressure-level temperature for the analysis hour
    imerg_file: str                         # IMERG half-hourly precipitation
    storm_name: str = "MILTON"
    mw_dt_min: Optional[float] = None       # overpass time minus analysis time (minutes), for provenance


def _open_atms(paths: Sequence[str], channels: Sequence[int], limb_correct: bool = True):
    """Return lat [ns, nf], lon [ns, nf], tb [C, ns, nf] from one or more consecutive ATMS L1B granules.

    NASA Sounder SIPS ATMS L1B v3 (SNPPATMSL1B, SNDRJ1ATMSL1B, SNDRJ2ATMSL1B): 6-minute granules with
    antenna_temp(atrack=135, xtrack=96, channel=22), lat/lon(atrack, xtrack), geo_qualflag(atrack, xtrack).
    Consecutive granules are concatenated along the scan (atrack) axis. NOAA CLASS SDR names
    (BrightnessTemperature/Latitude/Longitude) are accepted as a fallback. Returns
    lat, lon, tb (limb-adjusted, see physics/limb.py), satellite zenith angle (deg), land fraction.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    lats, lons, tbs, zens, lands = [], [], [], [], []
    for path in paths:
        with xr.open_dataset(path) as ds:
            tb = ds["antenna_temp"] if "antenna_temp" in ds else ds["BrightnessTemperature"]
            tb_np = tb.values.astype(np.float32)                                # [ns, nf, 22]
            lat = (ds["lat"] if "lat" in ds else ds["Latitude"]).values.astype(np.float64)
            lon = (ds["lon"] if "lon" in ds else ds["Longitude"]).values.astype(np.float64)
            zen = ds["sat_zen"].values.astype(np.float32) if "sat_zen" in ds else np.zeros(lat.shape, np.float32)
            land = ds["land_frac"].values.astype(np.float32) if "land_frac" in ds else np.zeros(lat.shape, np.float32)
            if "geo_qualflag" in ds:                                            # drop bad geolocation
                bad = ds["geo_qualflag"].values != 0
                lat[bad], lon[bad] = np.nan, np.nan
        lats.append(lat), lons.append(lon), tbs.append(tb_np), zens.append(zen), lands.append(land)
    lat, lon, tb_np = np.concatenate(lats, 0), np.concatenate(lons, 0), np.concatenate(tbs, 0)
    zen, land = np.concatenate(zens, 0), np.concatenate(lands, 0)
    idx = [c - 1 for c in channels]                   # ATMS channels are 1-indexed
    tb_sel = np.moveaxis(tb_np[..., idx], -1, 0)      # [C, ns, nf]
    tb_sel[(tb_sel < 50) | (tb_sel > 350)] = np.nan   # physical bounds -> fill
    if limb_correct and np.nanmax(zen) > 5.0:
        tb_sel, _ = limb_adjust(tb_sel, zen)
    return lat, lon, tb_sel, zen, land


def build_scene(paths: RawScenePaths, cfg: DataConfig) -> xr.Dataset:
    """Co-register all sources for one analysis time into a single xarray.Dataset."""
    g = cfg.grid
    target = TargetGrid.storm_centred(paths.storm_lat, paths.storm_lon, g)
    coarse = target.coarsen(g.mw_downscale)
    H, W = target.shape
    h, w = coarse.shape

    # --- IR --------------------------------------------------------------------------------
    ir = np.zeros((len(cfg.ir_channels), H, W), np.float32)
    ir_mask = np.ones((H, W), bool)
    for i, ch in enumerate(cfg.ir_channels):
        if ch not in paths.goes_files:
            ir_mask[:] = False
            continue
        with xr.open_dataset(paths.goes_files[ch]) as gds:
            vals, valid = regrid_goes_to_target(gds, "CMI", target)
        ir[i] = vals
        ir_mask &= valid

    # --- MW --------------------------------------------------------------------------------
    mw = np.zeros((len(cfg.mw_channels), h, w), np.float32)
    mw_mask = np.zeros((h, w), bool)
    mw_zen = np.zeros((h, w), np.float32)
    mw_land = np.zeros((h, w), np.float32)
    if paths.atms_files:
        lat, lon, tb, zen, land = _open_atms(paths.atms_files, cfg.mw_channels)
        mw_mask[:] = True
        for c in range(tb.shape[0]):
            vals, valid = regrid_swath_to_target(lat, lon, tb[c], coarse)
            mw[c] = vals
            mw_mask &= valid
        mw_zen, _ = regrid_swath_to_target(lat, lon, zen, coarse)
        mw_land, _ = regrid_swath_to_target(lat, lon, land, coarse)

    # --- Labels ----------------------------------------------------------------------------
    with xr.open_dataset(paths.era5_file) as eds:
        lvl_name = "pressure_level" if "pressure_level" in eds.dims else "level"
        t = eds["t"].sel({lvl_name: list(cfg.levels_hpa)})
        if "valid_time" in t.dims or "time" in t.dims:
            tname = "valid_time" if "valid_time" in t.dims else "time"
            t = t.sel({tname: paths.time}, method="nearest")
        temp = regrid_latlon_to_target(t, target)                     # [L, H, W]
    with xr.open_dataset(paths.imerg_file, group="Grid") as ids:
        p = ids["precipitation"] if "precipitation" in ids else ids["precipitationCal"]
        p = p.isel(time=0) if "time" in p.dims else p
        p = p.transpose("lat", "lon")
        precip = regrid_latlon_to_target(p, target, lat_name="lat", lon_name="lon")   # [H, W]
        precip = np.clip(np.nan_to_num(precip, nan=0.0), 0, None)

    ds = xr.Dataset(
        {
            "ir": (("ir_channel", "y", "x"), ir, {"units": "K", "long_name": "GOES ABI brightness temperature"}),
            "ir_mask": (("y", "x"), ir_mask),
            "mw": (("mw_channel", "yc", "xc"), mw, {"units": "K", "long_name": "ATMS brightness temperature"}),
            "mw_mask": (("yc", "xc"), mw_mask),
            "mw_zenith": (("yc", "xc"), mw_zen, {"units": "degree", "long_name": "ATMS satellite zenith angle"}),
            "mw_landfrac": (("yc", "xc"), mw_land, {"long_name": "ATMS footprint land fraction"}),
            "temp": (("level", "y", "x"), temp, {"units": "K", "long_name": "ERA5 temperature"}),
            "precip": (("y", "x"), precip, {"units": "mm h-1", "long_name": "IMERG precipitation rate"}),
            "lat": (("y", "x"), target.lat2d), "lon": (("y", "x"), target.lon2d),
            "latc": (("yc", "xc"), coarse.lat2d), "lonc": (("yc", "xc"), coarse.lon2d),
        },
        coords={"ir_channel": list(cfg.ir_channels), "mw_channel": list(cfg.mw_channels), "level": list(cfg.levels_hpa)},
        attrs={"time": str(paths.time), "storm_lat": paths.storm_lat, "storm_lon": paths.storm_lon, "storm_name": paths.storm_name,
               "mw_dt_min": float(paths.mw_dt_min) if paths.mw_dt_min is not None else -9999.0,
               "goes_files": ";".join(os.path.basename(v) for v in paths.goes_files.values()),
               "atms_files": ";".join(os.path.basename(v) for v in paths.atms_files)},
    )
    return ds


def build_scene_cache(scenes: Sequence[RawScenePaths], cfg: DataConfig, out_dir: str, overwrite: bool = False) -> List[str]:
    """Build and persist every scene as NetCDF; returns the list of cached paths."""
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for sp in scenes:
        path = os.path.join(out_dir, f"{sp.storm_name}_{np.datetime_as_string(sp.time, unit='m').replace(':', '')}.nc")
        if overwrite or not os.path.exists(path):
            build_scene(sp, cfg).to_netcdf(path)
        out.append(path)
    return out


# ----------------------------------------------------------------------------------------------
# PyTorch Dataset
# ----------------------------------------------------------------------------------------------
class HurricaneSceneDataset(Dataset):
    """Serves normalised tensors from cached scene NetCDF files (or in-memory xarray Datasets)."""

    def __init__(self, scenes: Sequence, cfg: DataConfig, normalizer: Normalizer, augment: bool = False, downscale: int = 1):
        self.scenes = list(scenes)          # paths (str) or xr.Dataset
        self.cfg = cfg
        self.norm = normalizer
        self.augment = augment
        self.downscale = int(downscale)     # 2 -> train at half resolution (128 x 128 fine, 16 x 16 coarse)

    def __len__(self) -> int:
        return len(self.scenes)

    def _load(self, i: int) -> xr.Dataset:
        s = self.scenes[i]
        return xr.load_dataset(s) if isinstance(s, (str, os.PathLike)) else s

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s = self._load(i)
        ir_raw = torch.from_numpy(s["ir"].values).float()                     # [C_ir, H, W]
        mw_raw = torch.from_numpy(s["mw"].values).float()                     # [C_mw, h, w]
        ir_mask = torch.from_numpy(s["ir_mask"].values).float()[None]         # [1, H, W]
        mw_mask = torch.from_numpy(s["mw_mask"].values).float()[None]         # [1, h, w]
        mw_zen = (torch.from_numpy(s["mw_zenith"].values).float()[None] / 60.0) if "mw_zenith" in s else torch.zeros_like(mw_mask)   # [1, h, w], ~0..1
        temp = torch.from_numpy(s["temp"].values).float()                     # [L, H, W]
        precip = torch.from_numpy(s["precip"].values).float()[None]           # [1, H, W]

        ir = self.norm.normalize("ir", ir_raw) * ir_mask                      # masked -> 0 after normalisation
        mw = self.norm.normalize("mw", mw_raw) * mw_mask
        state = self.norm.physical_to_state(temp[None], precip[None], self.cfg.precip_log_transform)[0]   # [L+1, H, W]

        sample = {"ir": ir, "ir_mask": ir_mask, "mw": mw, "mw_mask": mw_mask, "mw_zen": mw_zen * mw_mask, "state": state, "ir_raw": ir_raw, "mw_raw": mw_raw}
        if self.downscale > 1:
            sample = _coarsen_sample(sample, self.downscale)
        if self.augment:
            sample = _random_flip_rot(sample)
        return sample


def _coarsen_sample(sample: Dict[str, torch.Tensor], f: int) -> Dict[str, torch.Tensor]:
    """Block-average every field by f (masks by min so a block is valid only if fully valid;
    masked MW fields are mask-weighted so zeros outside the swath do not bias the average)."""
    import torch.nn.functional as F

    out = {}
    for k in ("ir", "ir_raw", "state"):
        out[k] = F.avg_pool2d(sample[k][None], f)[0]
    out["ir_mask"] = -F.max_pool2d(-sample["ir_mask"][None], f)[0]
    m = sample["mw_mask"][None]
    msum = F.avg_pool2d(m, f)
    for k in ("mw", "mw_raw", "mw_zen"):
        out[k] = (F.avg_pool2d(sample[k][None] * m, f) / msum.clamp(min=1e-6))[0]
    out["mw_mask"] = (msum[0] >= 0.5).float()
    for k in ("mw", "mw_raw", "mw_zen"):
        out[k] = out[k] * out["mw_mask"]
    return out


def _random_flip_rot(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Dihedral augmentation applied consistently to fine and coarse grids (geometry-preserving)."""
    k = int(torch.randint(0, 4, (1,)))
    flip = bool(torch.rand(1) < 0.5)
    out = {}
    for key, v in sample.items():
        v = torch.rot90(v, k, dims=(-2, -1))
        if flip:
            v = torch.flip(v, dims=(-1,))
        out[key] = v.contiguous()
    return out


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}
