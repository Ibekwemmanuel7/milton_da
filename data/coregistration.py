"""
Co-registration of heterogeneous satellite / reanalysis geometries onto one storm-centred grid.

Three source geometries are handled:

1. Geostationary fixed grid (GOES-16 ABI)   : regular in scan angle (y, x), curvilinear in lat/lon.
   -> exact inverse projection of the target lat/lon into scan-angle space, then bilinear sampling
      on the *regular* (y, x) source grid. No scattered interpolation, no approximation.
2. Polar-orbiter swath (ATMS)               : irregular (scanline, field-of-view) point cloud.
   -> nearest / linear scattered interpolation with a validity mask (swath edges, gaps).
3. Regular lat/lon (ERA5 0.25 deg, IMERG 0.1 deg) : xarray advanced interpolation.

All functions return numpy arrays on the target grid plus a boolean validity mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata

from ..config import GridConfig


# ----------------------------------------------------------------------------------------------
# Target grid
# ----------------------------------------------------------------------------------------------
@dataclass
class TargetGrid:
    """Regular lat/lon grid centred on the storm at analysis time.

    lat2d, lon2d : [H, W] float64 (degrees)
    """
    lat2d: np.ndarray
    lon2d: np.ndarray
    center_lat: float
    center_lon: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.lat2d.shape

    @classmethod
    def storm_centred(cls, center_lat: float, center_lon: float, grid: GridConfig) -> "TargetGrid":
        half_y = (grid.ny - 1) / 2.0
        half_x = (grid.nx - 1) / 2.0
        lat1d = center_lat + (np.arange(grid.ny) - half_y) * grid.dlat_deg   # south -> north
        lon1d = center_lon + (np.arange(grid.nx) - half_x) * grid.dlon_deg   # west  -> east
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return cls(lat2d=lat2d, lon2d=lon2d, center_lat=center_lat, center_lon=center_lon)

    def coarsen(self, factor: int) -> "TargetGrid":
        """Block-mean coarse grid used for the microwave sounder (h = H / factor)."""
        H, W = self.shape
        lat = self.lat2d[: H - H % factor, : W - W % factor].reshape(H // factor, factor, W // factor, factor).mean((1, 3))
        lon = self.lon2d[: H - H % factor, : W - W % factor].reshape(H // factor, factor, W // factor, factor).mean((1, 3))
        return TargetGrid(lat, lon, self.center_lat, self.center_lon)

    def to_xarray_coords(self) -> Dict[str, xr.DataArray]:
        return {
            "lat": xr.DataArray(self.lat2d, dims=("y", "x")),
            "lon": xr.DataArray(self.lon2d, dims=("y", "x")),
        }


# ----------------------------------------------------------------------------------------------
# 1. Geostationary fixed grid -> target grid
# ----------------------------------------------------------------------------------------------
def _geos_transformer(goes_ds: xr.Dataset):
    """Build a pyproj Transformer lon/lat -> GOES scan-angle (x, y) in radians."""
    import pyproj  # imported lazily so the package imports without pyproj (synthetic runs)

    p = goes_ds["goes_imager_projection"].attrs
    proj = pyproj.CRS.from_proj4(
        f"+proj=geos +h={p['perspective_point_height']} +lon_0={p['longitude_of_projection_origin']} "
        f"+sweep={p['sweep_angle_axis']} +a={p['semi_major_axis']} +b={p['semi_minor_axis']} +units=m +no_defs"
    )
    tr = pyproj.Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    return tr, float(p["perspective_point_height"])


def regrid_goes_to_target(goes_ds: xr.Dataset, var: str, target: TargetGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a GOES ABI L2 CMI variable on the target grid.

    Parameters
    ----------
    goes_ds : ABI L1b/L2 dataset (e.g. OR_ABI-L2-CMIPF-M6C13_G16_*.nc) with coords x, y in radians.
    var     : variable name, e.g. "CMI" (brightness temperature, K).
    Returns
    -------
    values [H, W] float32, valid [H, W] bool
    """
    tr, h = _geos_transformer(goes_ds)
    # Project target lat/lon to fixed-grid metres, convert to scan angle (radians) as stored in x/y.
    xm, ym = tr.transform(target.lon2d, target.lat2d)
    xs, ys = xm / h, ym / h
    finite = np.isfinite(xs) & np.isfinite(ys)   # points beyond the Earth disc are inf/nan

    src = goes_ds[var].values.astype(np.float32)   # [ny_src, nx_src]
    y_src = goes_ds["y"].values.astype(np.float64)
    x_src = goes_ds["x"].values.astype(np.float64)
    # RegularGridInterpolator requires ascending coordinates; ABI y is descending.
    if y_src[0] > y_src[-1]:
        y_src, src = y_src[::-1], src[::-1]
    interp = RegularGridInterpolator((y_src, x_src), src, method="linear", bounds_error=False, fill_value=np.nan)
    pts = np.stack([np.where(finite, ys, y_src[0]), np.where(finite, xs, x_src[0])], -1).reshape(-1, 2)
    out = interp(pts).reshape(target.shape).astype(np.float32)
    valid = finite & np.isfinite(out)
    return np.where(valid, out, 0.0).astype(np.float32), valid


# ----------------------------------------------------------------------------------------------
# 2. Swath (ATMS) -> coarse target grid
# ----------------------------------------------------------------------------------------------
def regrid_swath_to_target(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    target: TargetGrid,
    method: str = "linear",
    max_gap_deg: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scattered interpolation of a swath variable onto the (coarse) target grid.

    lat, lon, values : [n_scan, n_fov] (or already flattened). NaNs in `values` are dropped.
    Returns values [h, w] float32 and a validity mask that is False outside the swath
    (nearest-neighbour distance > max_gap_deg) or where the sensor reported fill values.
    """
    lat, lon, values = (np.asarray(a).ravel() for a in (lat, lon, values))
    ok = np.isfinite(values) & np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 4:
        return np.zeros(target.shape, np.float32), np.zeros(target.shape, bool)
    pts = np.stack([lat[ok], lon[ok]], -1)
    tgt = np.stack([target.lat2d.ravel(), target.lon2d.ravel()], -1)
    out = griddata(pts, values[ok], tgt, method=method).reshape(target.shape)
    # Coverage mask: distance to nearest real observation.
    from scipy.spatial import cKDTree

    d, _ = cKDTree(pts).query(tgt, k=1)
    valid = (d.reshape(target.shape) <= max_gap_deg) & np.isfinite(out)
    return np.where(valid, out, 0.0).astype(np.float32), valid


# ----------------------------------------------------------------------------------------------
# 3. Regular lat/lon (ERA5 / IMERG) -> target grid
# ----------------------------------------------------------------------------------------------
def regrid_latlon_to_target(
    da: xr.DataArray, target: TargetGrid, lat_name: str = "latitude", lon_name: str = "longitude"
) -> np.ndarray:
    """Bilinear interpolation of a regular lat/lon DataArray onto the target grid.

    Works for 2D [lat, lon] (IMERG) and 3D [level, lat, lon] (ERA5) arrays; leading dims are kept.
    Longitudes are harmonised to the target convention (-180..180 or 0..360) automatically.
    """
    lon_src = da[lon_name].values
    lon_tgt = target.lon2d.copy()
    if lon_src.max() > 180 and lon_tgt.min() < 0:
        lon_tgt = np.where(lon_tgt < 0, lon_tgt + 360, lon_tgt)
    elif lon_src.min() < 0 and lon_tgt.max() > 180:
        lon_tgt = np.where(lon_tgt > 180, lon_tgt - 360, lon_tgt)
    if da[lat_name].values[0] > da[lat_name].values[-1]:
        da = da.sortby(lat_name)
    out = da.interp(
        {lat_name: xr.DataArray(target.lat2d, dims=("y", "x")), lon_name: xr.DataArray(lon_tgt, dims=("y", "x"))},
        method="linear",
    )
    return np.ascontiguousarray(out.values.astype(np.float32))


def block_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Area-average a [..., H, W] array to [..., H/f, W/f]. Used to derive coarse fields."""
    *lead, H, W = a.shape
    a = a[..., : H - H % factor, : W - W % factor]
    return a.reshape(*lead, H // factor, factor, W // factor, factor).mean((-3, -1))
