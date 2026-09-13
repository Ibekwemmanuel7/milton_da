"""
Patch existing scene files with ERA5 cloud ice (ciwc profile and ice water path) without rebuilding them.

    python -m milton_da.scripts.add_cloud_ice --scenes "data/archive/scenes/*.nc" "data/milton/scenes/*.nc" \
        --raw data/raw/era5_ice [--overwrite] [--dry-run]

Scenes are grouped by storm (storm_name attribute and season); one CDS request per storm fetches
specific_cloud_ice_water_content on the state levels plus 100 to 175 hPa for every analysis day and hour
of that storm over a box around its track. Each scene then gets 'ciwc' (level, y, x) and 'iwp' (y, x) on
its own grid, exactly as build_scene() would have written them. Requests are cached under --raw, so the
job is resumable. About 100 storms for the 2017 to 2023 archive: expect an hour or two of CDS queue.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
import xarray as xr

from ..config import GridConfig, PipelineConfig
from ..data.coregistration import TargetGrid
from ..data.dataset import cloud_ice_from_era5
from ..data.download import ERA5_ICE_EXTRA_LEVELS_HPA, _retry, storm_area

log = logging.getLogger("milton_da")


def ice_request(days: List[str], hours: List[int], levels_hpa: List[int], area: List[float]) -> Dict:
    ds = sorted(set(days))
    years = sorted({d[:4] for d in ds}); months = sorted({d[5:7] for d in ds}); dd = sorted({d[8:10] for d in ds})
    return {
        "product_type": ["reanalysis"],
        "variable": ["specific_cloud_ice_water_content"],
        "pressure_level": [str(p) for p in sorted(set(levels_hpa) | set(ERA5_ICE_EXTRA_LEVELS_HPA))],
        "year": years, "month": months, "day": dd,
        "time": [f"{h:02d}:00" for h in sorted(set(hours))],
        "area": [float(a) for a in area],
        "data_format": "netcdf", "download_format": "unarchived",
    }


def scene_key(path: str) -> tuple:
    with xr.open_dataset(path) as s:
        t = str(s.attrs.get("time"))[:16]
        return (str(s.attrs.get("storm_name", "STORM")), t[:4]), t, float(s.attrs["storm_lat"]), float(s.attrs["storm_lon"]), int(s.sizes["y"]), int(s.sizes["x"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True, help="glob patterns of scene NetCDF files")
    ap.add_argument("--raw", default="data/raw/era5_ice", help="cache folder for the ERA5 cloud-ice downloads")
    ap.add_argument("--overwrite", action="store_true", help="recompute scenes that already have 'iwp'")
    ap.add_argument("--dry-run", action="store_true", help="list the storms and requests, download nothing")
    ap.add_argument("--max-storms", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = PipelineConfig()
    levels = list(cfg.data.levels_hpa)
    paths = sorted({p for pat in args.scenes for p in glob.glob(pat)})
    groups: Dict[tuple, List[tuple]] = defaultdict(list)
    skipped = 0
    for p in paths:
        with xr.open_dataset(p) as s:
            if "iwp" in s and not args.overwrite:
                skipped += 1; continue
        key, t, lat, lon, ny, nx = scene_key(p)
        groups[key].append((p, t, lat, lon, ny, nx))
    log.info(f"{len(paths)} scenes: {skipped} already have cloud ice, {sum(len(v) for v in groups.values())} to patch in {len(groups)} storms")
    if args.max_storms:
        groups = dict(list(sorted(groups.items()))[: args.max_storms])
    os.makedirs(args.raw, exist_ok=True)

    for (storm, season), items in sorted(groups.items()):
        times = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for _, t, *_ in items]
        area = storm_area([i[2] for i in items], [i[3] for i in items], pad_deg=4.0)
        req = ice_request([t.strftime("%Y-%m-%d") for t in times], [t.hour for t in times], levels, area)
        dst = os.path.join(args.raw, f"era5_ciwc_{storm}{season}.nc")
        log.info(f"{storm} {season}: {len(items)} scenes, {len(req['day'])} days x {len(req['time'])} hours, area {area} -> {os.path.basename(dst)}")
        if args.dry_run:
            continue
        if not os.path.exists(dst):
            import cdsapi
            client = cdsapi.Client()
            _retry(lambda: client.retrieve("reanalysis-era5-pressure-levels", req, dst), f"ERA5 ciwc {storm} {season}", attempts=4, base_delay=30.0)
        with xr.open_dataset(dst) as eds:
            if "ciwc" not in eds:
                log.error(f"{dst} has no 'ciwc' variable ({list(eds.data_vars)}); delete it and retry"); continue
            for p, t, lat, lon, ny, nx in items:
                grid = GridConfig(ny=ny, nx=nx, dlat_deg=cfg.data.grid.dlat_deg, dlon_deg=cfg.data.grid.dlon_deg, mw_downscale=cfg.data.grid.mw_downscale)
                target = TargetGrid.storm_centred(lat, lon, grid)
                sc = xr.load_dataset(p)
                # sanity: the rebuilt grid must be the scene's grid
                if not np.allclose(target.lat2d, sc["lat"].values, atol=1e-4) or not np.allclose(target.lon2d, sc["lon"].values, atol=1e-4):
                    log.error(f"{p}: grid mismatch, skipped"); continue
                ciwc, iwp = cloud_ice_from_era5(eds, np.datetime64(t), levels, target)
                sc["ciwc"] = (("level", "y", "x"), ciwc, {"units": "kg kg-1", "long_name": "ERA5 specific cloud ice water content"})
                sc["iwp"] = (("y", "x"), iwp, {"units": "kg m-2", "long_name": "ERA5 ice water path (integral of ciwc over all requested levels)"})
                tmp = p + ".tmp"
                sc.to_netcdf(tmp); sc.close(); os.replace(tmp, p)
                log.info(f"  {os.path.basename(p)}: iwp max {iwp.max():.2f} kg m-2, mean {iwp.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
