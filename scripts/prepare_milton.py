"""
Build the Hurricane Milton scene cache from the public archives.

    python -m milton_da.scripts.prepare_milton --root data/milton \
        --start 2024-10-06T00 --end 2024-10-10T00 --step-hours 6 \
        [--times 2024-10-07T06 2024-10-07T18 ...] [--dry-run]

Any storm in IBTrACS works: --storm MELISSA --season 2025 --start 2025-10-25T00 --end 2025-10-29T00 --goes-satellite goes19
(GOES-East has been GOES-19 since April 2025; the Milton scenes came from GOES-16).

Steps: IBTrACS track -> analysis times and centres -> GOES-East (nearest scan), ATMS (nearest overpass
from SNPP/NOAA-20/NOAA-21), IMERG (containing half hour), ERA5 (one request per day) -> manifest.json
-> scene NetCDFs under <root>/scenes. Re-runs are incremental: existing files are not fetched again.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data.best_track import BestTrack
from ..data.dataset import build_scene_cache
from ..data.download import (assemble_manifest, download_atms, download_era5, download_goes, download_ibtracs,
                             download_imerg, save_manifest, storm_area)

log = logging.getLogger("milton_da")


def analysis_times(start: str, end: str, step_hours: int, explicit) -> list:
    if explicit:
        return [np.datetime64(t) for t in explicit]
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{step_hours}h").to_numpy())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--storm", default="MILTON")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--start", default="2024-10-06T00:00")
    ap.add_argument("--end", default="2024-10-10T00:00")
    ap.add_argument("--step-hours", type=int, default=6)
    ap.add_argument("--times", nargs="*", default=None, help="explicit analysis times (override start/end/step)")
    ap.add_argument("--goes-product", default=None, help="force ABI-L2-CMIPF or ABI-L2-CMIPC")
    ap.add_argument("--goes-satellite", default="goes16", choices=["goes16", "goes18", "goes19"],
                    help="GOES-East was GOES-16 until April 2025 and GOES-19 after; use goes19 for 2025 storms")
    ap.add_argument("--dry-run", action="store_true", help="resolve track and times, download nothing")
    ap.add_argument("--rebuild", action="store_true", help="rebuild scene NetCDFs even if they exist (raw files are reused)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = PipelineConfig()
    root = args.root
    raw = os.path.join(root, "raw")
    ib = download_ibtracs(os.path.join(raw, "ibtracs"))
    bt = BestTrack.from_ibtracs_csv(ib, name=args.storm, season=args.season)
    times = analysis_times(args.start, args.end, args.step_hours, args.times)
    times = [t for t in times if bt.times[0] <= np.datetime64(t, "ns") <= bt.times[-1]]
    centers = [bt.center_at(t) for t in times]
    log.info(f"{args.storm} {args.season} SID {bt.sid}: {len(times)} analysis times from {times[0]} to {times[-1]}")
    for t, (la, lo) in zip(times, centers):
        log.info(f"  {np.datetime_as_string(t, unit='m')}  centre {la:6.2f}N {lo:7.2f}E")
    if args.dry_run:
        return 0

    goes = download_goes(times, centers, cfg.data.ir_channels, os.path.join(raw, args.goes_satellite), satellite=args.goes_satellite,
                         tol_min=cfg.data.ir_time_tolerance_min, product=args.goes_product)
    atms = download_atms(times, centers, os.path.join(raw, "atms"), tol_min=cfg.data.mw_time_tolerance_min)
    imerg = download_imerg(times, os.path.join(raw, "imerg"))
    area = storm_area([c[0] for c in centers], [c[1] for c in centers])
    era5 = download_era5(times, cfg.data.levels_hpa, area, os.path.join(raw, "era5"), tag=f"{args.storm}{args.season}")

    scenes = assemble_manifest(times, centers, goes, atms, era5, imerg, storm_name=args.storm)
    manifest = os.path.join(root, "manifest.json")
    save_manifest(scenes, manifest)
    n_mw = sum(1 for s in scenes if s.atms_files)
    log.info(f"manifest: {len(scenes)} scenes ({n_mw} with an ATMS overpass) -> {manifest}")
    paths = build_scene_cache(scenes, cfg.data, os.path.join(root, "scenes"), overwrite=args.rebuild)
    log.info(f"scene cache: {len(paths)} files under {os.path.join(root, 'scenes')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
