"""
Build the multi-storm training archive (North Atlantic hurricanes, GOES-16 era) for the prior and proxy.

    python -m milton_da.scripts.prepare_archive --root data/archive --seasons 2018 2019 2020 2021 2022 2023 \
        --min-wind 64 --step-hours 3 [--max-storms 40] [--exclude MILTON:2024] [--list-only]

Selection: IBTrACS storms in the seasons/basins that reached the wind threshold; analysis times every
step_hours along each track while the storm is at or above 34 kt (so the archive is not dominated by
weak remnants). The full 2018-2023 set is ~51 storms / ~3,400 analysis times; raw satellite files
would exceed 150 GB, so use --purge-raw (scenes are ~5 MB each) and start with --max-storms 12.
The job is resumable: per-storm manifests and existing scene files are skipped on re-run. Sources and matching rules are the same as for Milton. ERA5 is requested once per
storm-day; GOES uses the CONUS sector when the storm is inside it. Scenes without an ATMS overpass
are kept (mask = 0), because the prior and the infrared path of the proxy still learn from them.
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
                             download_imerg, load_manifest, save_manifest, storm_area)

log = logging.getLogger("milton_da")
GOES16_EAST_START = np.datetime64("2017-12-18")   # GOES-16 declared operational as GOES-East


def storm_times(bt: BestTrack, step_hours: int, min_wind_kt: float = 34.0) -> list:
    t0 = pd.Timestamp(bt.times[0]).ceil(f"{step_hours}h")
    t1 = pd.Timestamp(bt.times[-1]).floor(f"{step_hours}h")
    times = pd.date_range(t0, t1, freq=f"{step_hours}h")
    x = (bt.times - bt.times[0]) / np.timedelta64(1, "s")
    wind = np.where(np.isfinite(bt.wind_kt), bt.wind_kt, 0.0)
    keep = []
    for t in times:
        xt = (np.datetime64(t, "ns") - bt.times[0]) / np.timedelta64(1, "s")
        if np.interp(xt, x, wind) >= min_wind_kt:
            keep.append(t.to_numpy())
    return keep


def purge_raw(scenes) -> int:
    """Delete the GOES, ATMS and IMERG raw files referenced by these scenes. Returns bytes freed."""
    freed = 0
    for sc in scenes:
        for path in list(sc.goes_files.values()) + list(sc.atms_files) + [sc.imerg_file]:
            if path and os.path.exists(path):
                freed += os.path.getsize(path)
                os.remove(path)
    return freed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2018, 2019, 2020, 2021, 2022, 2023])
    ap.add_argument("--basins", nargs="+", default=["NA"])
    ap.add_argument("--min-wind", type=float, default=64.0, help="storm must reach this peak wind (kt)")
    ap.add_argument("--step-hours", type=int, default=3)
    ap.add_argument("--max-storms", type=int, default=None)
    ap.add_argument("--exclude", nargs="*", default=["MILTON:2024"], help="NAME:SEASON to hold out")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--purge-raw", action="store_true", help="delete raw GOES/ATMS/IMERG files as soon as their scenes are built (keeps ERA5 and scenes)")
    ap.add_argument("--chunk", type=int, default=16, help="analysis times downloaded per block before scenes are built and raw files purged")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = PipelineConfig()
    root, raw = args.root, os.path.join(args.root, "raw")
    ib = download_ibtracs(os.path.join(raw, "ibtracs"))
    storms = BestTrack.list_storms(ib, args.seasons, args.basins, args.min_wind)
    excl = {tuple(e.split(":")) for e in args.exclude}
    storms = storms[~storms.apply(lambda r: (str(r["name"]).upper(), str(int(r["season"]))) in excl, axis=1)]
    storms = storms[pd.to_datetime(storms["start"]) >= pd.Timestamp(GOES16_EAST_START)]
    if args.max_storms:
        storms = storms.sort_values("max_wind", ascending=False).head(args.max_storms)
    log.info(f"{len(storms)} storms selected")
    total_scenes = 0
    all_scenes = []
    for _, row in storms.sort_values("start").iterrows():
        bt = BestTrack.from_ibtracs_csv(ib, sid=row["SID"])
        times = storm_times(bt, args.step_hours)
        total_scenes += len(times)
        log.info(f"  {row['name']:>10} {int(row['season'])}  peak {row['max_wind']:.0f} kt  {len(times)} analysis times")
        if args.list_only or not times:
            continue
        centers = [bt.center_at(t) for t in times]
        tag = f"{row['name']}{int(row['season'])}"
        manifest = os.path.join(root, "manifests", f"{tag}.json")
        if os.path.exists(manifest):
            scenes = load_manifest(manifest)
            build_scene_cache(scenes, cfg.data, os.path.join(root, "scenes"))
        else:
            # Work in blocks of `chunk` analysis times so raw full-disk GOES files never pile up on disk.
            scenes = []
            area = storm_area([c[0] for c in centers], [c[1] for c in centers])
            for k in range(0, len(times), args.chunk):
                tt, cc = times[k : k + args.chunk], centers[k : k + args.chunk]
                goes = download_goes(tt, cc, cfg.data.ir_channels, os.path.join(raw, "goes16"), tol_min=cfg.data.ir_time_tolerance_min)
                atms = download_atms(tt, cc, os.path.join(raw, "atms"), tol_min=cfg.data.mw_time_tolerance_min)
                imerg = download_imerg(tt, os.path.join(raw, "imerg"))
                era5 = download_era5(tt, cfg.data.levels_hpa, area, os.path.join(raw, "era5"), tag=tag)
                block = assemble_manifest(tt, cc, goes, atms, era5, imerg, storm_name=str(row["name"]))
                build_scene_cache(block, cfg.data, os.path.join(root, "scenes"))
                if args.purge_raw:
                    freed = purge_raw(block)
                    log.info(f"  {tag} block {k // args.chunk + 1}: {len(block)} scenes built, {freed / 1e9:.2f} GB raw purged")
                scenes.extend(block)
            save_manifest(scenes, manifest)
        all_scenes.extend(scenes)
    log.info(f"total analysis times: {total_scenes}; scenes built: {len(all_scenes)}")
    if not args.list_only:
        save_manifest(all_scenes, os.path.join(root, "manifest_all.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
