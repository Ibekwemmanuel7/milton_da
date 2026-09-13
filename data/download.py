"""
Archive access for the real-data runs: GOES-16 ABI (NOAA AWS), ATMS L1B and IMERG (NASA Earthdata),
ERA5 (Copernicus CDS) and IBTrACS (NCEI). Each source has one pure "selection" helper that is unit
tested offline and one thin I/O function.

Credentials
-----------
GOES     : none (anonymous S3).
Earthdata: free account at https://urs.earthdata.nasa.gov, then either ~/.netrc
           (machine urs.earthdata.nasa.gov login USER password PASS) or the environment variables
           EARTHDATA_USERNAME / EARTHDATA_PASSWORD. Approve the "NASA GESDISC DATA ARCHIVE"
           application once in the Earthdata profile or the ATMS/IMERG downloads return 401.
CDS      : free account at https://cds.climate.copernicus.eu, accept the ERA5 licence, and write
           ~/.cdsapirc with  url: https://cds.climate.copernicus.eu/api  and  key: <personal token>.

Products
--------
GOES-16 ABI L2 CMIP  s3://noaa-goes16/ABI-L2-CMIPF/YYYY/DDD/HH/   full disk, 10-min (mode 6)
                     s3://noaa-goes16/ABI-L2-CMIPC/YYYY/DDD/HH/   CONUS, 5-min, ~10x smaller files
ATMS L1B v3          short names SNPPATMSL1B (Suomi-NPP), SNDRJ1ATMSL1B (NOAA-20), SNDRJ2ATMSL1B (NOAA-21)
                     6-minute granules, antenna_temp(atrack, xtrack, channel), lat, lon, geo_qualflag
IMERG                short name GPM_3IMERGHH version 07 (Final), GPM_3IMERGHHL (Late, lower latency)
ERA5                 dataset reanalysis-era5-pressure-levels, variable temperature
IBTrACS              v04r01 North Atlantic CSV
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .dataset import RawScenePaths

log = logging.getLogger("milton_da")

IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
ATMS_SHORT_NAMES = ["SNPPATMSL1B", "SNDRJ1ATMSL1B", "SNDRJ2ATMSL1B"]
GOES_BUCKET = {"goes16": "noaa-goes16", "goes18": "noaa-goes18", "goes19": "noaa-goes19"}


def _retry(fn, what: str, attempts: int = 5, base_delay: float = 3.0, swallow: bool = False):
    """Call fn() with exponential backoff on any exception (network hiccups, 5xx, throttling).
    If it still fails and swallow=True, log and return None instead of raising."""
    import time

    for i in range(attempts):
        try:
            return fn()
        except Exception as ex:  # noqa: BLE001
            if i == attempts - 1:
                if swallow:
                    log.warning(f"{what}: giving up after {attempts} attempts ({ex})")
                    return None
                raise
            delay = base_delay * (2 ** i)
            log.warning(f"{what}: {ex}; retry {i + 1}/{attempts - 1} in {delay:.0f}s")
            time.sleep(delay)


def _to_dt(t) -> datetime:
    """numpy datetime64 / pandas Timestamp / datetime -> timezone-aware UTC datetime."""
    ts = pd.Timestamp(t)
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).to_pydatetime()


# ==============================================================================================
# GOES-16 ABI (anonymous S3)
# ==============================================================================================
_GOES_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_e(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_")


def parse_goes_times(key: str) -> Tuple[datetime, datetime]:
    """Start and end scan times from an ABI filename such as
    OR_ABI-L2-CMIPF-M6C13_G16_s20242811200205_e20242811209525_c20242811210001.nc"""
    m = _GOES_RE.search(key)
    if not m:
        raise ValueError(f"not an ABI filename: {key}")
    g = m.groups()
    mk = lambda i: datetime(int(g[i]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(g[i + 1]) - 1, hours=int(g[i + 2]), minutes=int(g[i + 3]), seconds=int(g[i + 4]) + int(g[i + 5]) / 10)
    return mk(0), mk(6)


def goes_sector_for(lat: float, lon: float) -> str:
    """CMIPC (CONUS, 5-min, small files) when the storm is inside the GOES-East CONUS sector, else CMIPF."""
    return "ABI-L2-CMIPC" if (15.0 <= lat <= 50.0 and -125.0 <= lon <= -60.0) else "ABI-L2-CMIPF"


def goes_prefixes(t: datetime, product: str, satellite: str = "goes16") -> List[str]:
    """S3 prefixes for the hour containing t and its neighbours (the scan may straddle the hour)."""
    out = []
    for dh in (-1, 0, 1):
        th = t + timedelta(hours=dh)
        out.append(f"{GOES_BUCKET[satellite]}/{product}/{th.year}/{th.timetuple().tm_yday:03d}/{th.hour:02d}/")
    return out


def select_goes_key(keys: Iterable[str], t: datetime, channel: str, tol_min: float) -> Optional[str]:
    """Among S3 keys, the file of `channel` whose scan *start* is nearest to t, within tolerance
    (the convention used by goes2go; a full-disk scan takes ~10 min so pixel times vary anyway)."""
    best, best_dt = None, None
    tag = f"C{int(channel[1:]):02d}_"
    for k in keys:
        if tag not in k:
            continue
        try:
            s, e = parse_goes_times(k)
        except ValueError:
            continue
        dt = abs((s - t).total_seconds()) / 60.0
        if dt <= tol_min and (best_dt is None or dt < best_dt):
            best, best_dt = k, dt
    return best


def download_goes(times: Sequence, centers: Sequence[Tuple[float, float]], channels: Sequence[str], out_dir: str,
                  satellite: str = "goes16", tol_min: float = 7.5, product: Optional[str] = None) -> List[Dict[str, str]]:
    """For each analysis time, download the nearest scan of every channel. Returns one {channel: path} per time."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    listing_cache: Dict[str, List[str]] = {}
    for t_raw, (lat, lon) in zip(times, centers):
        t = _to_dt(t_raw)
        prod = product or goes_sector_for(lat, lon)
        keys: List[str] = []
        for prefix in goes_prefixes(t, prod, satellite):
            if prefix not in listing_cache:
                def _ls(p=prefix):
                    try:
                        return fs.ls(p)
                    except FileNotFoundError:
                        return []
                listing_cache[prefix] = _retry(_ls, f"GOES ls {prefix}") or []
            keys.extend(listing_cache[prefix])
        files = {}
        for ch in channels:
            key = select_goes_key(keys, t, ch, tol_min)
            if key is None:
                log.warning(f"GOES {ch} not found within {tol_min} min of {t:%Y-%m-%dT%H:%M} ({prod})")
                continue
            dst = os.path.join(out_dir, os.path.basename(key))
            if not os.path.exists(dst):
                log.info(f"GOES get {key}")
                tmp = dst + ".part"
                ok = _retry(lambda: (fs.get(key, tmp), os.replace(tmp, dst)), f"GOES get {os.path.basename(key)}", swallow=True)
                if ok is None and not os.path.exists(dst):
                    continue
            files[ch] = dst
        results.append(files)
    return results


# ==============================================================================================
# NASA Earthdata (ATMS L1B, IMERG) via earthaccess
# ==============================================================================================
def _earthaccess_login():
    import earthaccess

    auth = earthaccess.login(strategy="environment") if os.environ.get("EARTHDATA_USERNAME") else earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError("Earthdata login failed: set EARTHDATA_USERNAME/EARTHDATA_PASSWORD or ~/.netrc")
    return earthaccess


@dataclass
class Granule:
    """Minimal, serialisable view of a CMR granule for offline selection logic."""
    short_name: str
    start: datetime
    end: datetime
    url: str
    name: str
    polygon: Optional[List[Tuple[float, float]]] = None    # [(lon, lat), ...] footprint boundary, if known

    def contains(self, lat: float, lon: float) -> Optional[bool]:
        """True/False if the footprint polygon is known, None otherwise."""
        if not self.polygon or len(self.polygon) < 3:
            return None
        pts = np.array(self.polygon, dtype=float)
        # unwrap longitudes so a polygon straddling the dateline is contiguous
        lon0 = pts[0, 0]
        pts[:, 0] = np.where(pts[:, 0] - lon0 > 180, pts[:, 0] - 360, np.where(pts[:, 0] - lon0 < -180, pts[:, 0] + 360, pts[:, 0]))
        q = lon if abs(lon - lon0) <= 180 else (lon - 360 if lon - lon0 > 180 else lon + 360)
        return _point_in_polygon(q, lat, pts)


def _point_in_polygon(x: float, y: float, pts: np.ndarray) -> bool:
    """Even-odd ray casting; pts [N, 2] as (x, y). Pure numpy, no plotting dependency."""
    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.sum(crosses & (x < x_int)) % 2 == 1)


def _cmr_polygon(r) -> Optional[List[Tuple[float, float]]]:
    try:
        geom = r["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        if "GPolygons" in geom:
            pts = geom["GPolygons"][0]["Boundary"]["Points"]
            return [(float(p["Longitude"]), float(p["Latitude"])) for p in pts]
        if "BoundingRectangles" in geom:
            b = geom["BoundingRectangles"][0]
            w, e_, s_, n = (float(b[k]) for k in ("WestBoundingCoordinate", "EastBoundingCoordinate", "SouthBoundingCoordinate", "NorthBoundingCoordinate"))
            return [(w, s_), (e_, s_), (e_, n), (w, n)]
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return None


def granules_from_cmr(results, short_name: str) -> List[Granule]:
    out = []
    for r in results:
        rng = r["umm"]["TemporalExtent"]["RangeDateTime"]
        s, e = _to_dt(rng["BeginningDateTime"]), _to_dt(rng["EndingDateTime"])
        links = r.data_links()
        out.append(Granule(short_name, s, e, links[0], os.path.basename(links[0]), _cmr_polygon(r)))
    return out


def dedupe_granule_versions(granules: Sequence[Granule]) -> List[Granule]:
    """GES DISC can return the same 6-min granule in several product versions (e.g. v03_15 and
    v02_11). Keep one per (platform, start time): the lexicographically highest version string."""
    best: Dict[Tuple[str, datetime], Granule] = {}
    for g in granules:
        key = (g.short_name, g.start)
        ver = re.search(r"\.v(\d+_\d+)\.", g.name)
        cur = best.get(key)
        cur_ver = re.search(r"\.v(\d+_\d+)\.", cur.name) if cur else None
        if cur is None or (ver and (cur_ver is None or ver.group(1) > cur_ver.group(1))):
            best[key] = g
    return list(best.values())


def select_atms_overpass(granules: Sequence[Granule], t: datetime, tol_min: float, neighbour_min: float = 7.0,
                         center: Optional[Tuple[float, float]] = None) -> Tuple[List[Granule], Optional[float]]:
    """Pick the overpass (one platform, consecutive 6-min granules) nearest to t.

    Granules from the CMR spatial query already intersect the storm box. Duplicate product versions
    are removed first. Within each platform, granules are clustered into overpasses by time gaps.
    If `center` = (lat, lon) is given, overpasses whose footprint polygon contains the storm centre
    are preferred over ones that only clip the domain; among equals the one nearest to t wins.
    Returns (granules of that overpass sorted by time, dt in minutes).
    """
    by_platform: Dict[str, List[Granule]] = {}
    for g in dedupe_granule_versions(granules):
        by_platform.setdefault(g.short_name, []).append(g)
    best, best_dt, best_cov = [], None, -1
    for plat, gs in by_platform.items():
        gs = sorted(gs, key=lambda g: g.start)
        clusters: List[List[Granule]] = []
        for g in gs:
            if clusters and (g.start - clusters[-1][-1].end).total_seconds() / 60.0 <= neighbour_min:
                clusters[-1].append(g)
            else:
                clusters.append([g])
        for c in clusters:
            mid = c[0].start + (c[-1].end - c[0].start) / 2
            dt = (mid - t).total_seconds() / 60.0
            if abs(dt) > tol_min:
                continue
            cov = 0
            if center is not None:
                flags = [g.contains(*center) for g in c]
                cov = 1 if any(f is True for f in flags) else (0 if all(f is None for f in flags) else -1)
            # rank: centre covered (1) > unknown (0) > known not covered (-1); then nearest in time
            if (cov, -abs(dt)) > (best_cov, -abs(best_dt) if best_dt is not None else -1e9):
                best, best_dt, best_cov = c, dt, cov
    return best, best_dt


def download_atms(times: Sequence, centers: Sequence[Tuple[float, float]], out_dir: str, tol_min: float = 90.0,
                  box_half_deg: float = 3.0, short_names: Sequence[str] = ATMS_SHORT_NAMES) -> List[Tuple[List[str], Optional[float]]]:
    """For each analysis time, the nearest ATMS overpass over the storm from any of the three platforms.
    Returns one ([granule paths], dt_minutes) per time; ([], None) when no overpass is within tolerance."""
    ea = _earthaccess_login()
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for t_raw, (lat, lon) in zip(times, centers):
        t = _to_dt(t_raw)
        t0, t1 = t - timedelta(minutes=tol_min + 10), t + timedelta(minutes=tol_min + 10)
        bbox = (lon - box_half_deg, lat - box_half_deg, lon + box_half_deg, lat + box_half_deg)
        found: List[Granule] = []
        for sn in short_names:
            res = _retry(lambda: ea.search_data(short_name=sn, temporal=(t0.strftime("%Y-%m-%dT%H:%M:%S"), t1.strftime("%Y-%m-%dT%H:%M:%S")), bounding_box=bbox),
                         f"CMR {sn} {t:%Y-%m-%dT%H:%M}", attempts=4, swallow=True)
            if res is None:
                continue          # collection may not exist for the period (e.g. NOAA-21 before 2023) or CMR is down
            found.extend(granules_from_cmr(res, sn))
        chosen, dt = select_atms_overpass(found, t, tol_min, center=(lat, lon))
        if not chosen:
            log.warning(f"ATMS: no overpass within {tol_min} min of {t:%Y-%m-%dT%H:%M}")
            results.append(([], None))
            continue
        if not any(g.contains(lat, lon) for g in chosen):
            log.warning(f"ATMS: nearest overpass ({chosen[0].short_name}, dt={dt:+.0f} min) does not contain the storm centre; swath will be partial")
        paths = []
        for g in chosen:
            dst = os.path.join(out_dir, g.name)
            if not os.path.exists(dst):
                log.info(f"ATMS get {g.name}")
                if _retry(lambda: _earthdata_fetch(g.url, dst) or True, f"ATMS get {g.name}", swallow=True) is None:
                    continue
            paths.append(dst)
        if not paths:
            log.warning(f"ATMS: download failed for {t:%Y-%m-%dT%H:%M}; scene will be infrared-only")
            results.append(([], None))
        else:
            results.append((paths, dt))
    return results


def _earthdata_fetch(url: str, dst: str) -> None:
    """Authenticated HTTPS download through the earthaccess session (handles URS redirects)."""
    import earthaccess

    session = earthaccess.get_requests_https_session()
    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        os.replace(tmp, dst)


def select_imerg(granules: Sequence[Granule], t: datetime) -> Optional[Granule]:
    """The half-hourly file whose window contains t (or is nearest to it)."""
    best, best_dt = None, None
    for g in granules:
        mid = g.start + (g.end - g.start) / 2
        dt = abs((mid - t).total_seconds()) / 60.0
        if best_dt is None or dt < best_dt:
            best, best_dt = g, dt
    return best if best is not None and best_dt <= 30 else None


IMERG_RUNS = ["GPM_3IMERGHH", "GPM_3IMERGHHL", "GPM_3IMERGHHE"]   # Final, Late, Early: same Grid/precipitation layout


def download_imerg(times: Sequence, out_dir: str, short_name: str = "GPM_3IMERGHH", version: str = "07",
                   fallback_runs: Sequence[str] = ("GPM_3IMERGHHL", "GPM_3IMERGHHE")) -> List[Optional[str]]:
    """The IMERG Final Run lags real time by months (the V07 Final archive ended at 2025-09-30 when checked in
    September 2026). For recent storms the search falls back to the Late Run and then the Early Run; the
    file layout is identical and the run used is logged so the scene provenance is explicit."""
    ea = _earthaccess_login()
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for t_raw in times:
        t = _to_dt(t_raw)
        g = None
        for sn in [short_name] + [r for r in fallback_runs if r != short_name]:
            res = _retry(lambda sn=sn: ea.search_data(short_name=sn, version=version, temporal=((t - timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%S"), (t + timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%S"))),
                         f"CMR IMERG {sn} {t:%Y-%m-%dT%H:%M}", attempts=4, swallow=True)
            g = select_imerg(granules_from_cmr(res, sn), t) if res is not None else None
            if g is not None:
                if sn != short_name:
                    log.warning(f"IMERG: {short_name} has nothing near {t:%Y-%m-%dT%H:%M}; using {sn} (Late/Early run)")
                break
        if g is None:
            log.warning(f"IMERG: nothing near {t:%Y-%m-%dT%H:%M} in any run")
            out.append(None)
            continue
        dst = os.path.join(out_dir, g.name)
        if not os.path.exists(dst):
            log.info(f"IMERG get {g.name}")
            if _retry(lambda: _earthdata_fetch(g.url, dst) or True, f"IMERG get {g.name}", swallow=True) is None:
                out.append(None)
                continue
        out.append(dst)
    return out


# ==============================================================================================
# ERA5 (Copernicus CDS)
# ==============================================================================================
# Cloud ice is requested on the state levels plus these levels above the state top (ERA5 has them), so the
# ice water path integrates the anvil ice that sits above 200 hPa in deep convection.
ERA5_ICE_EXTRA_LEVELS_HPA = [100, 125, 150, 175]


def era5_request(day: datetime, hours: Sequence[int], levels_hpa: Sequence[int], area: Sequence[float], cloud_ice: bool = True) -> Dict:
    """CDS request body for one day. area = [north, west, south, east] in degrees. With cloud_ice, the
    request also carries specific_cloud_ice_water_content (ciwc) and the extra upper levels."""
    levels = sorted(set(int(p) for p in levels_hpa) | (set(ERA5_ICE_EXTRA_LEVELS_HPA) if cloud_ice else set()))
    return {
        "product_type": ["reanalysis"],
        "variable": ["temperature"] + (["specific_cloud_ice_water_content"] if cloud_ice else []),
        "pressure_level": [str(p) for p in levels],
        "year": [f"{day.year}"],
        "month": [f"{day.month:02d}"],
        "day": [f"{day.day:02d}"],
        "time": [f"{h:02d}:00" for h in sorted(set(hours))],
        "area": [float(a) for a in area],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def download_era5(times: Sequence, levels_hpa: Sequence[int], area: Sequence[float], out_dir: str, tag: str = "storm", cloud_ice: bool = True) -> Dict[str, str]:
    """One CDS request per UTC day covering all requested hours. Returns {YYYY-MM-DD: path}.
    Files written before cloud ice was added (temperature only) are reused as they are; the scene builder
    simply gets no ice for those days (patch them with scripts/add_cloud_ice.py)."""
    import cdsapi

    os.makedirs(out_dir, exist_ok=True)
    by_day: Dict[str, List[int]] = {}
    for t_raw in times:
        t = _to_dt(t_raw)
        by_day.setdefault(t.strftime("%Y-%m-%d"), []).append(t.hour)
    client = cdsapi.Client()
    out = {}
    for day, hours in sorted(by_day.items()):
        dst = os.path.join(out_dir, f"era5_t_{tag}_{day}.nc")
        if not os.path.exists(dst):
            d = datetime.strptime(day, "%Y-%m-%d")
            log.info(f"ERA5 request {day} hours {sorted(set(hours))}")
            _retry(lambda: client.retrieve("reanalysis-era5-pressure-levels", era5_request(d, hours, levels_hpa, area, cloud_ice=cloud_ice), dst), f"ERA5 {day}", attempts=4, base_delay=30.0)
        out[day] = dst
    return out


# ==============================================================================================
# IBTrACS
# ==============================================================================================
def download_ibtracs(out_dir: str, url: str = IBTRACS_URL) -> str:
    import requests

    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.basename(url))
    if not os.path.exists(dst):
        log.info(f"IBTrACS get {url}")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return dst


# ==============================================================================================
# Manifest: ties everything together for build_scene_cache()
# ==============================================================================================
def storm_area(lats: Sequence[float], lons: Sequence[float], pad_deg: float = 4.0) -> List[float]:
    """[north, west, south, east] bounding box around a track with padding (ERA5 request area)."""
    return [float(max(lats) + pad_deg), float(min(lons) - pad_deg), float(min(lats) - pad_deg), float(max(lons) + pad_deg)]


def assemble_manifest(times: Sequence, centers: Sequence[Tuple[float, float]], goes: List[Dict[str, str]], atms: List[Tuple[List[str], Optional[float]]],
                      era5_by_day: Dict[str, str], imerg: List[Optional[str]], storm_name: str, require_all_ir: bool = True) -> List[RawScenePaths]:
    scenes = []
    for t_raw, (lat, lon), g, (a, dt), im in zip(times, centers, goes, atms, imerg):
        t = _to_dt(t_raw)
        era = era5_by_day.get(t.strftime("%Y-%m-%d"))
        if era is None or im is None or (require_all_ir and len(g) == 0):
            log.warning(f"skip {t:%Y-%m-%dT%H:%M}: missing labels or IR")
            continue
        scenes.append(RawScenePaths(time=np.datetime64(t.replace(tzinfo=None)), storm_lat=lat, storm_lon=lon, goes_files=g, atms_files=a, era5_file=era, imerg_file=im, storm_name=storm_name, mw_dt_min=dt))
    return scenes


def save_manifest(scenes: Sequence[RawScenePaths], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([{**asdict(s), "time": str(s.time)} for s in scenes], f, indent=1)


def load_manifest(path: str) -> List[RawScenePaths]:
    with open(path) as f:
        rows = json.load(f)
    return [RawScenePaths(**{**r, "time": np.datetime64(r["time"])}) for r in rows]
