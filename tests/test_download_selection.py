"""Offline tests of the archive selection logic (no network)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from milton_da.data.best_track import BestTrack
from milton_da.data.download import (Granule, era5_request, goes_prefixes, goes_sector_for, parse_goes_times,
                                     select_atms_overpass, select_goes_key, select_imerg, storm_area)

UTC = timezone.utc


def test_goes_filename_parsing_and_selection():
    keys = [
        "noaa-goes16/ABI-L2-CMIPF/2024/281/12/OR_ABI-L2-CMIPF-M6C13_G16_s20242811150205_e20242811159525_c20242811200001.nc",
        "noaa-goes16/ABI-L2-CMIPF/2024/281/12/OR_ABI-L2-CMIPF-M6C13_G16_s20242811200205_e20242811209525_c20242811210001.nc",
        "noaa-goes16/ABI-L2-CMIPF/2024/281/12/OR_ABI-L2-CMIPF-M6C13_G16_s20242811210205_e20242811219525_c20242811220001.nc",
        "noaa-goes16/ABI-L2-CMIPF/2024/281/12/OR_ABI-L2-CMIPF-M6C08_G16_s20242811200205_e20242811209525_c20242811210001.nc",
    ]
    s, e = parse_goes_times(keys[1])
    assert s == datetime(2024, 10, 7, 12, 0, 20, 500000, tzinfo=UTC) and (e - s).total_seconds() > 500
    t = datetime(2024, 10, 7, 12, 0, tzinfo=UTC)
    assert select_goes_key(keys, t, "C13", 7.5) == keys[1]        # mid-time 12:05 is nearest to 12:00
    assert select_goes_key(keys, t, "C08", 7.5) == keys[3]
    assert select_goes_key(keys, t, "C10", 7.5) is None
    assert select_goes_key(keys, t + timedelta(minutes=30), "C13", 7.5) is None
    assert goes_prefixes(t, "ABI-L2-CMIPF")[1] == "noaa-goes16/ABI-L2-CMIPF/2024/281/12/"
    assert goes_sector_for(22.0, -91.0) == "ABI-L2-CMIPC" and goes_sector_for(12.0, -40.0) == "ABI-L2-CMIPF"


def test_atms_overpass_clustering():
    t = datetime(2024, 10, 7, 12, 0, tzinfo=UTC)
    mk = lambda sn, m0: Granule(sn, t + timedelta(minutes=m0), t + timedelta(minutes=m0 + 6), f"https://x/{sn}_{m0}.nc", f"{sn}_{m0}.nc")
    gr = [mk("SNPPATMSL1B", -80), mk("SNPPATMSL1B", -74),                 # one overpass, two granules
          mk("SNDRJ1ATMSL1B", 20), mk("SNDRJ1ATMSL1B", 26), mk("SNDRJ1ATMSL1B", 32),   # nearer overpass
          mk("SNDRJ1ATMSL1B", 130)]                                        # outside tolerance
    chosen, dt = select_atms_overpass(gr, t, tol_min=90)
    assert [g.name for g in chosen] == ["SNDRJ1ATMSL1B_20.nc", "SNDRJ1ATMSL1B_26.nc", "SNDRJ1ATMSL1B_32.nc"]
    assert abs(dt - 29.0) < 1e-6
    assert select_atms_overpass(gr, t + timedelta(hours=6), tol_min=90) == ([], None)
    # same granule in two product versions (seen on GES DISC for Milton): keep only the newest
    dup = [Granule("SNDRJ1ATMSL1B", t, t + timedelta(minutes=6), "u", "SNDR.J1.ATMS.20241007T1824.m06.g185.L1B.std.v03_15.G.x.nc"),
           Granule("SNDRJ1ATMSL1B", t, t + timedelta(minutes=6), "u", "SNDR.J1.ATMS.20241007T1824.m06.g185.L1B.std.v02_11.G.x.nc")]
    chosen, _ = select_atms_overpass(dup, t, tol_min=90)
    assert len(chosen) == 1 and "v03_15" in chosen[0].name
    # centre coverage beats time proximity: a nearer overpass that only clips the domain loses
    box = lambda w, e, s_, n: [(w, s_), (e, s_), (e, n), (w, n)]
    near_miss = Granule("SNPPATMSL1B", t + timedelta(minutes=5), t + timedelta(minutes=11), "u", "near.nc", polygon=box(-95, -92.5, 18, 26))
    covering = Granule("SNDRJ1ATMSL1B", t + timedelta(minutes=70), t + timedelta(minutes=76), "u", "cover.nc", polygon=box(-93, -85, 18, 26))
    chosen, dt = select_atms_overpass([near_miss, covering], t, tol_min=90, center=(21.7, -91.3))
    assert chosen[0].name == "cover.nc" and abs(dt - 73) < 1e-6
    assert covering.contains(21.7, -91.3) is True and near_miss.contains(21.7, -91.3) is False


def test_imerg_and_era5_and_area():
    t = datetime(2024, 10, 7, 12, 10, tzinfo=UTC)
    g = [Granule("GPM_3IMERGHH", t.replace(minute=0), t.replace(minute=29, second=59), "u1", "a"), Granule("GPM_3IMERGHH", t.replace(minute=30), t.replace(minute=59), "u2", "b")]
    assert select_imerg(g, t).name == "a"
    req = era5_request(datetime(2024, 10, 7), [0, 6, 6, 12], [200, 500, 1000], [30, -100, 15, -80], cloud_ice=False)
    assert req["time"] == ["00:00", "06:00", "12:00"] and req["pressure_level"] == ["200", "500", "1000"] and req["day"] == ["07"]
    req = era5_request(datetime(2024, 10, 7), [0], [200, 500, 1000], [30, -100, 15, -80])
    assert req["variable"] == ["temperature", "specific_cloud_ice_water_content"]
    assert req["pressure_level"] == ["100", "125", "150", "175", "200", "500", "1000"]
    assert storm_area([20, 25], [-95, -85], 4) == [29.0, -99.0, 16.0, -81.0]


def test_ibtracs_name_lookup_and_listing(tmp_path):
    rows = ["SID,SEASON,NUMBER,BASIN,NAME,ISO_TIME,LAT,LON,USA_WIND", ",Year,,,,,degrees_north,degrees_east,kts"]
    for i, (name, season, w) in enumerate([("MILTON", 2024, 155), ("HELENE", 2024, 120), ("MILTON", 2018, 40)]):
        for k in range(3):
            rows.append(f"S{i},{season},1,NA,{name},{season}-10-0{k+5} 00:00:00,{20+k},{-90+k},{w - 10*k}")
    p = tmp_path / "ib.csv"; p.write_text("\n".join(rows))
    bt = BestTrack.from_ibtracs_csv(str(p), name="milton", season=2024)
    assert bt.sid == "S0" and bt.center_at(np.datetime64("2024-10-06T12:00")) == (21.5, -88.5)
    st = BestTrack.list_storms(str(p), [2024], ["NA"], 64)
    assert sorted(st["name"]) == ["HELENE", "MILTON"]
