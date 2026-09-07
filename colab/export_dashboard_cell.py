# Export everything the presentation dashboard needs into one JSON on Drive.
# Reads every run folder under milton_artifacts that contains *_analysis.nc files, the prior samples,
# the RTM audit and the ERA5/IMERG truth from the scene files. Fields are block-averaged to 64 x 64
# and rounded so the file stays a few MB. Paste as one Colab cell (needs cells 0 and 1).
import glob, json, os
import numpy as np, xarray as xr

A = "/content/drive/MyDrive/milton_artifacts"
SCENES = "/content/data/milton/scenes"
OUT = f"{A}/dashboard_data.json"

def coarsen(a, n=64):
    a = np.asarray(a, np.float32)
    f = max(a.shape[-1] // n, 1)
    if f > 1:
        H, W = a.shape[-2:]
        a = a[..., : H - H % f, : W - W % f].reshape(*a.shape[:-2], H // f, f, W // f, f).mean((-3, -1))
    return np.round(np.nan_to_num(a, nan=0.0), 1).tolist()

def warm_core(t, radius_frac=0.375):
    L, H, W = t.shape
    yy, xx = np.mgrid[:H, :W]; r = np.hypot(yy - H / 2, xx - W / 2)
    env = r > radius_frac * H
    core = t[:, H // 2 - 2 : H // 2 + 3, W // 2 - 2 : W // 2 + 3].mean((-2, -1))
    return core - t[:, env].mean(-1)

runs = {}
for folder in sorted(glob.glob(f"{A}/*/")):
    files = sorted(glob.glob(f"{folder}/MILTON_*_analysis.nc"))
    if not files:
        continue
    name = os.path.basename(folder.rstrip("/"))
    scenes = []
    for f in files:
        ds = xr.load_dataset(f)
        lev = [int(v) for v in ds["level"].values]
        t = ds["temperature"].values
        rec = {
            "time": str(ds.attrs.get("time", os.path.basename(f)[7:23])),
            "levels_hpa": lev,
            "warm_core_K": np.round(ds["warm_core_anomaly"].values, 2).tolist(),
            "warm_core_unet_K": np.round(warm_core(ds["temperature_unet"].values), 2).tolist(),
            "temp_rmse_vs_era5_K": np.round(ds["temperature_rmse_vs_era5"].values, 2).tolist() if "temperature_rmse_vs_era5" in ds else None,
            "precip_rmse_vs_imerg": float(ds["precip_rmse_vs_imerg"]) if "precip_rmse_vs_imerg" in ds else None,
            "spread_mean_K": np.round(ds["temperature_spread"].values.mean((-2, -1)), 2).tolist(),
            "ensemble_size": int(ds.attrs.get("ensemble_size", 0)), "n_steps": int(ds.attrs.get("n_steps", 0)),
            "fields": {
                "T300": coarsen(t[lev.index(300)]), "T500": coarsen(t[lev.index(500)]), "T850": coarsen(t[lev.index(850)]),
                "precip": coarsen(ds["precip"].values), "precip_unet": coarsen(ds["precip_unet"].values),
                "T300_unet": coarsen(ds["temperature_unet"].values[lev.index(300)]),
                "xsec_T": coarsen(t[:, t.shape[1] // 2, :], 128),                      # level x east-west line through centre
                "ir_obs_C13": coarsen(ds["ir_tb_observed"].sel(ir_channel="C13").values),
                "ir_sim_C13": coarsen(ds["ir_tb_simulated"].sel(ir_channel="C13").values),
                "mw_obs_7": coarsen(ds["mw_tb_observed"].sel(mw_channel=7).values, 32),
                "mw_sim_7": coarsen(ds["mw_tb_simulated"].sel(mw_channel=7).values, 32),
            },
        }
        # truth from the scene file (ERA5 / IMERG), matched by basename
        sp = os.path.join(SCENES, os.path.basename(f).replace("_analysis", ""))
        if os.path.exists(sp):
            s = xr.load_dataset(sp)
            tt = s["temp"].values; slev = [int(v) for v in s["level"].values]
            rec["warm_core_era5_K"] = np.round(warm_core(tt), 2).tolist()
            rec["truth"] = {"T300": coarsen(tt[slev.index(300)]), "T850": coarsen(tt[slev.index(850)]),
                            "precip": coarsen(s["precip"].values), "xsec_T": coarsen(tt[:, tt.shape[1] // 2, :], 128)}
            rec["storm_lat"], rec["storm_lon"] = float(s.attrs.get("storm_lat", np.nan)), float(s.attrs.get("storm_lon", np.nan))
        scenes.append(rec)
    runs[name] = scenes
    print(f"{name}: {len(scenes)} scenes")

out = {"runs": runs}
if os.path.exists(f"{A}/rtm_audit.json"):
    out["rtm_audit"] = json.load(open(f"{A}/rtm_audit.json"))
if os.path.exists(f"{A}/prior_samples.npz"):
    z = np.load(f"{A}/prior_samples.npz")
    lev = [int(v) for v in z["levels_hpa"]]
    out["prior_samples"] = {"levels_hpa": lev, "T400": [coarsen(z["temperature"][i, lev.index(400)]) for i in range(len(z["temperature"]))],
                            "T1000": [coarsen(z["temperature"][i, -1]) for i in range(len(z["temperature"]))],
                            "precip": [coarsen(z["precip"][i]) for i in range(len(z["precip"]))]}
out["training"] = {"unet": {"steps": 6000, "lambda_rtm": 0.01, "val_temp_rmse_K": 2.55, "val_precip_rmse_mmh": 4.05},
                   "prior": {"steps": 60000, "dsm_final": 0.005}, "archive": {"train_scenes": 805, "val_scenes": 252, "val_season": 2023}}
json.dump(out, open(OUT, "w"))
print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
