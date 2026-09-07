# Observation-operator audit: simulate TB from the ERA5/IMERG truth with the analytic RTM
# and compare channel by channel with what GOES-16 and ATMS actually measured.
# Runs on CPU or GPU in about a minute. Paste as one Colab cell after cells 0 and 1.
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, "/content")
from milton_da.config import PipelineConfig
from milton_da.data.dataset import HurricaneSceneDataset, Normalizer, collate
from milton_da.physics.rtm import AnalyticRTM
from milton_da.scripts.train import apply_preset

A = "/content/drive/MyDrive/milton_artifacts"
cfg = PipelineConfig(); apply_preset(cfg, "small"); d = cfg.data
norm = Normalizer.load(f"{A}/norm_stats.json")
rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)

def audit(paths, name, max_scenes=60):
    ds = HurricaneSceneDataset(paths[:max_scenes], d, norm, augment=False, downscale=2)
    n_ir, n_mw = len(d.ir_channels), len(d.mw_channels)
    s_ir = np.zeros((n_ir, 3)); s_mw = np.zeros((n_mw, 3))      # sum err, sum err^2, count
    with torch.no_grad():
        for i in range(len(ds)):
            b = collate([ds[i]])
            temp, precip = norm.state_to_physical(b["state"], d.precip_log_transform)
            ir_sim, mw_sim = rtm(temp, precip)
            for c in range(n_ir):
                m = b["ir_mask"][0, 0] > 0.5
                e = (ir_sim[0, c] - b["ir_raw"][0, c])[m]; e = e[torch.isfinite(e)]
                s_ir[c] += [e.sum(), (e**2).sum(), e.numel()]
            for c in range(n_mw):
                m = b["mw_mask"][0, 0] > 0.5
                e = (mw_sim[0, c] - b["mw_raw"][0, c])[m]; e = e[torch.isfinite(e)]
                s_mw[c] += [e.sum(), (e**2).sum(), e.numel()]
    print(f"\n=== {name}: {len(ds)} scenes, simulated(truth) minus observed ===")
    print(f"{'channel':>12} {'bias K':>8} {'rmse K':>8} {'std K':>8}   verdict")
    out = {}
    for lab, s, sig in [(d.ir_channels, s_ir, cfg.guidance.sigma_ir_K), (d.mw_channels, s_mw, cfg.guidance.sigma_mw_K)]:
        for c, ch in enumerate(lab):
            n = max(s[c, 2], 1); bias = s[c, 0] / n; rmse = np.sqrt(s[c, 1] / n); std = np.sqrt(max(rmse**2 - bias**2, 0))
            verdict = "usable" if std < 3 * sig else ("bias-correctable" if std < 6 * sig else "drop")
            print(f"{str(ch):>12} {bias:8.2f} {rmse:8.2f} {std:8.2f}   {verdict}")
            out[str(ch)] = {"bias_K": float(bias), "std_K": float(std), "rmse_K": float(rmse), "verdict": verdict}
    return out

milton = sorted(glob.glob("/content/data/milton/scenes/MILTON_*.nc"))
val = sorted(p for p in glob.glob("/content/data/archive/scenes/*.nc") if os.path.basename(p).split("_")[-1][:4] == "2023")
res = {"milton": audit(milton, "Milton 2024"), "archive_2023": audit(val, "archive validation (2023 season)")}
json.dump(res, open(f"{A}/rtm_audit.json", "w"), indent=1)
print(f"\nsaved {A}/rtm_audit.json")
