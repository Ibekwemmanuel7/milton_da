"""
Hurricane Melissa (October 2025) with the Milton-trained system: Colab cells.

Second storm, never seen in training (archive 2017 to 2023) or in development (Milton 2024). Scenes were
built on the PC with scripts/prepare_milton.py (--storm MELISSA --season 2025, GOES-19 full disk, ATMS from
NOAA-20 and Suomi-NPP, ERA5, IMERG Late Run) and zipped to MyDrive/milton_da_upload/melissa_scenes.zip.

Runtime: T4 GPU on the main account. Run the usual setup cell first (unzips milton_da.zip and the latest
patch into /content, installs xarray netCDF4 h5netcdf pyproj, runs pytest), then cells M1 to M5 in order.
Checkpoints, normaliser and rtm_audit.json are the Milton ones in MyDrive/milton_artifacts: nothing is retrained.
"""

# ---------- CELL M1: unpack the Melissa scenes ----------
"""
import glob, os
UP = '/content/drive/MyDrive/milton_da_upload'
A = '/content/drive/MyDrive/milton_artifacts'
os.makedirs('/content/data/melissa/scenes', exist_ok=True)
!unzip -o -q $UP/melissa_scenes.zip -d /content/data/melissa/scenes
# if the zip carried a scenes/ folder inside it, flatten it
import shutil
for f in glob.glob('/content/data/melissa/scenes/**/MELISSA_*.nc', recursive=True):
    if os.path.dirname(f) != '/content/data/melissa/scenes':
        shutil.move(f, '/content/data/melissa/scenes/' + os.path.basename(f))
S = sorted(glob.glob('/content/data/melissa/scenes/MELISSA_*.nc'))
print(len(S), 'scenes'); print('\\n'.join(os.path.basename(s) for s in S))
import xarray as xr
ds = xr.open_dataset(S[8]); print(ds.attrs.get('time'), 'ATMS coverage:', round(float(ds['mw_mask'].mean()), 2), '| ATMS files:', ds.attrs.get('atms_files', '')[:60]); ds.close()
"""

# ---------- CELL M2: observation-operator audit on Melissa (compare with the Milton audit) ----------
# Same code as colab/audit_rtm_cell.py, pointed at the Melissa scenes. The retrieval below still calibrates
# from the archive_2023 table, exactly as for Milton; this cell is the diagnostic that says whether the
# operator's per-channel bias on a 2025 GOES-19 storm looks like the one you audited on Milton.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, '/content')
from milton_da.config import PipelineConfig
from milton_da.data.dataset import HurricaneSceneDataset, Normalizer, collate
from milton_da.physics.rtm import AnalyticRTM
from milton_da.scripts.train import apply_preset
cfg = PipelineConfig(); apply_preset(cfg, 'small'); d = cfg.data
norm = Normalizer.load(f'{A}/norm_stats.json')
rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)

def audit(paths, name):
    ds = HurricaneSceneDataset(paths, d, norm, augment=False, downscale=2)
    n_ir, n_mw = len(d.ir_channels), len(d.mw_channels)
    s_ir = np.zeros((n_ir, 3)); s_mw = np.zeros((n_mw, 3))
    with torch.no_grad():
        for i in range(len(ds)):
            b = collate([ds[i]])
            temp, precip = norm.state_to_physical(b['state'], d.precip_log_transform)
            ir_sim, mw_sim = rtm(temp, precip)
            for c in range(n_ir):
                m = b['ir_mask'][0, 0] > 0.5
                e = (ir_sim[0, c] - b['ir_raw'][0, c])[m]; e = e[torch.isfinite(e)]
                s_ir[c] += [e.sum(), (e ** 2).sum(), e.numel()]
            for c in range(n_mw):
                m = b['mw_mask'][0, 0] > 0.5
                e = (mw_sim[0, c] - b['mw_raw'][0, c])[m]; e = e[torch.isfinite(e)]
                s_mw[c] += [e.sum(), (e ** 2).sum(), e.numel()]
    print(f'\\n=== {name}: {len(ds)} scenes, simulated(truth) minus observed ===')
    print(f"{'channel':>12} {'bias K':>8} {'rmse K':>8} {'std K':>8}   verdict")
    out = {}
    for lab, s, sig in [(d.ir_channels, s_ir, cfg.guidance.sigma_ir_K), (d.mw_channels, s_mw, cfg.guidance.sigma_mw_K)]:
        for c, ch in enumerate(lab):
            n = max(s[c, 2], 1); bias = s[c, 0] / n; rmse = np.sqrt(s[c, 1] / n); std = np.sqrt(max(rmse ** 2 - bias ** 2, 0))
            verdict = 'usable' if std < 3 * sig else ('bias-correctable' if std < 6 * sig else 'drop')
            print(f'{str(ch):>12} {bias:8.2f} {rmse:8.2f} {std:8.2f}   {verdict}')
            out[str(ch)] = {'bias_K': float(bias), 'std_K': float(std), 'rmse_K': float(rmse), 'verdict': verdict}
    return out

old = json.load(open(f'{A}/rtm_audit.json'))
res = {'melissa': audit(S, 'Melissa 2025'), 'milton': old.get('milton'), 'archive_2023': old.get('archive_2023')}
json.dump(res, open(f'{A}/rtm_audit_melissa.json', 'w'), indent=1)
print('\\nMilton audit for comparison (ATMS 5 to 9 bias K):', {k: round(v['bias_K'], 1) for k, v in old['milton'].items() if k in ['5','6','7','8','9']})
"""

# ---------- CELL M3: the full retrieval on all 17 scenes (about 65 minutes on a T4) ----------
# Same command as the Milton runs: 8 members, 500 steps, small preset at 128 x 128, ATMS 5 to 9 in the
# physics likelihood, calibrated from the archive_2023 audit table. Output: one *_analysis.nc per scene,
# each holding the ensemble mean, spread, the U-Net proxy and the RMSE against ERA5.
"""
%cd /content
!python -m milton_da.inference.run_milton --scenes /content/data/melissa/scenes/MELISSA_*.nc \
    --stats $A/norm_stats.json --unet $A/checkpoints/unet.pt --score $A/checkpoints/score.pt \
    --out $A/melissa_analysis --preset small --downscale 2 --ensemble 8 --steps 500 \
    --rtm-audit $A/rtm_audit.json --audit-table archive_2023
"""

# ---------- CELL M4 (optional, another 65 minutes): physics-only microwave, the third bar of the comparison ----------
"""
%cd /content
!python -m milton_da.inference.run_milton --scenes /content/data/melissa/scenes/MELISSA_*.nc \
    --stats $A/norm_stats.json --unet $A/checkpoints/unet.pt --score $A/checkpoints/score.pt \
    --out $A/melissa_physics_only --preset small --downscale 2 --ensemble 8 --steps 500 \
    --rtm-audit $A/rtm_audit.json --audit-table archive_2023 --proxy-no-mw
"""

# ---------- CELL M5: score Melissa the way Milton was scored, next to the observed intensity ----------
"""
import glob, json, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
CORE_HALF = 8      # central 16 x 16 pixels at the 128 grid, about 70 km

def score_run(folder, label):
    rows = []
    for f in sorted(glob.glob(f'{folder}/MELISSA_*_analysis.nc')):
        ds = xr.load_dataset(f); sc = xr.load_dataset('/content/data/melissa/scenes/' + os.path.basename(f).replace('_analysis', ''))
        lev = [int(v) for v in ds['level'].values]; k = lev.index(300)
        t_a = ds['temperature'].values[k]; t_u = ds['temperature_unet'].values[k]
        # ERA5 truth at the analysis resolution: block-mean the scene's label to the analysis grid
        tt = sc['temp'].values[k] if 'temp' in sc else None
        if tt is not None and tt.shape != t_a.shape:
            fct = tt.shape[0] // t_a.shape[0]
            tt = tt[: tt.shape[0] - tt.shape[0] % fct, : tt.shape[1] - tt.shape[1] % fct].reshape(t_a.shape[0], fct, t_a.shape[1], fct).mean((1, 3))
        H = t_a.shape[0]; c = H // 2; s = slice(c - CORE_HALF, c + CORE_HALF)
        def rm(a, b, sl=None):
            if b is None: return np.nan
            aa, bb = (a[sl, sl], b[sl, sl]) if sl is not None else (a, b)
            ok = np.isfinite(bb)
            return float(np.sqrt(np.mean((aa[ok] - bb[ok]) ** 2))) if ok.mean() > 0.5 else np.nan
        wc = ds['warm_core_anomaly'].values
        rows.append({'time': str(ds.attrs.get('time'))[:16], 'atms': float(sc['mw_mask'].mean()) > 0.05 if 'mw_mask' in sc else None,
                     'warm_core_300_K': float(wc[k]), 'warm_core_peak_hpa': int(lev[int(np.argmax(wc))]),
                     'core_rmse_300_K': rm(t_a, tt, s), 'core_rmse_300_unet_K': rm(t_u, tt, s),
                     'domain_rmse_300_K': rm(t_a, tt), 'domain_rmse_300_unet_K': rm(t_u, tt),
                     'spread_core_300_K': float(ds['temperature_spread'].values[k][s, s].mean())})
        ds.close(); sc.close()
    df = pd.DataFrame(rows); df['run'] = label
    return df

runs = {'analysis': f'{A}/melissa_analysis'}
if glob.glob(f'{A}/melissa_physics_only/MELISSA_*_analysis.nc'): runs['physics_only'] = f'{A}/melissa_physics_only'
D = pd.concat([score_run(p, n) for n, p in runs.items()], ignore_index=True)
pd.set_option('display.width', 220)
print(D.round(2).to_string(index=False))
a = D[D.run == 'analysis']
print('\\nMelissa, analysis: core RMSE ATMS scenes %.2f K (U-Net %.2f), all scenes %.2f K (U-Net %.2f), domain %.2f K; warm core at 250 to 400 hPa in %d of %d'
      % (a[a.atms == True].core_rmse_300_K.mean(), a[a.atms == True].core_rmse_300_unet_K.mean(), a.core_rmse_300_K.mean(), a.core_rmse_300_unet_K.mean(),
         a.domain_rmse_300_K.mean(), int(a.warm_core_peak_hpa.between(250, 400).sum()), len(a)))
print('Milton, for reference: 1.13 K on the 8 ATMS scenes, 1.77 K on all 16, domain 1.87 K; 15 of 16 warm cores at 250 to 400 hPa')

# observed intensity from IBTrACS, saved by the scene builder
IB = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv'
try:
    ib = pd.read_csv(IB, skiprows=[1], low_memory=False)
except Exception as e:
    print('IBTrACS download failed, skipping the intensity panel:', e); ib = None
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
t = pd.to_datetime(a.time)
ax[0].plot(t, a.warm_core_300_K, 'o-', color='navy', label='analysis warm core, 300 hPa (K)')
ax[0].plot(t, D[D.run=='analysis'].core_rmse_300_K, 's--', color='tab:red', label='inner-core RMSE vs ERA5 (K)')
ax[0].set_title('Melissa: retrieved warm core and inner-core error'); ax[0].grid(alpha=.3); ax[0].legend(); ax[0].tick_params(axis='x', rotation=30)
if ib is not None:
    m = ib[(ib.NAME == 'MELISSA') & (ib.SEASON.astype(int) == 2025)].copy(); m['t'] = pd.to_datetime(m.ISO_TIME)
    m = m[(m.t >= t.min()) & (m.t <= t.max())]
    ax[1].plot(m.t, pd.to_numeric(m.USA_PRES, errors='coerce'), 'k-o', ms=3, label='IBTrACS min SLP (hPa)')
    ax[1].set_title('Observed intensity'); ax[1].grid(alpha=.3); ax[1].legend(); ax[1].tick_params(axis='x', rotation=30)
plt.tight_layout(); os.makedirs(f'{A}/melissa_results', exist_ok=True)
plt.savefig(f'{A}/melissa_results/melissa_summary.png', dpi=140); plt.show()
D.to_csv(f'{A}/melissa_results/melissa_scores.csv', index=False); print('saved to', f'{A}/melissa_results')
"""

# ---------- CELL M6: the warm core through time, with ERA5 as the reference and the overpass markers ----------
# CPU is enough. Needs Drive mounted (A defined), the scenes unpacked (cell M1) and the analyses in $A/melissa_analysis.
# Adds era5_warm_core_300_K and unet_warm_core_300_K to melissa_scores.csv and redraws melissa_summary.png.
"""
import glob, json, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt

def warm_core(t, radius_frac=0.375):
    # same definition as physics/constraints.warm_core_anomaly (radius 48 px on the 128 grid): 5 x 5 centre minus the environment beyond the radius
    H, W = t.shape
    yy, xx = np.mgrid[:H, :W]
    env = np.hypot(yy - H / 2, xx - W / 2) > radius_frac * H
    core = np.nanmean(t[H // 2 - 2 : H // 2 + 3, W // 2 - 2 : W // 2 + 3])
    return float(core - np.nanmean(t[env]))

def to_grid(tt, shape):
    if tt.shape == shape: return tt
    fct = tt.shape[0] // shape[0]
    return tt[: tt.shape[0] - tt.shape[0] % fct, : tt.shape[1] - tt.shape[1] % fct].reshape(shape[0], fct, shape[1], fct).mean((1, 3))

rows = []
for f in sorted(glob.glob(f'{A}/melissa_analysis/MELISSA_*_analysis.nc')):
    ds = xr.load_dataset(f); sc = xr.load_dataset('/content/data/melissa/scenes/' + os.path.basename(f).replace('_analysis', ''))
    lev = [int(v) for v in ds['level'].values]; k = lev.index(300)
    t_a = ds['temperature'].values[k]; t_u = ds['temperature_unet'].values[k]
    tt = to_grid(sc['temp'].values[k], t_a.shape) if 'temp' in sc else None
    rows.append({'time': str(ds.attrs.get('time'))[:16],
                 'analysis_wc': warm_core(t_a), 'unet_warm_core_300_K': warm_core(t_u),
                 'era5_warm_core_300_K': warm_core(tt) if tt is not None else np.nan})
    ds.close(); sc.close()
W = pd.DataFrame(rows)

S = pd.read_csv(f'{A}/melissa_results/melissa_scores.csv')
for c in ['era5_warm_core_300_K', 'unet_warm_core_300_K']:
    if c in S: S = S.drop(columns=c)
S = S.merge(W[['time', 'era5_warm_core_300_K', 'unet_warm_core_300_K']], on='time', how='left')
a = S[S.run == 'analysis'].copy()
chk = float(np.abs(a.warm_core_300_K - W.set_index('time').loc[a.time, 'analysis_wc'].values).max())
print('recomputed analysis warm core matches the saved one to %.3f K' % chk)
pd.set_option('display.width', 220)
print(a[['time', 'atms', 'era5_warm_core_300_K', 'warm_core_300_K', 'unet_warm_core_300_K', 'core_rmse_300_K']].round(2).to_string(index=False))
on, off = a[a.atms == True], a[a.atms == False]
print('\nmean 300 hPa warm core, sounder scenes vs infrared-only scenes:')
print('  ERA5      %.2f vs %.2f K' % (on.era5_warm_core_300_K.mean(), off.era5_warm_core_300_K.mean()))
print('  analysis  %.2f vs %.2f K' % (on.warm_core_300_K.mean(), off.warm_core_300_K.mean()))
print('  U-Net     %.2f vs %.2f K' % (on.unet_warm_core_300_K.mean(), off.unet_warm_core_300_K.mean()))
print('ERA5 peak warm core %.2f K at %s' % (a.era5_warm_core_300_K.max(), a.loc[a.era5_warm_core_300_K.idxmax(), 'time']))

IB = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv'
try:
    ib = pd.read_csv(IB, skiprows=[1], low_memory=False)
except Exception as e:
    print('IBTrACS download failed, skipping the intensity panel:', e); ib = None
t = pd.to_datetime(a.time)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot(t, a.era5_warm_core_300_K, '--', color='gray', lw=1.6, label='ERA5 (reference)')
ax[0].plot(t, a.warm_core_300_K, 'o-', color='navy', label='analysis')
ax[0].plot(t, a.unet_warm_core_300_K, '-', color='tab:orange', lw=1.2, alpha=.9, label='direct U-Net proxy')
ax[0].plot(t[a.atms == True], a.warm_core_300_K[a.atms == True], 'o', color='black', ms=9, mfc='none', mew=1.6, label='ATMS overpass')
ax[0].set_ylabel('core minus environment at 300 hPa (K)'); ax[0].set_title('Melissa: the warm core through time')
ax[0].grid(alpha=.3); ax[0].legend(loc='upper left'); ax[0].tick_params(axis='x', rotation=30)
if ib is not None:
    m = ib[(ib.NAME == 'MELISSA') & (ib.SEASON.astype(int) == 2025)].copy(); m['t'] = pd.to_datetime(m.ISO_TIME)
    m = m[(m.t >= t.min()) & (m.t <= t.max())]
    ax[1].plot(m.t, pd.to_numeric(m.USA_PRES, errors='coerce'), 'k-o', ms=3, label='IBTrACS min SLP (hPa)')
    ax[1].set_title('Observed intensity'); ax[1].grid(alpha=.3); ax[1].legend(); ax[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f'{A}/melissa_results/melissa_summary.png', dpi=140); plt.show()
S.to_csv(f'{A}/melissa_results/melissa_scores.csv', index=False)
a[['time', 'atms', 'era5_warm_core_300_K', 'warm_core_300_K', 'unet_warm_core_300_K', 'core_rmse_300_K', 'core_rmse_300_unet_K', 'domain_rmse_300_K', 'spread_core_300_K']].to_json(
    f'{A}/melissa_results/melissa_warmcore.json', orient='records', indent=1)
print('saved melissa_scores.csv (two new columns), melissa_summary.png and melissa_warmcore.json to', f'{A}/melissa_results')
"""
