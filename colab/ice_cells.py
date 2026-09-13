"""
Cloud-ice cells (levels one to three of the ice work). Paste one at a time after the setup cell
(Drive mounted, A = /content/drive/MyDrive/milton_artifacts, package importable, scenes unpacked under
/content/data/{archive,milton,melissa}/scenes as the other notebooks do).

I0  patch scene files with ERA5 cloud ice (CDS download, one request per storm)      CPU, CDS queue time
I1  ice audit: does ice explain the 89 to 183 GHz depression? fits a_k, I0_k per channel, checks the
    two-stream operator with no fitted parameter, writes rtm_audit_ice.json                         CPU, minutes
I2  level three on the existing checkpoints: all-sky Melissa run, every ATMS channel kept          T4, ~1 h
I3  retrain U-Net and prior with ice water path in the state (new artifacts folder)                T4, hours
I4  Melissa with the ice checkpoints, the two-stream operator and all-sky errors                    T4, ~1 h
"""

# ---------- CELL I0: ERA5 cloud ice into the scene files ----------
"""
import os, getpass, subprocess, sys
if not os.path.exists(os.path.expanduser('~/.cdsapirc')):
    key = getpass.getpass('CDS API key (from https://cds.climate.copernicus.eu/profile): ')
    open(os.path.expanduser('~/.cdsapirc'), 'w').write(f'url: https://cds.climate.copernicus.eu/api\\nkey: {key}\\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cdsapi'], check=True)

RAW = f'{A}/era5_ice'                      # cached CDS downloads, one file per storm, resumable
os.makedirs(RAW, exist_ok=True)
# start with what the audit and the runs need; the full archive (all seasons) is the last line, run it when you have the time
sets = {
    'archive 2023 (validation)': [p for p in __import__('glob').glob('/content/data/archive/scenes/*.nc') if os.path.basename(p).split('_')[-1][:4] == '2023'],
    'milton':  __import__('glob').glob('/content/data/milton/scenes/MILTON_*.nc'),
    'melissa': __import__('glob').glob('/content/data/melissa/scenes/MELISSA_*.nc'),
    # 'archive all seasons': __import__('glob').glob('/content/data/archive/scenes/*.nc'),
}
for name, paths in sets.items():
    if not paths: print(name, ': no scenes found'); continue
    print(f'=== {name}: {len(paths)} scenes')
    subprocess.run([sys.executable, '-m', 'milton_da.scripts.add_cloud_ice', '--scenes', *paths, '--raw', RAW], cwd='/content', check=True)

# keep the patched scenes: zip them back to Drive next to the originals (the originals are untouched on Drive)
import shutil
for folder, out in [('/content/data/archive/scenes', f'{A}/archive_scenes_ice'), ('/content/data/milton/scenes', f'{A}/milton_scenes_ice'), ('/content/data/melissa/scenes', f'{A}/melissa_scenes_ice')]:
    if os.path.isdir(folder):
        shutil.make_archive(out, 'zip', folder); print('saved', out + '.zip')
"""

# ---------- CELL I1: the ice audit ----------
"""
import glob, json, os
import numpy as np, torch, xarray as xr
from milton_da.config import PipelineConfig
from milton_da.data.dataset import HurricaneSceneDataset, Normalizer, collate
from milton_da.physics.rtm import AnalyticRTM
from milton_da.physics.scatter import ScatteringRTM
from milton_da.physics.audit import fit_scatter_depression
from milton_da.scripts.train import apply_preset

cfg = PipelineConfig(); apply_preset(cfg, 'small'); d = cfg.data
norm = Normalizer.load(f'{A}/norm_stats.json')
ana = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
sca = ScatteringRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
F = 2 * d.grid.mw_downscale          # scene (256) -> coarse microwave grid at downscale 2 (16)

def block(a, f):
    H, W = a.shape[-2:]; return a[..., :H - H % f, :W - W % f].reshape(*a.shape[:-2], H // f, f, W // f, f).mean((-3, -1))

def ice_audit(paths, name, max_scenes=80):
    ds = HurricaneSceneDataset(paths[:max_scenes], d, norm, augment=False, downscale=2)
    pairs = {ch: {'iwp': [], 'depression': [], 'sca_err': [], 'rain_err': []} for ch in d.mw_channels}
    n_used = 0
    with torch.no_grad():
        for i in range(len(ds)):
            sc = xr.open_dataset(paths[i])
            if 'iwp' not in sc or float(sc['mw_mask'].mean()) < 0.05: sc.close(); continue
            b = collate([ds[i]]); n_used += 1
            temp, precip = norm.state_to_physical(b['state'], d.precip_log_transform)
            iwp = torch.from_numpy(sc['iwp'].values).float()[None, None]; ciwc = torch.from_numpy(sc['ciwc'].values).float()[None]
            iwp2 = torch.nn.functional.avg_pool2d(iwp, 2); ciwc2 = torch.nn.functional.avg_pool2d(ciwc, 2)       # to the 128 grid
            _, mw_clear = ana.clear_sky(temp)
            _, mw_rain = ana(temp, precip)                                              # the original rain-based depression
            _, mw_sca = sca(temp, precip, {'ciwc': ciwc2, 'iwp': iwp2})                 # two-stream, no fitted parameter
            iwp_c = torch.nn.functional.avg_pool2d(iwp2, d.grid.mw_downscale)[0, 0]
            m = b['mw_mask'][0, 0] > 0.5
            for c, ch in enumerate(d.mw_channels):
                obs = b['mw_raw'][0, c]
                pairs[ch]['iwp'].append(iwp_c[m].numpy()); pairs[ch]['depression'].append((mw_clear[0, c] - obs)[m].numpy())
                pairs[ch]['sca_err'].append((mw_sca[0, c] - obs)[m].numpy()); pairs[ch]['rain_err'].append((mw_rain[0, c] - obs)[m].numpy())
            sc.close()
    print(f'\\n=== {name}: {n_used} scenes with ATMS and cloud ice ===')
    print(f"{'ch':>4} {'clear bias':>10} {'clear std':>9} | {'rain-proxy std':>14} | {'ice fit a_K':>11} {'I0':>6} {'resid std':>9} | {'2-stream bias':>13} {'std':>6} | {'clear-sky std (IWP<0.05)':>24}")
    out = {}
    for ch, v in pairs.items():
        x = np.concatenate(v['iwp']); dep = np.concatenate(v['depression']); se = np.concatenate(v['sca_err']); re = np.concatenate(v['rain_err'])
        fit = fit_scatter_depression(x, dep)
        clear = x < 0.05
        cs = float(dep[clear].std()) if clear.sum() > 100 else float('nan'); cb = float(dep[clear].mean()) if clear.sum() > 100 else float('nan')
        print(f'{ch:>4} {dep.mean():10.2f} {dep.std():9.2f} | {re.std():14.2f} | {fit["a_K"]:11.1f} {fit["I0"]:6.2f} {fit["rmse_K"]:9.2f} | {se.mean():13.2f} {se.std():6.2f} | {cs:24.2f}')
        out[str(ch)] = {**fit, 'clear_bias_K': cb, 'clear_std_K': cs, 'rain_proxy_std_K': float(re.std()),
                        'two_stream_bias_K': float(se.mean()), 'two_stream_std_K': float(se.std()), 'n': int(x.size)}
    return out

sets = {'archive_2023': sorted(p for p in glob.glob('/content/data/archive/scenes/*.nc') if os.path.basename(p).split('_')[-1][:4] == '2023'),
        'milton': sorted(glob.glob('/content/data/milton/scenes/MILTON_*.nc')),
        'melissa': sorted(glob.glob('/content/data/melissa/scenes/MELISSA_*.nc'))}
res = {k: ice_audit(v, k) for k, v in sets.items() if v}
# a clear-sky calibration table for the all-sky runs: bias and sigma measured where ERA5 has no ice
for k in list(res):
    res[k + '_clear'] = {ch: {'bias_K': f['clear_bias_K'], 'std_K': f['clear_std_K'], 'rmse_K': float(np.hypot(f['clear_bias_K'], f['clear_std_K'])),
                              'verdict': 'usable' if f['clear_std_K'] < 6 else 'drop'} for ch, f in res[k].items() if np.isfinite(f['clear_std_K'])}
json.dump(res, open(f'{A}/rtm_audit_ice.json', 'w'), indent=1); print('saved', f'{A}/rtm_audit_ice.json')
print('\\nRead it like this: "clear std" is the scatter the old audit saw (all sky); "rain-proxy std" is what the original operator left;')
print('"resid std" is what is left after the fitted IWP depression; "2-stream std" is the physics with nothing fitted. Smaller is better.')
"""

# ---------- CELL I2: level three now, on the existing checkpoints: all-sky Melissa, every ATMS channel kept ----------
# The audit's clear-sky table gives each channel its bias and clear-sky sigma; the symmetric cloud predictor
# then inflates the error pixel by pixel where either the observation or the state says there is cloud.
"""
!cd /content && python -m milton_da.inference.run_milton \
    --scenes /content/data/melissa/scenes/MELISSA_*.nc --stats $A/norm_stats.json \
    --unet $A/checkpoints/unet.pt --score $A/checkpoints/score.pt \
    --out $A/melissa_allsky --preset small --downscale 2 --ensemble 8 --steps 500 \
    --rtm-audit $A/rtm_audit_ice.json --audit-table archive_2023_clear \
    --mw-channels 5 6 7 8 9 16 17 18 22 --allsky --allsky-slope 0.5
"""

# ---------- CELL I3: retrain with ice water path in the state (after I0 has patched the whole archive) ----------
# New artifacts folder: the normaliser must be refitted with the iwp statistics. Same steps, seed and lambda as
# the baseline so the comparison is fair. Checkpoints record ice='iwp'; run_milton picks it up automatically.
"""
!cd /content && python -m milton_da.scripts.train --archive /content/data/archive/scenes --out $A/ice \
    --preset small --downscale 2 --ice iwp --unet-steps 6000 --score-steps 20000 --lambda-rtm 0.01 --seed 0 --num-workers 4
"""

# ---------- CELL I4: Melissa with the ice checkpoints, the two-stream operator and all-sky errors ----------
"""
!cd /content && python -m milton_da.inference.run_milton \
    --scenes /content/data/melissa/scenes/MELISSA_*.nc --stats $A/ice/norm_stats.json \
    --unet $A/ice/checkpoints/unet.pt --score $A/ice/checkpoints/score.pt \
    --out $A/melissa_ice --preset small --downscale 2 --ensemble 8 --steps 500 \
    --rtm scattering --rtm-audit $A/rtm_audit_ice.json --audit-table archive_2023_clear \
    --mw-channels 5 6 7 8 9 16 17 18 22 --allsky
"""
