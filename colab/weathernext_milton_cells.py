"""
WeatherNext Cyclones Mini on Hurricane Milton: cells to append to the official demo notebook.

The official notebook
  https://colab.research.google.com/github/google-deepmind/weathernext/blob/master/docs/weathernext2/wn2_demo.ipynb
ships with Milton as its sample case: model WeatherNextCyclones_Mini_<2024 (1 degree, trained on data
before 2024, so Milton is out of sample), HRES initial conditions at 2024-10-07 00 UTC, 20 six-hour
steps to 2024-10-12 00 UTC. Runtime: Colab "TPU v5e-1" (free tier). Run the notebook's own cells
through "Run the tracker" first (cells 3 to 18), with num_ensemble_members = 8, then append W1 to W5.

Nothing here touches the model; these cells only score and save what the notebook produced.
"""

# ---------- CELL W0 (run BEFORE the notebook's cells, once): mount Drive for outputs ----------
"""
from google.colab import drive
drive.mount('/content/drive')
import os
WN = '/content/drive/MyDrive/milton_artifacts/weathernext'
os.makedirs(WN, exist_ok=True)
print(WN)
"""

# ---------- CELL W1: identify Milton in IBTrACS and in every predicted ensemble track ----------
"""
import numpy as np, pandas as pd
from weathernext.cyclones import constants as C

LAT, LON, T, P, V, ID = C.LAT, C.LON, C.VALID_TIME, C.MIN_SEA_LEVEL_PRESSURE_HPA, C.MAX_SUSTAINED_WIND_SPEED_KNOTS, C.TRACK_ID
init = pd.Timestamp(init_time)

# Milton at 2024-10-07 00 UTC was near 22.0 N, 93.9 W (266.1 E). Pick the observed storm closest to that.
obs0 = initial_storms_df[initial_storms_df[T] == init].copy()
d = np.hypot(obs0[LAT] - 22.0, (obs0[LON] % 360) - 266.1)
milton_id = obs0.loc[d.idxmin(), ID]
obs = all_storms_df[(all_storms_df[ID] == milton_id) & (all_storms_df[T] >= init)].sort_values(T).reset_index(drop=True)
print('Milton IBTrACS id:', milton_id, '| observed points from init:', len(obs))
print(obs[[T, LAT, LON, P, V]].head(24).to_string(index=False))

def pick_milton(df):
    # the predicted track that starts closest to Milton's observed initial position
    tcol = T if T in df.columns else None
    if tcol is None:
        df = df.copy(); df[T] = init + pd.to_timedelta(df[C.LEAD_TIME])
    first = df.sort_values(T).groupby(ID).first()
    dd = np.hypot(first[LAT] - obs.loc[0, LAT], (first[LON] % 360) - (obs.loc[0, LON] % 360))
    return df[df[ID] == dd.idxmin()].sort_values(T).reset_index(drop=True)

members = [pick_milton(t) for t in predicted_ensemble_tracks]
print('predicted track columns:', list(members[0].columns))
for i, m in enumerate(members):
    print(f'member {i}: {len(m)} points, last {m[T].iloc[-1]}, min slp {m[P].min():.0f} hPa, max wind {m[V].max():.0f} kt')
"""

# ---------- CELL W2: score track and intensity against IBTrACS, per lead time ----------
"""
R = 6371.0
def gc_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians((lon2 - lon1 + 180) % 360 - 180)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

rows = []
for i, m in enumerate(members):
    j = m.merge(obs[[T, LAT, LON, P, V]], on=T, suffixes=('', '_obs'))
    for _, r in j.iterrows():
        rows.append({'member': i, 'valid_time': r[T], 'lead_h': int((r[T] - init) / pd.Timedelta('1h')),
                     'track_err_km': gc_km(r[LAT], r[LON], r[LAT + '_obs'], r[LON + '_obs']),
                     'slp_pred': r[P], 'slp_obs': r[P + '_obs'], 'wind_pred_kt': r[V], 'wind_obs_kt': r[V + '_obs']})
S = pd.DataFrame(rows)
S['slp_err'] = S.slp_pred - S.slp_obs
S['wind_err_kt'] = S.wind_pred_kt - S.wind_obs_kt

# ensemble-mean position error uses the mean position, not the mean of member errors
mean_pos = (pd.concat(members).groupby(T)[[LAT, LON]].mean().reset_index()
            .merge(obs[[T, LAT, LON]], on=T, suffixes=('', '_obs')))
mean_pos['lead_h'] = ((mean_pos[T] - init) / pd.Timedelta('1h')).astype(int)
mean_pos['track_err_km'] = gc_km(mean_pos[LAT], mean_pos[LON], mean_pos[LAT + '_obs'], mean_pos[LON + '_obs'])

summary = (S.groupby('lead_h').agg(members=('member', 'nunique'),
                                   mean_member_track_err_km=('track_err_km', 'mean'),
                                   slp_pred_mean=('slp_pred', 'mean'), slp_obs=('slp_obs', 'first'),
                                   wind_pred_mean_kt=('wind_pred_kt', 'mean'), wind_obs_kt=('wind_obs_kt', 'first'))
           .join(mean_pos.set_index('lead_h')['track_err_km'].rename('ens_mean_track_err_km')).reset_index())
pd.set_option('display.width', 200)
print(summary.round(1).to_string(index=False))

# landfall: first time the observed track is over Florida (lon > 277.0 E, i.e. west of 83 W is Gulf); use obs lat/lon crossing
land = obs[(obs[LON] % 360) > 277.3]
print('\\nobserved: min slp %.0f hPa at %s; max wind %.0f kt; first point east of 82.7 W: %s'
      % (obs[P].min(), obs.loc[obs[P].idxmin(), T], obs[V].max(), land[T].iloc[0] if len(land) else 'n/a'))
mins = [m[P].min() for m in members]
print('members: min slp mean %.0f hPa (range %.0f to %.0f); max wind mean %.0f kt (range %.0f to %.0f)'
      % (np.mean(mins), min(mins), max(mins), np.mean([m[V].max() for m in members]),
         min(m[V].max() for m in members), max(m[V].max() for m in members)))
"""

# ---------- CELL W3: figures (track map, intensity vs time, track error vs lead) ----------
"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for i, m in enumerate(members):
    ax[0].plot(((m[LON] + 180) % 360) - 180, m[LAT], '-', color='tab:blue', alpha=.45, lw=1.2, label='members' if i == 0 else None)
mp = pd.concat(members).groupby(T)[[LAT, LON]].mean()
ax[0].plot(((mp[LON] + 180) % 360) - 180, mp[LAT], '-', color='navy', lw=2.2, label='ensemble mean')
ax[0].plot(((obs[LON] + 180) % 360) - 180, obs[LAT], 'k-o', ms=3, lw=1.8, label='IBTrACS (observed)')
ax[0].set_xlim(-98, -70); ax[0].set_ylim(16, 34); ax[0].grid(alpha=.3); ax[0].legend(loc='lower right')
ax[0].set_title('Track: WeatherNext Cyclones Mini (1 deg), init 2024-10-07 00 UTC')
for i, m in enumerate(members):
    ax[1].plot(m[T], m[P], color='tab:blue', alpha=.45, lw=1.2)
ax[1].plot(obs[T], obs[P], 'k-o', ms=3, lw=1.8)
ax[1].set_ylabel('minimum sea-level pressure (hPa)'); ax[1].grid(alpha=.3); ax[1].tick_params(axis='x', rotation=30)
ax[1].set_title('Intensity: members (blue) vs observed (black)')
ax[2].plot(summary.lead_h, summary.ens_mean_track_err_km, 'o-', color='navy', label='ensemble-mean position')
ax[2].plot(summary.lead_h, summary.mean_member_track_err_km, 's--', color='tab:blue', label='mean member error')
ax[2].set_xlabel('lead time (h)'); ax[2].set_ylabel('track error (km)'); ax[2].grid(alpha=.3); ax[2].legend()
ax[2].set_title('Track error vs lead time')
plt.tight_layout(); plt.savefig(f'{WN}/milton_weathernext_mini.png', dpi=140); plt.show()
"""

# ---------- CELL W4: save everything the dashboard and drill need ----------
"""
import json
S.to_csv(f'{WN}/milton_member_scores.csv', index=False)
summary.to_csv(f'{WN}/milton_summary_by_lead.csv', index=False)
pd.concat([m.assign(member=i) for i, m in enumerate(members)]).to_csv(f'{WN}/milton_member_tracks.csv', index=False)
obs.to_csv(f'{WN}/milton_ibtracs.csv', index=False)
meta = {'model': model_name, 'split': split, 'resolution_deg': data_resolution, 'steps': int(steps),
        'init_time': str(init), 'members': len(members), 'data_path': data_path, 'weights_path': weights_path,
        'backend': jax.default_backend(), 'devices': len(jax.local_devices())}
json.dump(meta, open(f'{WN}/milton_run_meta.json', 'w'), indent=1)
print(json.dumps(meta, indent=1)); print('saved to', WN)
"""

# ---------- CELL W5 (optional): the 300 hPa temperature the model carries at Milton's centre ----------
# Compares the forecast model's own view of the warm core, at 1 degree, with the observed track position.
"""
tmp = predictions['temperature'].sel(level=300).isel(batch=0)
lat_c, lon_c = obs.loc[0, LAT], obs.loc[0, LON] % 360
box = tmp.sel(lat=slice(lat_c - 5, lat_c + 5), lon=slice(lon_c - 5, lon_c + 5)) if tmp.lat[0] < tmp.lat[-1] else tmp.sel(lat=slice(lat_c + 5, lat_c - 5), lon=slice(lon_c - 5, lon_c + 5))
anom = (box.max(('lat', 'lon')) - box.mean(('lat', 'lon')))
print('300 hPa warm anomaly at first step, ensemble mean: %.2f K (members %.2f to %.2f)' % (
    float(anom.isel(time=0).mean('sample')), float(anom.isel(time=0).min('sample')), float(anom.isel(time=0).max('sample'))))
print('Compare with the warm-core panel on the milton_da dashboard at 07 Oct 00 UTC (an IR-only scene) and the ERA5 peak of 4.4 K.')
"""
