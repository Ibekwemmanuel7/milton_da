# Real-data run plan: Hurricane Milton and the training archive

## 0. Accounts and environment (day 1)

1. NASA Earthdata account (free): https://urs.earthdata.nasa.gov. In the profile, approve the
   application "NASA GESDISC DATA ARCHIVE". Put credentials in `~/.netrc`
   (`machine urs.earthdata.nasa.gov login USER password PASS`, `chmod 600`) or export
   `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.
2. Copernicus CDS account (free): https://cds.climate.copernicus.eu. Accept the ERA5 licence on the
   dataset page once, then write `~/.cdsapirc` with `url: https://cds.climate.copernicus.eu/api` and
   `key: <token>`.
3. GOES needs nothing (anonymous S3).
4. `pip install -e . s3fs earthaccess cdsapi h5netcdf` and run `pytest -q milton_da/tests` (9 tests,
   the 4 archive-selection tests are offline).

Smoke-check each source with one scene before launching anything large:

    python -m milton_da.scripts.prepare_milton --root data/milton --times 2024-10-07T18:00 --dry-run
    python -m milton_da.scripts.prepare_milton --root data/milton --times 2024-10-07T18:00

## 1. Milton case (week 1)

Best track (NHC report AL142024): depression 1200 UTC 5 Oct east of Tampico; explosive
intensification from about 0325 UTC 7 Oct, 977 mb to 895 mb by 2000 UTC 7 Oct (155 kt, 21.8N 90.9W);
landfall 0030 UTC 10 Oct at Siesta Key at 100 kt / ~958 mb.

Analysis times (UTC), chosen to bracket the three regimes and to fall on ERA5 hours:

| Regime | Times | Why |
|---|---|---|
| Pre-RI, small Gulf storm | 06 Oct 12, 18 | weak warm core, cirrus shield forming; baseline for the retrieval |
| Explosive intensification | 07 Oct 00, 06, 12, 18, 20 | warm core deepens by >10 K in 16 h; pinhole eye ~ 5 km, hardest resolution test |
| Peak and Yucatan approach | 08 Oct 00, 06, 12, 18 | mature Cat 5, eyewall replacement cycle begins |
| Shear and expansion | 09 Oct 00, 06, 12, 18 | asymmetric convection, eye opening, wind field broadening |
| Landfall | 10 Oct 00 | coastal, land in the domain, ERA5 label quality drops |

That is 16 scenes; `prepare_milton.py` defaults to 6-hourly from 06 Oct 00 to 10 Oct 00 and
accepts `--times` for the list above. The storm stays inside the GOES-East CONUS sector for the whole
period, so `ABI-L2-CMIPC` (5-min, ~5 MB per channel) is selected automatically.

ATMS coverage: three platforms (SNPP, NOAA-20, NOAA-21) in the same 1330 local-time orbit give
overpasses clustered around 07-09 UTC and 18-20 UTC each day over the Gulf. Expect roughly half of
the 16 analysis times to have an overpass within the 90-minute tolerance; the manifest records the
offset (`mw_dt_min`) and the scene attribute carries it. Analysis times at 00 and 12 UTC will mostly
be infrared-only, which is itself an experiment (Section 4).

Volume: 16 scenes x (3 x 5 MB GOES + 2-3 x 10 MB ATMS + 10 MB IMERG) plus ~5 ERA5 day files
(~20 MB each) is under 1 GB. Wall time is dominated by CDS queueing (minutes to an hour per request).

## 2. Training archive (weeks 1-2, runs in the background)

    python -m milton_da.scripts.prepare_archive --root data/archive --list-only
    python -m milton_da.scripts.prepare_archive --root data/archive --max-storms 12      # first pass
    python -m milton_da.scripts.prepare_archive --root data/archive                     # full

Selection: North Atlantic storms 2018-2023 that reached hurricane strength (64 kt), 3-hourly
analysis times while the storm is at or above 34 kt, Milton and all 2024 storms held out. GOES-16
has been GOES-East since 18 Dec 2017, so every season is covered by one satellite. Expected size:
about 35-45 storms, 1,500-2,500 scenes, 40-80 GB raw (full-disk files dominate for storms east of
60W), 15-25 GB of scene NetCDF. Start with `--max-storms 12` (the most intense storms) to get the
first prior trained while the rest downloads.

Suggested split: hold out 2023 entirely for validation (Idalia, Lee, Franklin) in addition to Milton
as the test case. Fit the `Normalizer` on the training split only.

## 3. Training (weeks 2-3, one GPU)

| Stage | Data | Steps | Approximate time on one A100 |
|---|---|---|---|
| 0 RTM residual (optional first pass) | all scenes with ATMS | 5k | 1 h |
| 1 U-Net proxy | train split, augment on | 30-50k at batch 8 | 8-12 h |
| 2 Score prior | train split states only | 150-300k at batch 8 | 2-3 days |

Start the prior at 128 x 128 (crop or 2x coarsen the scenes) to get a usable model in a day, then
fine-tune at 256 x 256. Watch the DSM loss and, every few thousand steps, sample the unconditional
prior and look at it: eyewall rings and a warm core aloft should appear before any guidance is used.
If they do not, the prior is not ready and the guided results will not be meaningful.

## 4. Milton retrievals and experiments (week 3-4)

    python -m milton_da.inference.run_milton --scenes data/milton/scenes/MILTON_*.nc \
        --stats artifacts/norm_stats.json --unet artifacts/checkpoints/unet.pt \
        --score artifacts/checkpoints/score.pt --out artifacts/milton --ensemble 8 --steps 500

Experiments, each a single flag change:

1. Full system vs U-Net proxy vs unguided prior (the three bars of the report, on real data).
2. Infrared-only vs infrared + ATMS at the times that have an overpass: quantifies what the sounder
   adds to the warm core (`run_milton --no-mw`).
3. Guidance ablation: `--no-rtm` drops the radiative terms; set `lambda_stability = 0` or `sigma_unet = 1e6` in the config for the others.
4. Observation-error sensitivity: sigma_ir in {3, 5, 8} K, sigma_mw in {1.5, 2, 3} K.
5. Ensemble calibration: spread-skill scatter and rank histograms against ERA5 over all 16 scenes.

## 5. Verification against independent data

* NOAA P-3 / G-IV dropsondes: 13 NOAA and 9 USAF missions flew Milton. The HRD dropsonde archive
  (https://www.aoml.noaa.gov/hrd/data_sub/dropsonde.html) provides quality-controlled profiles; use
  the eye and eyewall sondes to verify the warm-core anomaly at 300-500 hPa and the low-level
  temperature, which ERA5 cannot resolve. Match sondes to the nearest analysis time within 3 h and
  to the retrieval by storm-relative position.
* HAFS-A/B analyses (NOAA NOMADS archive) and the NHC best-track intensity: compare the retrieved
  warm-core magnitude with the pressure deficit through the hypsometric relation.
* IMERG and ground radar (KTBW Tampa NEXRAD near landfall) for the precipitation field.
* Observation space: simulated vs observed brightness temperature per channel, including the
  channels not used in the likelihood if you hold one out (e.g. train the likelihood without ATMS 7
  and verify against it).

## 6. Risks and mitigations

* CDS queue times are unpredictable: request whole storm-days, never single hours, and start the
  archive ERA5 pulls first.
* GES DISC returns 401 until the application is approved in the Earthdata profile.
* ATMS granules straddle the swath edge: the coverage mask handles it, but check `mw_mask.mean()`
  per scene and drop scenes with less than 30 percent coverage from the MW ablation.
* Land in the domain near landfall breaks the ocean-surface assumptions of the analytic operator's
  window channels; either mask land pixels in the likelihood (add a land mask to the scene) or stop
  the case at 09 Oct 18 UTC for the headline figures.
* The analytic RTM's representativeness error on real data will be larger than on synthetic data:
  start with sigma_ir = 8 K, then tighten after fitting the residual.

## 7. Deliverables for the portfolio

1. Real Milton retrieval figures: warm-core cross-sections at the five RI times with dropsonde
   overlays, precipitation vs IMERG, ensemble spread maps.
2. The three-bar comparison (prior, proxy, guided) on real data, plus the IR-only vs IR+MW ablation.
3. A public repository with the scene cache manifest (not the raw data), trained checkpoints,
   CI running the 9 tests, and a README that reproduces one figure end to end.
4. A short write-up (blog or arXiv-style note) built from the technical report with the real
   results substituted for the synthetic ones.
