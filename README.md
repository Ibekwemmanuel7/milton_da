# milton_da: physics-guided score-based data assimilation for Hurricane Milton (October 2024)

A PyTorch package that reconstructs the three-dimensional temperature structure (200 to 1000 hPa)
and surface rain rate of Hurricane Milton from GOES-16 infrared imagery and NOAA-20 ATMS microwave
radiances, with no forecast model. A cross-attention U-Net provides a deterministic multi-sensor
proxy, an unconditional diffusion prior trained on six seasons of storm-centred ERA5/IMERG states
encodes what real hurricanes look like, and a physics-guided reverse diffusion (score-based data
assimilation) samples the posterior with a calibrated microwave observation operator and a
static-stability constraint in the likelihood.

**Interactive results:** `dashboard/index.html` (16 analysis times, 6 to 10 October 2024, with GOES
imagery, warm-core evolution, diagnostics and the direct-versus-generative comparison).
**Technical report:** see `docs/`.

## Results in brief

Trained on 805 storm-centred scenes from the 2017 to 2022 Atlantic and East Pacific seasons,
validated on the 2023 season, with Milton held out entirely. Verified against ERA5 and IMERG.

| quantity (16 Milton scenes, 8 members x 500 steps) | value |
|---|---|
| domain temperature RMSE against ERA5, 10 levels | 1.0 to 2.8 K |
| inner-core temperature RMSE at 300 hPa, 130 km box, scenes with ATMS | 1.12 K (direct U-Net 1.13 K, physics-only microwave 1.92 K) |
| warm core peaking at 250 to 400 hPa | 15 of 16 scenes |
| ATMS sounding-channel misfit after bias correction | 1.5 to 2.8 K (the ERA5 truth itself sits at 2.5 to 3 K) |
| cost | 3.7 min per scene on a Colab T4 |

Two findings shaped the final design, both reproducible from the code here:

1. **The analytic observation operator must be audited per channel before it enters a likelihood.**
   Against ERA5 truth, the ATMS oxygen sounding channels (5 to 9) carry 14 to 23 K of bias with only
   1 to 3 K of scatter, so a per-channel offset makes them usable; the GOES window channels carry
   35 to 40 K of bias with 20 K of scatter under Milton's ice canopy because the state has no cloud
   variable, so they are removed from the physical likelihood and carried by the learned proxy.
   `colab/audit_rtm_cell.py` produces the table; `JointLikelihood.calibrate_from_audit` applies it.
2. **Likelihood guidance applied where the Tweedie estimate is meaningless destabilises the chain.**
   The first Milton run produced 14 K errors on every scene. Instrumenting a single chain showed
   gradients applied near t = 1 pushed the sample off the data manifold within two steps. The fix
   (`assimilation/guidance.py`, `assimilation/sampler.py`): inflate every observation error by the
   Tweedie variance of the clean-state estimate (pseudo-inverse guidance, Song et al. 2023) and apply
   no guidance above `guide_t_max = 0.65`. Same weights, same data: 15.9 K to 1.7 K on the test scene.

On Milton the full system and the direct U-Net proxy agree to 0.01 K in the inner core, which is the
same conclusion Tomorrow.io reported for ICGen: direct injection of sounder channels is hard to beat
when the proxy is in distribution. The physics term shows its value in the physics-only experiment
(`--proxy-no-mw`): starting from an infrared-only proxy it drives the sounding-channel misfit to the
same floor as the full system without any retraining, but recovers a weaker warm core because a
16 x 16 sounder grid cannot resolve a 30 km eye.

## Architecture

```
observations                     deterministic proxy                 generative DA
GOES ABI  [B,3,H,W]  ─┐                                        prior score s_theta(x_t,t)
                      ├─ CrossAttentionUNet ─► x_det ──┐        (ScoreUNet, ERA5/IMERG states)
ATMS      [B,9,h,w]  ─┘   (IR queries attend            │               │
                           to MW tokens)                ▼               ▼
                                          JointLikelihood  ◄──  GuidedScoreSampler (PC, DPS)
                                          = proxy term                  │
                                          + calibrated ATMS 5-9         ▼
                                          + static stability       posterior ensemble
                                                                   T [N,10,H,W], P [N,1,H,W]
```

| module | contents |
|---|---|
| `config.py` | dataclass configs and tensor conventions |
| `data/` | co-registration (GOES fixed-grid inverse projection, ATMS swath regridding), scene cache, `Normalizer`, dataset, best-track centring, synthetic scenes for tests |
| `models/` | `CrossAttentionUNet`, `ScoreUNet`, `VPSDE`, attention and residual blocks |
| `physics/` | `AnalyticRTM`, `HybridRTM` and `NeuralRTMResidual`, static-stability penalty, hypsometric thickness, warm-core diagnostic |
| `assimilation/` | `JointLikelihood` (per-channel calibration, Tweedie-variance inflation), `GuidedScoreSampler` (predictor-corrector, DPS, guidance gate) |
| `train/` | U-Net training (MSE plus radiance-consistency loss), prior training (DSM, EMA, resumable), optional RTM residual fit |
| `inference/run_milton.py` | `RetrievalEngine` and the CLI writing CF NetCDF analyses |
| `scripts/` | archive and Milton scene preparation, training driver |
| `colab/` | the notebook and cells used for training on a T4, the operator audit, and the dashboard export |
| `tests/` | CPU smoke tests, including a closed-form-prior check that guidance reduces the misfit |
| `dashboard/` | the self-contained results page |

State `x` is `[B, L+1, H, W]`: temperature on 10 pressure levels then `log1p(rain)`, normalised per
channel. IR is on the target grid (2 km, storm-centred), MW on a grid coarsened by 8.

## Running it

The package is imported as `milton_da`, so clone into a directory of that name and run from its parent:

```bash
git clone https://github.com/Ibekwemmanuel7/milton_da.git
cd milton_da && pip install -r requirements.txt && cd ..
python -m pytest milton_da/tests -q
```

Scene preparation (`scripts/prepare_archive.py`, `scripts/prepare_milton.py`) needs GOES-16 ABI L2
CMIP files, ATMS L1B/SDR granules, ERA5 pressure-level temperature, IMERG half-hourly rain and the
IBTrACS best track; the data are not in the repository. Training (`scripts/train.py --preset small
--downscale 2`) and retrieval (`inference/run_milton.py`) commands, and the Colab cells used for the
runs behind the dashboard, are in `colab/`.

```bash
python -m milton_da.inference.run_milton --scenes data/milton/scenes/MILTON_*.nc \
  --stats artifacts/norm_stats.json --unet artifacts/checkpoints/unet.pt --score artifacts/checkpoints/score.pt \
  --rtm-audit artifacts/rtm_audit.json --out artifacts/milton --preset small --downscale 2 --ensemble 8 --steps 500
```

Flags: `--rtm-audit` applies the per-channel calibration, `--mw-channels` and `--ir-channels` select
what enters the physical likelihood (default ATMS 5 to 9, no IR), `--proxy-no-mw` hides the
microwave from the proxy for the physics-only experiment, `--sigma-unet` sets the trust in the proxy,
`--no-rtm` and `--no-mw` are ablations.

## Limitations

Small preset (U-Net base 32, score base 48) trained at 128 x 128 on a free GPU. ERA5 as truth smooths
the eye: its Milton warm core peaks at 4.4 K where reconnaissance measured well over 10 K. The prior
has no positional embedding. The analytic operator has no cloud variable and the state has no
humidity. Coverage is intermittent: 8 of 16 Milton analysis times had an ATMS overpass in the window.
The production path is a learned observation operator emulated from line-by-line and multiple-scattering
references (LBLRTM, DISORT) with ice cloud properties, cloud and humidity in the state, a positional
embedding and 256 x 256 fine-tuning for the prior, few-step distillation, a forecast background term
for cycling, and inner-core verification against dropsondes.

## References

Cannon et al. 2024, Deep Learning for Multi-Satellite Precipitation Retrievals: Impact of Tomorrow.io's
Microwave Sounders (ESS Open Archive, 10.22541/essoar.173430371.11882843). Guerrette et al. 2026,
All-sky assimilation impacts of the Tomorrow.io microwave sounder constellation (QJRMS,
10.1002/qj.70106). Chung et al. 2023, Diffusion Posterior Sampling. Song et al. 2023, Pseudoinverse-Guided
Diffusion Models. Song et al. 2021, Score-Based Generative Modeling through SDEs.

## Author

Chidi Ibekwe. Built with AI-assisted development; the physics and design decisions are the author's.
MIT licence.
