"""Cloud ice in the state (level one), two-stream scattering (level two) and the all-sky error model (level three)."""
import copy

import numpy as np
import pytest
import torch

from milton_da.assimilation.guidance import JointLikelihood, Observations
from milton_da.config import PipelineConfig
from milton_da.data.dataset import HurricaneSceneDataset, Normalizer, collate, integrate_ice
from milton_da.data.synthetic import make_synthetic_scenes
from milton_da.inference.run_milton import RetrievalEngine
from milton_da.physics.audit import fit_scatter_depression
from milton_da.physics.rtm import AnalyticRTM
from milton_da.physics.scatter import IceOptics, ScatteringRTM, _mie_q, two_stream_layer
from milton_da.train.train_score import train_score
from milton_da.train.train_unet import train_unet


@pytest.fixture(scope="module")
def scenes():
    torch.manual_seed(0)
    cfg = PipelineConfig.small_debug()
    return cfg, make_synthetic_scenes(cfg.data, 6)


@pytest.mark.parametrize("ice", ["iwp", "profile"])
def test_state_round_trip_with_ice(scenes, ice):
    cfg, sc = scenes
    cfg = copy.deepcopy(cfg); cfg.data.ice = ice
    norm = Normalizer.fit(sc)
    assert "iwp" in norm.stats and "ciwc" in norm.stats
    ds = HurricaneSceneDataset(sc, cfg.data, norm)
    b = collate([ds[0], ds[1]])
    assert b["state"].shape[1] == cfg.data.state_channels == cfg.data.n_levels + 1 + (1 if ice == "iwp" else cfg.data.n_levels)
    temp, precip = norm.state_to_physical(b["state"])
    assert torch.allclose(temp, torch.from_numpy(np.stack([s["temp"].values for s in sc[:2]])), atol=1e-2)
    got = norm.state_ice(b["state"], cfg.data.levels_hpa)
    truth_iwp = torch.from_numpy(np.stack([s["iwp"].values for s in sc[:2]]))[:, None]
    if ice == "iwp":
        assert torch.allclose(got["iwp"], truth_iwp, atol=1e-3)
    else:
        truth_ciwc = torch.from_numpy(np.stack([s["ciwc"].values for s in sc[:2]]))
        assert torch.allclose(got["ciwc"], truth_ciwc, rtol=1e-3, atol=1e-7)
        # the integral of the profile over the state levels is the IWP (synthetic ice sits on the state levels)
        assert torch.allclose(integrate_ice(truth_ciwc, cfg.data.levels_hpa), truth_iwp, rtol=0.25, atol=0.02)


def test_analytic_rtm_ice_depression_monotone(scenes):
    cfg, sc = scenes
    d = cfg.data
    rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    temp = torch.from_numpy(sc[0]["temp"].values)[None]
    precip = torch.zeros_like(temp[:, :1])
    _, clear = rtm(temp, precip)
    prev = clear
    for iwp in (0.1, 0.5, 2.0):
        _, mw = rtm(temp, precip, {"iwp": torch.full_like(precip, iwp)})
        assert (mw <= prev + 1e-4).all()                          # more ice, colder everywhere
        prev = mw
    ch = list(d.mw_channels)
    dep = (clear - prev)[0, :, 0, 0]
    assert dep[ch.index(18)] > dep[ch.index(16)] > dep[ch.index(9)]   # 183 GHz > 89 GHz > O2 sounding channel
    # IR: an ice cloud lowers the window channel toward the cloud-top temperature
    ir_clear, _ = rtm(temp, precip)
    ir_ice, _ = rtm(temp, precip, {"iwp": torch.full_like(precip, 1.0)})
    assert (ir_ice[:, -1] < ir_clear[:, -1]).float().mean() > 0.95


def test_mie_rayleigh_limit_and_optics_tables():
    m = complex(1.33, 0.0)
    x = np.array([0.05, 0.1])
    _, qs, _ = _mie_q(m, x)
    ray = 8 / 3 * x**4 * abs((m**2 - 1) / (m**2 + 2)) ** 2
    assert np.allclose(qs, ray, rtol=2e-3)
    opt = IceOptics([52.8, 88.2, 165.5, 183.31])
    assert np.all(np.diff(opt.kext[:, 20]) > 0)                       # extinction per kg grows with frequency
    assert np.all((opt.ssa > 0) & (opt.ssa < 1)) and np.all((opt.asy >= 0) & (opt.asy < 1))


def test_two_stream_layer_limits():
    tau = torch.tensor([0.0, 0.5, 5.0])
    R, T = two_stream_layer(tau, torch.zeros(3), torch.zeros(3), D=1.66)
    assert torch.allclose(R, torch.zeros(3)) and torch.allclose(T, torch.exp(-1.66 * tau), atol=1e-5)
    R, T = two_stream_layer(torch.tensor(2.0), torch.tensor(0.95), torch.tensor(0.7))
    assert 0 < R < 1 and 0 < T < 1 and R + T <= 1


def test_scattering_rtm_matches_analytic_in_clear_sky_and_scatters(scenes):
    cfg, sc = scenes
    d = cfg.data
    ana = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    sca = ScatteringRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    temp = torch.from_numpy(sc[0]["temp"].values)[None]
    precip = torch.zeros_like(temp[:, :1])
    _, a = ana(temp, precip); _, s = sca(temp, precip)
    assert torch.allclose(a, s, atol=1e-3)                            # exact clear-sky equivalence
    ciwc = torch.from_numpy(sc[0]["ciwc"].values)[None].clone().requires_grad_(True)
    iwp = integrate_ice(ciwc, d.levels_hpa)
    _, s_ice = sca(temp, precip, {"ciwc": ciwc, "iwp": iwp})
    assert (s_ice <= s + 1e-3).all() and (s - s_ice).max() > 5.0    # the anvil scatters, tens of K at 183 GHz
    s_ice.sum().backward()
    assert torch.isfinite(ciwc.grad).all() and ciwc.grad.abs().sum() > 0
    # IWP-only state: the fixed vertical profile gives a depression of the same order
    _, s_iwp = sca(temp, precip, {"iwp": iwp.detach()})
    ch = list(d.mw_channels).index(18)
    assert 0.3 < float((s - s_iwp)[0, ch].max() / (s - s_ice)[0, ch].max().detach()) < 3.0


def test_fit_scatter_depression_recovers_parameters():
    rng = np.random.default_rng(0)
    iwp = rng.gamma(1.0, 0.6, 3000)
    dep = 5.0 + 60.0 * (1 - np.exp(-iwp / 0.4)) + rng.normal(0, 2.0, iwp.size)
    fit = fit_scatter_depression(iwp, dep)
    assert fit["fitted"] and abs(fit["a_K"] - 60) < 6 and 0.28 < fit["I0"] < 0.55 and abs(fit["bias_K"] - 5) < 2
    assert fit["rmse_K"] < 0.3 * fit["rmse_clear_K"]
    rtm = AnalyticRTM([200, 300, 500, 700, 1000], ["C13"], [16, 18], 2)
    rtm.set_scatter_fit({"18": fit})
    assert abs(float(rtm.mw_ice_a[1]) - fit["a_K"]) < 1e-4 and float(rtm.mw_ice_a[0]) == 40.0


def test_allsky_weights_downweight_cloudy_pixels(scenes):
    cfg, sc = scenes
    cfg = copy.deepcopy(cfg); d = cfg.data
    norm = Normalizer.fit(sc)
    ds = HurricaneSceneDataset(sc, d, norm)
    b = collate([ds[0]])
    rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    cfg.guidance.allsky = True
    lik = JointLikelihood(rtm, norm, d, cfg.guidance)
    obs = Observations(b["state"], b["ir_raw"], b["ir_mask"], b["mw_raw"], b["mw_mask"])
    temp, precip = norm.state_to_physical(b["state"])
    ir_sim, mw_sim = rtm(temp, precip)
    ir_w, mw_w = lik._allsky_weights(temp, ir_sim, mw_sim, obs, lik.ir_w, lik.mw_w, 0.0, lik._temp_var_K2())
    assert mw_w.shape == mw_sim.shape and ir_w.shape == ir_sim.shape
    ch = list(d.mw_channels).index(18)
    rain_c = torch.nn.functional.avg_pool2d(precip, d.grid.mw_downscale)[0, 0]
    cloudy, clear = rain_c > rain_c.quantile(0.9), rain_c < 0.05
    assert mw_w[0, ch][cloudy].mean() < 0.5 * mw_w[0, ch][clear].mean()          # eyewall pixels trusted less
    assert torch.allclose(mw_w[0, ch][clear], torch.full_like(mw_w[0, ch][clear], 1.0 / cfg.guidance.sigma_mw_K**2), rtol=0.1)
    # the full cost still evaluates and stays finite with the per-pixel weights
    t = lik.terms(b["state"], obs, r2=0.1)
    assert all(torch.isfinite(v).all() for v in t.values())


@pytest.mark.parametrize("ice,rtm_kind", [("iwp", "analytic"), ("profile", "scattering")])
def test_end_to_end_with_ice(scenes, ice, rtm_kind):
    cfg, sc = scenes
    cfg = copy.deepcopy(cfg); cfg.data.ice = ice; cfg.guidance.allsky = True; cfg.guidance.n_steps = 12
    d = cfg.data
    norm = Normalizer.fit(sc)
    ds = HurricaneSceneDataset(sc, d, norm, augment=True)
    rtm = (ScatteringRTM if rtm_kind == "scattering" else AnalyticRTM)(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    device = torch.device("cpu")
    unet = train_unet(cfg, ds, None, norm, rtm, max_steps=12, device=device)
    score, ema = train_score(cfg, ds, max_steps=20, device=device)
    engine = RetrievalEngine(cfg, norm, unet, ema.shadow, rtm, device=device)
    batch = collate([ds[0]])
    assert batch["state"].shape[1] == d.state_channels
    out, obs = engine.analyse(batch, ensemble_size=2, log_every=0)
    assert out.mean.shape == batch["state"].shape and torch.isfinite(out.samples).all()
    result = engine.to_dataset(out, obs, sc[0], truth=batch["state"])
    assert "iwp" in result and "iwp_unet" in result and float(result["iwp"].min()) >= 0
    if ice == "profile":
        assert "ciwc" in result and result["ciwc"].shape == (d.n_levels, d.grid.ny, d.grid.nx)
