"""End-to-end CPU smoke test on synthetic scenes (run: pytest -q milton_da/tests)."""
import numpy as np
import pytest
import torch

from milton_da.assimilation.guidance import JointLikelihood, Observations
from milton_da.assimilation.sampler import GuidedScoreSampler
from milton_da.config import PipelineConfig
from milton_da.data.dataset import HurricaneSceneDataset, Normalizer, collate
from milton_da.data.synthetic import make_synthetic_scenes
from milton_da.inference.run_milton import RetrievalEngine
from milton_da.models.score_net import ScoreUNet
from milton_da.models.sde import VPSDE
from milton_da.models.unet_xattn import CrossAttentionUNet
from milton_da.physics.constraints import static_stability_penalty
from milton_da.physics.rtm import AnalyticRTM, HybridRTM, NeuralRTMResidual
from milton_da.train.train_score import train_score
from milton_da.train.train_unet import train_unet


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = PipelineConfig.small_debug()
    scenes = make_synthetic_scenes(cfg.data, 8)
    norm = Normalizer.fit(scenes)
    ds = HurricaneSceneDataset(scenes, cfg.data, norm, augment=True)
    return cfg, scenes, norm, ds


def test_dataset_shapes(setup):
    cfg, scenes, norm, ds = setup
    b = collate([ds[0], ds[1]])
    H, W = cfg.data.grid.ny, cfg.data.grid.nx
    h, w = cfg.data.grid.coarse_shape
    assert b["ir"].shape == (2, len(cfg.data.ir_channels), H, W)
    assert b["mw"].shape == (2, len(cfg.data.mw_channels), h, w)
    assert b["state"].shape == (2, cfg.data.state_channels, H, W)
    temp, precip = norm.state_to_physical(b["state"])
    assert 180 < temp.min() < temp.max() < 320 and precip.min() >= 0


def test_rtm_is_differentiable_and_physical(setup):
    cfg, scenes, norm, ds = setup
    d = cfg.data
    rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    b = collate([ds[0]])
    temp, precip = norm.state_to_physical(b["state"])
    temp.requires_grad_(True), precip.requires_grad_(True)
    ir, mw = rtm(temp, precip)
    (ir.sum() + mw.sum()).backward()
    assert torch.isfinite(temp.grad).all() and torch.isfinite(precip.grad).all()
    # heavier rain -> colder IR window (higher cloud tops) and colder 183 GHz (scattering)
    ir2, mw2 = rtm(temp.detach(), precip.detach() * 3)
    assert (ir2[:, -1] <= ir[:, -1].detach() + 1e-3).float().mean() > 0.95
    assert (mw2[:, 7] <= mw[:, 7].detach() + 1e-3).all()
    assert static_stability_penalty(temp.detach(), d.levels_hpa).item() < 1e-3
    hyb = HybridRTM(rtm, NeuralRTMResidual(d.n_levels, len(d.ir_channels), len(d.mw_channels), d.grid.mw_downscale, 8))
    ir3, _ = hyb(temp.detach(), precip.detach())
    assert torch.allclose(ir3, ir.detach(), atol=1e-5)   # zero-initialised residual


def test_cross_attention_unet_masks(setup):
    cfg, scenes, norm, ds = setup
    model = CrossAttentionUNet(cfg.data, cfg.unet)
    b = collate([ds[0], ds[1]])
    b["mw_mask"][1] = 0.0                       # sample with no MW coverage must still work
    x = model(b["ir"], b["ir_mask"], b["mw"], b["mw_mask"])
    assert x.shape == b["state"].shape and torch.isfinite(x).all()


def test_end_to_end_train_and_assimilate(setup):
    cfg, scenes, norm, ds = setup
    d = cfg.data
    rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    device = torch.device("cpu")
    unet = train_unet(cfg, ds, None, norm, rtm, max_steps=30, device=device)
    score, ema = train_score(cfg, ds, max_steps=60, device=device)
    engine = RetrievalEngine(cfg, norm, unet, ema.shadow, rtm, device=device)
    batch = collate([ds[0]])
    out, obs = engine.analyse(batch, ensemble_size=2)
    assert out.mean.shape == batch["state"].shape and torch.isfinite(out.samples).all()
    trace = [t for t in out.trace if t["member"] == 0]
    assert np.isfinite(trace[-1]["ir_rmse_K"])
    result = engine.to_dataset(out, obs, scenes[0], truth=batch["state"])
    assert "temperature" in result and result["temperature"].shape == (d.n_levels, d.grid.ny, d.grid.nx)
    assert "warm_core_anomaly" in result


def test_guidance_reduces_observation_misfit(setup):
    """Exact-prior check of the DA machinery: with a closed-form Gaussian climatology prior, turning on
    the likelihood guidance must reduce the simulated-vs-observed brightness temperature misfit."""
    cfg, scenes, norm, ds = setup
    d = cfg.data
    from milton_da.models.score_net import GaussianClimatologyScore

    cfg.guidance.n_steps = 60
    rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    prior = GaussianClimatologyScore.fit_from_dataset(HurricaneSceneDataset(scenes, d, norm), VPSDE(cfg.sde))
    unet = train_unet(cfg, ds, None, norm, rtm, max_steps=40, device=torch.device("cpu"))
    batch = collate([HurricaneSceneDataset(make_synthetic_scenes(d, 1, seed=123), d, norm)[0]])
    misfit = {}
    for gs in (0.0, 1.0):
        cfg.guidance.guidance_scale = gs
        torch.manual_seed(1)
        engine = RetrievalEngine(cfg, norm, unet, prior, rtm, device=torch.device("cpu"))
        out, obs = engine.analyse(batch, ensemble_size=2)
        misfit[gs] = engine.likelihood.diagnostics(out.mean, obs)
    assert misfit[1.0]["ir_rmse_K"] < 0.7 * misfit[0.0]["ir_rmse_K"]
    assert misfit[1.0]["mw_rmse_K"] < misfit[0.0]["mw_rmse_K"]
    assert misfit[1.0]["stability"] < 1e-3
