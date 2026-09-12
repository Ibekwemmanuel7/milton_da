"""
Central configuration for the Hurricane Milton 3D retrieval / score-based DA pipeline.

Tensor layout conventions used everywhere in this package
---------------------------------------------------------
B  : batch
L  : number of pressure levels (len(PRESSURE_LEVELS_HPA), top -> bottom)
C  : generic channel axis
H, W : rows / cols of the storm-centred *target* grid (IR native resolution)
h, w : rows / cols of the coarse microwave grid, h = H // mw_downscale

State vector x  : [B, L + 1, H, W]
                  channels 0..L-1  = temperature at PRESSURE_LEVELS_HPA (K, normalised)
                  channel  L       = surface precipitation rate (log1p(mm/h), normalised)
IR observations : [B, C_ir, H, W]   brightness temperature (K, normalised)
MW observations : [B, C_mw, h, w]   brightness temperature (K, normalised)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

# Pressure levels ordered top-of-atmosphere -> surface. ERA5 provides all of these.
PRESSURE_LEVELS_HPA: List[int] = [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# GOES-16 ABI channels used. C13 (10.3 um clean IR window) is the primary cloud-top channel;
# C08/C10 water-vapour channels add mid/upper-troposphere moisture and height context.
GOES_CHANNELS: List[str] = ["C08", "C10", "C13"]
GOES_CENTRAL_WAVELENGTH_UM = {"C08": 6.19, "C10": 7.34, "C13": 10.35}

# ATMS channels (SNPP / NOAA-20 / NOAA-21).
#   5-9  : 52.8-55.5 GHz O2 temperature sounding (pierce cloud, see the warm core)
#   16   : 88.2 GHz  window / scattering
#   17   : 165.5 GHz scattering
#   18-22: 183 GHz water-vapour sounding (18 = 183+-7 GHz is the most precip-sensitive)
ATMS_CHANNELS: List[int] = [5, 6, 7, 8, 9, 16, 17, 18, 22]
ATMS_FREQ_GHZ = {5: 52.8, 6: 53.596, 7: 54.4, 8: 54.94, 9: 55.5, 16: 88.2, 17: 165.5, 18: 183.31, 22: 183.31}


@dataclass
class GridConfig:
    """Storm-centred, regular lat/lon target grid on which every source is co-registered."""
    ny: int = 256                 # H
    nx: int = 256                 # W
    dlat_deg: float = 0.02        # ~2.2 km, comparable to ABI IR nadir resolution
    dlon_deg: float = 0.02
    mw_downscale: int = 8         # h = H / 8 -> ~16-18 km, comparable to ATMS 50 GHz footprint

    @property
    def coarse_shape(self) -> Tuple[int, int]:
        return self.ny // self.mw_downscale, self.nx // self.mw_downscale


@dataclass
class DataConfig:
    levels_hpa: Sequence[int] = field(default_factory=lambda: list(PRESSURE_LEVELS_HPA))
    ir_channels: Sequence[str] = field(default_factory=lambda: list(GOES_CHANNELS))
    mw_channels: Sequence[int] = field(default_factory=lambda: list(ATMS_CHANNELS))
    grid: GridConfig = field(default_factory=GridConfig)
    # Maximum |dt| between the analysis time and a sensor overpass for a match to be accepted.
    ir_time_tolerance_min: float = 7.5      # ABI CONUS/FD cadence 5-10 min
    mw_time_tolerance_min: float = 90.0     # polar orbiter revisit
    precip_log_transform: bool = True
    stats_path: str = "artifacts/norm_stats.json"

    @property
    def n_levels(self) -> int:
        return len(self.levels_hpa)

    @property
    def state_channels(self) -> int:
        return self.n_levels + 1


@dataclass
class UNetConfig:
    base_channels: int = 64
    channel_mults: Sequence[int] = (1, 2, 4, 8)   # 256 -> 128 -> 64 -> 32 (bottleneck)
    n_res_blocks: int = 2
    attn_heads: int = 8
    cross_attn_levels: Sequence[int] = (2, 3)      # encoder depths at which MW cross-attention is applied
    dropout: float = 0.0
    # Attention-encoder variant (experiment, off by default so existing checkpoints load unchanged):
    pos_embed: bool = False                        # learned 2D positional embedding added to the level-0 IR features
    self_attn_levels: Sequence[int] = ()           # encoder/decoder depths given global self-attention over IR tokens
    mw_encoder: str = "grid"                       # "grid": convolutional context encoder (default); "graph": kNN message passing (models/gnn.py)
    graph_k: int = 16                              # neighbours per node for the graph encoder
    graph_rounds: int = 3                          # message-passing rounds


@dataclass
class ScoreNetConfig:
    base_channels: int = 96
    channel_mults: Sequence[int] = (1, 2, 4, 8)
    n_res_blocks: int = 2
    attn_heads: int = 8
    time_embed_dim: int = 384
    self_attn_levels: Sequence[int] = (3,)
    cond_channels: int = 0    # 0 -> pure unconditional prior p(x). >0 -> concatenates conditioning maps.


@dataclass
class SDEConfig:
    beta_min: float = 0.1
    beta_max: float = 20.0
    t_eps: float = 1e-3       # smallest diffusion time used for training / sampling


@dataclass
class GuidanceConfig:
    """Observation-error standard deviations (in *normalised* units unless noted) and weights."""
    n_steps: int = 500
    final_denoise_t: float = 0.03     # stop the chain here and return the Tweedie estimate (see sampler)
    n_corrector: int = 1
    snr: float = 0.16                 # Langevin corrector signal-to-noise
    sigma_unet: float = 0.5           # trust in deterministic U-Net proxy (normalised state units)
    # Observation errors include RTM representativeness error (the analytic proxy is not CRTM);
    # tighten these when a trained HybridRTM is used.
    sigma_ir_K: float = 5.0           # IR brightness temperature error (K)
    sigma_mw_K: float = 2.0           # MW brightness temperature error (K)
    lambda_stability: float = 1.0     # static-stability (dry adiabatic) penalty weight
    lambda_precip_nonneg: float = 1.0
    guidance_scale: float = 1.0       # multiplier on grad log p(y|x); 1.0 = Bayesian posterior score
    guide_t_max: float = 0.65         # no likelihood guidance for t above this (alpha_t < ~0.5: the Tweedie
                                      # estimate is not yet meaningful and DPS gradients through the network
                                      # only perturb the chain off the data manifold)
    max_step_rms: float = 0.25        # cap on per-step RMS guidance displacement (normalised units)
    x0_clip: float = 6.0              # soft clamp of the Tweedie estimate (normalised units)
    ensemble_size: int = 8


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 50
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    amp: bool = True
    num_workers: int = 4
    ckpt_dir: str = "artifacts/checkpoints"
    log_every: int = 50
    lambda_rtm_consistency: float = 0.1   # U-Net auxiliary loss: RTM(pred) vs observed TB


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    score: ScoreNetConfig = field(default_factory=ScoreNetConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def small_debug() -> "PipelineConfig":
        """A tiny configuration for CPU smoke tests."""
        cfg = PipelineConfig()
        cfg.data.grid = GridConfig(ny=64, nx=64, mw_downscale=8)
        cfg.unet = UNetConfig(base_channels=16, channel_mults=(1, 2, 4), n_res_blocks=1, attn_heads=4, cross_attn_levels=(2,))
        cfg.score = ScoreNetConfig(base_channels=16, channel_mults=(1, 2, 4), n_res_blocks=1, attn_heads=4, time_embed_dim=64, self_attn_levels=(2,))
        cfg.guidance = GuidanceConfig(n_steps=20, n_corrector=1, ensemble_size=2)
        cfg.train = TrainConfig(batch_size=2, num_workers=0, amp=False)
        return cfg
