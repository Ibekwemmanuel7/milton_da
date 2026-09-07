"""
Time-conditioned U-Net that parameterises the prior score  s_theta(x_t, t) ~ grad_x log p_t(x).

The network predicts eps (noise); `VPSDE.score_from_eps` converts to the score. Trained purely on
ERA5/IMERG *states* (no observations), so it is an unconditional prior over physically realised
3D atmospheric structures. Optional `cond_channels` allows a conditional variant that concatenates
extra maps (e.g. the U-Net proxy) to the input; the default pipeline keeps it unconditional and
injects observations only through the likelihood guidance at inference.

x_t [B, L+1, H, W], t [B] -> eps [B, L+1, H, W]
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..config import DataConfig, ScoreNetConfig
from .blocks import Downsample, ResBlock, SelfAttention2D, SinusoidalTimeEmbedding, Upsample, _gn


class ScoreUNet(nn.Module):
    def __init__(self, data_cfg: DataConfig, cfg: ScoreNetConfig):
        super().__init__()
        c_state = data_cfg.state_channels
        c_in = c_state + cfg.cond_channels
        chans = [cfg.base_channels * m for m in cfg.channel_mults]
        T = cfg.time_embed_dim
        self.t_emb = SinusoidalTimeEmbedding(T)
        self.inp = nn.Conv2d(c_in, chans[0], 3, padding=1)

        self.enc, self.enc_attn, self.downs = nn.ModuleList(), nn.ModuleDict(), nn.ModuleList()
        c_prev = chans[0]
        for i, c in enumerate(chans):
            self.enc.append(nn.ModuleList([ResBlock(c_prev if j == 0 else c, c, T) for j in range(cfg.n_res_blocks)]))
            if i in cfg.self_attn_levels:
                self.enc_attn[str(i)] = SelfAttention2D(c, cfg.attn_heads)
            self.downs.append(Downsample(c) if i < len(chans) - 1 else nn.Identity())
            c_prev = c
        self.mid = nn.ModuleList([ResBlock(chans[-1], chans[-1], T), SelfAttention2D(chans[-1], cfg.attn_heads), ResBlock(chans[-1], chans[-1], T)])
        self.dec, self.dec_attn, self.ups = nn.ModuleList(), nn.ModuleDict(), nn.ModuleList()
        for i in reversed(range(len(chans))):
            c = chans[i]
            c_in_dec = chans[min(i + 1, len(chans) - 1)] + c
            self.dec.append(nn.ModuleList([ResBlock(c_in_dec if j == 0 else c, c, T) for j in range(cfg.n_res_blocks)]))
            if i in cfg.self_attn_levels:
                self.dec_attn[str(i)] = SelfAttention2D(c, cfg.attn_heads)
            self.ups.append(Upsample(c) if i > 0 else nn.Identity())
        self.out = nn.Sequential(_gn(chans[0]), nn.SiLU(), nn.Conv2d(chans[0], c_state, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight), nn.init.zeros_(self.out[-1].bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        temb = self.t_emb(t)                                             # [B, T]
        h = self.inp(x_t if cond is None else torch.cat([x_t, cond], 1))
        skips: List[torch.Tensor] = []
        for i, (blocks, down) in enumerate(zip(self.enc, self.downs)):
            for b in blocks:
                h = b(h, temb)
            if str(i) in self.enc_attn:
                h = self.enc_attn[str(i)](h)
            skips.append(h)
            h = down(h)
        h = self.mid[0](h, temb)
        h = self.mid[1](h)
        h = self.mid[2](h, temb)
        for k, (blocks, up) in enumerate(zip(self.dec, self.ups)):
            i = len(self.enc) - 1 - k
            h = torch.cat([h, skips[i]], 1)
            for b in blocks:
                h = b(h, temb)
            if str(i) in self.dec_attn:
                h = self.dec_attn[str(i)](h)
            h = up(h)
        return self.out(h)                                               # eps [B, L+1, H, W]


class GaussianClimatologyScore(nn.Module):
    """Closed-form eps-predictor for a diagonal Gaussian prior  x0 ~ N(mu, diag(var))  (per pixel).

    This is the generative analogue of a static background-error covariance B in classical DA and
    serves two purposes: (i) a drop-in baseline prior (`climatology`) to compare against the learned
    score network, and (ii) an exact reference for unit-testing the guided sampler, because the
    marginal p_t(x_t) is Gaussian with mean alpha_t mu and variance alpha_t^2 var + sigma_t^2, so
        score(x_t, t) = -(x_t - alpha_t mu) / (alpha_t^2 var + sigma_t^2),   eps = -sigma_t * score.
    mu, var : [L+1, H, W] in normalised state units (e.g. from `fit_from_dataset`).
    """

    def __init__(self, mu: torch.Tensor, var: torch.Tensor, sde):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("var", var.clamp(min=1e-4))
        self.sde = sde

    @staticmethod
    def fit_from_dataset(dataset, sde, max_items: int = 512) -> "GaussianClimatologyScore":
        xs = torch.stack([dataset[i]["state"] for i in range(min(len(dataset), max_items))])   # [N, L+1, H, W]
        return GaussianClimatologyScore(xs.mean(0), xs.var(0, unbiased=False), sde)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        alpha, sigma = self.sde.marginal_prob(t)
        a, s = alpha[:, None, None, None], sigma[:, None, None, None]
        score = -(x_t - a * self.mu) / (a**2 * self.var + s**2)
        return -s * score
