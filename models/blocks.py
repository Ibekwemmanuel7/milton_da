"""
Shared building blocks: residual conv blocks, time embeddings, self- and cross-attention.

Shape notation: [B, C, H, W] feature maps; attention operates on token sequences [B, N, C].
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    """GroupNorm with the largest group count <= 32 that divides the channel count."""
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0 and c // g >= 4 or g == 1:
            return nn.GroupNorm(num_groups=g, num_channels=c)
    return nn.GroupNorm(num_groups=1, num_channels=c)


class SinusoidalTimeEmbedding(nn.Module):
    """Diffusion time t in [0, 1] -> [B, dim] sinusoidal features, then an MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:              # t [B]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * 1000.0 * freqs[None]                # scale t to a 0..1000 "step" range
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)     # [B, dim]
        return self.mlp(emb)


class ResBlock(nn.Module):
    """Pre-norm residual block with optional FiLM-style time conditioning.

    x [B, C_in, H, W] (+ t_emb [B, T]) -> [B, C_out, H, W]
    """

    def __init__(self, c_in: int, c_out: int, t_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.norm1, self.conv1 = _gn(c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2, self.conv2 = _gn(c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.t_proj = nn.Linear(t_dim, 2 * c_out) if t_dim else None
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        nn.init.zeros_(self.conv2.weight), nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.t_proj is not None and t_emb is not None:
            scale, shift = self.t_proj(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)   # each [B, C_out, 1, 1]
            h = h * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):                                              # [B,C,H,W] -> [B,C,H/2,W/2]
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):                                              # [B,C,H,W] -> [B,C,2H,2W]
        return self.op(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class LearnedGridPosEmb(nn.Module):
    """Learned 2D positional embedding for a token grid of at most (max_h, max_w)."""

    def __init__(self, c: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.row = nn.Parameter(torch.zeros(max_h, c))
        self.col = nn.Parameter(torch.zeros(max_w, c))
        nn.init.normal_(self.row, std=0.02), nn.init.normal_(self.col, std=0.02)

    def forward(self, h: int, w: int) -> torch.Tensor:                  # -> [h*w, C]
        return (self.row[:h, None, :] + self.col[None, :w, :]).reshape(h * w, -1)


class CrossAttention2D(nn.Module):
    """Condition a fine feature map on a coarse feature map via multi-head cross-attention.

    Queries  : every pixel of the fine map            x    [B, C_q, H, W]   -> N_q = H*W tokens
    Keys/Vals: every pixel of the coarse map          ctx  [B, C_kv, h, w]  -> N_kv = h*w tokens
    ctx_mask : [B, 1, h, w] 1 = valid observation, 0 = no swath coverage (masked out of attention)

    Fine pixels attend over the *whole* coarse field, so the microwave signal (e.g. the warm core
    seen at 54 GHz) can be redistributed by the IR texture regardless of the misaligned grids.
    Returns x + gated attention output, same shape as x. The output gate is zero-initialised so the
    block is an identity at initialisation (stable training).
    """

    def __init__(self, c_q: int, c_kv: int, n_heads: int = 8, max_ctx_hw: int = 64):
        super().__init__()
        assert c_q % n_heads == 0
        self.n_heads, self.d = n_heads, c_q // n_heads
        self.norm_q, self.norm_kv = _gn(c_q), _gn(c_kv)
        self.q = nn.Conv2d(c_q, c_q, 1)
        self.kv = nn.Conv2d(c_kv, 2 * c_q, 1)
        self.pos_q = LearnedGridPosEmb(c_q, 256, 256)
        self.pos_kv = LearnedGridPosEmb(c_q, max_ctx_hw, max_ctx_hw)
        self.out = nn.Conv2d(c_q, c_q, 1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        _, Ck, h, w = ctx.shape
        q = self.q(self.norm_q(x)).flatten(2).transpose(1, 2) + self.pos_q(H, W)[None]          # [B, N_q, C]
        k, v = self.kv(self.norm_kv(ctx)).flatten(2).transpose(1, 2).chunk(2, dim=-1)            # [B, N_kv, C] each
        k = k + self.pos_kv(h, w)[None]
        split = lambda t: t.view(B, -1, self.n_heads, self.d).transpose(1, 2)                   # [B, heads, N, d]
        q, k, v = split(q), split(k), split(v)
        attn_mask = None
        if ctx_mask is not None:
            valid = ctx_mask.flatten(1) > 0.5                                                   # [B, N_kv]
            valid = valid | (~valid.any(1, keepdim=True))       # if a sample has no MW at all, allow all (values are 0)
            attn_mask = valid[:, None, None, :]                                                 # [B,1,1,N_kv] True = attend
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)                        # [B, heads, N_q, d]
        o = o.transpose(1, 2).reshape(B, H * W, C).transpose(1, 2).view(B, C, H, W)
        return x + self.gate * self.out(o)


class SelfAttention2D(nn.Module):
    """Global self-attention over a feature map (used at coarse resolutions of the score network)."""

    def __init__(self, c: int, n_heads: int = 8):
        super().__init__()
        self.norm = _gn(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.out = nn.Conv2d(c, c, 1)
        self.n_heads = n_heads
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, 3, self.n_heads, C // self.n_heads, H * W).transpose(-1, -2).unbind(1)  # [B,heads,N,d]
        o = F.scaled_dot_product_attention(q, k, v)
        return x + self.out(o.transpose(-1, -2).reshape(B, C, H, W))
