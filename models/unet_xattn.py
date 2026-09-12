"""
Cross-Attention U-Net: deterministic multi-modal retrieval proxy.

Encoder path   : high-resolution IR imagery [B, C_ir + 1, H, W] (channels + validity mask)
Context path   : coarse microwave sounder [B, C_mw + 2, h, w] (TB, mask, zenith) -> token grid
Cross-attention: at the deepest encoder levels (and mirrored in the decoder), IR feature maps
                 (queries) attend over the MW token grid (keys/values). This is how the deep
                 eyewall / warm-core signal captured only by the sounder is injected and spatially
                 redistributed by the fine IR texture.
Output         : deterministic proxy state x_det [B, L + 1, H, W] in *normalised* state units:
                 channels 0..L-1 temperature levels, channel L log-precipitation.

Resolution bookkeeping (default config, H = W = 256, mw_downscale = 8 -> h = w = 32):
    level 0 : 256 x 256  (base_channels)
    level 1 : 128 x 128
    level 2 :  64 x  64   <- cross-attention (IR tokens 4096, MW tokens 1024)
    level 3 :  32 x  32   <- cross-attention / bottleneck (IR tokens 1024)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DataConfig, UNetConfig
from .blocks import CrossAttention2D, Downsample, LearnedGridPosEmb, ResBlock, SelfAttention2D, Upsample, _gn
from .gnn import GraphMicrowaveEncoder


class MicrowaveContextEncoder(nn.Module):
    """[B, C_mw + 2, h, w] (TB channels, validity mask, zenith angle) -> [B, C_ctx, h, w]. Keeps native
    coarse resolution (no upsampling): the alignment is learned inside the cross-attention rather than
    baked in by interpolation. The zenith angle lets the network absorb residual limb effects."""

    def __init__(self, c_in: int, c_ctx: int, n_blocks: int = 3):
        super().__init__()
        self.inp = nn.Conv2d(c_in, c_ctx, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(c_ctx, c_ctx) for _ in range(n_blocks)])
        self.attn = SelfAttention2D(c_ctx, n_heads=4)   # lets sounder tokens share swath-wide context

    def forward(self, mw: torch.Tensor, mw_mask: torch.Tensor, mw_zen: torch.Tensor) -> torch.Tensor:
        h = self.inp(torch.cat([mw, mw_mask, mw_zen], dim=1))
        for b in self.blocks:
            h = b(h)
        return self.attn(h)


class CrossAttentionUNet(nn.Module):
    def __init__(self, data_cfg: DataConfig, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        c_ir = len(data_cfg.ir_channels) + 1                 # + validity mask
        c_mw = len(data_cfg.mw_channels) + 2                 # + validity mask + satellite zenith angle
        self.n_levels, self.out_channels = data_cfg.n_levels, data_cfg.state_channels
        chans = [cfg.base_channels * m for m in cfg.channel_mults]
        c_ctx = chans[-1] // 2
        if getattr(cfg, "mw_encoder", "grid") == "graph":
            self.mw_encoder = GraphMicrowaveEncoder(c_mw, c_ctx, k=cfg.graph_k, n_rounds=cfg.graph_rounds)   # experiment: message passing
        else:
            self.mw_encoder = MicrowaveContextEncoder(c_mw, c_ctx)
        self.inp = nn.Conv2d(c_ir, chans[0], 3, padding=1)
        # Optional learned positional embedding on the level-0 feature map: gives the fully convolutional
        # network a notion of where the storm centre is in the frame (zero-initialised: identity at start).
        self.pos = LearnedGridPosEmb(chans[0], 256, 256) if cfg.pos_embed else None
        if self.pos is not None:
            nn.init.zeros_(self.pos.row), nn.init.zeros_(self.pos.col)
        sa_levels = tuple(getattr(cfg, "self_attn_levels", ()) or ())

        # ---- encoder -----------------------------------------------------------------------
        self.enc_blocks = nn.ModuleList()
        self.enc_xattn = nn.ModuleDict()
        self.enc_sattn = nn.ModuleDict()
        self.downs = nn.ModuleList()
        c_prev = chans[0]
        for i, c in enumerate(chans):
            self.enc_blocks.append(nn.ModuleList([ResBlock(c_prev if j == 0 else c, c, dropout=cfg.dropout) for j in range(cfg.n_res_blocks)]))
            if i in cfg.cross_attn_levels:
                self.enc_xattn[str(i)] = CrossAttention2D(c, c_ctx, cfg.attn_heads)
            if i in sa_levels:
                self.enc_sattn[str(i)] = SelfAttention2D(c, cfg.attn_heads)       # global self-attention over IR tokens
            self.downs.append(Downsample(c) if i < len(chans) - 1 else nn.Identity())
            c_prev = c

        # ---- bottleneck --------------------------------------------------------------------
        self.mid = nn.ModuleList([ResBlock(chans[-1], chans[-1]), SelfAttention2D(chans[-1], cfg.attn_heads), ResBlock(chans[-1], chans[-1])])

        # ---- decoder -----------------------------------------------------------------------
        self.dec_blocks = nn.ModuleList()
        self.dec_xattn = nn.ModuleDict()
        self.dec_sattn = nn.ModuleDict()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(chans))):
            c = chans[i]
            c_in = chans[min(i + 1, len(chans) - 1)] + c            # upsampled deeper features + skip
            self.dec_blocks.append(nn.ModuleList([ResBlock(c_in if j == 0 else c, c, dropout=cfg.dropout) for j in range(cfg.n_res_blocks)]))
            if i in cfg.cross_attn_levels:
                self.dec_xattn[str(i)] = CrossAttention2D(c, c_ctx, cfg.attn_heads)
            if i in sa_levels:
                self.dec_sattn[str(i)] = SelfAttention2D(c, cfg.attn_heads)
            self.ups.append(Upsample(c) if i > 0 else nn.Identity())

        self.out = nn.Sequential(_gn(chans[0]), nn.SiLU(), nn.Conv2d(chans[0], self.out_channels, 3, padding=1))

    def forward(self, ir: torch.Tensor, ir_mask: torch.Tensor, mw: torch.Tensor, mw_mask: torch.Tensor, mw_zen: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ir      [B, C_ir, H, W]   normalised IR brightness temperatures
        ir_mask [B, 1, H, W]
        mw      [B, C_mw, h, w]   normalised MW brightness temperatures (0 where invalid)
        mw_mask [B, 1, h, w]
        mw_zen  [B, 1, h, w]      satellite zenith angle / 60 (0 where invalid); zeros if None
        returns x_det [B, L+1, H, W]
        """
        if mw_zen is None:
            mw_zen = torch.zeros_like(mw_mask)
        ctx = self.mw_encoder(mw, mw_mask, mw_zen)                      # [B, C_ctx, h, w]
        h = self.inp(torch.cat([ir, ir_mask], dim=1))                   # [B, C0, H, W]
        if self.pos is not None:
            B_, C0, H, W = h.shape
            h = h + self.pos(H, W).transpose(0, 1).reshape(1, C0, H, W)     # learned "where am I in the frame"
        skips: List[torch.Tensor] = []
        for i, (blocks, down) in enumerate(zip(self.enc_blocks, self.downs)):
            for b in blocks:
                h = b(h)
            if str(i) in self.enc_xattn:
                h = self.enc_xattn[str(i)](h, ctx, mw_mask)             # IR queries attend to MW tokens
            if str(i) in self.enc_sattn:
                h = self.enc_sattn[str(i)](h)                           # IR tokens attend to each other (global)
            skips.append(h)
            h = down(h)                                                  # [B, C_i, H/2^(i+1), ...]
        for m in self.mid:
            h = m(h)
        for k, (blocks, up) in enumerate(zip(self.dec_blocks, self.ups)):
            i = len(self.enc_blocks) - 1 - k
            h = torch.cat([h, skips[i]], dim=1)
            for b in blocks:
                h = b(h)
            if str(i) in self.dec_xattn:
                h = self.dec_xattn[str(i)](h, ctx, mw_mask)
            if str(i) in self.dec_sattn:
                h = self.dec_sattn[str(i)](h)
            h = up(h)
        return self.out(h)                                               # [B, L+1, H, W]

    def split(self, x_det: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_det [B, L+1, H, W] -> temp_n [B, L, H, W], precip_n [B, 1, H, W] (normalised units)."""
        return x_det[:, : self.n_levels], x_det[:, self.n_levels :]
