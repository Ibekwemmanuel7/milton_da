"""
Graph microwave context encoder: message passing over the coarse sounder grid.

Experiment (September 2026): a second, switchable microwave path for the proxy. The existing
MicrowaveContextEncoder is convolutional (3 x 3 stencils on the regridded h x w field, invalid pixels
seen as zeros). This encoder treats every coarse pixel as a graph node and builds, per sample, a
k-nearest-neighbour graph over the *valid* nodes only, so a swath edge or a gap changes the graph
rather than injecting zeros. Messages carry the relative displacement between nodes as an edge
feature (a continuous-kernel convolution in the GraphCast / MeshGraphNet style), and aggregation is a
mask-aware mean. Output has the same shape as the convolutional encoder, [B, C_ctx, h, w], so the
cross-attention downstream is untouched and the two encoders can be compared like for like.

Honest scope: the nodes are the regridded pixels, not the ATMS footprints, so this tests message
passing against convolution on identical information. It cannot show the footprint advantage; that
needs footprint tables saved at scene-preparation time (see the roadmap).

Shapes: nodes N = h * w (256 at 16 x 16, 1024 at 32 x 32); kNN over N x N distances is cheap here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingLayer(nn.Module):
    """h_i <- h_i + psi(h_i, mean_j phi(h_i, h_j, e_ij)) over the k valid neighbours j of i."""

    def __init__(self, c: int, c_edge: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or 2 * c
        self.edge_mlp = nn.Sequential(nn.Linear(2 * c + c_edge, hidden), nn.SiLU(), nn.Linear(hidden, c))
        self.node_mlp = nn.Sequential(nn.Linear(2 * c, hidden), nn.SiLU(), nn.Linear(hidden, c))
        self.norm = nn.LayerNorm(c)
        nn.init.zeros_(self.node_mlp[-1].weight), nn.init.zeros_(self.node_mlp[-1].bias)   # identity at init

    def forward(self, h: torch.Tensor, nbr: torch.Tensor, edge: torch.Tensor, nbr_valid: torch.Tensor) -> torch.Tensor:
        """
        h          [B, N, C]      node states
        nbr        [B, N, K]      neighbour indices
        edge       [B, N, K, E]   edge features (relative displacement, distance)
        nbr_valid  [B, N, K]      1 where the neighbour is a valid observation, else 0
        """
        B, N, C = h.shape
        K = nbr.shape[-1]
        h_j = torch.gather(h, 1, nbr.reshape(B, N * K, 1).expand(-1, -1, C)).view(B, N, K, C)   # [B, N, K, C]
        h_i = h[:, :, None, :].expand(-1, -1, K, -1)
        m = self.edge_mlp(torch.cat([h_i, h_j, edge], dim=-1)) * nbr_valid[..., None]            # masked messages
        agg = m.sum(2) / nbr_valid.sum(-1, keepdim=True).clamp(min=1.0)                          # mask-aware mean
        return h + self.node_mlp(torch.cat([self.norm(h), agg], dim=-1))


class GraphMicrowaveEncoder(nn.Module):
    """[B, C_mw + 2, h, w] (TB, mask, zenith) -> [B, C_ctx, h, w] via message passing on a kNN graph of valid pixels."""

    def __init__(self, c_in: int, c_ctx: int, k: int = 16, n_rounds: int = 3):
        super().__init__()
        self.k, self.n_rounds = k, n_rounds
        self.embed = nn.Sequential(nn.Linear(c_in + 2, c_ctx), nn.SiLU(), nn.Linear(c_ctx, c_ctx))   # + normalised (y, x)
        self.layers = nn.ModuleList([MessagePassingLayer(c_ctx, c_edge=3) for _ in range(n_rounds)])
        self.out = nn.Linear(c_ctx, c_ctx)

    @staticmethod
    def _coords(h: int, w: int, device) -> torch.Tensor:                       # [N, 2] in [-1, 1]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing="ij")
        return torch.stack([yy, xx], -1).reshape(h * w, 2)

    def build_graph(self, valid: torch.Tensor, h: int, w: int):
        """valid [B, N] -> nbr [B, N, K], edge [B, N, K, 3], nbr_valid [B, N, K].
        Each node's K nearest *valid* nodes (excluding itself). Nodes with fewer than K valid
        neighbours (or none at all) get padded links that are masked out of aggregation."""
        B, N = valid.shape
        K = min(self.k, N - 1)
        xy = self._coords(h, w, valid.device)                                   # [N, 2]
        d = torch.cdist(xy, xy)                                                 # [N, N]
        d = d[None].expand(B, -1, -1).clone()
        d.diagonal(dim1=1, dim2=2).fill_(float("inf"))                          # no self-loops
        d = d.masked_fill(~valid[:, None, :].bool(), float("inf"))             # only valid targets
        dist, nbr = torch.topk(d, K, dim=-1, largest=False)                     # [B, N, K]
        nbr_valid = torch.isfinite(dist).float()
        nbr = torch.where(torch.isfinite(dist), nbr, torch.zeros_like(nbr))     # padded links point at node 0, masked
        rel = xy[nbr] - xy[:, None, :][None]                                    # [B, N, K, 2] displacement j - i
        edge = torch.cat([rel, torch.where(torch.isfinite(dist), dist, torch.zeros_like(dist))[..., None]], -1)
        return nbr, edge, nbr_valid

    def forward(self, mw: torch.Tensor, mw_mask: torch.Tensor, mw_zen: torch.Tensor) -> torch.Tensor:
        B, C, h, w = mw.shape
        N = h * w
        x = torch.cat([mw, mw_mask, mw_zen], dim=1).flatten(2).transpose(1, 2)  # [B, N, C+2]
        xy = self._coords(h, w, mw.device)[None].expand(B, -1, -1)
        hnode = self.embed(torch.cat([x, xy], dim=-1))                          # [B, N, C_ctx]
        valid = mw_mask.flatten(1)                                              # [B, N]
        nbr, edge, nbr_valid = self.build_graph(valid, h, w)
        for layer in self.layers:
            hnode = layer(hnode, nbr, edge, nbr_valid)
        hnode = self.out(hnode) * valid[..., None]                              # invalid pixels contribute nothing downstream
        return hnode.transpose(1, 2).reshape(B, -1, h, w)
