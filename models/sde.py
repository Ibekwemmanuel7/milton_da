"""
Variance-Preserving SDE (Song et al., 2021) and the denoising score-matching objective.

Forward SDE :  dx = -1/2 beta(t) x dt + sqrt(beta(t)) dW,          t in [0, 1]
Marginal    :  x_t = alpha(t) x_0 + sigma(t) z,   alpha(t) = exp(-1/2 int beta),  sigma^2 = 1 - alpha^2
Reverse SDE :  dx = [-1/2 beta(t) x - beta(t) s(x, t)] dt + sqrt(beta(t)) dW_bar
Tweedie     :  E[x_0 | x_t] = (x_t + sigma(t)^2 s(x_t, t)) / alpha(t)

The score network is parameterised to output eps-prediction internally; `score()` converts to
grad log p_t(x) = -eps / sigma(t). This keeps the DSM loss well-scaled across t.
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch

from ..config import SDEConfig


class VPSDE:
    def __init__(self, cfg: SDEConfig):
        self.beta_min, self.beta_max, self.t_eps = cfg.beta_min, cfg.beta_max, cfg.t_eps

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def log_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def marginal_prob(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """t [B] -> alpha [B], sigma [B] such that x_t = alpha x_0 + sigma z."""
        alpha = torch.exp(self.log_alpha(t))
        sigma = torch.sqrt((1.0 - alpha**2).clamp(min=1e-10))
        return alpha, sigma

    def perturb(self, x0: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """x0 [B,C,H,W], t [B], z [B,C,H,W] -> x_t [B,C,H,W]."""
        alpha, sigma = self.marginal_prob(t)
        return alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * z

    def prior_sample(self, shape, device) -> torch.Tensor:
        return torch.randn(shape, device=device)

    def drift_diffusion(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.beta(t)[:, None, None, None]
        return -0.5 * b * x, torch.sqrt(b)

    # -- score / x0 conversions --------------------------------------------------------------
    def score_from_eps(self, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, sigma = self.marginal_prob(t)
        return -eps / sigma[:, None, None, None]

    def tweedie_x0(self, x_t: torch.Tensor, score: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.marginal_prob(t)
        return (x_t + sigma[:, None, None, None] ** 2 * score) / alpha[:, None, None, None]

    # -- training objective --------------------------------------------------------------------
    def dsm_loss(self, eps_model: Callable, x0: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Denoising score matching with eps-parameterisation (equivalent to weighted score matching).
        x0 [B,C,H,W] normalised states. Returns scalar loss."""
        B = x0.shape[0]
        t = torch.rand(B, device=x0.device) * (1.0 - self.t_eps) + self.t_eps
        z = torch.randn_like(x0)
        x_t = self.perturb(x0, t, z)
        eps_hat = eps_model(x_t, t, cond)
        return torch.mean((eps_hat - z) ** 2)
