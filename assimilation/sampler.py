"""
Physics-guided conditional sampling: the generative inverse solver / data-assimilation step.

At every reverse-time step the sample is moved by the prior score and by the likelihood gradient:

    s_post(x_t, t) = s_theta(x_t, t)  +  grad_{x_t} log p(y | x0_hat(x_t))

with x0_hat the Tweedie posterior-mean estimate  (x_t + sigma_t^2 s_theta) / alpha_t  and the
gradient taken through both the likelihood *and* the score network (Diffusion Posterior Sampling,
Chung et al. 2023). The likelihood contains the U-Net proxy, the RTM observation operator and the
thermodynamic penalties (see guidance.py). Integration is a predictor-corrector scheme on the
posterior score: `n_corrector` annealed Langevin steps followed by Euler-Maruyama on the reverse
VP-SDE.

Running `ensemble_size` chains from independent noise gives a posterior ensemble whose spread is
a flow-dependent uncertainty estimate for the retrieved 3D state, exactly the quantity an ensemble
Kalman / 4D-Var system produces at far greater cost.

Shapes: x_t, x0_hat, eps [B, L+1, H, W]; t [B].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..config import GuidanceConfig
from ..models.sde import VPSDE
from .guidance import JointLikelihood, Observations


@dataclass
class SamplerOutput:
    samples: torch.Tensor          # [N, B, L+1, H, W]  ensemble members (normalised state)
    mean: torch.Tensor             # [B, L+1, H, W]
    std: torch.Tensor              # [B, L+1, H, W]
    trace: List[Dict[str, float]]  # likelihood diagnostics logged along the trajectory


class GuidedScoreSampler:
    def __init__(self, score_net: nn.Module, sde: VPSDE, likelihood: JointLikelihood, cfg: GuidanceConfig):
        self.net, self.sde, self.lik, self.cfg = score_net, sde, likelihood, cfg

    # ------------------------------------------------------------------------------------------
    def _guided_score(self, x: torch.Tensor, t: torch.Tensor, dt_abs: float, obs: Observations):
        """Return (posterior score, x0_hat) at (x, t).

        posterior score = s_theta(x_t, t) - guidance_scale * grad_{x_t} J(x0_hat(x_t)),
        J = -log p(y | x0_hat) summed over observations (guidance.py). The gradient is taken through
        the score network (DPS). Because J is a proper sum-form log-likelihood, the reverse SDE
        multiplies the gradient by beta(t)|dt| exactly as it does the prior score, so the observation
        term is weighted consistently with the prior (guidance_scale = 1 is the Bayesian choice).
        Safety: the per-step displacement beta|dt| * gs * grad is capped at `max_step_rms` RMS
        (normalised units), which matters early in the chain where x0_hat is unreliable.
        """
        c = self.cfg
        if float(t[0]) > c.guide_t_max:                                         # prior only, early in the chain
            with torch.no_grad():
                score = self.sde.score_from_eps(self.net(x, t), t)
                x0_hat = self.sde.tweedie_x0(x, score, t)
                return score, c.x0_clip * torch.tanh(x0_hat / c.x0_clip)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            eps = self.net(x, t)
            score = self.sde.score_from_eps(eps, t)
            x0_hat = self.sde.tweedie_x0(x, score, t)
            # Soft clamp of the clean estimate to a plausible range of normalised units: early in the
            # reverse process (alpha_t -> 0) Tweedie estimates are unreliable and would otherwise
            # push the RTM into unphysical temperatures.
            x0_soft = c.x0_clip * torch.tanh(x0_hat / c.x0_clip)
            alpha, sigma = self.sde.marginal_prob(t)
            r2 = float((sigma[0] / alpha[0]) ** 2)                            # Tweedie variance of x0 | x_t
            J = self.lik.neg_log_likelihood(x0_soft, obs, r2=r2)
            grad = torch.autograd.grad(J, x)[0]                             # [B, C, H, W]
        g = c.guidance_scale * grad
        step_rms = (self.sde.beta(t)[:, None, None, None] * dt_abs * g).flatten(1).pow(2).mean(1).sqrt()
        clip = torch.clamp(c.max_step_rms / step_rms.clamp(min=1e-12), max=1.0)[:, None, None, None]
        return (score - clip * g).detach(), x0_soft.detach()

    def _predictor(self, x: torch.Tensor, t: torch.Tensor, dt: float, score: torch.Tensor) -> torch.Tensor:
        """Euler-Maruyama step of the reverse VP-SDE (dt < 0) driven by the posterior score."""
        drift, diffusion = self.sde.drift_diffusion(x, t)
        rev_drift = drift - diffusion**2 * score
        z = torch.randn_like(x)
        return x + rev_drift * dt + diffusion * (-dt) ** 0.5 * z

    def _corrector(self, x: torch.Tensor, t: torch.Tensor, dt_abs: float, obs: Observations) -> torch.Tensor:
        """Annealed Langevin MCMC step on the *posterior* score (Song et al. 2021, Alg. 4 step rule)."""
        for _ in range(self.cfg.n_corrector):
            score, _ = self._guided_score(x, t, dt_abs, obs)
            z = torch.randn_like(x)
            g_norm = score.flatten(1).norm(dim=1).mean()
            z_norm = z.flatten(1).norm(dim=1).mean()
            alpha, _ = self.sde.marginal_prob(t)
            step = 2 * (self.cfg.snr * z_norm / g_norm.clamp(min=1e-8)) ** 2 * alpha.mean()
            x = x + step * score + torch.sqrt(2 * step) * z
        return x

    # ------------------------------------------------------------------------------------------
    def sample_chain(self, obs: Observations, x_init: Optional[torch.Tensor] = None, log_every: int = 50, callback: Optional[Callable] = None):
        """One posterior sample per batch element. Returns (x0 [B,L+1,H,W], trace).

        Per step: Langevin corrector -> Euler-Maruyama predictor, both driven by the posterior score
        (prior score + U-Net proxy + RTM + thermodynamic penalties).
        """
        B = obs.x_det.shape[0]
        device = obs.x_det.device
        x = self.sde.prior_sample(obs.x_det.shape, device) if x_init is None else x_init
        ts = torch.linspace(1.0, self.cfg.final_denoise_t, self.cfg.n_steps + 1, device=device)
        trace = []
        self.net.eval()
        for i in range(self.cfg.n_steps):
            t = ts[i].expand(B)
            dt = float(ts[i + 1] - ts[i])                  # negative
            x = self._corrector(x, t, -dt, obs)
            score, x0_hat = self._guided_score(x, t, -dt, obs)
            x = self._predictor(x, t, dt, score)
            if log_every and (i % log_every == 0 or i == self.cfg.n_steps - 1):
                d = self.lik.diagnostics(x0_hat, obs)
                d["step"], d["t"] = i, float(ts[i])
                trace.append(d)
                if callback:
                    callback(d)
        # Final denoising: the Tweedie estimate at t = final_denoise_t is the posterior mean over the
        # remaining small-scale noise (sigma ~ 0.1 - 0.2), i.e. a sample of the large-scale posterior
        # without pixel-level diffusion noise leaking into the precipitation field.
        t = ts[-1].expand(B)
        _, x0 = self._guided_score(x, t, -dt, obs)
        return x0, trace

    def sample(self, obs: Observations, ensemble_size: Optional[int] = None, **kw) -> SamplerOutput:
        n = ensemble_size or self.cfg.ensemble_size
        members, trace = [], []
        for k in range(n):
            torch.manual_seed(torch.initial_seed() + k)
            x0, tr = self.sample_chain(obs, **kw)
            members.append(x0)
            trace.extend([{**d, "member": k} for d in tr])
        samples = torch.stack(members, 0)                                # [N, B, L+1, H, W]
        return SamplerOutput(samples=samples, mean=samples.mean(0), std=samples.std(0) if n > 1 else torch.zeros_like(samples[0]), trace=trace)
