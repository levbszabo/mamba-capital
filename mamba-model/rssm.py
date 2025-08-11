"""
RSSM-style world model for FX latent dynamics.

This module provides a light-weight deterministic RSSM variant that learns to
predict the next latent state z_{t+1} given the current latent z_t and an
optional action a_t. It is designed to sit on top of the encoder provided by
ForecastingModel.encode(...), so you can train it on sequences of window-level
latents extracted from your trained forecaster.

The design keeps the interface compatible with a Dreamer-style training loop,
while staying minimal and robust for first use:

- Deterministic transition: GRUCell(h_t) with linear readout to z_{t+1}
- Optional action conditioning (concatenated to z_t)
- MSE loss on next-latent prediction; optionally KL to a unit Gaussian prior
  can be added later if you introduce a stochastic head

Usage:
  - Train the forecaster
  - Extract z_t per time step (one z per window, using encode())
  - Train the RSSM on (z_t, a_t -> z_{t+1}) pairs
  - Use rssm.imagine(...) for latent rollouts during RL planning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class RSSMConfig:
    latent_dim: int
    hidden_dim: int = 256
    action_dim: int = 0  # set >0 to condition on actions
    dropout: float = 0.0
    stochastic: bool = True


class RSSM(nn.Module):
    """RSSM with deterministic core and optional stochastic latent z.

    - Deterministic-only mode (stochastic=False): matches prior impl
    - Stochastic mode (default): prior p(z_t|h_{t-1}), posterior q(z_t|h_{t-1}, e_t)
    """

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        # Core transition uses z (sampled or deterministic) as input
        core_in_dim = cfg.latent_dim + (cfg.action_dim if cfg.action_dim > 0 else 0)
        self.core_in = nn.Sequential(
            nn.LayerNorm(core_in_dim) if core_in_dim > 1 else nn.Identity(),
            nn.Linear(core_in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)

        if cfg.stochastic:
            # Prior and posterior parameter heads (μ, logσ^2)
            self.prior_net = nn.Linear(cfg.hidden_dim, 2 * cfg.latent_dim)
            self.post_net = nn.Linear(
                cfg.hidden_dim + cfg.latent_dim, 2 * cfg.latent_dim
            )
        else:
            # Deterministic projection for next z
            self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward_deterministic(
        self, z_t: torch.Tensor, h_t: torch.Tensor, a_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if a_t is not None and self.cfg.action_dim > 0:
            x = torch.cat([z_t, a_t], dim=-1)
        else:
            x = z_t
        x = self.core_in(x)
        h_tp1 = self.gru(x, h_t)
        z_hat_tp1 = self.out_proj(h_tp1)
        return z_hat_tp1, h_tp1

    @staticmethod
    def _split_stats(stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        return mu, logvar

    @staticmethod
    def _rsample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_gaussian(
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
        # KL(q||p) for diagonal Gaussians
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * ((logvar_p - logvar_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
        return kl.sum(dim=-1)  # sum over latent dims

    def posterior(
        self, h_prev: torch.Tensor, obs_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.cfg.stochastic, "posterior only in stochastic mode"
        stats = self.post_net(torch.cat([h_prev, obs_embed], dim=-1))
        return self._split_stats(stats)

    def prior(self, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.cfg.stochastic, "prior only in stochastic mode"
        stats = self.prior_net(h_prev)
        return self._split_stats(stats)

    def core(
        self,
        z_t: torch.Tensor,
        h_prev: torch.Tensor,
        a_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if a_t is not None and self.cfg.action_dim > 0:
            x = torch.cat([z_t, a_t], dim=-1)
        else:
            x = z_t
        x = self.core_in(x)
        h_t = self.gru(x, h_prev)
        return h_t

    def imagine(
        self,
        z0: torch.Tensor,
        horizon: int,
        policy_fn=None,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Roll forward in latent space for `horizon` steps.

        Args:
          z0: [B, latent_dim]
          horizon: number of steps to roll
          policy_fn: callable taking (z_t) -> a_t tensor or None
          h0: optional initial hidden state [B, hidden_dim]

        Returns:
          Z: [B, horizon+1, latent_dim] including the initial z0
        """
        batch_size = z0.size(0)
        device = z0.device
        h = (
            h0
            if h0 is not None
            else torch.zeros(batch_size, self.cfg.hidden_dim, device=device)
        )
        z = z0
        traj = [z]
        for _ in range(horizon):
            a = None
            if self.cfg.action_dim > 0 and policy_fn is not None:
                a = policy_fn(z)
            if self.cfg.stochastic:
                # Use prior for imagination
                mu_p, logvar_p = self.prior(h)
                z = self._rsample(mu_p, logvar_p)
                h = self.core(z, h, a)
            else:
                z, h = self.forward_deterministic(z, h, a)
            traj.append(z)
        return torch.stack(traj, dim=1)

    @staticmethod
    def loss_mse(pred_next_z: torch.Tensor, true_next_z: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred_next_z - true_next_z) ** 2)
