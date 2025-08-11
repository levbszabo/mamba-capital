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


class RSSM(nn.Module):
    """Deterministic RSSM variant with a GRU transition and linear decoder.

    Transition:
      h_{t+1} = GRUCell([z_t, a_t], h_t)
      zhat_{t+1} = W_out h_{t+1}

    Loss:
      L = MSE(zhat_{t+1}, z_{t+1}) averaged over batch/sequence.
    """

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.latent_dim + (cfg.action_dim if cfg.action_dim > 0 else 0)
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim) if input_dim > 1 else nn.Identity(),
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        # Small init helps stability
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        a_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One transition step.

        Args:
          z_t: [B, latent_dim]
          h_t: [B, hidden_dim]
          a_t: [B, action_dim] or None
        Returns:
          z_hat_tp1: [B, latent_dim]
          h_tp1: [B, hidden_dim]
        """
        if a_t is not None and self.cfg.action_dim > 0:
            x = torch.cat([z_t, a_t], dim=-1)
        else:
            x = z_t
        x = self.in_proj(x)
        h_tp1 = self.gru(x, h_t)
        z_hat_tp1 = self.out_proj(h_tp1)
        return z_hat_tp1, h_tp1

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
            z, h = self.forward(z, h, a)
            traj.append(z)
        return torch.stack(traj, dim=1)

    @staticmethod
    def loss_mse(pred_next_z: torch.Tensor, true_next_z: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred_next_z - true_next_z) ** 2)
