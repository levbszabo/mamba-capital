from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, in_dim: int, action_space: str = "discrete") -> None:
        super().__init__()
        self.action_space = action_space
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        if action_space == "discrete":
            self.head = nn.Linear(256, 3)  # short / flat / long
        else:
            self.head = nn.Linear(256, 1)  # position size in [-1,1]

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.trunk(s)
        return self.head(x)


class Critic(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


def lambda_returns(
    rewards: torch.Tensor,
    values_tp1: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Compute λ-returns with bootstrap V(s_{t+1}).

    rewards: [T, B]
    values_tp1: [T, B]
    Returns: targets G_t [T, B]
    """
    T, B = rewards.shape
    G = torch.zeros_like(rewards)
    next_val = values_tp1[-1]
    for t in reversed(range(T)):
        td = rewards[t] + gamma * next_val
        G[t] = (
            td
            if t == T - 1
            else rewards[t] + gamma * ((1 - lam) * values_tp1[t] + lam * G[t + 1])
        )
        next_val = values_tp1[t]
    return G
