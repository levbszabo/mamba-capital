"""
Dreamer-style world model components for FX hourly data.

Provides:
- ObsEncoder: stepwise embedding of observation (continuous + categorical)
- RSSM (imported): stochastic latent dynamics p(z_t|h_{t-1}), q(z_t|h_{t-1}, e_t)
- ObsDecoder: reconstructs continuous features p(x_t|z_t)
- ReturnHead: heteroscedastic prediction p(y|z_last) for multi-horizon returns

Also exposes small utility losses for Gaussian NLL and KL handling with free-nats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rssm import RSSM, RSSMConfig


def gaussian_nll(
    x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Mean NLL per batch for diagonal Gaussian.

    Reduces over all non-batch dims and returns mean over batch.
    """
    var = torch.exp(logvar)
    nll = 0.5 * (
        (x - mu) ** 2 / var
        + logvar
        + torch.log(
            torch.tensor(2.0 * 3.141592653589793, device=x.device, dtype=x.dtype)
        )
    )
    # Reduce over feature/time dims, mean over batch
    while nll.dim() > 1:
        nll = nll.mean(dim=-1)
    return nll.mean()


def student_t_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_s: torch.Tensor,
    nu: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean NLL per batch for independent Student-t per feature/horizon.

    x, mu, log_s broadcast across last dim; nu should be broadcastable too.
    Returns mean over all non-batch dims and mean over batch.
    """
    s = F.softplus(log_s) + eps
    # Terms per element
    t1 = torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0)
    t2 = 0.5 * (torch.log(nu) + math.log(math.pi)) + torch.log(s)
    t3 = ((nu + 1.0) / 2.0) * torch.log1p(((x - mu) ** 2) / (nu * s * s + eps))
    nll = t1 + t2 + t3
    while nll.dim() > 1:
        nll = nll.mean(dim=-1)
    return nll.mean()


def apply_free_nats(kl: torch.Tensor, free_nats: float) -> torch.Tensor:
    """Clamp KL per-sample by free nats, then mean."""
    return torch.clamp(kl - float(free_nats), min=0.0).mean()


@dataclass
class WorldModelConfig:
    num_cont_features: int
    num_symbols: int
    hour_size: int
    dow_size: int
    num_bases: Optional[int] = None
    num_quotes: Optional[int] = None

    embed_dim: int = 128
    latent_dim: int = 64
    hidden_dim: int = 256
    dropout: float = 0.1

    # Loss weights
    recon_weight: float = 1.0
    kl_beta: float = 1.0
    kl_free_nats: float = 2.0
    ret_weight: float = 0.3


class ObsEncoder(nn.Module):
    """Stepwise embedding encoder producing e_t per step.

    Input tensors are shape [B, L, ...]. Returns [B, L, embed_dim].
    """

    def __init__(
        self,
        num_cont_features: int,
        num_symbols: int,
        hour_size: int,
        dow_size: int,
        num_bases: Optional[int],
        num_quotes: Optional[int],
        embed_dim: int = 128,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        symbol_emb_dim: int = 32,
        hour_emb_dim: int = 16,
        dow_emb_dim: int = 8,
        base_emb_dim: int = 8,
        quote_emb_dim: int = 8,
    ) -> None:
        super().__init__()
        self.num_bases = num_bases
        self.num_quotes = num_quotes

        self.cont_proj = nn.Sequential(
            nn.Linear(num_cont_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU(),
        )
        self.symbol_emb = nn.Embedding(num_symbols, symbol_emb_dim)
        self.hour_emb = nn.Embedding(hour_size, hour_emb_dim)
        self.dow_emb = nn.Embedding(dow_size, dow_emb_dim)
        self.base_emb = (
            nn.Embedding(num_bases, base_emb_dim)
            if (num_bases is not None and num_bases > 0)
            else None
        )
        self.quote_emb = (
            nn.Embedding(num_quotes, quote_emb_dim)
            if (num_quotes is not None and num_quotes > 0)
            else None
        )
        cat_total = symbol_emb_dim + hour_emb_dim + dow_emb_dim
        if self.base_emb is not None:
            cat_total += base_emb_dim
        if self.quote_emb is not None:
            cat_total += quote_emb_dim
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_total, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, embed_dim))

    def forward(
        self,
        x_cont: torch.Tensor,
        x_sym: torch.Tensor,
        x_hour: torch.Tensor,
        x_dow: torch.Tensor,
        x_base: Optional[torch.Tensor] = None,
        x_quote: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cont = self.cont_proj(x_cont)
        embeds = [self.symbol_emb(x_sym), self.hour_emb(x_hour), self.dow_emb(x_dow)]
        if self.base_emb is not None and x_base is not None:
            embeds.append(self.base_emb(x_base))
        if self.quote_emb is not None and x_quote is not None:
            embeds.append(self.quote_emb(x_quote))
        cat = torch.cat(embeds, dim=-1)
        cat = self.cat_proj(cat)
        x = cont + cat
        y, _ = self.gru(x)
        e_seq = self.out(y)
        return e_seq  # [B, L, embed_dim]


class ObsDecoder(nn.Module):
    """Decode z_t to continuous feature distribution N(mu, diag(var))."""

    def __init__(self, latent_dim: int, num_cont_features: int) -> None:
        super().__init__()
        self.mu = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_cont_features),
        )
        self.logvar = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_cont_features),
        )

    def forward(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(z_t), self.logvar(z_t)


class ReturnHead(nn.Module):
    """Predict heteroscedastic future returns for multiple horizons from z_last.

    Uses a tiny MLP for mild nonlinearity to help μ move without overfitting.
    """

    def __init__(self, latent_dim: int, num_horizons: int) -> None:
        super().__init__()
        self.mu = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_horizons),
        )
        self.logvar = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_horizons),
        )

    def forward(self, z_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(z_last), self.logvar(z_last)


class StudentTHead(nn.Module):
    """Predict Student-t parameters per horizon from latent.

    Outputs:
      mu: [B, H]
      log_s: [B, H] (scale > 0 via softplus)
      nu: [H] degrees of freedom (learned, shared across batch)
    """

    def __init__(self, latent_dim: int, num_horizons: int) -> None:
        super().__init__()
        self.mu = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_horizons),
        )
        self.log_s = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_horizons),
        )
        # One nu per horizon; constrained via softplus+2 in forward users
        self.nu_raw = nn.Parameter(torch.zeros(num_horizons))

    def forward(
        self, z_last: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu(z_last)
        log_s = self.log_s(z_last)
        nu = F.softplus(self.nu_raw) + 2.0
        return mu, log_s, nu


class SignHead(nn.Module):
    """Predict logits for sign(y) per horizon from latent.

    Logits > 0 => predict positive return.
    """

    def __init__(self, latent_dim: int, num_horizons: int) -> None:
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, num_horizons)
        )

    def forward(self, z_last: torch.Tensor) -> torch.Tensor:
        return self.out(z_last)


def roll_posterior_through_time(
    encoder: ObsEncoder,
    rssm: RSSM,
    decoder: ObsDecoder,
    x_cont: torch.Tensor,
    x_sym: torch.Tensor,
    x_hour: torch.Tensor,
    x_dow: torch.Tensor,
    x_base: Optional[torch.Tensor],
    x_quote: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs posterior over a sequence and returns:
    - z_seq [B, L, Dz]
    - mu_x [B, L, C]
    - logv_x [B, L, C]
    - h_last [B, H]
    """
    # We compute with grad in training; callers may wrap in no_grad if needed.
    E = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)  # [B, L, De]
    B, L, _ = E.shape
    device = x_cont.device
    Dz = rssm.cfg.latent_dim
    H = rssm.cfg.hidden_dim
    h = torch.zeros(B, H, device=device, dtype=E.dtype)
    z_seq = []
    mu_x_seq = []
    logv_x_seq = []
    for t in range(L):
        mu_p, logv_p = rssm.prior(h)
        mu_q, logv_q = rssm.posterior(h, E[:, t, :])
        z = rssm._rsample(mu_q, logv_q)
        h = rssm.core(z, h, None)
        mu_x_t, logv_x_t = decoder(z)
        z_seq.append(z)
        mu_x_seq.append(mu_x_t)
        logv_x_seq.append(logv_x_t)
    Z = torch.stack(z_seq, dim=1)
    MU_X = torch.stack(mu_x_seq, dim=1)
    LOGV_X = torch.stack(logv_x_seq, dim=1)
    return Z, MU_X, LOGV_X, h
