"""
Model definitions for Mamba-style multihorizon forecasting with Transformer fallback.

The model consumes:
- x_cont: [batch, seq_len, num_cont_features]
- x_symbol_id: [batch, seq_len]
- x_ny_hour_id: [batch, seq_len]
- x_ny_dow_id: [batch, seq_len]

Outputs (per sample/window, at the last time step):
- mean: [batch, num_horizons]
- log_var: [batch, num_horizons]

If `mamba-ssm` is available, the backbone can be a stack of Mamba layers.
Otherwise, we fallback to a TransformerEncoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional

import torch
import torch.nn as nn


def try_import_mamba():
    try:
        from mamba_ssm import Mamba  # type: ignore

        return Mamba
    except Exception:
        pass
    try:
        from mamba_ssm.models.mamba import Mamba  # type: ignore

        return Mamba
    except Exception:
        return None


@dataclass
class ModelConfig:
    num_cont_features: int
    num_symbols: int
    hour_size: int
    dow_size: int
    num_bases: int | None = None
    num_quotes: int | None = None
    num_horizons: int
    d_model: int = 384
    n_layers: int = 6
    dropout: float = 0.1
    # Transformer specific
    n_heads: int = 8
    dim_feedforward: int = 1536
    # Embedding sizes before projection
    symbol_emb_dim: int = 32
    hour_emb_dim: int = 16
    dow_emb_dim: int = 8
    base_emb_dim: int = 8
    quote_emb_dim: int = 8
    # Backbone selection
    backbone: Literal["auto", "mamba", "transformer"] = "auto"
    # Mamba-specific hyperparameters (used if available)
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    # Sequence pooling
    pooling: Literal["last", "mean", "attn"] = "last"
    # Latent representation size for downstream tasks (e.g., RL)
    latent_dim: int = 64


class ResidualBlock(nn.Module):
    def __init__(self, module: nn.Module, d_model: int, dropout: float):
        super().__init__()
        self.module = module
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.module(x)
        x = self.dropout(x)
        return x + residual


class MambaBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
    ):
        super().__init__()
        Mamba = try_import_mamba()
        if Mamba is None:
            raise RuntimeError("mamba-ssm is not available")
        blocks = []
        for _ in range(n_layers):
            block = Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            blocks.append(ResidualBlock(block, d_model=d_model, dropout=dropout))
        self.layers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ForecastingModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Project continuous features to d_model
        self.cont_proj = nn.Sequential(
            nn.Linear(cfg.num_cont_features, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

        # Categorical embeddings → concat → project to d_model
        self.symbol_emb = nn.Embedding(cfg.num_symbols, cfg.symbol_emb_dim)
        self.hour_emb = nn.Embedding(cfg.hour_size, cfg.hour_emb_dim)
        self.dow_emb = nn.Embedding(cfg.dow_size, cfg.dow_emb_dim)
        self.base_emb = (
            nn.Embedding(cfg.num_bases, cfg.base_emb_dim)
            if cfg.num_bases is not None and cfg.num_bases > 0
            else None
        )
        self.quote_emb = (
            nn.Embedding(cfg.num_quotes, cfg.quote_emb_dim)
            if cfg.num_quotes is not None and cfg.num_quotes > 0
            else None
        )

        cat_total = cfg.symbol_emb_dim + cfg.hour_emb_dim + cfg.dow_emb_dim
        if self.base_emb is not None:
            cat_total += cfg.base_emb_dim
        if self.quote_emb is not None:
            cat_total += cfg.quote_emb_dim
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_total, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

        # Backbone selection
        backbone_choice = cfg.backbone
        if backbone_choice == "auto":
            Mamba = try_import_mamba()
            backbone_choice = "mamba" if Mamba is not None else "transformer"

        if backbone_choice == "mamba":
            self.backbone = MambaBackbone(
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout,
                d_state=cfg.mamba_d_state,
                d_conv=cfg.mamba_d_conv,
                expand=cfg.mamba_expand,
            )
        elif backbone_choice == "transformer":
            self.backbone = TransformerBackbone(
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")

        # Output heads: independent Gaussian per horizon
        self.head_mean = nn.Linear(cfg.d_model, cfg.num_horizons)
        self.head_logvar = nn.Linear(cfg.d_model, cfg.num_horizons)

        # Latent projection for representation learning / RL
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.latent_dim),
        )

        # Small init for stability
        nn.init.zeros_(self.head_mean.weight)
        nn.init.zeros_(self.head_mean.bias)
        nn.init.zeros_(self.head_logvar.weight)
        nn.init.constant_(self.head_logvar.bias, -2.0)

    def forward(
        self,
        x_cont: torch.Tensor,
        x_symbol_id: torch.Tensor,
        x_ny_hour_id: torch.Tensor,
        x_ny_dow_id: torch.Tensor,
        x_base_id: torch.Tensor | None = None,
        x_quote_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        # x_cont: [B, L, C]
        # x_*_id: [B, L]
        b, l, _ = x_cont.shape

        cont = self.cont_proj(x_cont)

        sym_e = self.symbol_emb(x_symbol_id)
        hour_e = self.hour_emb(x_ny_hour_id)
        dow_e = self.dow_emb(x_ny_dow_id)
        embeds = [sym_e, hour_e, dow_e]
        if self.base_emb is not None and x_base_id is not None:
            embeds.append(self.base_emb(x_base_id))
        if self.quote_emb is not None and x_quote_id is not None:
            embeds.append(self.quote_emb(x_quote_id))
        cat = torch.cat(embeds, dim=-1)
        cat = self.cat_proj(cat)

        x = cont + cat
        x = self.backbone(x)
        if self.cfg.pooling == "mean":
            h = x.mean(dim=1)
        elif self.cfg.pooling == "attn":
            # Lightweight attention pooling: parameterized query vector
            if not hasattr(self, "attn_q"):
                self.attn_q = nn.Parameter(torch.randn(self.cfg.d_model))
            q = self.attn_q.view(1, 1, -1)
            attn = torch.softmax(
                torch.matmul(x, q.transpose(1, 2)).squeeze(-1)
                / (self.cfg.d_model**0.5),
                dim=1,
            )
            h = (x * attn.unsqueeze(-1)).sum(dim=1)
        else:
            h = x[:, -1, :]

        mean = self.head_mean(h)
        log_var = self.head_logvar(h)
        return mean, log_var

    @torch.no_grad()
    def encode(
        self,
        x_cont: torch.Tensor,
        x_symbol_id: torch.Tensor,
        x_ny_hour_id: torch.Tensor,
        x_ny_dow_id: torch.Tensor,
        x_base_id: torch.Tensor | None = None,
        x_quote_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns a low-dimensional latent representation for downstream tasks.

        Shape: inputs as forward; output [batch, latent_dim]
        """
        cont = self.cont_proj(x_cont)
        sym_e = self.symbol_emb(x_symbol_id)
        hour_e = self.hour_emb(x_ny_hour_id)
        dow_e = self.dow_emb(x_ny_dow_id)
        embeds = [sym_e, hour_e, dow_e]
        if self.base_emb is not None and x_base_id is not None:
            embeds.append(self.base_emb(x_base_id))
        if self.quote_emb is not None and x_quote_id is not None:
            embeds.append(self.quote_emb(x_quote_id))
        cat = torch.cat(embeds, dim=-1)
        cat = self.cat_proj(cat)
        x = cont + cat
        x = self.backbone(x)
        if self.cfg.pooling == "mean":
            h = x.mean(dim=1)
        elif self.cfg.pooling == "attn":
            if not hasattr(self, "attn_q"):
                self.attn_q = nn.Parameter(torch.randn(self.cfg.d_model))
            q = self.attn_q.view(1, 1, -1)
            attn = torch.softmax(
                torch.matmul(x, q.transpose(1, 2)).squeeze(-1)
                / (self.cfg.d_model**0.5),
                dim=1,
            )
            h = (x * attn.unsqueeze(-1)).sum(dim=1)
        else:
            h = x[:, -1, :]
        return self.latent_proj(h)


def gaussian_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # mean/log_var/target: [batch, H]
    var = torch.exp(log_var)
    # Use numeric constant for log(2*pi) to avoid torch.log on float
    log_two_pi = math.log(2.0 * math.pi)
    nll = 0.5 * (log_var + (target - mean) ** 2 / var) + 0.5 * log_two_pi
    if weights is not None:
        nll = nll * weights.view(1, -1)
    return nll.mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
