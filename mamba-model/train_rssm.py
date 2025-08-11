"""
Train a light-weight deterministic RSSM on top of a trained forecaster.

Flow:
 1) Load dataset tensors and the best trained forecaster checkpoint
 2) For each batch, get encoder latents z_t for all sequence positions
 3) Train RSSM to predict z_{t+1} from z_t (no actions for now)

This learns a latent dynamics model that can be used for Dreamer-style
imagination. PPO/actor-critic training can use imagine() to plan ahead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import ForecastingModel, ModelConfig
from rssm import RSSM, RSSMConfig


def load_tensors(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict]:
    data: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ["train", "val", "test"]:
        data[split] = torch.load(
            dataset_dir / f"dataset_{split}.pt", map_location="cpu"
        )
    meta = torch.load(dataset_dir / "meta.pt", map_location="cpu")
    return data, meta


def make_dataloader(
    tensors: Dict[str, torch.Tensor], batch_size: int, num_workers: int
) -> DataLoader:
    # We need full sequences to extract z for each timestep
    x_cont = tensors["x_cont"]
    x_sym = tensors["x_symbol_id"]
    x_hour = tensors["x_ny_hour_id"]
    x_dow = tensors["x_ny_dow_id"]
    y = tensors["y"]
    x_base = tensors.get("x_base_id")
    x_quote = tensors.get("x_quote_id")
    if x_base is not None and x_quote is not None:
        ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y)
    else:
        ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


@torch.no_grad()
def encode_all_steps(
    model: ForecastingModel,
    x_cont: torch.Tensor,
    x_sym: torch.Tensor,
    x_hour: torch.Tensor,
    x_dow: torch.Tensor,
    x_base: torch.Tensor | None,
    x_quote: torch.Tensor | None,
) -> torch.Tensor:
    """Return latent per time step using the encoder+backbone with mean pooling over prefixes.

    For step t, we use the subsequence up to t to produce a pooled representation,
    which yields a consistent summary latent z_t per step.
    """
    B, L, _ = x_cont.shape
    zs = []
    for t in range(1, L + 1):
        z_t = model.encode(
            x_cont[:, :t, :],
            x_sym[:, :t],
            x_hour[:, :t],
            x_dow[:, :t],
            x_base[:, :t] if x_base is not None else None,
            x_quote[:, :t] if x_quote is not None else None,
        )
        zs.append(z_t)
    return torch.stack(zs, dim=1)  # [B, L, latent_dim]


def parse_args():
    p = argparse.ArgumentParser(description="Train RSSM on top of trained forecaster")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument(
        "--checkpoint", type=str, required=True, help="Path to best_model.pt"
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, meta = load_tensors(Path(args.dataset_dir))
    vocab = meta["vocab"]
    feature_names = meta["feature_names"]
    horizons = meta["horizons"]

    # Rebuild model and load checkpoint
    cfg = ModelConfig(
        num_cont_features=len(feature_names),
        num_symbols=int(vocab["num_symbols"]),
        hour_size=int(vocab["hour_size"]),
        dow_size=int(vocab["dow_size"]),
        num_bases=int(vocab.get("num_bases", 0)) or None,
        num_quotes=int(vocab.get("num_quotes", 0)) or None,
        num_horizons=len(horizons),
    )
    forecaster = ForecastingModel(cfg).to(device)
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    forecaster.load_state_dict(ckpt["model_state"])  # type: ignore
    forecaster.eval()

    latent_dim = int(getattr(forecaster.cfg, "latent_dim", 64))
    rssm = RSSM(RSSMConfig(latent_dim=latent_dim)).to(device)
    optim = torch.optim.Adam(rssm.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    train_loader = make_dataloader(data["train"], args.batch_size, args.num_workers)
    val_loader = make_dataloader(data["val"], args.batch_size, args.num_workers)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        rssm.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            if len(batch) == 7:
                x_cont, x_sym, x_hour, x_dow, x_base, x_quote, _ = batch
            else:
                x_cont, x_sym, x_hour, x_dow, _ = batch
                x_base = None
                x_quote = None
            x_cont = x_cont.to(device)
            x_sym = x_sym.to(device)
            x_hour = x_hour.to(device)
            x_dow = x_dow.to(device)
            if x_base is not None:
                x_base = x_base.to(device)
            if x_quote is not None:
                x_quote = x_quote.to(device)

            with torch.no_grad():
                Z = encode_all_steps(
                    forecaster, x_cont, x_sym, x_hour, x_dow, x_base, x_quote
                )  # [B,L,D]

            z_t = Z[:, :-1, :].contiguous().view(-1, Z.size(-1))
            z_tp1 = Z[:, 1:, :].contiguous().view(-1, Z.size(-1))

            # Train one-step prediction without actions
            B = z_t.size(0)
            h = torch.zeros(B, rssm.cfg.hidden_dim, device=device)
            z_hat, _ = rssm(z_t, h, None)
            loss = crit(z_hat, z_tp1)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 1.0)
            optim.step()

            total += loss.item() * B
            count += B

        # Validation
        rssm.eval()
        with torch.no_grad():
            total_val = 0.0
            count_val = 0
            for batch in val_loader:
                if len(batch) == 7:
                    x_cont, x_sym, x_hour, x_dow, x_base, x_quote, _ = batch
                else:
                    x_cont, x_sym, x_hour, x_dow, _ = batch
                    x_base = None
                    x_quote = None
                x_cont = x_cont.to(device)
                x_sym = x_sym.to(device)
                x_hour = x_hour.to(device)
                x_dow = x_dow.to(device)
                if x_base is not None:
                    x_base = x_base.to(device)
                if x_quote is not None:
                    x_quote = x_quote.to(device)

                Z = encode_all_steps(
                    forecaster, x_cont, x_sym, x_hour, x_dow, x_base, x_quote
                )
                z_t = Z[:, :-1, :].contiguous().view(-1, Z.size(-1))
                z_tp1 = Z[:, 1:, :].contiguous().view(-1, Z.size(-1))

                B = z_t.size(0)
                h = torch.zeros(B, rssm.cfg.hidden_dim, device=device)
                z_hat, _ = rssm(z_t, h, None)
                loss = crit(z_hat, z_tp1)
                total_val += loss.item() * B
                count_val += B

            avg_train = total / max(count, 1)
            avg_val = total_val / max(count_val, 1)
            print(
                {"epoch": epoch, "rssm_train_mse": avg_train, "rssm_val_mse": avg_val}
            )
            if avg_val < best_val:
                best_val = avg_val
                out = {
                    "rssm_state": rssm.state_dict(),
                    "rssm_cfg": rssm.cfg.__dict__,
                    "latent_dim": latent_dim,
                }
                torch.save(out, Path(args.dataset_dir) / "rssm_best.pt")


if __name__ == "__main__":
    main()
