"""
Direct RSSM training on raw time series windows (no pre-trained forecaster).

This trains:
  - An Encoder that maps per-step inputs (continuous + categorical) to a latent z_t
    using a GRU sequence encoder
  - A deterministic RSSM transition that predicts z_{t+1} from z_t with a GRUCell

Losses:
  - Latent one-step consistency: MSE(z_hat_{t+1}, stopgrad(z_{t+1}))

Inputs are taken from datasets built by build_data.py:
  x_cont [B,L,C], x_symbol_id/x_ny_hour_id/x_ny_dow_id (and optional base/quote ids)

Usage example:
  python mamba-model/train_rssm_direct.py \
    --dataset_dir mamba-model/datasets/fx_1h_l512_s2 \
    --epochs 10 --batch_size 64 --latent_dim 64 --hidden_dim 256 --precision bf16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rssm import RSSM, RSSMConfig


def load_tensors(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict]:
    data: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ["train", "val", "test"]:
        data[split] = torch.load(
            dataset_dir / f"dataset_{split}.pt", map_location="cpu", weights_only=True
        )
    meta = torch.load(dataset_dir / "meta.pt", map_location="cpu", weights_only=True)
    return data, meta


def make_dataloaders(
    data: Dict[str, Dict[str, torch.Tensor]],
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split, tensors in data.items():
        x_cont = tensors["x_cont"]
        x_sym = tensors["x_symbol_id"]
        x_hour = tensors["x_ny_hour_id"]
        x_dow = tensors["x_ny_dow_id"]
        y = tensors["y"]  # unused, but kept for layout
        x_base = tensors.get("x_base_id")
        x_quote = tensors.get("x_quote_id")
        if x_base is not None and x_quote is not None:
            ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y)
        else:
            ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, y)
        shuffle = split == "train"
        drop_last = split == "train"
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            pin_memory=True,
            drop_last=drop_last,
        )
    return loaders


class SequenceEncoder(nn.Module):
    """Stepwise encoder: (cont + categorical embeddings) -> GRU -> per-step z.

    Produces Z_seq: [B, L, latent_dim] without pooling.
    """

    def __init__(
        self,
        num_cont_features: int,
        num_symbols: int,
        hour_size: int,
        dow_size: int,
        num_bases: int | None,
        num_quotes: int | None,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        latent_dim: int = 64,
        symbol_emb_dim: int = 32,
        hour_emb_dim: int = 16,
        dow_emb_dim: int = 8,
        base_emb_dim: int = 8,
        quote_emb_dim: int = 8,
    ):
        super().__init__()
        self.num_bases = num_bases
        self.num_quotes = num_quotes

        self.cont_proj = nn.Sequential(
            nn.Linear(num_cont_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
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
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim)
        )

    def forward(
        self,
        x_cont: torch.Tensor,
        x_sym: torch.Tensor,
        x_hour: torch.Tensor,
        x_dow: torch.Tensor,
        x_base: torch.Tensor | None = None,
        x_quote: torch.Tensor | None = None,
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
        z_seq = self.latent_head(y)
        return z_seq  # [B, L, latent_dim]


def parse_args():
    p = argparse.ArgumentParser(description="Direct RSSM training on raw windows")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="mamba-model/checkpoints")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument(
        "--diag_every_steps",
        type=int,
        default=500,
        help="Print latent diagnostics every N train steps",
    )
    p.add_argument(
        "--diag_max_points",
        type=int,
        default=20000,
        help="Max points for linear probe diagnostics",
    )
    p.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default="bf16" if torch.cuda.is_available() else "fp32",
    )
    return p.parse_args()


def train_one_epoch(
    encoder: SequenceEncoder,
    rssm: RSSM,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    optimizer: torch.optim.Optimizer,
    diag_every_steps: int,
    diag_max_points: int,
) -> float:
    encoder.train()
    rssm.train()
    total = 0.0
    count = 0
    start = time.time()
    for step, batch in enumerate(loader, 1):
        if len(batch) == 7:
            x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y = batch
        else:
            x_cont, x_sym, x_hour, x_dow, y = batch
            x_base = None
            x_quote = None
        x_cont = x_cont.to(device, non_blocking=True)
        x_sym = x_sym.to(device, non_blocking=True)
        x_hour = x_hour.to(device, non_blocking=True)
        x_dow = x_dow.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if x_base is not None:
            x_base = x_base.to(device, non_blocking=True)
        if x_quote is not None:
            x_quote = x_quote.to(device, non_blocking=True)

        with (
            torch.autocast("cuda", dtype=autocast_dtype)
            if autocast_dtype
            else torch.cuda.amp.autocast(enabled=False)
        ):
            Z = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)  # [B,L,Dz]
            B, L, Dz = Z.shape
            z_t = Z[:, :-1, :]
            z_tp1_true = Z[:, 1:, :]
            # roll GRUCell over time with batch hidden state
            h = torch.zeros(B, rssm.cfg.hidden_dim, device=device, dtype=Z.dtype)
            loss_mse = 0.0
            steps = 0
            for t in range(L - 1):
                z_t_step = z_t[:, t, :]
                z_pred, h = rssm(z_t_step, h, None)
                loss_mse = loss_mse + torch.mean(
                    (z_pred - z_tp1_true[:, t, :].detach()) ** 2
                )
                steps += 1
            loss = loss_mse / max(steps, 1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(rssm.parameters()), 1.0
        )
        optimizer.step()

        total += loss.item() * B
        count += B
        if step % 100 == 0:
            print(
                f"train step {step}/{len(loader)}: avg_loss={total/max(count,1):.6f} elapsed={time.time()-start:.1f}s"
            )
        # Diagnostics
        if diag_every_steps > 0 and (step % diag_every_steps == 0):
            with torch.no_grad():
                z_flat = Z.reshape(-1, Dz)
                z_std = z_flat.std(dim=0)
                zs_t = z_t.reshape(-1, Dz)
                zs_tp1 = z_tp1_true.reshape(-1, Dz)
                cos = torch.nn.functional.cosine_similarity(zs_t, zs_tp1, dim=-1).mean()
                # Linear probe: predict next target from z_t (first horizon if multi)
                y_tp1 = y[:, 1:, 0] if y.dim() == 3 else y[:, 1:]
                Y = y_tp1.reshape(-1, 1)
                F = zs_t
                if F.size(0) > diag_max_points:
                    idx = torch.randperm(F.size(0), device=F.device)[:diag_max_points]
                    F = F[idx]
                    Y = Y[idx]
                Dz_local = F.size(1)
                lam = 1e-3
                FtF = F.T @ F + lam * torch.eye(
                    Dz_local, device=F.device, dtype=F.dtype
                )
                FtY = F.T @ Y
                W = torch.linalg.solve(FtF, FtY)
                Y_hat = F @ W
                y_std = Y.std().clamp_min(1e-8)
                r2 = float(1.0 - torch.mean((Y_hat - Y) ** 2) / (y_std**2))
                corr = float(
                    torch.corrcoef(torch.stack([Y.squeeze(), Y_hat.squeeze()]))[
                        0, 1
                    ].clamp(-1, 1)
                )
                diracc = float((torch.sign(Y_hat) == torch.sign(Y)).float().mean())
                print(
                    {
                        "diag": True,
                        "z_std_min": float(z_std.min()),
                        "z_std_med": float(z_std.median()),
                        "z_std_max": float(z_std.max()),
                        "z_cos_t_tp1": float(cos),
                        "probe_r2": r2,
                        "probe_corr": corr,
                        "probe_diracc": diracc,
                    }
                )
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    encoder: SequenceEncoder,
    rssm: RSSM,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> float:
    encoder.eval()
    rssm.eval()
    total = 0.0
    count = 0
    for batch in loader:
        if len(batch) == 7:
            x_cont, x_sym, x_hour, x_dow, x_base, x_quote, _ = batch
        else:
            x_cont, x_sym, x_hour, x_dow, _ = batch
            x_base = None
            x_quote = None
        x_cont = x_cont.to(device, non_blocking=True)
        x_sym = x_sym.to(device, non_blocking=True)
        x_hour = x_hour.to(device, non_blocking=True)
        x_dow = x_dow.to(device, non_blocking=True)
        if x_base is not None:
            x_base = x_base.to(device, non_blocking=True)
        if x_quote is not None:
            x_quote = x_quote.to(device, non_blocking=True)

        with (
            torch.autocast("cuda", dtype=autocast_dtype)
            if autocast_dtype
            else torch.cuda.amp.autocast(enabled=False)
        ):
            Z = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)
            B, L, Dz = Z.shape
            z_t = Z[:, :-1, :]
            z_tp1_true = Z[:, 1:, :]
            h = torch.zeros(B, rssm.cfg.hidden_dim, device=device, dtype=Z.dtype)
            loss_mse = 0.0
            steps = 0
            for t in range(L - 1):
                z_t_step = z_t[:, t, :]
                z_pred, h = rssm(z_t_step, h, None)
                loss_mse = loss_mse + torch.mean((z_pred - z_tp1_true[:, t, :]) ** 2)
                steps += 1
            loss = loss_mse / max(steps, 1)

        total += float(loss) * B
        count += B
    return total / max(count, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data, meta = load_tensors(Path(args.dataset_dir))
    vocab = meta["vocab"]
    feature_names = meta["feature_names"]

    loaders = make_dataloaders(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
    )

    encoder = SequenceEncoder(
        num_cont_features=len(feature_names),
        num_symbols=int(vocab["num_symbols"]),
        hour_size=int(vocab["hour_size"]),
        dow_size=int(vocab["dow_size"]),
        num_bases=int(vocab.get("num_bases", 0)) or None,
        num_quotes=int(vocab.get("num_quotes", 0)) or None,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
    ).to(device)

    rssm = RSSM(RSSMConfig(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)).to(
        device
    )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(rssm.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    autocast_dtype = None
    if device.type == "cuda":
        if args.precision == "bf16" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
            print("Using bf16 autocast")
        elif args.precision == "fp16":
            autocast_dtype = torch.float16
            print("Using fp16 autocast")

    best_val = float("inf")
    best_path = save_dir / "rssm_direct_best.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t0 = time.time()
        train_loss = train_one_epoch(
            encoder, rssm, loaders["train"], device, autocast_dtype, optimizer
        )
        val_loss = evaluate(encoder, rssm, loaders["val"], device, autocast_dtype)
        scheduler.step()
        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - t0,
        }
        print(json.dumps(log))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "encoder_state": encoder.state_dict(),
                    "rssm_state": rssm.state_dict(),
                    "cfg": vars(args),
                    "val_loss": best_val,
                },
                best_path,
            )
            print(f"Saved improved checkpoint: {best_path} (val_loss={best_val:.6f})")

    # Final eval on test
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])  # type: ignore
        rssm.load_state_dict(ckpt["rssm_state"])  # type: ignore
    test_loss = evaluate(encoder, rssm, loaders["test"], device, autocast_dtype)
    print(json.dumps({"test_loss": test_loss}))


if __name__ == "__main__":
    main()
