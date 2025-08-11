"""
Train Dreamer-style world model on FX hourly windows (2018–2025, 7 majors).

Objective per sequence:
  - Reconstruction ELBO: E_q[log p(x_t|z_t)] - beta * KL(q||p) with free-nats
  - Return head NLL on z_last for multiple horizons (heteroscedastic)

Data input: tensors built via build_data.py
  x_cont [B,L,C], x_symbol_id/x_ny_hour_id/x_ny_dow_id/(x_base_id,x_quote_id optional), y [B,H]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rssm import RSSM, RSSMConfig
from world_model import (
    WorldModelConfig,
    ObsEncoder,
    ObsDecoder,
    ReturnHead,
    gaussian_nll,
    apply_free_nats,
)


def load_tensors(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict]:
    data: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ["train", "val", "test"]:
        data[split] = torch.load(
            dataset_dir / f"dataset_{split}.pt", map_location="cpu"
        )
    meta = torch.load(dataset_dir / "meta.pt", map_location="cpu")
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
        y = tensors["y"]
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


def parse_args():
    p = argparse.ArgumentParser(description="Train Dreamer-style world model")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="mamba-model/checkpoints")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kl_beta", type=float, default=1.0)
    p.add_argument("--kl_free_nats", type=float, default=2.0)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--ret_weight", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default="bf16" if torch.cuda.is_available() else "fp32",
    )
    p.add_argument("--diag_every_steps", type=int, default=500)
    return p.parse_args()


def train_one_epoch(
    encoder: ObsEncoder,
    rssm: RSSM,
    decoder: ObsDecoder,
    head: ReturnHead,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    optimizer: torch.optim.Optimizer,
    kl_beta: float,
    kl_free_nats: float,
    recon_weight: float,
    ret_weight: float,
    diag_every_steps: int,
) -> float:
    encoder.train()
    rssm.train()
    decoder.train()
    head.train()
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
            E = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)  # [B,L,De]
            B, L, _ = E.shape
            Hdim = rssm.cfg.hidden_dim
            h = torch.zeros(B, Hdim, device=device, dtype=E.dtype)
            recon_loss = torch.tensor(0.0, device=device, dtype=E.dtype)
            kl_accum = torch.tensor(0.0, device=device, dtype=E.dtype)

            for t in range(L):
                mu_p, logv_p = rssm.prior(h)
                mu_q, logv_q = rssm.posterior(h, E[:, t, :])
                z_t = rssm._rsample(mu_q, logv_q)
                h = rssm.core(z_t, h, None)
                mu_x, logv_x = decoder(z_t)
                recon_loss = recon_loss + gaussian_nll(x_cont[:, t, :], mu_x, logv_x)
                kl_t = rssm.kl_gaussian(mu_q, logv_q, mu_p, logv_p)
                kl_accum = kl_accum + apply_free_nats(kl_t, kl_free_nats)

            recon_loss = recon_loss / max(L, 1)
            kl_loss = kl_accum / max(L, 1)
            # Return head on last latent h state: need last z_t; reuse mu_q of last step's sample
            # For numerical stability, feed last latent derived from last posterior sample
            mu_y, logv_y = head(z_t)
            y_loss = gaussian_nll(y, mu_y, logv_y)

            loss = recon_weight * recon_loss + kl_beta * kl_loss + ret_weight * y_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(encoder.parameters())
            + list(rssm.parameters())
            + list(decoder.parameters())
            + list(head.parameters()),
            1.0,
        )
        optimizer.step()

        total += float(loss) * B
        count += B
        if step % 100 == 0:
            print(
                f"train step {step}/{len(loader)} avg_loss={total/max(count,1):.6f} elapsed={time.time()-start:.1f}s"
            )
        if diag_every_steps > 0 and (step % diag_every_steps == 0):
            with torch.no_grad():
                # Simple z statistic: run encoder again last batch
                z_std = z_t.std(dim=-1).mean().item()
                print(
                    {
                        "diag": True,
                        "z_std_mean": z_std,
                        "recon_loss": float(recon_loss),
                        "kl": float(kl_loss),
                        "y_nll": float(y_loss),
                    }
                )

    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    encoder: ObsEncoder,
    rssm: RSSM,
    decoder: ObsDecoder,
    head: ReturnHead,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
) -> float:
    encoder.eval()
    rssm.eval()
    decoder.eval()
    head.eval()
    total = 0.0
    count = 0
    for batch in loader:
        if len(batch) == 7:
            x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y = batch
        else:
            x_cont, x_sym, x_hour, x_dow, y = batch
            x_base = None
            x_quote = None
        x_cont = x_cont.to(device)
        x_sym = x_sym.to(device)
        x_hour = x_hour.to(device)
        x_dow = x_dow.to(device)
        y = y.to(device)
        if x_base is not None:
            x_base = x_base.to(device)
        if x_quote is not None:
            x_quote = x_quote.to(device)

        with (
            torch.autocast("cuda", dtype=autocast_dtype)
            if autocast_dtype
            else torch.cuda.amp.autocast(enabled=False)
        ):
            E = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)
            B, L, _ = E.shape
            h = torch.zeros(B, rssm.cfg.hidden_dim, device=device, dtype=E.dtype)
            recon_loss = torch.tensor(0.0, device=device, dtype=E.dtype)
            kl_accum = torch.tensor(0.0, device=device, dtype=E.dtype)
            for t in range(L):
                mu_p, logv_p = rssm.prior(h)
                mu_q, logv_q = rssm.posterior(h, E[:, t, :])
                z_t = mu_q  # use mean at eval
                h = rssm.core(z_t, h, None)
                mu_x, logv_x = decoder(z_t)
                recon_loss = recon_loss + gaussian_nll(x_cont[:, t, :], mu_x, logv_x)
                kl_t = rssm.kl_gaussian(mu_q, logv_q, mu_p, logv_p)
                kl_accum = kl_accum + apply_free_nats(kl_t, 2.0)

            recon_loss = recon_loss / max(L, 1)
            kl_loss = kl_accum / max(L, 1)
            mu_y, logv_y = head(z_t)
            y_loss = gaussian_nll(y, mu_y, logv_y)
            loss = recon_loss + kl_loss + 0.3 * y_loss

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

    cfg = WorldModelConfig(
        num_cont_features=len(feature_names),
        num_symbols=int(vocab["num_symbols"]),
        hour_size=int(vocab["hour_size"]),
        dow_size=int(vocab["dow_size"]),
        num_bases=int(vocab.get("num_bases", 0)) or None,
        num_quotes=int(vocab.get("num_quotes", 0)) or None,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        recon_weight=args.recon_weight,
        kl_beta=args.kl_beta,
        kl_free_nats=args.kl_free_nats,
        ret_weight=args.ret_weight,
    )

    encoder = ObsEncoder(
        num_cont_features=cfg.num_cont_features,
        num_symbols=cfg.num_symbols,
        hour_size=cfg.hour_size,
        dow_size=cfg.dow_size,
        num_bases=cfg.num_bases,
        num_quotes=cfg.num_quotes,
        embed_dim=cfg.latent_dim,
        d_model=args.hidden_dim,
        n_layers=2,
        dropout=cfg.dropout,
    ).to(device)
    rssm = RSSM(
        RSSMConfig(
            latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim, stochastic=True
        )
    ).to(device)
    decoder = ObsDecoder(
        latent_dim=cfg.latent_dim, num_cont_features=cfg.num_cont_features
    ).to(device)
    num_horizons = int(len(meta["horizons"]))
    head = ReturnHead(latent_dim=cfg.latent_dim, num_horizons=num_horizons).to(device)

    params = (
        list(encoder.parameters())
        + list(rssm.parameters())
        + list(decoder.parameters())
        + list(head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    autocast_dtype: Optional[torch.dtype] = None
    if device.type == "cuda":
        if args.precision == "bf16" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
            print("Using bf16 autocast")
        elif args.precision == "fp16":
            autocast_dtype = torch.float16
            print("Using fp16 autocast")

    best_val = float("inf")
    best_path = save_dir / "world_model_best.pt"
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t0 = time.time()
        train_loss = train_one_epoch(
            encoder,
            rssm,
            decoder,
            head,
            loaders["train"],
            device,
            autocast_dtype,
            optimizer,
            args.kl_beta,
            args.kl_free_nats,
            args.recon_weight,
            args.ret_weight,
            args.diag_every_steps,
        )
        val_loss = evaluate(
            encoder, rssm, decoder, head, loaders["val"], device, autocast_dtype
        )
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
                    "decoder_state": decoder.state_dict(),
                    "ret_head_state": head.state_dict(),
                    "cfg": cfg.__dict__,
                    "val_loss": best_val,
                },
                best_path,
            )
            print(f"Saved improved checkpoint: {best_path} (val_loss={best_val:.6f})")

    # Final test
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])  # type: ignore
        rssm.load_state_dict(ckpt["rssm_state"])  # type: ignore
        decoder.load_state_dict(ckpt["decoder_state"])  # type: ignore
        head.load_state_dict(ckpt["ret_head_state"])  # type: ignore
    test_loss = evaluate(
        encoder, rssm, decoder, head, loaders["test"], device, autocast_dtype
    )
    print(json.dumps({"test_loss": test_loss}))


if __name__ == "__main__":
    main()
