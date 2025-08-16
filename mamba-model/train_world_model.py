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
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rssm import RSSM, RSSMConfig
from world_model import (
    WorldModelConfig,
    ObsEncoder,
    ObsDecoder,
    ReturnHead,
    StudentTHead,
    SignHead,
    gaussian_nll,
    student_t_nll,
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
    p.add_argument(
        "--likelihood",
        type=str,
        choices=["gaussian", "studentt"],
        default="gaussian",
        help="Observation model for return head",
    )
    p.add_argument(
        "--sign_weight",
        type=float,
        default=0.0,
        help="Weight for sign (direction) BCE loss per horizon",
    )
    p.add_argument("--kl_beta", type=float, default=1.0)
    p.add_argument("--kl_beta_start", type=float, default=0.1)
    p.add_argument("--kl_warmup_epochs", type=int, default=5)
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
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic dynamics (no KL, no sampling; z_t = encoder output)",
    )
    # Validation metrics printing
    p.add_argument(
        "--val_metrics_every_epochs",
        type=int,
        default=1,
        help="Print detailed validation metrics every N epochs (0 to disable)",
    )
    p.add_argument(
        "--val_metrics_max_points",
        type=int,
        default=8000,
        help="Max number of validation samples to use for quick metrics",
    )
    p.add_argument(
        "--val_use_zscore",
        action="store_true",
        help="Use |mu/sigma| for confidence gating metrics",
    )
    p.add_argument(
        "--val_txn_cost_bp",
        type=float,
        default=0.5,
        help="Txn costs (basis points) used for quick validation PnL",
    )
    return p.parse_args()


def train_one_epoch(
    encoder: ObsEncoder,
    rssm: RSSM,
    decoder: ObsDecoder,
    ret_head,
    sign_head: SignHead,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    optimizer: torch.optim.Optimizer,
    kl_beta: float,
    kl_free_nats: float,
    recon_weight: float,
    ret_weight: float,
    diag_every_steps: int,
    deterministic: bool,
    likelihood: str,
    sign_weight: float,
) -> float:
    encoder.train()
    rssm.train()
    decoder.train()
    ret_head.train()
    sign_head.train()
    total = 0.0
    count = 0
    recon_total = 0.0
    kl_total = 0.0
    y_total = 0.0
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

            if deterministic:
                for t in range(L):
                    z_t = E[:, t, :]
                    h = rssm.core(z_t, h, None)
                kl_loss = torch.tensor(0.0, device=device, dtype=E.dtype)
            else:
                for t in range(L):
                    mu_p, logv_p = rssm.prior(h)
                    mu_q, logv_q = rssm.posterior(h, E[:, t, :])
                    z_t = rssm._rsample(mu_q, logv_q)
                    h = rssm.core(z_t, h, None)
                    if recon_weight > 0.0:
                        mu_x, logv_x = decoder(z_t)
                        recon_loss = recon_loss + gaussian_nll(
                            x_cont[:, t, :], mu_x, logv_x
                        )
                    kl_t = rssm.kl_gaussian(mu_q, logv_q, mu_p, logv_p)
                    kl_accum = kl_accum + apply_free_nats(kl_t, kl_free_nats)
                kl_loss = kl_accum / max(L, 1)

            recon_loss = recon_loss / max(L, 1)
            # Return head on last latent
            if likelihood == "gaussian":
                mu_y, logv_y = ret_head(z_t)
                logv_y = logv_y.clamp(min=-6.0, max=2.0)
                y_loss = gaussian_nll(y, mu_y, logv_y) + 1e-6 * (logv_y**2).mean()
            else:
                mu_y, log_s_y, nu_y = ret_head(z_t)
                y_loss = student_t_nll(y, mu_y, log_s_y, nu_y)
            # Sign loss per horizon
            logits = sign_head(z_t)
            sign_targets = (y > 0).float()
            sign_loss = F.binary_cross_entropy_with_logits(logits, sign_targets)

            loss = (
                recon_weight * recon_loss
                + (0.0 if deterministic else kl_beta * kl_loss)
                + ret_weight * y_loss
                + sign_weight * sign_loss
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(encoder.parameters())
            + list(rssm.parameters())
            + list(decoder.parameters())
            + list(ret_head.parameters())
            + list(sign_head.parameters()),
            1.0,
        )
        optimizer.step()

        # Running aggregates (weighted by batch size)
        total += float(loss) * B
        recon_total += float(recon_loss) * B
        kl_total += float(kl_loss) * B
        y_total += float(y_loss) * B
        count += B
        if step % 100 == 0:
            avg_total = total / max(count, 1)
            avg_recon = recon_total / max(count, 1)
            avg_kl = kl_total / max(count, 1)
            avg_y = y_total / max(count, 1)
            # Fun, compact status line
            print(
                f"[🚀 Train] step {step:>4}/{len(loader):<4} | 🎯 total {avg_total:8.5f} "
                f"= 🧩 recon {recon_weight}×{avg_recon:7.5f} + 🔗 β{kl_beta}×KL {avg_kl:7.5f} + 📈 λ{ret_weight}×NLL {avg_y:7.5f} | ⏱ {time.time()-start:.1f}s"
            )
            # Quick edge proxies on current batch (first horizon)
            with torch.no_grad():
                mu_h = mu_y[:, 0]
                y_h = y[:, 0]
                if likelihood == "gaussian":
                    sig_h = torch.exp(0.5 * logv_y[:, 0]).clamp_min(1e-8)
                else:
                    # Student-t: use scale s = softplus(log_s)
                    _, log_s_temp, _ = ret_head(z_t)
                    sig_h = F.softplus(log_s_temp[:, 0]).clamp_min(1e-8)
                # Ensure float32 for stats (quantile doesn't support bf16)
                mu_h_f = mu_h.float()
                y_h_f = y_h.float()
                sig_h_f = sig_h.float()
                z = mu_h_f / sig_h_f
                # Pearson IC
                if mu_h_f.std() > 0 and y_h_f.std() > 0:
                    ic = float(
                        torch.corrcoef(torch.stack([mu_h_f, y_h_f]))[0, 1].clamp(-1, 1)
                    )
                else:
                    ic = 0.0
                # Spearman via ranks
                r_mu = torch.argsort(torch.argsort(mu_h_f))
                r_y = torch.argsort(torch.argsort(y_h_f))
                if r_mu.float().std() > 0 and r_y.float().std() > 0:
                    ic_s = float(
                        torch.corrcoef(torch.stack([r_mu.float(), r_y.float()]))[0, 1]
                    )
                else:
                    ic_s = 0.0
                diracc = float((torch.sign(mu_h_f) == torch.sign(y_h_f)).float().mean())
                edge = float((torch.sign(mu_h_f) * y_h_f).mean())
                # Top-20% by |z|
                thr = torch.quantile(z.abs().float(), 0.8)
                mask = z.abs() >= thr
                if mask.any():
                    diracc_top = float(
                        (torch.sign(mu_h_f[mask]) == torch.sign(y_h_f[mask]))
                        .float()
                        .mean()
                    )
                    edge_top = float((torch.sign(mu_h_f[mask]) * y_h_f[mask]).mean())
                else:
                    diracc_top = float("nan")
                    edge_top = float("nan")
                z_mean = float(z.abs().mean())
            print(
                {
                    "ic": ic,
                    "ic_s": ic_s,
                    "diracc": diracc,
                    "diracc@20": diracc_top,
                    "edge": edge,
                    "edge@20": edge_top,
                    "z_mean": z_mean,
                }
            )
        if diag_every_steps > 0 and (step % diag_every_steps == 0):
            with torch.no_grad():
                # Simple z statistic: run encoder again last batch
                z_std = z_t.std(dim=-1).mean().item()
                print(
                    f"   ↳ [🔎 Diag] z_std_mean={z_std:.4f} | recon={float(recon_loss):.5f} KL={float(kl_loss):.5f} yNLL={float(y_loss):.5f}"
                )

    # End-of-epoch summary
    avg_total = total / max(count, 1)
    avg_recon = recon_total / max(count, 1)
    avg_kl = kl_total / max(count, 1)
    avg_y = y_total / max(count, 1)
    print(
        "\n"
        + f"🏁 [Train Epoch] 🎯 total {avg_total:.6f} | 🧩 recon {recon_weight}×{avg_recon:.6f} | "
        f"🔗 KL β={kl_beta}, free_nats={kl_free_nats} → {avg_kl:.6f} | 📈 λ={ret_weight}×yNLL {avg_y:.6f}\n"
    )
    return avg_total


@torch.no_grad()
def evaluate(
    encoder: ObsEncoder,
    rssm: RSSM,
    decoder: ObsDecoder,
    ret_head: ReturnHead,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    kl_free_nats: float,
    kl_beta: float,
    ret_weight: float,
    recon_weight: float = 1.0,
    deterministic: bool = False,
    likelihood: str = "gaussian",
) -> float:
    encoder.eval()
    rssm.eval()
    decoder.eval()
    ret_head.eval()
    total = 0.0
    count = 0
    recon_total = 0.0
    kl_total = 0.0
    y_total = 0.0
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
            if deterministic:
                for t in range(L):
                    z_t = E[:, t, :]
                    h = rssm.core(z_t, h, None)
                kl_loss = torch.tensor(0.0, device=device, dtype=E.dtype)
            else:
                for t in range(L):
                    mu_p, logv_p = rssm.prior(h)
                    mu_q, logv_q = rssm.posterior(h, E[:, t, :])
                    z_t = mu_q  # use mean at eval
                    h = rssm.core(z_t, h, None)
                    if recon_weight > 0.0:
                        mu_x, logv_x = decoder(z_t)
                        recon_loss = recon_loss + gaussian_nll(
                            x_cont[:, t, :], mu_x, logv_x
                        )
                    kl_t = rssm.kl_gaussian(mu_q, logv_q, mu_p, logv_p)
                    kl_accum = kl_accum + apply_free_nats(kl_t, kl_free_nats)
                kl_loss = kl_accum / max(L, 1)

            recon_loss = recon_loss / max(L, 1)
            mu_y, logv_y = ret_head(z_t)
            y_loss = gaussian_nll(y, mu_y, logv_y)
            loss = (
                recon_weight * recon_loss
                + ((0.0 if deterministic else kl_beta * kl_loss))
                + ret_weight * y_loss
            )

        total += float(loss) * B
        recon_total += float(recon_loss) * B
        kl_total += float(kl_loss) * B
        y_total += float(y_loss) * B
        count += B

    avg_total = total / max(count, 1)
    avg_recon = recon_total / max(count, 1)
    avg_kl = kl_total / max(count, 1)
    avg_y = y_total / max(count, 1)
    print(
        f"🧪 [Val] 🎯 total {avg_total:.6f} | 🧩 recon {recon_weight}×{avg_recon:.6f} | "
        f"🔗 KL β={kl_beta}, free_nats={kl_free_nats} → {avg_kl:.6f} | 📈 λ={ret_weight}×yNLL {avg_y:.6f}"
    )
    return avg_total


@torch.no_grad()
def quick_val_metrics(
    encoder: ObsEncoder,
    rssm: RSSM,
    ret_head: ReturnHead,
    loader: DataLoader,
    device: torch.device,
    max_points: int = 8000,
    use_zscore: bool = True,
    txn_cost_bp: float = 0.5,
) -> None:
    """Lightweight validation metrics for immediate feedback.

    Prints per-horizon: corr, diracc, diracc@20, RMSE, and a simple costs-aware Sharpe using sign(mu).
    Uses at most `max_points` samples for speed.
    """
    MU_list: list[torch.Tensor] = []
    SIG_list: list[torch.Tensor] = []
    Y_list: list[torch.Tensor] = []
    N_total = 0
    for xb in loader:
        # Support optional base/quote
        if len(xb) == 7:
            x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y = xb
        else:
            x_cont, x_sym, x_hour, x_dow, y = xb
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
        y = y.to(device)

        E = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)
        B, L, _ = E.shape
        h = torch.zeros(B, rssm.cfg.hidden_dim, device=device, dtype=E.dtype)
        z_t = None
        for t in range(L):
            # Use deterministic path if model is deterministic
            if hasattr(rssm.cfg, "stochastic") and not rssm.cfg.stochastic:
                z_t = E[:, t, :]
                h = rssm.core(z_t, h, None)
            else:
                mu_p, logv_p = rssm.prior(h)
                mu_q, logv_q = rssm.posterior(h, E[:, t, :])
                z_t = mu_q
                h = rssm.core(z_t, h, None)
        assert z_t is not None
        mu_y, logv_y = ret_head(z_t)
        MU_list.append(mu_y.detach().cpu())
        SIG_list.append(torch.exp(0.5 * logv_y).detach().cpu())
        Y_list.append(y.detach().cpu())
        N_total += y.size(0)
        if N_total >= max_points:
            break

    if not MU_list:
        print('{"val_metrics": "no samples"}')
        return

    MU = torch.cat(MU_list, dim=0).numpy()
    SIG = torch.cat(SIG_list, dim=0).numpy()
    Y = torch.cat(Y_list, dim=0).numpy()

    num_horizons = MU.shape[1]
    cost = txn_cost_bp * 1e-4
    for h_idx in range(num_horizons):
        mu = MU[:, h_idx]
        sig = SIG[:, h_idx]
        yt = Y[:, h_idx]
        rmse = float(np.sqrt(np.mean((mu - yt) ** 2)))
        corr = (
            float(np.corrcoef(mu, yt)[0, 1]) if mu.std() > 0 and yt.std() > 0 else 0.0
        )
        diracc = float(np.mean(np.sign(mu) == np.sign(yt)))
        score = np.abs(mu / np.maximum(sig, 1e-6)) if use_zscore else np.abs(mu)
        thr = np.quantile(score, 0.8)
        mask = score >= thr
        diracc_top = (
            float(np.mean(np.sign(mu[mask]) == np.sign(yt[mask])))
            if mask.any()
            else float("nan")
        )
        # Simple costs-aware Sharpe for baseline sign(mu)
        pos = np.sign(mu)
        pnl = pos * yt - np.abs(np.diff(pos, prepend=0)) * cost
        shrp = float(pnl.mean() / (pnl.std() + 1e-9))
        print(
            {
                "val_quick": True,
                "h_idx": h_idx,
                "rmse": rmse,
                "corr": corr,
                "diracc": diracc,
                "diracc@20": diracc_top,
                "baseline_sign_mu_sharpe": shrp,
            }
        )


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
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            stochastic=not args.deterministic,
        )
    ).to(device)
    decoder = ObsDecoder(
        latent_dim=cfg.latent_dim, num_cont_features=cfg.num_cont_features
    ).to(device)
    num_horizons = int(len(meta["horizons"]))
    if args.likelihood == "gaussian":
        ret_head = ReturnHead(latent_dim=cfg.latent_dim, num_horizons=num_horizons).to(
            device
        )
    else:
        ret_head = StudentTHead(
            latent_dim=cfg.latent_dim, num_horizons=num_horizons
        ).to(device)
    sign_head = SignHead(latent_dim=cfg.latent_dim, num_horizons=num_horizons).to(
        device
    )

    params = (
        list(encoder.parameters())
        + list(rssm.parameters())
        + list(decoder.parameters())
        + list(ret_head.parameters())
        + list(sign_head.parameters())
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
        # KL warm-up schedule
        beta_start = float(args.kl_beta_start)
        warm_epochs = max(1, int(args.kl_warmup_epochs))
        progress = min((epoch - 1) / float(warm_epochs), 1.0)
        kl_beta_epoch = beta_start + (float(args.kl_beta) - beta_start) * progress
        train_loss = train_one_epoch(
            encoder,
            rssm,
            decoder,
            ret_head,
            sign_head,
            loaders["train"],
            device,
            autocast_dtype,
            optimizer,
            kl_beta_epoch,
            args.kl_free_nats,
            args.recon_weight,
            args.ret_weight,
            args.diag_every_steps,
            args.deterministic,
            args.likelihood,
            float(args.sign_weight),
        )
        val_loss = evaluate(
            encoder,
            rssm,
            decoder,
            ret_head,
            loaders["val"],
            device,
            autocast_dtype,
            args.kl_free_nats,
            kl_beta_epoch,
            args.ret_weight,
            args.recon_weight,
            args.deterministic,
            args.likelihood,
        )
        if args.val_metrics_every_epochs and (
            epoch % args.val_metrics_every_epochs == 0
        ):
            quick_val_metrics(
                encoder,
                rssm,
                ret_head,
                loaders["val"],
                device,
                max_points=args.val_metrics_max_points,
                use_zscore=args.val_use_zscore,
                txn_cost_bp=args.val_txn_cost_bp,
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
                    "ret_head_state": ret_head.state_dict(),
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
        ret_head.load_state_dict(ckpt["ret_head_state"])  # type: ignore
    test_loss = evaluate(
        encoder,
        rssm,
        decoder,
        ret_head,
        loaders["test"],
        device,
        autocast_dtype,
        args.kl_free_nats,
        float(args.kl_beta),
        args.ret_weight,
        args.recon_weight,
        args.deterministic,
        args.likelihood,
    )
    print(json.dumps({"test_loss": test_loss}))


if __name__ == "__main__":
    main()
