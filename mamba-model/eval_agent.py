"""
Evaluate a trained Dreamer policy on real data (val/test) with costs-aware PnL.

- Loads frozen world model checkpoint and Dreamer policy checkpoint
- Builds policy state from posterior (h,(z)) and horizon z-scores
- Rolls one-step decisions sample-by-sample (no peeking), computes:
  cumulative PnL, mean/std PnL, Sharpe, turnover, hit ratio, and prints summary
  Optionally writes per-sample CSV with symbol, ts_utc, action, pnl parts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from world_model import (
    ObsEncoder,
    ObsDecoder,
    ReturnHead,
    StudentTHead,
    predict_mu_sigma,
)
from rssm import RSSM, RSSMConfig
from policy import Actor


def load_tensors(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict]:
    data: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ["train", "val", "test"]:
        data[split] = torch.load(
            dataset_dir / f"dataset_{split}.pt", map_location="cpu"
        )
    meta = torch.load(dataset_dir / "meta.pt", map_location="cpu")
    return data, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Dreamer policy")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--wm_checkpoint", type=str, required=True)
    p.add_argument("--policy_checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    p.add_argument(
        "--h_eval", type=int, default=6, help="Horizon (hours) to evaluate PnL over"
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--likelihood", type=str, choices=["gaussian", "studentt"], default="studentt"
    )
    p.add_argument("--txn_cost_bp", type=float, default=0.5)
    p.add_argument(
        "--action_space",
        type=str,
        choices=["discrete", "continuous"],
        default="discrete",
    )
    p.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Optional path to write per-sample actions and PnL",
    )
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, meta = load_tensors(Path(args.dataset_dir))
    vocab = meta["vocab"]
    feature_names = meta["feature_names"]
    horizons = list(meta["horizons"])  # type: ignore
    num_horizons = len(horizons)
    # map horizon to index
    h_idx = horizons.index(args.h_eval) if args.h_eval in horizons else 0

    # Loader
    tensors = data[args.split]
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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Load checkpoints
    wm_ckpt = torch.load(Path(args.wm_checkpoint), map_location=device)
    pol_ckpt = torch.load(Path(args.policy_checkpoint), map_location=device)
    pol_cfg = pol_ckpt.get("config", {})

    latent_dim = int(wm_ckpt.get("cfg", {}).get("latent_dim", 64))
    hidden_dim = int(wm_ckpt.get("cfg", {}).get("hidden_dim", 256))

    # Build WM
    encoder = ObsEncoder(
        num_cont_features=len(feature_names),
        num_symbols=int(vocab["num_symbols"]),
        hour_size=int(vocab["hour_size"]),
        dow_size=int(vocab["dow_size"]),
        num_bases=int(vocab.get("num_bases", 0)) or None,
        num_quotes=int(vocab.get("num_quotes", 0)) or None,
        embed_dim=latent_dim,
        d_model=hidden_dim,
        n_layers=2,
        dropout=0.1,
    ).to(device)
    rssm = RSSM(
        RSSMConfig(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            stochastic=(not args.deterministic),
        )
    ).to(device)
    decoder = ObsDecoder(
        latent_dim=latent_dim, num_cont_features=len(feature_names)
    ).to(device)
    if args.likelihood == "gaussian":
        ret_head = ReturnHead(latent_dim=latent_dim, num_horizons=num_horizons).to(
            device
        )
    else:
        ret_head = StudentTHead(latent_dim=latent_dim, num_horizons=num_horizons).to(
            device
        )

    encoder.load_state_dict(wm_ckpt["encoder_state"])  # type: ignore
    rssm.load_state_dict(wm_ckpt["rssm_state"], strict=False)  # type: ignore
    decoder.load_state_dict(wm_ckpt["decoder_state"])  # type: ignore
    ret_head.load_state_dict(wm_ckpt["ret_head_state"])  # type: ignore
    encoder.eval()
    rssm.eval()
    decoder.eval()
    ret_head.eval()

    # Actor
    wm_state_dim = hidden_dim if args.deterministic else hidden_dim + latent_dim
    state_dim = wm_state_dim + 3 + 1
    action_space = pol_cfg.get("action_space", args.action_space)
    actor = Actor(state_dim, action_space=action_space).to(device)
    actor.load_state_dict(pol_ckpt["actor_state"])  # type: ignore
    actor.eval()

    # Prepare per-symbol prev action for costs
    sym_to_prev_action: Dict[int, float] = {}

    # If targets standardized, optionally de-standardize
    target_scalers = meta.get("target_scalers")
    symbol_to_id = vocab.get("symbol_to_id", {})
    id_to_symbol = vocab.get("id_to_symbol", None)
    # Build inverse map id->symbol string if not present
    if not id_to_symbol:
        id_to_symbol = {v: k for k, v in symbol_to_id.items()}

    # Accumulators
    rows = []
    pnl_series = []
    turns = 0
    total_steps = 0
    txn_cost = args.txn_cost_bp * 1e-4

    for xb in loader:
        if len(xb) == 7:
            xc, xs, xh, xd, xb_id, xq_id, yb = [t.to(device) for t in xb]
        else:
            xc, xs, xh, xd, yb = [t.to(device) for t in xb]
            xb_id = None
            xq_id = None
        E = encoder(xc, xs, xh, xd, xb_id, xq_id)
        B, L, _ = E.shape
        h = torch.zeros(B, rssm.cfg.hidden_dim, device=device, dtype=E.dtype)
        z_t = None
        if args.deterministic:
            for t in range(L):
                z_t = E[:, t, :]
                h = rssm.core(z_t, h, None)
            wm_state = h
        else:
            for t in range(L):
                mu_p, logv_p = rssm.prior(h)
                mu_q, logv_q = rssm.posterior(h, E[:, t, :])
                z_t = mu_q
                h = rssm.core(z_t, h, None)
            wm_state = torch.cat([h, z_t], dim=-1)  # type: ignore
        assert z_t is not None

        mu, sigma = predict_mu_sigma(ret_head, z_t, args.likelihood)
        # Build state for actor per sample using previous action per symbol
        # Compute decision and realized PnL per sample
        for b in range(B):
            sym_id = int(xs[b, 0].detach().cpu())
            prev_a = sym_to_prev_action.get(sym_id, 0.0)
            prev_a_tensor = torch.tensor([[prev_a]], device=device, dtype=E.dtype)
            s = torch.cat(
                [
                    wm_state[b : b + 1, :],
                    (
                        mu[b, h_idx : h_idx + 1] / (sigma[b, h_idx : h_idx + 1] + 1e-6)
                    ).unsqueeze(0),
                    (
                        mu[
                            b,
                            horizons.index(24) if 24 in horizons else h_idx : (
                                horizons.index(24) + 1 if 24 in horizons else h_idx + 1
                            ),
                        ]
                        / (
                            sigma[
                                b,
                                horizons.index(24) if 24 in horizons else h_idx : (
                                    horizons.index(24) + 1
                                    if 24 in horizons
                                    else h_idx + 1
                                ),
                            ]
                            + 1e-6
                        )
                    ).unsqueeze(0),
                    (
                        mu[
                            b,
                            horizons.index(168) if 168 in horizons else h_idx : (
                                horizons.index(168) + 1
                                if 168 in horizons
                                else h_idx + 1
                            ),
                        ]
                        / (
                            sigma[
                                b,
                                horizons.index(168) if 168 in horizons else h_idx : (
                                    horizons.index(168) + 1
                                    if 168 in horizons
                                    else h_idx + 1
                                ),
                            ]
                            + 1e-6
                        )
                    ).unsqueeze(0),
                    prev_a_tensor,
                ],
                dim=-1,
            )
            logits_or_size = actor(s)
            if action_space == "discrete":
                a_idx = int(torch.argmax(logits_or_size, dim=-1).item())
                a = float(np.tanh(a_idx - 1.0))  # map {0,1,2} -> ~{-1,0,1}
            else:
                a = float(torch.tanh(logits_or_size).item())

            # Realized return
            y_std = float(yb[b, h_idx].detach().cpu().item())
            if target_scalers:
                sym = id_to_symbol.get(sym_id, None)
                col = f"fwd_ret_log_{horizons[h_idx]}h"
                if (
                    sym is not None
                    and sym in target_scalers
                    and col in target_scalers[sym]
                ):
                    mean, std = target_scalers[sym][col]
                    y_real = y_std * std + mean
                else:
                    y_real = y_std
            else:
                y_real = y_std

            # Costs and PnL (per sample)
            cost = abs(a - prev_a) * txn_cost
            pnl = a * y_real - cost
            pnl_series.append(pnl)
            total_steps += 1
            if abs(a - prev_a) > 1e-6:
                turns += 1
            sym_to_prev_action[sym_id] = a

            if args.save_csv:
                # Optional rich row
                samples = meta.get("samples", {}).get(args.split, [])
                j = len(rows)
                meta_j = samples[j] if j < len(samples) else {}
                rows.append(
                    {
                        "idx": j,
                        "symbol": id_to_symbol.get(sym_id, sym_id),
                        "ts_utc": meta_j.get("ts_utc")
                        or meta_j.get("t_end")
                        or meta_j.get("timestamp"),
                        "a": a,
                        "prev_a": prev_a,
                        "pnl": pnl,
                        "y_real": y_real,
                        "cost": cost,
                    }
                )

    # Metrics
    pnl_np = np.array(pnl_series, dtype=np.float64)
    mean_pnl = float(np.mean(pnl_np)) if pnl_np.size else 0.0
    std_pnl = float(np.std(pnl_np) + 1e-12) if pnl_np.size else 0.0
    sharpe = mean_pnl / (std_pnl if std_pnl > 0 else 1e-12)
    turnover = turns / max(total_steps, 1)
    print(
        {
            "split": args.split,
            "h_eval": args.h_eval,
            "steps": int(total_steps),
            "turnover": turnover,
            "avg_pnl": mean_pnl,
            "std_pnl": std_pnl,
            "sharpe": sharpe,
            "cum_pnl": float(np.sum(pnl_np)) if pnl_np.size else 0.0,
        }
    )

    if args.save_csv and rows:
        import pandas as pd

        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        print(f"Saved per-sample policy outputs to {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
