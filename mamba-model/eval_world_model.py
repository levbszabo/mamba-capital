"""
Evaluate Dreamer-style world model on held-out splits with finance metrics.

Reports per-horizon:
- MSE/MAE/RMSE, correlation, DirAcc
- Threshold buckets by |mu| or |mu/sigma|
- Simple PnL with txn costs using sign(mu)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rssm import RSSM, RSSMConfig
from world_model import ObsEncoder, ObsDecoder, ReturnHead


def load_tensors(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict]:
    data: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ["train", "val", "test"]:
        data[split] = torch.load(
            dataset_dir / f"dataset_{split}.pt", map_location="cpu"
        )
    meta = torch.load(dataset_dir / "meta.pt", map_location="cpu")
    return data, meta


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-12) -> float:
    return float(returns.mean() / (returns.std() + eps))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    return float(dd.min())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Evaluate Dreamer-style world model")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--use_zscore", action="store_true", help="Use |mu/sigma| for thresholding"
    )
    p.add_argument("--txn_cost_bp", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, meta = load_tensors(Path(args.dataset_dir))
    vocab = meta["vocab"]
    feature_names = meta["feature_names"]
    horizons: List[int] = list(meta["horizons"])  # type: ignore
    num_horizons = len(horizons)

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

    # Load checkpoint to get architecture dims
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    cfg = ckpt.get("cfg", {})
    latent_dim = int(cfg.get("latent_dim", 64))
    hidden_dim = int(cfg.get("hidden_dim", 256))

    # Rebuild model using checkpoint dims
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
        RSSMConfig(latent_dim=latent_dim, hidden_dim=hidden_dim, stochastic=True)
    ).to(device)
    decoder = ObsDecoder(
        latent_dim=latent_dim, num_cont_features=len(feature_names)
    ).to(device)
    head = ReturnHead(latent_dim=latent_dim, num_horizons=num_horizons).to(device)

    encoder.load_state_dict(ckpt["encoder_state"])  # type: ignore
    rssm.load_state_dict(ckpt["rssm_state"])  # type: ignore
    decoder.load_state_dict(ckpt["decoder_state"])  # type: ignore
    head.load_state_dict(ckpt["ret_head_state"])  # type: ignore
    encoder.eval()
    rssm.eval()
    decoder.eval()
    head.eval()

    # Collect preds
    Y_list: List[torch.Tensor] = []
    MU_list: List[torch.Tensor] = []
    SIG_list: List[torch.Tensor] = []
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
        for t in range(L):
            mu_p, logv_p = rssm.prior(h)
            mu_q, logv_q = rssm.posterior(h, E[:, t, :])
            z_t = mu_q  # mean
            h = rssm.core(z_t, h, None)
        assert z_t is not None
        mu_y, logv_y = head(z_t)
        Y_list.append(yb)
        MU_list.append(mu_y.cpu())
        SIG_list.append(torch.exp(0.5 * logv_y).cpu())

    Y = torch.cat(Y_list, dim=0).cpu().numpy()
    MU = torch.cat(MU_list, dim=0).cpu().numpy()
    SIG = torch.cat(SIG_list, dim=0).cpu().numpy()

    # Per-horizon metrics
    for h_idx, h in enumerate(horizons):
        yt = Y[:, h_idx]
        mu = MU[:, h_idx]
        sig = SIG[:, h_idx]
        mse = float(np.mean((mu - yt) ** 2))
        mae = float(np.mean(np.abs(mu - yt)))
        rmse = math.sqrt(max(mse, 0.0))
        corr = (
            float(np.corrcoef(mu, yt)[0, 1]) if yt.std() > 0 and mu.std() > 0 else 0.0
        )
        diracc = float(np.mean(np.sign(mu) == np.sign(yt)))
        print(
            {
                "h": h,
                "N": int(len(yt)),
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "corr": corr,
                "diracc": diracc,
            }
        )

        # Threshold buckets
        score = np.abs(mu / np.maximum(sig, 1e-6)) if args.use_zscore else np.abs(mu)
        for pct in [10, 20, 30, 40, 50]:
            thr = np.percentile(score, 100 - pct)
            mask = score >= thr
            cov = float(mask.mean())
            if cov == 0.0:
                continue
            da = float(np.mean(np.sign(mu[mask]) == np.sign(yt[mask])))
            avg_ret = float(np.mean(yt[mask]))
            print(
                {
                    "h": h,
                    "coverage": cov,
                    "threshold_pct": pct,
                    "diracc": da,
                    "avg_realized_ret": avg_ret,
                }
            )

        # Simple PnL with costs
        pos = np.sign(mu)
        ret = pos * yt
        cost = np.abs(np.diff(pos, prepend=0)) * (args.txn_cost_bp * 1e-4)
        pnl = ret - cost
        equity = np.cumsum(pnl)
        print(
            {
                "h": h,
                "pnl_mean": float(pnl.mean()),
                "pnl_std": float(pnl.std()),
                "sharpe": sharpe_ratio(pnl),
                "max_drawdown": max_drawdown(equity),
                "final_equity": float(equity[-1]) if len(equity) else 0.0,
            }
        )


if __name__ == "__main__":
    main()
