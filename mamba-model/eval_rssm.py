"""
Evaluate RSSM effectiveness on held-out splits with financially relevant metrics.

Metrics reported:
  - Return head: MSE, MAE, RMSE, correlation, DirAcc
  - Thresholded performance by |y_hat| percentile: coverage, DirAcc, avg realized return
  - Simple PnL simulation with position = sign(y_hat) * 1.0 (optional threshold),
    transaction cost, Sharpe, max drawdown

Usage:
  python mamba-model/eval_rssm.py \
    --dataset_dir mamba-model/datasets/fx_1h_l512_s2 \
    --checkpoint mamba-model/checkpoints/rssm_1h_kl_best.pt \
    --split val --batch_size 256 --txn_cost_bp 0.5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rssm import RSSM, RSSMConfig
from train_rssm_direct import SequenceEncoder, ReturnHead


def load_tensors(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict]:
    data: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ["train", "val", "test"]:
        data[split] = torch.load(
            dataset_dir / f"dataset_{split}.pt", map_location="cpu"
        )
    meta = torch.load(dataset_dir / "meta.pt", map_location="cpu")
    return data, meta


def make_loader(tensors: Dict[str, torch.Tensor], batch_size: int) -> DataLoader:
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
    return DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-12) -> float:
    mean = returns.mean()
    std = returns.std() + eps
    return float(mean / std)


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    return float(dd.min())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Evaluate RSSM effectiveness (return head)")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--txn_cost_bp",
        type=float,
        default=0.5,
        help="transaction cost in basis points per unit turnover",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, meta = load_tensors(Path(args.dataset_dir))
    vocab = meta["vocab"]
    feature_names = meta["feature_names"]
    horizons: List[int] = meta["horizons"]
    num_horizons = len(horizons)

    loader = make_loader(data[args.split], args.batch_size)

    # Rebuild models
    encoder = SequenceEncoder(
        num_cont_features=len(feature_names),
        num_symbols=int(vocab["num_symbols"]),
        hour_size=int(vocab["hour_size"]),
        dow_size=int(vocab["dow_size"]),
        num_bases=int(vocab.get("num_bases", 0)) or None,
        num_quotes=int(vocab.get("num_quotes", 0)) or None,
        d_model=256,
        n_layers=2,
        dropout=0.1,
        latent_dim=64,
    ).to(device)
    rssm = RSSM(RSSMConfig(latent_dim=64, hidden_dim=256, stochastic=True)).to(device)
    head = ReturnHead(latent_dim=64, num_horizons=num_horizons).to(device)

    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    encoder.load_state_dict(ckpt["encoder_state"])  # type: ignore
    rssm.load_state_dict(ckpt["rssm_state"])  # type: ignore
    if "ret_head_state" in ckpt:
        head.load_state_dict(ckpt["ret_head_state"])  # type: ignore
    encoder.eval()
    rssm.eval()
    head.eval()

    # Get target scalers for descaling
    target_scalers = meta.get("target_scalers", {})

    # Collect predictions and targets
    Y_list: List[torch.Tensor] = []
    YH_list: List[torch.Tensor] = []
    SYM_list: List[torch.Tensor] = []  # track symbols for descaling
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
        if x_base is not None:
            x_base = x_base.to(device)
        if x_quote is not None:
            x_quote = x_quote.to(device)

        Z = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)
        z_last = Z[:, -1, :]
        y_hat = head(z_last)
        Y_list.append(y)
        YH_list.append(y_hat.cpu())
        SYM_list.append(x_sym.cpu())

    Y = torch.cat(Y_list, dim=0).cpu().numpy()  # [N, H]
    YH = torch.cat(YH_list, dim=0).cpu().numpy()
    SYM = torch.cat(SYM_list, dim=0).cpu().numpy()  # [N]

    # Descale targets and predictions if scalers available
    if target_scalers:
        vocab_symbols = meta["vocab"]["symbols"]
        for i in range(Y.shape[0]):
            sym_id = int(SYM[i])
            if sym_id < len(vocab_symbols):
                symbol = vocab_symbols[sym_id]
                if symbol in target_scalers:
                    scaler = target_scalers[symbol]
                    mean_vals = np.array(scaler["mean"])
                    std_vals = np.array(scaler["std"])
                    # Descale: y_orig = y_scaled * std + mean
                    Y[i, :] = Y[i, :] * std_vals + mean_vals
                    YH[i, :] = YH[i, :] * std_vals + mean_vals

    # Evaluate per horizon
    for h_idx, h in enumerate(horizons):
        yt = Y[:, h_idx]
        yh = YH[:, h_idx]
        mse = float(np.mean((yh - yt) ** 2))
        mae = float(np.mean(np.abs(yh - yt)))
        rmse = math.sqrt(mse)
        corr = (
            float(np.corrcoef(yh, yt)[0, 1]) if yt.std() > 0 and yh.std() > 0 else 0.0
        )
        diracc = float(np.mean(np.sign(yh) == np.sign(yt)))
        print(
            {
                "horizon_h": h,
                "N": int(len(yt)),
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "corr": corr,
                "diracc": diracc,
            }
        )

        # Thresholded analysis by |y_hat|
        for pct in [10, 20, 30, 40, 50]:
            thr = np.percentile(np.abs(yh), 100 - pct)
            mask = np.abs(yh) >= thr
            cov = float(mask.mean())
            if cov == 0.0:
                continue
            da = float(np.mean(np.sign(yh[mask]) == np.sign(yt[mask])))
            avg_ret = float(np.mean(yt[mask]))
            print(
                {
                    "horizon_h": h,
                    "coverage": cov,
                    "threshold_pct": pct,
                    "diracc": da,
                    "avg_realized_ret": avg_ret,
                }
            )

        # Simple trading PnL: position = sign(y_hat)
        pos = np.sign(yh)
        ret = pos * yt
        # transaction cost per change in position
        cost = abs(np.diff(pos, prepend=0)) * (args.txn_cost_bp * 1e-4)
        pnl = ret - cost
        equity = np.cumsum(pnl)
        shrp = sharpe_ratio(pnl)
        mdd = max_drawdown(equity)
        print(
            {
                "horizon_h": h,
                "pnl_mean": float(pnl.mean()),
                "pnl_std": float(pnl.std()),
                "sharpe": shrp,
                "max_drawdown": mdd,
                "final_equity": float(equity[-1]) if len(equity) > 0 else 0.0,
            }
        )


if __name__ == "__main__":
    main()
