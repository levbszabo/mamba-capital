"""
Compute calibration parameters (a, b, sigma scale s) from one or more eval CSVs
produced by eval_world_model.py --save_csv. Use TRAIN/VAL (not TEST!) to avoid
look-ahead. The output is a calibration_summary.csv compatible with
backtest_from_csv.py --calibration_csv.

Example:
  python calibrate_from_csv.py \
    --csv mamba-model/checkpoints_det/train_forecasts.csv \
          mamba-model/checkpoints_det/val_forecasts.csv \
    --out mamba-capital/data/calibration_summary_trainval.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def reliability(mu: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    mask = np.isfinite(mu) & np.isfinite(y)
    mu = mu[mask]
    y = y[mask]
    if mu.size == 0 or np.var(mu) <= 0:
        return 0.0, 1.0, 0.0
    cov = np.cov(mu, y, ddof=0)[0, 1]
    var_mu = np.var(mu)
    b = float(cov / (var_mu + 1e-12))
    a = float(np.mean(y) - b * np.mean(mu))
    y_hat = a + b * mu
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return a, b, float(max(min(r2, 1.0), 0.0))


def sigma_scale(mu: np.ndarray, y: np.ndarray, sig: np.ndarray) -> float:
    mask = np.isfinite(mu) & np.isfinite(y) & np.isfinite(sig) & (sig > 0)
    mu = mu[mask]
    y = y[mask]
    sig = sig[mask]
    if mu.size == 0:
        return 1.0
    var_ratio = float(np.mean((y - mu) ** 2) / (np.mean(sig**2) + 1e-12))
    s = float(np.sqrt(max(var_ratio, 1e-12)))
    return s


def collect_horizons(cols: List[str]) -> List[int]:
    hs: List[int] = []
    for c in cols:
        if c.startswith("mu_") and c.endswith("h"):
            try:
                h = int(c.split("_")[1][:-1])
                hs.append(h)
            except Exception:
                pass
    return sorted(list(set(hs)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate a,b,s from eval CSVs")
    ap.add_argument("--csv", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in args.csv]
    df = pd.concat(frames, axis=0, ignore_index=True)
    horizons = collect_horizons(df.columns.tolist())
    rows: List[Dict[str, object]] = []
    for h in horizons:
        mu_col = f"mu_{h}h"
        sig_col = f"sig_{h}h"
        y_col = f"y_{h}h"
        if not {mu_col, sig_col, y_col}.issubset(df.columns):
            continue
        mu = df[mu_col].to_numpy()
        y = df[y_col].to_numpy()
        sig = df[sig_col].to_numpy()

        a, b, r2 = reliability(mu, y)
        s = sigma_scale(mu, y, sig)
        e = (y - mu) / (sig + 1e-12)
        row = {
            "horizon": f"{h}h",
            "N": int(np.isfinite(e).sum()),
            "mean_e": float(np.nanmean(e)),
            "std_e": float(np.nanstd(e)),
            "skew_e": float(pd.Series(e).skew()),
            "kurt_excess_e": float(pd.Series(e).kurt()),
            "var_ratio_E[(y-mu)^2]/E[sig^2]": float(
                np.mean((y - mu) ** 2) / (np.mean(sig**2) + 1e-12)
            ),
            "suggested_sigma_scale_s": s,
            "reliability_intercept_a": a,
            "reliability_slope_b": b,
            "reliability_R2": r2,
        }
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved calibration to {out} ({len(rows)} horizons)")


if __name__ == "__main__":
    main()
