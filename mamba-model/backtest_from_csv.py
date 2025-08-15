"""
Backtest trading rules directly from an evaluation CSV (per-asset forecasts).

Input CSV (from eval_world_model.py --save_csv):
  - Required columns: symbol, ts_utc, mu_{h}h, sig_{h}h, z_{h}h, y_{h}h
  - May also contain extra metadata columns (preserved by eval script)

This script:
  1) Optionally calibrates mu and sigma using a reliability/sigma scaling file
     (e.g., calibration_summary.csv) or manual flags
  2) Builds cross-sectional signals each timestamp across symbols
  3) Applies gates (|z|≥thr or top-pct), position sizing, and txn costs
  4) Produces metrics (Sharpe, turnover, coverage) and plots

Usage example:
  python backtest_from_csv.py \
    --csv mamba-model/checkpoints_det/test_forecasts.csv \
    --horizon 24 --z_thr 0.05 --sizing sign --cost_bp 0.5 \
    --calibration_csv mamba-capital/data/calibration_summary.csv \
    --out_dir mamba-model/results/backtest_csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Calibration:
    intercept: float = 0.0  # a in y ≈ a + b*mu
    slope: float = 1.0  # b
    sigma_scale: float = 1.0  # s: sigma' = s * sigma

    def apply(
        self, mu: np.ndarray, sig: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu_hat = self.intercept + self.slope * mu
        sig_hat = np.maximum(self.sigma_scale * sig, 1e-12)
        z_hat = mu_hat / sig_hat
        return mu_hat, sig_hat, z_hat


def load_calibration_from_csv(calib_csv: Path, horizon: int) -> Calibration:
    df = pd.read_csv(calib_csv)
    key = f"{horizon}h"
    row = df.loc[df["horizon"] == key]
    if row.empty:
        return Calibration()
    r = row.iloc[0]
    # Fallbacks if columns missing
    a = float(r.get("reliability_intercept_a", 0.0))
    b = float(r.get("reliability_slope_b", 1.0))
    s = float(r.get("suggested_sigma_scale_s", 1.0))
    return Calibration(intercept=a, slope=b, sigma_scale=s)


def sharpe_ratio(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(x) / (np.std(x) + eps))


def run_backtest(
    csv_path: Path,
    out_dir: Path,
    horizon: int,
    z_thr: float,
    top_pct: float,
    sizing: str,
    k: float,
    cost_bp: float,
    calib: Calibration,
    stride: int = 0,
    recompute_y_from_prices: bool = False,
    price_dir: Optional[Path] = None,
    price_glob: str = "*.csv",
    price_ts_col: str = "ts_utc",
    price_sym_col: str = "symbol",
    price_close_col: str = "close",
    price_ret_col: str = "",
    overlap_scale: bool = False,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    # Parse timestamp; tolerate missing by creating an index if needed
    # In eval CSVs we may have ts_utc or ts_end. Prefer ts_utc; fall back to ts_end.
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    elif "ts_end" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_end"], errors="coerce")
    else:
        df["ts_utc"] = pd.NaT
    # Required columns for selected horizon
    mu_col = f"mu_{horizon}h"
    sig_col = f"sig_{horizon}h"
    z_col = f"z_{horizon}h"
    y_col = f"y_{horizon}h"
    for c in ["symbol", mu_col, sig_col, z_col, y_col]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    # Optionally recompute realized returns from source price files
    if recompute_y_from_prices:
        if price_dir is None:
            raise ValueError(
                "price_dir must be provided when recompute_y_from_prices=True"
            )
        price_rows = []
        matched_files = list(Path(price_dir).glob(price_glob))
        if not matched_files:
            raise RuntimeError(
                f"No files matched: dir={price_dir} glob='{price_glob}'."
            )
        for p in matched_files:
            try:
                pf = pd.read_csv(p)
            except Exception:
                continue
            # Column fallback mapping
            ts_col = (
                price_ts_col
                if price_ts_col in pf.columns
                else (
                    "ts_utc"
                    if "ts_utc" in pf.columns
                    else (
                        "date"
                        if "date" in pf.columns
                        else ("timestamp" if "timestamp" in pf.columns else None)
                    )
                )
            )
            sym_col = (
                price_sym_col
                if price_sym_col in pf.columns
                else (
                    "symbol"
                    if "symbol" in pf.columns
                    else ("pair" if "pair" in pf.columns else None)
                )
            )
            if ts_col is None or sym_col is None:
                continue
            pf = pf[
                [sym_col, ts_col]
                + ([price_close_col] if price_close_col in pf.columns else [])
                + (
                    [price_ret_col]
                    if price_ret_col and price_ret_col in pf.columns
                    else []
                )
            ].copy()
            pf.rename(columns={sym_col: "symbol", ts_col: "ts_utc"}, inplace=True)
            pf["ts_utc"] = pd.to_datetime(pf["ts_utc"], errors="coerce")
            pf.sort_values(["symbol", "ts_utc"], inplace=True)
            if price_ret_col and price_ret_col in pf.columns:
                pf["y_price"] = pf[price_ret_col]
            else:
                if price_close_col not in pf.columns:
                    continue
                # Future log return over horizon H
                pf["close"] = pf[price_close_col].astype(float)
                pf["close_fwd"] = pf.groupby("symbol")["close"].shift(-horizon)
                pf["y_price"] = np.log(pf["close_fwd"] / pf["close"])  # log-return
            price_rows.append(pf[["symbol", "ts_utc", "y_price"]])
        if not price_rows:
            raise RuntimeError(
                "No usable price files after parsing (check column names: symbol/date/ts_utc, close)."
            )
        prices = pd.concat(price_rows, axis=0, ignore_index=True)
        # Join onto df (requires ts_utc present)
        if df["ts_utc"].isna().any():
            raise RuntimeError(
                "ts_utc missing in eval CSV; cannot join prices. Re-export with --require_samples_meta."
            )
        df = df.merge(prices, on=["symbol", "ts_utc"], how="left")
        # Overwrite y with price-based
        df[y_col] = df["y_price"]

    # Drop rows lacking outcomes
    df = df.dropna(subset=[mu_col, sig_col, y_col])
    # Apply calibration
    mu_hat, sig_hat, z_hat = calib.apply(df[mu_col].to_numpy(), df[sig_col].to_numpy())
    df["mu_hat"] = mu_hat
    df["sig_hat"] = sig_hat
    df["z_hat"] = z_hat

    # Build time index; if timestamps missing, sort by original order per symbol
    if df["ts_utc"].isna().all():
        # Create synthetic timestamps by rank ordering
        df = df.sort_values(["symbol"]).copy()
        df["i"] = df.groupby("symbol").cumcount()
        # Create a combined time index by i; this assumes aligned windows
        df["ts_ix"] = df["i"]
        ts_key = "ts_ix"
    else:
        df = df.sort_values(["ts_utc", "symbol"])  # stable ordering
        ts_key = "ts_utc"

    # Cross-sectional selection per timestamp
    symbols = sorted(df["symbol"].unique().tolist())
    cost = cost_bp * 1e-4
    # Position table over time for turnover
    pos_prev: Dict[str, float] = {s: 0.0 for s in symbols}

    eq: Dict[pd.Timestamp, float] = {}
    ret_ts: Dict[pd.Timestamp, float] = {}
    turnover_ts: Dict[pd.Timestamp, float] = {}

    pnl_list = []
    dates = []
    # Apply stride over timestamps to avoid overlapping targets double-counting
    grouped = list(df.groupby(ts_key))
    if stride and stride > 0:
        grouped = [grp for idx, grp in enumerate(grouped) if (idx % stride) == 0]

    for ts, g in grouped:
        # Select set
        z = g["z_hat"].to_numpy()
        mu = g["mu_hat"].to_numpy()
        y = g[y_col].to_numpy()
        sym = g["symbol"].to_numpy()

        if top_pct > 0.0:
            thr = np.quantile(np.abs(z), 1.0 - top_pct)
            sel = np.abs(z) >= thr
        else:
            sel = np.abs(z) >= z_thr
        # Positions
        if sizing == "sign":
            pos = np.sign(mu) * sel.astype(float)
        elif sizing == "z":
            pos = np.clip(k * z, -1.0, 1.0) * sel.astype(float)
        else:  # tanh
            pos = np.tanh(k * z) * sel.astype(float)

        # Turnover / cost
        txn = 0.0
        pnl = 0.0
        for i in range(len(sym)):
            s = str(sym[i])
            p_prev = pos_prev.get(s, 0.0)
            p_now = float(pos[i])
            pnl += p_now * float(y[i])
            txn += abs(p_now - p_prev)
            pos_prev[s] = p_now
        # Overlap scale to avoid overcounting when trading every bar on H-hour labels
        scale = (
            (1.0 / max(horizon, 1))
            if (overlap_scale and (not stride or stride <= 1))
            else 1.0
        )
        pnl_net = (pnl * scale) - (txn * cost * scale)
        pnl_list.append(pnl_net)
        dates.append(ts)
        eq[ts] = pnl_net
        turnover_ts[ts] = txn
        ret_ts[ts] = pnl

    pnl_arr = np.array(pnl_list, dtype=float)
    sr_hourly = sharpe_ratio(pnl_arr)
    sr_annual = sr_hourly * np.sqrt(24 * 365)  # hourly to annualized (approx)
    avg_turnover = float(np.mean(list(turnover_ts.values())))

    # Plots
    t_index = (
        pd.to_datetime(pd.Series(dates)) if ts_key == "ts_utc" else pd.Series(dates)
    )
    eq_curve = np.cumsum(pnl_arr)
    plt.figure(figsize=(10, 5))
    plt.plot(t_index, eq_curve)
    title = f"Equity Curve — {horizon}h, {'top% '+str(top_pct) if top_pct>0 else '|z|≥'+str(z_thr)}, {sizing}, cost {cost_bp}bp"
    plt.title(title)
    plt.xlabel("timestamp (UTC)" if ts_key == "ts_utc" else "index")
    plt.ylabel("cumulative return (net)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / f"equity_h{horizon}.png", dpi=150)
    plt.close()

    # Save summary
    summary = {
        "horizon": horizon,
        "z_thr": z_thr,
        "top_pct": top_pct,
        "sizing": sizing,
        "k": k,
        "cost_bp": cost_bp,
        "N": int(len(pnl_arr)),
        "mean_hourly": float(np.mean(pnl_arr)),
        "std_hourly": float(np.std(pnl_arr)),
        "sharpe_hourly": float(sr_hourly),
        "sharpe_annual": float(sr_annual),
        "avg_turnover": avg_turnover,
        "coverage": float(
            np.mean(
                (
                    df["z_hat"].abs()
                    >= (
                        np.quantile(df["z_hat"].abs(), 1.0 - top_pct)
                        if top_pct > 0
                        else z_thr
                    )
                ).astype(float)
            )
        ),
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"metrics_h{horizon}.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest a simple trading rule from eval CSV"
    )
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="mamba-model/results/backtest_csv")
    p.add_argument("--horizon", type=int, choices=[6, 24], default=24)
    p.add_argument("--z_thr", type=float, default=0.1)
    p.add_argument(
        "--top_pct",
        type=float,
        default=0.0,
        help="If >0, ignore z_thr and trade top pct by |z| each hour",
    )
    p.add_argument("--sizing", type=str, choices=["sign", "z", "tanh"], default="sign")
    p.add_argument(
        "--k", type=float, default=0.5, help="Sizing scale (for sizing=z/tanh)"
    )
    p.add_argument("--cost_bp", type=float, default=0.5)
    p.add_argument(
        "--calibration_csv",
        type=str,
        default="",
        help="Use suggested a,b,s per horizon from this CSV",
    )
    p.add_argument("--a", type=float, default=0.0)
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--s", type=float, default=1.0)
    p.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Trade every Nth timestamp to avoid overlapping horizons (e.g., 6 or 24)",
    )
    p.add_argument(
        "--overlap_scale",
        action="store_true",
        help="Scale PnL by 1/horizon if trading every bar on H-hour labels",
    )
    p.add_argument(
        "--recompute_y",
        action="store_true",
        help="Recompute realized returns from source price files and overwrite y_{h}h",
    )
    p.add_argument(
        "--price_dir",
        type=str,
        default="",
        help="Directory of per-asset price CSVs to join (must include symbol and ts_utc)",
    )
    p.add_argument("--price_glob", type=str, default="*.csv")
    p.add_argument("--price_ts_col", type=str, default="ts_utc")
    p.add_argument("--price_sym_col", type=str, default="symbol")
    p.add_argument("--price_close_col", type=str, default="close")
    p.add_argument(
        "--price_ret_col",
        type=str,
        default="",
        help="If provided, use this column as realized return instead of computing from close",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    # Calibration priority: CSV > manual > identity
    if args.calibration_csv:
        calib = load_calibration_from_csv(Path(args.calibration_csv), args.horizon)
    else:
        calib = Calibration(intercept=args.a, slope=args.b, sigma_scale=args.s)
    summary = run_backtest(
        csv_path=Path(args.csv),
        out_dir=out_dir,
        horizon=args.horizon,
        z_thr=args.z_thr,
        top_pct=args.top_pct,
        sizing=args.sizing,
        k=args.k,
        cost_bp=args.cost_bp,
        calib=calib,
        stride=int(args.stride),
        recompute_y_from_prices=bool(args.recompute_y),
        price_dir=Path(args.price_dir) if args.price_dir else None,
        price_glob=args.price_glob,
        price_ts_col=args.price_ts_col,
        price_sym_col=args.price_sym_col,
        price_close_col=args.price_close_col,
        price_ret_col=args.price_ret_col,
        overlap_scale=bool(args.overlap_scale),
    )
    print(summary)


if __name__ == "__main__":
    main()
