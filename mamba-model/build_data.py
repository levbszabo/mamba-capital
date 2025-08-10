"""
Builds PyTorch-ready tensors from cleaned FX hourly CSVs.

Functionality:
- Read `forex_cleaned/processed_*_hourly.csv` files
- Compute forward targets: fwd_ret_log_{1h,6h,24h}
- Create continuous features and categorical IDs (symbol, ny_hour, ny_dow)
- Fit per-symbol scalers on train split; apply to all splits
- Construct fixed-length windows (sequence_length) with configurable strides
- Save train/val/test tensors and metadata as .pt files

Usage:
  python build_data.py \
    --input_dir /Users/leventeszabo/mamba-capital/data/forex_cleaned \
    --output_dir /Users/leventeszabo/mamba-capital/mamba-model/datasets/fx_mamba_v1 \
    --sequence_length 512 \
    --train_stride 4 --eval_stride 1

Notes:
- Expects columns: symbol, ts_utc, open, high, low, close, volume, is_gap,
  ny_hour, ny_dow, ret_log_1h, ret_log_6h, ret_log_24h, vol_real_24h (some may be missing).
- `ret_log_*` are used as backward-looking features (inputs), not targets.
- Targets are forward-looking log returns computed from close prices.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# -----------------------------
# Configuration dataclasses
# -----------------------------


@dataclass
class TimeSplits:
    train_start: Optional[pd.Timestamp]
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: Optional[pd.Timestamp]


@dataclass
class BuildConfig:
    input_dir: Path
    output_dir: Path
    sequence_length: int
    train_stride: int
    eval_stride: int
    symbols: Optional[List[str]]
    standardize_targets: bool
    splits: TimeSplits
    horizons: Tuple[int, int, int] = (1, 6, 24)


# -----------------------------
# IO helpers
# -----------------------------


def _discover_symbol_files(input_dir: Path, symbols: Optional[List[str]]) -> List[Path]:
    files = sorted(input_dir.glob("processed_*_hourly.csv"))
    if symbols is None:
        return files
    normalized_symbols = {s.upper() for s in symbols}
    selected: List[Path] = []
    for f in files:
        # Expect filenames like processed_EURUSD_hourly.csv
        name = f.name
        try:
            sym = name.split("processed_")[1].split("_hourly")[0].upper()
        except Exception:
            continue
        if sym in normalized_symbols:
            selected.append(f)
    return selected


def _parse_timestamp(ts_series: pd.Series) -> pd.Series:
    # Robust parsing of UTC timestamps; keep tz-aware if present
    parsed = pd.to_datetime(ts_series, utc=True, errors="coerce")
    return parsed


def load_cleaned_forex(
    input_dir: Path, symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    csv_files = _discover_symbol_files(input_dir, symbols)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir} matching processed_*_hourly.csv"
        )

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        # Minimal required columns
        required = [
            "symbol",
            "ts_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ny_hour",
            "ny_dow",
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_path}")

        # Parse & normalize
        df["symbol"] = df["symbol"].astype(str)
        df["ts_utc"] = _parse_timestamp(df["ts_utc"])  # type: ignore
        if "is_gap" in df.columns:
            # Normalize to 0/1 robustly (handle booleans and string literals)
            if df["is_gap"].dtype == bool:
                df["is_gap"] = df["is_gap"].astype(int)
            else:
                df["is_gap"] = (
                    df["is_gap"]
                    .astype(str)
                    .str.lower()
                    .map(
                        {
                            "true": 1,
                            "false": 0,
                            "1": 1,
                            "0": 0,
                        }
                    )
                    .fillna(0)
                ).astype(int)
        else:
            df["is_gap"] = 0

        frames.append(df)

    data = pd.concat(frames, axis=0, ignore_index=True)
    # Sort and reset index
    data = data.sort_values(["symbol", "ts_utc"], kind="mergesort").reset_index(
        drop=True
    )
    return data


# -----------------------------
# Feature engineering
# -----------------------------


def compute_forward_returns_per_symbol(
    data: pd.DataFrame, horizons: Iterable[int]
) -> pd.DataFrame:
    # Operate per symbol to avoid cross-symbol leakage
    data = data.copy()
    data["log_close"] = np.log(data["close"].astype(float))
    group = data.groupby("symbol", sort=False, group_keys=False)
    for h in horizons:
        data[f"fwd_ret_log_{h}h"] = group["log_close"].shift(-h) - group[
            "log_close"
        ].shift(0)
    return data


def add_derived_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    data = data.copy()

    # Continuous base features
    data["log_close"] = np.log(data["close"].astype(float))
    data["log_volume"] = np.log1p(data["volume"].astype(float))
    data["log_co"] = np.log(data["close"].astype(float)) - np.log(
        data["open"].astype(float)
    )
    # Range relative to close (safe if close>0)
    with np.errstate(divide="ignore", invalid="ignore"):
        data["hl_range"] = (
            data["high"].astype(float) - data["low"].astype(float)
        ) / data["close"].astype(float)

    # Use backward-looking returns if present
    candidate_cols = [
        "ret_log_1h",
        "ret_log_6h",
        "ret_log_24h",
        "vol_real_24h",
    ]
    available = [c for c in candidate_cols if c in data.columns]

    # is_gap as numeric feature
    if "is_gap" not in data.columns:
        data["is_gap"] = 0

    continuous_features = (
        [
            "log_close",
            "log_volume",
            "log_co",
            "hl_range",
        ]
        + available
        + ["is_gap"]
    )

    # Clean up infs
    for col in continuous_features:
        vals = data[col].astype(float)
        vals[~np.isfinite(vals)] = np.nan
        data[col] = vals

    return data, continuous_features


def build_vocabularies(data: pd.DataFrame) -> Dict[str, Dict]:
    symbols = sorted(data["symbol"].astype(str).unique().tolist())
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    id_to_symbol = {i: s for s, i in symbol_to_id.items()}

    # ny_hour and ny_dow are already numeric in 0..23 and 0..6
    vocab = {
        "symbol_to_id": symbol_to_id,
        "id_to_symbol": id_to_symbol,
        "num_symbols": len(symbols),
        "hour_size": 24,
        "dow_size": 7,
    }
    return vocab


# -----------------------------
# Splitting and scaling
# -----------------------------


def assign_splits(data: pd.DataFrame, splits: TimeSplits) -> pd.DataFrame:
    data = data.copy()
    ts = data["ts_utc"]

    def in_range(
        start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
    ) -> pd.Series:
        cond = pd.Series(True, index=data.index)
        if start is not None:
            cond &= ts >= start
        if end is not None:
            cond &= ts <= end
        return cond

    data["split"] = "train"
    data.loc[in_range(splits.val_start, splits.val_end), "split"] = "val"
    data.loc[in_range(splits.test_start, splits.test_end), "split"] = "test"
    data.loc[in_range(None, splits.train_end) & (ts < splits.val_start), "split"] = (
        "train"
    )
    return data


def fit_per_symbol_scalers(
    data: pd.DataFrame,
    continuous_features: List[str],
    split_col: str = "split",
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    # Fit mean/std per symbol on the train split only
    scalers: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for symbol, df_sym in data.groupby("symbol", sort=False):
        df_train = df_sym[df_sym[split_col] == "train"]
        feature_to_stats: Dict[str, Tuple[float, float]] = {}
        for feat in continuous_features:
            vals = pd.to_numeric(df_train[feat], errors="coerce").astype(float)
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            if not np.isfinite(mean):
                mean = 0.0
            feature_to_stats[feat] = (float(mean), float(std))
        scalers[symbol] = feature_to_stats
    return scalers


def apply_per_symbol_scalers(
    data: pd.DataFrame,
    continuous_features: List[str],
    scalers: Dict[str, Dict[str, Tuple[float, float]]],
) -> pd.DataFrame:
    data = data.copy()

    # Scale features per symbol using train-fitted stats
    def scale_row(row: pd.Series) -> pd.Series:
        symbol = row["symbol"]
        stats = scalers[symbol]
        for feat in continuous_features:
            mean, std = stats[feat]
            val = row[feat]
            if pd.isna(val):
                row[feat] = 0.0
            else:
                row[feat] = (float(val) - mean) / std
        return row

    data = data.apply(scale_row, axis=1)
    return data


def fit_target_scalers(
    data: pd.DataFrame,
    horizons: Tuple[int, int, int],
    split_col: str = "split",
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    target_cols = [f"fwd_ret_log_{h}h" for h in horizons]
    scalers: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for symbol, df_sym in data.groupby("symbol", sort=False):
        df_train = df_sym[df_sym[split_col] == "train"]
        stats: Dict[str, Tuple[float, float]] = {}
        for col in target_cols:
            vals = pd.to_numeric(df_train[col], errors="coerce").astype(float)
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            if not np.isfinite(mean):
                mean = 0.0
            stats[col] = (float(mean), float(std))
        scalers[symbol] = stats
    return scalers


def apply_target_scalers(
    data: pd.DataFrame,
    horizons: Tuple[int, int, int],
    target_scalers: Dict[str, Dict[str, Tuple[float, float]]],
) -> pd.DataFrame:
    data = data.copy()
    target_cols = [f"fwd_ret_log_{h}h" for h in horizons]

    def scale_row(row: pd.Series) -> pd.Series:
        symbol = row["symbol"]
        stats = target_scalers[symbol]
        for col in target_cols:
            val = row[col]
            mean, std = stats[col]
            if pd.isna(val):
                row[col] = np.nan  # Keep NaNs; we'll filter later
            else:
                row[col] = (float(val) - mean) / std
        return row

    data = data.apply(scale_row, axis=1)
    return data


# -----------------------------
# Window construction
# -----------------------------


def _windowize_symbol(
    df_sym: pd.DataFrame,
    symbol_id: int,
    continuous_features: List[str],
    horizons: Tuple[int, int, int],
    sequence_length: int,
    stride: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]
]:
    # Returns: x_cont [N,L,C], x_sym_id [N,L], x_hour_id [N,L], x_dow_id [N,L], y [N,3], end_timestamps [N]
    df = df_sym.reset_index(drop=True)
    num_rows = df.shape[0]

    feature_matrix = df[continuous_features].to_numpy(dtype=np.float32)
    ny_hour = df["ny_hour"].astype(int).to_numpy()
    ny_dow = df["ny_dow"].astype(int).to_numpy()
    ts = df["ts_utc"].to_numpy()

    target_cols = [f"fwd_ret_log_{h}h" for h in horizons]
    targets = df[target_cols].to_numpy(dtype=np.float32)

    c = feature_matrix.shape[1]
    max_h = max(horizons)

    x_cont_list: List[np.ndarray] = []
    x_sym_id_list: List[np.ndarray] = []
    x_hour_id_list: List[np.ndarray] = []
    x_dow_id_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    ts_end_list: List[pd.Timestamp] = []

    # Precompute valid mask for feature finiteness
    finite_mask = np.isfinite(feature_matrix).all(axis=1)
    target_valid_mask = np.isfinite(targets).all(axis=1)

    # The end index i must allow forward targets up to max horizon
    start_i = sequence_length - 1
    end_i = (num_rows - 1) - max_h  # i inclusive upper bound

    if end_i < start_i:
        return (
            np.zeros((0, sequence_length, c), dtype=np.float32),
            np.zeros((0, sequence_length), dtype=np.int64),
            np.zeros((0, sequence_length), dtype=np.int64),
            np.zeros((0, sequence_length), dtype=np.int64),
            np.zeros((0, len(horizons)), dtype=np.float32),
            [],
        )

    # Sliding window
    for i in range(start_i, end_i + 1, stride):
        window_slice = slice(i - sequence_length + 1, i + 1)
        # Validate features in the window
        if not finite_mask[window_slice].all():
            continue
        # Validate targets at time i
        if not target_valid_mask[i]:
            continue

        x_cont = feature_matrix[window_slice, :]
        x_hour = ny_hour[window_slice]
        x_dow = ny_dow[window_slice]
        # Symbol id is constant along the window
        x_sym = np.full((sequence_length,), symbol_id, dtype=np.int64)

        y = targets[i, :]
        timestamp_end = pd.Timestamp(ts[i])

        x_cont_list.append(x_cont)
        x_sym_id_list.append(x_sym)
        x_hour_id_list.append(x_hour)
        x_dow_id_list.append(x_dow)
        y_list.append(y)
        ts_end_list.append(timestamp_end)

    if not x_cont_list:
        return (
            np.zeros((0, sequence_length, c), dtype=np.float32),
            np.zeros((0, sequence_length), dtype=np.int64),
            np.zeros((0, sequence_length), dtype=np.int64),
            np.zeros((0, sequence_length), dtype=np.int64),
            np.zeros((0, len(horizons)), dtype=np.float32),
            [],
        )

    X_cont = np.stack(x_cont_list, axis=0).astype(np.float32)
    X_sym = np.stack(x_sym_id_list, axis=0).astype(np.int64)
    X_hour = np.stack(x_hour_id_list, axis=0).astype(np.int64)
    X_dow = np.stack(x_dow_id_list, axis=0).astype(np.int64)
    Y = np.stack(y_list, axis=0).astype(np.float32)

    return X_cont, X_sym, X_hour, X_dow, Y, ts_end_list


def build_windows(
    data: pd.DataFrame,
    vocab: Dict[str, Dict],
    continuous_features: List[str],
    horizons: Tuple[int, int, int],
    sequence_length: int,
    train_stride: int,
    eval_stride: int,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, List[Dict[str, object]]]]:
    # Returns tensors per split and lightweight per-sample metadata (symbol, ts_end)
    split_to_stride = {"train": train_stride, "val": eval_stride, "test": eval_stride}
    datasets: Dict[str, Dict[str, torch.Tensor]] = {}
    metas: Dict[str, List[Dict[str, object]]] = {}

    symbol_to_id: Dict[str, int] = vocab["symbol_to_id"]  # type: ignore
    target_cols = [f"fwd_ret_log_{h}h" for h in horizons]

    for split_name in ["train", "val", "test"]:
        X_cont_all: List[np.ndarray] = []
        X_sym_all: List[np.ndarray] = []
        X_hour_all: List[np.ndarray] = []
        X_dow_all: List[np.ndarray] = []
        Y_all: List[np.ndarray] = []
        meta_list: List[Dict[str, object]] = []

        stride = split_to_stride[split_name]
        df_split = data[data["split"] == split_name]

        # Group by symbol to windowize independently
        for symbol, df_sym in df_split.groupby("symbol", sort=False):
            sym_id = symbol_to_id[symbol]
            X_cont, X_sym, X_hour, X_dow, Y, ts_end = _windowize_symbol(
                df_sym=df_sym,
                symbol_id=sym_id,
                continuous_features=continuous_features,
                horizons=horizons,
                sequence_length=sequence_length,
                stride=stride,
            )

            if X_cont.shape[0] == 0:
                continue

            X_cont_all.append(X_cont)
            X_sym_all.append(X_sym)
            X_hour_all.append(X_hour)
            X_dow_all.append(X_dow)
            Y_all.append(Y)
            # Prepare metadata entries
            for t in ts_end:
                meta_list.append(
                    {"symbol": symbol, "ts_end": pd.Timestamp(t).isoformat()}
                )

        if not X_cont_all:
            # Empty split
            datasets[split_name] = {
                "x_cont": torch.zeros(
                    (0, sequence_length, len(continuous_features)), dtype=torch.float32
                ),
                "x_symbol_id": torch.zeros((0, sequence_length), dtype=torch.long),
                "x_ny_hour_id": torch.zeros((0, sequence_length), dtype=torch.long),
                "x_ny_dow_id": torch.zeros((0, sequence_length), dtype=torch.long),
                "y": torch.zeros((0, len(horizons)), dtype=torch.float32),
            }
            metas[split_name] = []
            continue

        X_cont = torch.from_numpy(np.concatenate(X_cont_all, axis=0))
        X_sym = torch.from_numpy(np.concatenate(X_sym_all, axis=0))
        X_hour = torch.from_numpy(np.concatenate(X_hour_all, axis=0))
        X_dow = torch.from_numpy(np.concatenate(X_dow_all, axis=0))
        Y = torch.from_numpy(np.concatenate(Y_all, axis=0))

        datasets[split_name] = {
            "x_cont": X_cont,
            "x_symbol_id": X_sym,
            "x_ny_hour_id": X_hour,
            "x_ny_dow_id": X_dow,
            "y": Y,
        }
        metas[split_name] = meta_list

    return datasets, metas


# -----------------------------
# Main entry
# -----------------------------


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Build PyTorch-ready datasets for Mamba FX forecasting"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=False,
        default="/Users/leventeszabo/mamba-capital/data/forex_cleaned",
        help="Directory containing processed_*_hourly.csv files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="/Users/leventeszabo/mamba-capital/mamba-model/datasets/fx_mamba_v1",
        help="Directory to write .pt datasets and metadata",
    )
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--train_stride", type=int, default=4)
    parser.add_argument("--eval_stride", type=int, default=1)
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of symbols to include (e.g., EURUSD AUDUSD)",
    )
    parser.add_argument(
        "--no_standardize_targets",
        action="store_true",
        help="If set, do not standardize forward targets",
    )
    # Defaults favoring larger train, compact validation (Q4 2024), and test (2025+)
    parser.add_argument("--train_end", type=str, default="2024-09-30 23:00:00+00:00")
    parser.add_argument("--val_start", type=str, default="2024-10-01 00:00:00+00:00")
    parser.add_argument("--val_end", type=str, default="2024-12-31 23:00:00+00:00")
    parser.add_argument("--test_start", type=str, default="2025-01-01 00:00:00+00:00")
    parser.add_argument(
        "--test_end",
        type=str,
        default=None,
        help="Optional end datetime for test; if omitted, runs to end of data",
    )

    args = parser.parse_args()

    splits = TimeSplits(
        train_start=None,
        train_end=pd.to_datetime(args.train_end, utc=True),
        val_start=pd.to_datetime(args.val_start, utc=True),
        val_end=pd.to_datetime(args.val_end, utc=True),
        test_start=pd.to_datetime(args.test_start, utc=True),
        test_end=pd.to_datetime(args.test_end, utc=True) if args.test_end else None,
    )

    cfg = BuildConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        sequence_length=int(args.sequence_length),
        train_stride=int(args.train_stride),
        eval_stride=int(args.eval_stride),
        symbols=args.symbols,
        standardize_targets=not bool(args.no_standardize_targets),
        splits=splits,
    )
    return cfg


def main(cfg: BuildConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    data = load_cleaned_forex(cfg.input_dir, cfg.symbols)

    # 2) Compute forward targets
    data = compute_forward_returns_per_symbol(data, cfg.horizons)

    # 3) Add derived continuous features
    data, continuous_features = add_derived_features(data)

    # 4) Build vocabularies (categoricals)
    vocab = build_vocabularies(data)
    symbol_to_id: Dict[str, int] = vocab["symbol_to_id"]  # type: ignore

    # 5) Assign splits
    data = assign_splits(data, cfg.splits)

    # 6) Fit and apply per-symbol scalers (continuous features)
    feature_scalers = fit_per_symbol_scalers(data, continuous_features)
    data_scaled = apply_per_symbol_scalers(data, continuous_features, feature_scalers)

    # 7) Optionally standardize targets per symbol using train stats
    target_scalers: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    if cfg.standardize_targets:
        target_scalers = fit_target_scalers(data_scaled, cfg.horizons)
        data_scaled = apply_target_scalers(data_scaled, cfg.horizons, target_scalers)

    # 8) Windowize into tensors per split
    datasets, metas = build_windows(
        data=data_scaled,
        vocab=vocab,
        continuous_features=continuous_features,
        horizons=cfg.horizons,
        sequence_length=cfg.sequence_length,
        train_stride=cfg.train_stride,
        eval_stride=cfg.eval_stride,
    )

    # Print window counts per split for sanity check
    def _count_windows(d: Dict[str, torch.Tensor]) -> int:
        return int(d["y"].shape[0]) if "y" in d else 0

    counts = {split: _count_windows(ds) for split, ds in datasets.items()}
    print(
        f"Window counts by split -> train: {counts.get('train', 0)}, val: {counts.get('val', 0)}, test: {counts.get('test', 0)}"
    )

    # 9) Save tensors and metadata
    for split_name, tensors in datasets.items():
        out_path = cfg.output_dir / f"dataset_{split_name}.pt"
        torch.save(tensors, out_path)

    # Save meta
    meta = {
        "feature_names": continuous_features,
        "horizons": list(cfg.horizons),
        "sequence_length": cfg.sequence_length,
        "train_stride": cfg.train_stride,
        "eval_stride": cfg.eval_stride,
        "vocab": vocab,
        "feature_scalers": feature_scalers,
        "target_scalers": target_scalers,
        "splits": {
            "train_end": cfg.splits.train_end.isoformat(),
            "val_start": cfg.splits.val_start.isoformat(),
            "val_end": cfg.splits.val_end.isoformat(),
            "test_start": cfg.splits.test_start.isoformat(),
            "test_end": (
                cfg.splits.test_end.isoformat() if cfg.splits.test_end else None
            ),
        },
        "samples": {split: metas[split] for split in ["train", "val", "test"]},
    }

    # Save metadata both as .pt and .json for convenience
    torch.save(meta, cfg.output_dir / "meta.pt")
    with open(cfg.output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved datasets to:")
    for split_name in ["train", "val", "test"]:
        print(f"  - {cfg.output_dir / f'dataset_{split_name}.pt'}")
    print(f"Saved metadata to: {cfg.output_dir / 'meta.pt'} and meta.json")


if __name__ == "__main__":
    config = parse_args()
    main(config)
