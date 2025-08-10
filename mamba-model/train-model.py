"""
Training script for Mamba-style multihorizon FX forecasting with Transformer fallback.

Loads datasets built by build_data.py and trains a ForecastingModel with Gaussian NLL.

Usage example:
  python train-model.py \
    --dataset_dir /Users/leventeszabo/mamba-capital/mamba-model/datasets/fx_mamba_v1 \
    --batch_size 32 --epochs 30 --lr 2e-4 --weight_decay 0.01
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

from model import ForecastingModel, ModelConfig, gaussian_nll, count_parameters


def load_tensors(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict]:
    data: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ["train", "val", "test"]:
        path = dataset_dir / f"dataset_{split}.pt"
        data[split] = torch.load(path, map_location="cpu")
    meta_pt = torch.load(dataset_dir / "meta.pt", map_location="cpu")
    return data, meta_pt


def make_dataloaders(
    data: Dict[str, Dict[str, torch.Tensor]], batch_size: int, num_workers: int = 0
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split, tensors in data.items():
        x_cont = tensors["x_cont"]
        x_sym = tensors["x_symbol_id"]
        x_hour = tensors["x_ny_hour_id"]
        x_dow = tensors["x_ny_dow_id"]
        y = tensors["y"]
        ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, y)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def train_one_epoch(
    model: ForecastingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    horizon_weights: torch.Tensor,
    log_interval: int = 100,
    max_steps: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    num_steps = len(loader)
    step = 0
    start_time = time.time()
    for x_cont, x_sym, x_hour, x_dow, y in loader:
        x_cont = x_cont.to(device)
        x_sym = x_sym.to(device)
        x_hour = x_hour.to(device)
        x_dow = x_dow.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        mean, log_var = model(x_cont, x_sym, x_hour, x_dow)
        loss = gaussian_nll(mean, log_var, y, horizon_weights)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_count += y.size(0)
        step += 1

        if step % max(1, log_interval) == 0 or step == num_steps:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = total_loss / max(total_count, 1)
            elapsed = time.time() - start_time
            print(
                f"train step {step}/{num_steps}: batch_loss={loss.item():.6f} avg_loss={avg_loss:.6f} "
                f"lr={lr:.6e} grad_norm={float(grad_norm):.4f} elapsed={elapsed:.1f}s"
            )

        if max_steps is not None and step >= max_steps:
            break
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: ForecastingModel,
    loader: DataLoader,
    device: torch.device,
    horizon_weights: torch.Tensor,
    log_interval: int = 0,
    max_steps: int | None = None,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_count = 0

    # Additional point metrics: RMSE per horizon
    sum_sq_err = None
    sum_abs_err = None
    correct_dir = None  # counts per horizon
    n_obs = 0

    num_steps = len(loader)
    step = 0
    for x_cont, x_sym, x_hour, x_dow, y in loader:
        x_cont = x_cont.to(device)
        x_sym = x_sym.to(device)
        x_hour = x_hour.to(device)
        x_dow = x_dow.to(device)
        y = y.to(device)
        mean, log_var = model(x_cont, x_sym, x_hour, x_dow)
        loss = gaussian_nll(mean, log_var, y, horizon_weights)
        total_loss += loss.item() * y.size(0)
        total_count += y.size(0)

        err = (mean - y).detach()
        if sum_sq_err is None:
            sum_sq_err = (err**2).sum(dim=0)
            sum_abs_err = err.abs().sum(dim=0)
            correct_dir = (
                (torch.sign(mean) == torch.sign(y)).to(torch.float32).sum(dim=0)
            )
        else:
            sum_sq_err += (err**2).sum(dim=0)
            sum_abs_err += err.abs().sum(dim=0)
            correct_dir += (
                (torch.sign(mean) == torch.sign(y)).to(torch.float32).sum(dim=0)
            )
        n_obs += err.size(0)

        step += 1
        if log_interval and (step % log_interval == 0 or step == num_steps):
            print(f"eval step {step}/{num_steps}: batch_loss={loss.item():.6f}")
        if max_steps is not None and step >= max_steps:
            break

    rmse = (
        torch.sqrt(sum_sq_err / max(n_obs, 1)).cpu().numpy().tolist()
        if sum_sq_err is not None
        else [float("nan")]
    )
    mae = (
        (sum_abs_err / max(n_obs, 1)).cpu().numpy().tolist()
        if sum_abs_err is not None
        else [float("nan")]
    )
    diracc = (
        (correct_dir / max(n_obs, 1)).cpu().numpy().tolist()
        if correct_dir is not None
        else [float("nan")]
    )
    metrics = {f"rmse_h{i}": float(v) for i, v in enumerate(rmse)}
    metrics.update({f"mae_h{i}": float(v) for i, v in enumerate(mae)})
    metrics.update({f"diracc_h{i}": float(v) for i, v in enumerate(diracc)})
    return total_loss / max(total_count, 1), metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba FX forecasting model")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        default="/Users/leventeszabo/mamba-capital/mamba-model/datasets/fx_mamba_v1",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--backbone", type=str, choices=["auto", "mamba", "transformer"], default="auto"
    )
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=1536)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="/Users/leventeszabo/mamba-capital/mamba-model/checkpoints",
    )
    parser.add_argument(
        "--horizon_weights", type=float, nargs="*", default=[1.0, 0.8, 0.6]
    )
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data, meta = load_tensors(Path(args.dataset_dir))
    feature_names = meta["feature_names"]
    vocab = meta["vocab"]
    horizons = meta["horizons"]

    cfg = ModelConfig(
        num_cont_features=len(feature_names),
        num_symbols=int(vocab["num_symbols"]),
        hour_size=int(vocab["hour_size"]),
        dow_size=int(vocab["dow_size"]),
        num_horizons=len(horizons),
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        backbone=args.backbone,
    )

    model = ForecastingModel(cfg).to(device)
    print(f"Model params: {count_parameters(model):,}")

    loaders = make_dataloaders(data, batch_size=args.batch_size)

    # Print dataset sizes and basic setup
    n_train = int(loaders["train"].dataset.tensors[-1].shape[0])  # type: ignore
    n_val = int(loaders["val"].dataset.tensors[-1].shape[0])  # type: ignore
    n_test = int(loaders["test"].dataset.tensors[-1].shape[0])  # type: ignore
    seq_len = int(meta.get("sequence_length", 0))
    print(
        f"Data: train={n_train} val={n_val} test={n_test} | seq_len={seq_len} features={len(feature_names)} horizons={len(horizons)}"
    )
    print(f"Device: {device} | Backbone: {cfg.backbone}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    horizon_weights = torch.tensor(
        args.horizon_weights, dtype=torch.float32, device=device
    )

    best_val = float("inf")
    best_path = save_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        epoch_start = time.time()
        max_train_steps = args.max_train_steps if args.max_train_steps > 0 else None
        max_eval_steps = args.max_eval_steps if args.max_eval_steps > 0 else None
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            horizon_weights,
            log_interval=args.log_interval,
            max_steps=max_train_steps,
        )
        val_loss, val_metrics = evaluate(
            model,
            loaders["val"],
            device,
            horizon_weights,
            log_interval=0,
            max_steps=max_eval_steps,
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start
        log_obj = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        print(json.dumps(log_obj))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "val_loss": best_val,
                    "epoch": epoch,
                    "meta": meta,
                },
                best_path,
            )
            print(f"Saved improved checkpoint: {best_path} (val_loss={best_val:.6f})")

    # Final evaluation on test using the best checkpoint
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    test_loss, test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        horizon_weights,
        log_interval=0,
        max_steps=None,
    )
    print(json.dumps({"test_loss": test_loss, **test_metrics}))


if __name__ == "__main__":
    main()
