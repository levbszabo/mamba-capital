"""
Export per-window latents and forecasts (mu, sigma) from a trained forecaster.

Saves:
  latents_train.pt / latents_val.pt / latents_test.pt
  Each contains: {"z": [N,D], "mu": [N,H], "sigma": [N,H], "meta": meta_list}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ForecastingModel, ModelConfig


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


def parse_args():
    p = argparse.ArgumentParser(description="Export latents and forecasts")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, meta = load_tensors(Path(args.dataset_dir))
    vocab = meta["vocab"]
    feature_names = meta["feature_names"]
    horizons = meta["horizons"]

    cfg = ModelConfig(
        num_cont_features=len(feature_names),
        num_symbols=int(vocab["num_symbols"]),
        hour_size=int(vocab["hour_size"]),
        dow_size=int(vocab["dow_size"]),
        num_bases=int(vocab.get("num_bases", 0)) or None,
        num_quotes=int(vocab.get("num_quotes", 0)) or None,
        num_horizons=len(horizons),
    )
    model = ForecastingModel(cfg).to(device)
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt["model_state"])  # type: ignore
    model.eval()

    for split in ["train", "val", "test"]:
        loader = make_loader(data[split], args.batch_size)
        Z_list = []
        MU_list = []
        SIG_list = []
        for batch in loader:
            if len(batch) == 7:
                x_cont, x_sym, x_hour, x_dow, x_base, x_quote, _ = batch
            else:
                x_cont, x_sym, x_hour, x_dow, _ = batch
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
            z = model.encode(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)
            mu, log_var = model(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)
            sigma = torch.exp(0.5 * log_var)
            Z_list.append(z.cpu())
            MU_list.append(mu.cpu())
            SIG_list.append(sigma.cpu())
        Z = torch.cat(Z_list, dim=0)
        MU = torch.cat(MU_list, dim=0)
        SIG = torch.cat(SIG_list, dim=0)
        out = {"z": Z, "mu": MU, "sigma": SIG, "meta": meta["samples"][split]}
        torch.save(out, Path(args.dataset_dir) / f"latents_{split}.pt")
        print(f"Saved {split} latents: z={tuple(Z.shape)}, mu={tuple(MU.shape)}")


if __name__ == "__main__":
    main()
