"""
Plot diagnostics for a pretrained Dreamer-style FX world model.

Generates per-split figures:
- Latent distribution checks: histogram of standardized latents, Q-Q plot, per-dim std bars, correlation heatmap
- Forecast diagnostics: scatter of predicted vs realized returns, histogram of standardized residuals, simple calibration by confidence bins
- Simple PnL equity curve using sign of predicted return (optional costs)

Usage:
  python3 mamba-model/plot_world_model_diagnostics.py \
    --dataset_dir mamba-model/datasets/fx_mamba_v1 \
    --checkpoint mamba-model/checkpoints/world_model_best.pt \
    --output_dir mamba-model/diagnostics --splits train val test \
    --batch_size 256 --max_points 50000 --txn_cost_bp 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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


def make_loader(split_tensors: Dict[str, torch.Tensor], batch_size: int) -> DataLoader:
    x_cont = split_tensors["x_cont"]
    x_sym = split_tensors["x_symbol_id"]
    x_hour = split_tensors["x_ny_hour_id"]
    x_dow = split_tensors["x_ny_dow_id"]
    y = split_tensors["y"]
    x_base = split_tensors.get("x_base_id")
    x_quote = split_tensors.get("x_quote_id")
    if x_base is not None and x_quote is not None:
        ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y)
    else:
        ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)


def collect_split_arrays(
    device: torch.device,
    loader: DataLoader,
    encoder: ObsEncoder,
    rssm: RSSM,
    head: ReturnHead,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Z_last, MU, SIG) arrays for the split.

    Z_last: [N, Dz] posterior mean latent at last step
    MU: [N, H] predicted mean per horizon
    SIG: [N, H] predicted std per horizon
    """
    Zs: List[torch.Tensor] = []
    Mus: List[torch.Tensor] = []
    Sigs: List[torch.Tensor] = []
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

        E = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)
        B, L, _ = E.shape
        h = torch.zeros(B, rssm.cfg.hidden_dim, device=device, dtype=E.dtype)
        z_t = None
        if hasattr(rssm.cfg, "stochastic") and not rssm.cfg.stochastic:
            for t in range(L):
                z_t = E[:, t, :]
                h = rssm.core(z_t, h, None)
        else:
            for t in range(L):
                mu_p, logv_p = rssm.prior(h)
                mu_q, logv_q = rssm.posterior(h, E[:, t, :])
                z_t = mu_q
                h = rssm.core(z_t, h, None)
        assert z_t is not None
        mu_y, logv_y = head(z_t)
        Zs.append(z_t.detach().cpu())
        Mus.append(mu_y.detach().cpu())
        Sigs.append(torch.exp(0.5 * logv_y).detach().cpu())

    Z = torch.cat(Zs, dim=0).numpy()
    MU = torch.cat(Mus, dim=0).numpy()
    SIG = torch.cat(Sigs, dim=0).numpy()
    return Z, MU, SIG


def qq_plot(
    ax, samples: np.ndarray, n_points: int = 10000, title: str = "QQ plot"
) -> None:
    n = samples.shape[0]
    idx = np.random.choice(n, size=min(n_points, n), replace=False)
    s = np.sort(samples[idx])
    # Theoretical quantiles for standard normal
    q = np.sort(np.random.normal(size=s.size))
    ax.plot(q, s, "o", ms=2, alpha=0.5)
    lim = max(np.max(np.abs(q)), np.max(np.abs(s)))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel("Sample quantiles")


def correlation_heatmap(ax, Z_std: np.ndarray, title: str) -> None:
    corr = np.corrcoef(Z_std.T)
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("Latent dim")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    ap = argparse.ArgumentParser(description="Plot world model diagnostics")
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="mamba-model/diagnostics")
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_points", type=int, default=50000)
    ap.add_argument("--txn_cost_bp", type=float, default=0.5)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic dynamics for diagnostics",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, meta = load_tensors(Path(args.dataset_dir))
    vocab = meta["vocab"]
    feature_names = meta["feature_names"]
    horizons: List[int] = list(meta["horizons"])  # type: ignore
    num_horizons = len(horizons)

    # Rebuild model
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    # Auto-detect deterministic checkpoints and switch mode if needed
    try:
        rssm_state = ckpt.get("rssm_state", {})
        is_det_ckpt = any(k.startswith("out_proj.") for k in rssm_state.keys())
        if is_det_ckpt:
            args.deterministic = True
    except Exception:
        pass
    cfg = ckpt.get("cfg", {})
    latent_dim = int(cfg.get("latent_dim", 64))
    hidden_dim = int(cfg.get("hidden_dim", 256))

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
    head = ReturnHead(latent_dim=latent_dim, num_horizons=num_horizons).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])  # type: ignore
    rssm.load_state_dict(ckpt["rssm_state"], strict=False)  # type: ignore
    decoder.load_state_dict(ckpt["decoder_state"])  # type: ignore
    head.load_state_dict(ckpt["ret_head_state"])  # type: ignore
    encoder.eval()
    rssm.eval()
    decoder.eval()
    head.eval()

    # Process requested splits
    for split in args.splits:
        print(f"Generating diagnostics for split: {split}")
        loader = make_loader(data[split], args.batch_size)
        # Collect arrays
        Z_last, MU, SIG = collect_split_arrays(device, loader, encoder, rssm, head)
        N, Dz = Z_last.shape
        # Limit for heavy plots
        idx = np.arange(N)
        if N > args.max_points:
            idx = np.random.choice(N, size=args.max_points, replace=False)
        Zs = Z_last[idx]
        MUs = MU[idx]
        SIGs = SIG[idx]

        # Standardize latents per-dim
        mean = Zs.mean(axis=0)
        std = Zs.std(axis=0) + 1e-8
        Z_std = (Zs - mean) / std

        # 1) Latent histogram + QQ + std bars
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        axes[0].hist(
            Z_std.flatten(), bins=100, density=True, alpha=0.7, color="#4e79a7"
        )
        axes[0].set_title(f"{split.upper()} latent standardized histogram")
        axes[0].set_xlabel("Standardized latent values")
        axes[0].set_ylabel("Density")

        qq_plot(
            axes[1],
            Z_std.flatten(),
            n_points=10000,
            title=f"{split.upper()} latent QQ (std)",
        )

        axes[2].bar(np.arange(Dz), std, color="#59a14f")
        axes[2].set_title(f"{split.upper()} latent per-dim std")
        axes[2].set_xlabel("Latent dimension")
        axes[2].set_ylabel("Std")

        correlation_heatmap(axes[3], Z_std, title=f"{split.upper()} latent correlation")
        fig.tight_layout()
        fig.savefig(out_dir / f"latent_diagnostics_{split}.png", dpi=150)
        plt.close(fig)

        # 2) Forecast diagnostics per horizon
        for h_idx, h in enumerate(horizons):
            mu = MUs[:, h_idx]
            sig = SIGs[:, h_idx]
            y = data[split]["y"][idx, h_idx].numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            # Scatter mu vs y
            axes[0].scatter(mu, y, s=6, alpha=0.4, color="#e15759")
            corr = (
                np.corrcoef(mu, y)[0, 1] if (np.std(mu) > 0 and np.std(y) > 0) else 0.0
            )
            axes[0].set_title(f"{split.upper()} μ vs y (h={h}h) corr={corr:.3f}")
            axes[0].set_xlabel("Predicted mean")
            axes[0].set_ylabel("Realized return")
            lim = np.percentile(np.abs(np.concatenate([mu, y])), 99)
            axes[0].plot([-lim, lim], [-lim, lim], "k--", lw=1)

            # Std residuals
            zres = (y - mu) / (np.maximum(sig, 1e-8))
            axes[1].hist(zres, bins=80, density=True, alpha=0.7, color="#76b7b2")
            axes[1].set_title(f"Std residuals (h={h}h)")
            axes[1].set_xlabel("(y - μ) / σ")

            # Calibration by confidence bins (|μ| or |μ/σ|)
            score = np.abs(mu / np.maximum(sig, 1e-8))
            bins = np.quantile(score, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
            xs = []
            rmses = []
            counts = []
            prev = -np.inf
            for b in bins:
                mask = (score > prev) & (score <= b)
                if mask.sum() > 0:
                    rmses.append(float(np.sqrt(np.mean((y[mask] - mu[mask]) ** 2))))
                    xs.append(float(b))
                    counts.append(int(mask.sum()))
                prev = b
            mask = score > prev
            if mask.sum() > 0:
                rmses.append(float(np.sqrt(np.mean((y[mask] - mu[mask]) ** 2))))
                xs.append(float(np.max(score)))
                counts.append(int(mask.sum()))
            axes[2].plot(xs, rmses, marker="o")
            axes[2].set_title(f"Calibration vs confidence (h={h}h)")
            axes[2].set_xlabel("confidence (|μ/σ|) quantile upper bound")
            axes[2].set_ylabel("RMSE")
            axes[2].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"forecast_diagnostics_{split}_h{h}.png", dpi=150)
            plt.close(fig)

        # 3) Simple PnL equity curve using sign(mu) on first horizon
        mu = MUs[:, 0]
        y = data[split]["y"][idx, 0].numpy()
        pos = np.sign(mu)
        cost = np.abs(np.diff(pos, prepend=0)) * (args.txn_cost_bp * 1e-4)
        pnl = pos * y - cost
        eq = np.cumsum(pnl)
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(eq, color="#4e79a7")
        ax.set_title(
            f"{split.upper()} equity curve (h={horizons[0]}h), Sharpe={np.mean(pnl)/(np.std(pnl)+1e-9):.2f}"
        )
        ax.set_xlabel("sample index")
        ax.set_ylabel("equity (arb units)")
        fig.tight_layout()
        fig.savefig(out_dir / f"equity_{split}.png", dpi=150)
        plt.close(fig)

    print(f"Saved diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
