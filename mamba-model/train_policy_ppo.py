"""
Train a PPO policy on top of a pretrained Dreamer-style world model.

Observations: z_t (latent from encoder + RSSM posterior/prior)
Actions: continuous position in [-1, 1]
Reward (imagination): a * mu - tc * |Δa| - lambda_unc * sigma

Policy is trained purely in imagination (prior rollouts) with costs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.utils.data import DataLoader, TensorDataset

from rssm import RSSM, RSSMConfig
from world_model import ObsEncoder, ReturnHead


def load_split(dataset_dir: Path, split: str) -> Dict[str, torch.Tensor]:
    return torch.load(dataset_dir / f"dataset_{split}.pt", map_location="cpu")


class Actor(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.mean = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> Independent:
        h = self.net(z)
        mean = torch.tanh(self.mean(h))  # keep in [-1,1]
        std = torch.exp(self.log_std).clamp_min(1e-3)
        return Independent(Normal(mean, std), 1)


class Critic(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 3e-4
    train_iters: int = 10
    minibatches: int = 4
    entropy_coef: float = 0.001


def compute_gae(rews, vals, gamma, lam):
    adv = torch.zeros_like(rews)
    lastgaelam = 0
    for t in reversed(range(len(rews))):
        next_val = vals[t + 1] if t + 1 < len(vals) else 0.0
        delta = rews[t] + gamma * next_val - vals[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals[: len(adv)]
    return adv, ret


def sharpe_ratio(x: np.ndarray, eps: float = 1e-9) -> float:
    return float(x.mean() / (x.std() + eps))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-9)
    return float(dd.min())


@torch.no_grad()
def rollout_imagination(encoder, rssm, batch, device):
    # Encode to last posterior mean latent
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
    for t in range(L):
        mu_p, logv_p = rssm.prior(h)
        mu_q, logv_q = rssm.posterior(h, E[:, t, :])
        z_t = mu_q
        h = rssm.core(z_t, h, None)
    assert z_t is not None
    return z_t


def main():
    ap = argparse.ArgumentParser(description="Train PPO policy with RSSM imagination")
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--lambda_unc", type=float, default=0.0, help="uncertainty penalty in reward"
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--txn_cost_bp", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(Path(args.dataset_dir), "train")
    val = load_split(Path(args.dataset_dir), "val")

    # Models
    meta = torch.load(Path(args.dataset_dir) / "meta.pt", map_location="cpu")
    vocab = meta["vocab"]
    feature_names = meta["feature_names"]
    horizons = meta["horizons"]
    num_horizons = len(horizons)

    encoder = ObsEncoder(
        num_cont_features=len(feature_names),
        num_symbols=int(vocab["num_symbols"]),
        hour_size=int(vocab["hour_size"]),
        dow_size=int(vocab["dow_size"]),
        num_bases=int(vocab.get("num_bases", 0)) or None,
        num_quotes=int(vocab.get("num_quotes", 0)) or None,
        embed_dim=64,
        d_model=256,
        n_layers=2,
    ).to(device)
    rssm = RSSM(RSSMConfig(latent_dim=64, hidden_dim=256, stochastic=True)).to(device)
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    encoder.load_state_dict(ckpt["encoder_state"])  # type: ignore
    rssm.load_state_dict(ckpt["rssm_state"])  # type: ignore
    encoder.eval()
    rssm.eval()

    # Optional return head for reward proxy
    ret_head = None
    if "ret_head_state" in ckpt:
        ret_head = ReturnHead(latent_dim=64, num_horizons=num_horizons).to(device)
        ret_head.load_state_dict(ckpt["ret_head_state"])  # type: ignore
        ret_head.eval()

    actor = Actor(latent_dim=64).to(device)
    critic = Critic(latent_dim=64).to(device)
    cfg = PPOConfig()
    opt_pi = torch.optim.Adam(actor.parameters(), lr=cfg.pi_lr)
    opt_vf = torch.optim.Adam(critic.parameters(), lr=cfg.vf_lr)

    # DataLoaders
    def to_loader(split):
        x_cont = split["x_cont"]
        x_sym = split["x_symbol_id"]
        x_hour = split["x_ny_hour_id"]
        x_dow = split["x_ny_dow_id"]
        y = split["y"]
        x_base = split.get("x_base_id")
        x_quote = split.get("x_quote_id")
        if x_base is not None and x_quote is not None:
            ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y)
        else:
            ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, y)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    def to_eval_loader(split):
        x_cont = split["x_cont"]
        x_sym = split["x_symbol_id"]
        x_hour = split["x_ny_hour_id"]
        x_dow = split["x_ny_dow_id"]
        y = split["y"]
        x_base = split.get("x_base_id")
        x_quote = split.get("x_quote_id")
        if x_base is not None and x_quote is not None:
            ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y)
        else:
            ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, y)
        return DataLoader(ds, batch_size=256, shuffle=False, pin_memory=True)

    train_loader = to_loader(train)
    val_loader = to_loader(val)
    val_eval_loader = to_eval_loader(val)

    bp = args.txn_cost_bp * 1e-4

    for epoch in range(1, args.epochs + 1):
        # Collect imagined trajectories
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        entropy_list = []
        for batch in train_loader:
            z = rollout_imagination(encoder, rssm, batch, device)
            # rollout in latent space with policy
            h = None
            zs = [z]
            acts = []
            rews = []
            vals = []
            for t in range(args.horizon):
                dist = actor(z)
                a = dist.sample().clamp(-1.0, 1.0)
                logp = dist.log_prob(a)
                entropy_list.append(dist.entropy().mean().detach())
                # reward: head prediction (use first horizon) minus turnover cost
                # Predict reward proxy using return head if available in checkpoint
                if ret_head is not None:
                    mu, logv = ret_head(z)
                    mu = mu[:, 0:1].detach()
                    sig = torch.exp(0.5 * logv[:, 0:1]).detach()
                else:
                    mu = torch.zeros_like(a)
                    sig = torch.zeros_like(a)
                cost = (a - (acts[-1] if acts else torch.zeros_like(a))).abs() * bp
                r = (a * mu - cost - args.lambda_unc * sig).squeeze(-1)
                v = critic(z)
                # RSSM prior step
                mu_p, logv_p = rssm.prior(
                    h
                    if h is not None
                    else torch.zeros(z.size(0), rssm.cfg.hidden_dim, device=device)
                )
                z = mu_p  # use mean for stability
                h = rssm.core(
                    z,
                    (
                        h
                        if h is not None
                        else torch.zeros(z.size(0), rssm.cfg.hidden_dim, device=device)
                    ),
                )
                # store
                zs.append(z)
                acts.append(a)
                rews.append(r)
                vals.append(v)
                logp_buf.append(logp.detach())
            obs_buf.append(torch.stack(zs[:-1], dim=1))
            act_buf.append(torch.stack(acts, dim=1))
            rew_buf.append(torch.stack(rews, dim=1))
            val_buf.append(torch.stack(vals, dim=1))

        obs = torch.cat(obs_buf, dim=0)
        act = torch.cat(act_buf, dim=0)
        rew = torch.cat(rew_buf, dim=0)
        val = torch.cat(val_buf, dim=0)
        logp_old = torch.cat(logp_buf, dim=0)

        # Compute GAE
        B, T, _ = obs.shape
        adv_list = []
        ret_list = []
        for b in range(B):
            adv_b, ret_b = compute_gae(rew[b], val[b], cfg.gamma, cfg.lam)
            adv_list.append(adv_b)
            ret_list.append(ret_b)
        adv = torch.stack(adv_list)
        ret = torch.stack(ret_list)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten for updates
        obs_f = obs.reshape(B * T, -1)
        act_f = act.reshape(B * T, -1)
        adv_f = adv.reshape(B * T)
        ret_f = ret.reshape(B * T)
        logp_old_f = logp_old.reshape(B * T)

        # PPO updates
        idx = torch.randperm(obs_f.size(0))
        nb = cfg.minibatches
        mb_size = obs_f.size(0) // nb
        clip_fracs = []
        pi_losses = []
        v_losses = []
        for _ in range(cfg.train_iters):
            for i in range(nb):
                j = idx[i * mb_size : (i + 1) * mb_size]
                z_j = obs_f[j].to(device)
                a_j = act_f[j].to(device)
                adv_j = adv_f[j].to(device)
                ret_j = ret_f[j].to(device)
                logp_old_j = logp_old_f[j].to(device)

                dist = actor(z_j)
                logp = dist.log_prob(a_j)
                ratio = torch.exp(logp - logp_old_j)
                clip = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                pi_loss = -(
                    torch.min(ratio * adv_j, clip * adv_j).mean()
                    + cfg.entropy_coef * dist.entropy().mean()
                )
                v_pred = critic(z_j)
                v_loss = F.mse_loss(v_pred, ret_j)

                # Metrics
                clipped = (ratio < (1.0 - cfg.clip_ratio)) | (
                    ratio > (1.0 + cfg.clip_ratio)
                )
                clip_fracs.append(float(clipped.float().mean().cpu()))
                pi_losses.append(float(pi_loss.detach().cpu()))
                v_losses.append(float(v_loss.detach().cpu()))

                opt_pi.zero_grad(set_to_none=True)
                pi_loss.backward()
                opt_pi.step()

                opt_vf.zero_grad(set_to_none=True)
                v_loss.backward()
                opt_vf.step()

        # Training collection metrics
        avg_rew = float(rew.mean().cpu())
        avg_abs_a = float(act.abs().mean().cpu())
        # Turnover per trajectory
        da = act[:, 1:, :] - act[:, :-1, :]
        avg_turnover = float(da.abs().mean().cpu())
        ent_mean = (
            float(torch.stack(entropy_list).mean().cpu()) if entropy_list else 0.0
        )
        print(
            {
                "epoch": epoch,
                "train_samples": int(B * T),
                "avg_reward": avg_rew,
                "avg_action_abs": avg_abs_a,
                "avg_turnover": avg_turnover,
                "pi_loss": float(np.mean(pi_losses)) if pi_losses else 0.0,
                "v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
                "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
                "entropy": ent_mean,
            }
        )

        # Validation: actor vs baseline (sign(mu)) on real windows (first horizon)
        with torch.no_grad():
            # Build sequences in dataset order
            actor_pos = []
            base_pos = []
            y_true = []
            for xb in val_eval_loader:
                if len(xb) == 7:
                    x_cont, x_sym, x_hour, x_dow, x_base, x_quote, yb = [
                        t.to(device) for t in xb
                    ]
                else:
                    x_cont, x_sym, x_hour, x_dow, yb = [t.to(device) for t in xb]
                    x_base = None
                    x_quote = None
                E = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)
                Bv, Lv, _ = E.shape
                h_v = torch.zeros(Bv, rssm.cfg.hidden_dim, device=device, dtype=E.dtype)
                z_v = None
                for t in range(Lv):
                    mu_p, logv_p = rssm.prior(h_v)
                    mu_q, logv_q = rssm.posterior(h_v, E[:, t, :])
                    z_v = mu_q
                    h_v = rssm.core(z_v, h_v, None)
                assert z_v is not None
                dist_eval = actor(z_v)
                a_mean = dist_eval.mean.squeeze(-1)
                actor_pos.append(a_mean.cpu())
                if ret_head is not None:
                    mu_eval, _ = ret_head(z_v)
                    base_pos.append(torch.sign(mu_eval[:, 0]).cpu())
                y_true.append(yb[:, 0].cpu())
            actor_pos = torch.cat(actor_pos).numpy()
            y_true = torch.cat(y_true).numpy()
            # Costs-aware PnL for actor
            cost = np.abs(np.diff(actor_pos, prepend=0)) * (args.txn_cost_bp * 1e-4)
            pnl = actor_pos * y_true - cost
            equity = np.cumsum(pnl)
            print(
                {
                    "epoch": epoch,
                    "actor_sharpe": sharpe_ratio(pnl),
                    "actor_mdd": max_drawdown(equity),
                    "actor_final_equity": float(equity[-1]) if len(equity) else 0.0,
                }
            )
            if ret_head is not None:
                base_pos = torch.cat(base_pos).numpy()
                base_cost = np.abs(np.diff(base_pos, prepend=0)) * (
                    args.txn_cost_bp * 1e-4
                )
                base_pnl = base_pos * y_true - base_cost
                base_equity = np.cumsum(base_pnl)
                print(
                    {
                        "epoch": epoch,
                        "baseline_sign_mu_sharpe": sharpe_ratio(base_pnl),
                        "baseline_sign_mu_mdd": max_drawdown(base_equity),
                        "baseline_sign_mu_final_equity": (
                            float(base_equity[-1]) if len(base_equity) else 0.0
                        ),
                    }
                )


if __name__ == "__main__":
    main()
