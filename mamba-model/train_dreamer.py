"""
Lite Dreamer training: fine-tune Actor/Critic on top of a trained world model,
optionally alternating with world-model updates. Works with deterministic or
stochastic RSSM. Uses costs-aware reward from return head.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from world_model import (
    ObsEncoder,
    ObsDecoder,
    ReturnHead,
    StudentTHead,
    predict_mu_sigma,
)
from rssm import RSSM, RSSMConfig
from policy import Actor, Critic, lambda_returns


def load_tensors(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict]:
    data: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ["train", "val", "test"]:
        data[split] = torch.load(
            dataset_dir / f"dataset_{split}.pt", map_location="cpu"
        )
    meta = torch.load(dataset_dir / "meta.pt", map_location="cpu")
    return data, meta


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
        x_base = tensors.get("x_base_id")
        x_quote = tensors.get("x_quote_id")
        if x_base is not None and x_quote is not None:
            ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y)
        else:
            ds = TensorDataset(x_cont, x_sym, x_hour, x_dow, y)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=split == "train",
        )
    return loaders


def build_state(
    wm_state: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    prev_action: torch.Tensor,
    select_idx: Tuple[int, int, int],
) -> torch.Tensor:
    """state = concat([wm_state, z6, z24, z168, prev_action]).

    select_idx are horizon indices for (6h,24h,168h) inside mu/sigma.
    """
    i6, i24, i168 = select_idx
    z6 = (mu[:, i6] / (sigma[:, i6] + 1e-6)).unsqueeze(-1).detach()
    z24 = (mu[:, i24] / (sigma[:, i24] + 1e-6)).unsqueeze(-1).detach()
    z168 = (mu[:, i168] / (sigma[:, i168] + 1e-6)).unsqueeze(-1).detach()
    return torch.cat([wm_state, z6, z24, z168, prev_action], dim=-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lite Dreamer training")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--wm_checkpoint", type=str, default="")
    p.add_argument("--save_dir", type=str, default="mamba-model/checkpoints_dreamer")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--precision", type=str, choices=["fp32", "bf16", "fp16"], default="bf16"
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--likelihood", type=str, choices=["gaussian", "studentt"], default="studentt"
    )
    p.add_argument(
        "--action_space",
        type=str,
        choices=["discrete", "continuous"],
        default="discrete",
    )
    p.add_argument("--imagine_horizon", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=1e-3)
    p.add_argument("--turnover_coef", type=float, default=1e-3)
    p.add_argument("--risk_pen_sigma", type=float, default=0.2)
    p.add_argument("--txn_cost_bp", type=float, default=0.5)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--value_lr", type=float, default=3e-4)
    p.add_argument("--world_updates_per_epoch", type=int, default=0)
    p.add_argument("--rl_updates_per_epoch", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, meta = load_tensors(Path(args.dataset_dir))
    vocab = meta["vocab"]
    feature_names = meta["feature_names"]
    horizons = list(meta["horizons"])  # type: ignore
    num_horizons = len(horizons)

    # Select horizon indices roughly matching 6,24,168 when available
    def idx_of(h: int) -> int:
        return horizons.index(h) if h in horizons else 0

    select_idx = (idx_of(6), idx_of(24), idx_of(168))

    loaders = make_dataloaders(data, args.batch_size, args.num_workers)

    # Determine dims from checkpoint if available
    latent_dim = 64
    hidden_dim = 256
    ckpt = None
    if args.wm_checkpoint:
        ckpt = torch.load(Path(args.wm_checkpoint), map_location=device)
        cfg = ckpt.get("cfg", {})
        latent_dim = int(cfg.get("latent_dim", latent_dim))
        hidden_dim = int(cfg.get("hidden_dim", hidden_dim))

    # Build world model with inferred dims
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

    # Restore world model if checkpoint provided
    if ckpt is not None:
        encoder.load_state_dict(ckpt["encoder_state"])  # type: ignore
        rssm.load_state_dict(ckpt["rssm_state"], strict=False)  # type: ignore
        decoder.load_state_dict(ckpt["decoder_state"])  # type: ignore
        ret_head.load_state_dict(ckpt["ret_head_state"])  # type: ignore
        print(f"Loaded WM checkpoint: {args.wm_checkpoint}")
    encoder.eval()
    rssm.eval()
    decoder.eval()
    ret_head.eval()

    # Build Actor/Critic
    wm_state_dim = (
        rssm.cfg.hidden_dim
        if args.deterministic
        else (rssm.cfg.hidden_dim + rssm.cfg.latent_dim)
    )
    state_dim = wm_state_dim + 3 + 1  # z6,z24,z168, prev_action
    actor = Actor(state_dim, action_space=args.action_space).to(device)
    critic = Critic(state_dim).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    opt_value = torch.optim.Adam(critic.parameters(), lr=args.value_lr)

    # Precision
    autocast_dtype: Optional[torch.dtype] = None
    if device.type == "cuda":
        if args.precision == "bf16" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
            print("Using bf16 autocast (RL)")
        elif args.precision == "fp16":
            autocast_dtype = torch.float16
            print("Using fp16 autocast (RL)")

    # Training loop (RL fine-tune only)
    for epoch in range(1, args.epochs + 1):
        actor.train()
        critic.train()
        iters = 0
        for xb in loaders["train"]:
            if len(xb) == 7:
                x_cont, x_sym, x_hour, x_dow, x_base, x_quote, y = [
                    t.to(device) for t in xb
                ]
            else:
                x_cont, x_sym, x_hour, x_dow, y = [t.to(device) for t in xb]
                x_base = None
                x_quote = None
            with (
                torch.autocast("cuda", dtype=autocast_dtype)
                if autocast_dtype
                else torch.cuda.amp.autocast(enabled=False)
            ):
                # Posterior pass to get last latent and wm_state (no grads through WM)
                with torch.no_grad():
                    E = encoder(x_cont, x_sym, x_hour, x_dow, x_base, x_quote)
                    B, L, _ = E.shape
                    h = torch.zeros(
                        B, rssm.cfg.hidden_dim, device=device, dtype=E.dtype
                    )
                    z_t = None
                    if args.deterministic:
                        for t in range(L):
                            z_t = E[:, t, :]
                            h = rssm.core(z_t, h, None)
                        wm_state = h.detach()
                    else:
                        for t in range(L):
                            mu_p, logv_p = rssm.prior(h)
                            mu_q, logv_q = rssm.posterior(h, E[:, t, :])
                            z_t = mu_q
                            h = rssm.core(z_t, h, None)
                        wm_state = torch.cat([h, z_t], dim=-1).detach()  # type: ignore
                    assert z_t is not None
                    z_t = z_t.detach()
                    mu, sigma = predict_mu_sigma(ret_head, z_t, args.likelihood)
                prev_action = torch.zeros(B, 1, device=device, dtype=E.dtype)
                s_t = build_state(wm_state, mu, sigma, prev_action, select_idx)

                # Imagination rollouts (closed-loop prior)
                T = args.imagine_horizon
                states = [s_t]
                rewards = []
                values_tp1 = []
                entropies = []
                a_indices = []
                z_curr = z_t
                h_curr = h
                a_prev = prev_action
                for k in range(T):
                    # Actor step (grad flows into actor)
                    logits_or_size = actor(states[-1])
                    if args.action_space == "discrete":
                        dist = torch.distributions.Categorical(logits=logits_or_size)
                        a_idx = dist.sample()  # int in {0,1,2}
                        a = a_idx.float().unsqueeze(-1)
                        a_indices.append(a_idx)
                        a_signed = torch.tanh((a - 1.0))  # map {0,1,2} -> ~{-1,0,1}
                        entropies.append(dist.entropy())
                    else:
                        # Continuous sizing in [-1,1]
                        a_raw = logits_or_size
                        a_signed = torch.tanh(a_raw)
                        entropies.append(
                            (-(1 - a_signed.pow(2) + 1e-6).log()).mean(dim=-1)
                        )

                    # World model + reward under no_grad
                    with torch.no_grad():
                        # Reward from return head (use 6h index)
                        mu_k, sigma_k = predict_mu_sigma(
                            ret_head, z_curr, args.likelihood
                        )
                        mu6 = mu_k[:, select_idx[0]]
                        sig6 = sigma_k[:, select_idx[0]]
                        txn_cost = (args.txn_cost_bp * 1e-4) * (
                            a_signed - a_prev
                        ).abs().squeeze(-1)
                        reward = (
                            a_signed.squeeze(-1) * mu6
                            - txn_cost
                            - args.risk_pen_sigma * sig6
                        )
                        rewards.append(reward)

                        # Prior step
                        if rssm.cfg.stochastic:
                            mu_p, logv_p = rssm.prior(h_curr)
                            z_next = mu_p  # mean for stability
                            h_next = rssm.core(z_next, h_curr, None)
                        else:
                            z_next, h_next = rssm.forward_deterministic(
                                z_curr, h_curr, None
                            )

                        # Next state features
                        mu_next, sigma_next = predict_mu_sigma(
                            ret_head, z_next, args.likelihood
                        )
                        wm_next = (
                            h_next
                            if args.deterministic
                            else torch.cat([h_next, z_next], dim=-1)
                        )
                        s_next = build_state(
                            wm_next, mu_next, sigma_next, a_signed, select_idx
                        )
                        states.append(s_next)
                        z_curr, h_curr, a_prev = z_next, h_next, a_signed
                        # Critic V(s_{t+1}) (no grad)
                        values_tp1.append(critic(s_next).detach())

                # Stack tensors
                R = torch.stack(rewards, dim=0)  # [T,B]
                V_tp1 = torch.stack(values_tp1, dim=0)  # [T,B]
                targets = lambda_returns(R, V_tp1, gamma=args.gamma, lam=args.lam)
                V_pred = torch.stack([critic(s) for s in states[0:-1]], dim=0)

                # Losses
                value_loss = F.mse_loss(V_pred, targets.detach())
                if args.action_space == "discrete":
                    logits = torch.stack(
                        [actor(s) for s in states[0:-1]], dim=0
                    )  # [T,B,3]
                    dist = torch.distributions.Categorical(logits=logits)
                    entropy_mean = torch.stack(entropies, dim=0).mean()
                    a_idx_tensor = torch.stack(a_indices, dim=0)  # [T,B]
                    logp = dist.log_prob(a_idx_tensor)  # [T,B]
                    advantage = targets - V_pred.detach()  # [T,B]
                    actor_loss = (
                        -(logp * advantage.detach()).mean()
                        - args.entropy_coef * entropy_mean
                    )
                else:
                    logits = torch.stack([actor(s) for s in states[0:-1]], dim=0)
                    a_cont = torch.tanh(logits)
                    entropy_mean = (-(1 - a_cont.pow(2) + 1e-6).log()).mean()
                    # Simple deterministic PG surrogate: direct gradient on immediate value estimate
                    actor_loss = (
                        -(V_pred.detach()).mean() - args.entropy_coef * entropy_mean
                    )
                # Turnover penalty using imagined actions
                if args.action_space == "discrete":
                    # approximate turnover from logits via soft actions
                    a_soft = torch.tanh(logits.mean(dim=-1, keepdim=True) - 1.0)
                    turnover = (a_soft[1:] - a_soft[:-1]).abs().mean()
                else:
                    a_soft = torch.tanh(logits)
                    turnover = (a_soft[1:] - a_soft[:-1]).abs().mean()
                actor_loss = actor_loss + args.turnover_coef * turnover

            # Optimize
            opt_value.zero_grad(set_to_none=True)
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_value.step()

            opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt_actor.step()

            iters += 1
            if iters >= args.rl_updates_per_epoch:
                break

        print(
            {
                "epoch": epoch,
                "value_loss": float(value_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
            }
        )

    # Save actor/critic and minimal config
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "config": vars(args),
        },
        save_dir / "dreamer_policy.pt",
    )
    print(f"Saved Dreamer policy: {save_dir / 'dreamer_policy.pt'}")


if __name__ == "__main__":
    main()
