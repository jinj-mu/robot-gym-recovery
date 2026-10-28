"""Behavior-clone a toy-pickup policy from expert demos (fast, no huge RL)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int = 30, act_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC policy on pick-toys demos")
    parser.add_argument("--demos", type=Path, default=Path("runs/pick_toys/demos.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/pick_toys/bc_policy.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if not args.demos.exists():
        raise SystemExit(f"No demos at {args.demos}. Run: uv run robot-gym-pick-collect")

    data = np.load(args.demos)
    obs = torch.from_numpy(data["obs"])
    acts = torch.from_numpy(data["actions"])
    loader = DataLoader(TensorDataset(obs, acts), batch_size=args.batch_size, shuffle=True)

    policy = BCPolicy(obs_dim=obs.shape[1], act_dim=acts.shape[1])
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    policy.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for xb, yb in loader:
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        print(f"epoch {epoch:03d}  mse={total / max(n, 1):.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "obs_dim": obs.shape[1]}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
