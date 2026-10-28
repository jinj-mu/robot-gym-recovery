"""Toy world model: next-obs predictor vs MuJoCo (One Robot loop, low-dim).

Goal analogous to One Robot: keep open-loop (many-step) error small, not just
1-step MSE. Train by unrolling the model on its own predictions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from robot_gym.envs import PickToysFloorEnv
from robot_gym.envs.expert import ToyPickExpert

TOY_SLICE = slice(18, 21)
EE_SLICE = slice(15, 18)
BIN_SLICE = slice(21, 24)
REL_SLICE = slice(24, 27)  # toy - ee (privileged; WM should stay consistent)


class ResidualWorldModel(nn.Module):
    """Predict delta in normalized obs space: s_{t+1} = s_t + f(s_t, a_t)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("std", torch.ones(obs_dim))

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.std.copy_(torch.from_numpy(np.maximum(std.astype(np.float32), 1e-6)))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.std

    def denormalize(self, obs_n: torch.Tensor) -> torch.Tensor:
        return obs_n * self.std + self.mean

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        obs_n = self.normalize(obs)
        delta_n = self.net(torch.cat([obs_n, act], dim=-1))
        return self.denormalize(obs_n + delta_n)


class WindowDataset(Dataset):
    """Windows of (obs_0, actions[H], future_obs[H]) from episode lists."""

    def __init__(
        self,
        episodes: list[tuple[np.ndarray, np.ndarray]],
        horizon: int,
    ):
        self.horizon = horizon
        self.obs0: list[np.ndarray] = []
        self.acts: list[np.ndarray] = []
        self.future: list[np.ndarray] = []
        for obs, acts in episodes:
            t = len(acts)
            if t < horizon:
                continue
            for i in range(t - horizon + 1):
                self.obs0.append(obs[i])
                self.acts.append(acts[i : i + horizon])
                self.future.append(obs[i + 1 : i + 1 + horizon])

    def __len__(self) -> int:
        return len(self.obs0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.obs0[idx]),
            torch.from_numpy(self.acts[idx]),
            torch.from_numpy(self.future[idx]),
        )


def collect_episodes(episodes: int, quiet: bool = False) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    env = PickToysFloorEnv(toy="random")
    expert = ToyPickExpert(env)
    out: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        expert.reset()
        obs_l = [obs.copy()]
        act_l: list[np.ndarray] = []
        info: dict = {}
        for _ in range(env.max_episode_steps):
            action = expert.act(obs)
            nxt, _, terminated, truncated, info = env.step(action)
            act_l.append(action.copy())
            obs_l.append(nxt.copy())
            obs = nxt
            if terminated or truncated or expert.phase == "done":
                break
        success = bool(info.get("success"))
        if not quiet:
            print(f"collect {ep + 1}/{episodes} success={success} steps={len(act_l)}")
        out.append(
            (
                np.asarray(obs_l, dtype=np.float32),
                np.asarray(act_l, dtype=np.float32),
                success,
            )
        )
    env.close()
    return out


def rollout_open_loop(
    model: ResidualWorldModel,
    obs0: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    pred = []
    x = torch.from_numpy(obs0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for a in actions:
            a_t = torch.from_numpy(a).unsqueeze(0)
            x = model(x, a_t)
            pred.append(x.squeeze(0).numpy())
    return np.stack(pred)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def in_bin(obs: np.ndarray) -> bool:
    """Same success geometry as PickToysFloorEnv (toy in bin)."""
    toy = obs[TOY_SLICE]
    bin_p = obs[BIN_SLICE]
    dist_xy = float(np.linalg.norm(toy[:2] - bin_p[:2]))
    return dist_xy < 0.10 and 0.22 < float(toy[2]) < 0.40


def feature_inconsistency(obs: np.ndarray) -> float:
    """How much the WM broke s = concat(..., toy-ee, ...). High → don't trust a 'fail'."""
    return float(np.linalg.norm(obs[TOY_SLICE] - obs[EE_SLICE] - obs[REL_SLICE]))


def run_mujoco_episode(seed: int, toy: str = "toy_block") -> tuple[np.ndarray, np.ndarray, bool]:
    env = PickToysFloorEnv(toy=toy)
    expert = ToyPickExpert(env)
    obs, _ = env.reset(seed=seed)
    expert.reset()
    obs_l = [obs.copy()]
    act_l: list[np.ndarray] = []
    info: dict = {}
    o = obs
    for _ in range(env.max_episode_steps):
        a = expert.act(o)
        nxt, _, term, trunc, info = env.step(a)
        act_l.append(a.copy())
        obs_l.append(nxt.copy())
        o = nxt
        if term or trunc or expert.phase == "done":
            break
    env.close()
    return (
        np.asarray(obs_l, dtype=np.float32),
        np.asarray(act_l, dtype=np.float32),
        bool(info.get("success")),
    )


@dataclass
class ConfusionMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    abstain: int
    fp_called: int
    n_eval: int
    inconsistency_max: float
    model_error_max: float | None
    one_step_mse: float
    toy_t40_m: float

    @property
    def fp_rate(self) -> float:
        return self.fp / max(self.fp + self.tn, 1)

    @property
    def fn_rate(self) -> float:
        return self.fn / max(self.fn + self.tp, 1)


def evaluate_confusion(
    model: ResidualWorldModel,
    n_eval: int,
    seed0: int,
    inconsistency_max: float,
    hold_obs: np.ndarray,
    hold_acts: np.ndarray,
    hold_nxt: np.ndarray,
    eval_horizon: int = 40,
    model_error_max: float | None = None,
) -> ConfusionMetrics:
    """Eval WM vs MuJoCo. Gate abstains when dream is untrustworthy (inconsistent or drifts from truth)."""
    model.eval()
    with torch.no_grad():
        hold_pred = model(
            torch.from_numpy(hold_obs),
            torch.from_numpy(hold_acts),
        ).numpy()
    one_step = mse(hold_pred, hold_nxt)

    tp = fp = fn = tn = 0
    abstain = 0
    fp_called = 0
    toy_t40 = 0.0
    for i in range(n_eval):
        seed = seed0 + i
        obs, acts, real_ok = run_mujoco_episode(seed)
        wm_traj = rollout_open_loop(model, obs[0], acts)
        wm_ok = in_bin(wm_traj[-1])
        inconsist = feature_inconsistency(wm_traj[-1])
        model_err = float(
            np.linalg.norm(wm_traj[-1][TOY_SLICE] - obs[-1][TOY_SLICE])
        )
        gated = inconsist > inconsistency_max
        if model_error_max is not None:
            gated = gated or model_err > model_error_max
        if real_ok and wm_ok:
            tn += 1
        elif real_ok and not wm_ok:
            fp += 1
        elif (not real_ok) and wm_ok:
            fn += 1
        else:
            tp += 1
        if gated:
            abstain += 1
        elif real_ok and not wm_ok:
            fp_called += 1

    env = PickToysFloorEnv(toy="toy_block")
    expert = ToyPickExpert(env)
    obs0, _ = env.reset(seed=999)
    expert.reset()
    true_act: list[np.ndarray] = []
    true_obs: list[np.ndarray] = []
    o = obs0
    for _ in range(eval_horizon):
        a = expert.act(o)
        nxt_o, _, term, trunc, _ = env.step(a)
        true_act.append(a.copy())
        true_obs.append(nxt_o.copy())
        o = nxt_o
        if term or trunc or expert.phase == "done":
            break
    env.close()
    wm_obs = rollout_open_loop(model, obs0.astype(np.float32), np.asarray(true_act, dtype=np.float32))
    h = min(len(true_obs), len(wm_obs))
    if h > 0:
        toy_t40 = float(
            np.linalg.norm(wm_obs[h - 1, TOY_SLICE] - np.asarray(true_obs[h - 1])[TOY_SLICE])
        )

    return ConfusionMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        abstain=abstain,
        fp_called=fp_called,
        n_eval=n_eval,
        inconsistency_max=inconsistency_max,
        model_error_max=model_error_max,
        one_step_mse=one_step,
        toy_t40_m=toy_t40,
    )


def train_world_model(
    collected: list[tuple[np.ndarray, np.ndarray, bool]],
    train_horizon: int,
    epochs: int,
    batch_size: int,
    quiet: bool = False,
) -> tuple[ResidualWorldModel, list[tuple[np.ndarray, np.ndarray, bool]]]:
    n_ep = len(collected)
    split_ep = max(1, int(n_ep * 0.85))
    train_eps = [(o, a) for o, a, _ in collected[:split_ep]]
    hold_eps = collected[split_ep:]

    all_obs = np.concatenate([o for o, _, _ in collected[:split_ep]])
    mean = all_obs.mean(axis=0)
    std = all_obs.std(axis=0)

    train_h = max(1, train_horizon)
    dataset = WindowDataset(train_eps, train_h)
    if len(dataset) == 0:
        raise ValueError(f"No windows for train_horizon={train_h}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    obs_dim = collected[0][0].shape[1]
    act_dim = collected[0][1].shape[1]
    model = ResidualWorldModel(obs_dim, act_dim)
    model.set_norm(mean, std)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for obs0, acts, future in loader:
            x = obs0
            losses = []
            for t in range(train_h):
                x = model(x, acts[:, t])
                losses.append(loss_fn(x, future[:, t]))
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(obs0)
            count += len(obs0)
        if not quiet:
            print(
                f"epoch {epoch:03d}  {train_h}-step unroll mse="
                f"{total / max(count, 1):.6f}  windows={len(dataset)}"
            )
    return model, hold_eps


def confusion_and_gate(
    model: ResidualWorldModel,
    n_eval: int,
    seed0: int,
    inconsistency_max: float,
    hold_obs: np.ndarray | None = None,
    hold_acts: np.ndarray | None = None,
    hold_nxt: np.ndarray | None = None,
    model_error_max: float | None = None,
) -> ConfusionMetrics:
    """False-positive eval: WM says fail, MuJoCo succeeded."""
    if hold_obs is None or hold_acts is None or hold_nxt is None:
        hold_obs = np.zeros((1, model.mean.shape[0]), dtype=np.float32)
        hold_acts = np.zeros((1, 4), dtype=np.float32)
        hold_nxt = hold_obs

    metrics = evaluate_confusion(
        model,
        n_eval,
        seed0,
        inconsistency_max,
        hold_obs,
        hold_acts,
        hold_nxt,
        model_error_max=model_error_max,
    )
    print("\n=== false positives (One Robot main challenge, toy scale) ===")
    print("MuJoCo = real robot. World model 'fail' = last imagined state not in bin.")
    err_note = f" model_err<{model_error_max}" if model_error_max is not None else ""
    print(f"gate: inconsistency<{inconsistency_max}{err_note}")
    print(f"{'seed':>6}  real  wm   gate   inc    err    class")
    for i in range(n_eval):
        seed = seed0 + i
        obs, acts, real_ok = run_mujoco_episode(seed)
        wm_traj = rollout_open_loop(model, obs[0], acts)
        wm_ok = in_bin(wm_traj[-1])
        inconsist = feature_inconsistency(wm_traj[-1])
        model_err = float(np.linalg.norm(wm_traj[-1][TOY_SLICE] - obs[-1][TOY_SLICE]))
        gated = inconsist > inconsistency_max
        if model_error_max is not None:
            gated = gated or model_err > model_error_max
        if real_ok and wm_ok:
            tag = "TN"
        elif real_ok and not wm_ok:
            tag = "FP"
        elif (not real_ok) and wm_ok:
            tag = "FN"
        else:
            tag = "TP"
        gate_s = "abstain" if gated else "call"
        print(
            f"{seed:6d}  {str(real_ok):5} {str(wm_ok):5} {gate_s:7}  "
            f"{inconsist:.3f} {model_err:.3f}  {tag}"
        )

    print(f"\nraw: TP={metrics.tp}  FP={metrics.fp}  FN={metrics.fn}  TN={metrics.tn}")
    print(f"false positive rate P(WM fail | real success) = {metrics.fp_rate:.2f}")
    print(f"false negative rate P(WM success | real fail)  = {metrics.fn_rate:.2f}")
    print(
        f"abstain {metrics.abstain}/{n_eval}; FP still called (toxic) = {metrics.fp_called}"
    )
    print("FP = invented fail. Abstain = don't generate a fix (model lie or incoherent dream).")
    return metrics


def flatten_transitions(
    episodes: list[tuple[np.ndarray, np.ndarray, bool]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, acts, nxt = [], [], []
    for o, a, _ in episodes:
        obs.append(o[:-1])
        acts.append(a)
        nxt.append(o[1:])
    return np.concatenate(obs), np.concatenate(acts), np.concatenate(nxt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy world-model vs MuJoCo")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=40, help="Eval open-loop length")
    parser.add_argument(
        "--train-horizon",
        type=int,
        default=40,
        help="Unroll this many steps in the training loss (1 = old 1-step MSE)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-eps", type=int, default=15, help="Held-out episodes for FP/FN table")
    parser.add_argument(
        "--inconsistency-max",
        type=float,
        default=0.08,
        help="Abstain if predicted (toy-ee) disagrees with toy and ee (eval gate)",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/pick_toys/world_model_compare.npz"))
    args = parser.parse_args()

    collected = collect_episodes(args.episodes)
    train_h = max(1, args.train_horizon)
    model, hold_eps = train_world_model(
        collected, train_h, args.epochs, args.batch_size
    )
    hold_obs, hold_acts, hold_nxt = flatten_transitions(hold_eps)
    model.eval()
    with torch.no_grad():
        hold_pred = model(
            torch.from_numpy(hold_obs),
            torch.from_numpy(hold_acts),
        ).numpy()
    one_step = mse(hold_pred, hold_nxt)
    one_step_toy = mse(hold_pred[:, TOY_SLICE], hold_nxt[:, TOY_SLICE])

    env = PickToysFloorEnv(toy="toy_block")
    expert = ToyPickExpert(env)
    obs0, _ = env.reset(seed=999)
    expert.reset()
    true_obs = []
    true_act = []
    o = obs0
    for _ in range(args.horizon):
        a = expert.act(o)
        nxt_o, _, term, trunc, _ = env.step(a)
        true_obs.append(nxt_o.copy())
        true_act.append(a.copy())
        o = nxt_o
        if term or trunc or expert.phase == "done":
            break
    env.close()
    true_obs = np.asarray(true_obs, dtype=np.float32)
    true_act = np.asarray(true_act, dtype=np.float32)
    wm_obs = rollout_open_loop(model, obs0.astype(np.float32), true_act)

    h = min(len(true_obs), len(wm_obs))
    open_mse = mse(wm_obs[:h], true_obs[:h])
    toy_err = np.linalg.norm(wm_obs[:h, TOY_SLICE] - true_obs[:h, TOY_SLICE], axis=1)
    ee_err = np.linalg.norm(wm_obs[:h, EE_SLICE] - true_obs[:h, EE_SLICE], axis=1)

    print("\n=== world model (train unroll, then eval open-loop) ===")
    print(f"train unroll horizon:              {train_h}")
    print(f"1-step next-obs MSE (holdout):     {one_step:.5f}")
    print(f"1-step toy-xyz MSE (holdout):       {one_step_toy:.5f}")
    print(f"open-loop {h}-step next-obs MSE:    {open_mse:.5f}")
    print(f"open-loop toy xyz error t=1:      {toy_err[0]:.4f} m")
    print(f"open-loop toy xyz error t={h}:      {toy_err[h - 1]:.4f} m")
    print(f"open-loop EE xyz error  t={h}:      {ee_err[h - 1]:.4f} m")
    print("Same job as One Robot: keep the long rollout faithful. They do it on")
    print("wrist-cam video + contact; this is 30-d state. Metric that matters is")
    print("t=40 toy/EE error, not 1-step MSE.")

    confusion_and_gate(
        model,
        n_eval=args.eval_eps,
        seed0=10_000,
        inconsistency_max=args.inconsistency_max,
        hold_obs=hold_obs,
        hold_acts=hold_acts,
        hold_nxt=hold_nxt,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        true_obs=true_obs[:h],
        wm_obs=wm_obs[:h],
        toy_err=toy_err,
        ee_err=ee_err,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
