"""Eval layer: judge WM rollouts (policy fail vs model lie vs abstain)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from robot_gym.world_model_toy import (
    BIN_SLICE,
    ConfusionMetrics,
    feature_inconsistency,
    in_bin,
    mse,
    rollout_open_loop,
    run_mujoco_episode,
    TOY_SLICE,
    train_world_model,
    collect_episodes,
    flatten_transitions,
    ResidualWorldModel,
)

FEATURE_DIM = 10


class Verdict(str, Enum):
    ABSTAIN = "abstain"
    POLICY_SUCCESS = "policy_success"
    POLICY_FAIL = "policy_fail"


@dataclass
class EvalConfig:
    inconsistency_max: float = 0.12
    max_inconsistency_max: float = 0.25
    trust_min: float = 0.5
    model_error_max: float | None = None  # oracle only (calibration / toy debug)
    use_trust_head: bool = True


@dataclass
class EvalSignals:
    wm_success: bool
    inconsistency_end: float
    inconsistency_max: float
    inconsistency_mean: float
    model_error_m: float | None
    trust: float | None
    dist_bin_xy: float
    toy_z: float


@dataclass
class EvalResult:
    verdict: Verdict
    signals: EvalSignals
    abstain_reasons: list[str] = field(default_factory=list)
    should_mint_synthetic_data: bool = False


def wm_trust_features(wm_traj: np.ndarray) -> np.ndarray:
    inc = [feature_inconsistency(s) for s in wm_traj]
    last = wm_traj[-1]
    toy = last[TOY_SLICE]
    bin_p = last[BIN_SLICE]
    dist_bin_xy = float(np.linalg.norm(toy[:2] - bin_p[:2]))
    toy_dz = float(toy[2] - wm_traj[0][TOY_SLICE][2])
    toy_move = float(np.linalg.norm(toy - wm_traj[0][TOY_SLICE]))
    return np.array(
        [
            inc[-1],
            max(inc),
            float(np.mean(inc)),
            dist_bin_xy,
            float(toy[2]),
            float(last[14]),
            len(wm_traj),
            toy_dz,
            toy_move,
            1.0 if in_bin(last) else 0.0,
        ],
        dtype=np.float32,
    )


class TrustHead(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EvalLayer:
    """One Robot–style eval: WM proposes outcome; layer decides trust + action."""

    def __init__(
        self,
        config: EvalConfig = EvalConfig(),
        trust_head: TrustHead | None = None,
    ):
        self.config = config
        self.trust_head = trust_head

    def signals_from_traj(
        self,
        wm_traj: np.ndarray,
        real_obs_end: np.ndarray | None = None,
    ) -> EvalSignals:
        inc = [feature_inconsistency(s) for s in wm_traj]
        last = wm_traj[-1]
        toy = last[TOY_SLICE]
        bin_p = last[BIN_SLICE]
        model_err = None
        if real_obs_end is not None:
            model_err = float(np.linalg.norm(last[TOY_SLICE] - real_obs_end[TOY_SLICE]))
        trust = None
        if self.trust_head is not None and self.config.use_trust_head:
            feats = torch.from_numpy(wm_trust_features(wm_traj)).unsqueeze(0)
            self.trust_head.eval()
            with torch.no_grad():
                trust = float(torch.sigmoid(self.trust_head(feats)).item())
        return EvalSignals(
            wm_success=in_bin(last),
            inconsistency_end=inc[-1],
            inconsistency_max=max(inc),
            inconsistency_mean=float(np.mean(inc)),
            model_error_m=model_err,
            trust=trust,
            dist_bin_xy=float(np.linalg.norm(toy[:2] - bin_p[:2])),
            toy_z=float(toy[2]),
        )

    def judge(
        self,
        wm_traj: np.ndarray,
        real_obs_end: np.ndarray | None = None,
    ) -> EvalResult:
        sig = self.signals_from_traj(wm_traj, real_obs_end)
        reasons: list[str] = []
        cfg = self.config

        if sig.inconsistency_end > cfg.inconsistency_max:
            reasons.append(f"inc_end={sig.inconsistency_end:.3f}>{cfg.inconsistency_max}")
        if sig.inconsistency_max > cfg.max_inconsistency_max:
            reasons.append(f"inc_max={sig.inconsistency_max:.3f}>{cfg.max_inconsistency_max}")
        if cfg.model_error_max is not None and sig.model_error_m is not None:
            if sig.model_error_m > cfg.model_error_max:
                reasons.append(f"model_err={sig.model_error_m:.3f}>{cfg.model_error_max}")
        if cfg.use_trust_head and self.trust_head is not None and sig.trust is not None:
            if sig.trust < cfg.trust_min:
                reasons.append(f"trust={sig.trust:.3f}<{cfg.trust_min}")

        if reasons:
            return EvalResult(
                verdict=Verdict.ABSTAIN,
                signals=sig,
                abstain_reasons=reasons,
                should_mint_synthetic_data=False,
            )

        if sig.wm_success:
            return EvalResult(
                verdict=Verdict.POLICY_SUCCESS,
                signals=sig,
                should_mint_synthetic_data=False,
            )
        return EvalResult(
            verdict=Verdict.POLICY_FAIL,
            signals=sig,
            should_mint_synthetic_data=True,
        )

    def build_trust_dataset(
        self,
        wm_model: ResidualWorldModel,
        seeds: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for seed in seeds:
            obs, acts, real_ok = run_mujoco_episode(seed)
            wm_traj = rollout_open_loop(wm_model, obs[0], acts)
            wm_ok = in_bin(wm_traj[-1])
            xs.append(wm_trust_features(wm_traj))
            ys.append(1.0 if (real_ok == wm_ok) else 0.0)
        return np.stack(xs), np.asarray(ys, dtype=np.float32)

    def train_trust_head(
        self,
        wm_model: ResidualWorldModel,
        train_seeds: list[int],
        epochs: int = 80,
        quiet: bool = False,
    ) -> TrustHead:
        x, y = self.build_trust_dataset(wm_model, train_seeds)
        head = TrustHead(x.shape[1])
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        head.train()
        for ep in range(1, epochs + 1):
            logits = head(xt)
            loss = loss_fn(logits, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not quiet and (ep == epochs or ep % 20 == 0):
                with torch.no_grad():
                    acc = ((torch.sigmoid(logits) > 0.5) == yt).float().mean()
                print(f"  trust-head epoch {ep:03d}  loss={loss.item():.4f}  acc={acc.item():.2f}")
        self.trust_head = head
        self.config.use_trust_head = True
        return head

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "trust_state": self.trust_head.state_dict() if self.trust_head else None,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> EvalLayer:
        payload = torch.load(path, weights_only=False)
        cfg = payload["config"]
        layer = EvalLayer(config=cfg)
        if payload.get("trust_state") is not None:
            layer.trust_head = TrustHead()
            layer.trust_head.load_state_dict(payload["trust_state"])
        return layer


def confusion_from_eval_layer(
    wm_model: ResidualWorldModel,
    eval_layer: EvalLayer,
    n_eval: int,
    seed0: int,
    hold_obs: np.ndarray,
    hold_acts: np.ndarray,
    hold_nxt: np.ndarray,
) -> ConfusionMetrics:
    wm_model.eval()
    with torch.no_grad():
        hold_pred = wm_model(
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
        wm_traj = rollout_open_loop(wm_model, obs[0], acts)
        wm_ok = in_bin(wm_traj[-1])
        result = eval_layer.judge(wm_traj, real_obs_end=obs[-1])

        if real_ok and wm_ok:
            tn += 1
        elif real_ok and not wm_ok:
            fp += 1
        elif (not real_ok) and wm_ok:
            fn += 1
        else:
            tp += 1

        if result.verdict == Verdict.ABSTAIN:
            abstain += 1
        elif real_ok and not wm_ok and result.verdict == Verdict.POLICY_FAIL:
            fp_called += 1

    obs, acts, _ = run_mujoco_episode(999)
    wm_traj = rollout_open_loop(wm_model, obs[0], acts)
    if len(wm_traj) > 0:
        toy_t40 = float(np.linalg.norm(wm_traj[-1][TOY_SLICE] - obs[-1][TOY_SLICE]))

    return ConfusionMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        abstain=abstain,
        fp_called=fp_called,
        n_eval=n_eval,
        inconsistency_max=eval_layer.config.trust_min,
        model_error_max=eval_layer.config.model_error_max,
        one_step_mse=one_step,
        toy_t40_m=toy_t40,
    )


def demo_episode(
    wm_model: ResidualWorldModel,
    eval_layer: EvalLayer,
    seed: int,
    real_ok: bool | None = None,
) -> EvalResult:
    obs, acts, real_ok_env = run_mujoco_episode(seed)
    if real_ok is None:
        real_ok = real_ok_env
    wm_traj = rollout_open_loop(wm_model, obs[0], acts)
    result = eval_layer.judge(wm_traj, real_obs_end=obs[-1])
    wm_ok = in_bin(wm_traj[-1])
    tag = "TN"
    if real_ok and wm_ok:
        tag = "TN"
    elif real_ok and not wm_ok:
        tag = "FP"
    elif not real_ok and wm_ok:
        tag = "FN"
    else:
        tag = "TP"
    print(
        f"seed={seed}  real={real_ok}  wm_ok={wm_ok}  raw={tag}  "
        f"verdict={result.verdict.value}  mint={result.should_mint_synthetic_data}"
    )
    if result.abstain_reasons:
        print(f"  abstain: {', '.join(result.abstain_reasons)}")
    print(
        f"  inc_end={result.signals.inconsistency_end:.3f}  "
        f"inc_max={result.signals.inconsistency_max:.3f}  "
        f"trust={result.signals.trust}  model_err={result.signals.model_error_m}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run eval layer on WM rollouts")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-horizon", type=int, default=16)
    parser.add_argument("--trust-epochs", type=int, default=80)
    parser.add_argument("--eval-eps", type=int, default=12)
    parser.add_argument("--eval-seed0", type=int, default=10_000)
    parser.add_argument("--train-seed0", type=int, default=5000)
    parser.add_argument("--train-seed-count", type=int, default=40)
    parser.add_argument("--trust-min", type=float, default=0.5)
    parser.add_argument("--inconsistency-max", type=float, default=0.12)
    parser.add_argument("--model-error-max", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/pick_toys/eval_layer.pt"))
    args = parser.parse_args()

    print("1) Collect + train world model...")
    collected = collect_episodes(args.episodes, quiet=True)
    wm, hold_eps = train_world_model(
        collected, args.train_horizon, args.epochs, batch_size=64, quiet=True
    )
    hold_obs, hold_acts, hold_nxt = flatten_transitions(hold_eps)

    cfg = EvalConfig(
        inconsistency_max=args.inconsistency_max,
        trust_min=args.trust_min,
        model_error_max=args.model_error_max,
        use_trust_head=True,
    )
    layer = EvalLayer(config=cfg)
    train_seeds = list(range(args.train_seed0, args.train_seed0 + args.train_seed_count))
    print(f"2) Train trust head on {len(train_seeds)} episodes (labels from MuJoCo, not at inference)...")
    layer.train_trust_head(wm, train_seeds, epochs=args.trust_epochs)
    layer.save(args.out)
    print(f"   saved {args.out}")

    print("\n3) Per-episode eval (first 5 eval seeds)...")
    for seed in range(args.eval_seed0, args.eval_seed0 + min(5, args.eval_eps)):
        demo_episode(wm, layer, seed)

    print("\n4) Confusion table (eval layer gates raw WM)...")
    m = confusion_from_eval_layer(
        wm, layer, args.eval_eps, args.eval_seed0, hold_obs, hold_acts, hold_nxt
    )
    print(f"TP={m.tp} FP={m.fp} FN={m.fn} TN={m.tn}  FP%={m.fp_rate:.2f}  FN%={m.fn_rate:.2f}")
    print(f"abstain={m.abstain}/{m.n_eval}  FPcall(toxic)={m.fp_called}")
    print("POLICY_FAIL + mint=True → would generate synthetic fix data.")
    print("ABSTAIN → recapture on real robot / don't trust dream.")


if __name__ == "__main__":
    main()
