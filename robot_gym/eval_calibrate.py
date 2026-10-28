"""Sweep eval-layer thresholds: FPcall=0 with some trusted policy_fail (mint)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from robot_gym.eval_layer import (
    EvalConfig,
    EvalLayer,
    Verdict,
    demo_episode,
)
from robot_gym.world_model_toy import (
    collect_episodes,
    in_bin,
    rollout_open_loop,
    run_mujoco_episode,
    train_world_model,
)


@dataclass
class CalibRow:
    inc_max: float
    inc_peak_max: float
    trust_min: float
    abstain: int
    policy_success: int
    policy_fail: int
    tp_mint: int
    fp_called: int
    fn_missed: int
    tn_ok: int

    @property
    def n(self) -> int:
        return self.abstain + self.policy_success + self.policy_fail


def breakdown(
    wm_model: torch.nn.Module,
    layer: EvalLayer,
    seeds: list[int],
    use_oracle: bool = False,
) -> CalibRow:
    abstain = policy_success = policy_fail = 0
    tp_mint = fp_called = fn_missed = tn_ok = 0

    for seed in seeds:
        obs, acts, real_ok = run_mujoco_episode(seed)
        wm_traj = rollout_open_loop(wm_model, obs[0], acts)
        wm_ok = in_bin(wm_traj[-1])
        real_end = obs[-1] if use_oracle else None
        result = layer.judge(wm_traj, real_obs_end=real_end)

        if result.verdict == Verdict.ABSTAIN:
            abstain += 1
        elif result.verdict == Verdict.POLICY_SUCCESS:
            policy_success += 1
        else:
            policy_fail += 1

        if real_ok and wm_ok and result.verdict == Verdict.POLICY_SUCCESS:
            tn_ok += 1
        if not real_ok and not wm_ok and result.verdict == Verdict.POLICY_FAIL:
            tp_mint += 1
        if real_ok and not wm_ok and result.verdict == Verdict.POLICY_FAIL:
            fp_called += 1
        if not real_ok and wm_ok and result.verdict == Verdict.POLICY_SUCCESS:
            fn_missed += 1

    cfg = layer.config
    return CalibRow(
        inc_max=cfg.inconsistency_max,
        inc_peak_max=cfg.max_inconsistency_max,
        trust_min=cfg.trust_min,
        abstain=abstain,
        policy_success=policy_success,
        policy_fail=policy_fail,
        tp_mint=tp_mint,
        fp_called=fp_called,
        fn_missed=fn_missed,
        tn_ok=tn_ok,
    )


def print_table(rows: list[CalibRow], title: str) -> None:
    print(f"\n=== {title} ===")
    hdr = (
        f"{'inc':>5} {'peak':>5} {'trust':>5} "
        f"{'abst':>5} {'ok':>4} {'fail':>4} "
        f"{'TPmint':>6} {'FPcall':>6} {'FNmiss':>6} {'TNok':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.inc_max:5.2f} {r.inc_peak_max:5.2f} {r.trust_min:5.2f} "
            f"{r.abstain:5d} {r.policy_success:4d} {r.policy_fail:4d} "
            f"{r.tp_mint:6d} {r.fp_called:6d} {r.fn_missed:6d} {r.tn_ok:5d}"
        )
    print("TPmint = trusted real fails (mint synthetic). FPcall = toxic (block these).")


def pick_best(rows: list[CalibRow]) -> CalibRow | None:
    safe = [r for r in rows if r.fp_called == 0]
    if not safe:
        return None
    return max(safe, key=lambda r: (r.tp_mint, r.tn_ok, -r.abstain))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate eval layer thresholds")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-horizon", type=int, default=16)
    parser.add_argument("--trust-epochs", type=int, default=80)
    parser.add_argument("--eval-seed0", type=int, default=10_000)
    parser.add_argument("--eval-eps", type=int, default=12)
    parser.add_argument("--train-seed0", type=int, default=5000)
    parser.add_argument("--train-seed-count", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("runs/pick_toys/eval_layer_calibrated.pt"))
    parser.add_argument("--use-oracle", action="store_true", help="Pass MuJoCo end state into judge")
    args = parser.parse_args()

    eval_seeds = list(range(args.eval_seed0, args.eval_seed0 + args.eval_eps))

    print("Train world model + trust head...")
    collected = collect_episodes(args.episodes, quiet=True)
    wm, _ = train_world_model(
        collected, args.train_horizon, args.epochs, batch_size=64, quiet=True
    )
    layer = EvalLayer(config=EvalConfig())
    train_seeds = list(range(args.train_seed0, args.train_seed0 + args.train_seed_count))
    layer.train_trust_head(wm, train_seeds, epochs=args.trust_epochs)

    inc_grid = [0.12, 0.18, 0.25, 0.35, 0.50]
    peak_grid = [0.25, 0.35, 0.50, 0.70]
    trust_grid = [0.30, 0.40, 0.50, 0.60]

    rows: list[CalibRow] = []
    for inc in inc_grid:
        for peak in peak_grid:
            for trust in trust_grid:
                layer.config.inconsistency_max = inc
                layer.config.max_inconsistency_max = peak
                layer.config.trust_min = trust
                layer.config.use_trust_head = True
                layer.config.model_error_max = None
                rows.append(breakdown(wm, layer, eval_seeds, use_oracle=args.use_oracle))

    print_table(rows, "eval calibration sweep (no oracle at judge)" if not args.use_oracle else "with oracle")

    best = pick_best(rows)
    if best is None:
        print("\nNo config with FPcall=0. Loosen gates or improve WM.")
        return

    layer.config.inconsistency_max = best.inc_max
    layer.config.max_inconsistency_max = best.inc_peak_max
    layer.config.trust_min = best.trust_min
    layer.save(args.out)
    print(
        f"\nRecommended: inc={best.inc_max} peak={best.inc_peak_max} trust={best.trust_min}"
    )
    print(f"  TPmint={best.tp_mint}  FPcall={best.fp_called}  abstain={best.abstain}/{best.n}")
    print(f"  saved {args.out}")

    # Demo: one FP where raw WM lied but eval should abstain or not mint
    print("\n--- demo episodes (calibrated layer) ---")
    for seed in eval_seeds[:5]:
        demo_episode(wm, layer, seed)

    # Find first raw FP seed for highlight
    for seed in eval_seeds:
        obs, acts, real_ok = run_mujoco_episode(seed)
        wm_traj = rollout_open_loop(wm, obs[0], acts)
        if real_ok and not in_bin(wm_traj[-1]):
            print("\n--- highlight: raw FP (real ok, dream fail) ---")
            demo_episode(wm, layer, seed)
            break


if __name__ == "__main__":
    main()
