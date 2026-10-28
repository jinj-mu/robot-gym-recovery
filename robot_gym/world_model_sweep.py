"""Sweep train horizon + eval gates; print FP/FN/abstain table only."""

from __future__ import annotations

import argparse

from robot_gym.world_model_toy import (
    ConfusionMetrics,
    collect_episodes,
    evaluate_confusion,
    flatten_transitions,
    train_world_model,
)


def print_table(title: str, rows: list[tuple[str, ConfusionMetrics]]) -> None:
    print(f"\n=== {title} ===\n")
    header = (
        f"{'config':<28} {'1-step':>8} {'toy@40':>8} "
        f"{'FP':>4} {'FN':>4} {'TN':>4} {'TP':>4} "
        f"{'FP%':>6} {'FN%':>6} {'abst':>5} {'FPcall':>6}"
    )
    print(header)
    print("-" * len(header))
    for label, m in rows:
        print(
            f"{label:<28} {m.one_step_mse:8.4f} {m.toy_t40_m:8.3f} "
            f"{m.fp:4d} {m.fn:4d} {m.tn:4d} {m.tp:4d} "
            f"{m.fp_rate:6.2f} {m.fn_rate:6.2f} "
            f"{m.abstain:5d} {m.fp_called:6d}"
        )
    print()
    print("FP% = P(WM fail | real success). abst = gate abstains. FPcall = toxic FP not blocked.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep WM training + eval gate")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--eval-eps", type=int, default=12)
    parser.add_argument("--seed0", type=int, default=10_000)
    parser.add_argument(
        "--train-horizons",
        type=int,
        nargs="+",
        default=[16],
        help="Train each model with this unroll length",
    )
    parser.add_argument(
        "--gates",
        type=float,
        nargs="+",
        default=[0.08, 0.12, 0.20],
        help="Inconsistency thresholds (internal geometry gate)",
    )
    parser.add_argument(
        "--model-errors",
        type=float,
        nargs="+",
        default=[0.08, 0.12, 0.15],
        help="Oracle gate: max toy xyz error vs MuJoCo at episode end (m)",
    )
    args = parser.parse_args()

    print(f"Collecting {args.episodes} episodes (shared across all runs)...")
    collected = collect_episodes(args.episodes, quiet=True)
    print(f"done. successes={sum(s for _, _, s in collected)}/{len(collected)}")

    inc_rows: list[tuple[str, ConfusionMetrics]] = []
    oracle_rows: list[tuple[str, ConfusionMetrics]] = []

    for train_h in args.train_horizons:
        print(f"\nTraining train_horizon={train_h} ({args.epochs} epochs)...")
        model, hold_eps = train_world_model(
            collected, train_h, args.epochs, batch_size=64, quiet=True
        )
        hold_obs, hold_acts, hold_nxt = flatten_transitions(hold_eps)

        for gate in args.gates:
            m = evaluate_confusion(
                model,
                n_eval=args.eval_eps,
                seed0=args.seed0,
                inconsistency_max=gate,
                hold_obs=hold_obs,
                hold_acts=hold_acts,
                hold_nxt=hold_nxt,
                model_error_max=None,
            )
            inc_rows.append((f"h={train_h} inc={gate:.2f}", m))

        for err in args.model_errors:
            m = evaluate_confusion(
                model,
                n_eval=args.eval_eps,
                seed0=args.seed0,
                inconsistency_max=999.0,
                hold_obs=hold_obs,
                hold_acts=hold_acts,
                hold_nxt=hold_nxt,
                model_error_max=err,
            )
            oracle_rows.append((f"h={train_h} err={err:.2f}m", m))

            m2 = evaluate_confusion(
                model,
                n_eval=args.eval_eps,
                seed0=args.seed0,
                inconsistency_max=0.12,
                hold_obs=hold_obs,
                hold_acts=hold_acts,
                hold_nxt=hold_nxt,
                model_error_max=err,
            )
            oracle_rows.append((f"h={train_h} inc+err={err:.2f}", m2))

    print_table("inconsistency gate only (no MuJoCo oracle)", inc_rows)
    print_table("oracle model-error gate (dream vs MuJoCo at end; toy only)", oracle_rows)


if __name__ == "__main__":
    main()
