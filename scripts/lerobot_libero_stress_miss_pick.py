#!/usr/bin/env python3
"""Stress LeaveEmptyDeployableCritic (+ optional detect→retry recover).

Detect: action-replay all GT leave_empty fails + held_but_no_place + success sample.
GT (GraspRetryController) is for scoring only — not inside the critic.

Recover: same leave_empty trajs through deployable harness (cam+EE retry → π₀).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT_GYM_ROOT / "scripts"))

from lerobot_libero_collect import load_policy_bundle  # noqa: E402
from lerobot_libero_deployable_critic import LeaveEmptyDeployableCritic  # noqa: E402
from lerobot_libero_deployable_harness import run_detect_retry  # noqa: E402
from lerobot_libero_grasp_retry import GraspRetryController  # noqa: E402
from lerobot_libero_gt_label_fails import load_actions, make_env  # noqa: E402

ATLAS_DEFAULT = (
    ROBOT_GYM_ROOT
    / "outputs/libero_inference/paper_oracle_v1/failure_atlas_gt.json"
)
RUNS_DEFAULT = (
    ROBOT_GYM_ROOT
    / "outputs/libero_inference/paper_oracle_v1/pi0_baseline/runs.csv"
)
OUT_DEFAULT = (
    ROBOT_GYM_ROOT
    / "outputs/libero_inference/paper_oracle_v1/miss_pick_stress"
)


def _load_atlas(path: Path) -> list[dict]:
    return list(json.loads(path.read_text())["episodes"])


def _load_success_sample(runs_csv: Path, n: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader(runs_csv.open()))
    oks = [r for r in rows if str(r["success"]).lower() in ("true", "1")]
    rng = np.random.default_rng(seed)
    if n < len(oks):
        idx = rng.choice(len(oks), size=n, replace=False)
        oks = [oks[i] for i in sorted(idx)]
    out = []
    for r in oks:
        out.append(
            {
                "task_id": int(r["task_id"]),
                "episode_index": int(r["episode_index"]),
                "seed": int(r["seed"]),
                "trajectory_csv": r["trajectory_csv"],
                "mode_gt": "success",
                "first_leave_empty_step": None,
                "ever_held": True,
                "task_description": r.get("task_description", ""),
            }
        )
    return out


def eval_detect(ep: dict) -> dict:
    traj = Path(ep["trajectory_csv"])
    actions = load_actions(traj)
    task_id = int(ep["task_id"])
    seed = int(ep["seed"])

    env = make_env("libero_object", task_id)
    obs, _ = env.reset(seed=seed)
    gt = GraspRetryController(env)
    critic = LeaveEmptyDeployableCritic()
    critic.reset()

    fire_step = None
    gt_leave_step = None
    leave_streak = 0
    ever_gt_hold = False
    fire_while_holding = False
    n_fire = 0

    for step, action in enumerate(actions):
        image = obs["pixels"]["image"]
        ee = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        grip = np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float64)

        fired = critic.step(image=image, ee_pos=ee, grip_qpos=grip)
        holding = bool(gt.holding())
        leaving = bool(gt.leaving())
        ever_gt_hold = ever_gt_hold or holding

        if leaving and not holding:
            leave_streak += 1
        else:
            leave_streak = 0
        if gt_leave_step is None and leave_streak >= 8:
            gt_leave_step = step + 1

        if fired:
            n_fire += 1
            if fire_step is None:
                fire_step = step + 1
            if holding:
                fire_while_holding = True

        obs, _, _, _, info = env.step(action.astype(np.float32))
        if bool(info.get("is_success", False)) and fire_step is None:
            # Stop shortly after success if never fired.
            if step > 5:
                break
        if fire_step is not None and step >= fire_step + 20:
            break

    env.close()

    mode = str(ep.get("mode_gt", "unknown"))
    fired = fire_step is not None
    # Scoring relative to paper miss-pick (= GT leave_empty).
    if mode == "leave_empty":
        detect_label = "TP" if fired else "FN"
    elif mode == "success":
        detect_label = "FP" if fired else "TN"
    elif mode == "held_but_no_place":
        # Should not fire miss-pick while holding / after hold without drop.
        detect_label = "FP" if fired else "TN"
    else:
        detect_label = "other"

    return {
        "task_id": task_id,
        "episode_index": int(ep["episode_index"]),
        "seed": seed,
        "mode_gt": mode,
        "atlas_leave_step": ep.get("first_leave_empty_step"),
        "traj": str(traj),
        "deployable_fire_step": fire_step,
        "gt_leave_empty_step": gt_leave_step,
        "n_fire_steps": n_fire,
        "ever_gt_hold": ever_gt_hold,
        "fire_while_holding": fire_while_holding,
        "detect_label": detect_label,
        "delta_fire_minus_gt": (
            None
            if fire_step is None or gt_leave_step is None
            else int(fire_step) - int(gt_leave_step)
        ),
    }


def eval_recover(ep: dict, *, policy_bundle, max_retries: int) -> dict:
    traj = Path(ep["trajectory_csv"])
    actions = load_actions(traj)
    name = f"t{ep['task_id']}_ep{ep['episode_index']}_s{ep['seed']}"
    print(f"\n=== RECOVER {name} ({ep.get('mode_gt')}) ===", flush=True)
    result = run_detect_retry(
        name=name,
        task_id=int(ep["task_id"]),
        seed=int(ep["seed"]),
        actions=actions,
        policy_bundle=policy_bundle,
        enable_collide=False,
        max_retries=max_retries,
        max_steps=2500,
        record_frames=False,
    )
    frames = result.pop("frames", None)
    del frames
    return {
        "task_id": int(ep["task_id"]),
        "episode_index": int(ep["episode_index"]),
        "seed": int(ep["seed"]),
        "mode_gt": ep.get("mode_gt"),
        "success": bool(result.get("success")),
        "n_retries": int(result.get("n_retries", 0)),
        "fire_steps": list(result.get("fire_steps", [])),
        "handback_steps": list(result.get("handback_steps", [])),
        "n_frames": int(result.get("n_frames", 0)),
    }


def _summarize_detect(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["detect_label"], 0)
        by[r["detect_label"]] += 1
    leave = [r for r in rows if r["mode_gt"] == "leave_empty"]
    succ = [r for r in rows if r["mode_gt"] == "success"]
    held = [r for r in rows if r["mode_gt"] == "held_but_no_place"]
    tp = sum(1 for r in leave if r["detect_label"] == "TP")
    fn = sum(1 for r in leave if r["detect_label"] == "FN")
    fp_s = sum(1 for r in succ if r["detect_label"] == "FP")
    tn_s = sum(1 for r in succ if r["detect_label"] == "TN")
    fp_h = sum(1 for r in held if r["detect_label"] == "FP")
    deltas = [r["delta_fire_minus_gt"] for r in leave if r["delta_fire_minus_gt"] is not None]
    return {
        "counts_by_label": by,
        "leave_empty": {
            "n": len(leave),
            "recall": (tp / len(leave)) if leave else None,
            "TP": tp,
            "FN": fn,
            "mean_delta_fire_minus_gt": float(np.mean(deltas)) if deltas else None,
            "median_delta_fire_minus_gt": float(np.median(deltas)) if deltas else None,
        },
        "success_negatives": {
            "n": len(succ),
            "false_positive_rate": (fp_s / len(succ)) if succ else None,
            "FP": fp_s,
            "TN": tn_s,
        },
        "held_but_no_place": {
            "n": len(held),
            "false_positive_rate": (fp_h / len(held)) if held else None,
            "FP": fp_h,
            "TN": len(held) - fp_h,
        },
        "fire_while_gt_holding_any": sum(1 for r in rows if r["fire_while_holding"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, default=ATLAS_DEFAULT)
    parser.add_argument("--runs", type=Path, default=RUNS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--n-success", type=int, default=20)
    parser.add_argument("--success-sample-seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("detect", "recover", "both"),
        default="both",
    )
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--policy", default="lerobot/pi0_libero_finetuned")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--recover-only-leave-empty",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds to recover (default: all leave_empty)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    atlas = _load_atlas(args.atlas)
    if args.seeds.strip():
        want = {int(s) for s in args.seeds.split(",") if s.strip()}
        atlas = [e for e in atlas if int(e["seed"]) in want]
    successes = _load_success_sample(args.runs, args.n_success, args.success_sample_seed)
    detect_eps = atlas + successes

    detect_rows: list[dict] = []
    if args.mode in ("detect", "both"):
        print(f"=== DETECT stress: {len(atlas)} atlas + {len(successes)} success ===", flush=True)
        for i, ep in enumerate(detect_eps):
            tag = f"t{ep['task_id']}_ep{ep['episode_index']}_s{ep['seed']}"
            print(f"[{i+1}/{len(detect_eps)}] detect {tag} mode={ep.get('mode_gt')}", flush=True)
            row = eval_detect(ep)
            print(
                f"  → {row['detect_label']} fire@{row['deployable_fire_step']} "
                f"gt_leave@{row['gt_leave_empty_step']} "
                f"fire_while_hold={row['fire_while_holding']}",
                flush=True,
            )
            detect_rows.append(row)
        detect_summary = _summarize_detect(detect_rows)
        detect_path = args.output_dir / "detect_results.json"
        detect_path.write_text(
            json.dumps({"summary": detect_summary, "episodes": detect_rows}, indent=2)
        )
        print("\n=== DETECT SUMMARY ===", flush=True)
        print(json.dumps(detect_summary, indent=2), flush=True)
        print(f"wrote {detect_path}", flush=True)

    recover_rows: list[dict] = []
    if args.mode in ("recover", "both"):
        leave_eps = [e for e in atlas if e.get("mode_gt") == "leave_empty"]
        print(
            f"\n=== RECOVER stress: {len(leave_eps)} leave_empty (load π₀ once) ===",
            flush=True,
        )
        # Policy task_id is only used for task string wiring; per-ep task changes in env.
        bundle = load_policy_bundle(args.policy, args.device, "libero_object", 2)
        recover_path = args.output_dir / "recover_results.json"
        for i, ep in enumerate(leave_eps):
            print(f"[{i+1}/{len(leave_eps)}] recover…", flush=True)
            try:
                row = eval_recover(ep, policy_bundle=bundle, max_retries=args.max_retries)
            except Exception as exc:  # noqa: BLE001 — keep stress running
                row = {
                    "task_id": int(ep["task_id"]),
                    "episode_index": int(ep["episode_index"]),
                    "seed": int(ep["seed"]),
                    "mode_gt": ep.get("mode_gt"),
                    "success": False,
                    "n_retries": -1,
                    "fire_steps": [],
                    "handback_steps": [],
                    "n_frames": 0,
                    "error": repr(exc),
                }
                print(f"  → ERROR {exc!r}", flush=True)
            print(
                f"  → success={row.get('success')} retries={row.get('n_retries')} "
                f"fires={row.get('fire_steps')}",
                flush=True,
            )
            recover_rows.append(row)
            # Checkpoint after each ep so crashes don't wipe progress.
            n_ok = sum(1 for r in recover_rows if r.get("success"))
            recover_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "n": len(recover_rows),
                            "n_success": n_ok,
                            "success_rate": n_ok / len(recover_rows),
                        },
                        "episodes": recover_rows,
                    },
                    indent=2,
                )
            )
        n_ok = sum(1 for r in recover_rows if r["success"])
        recover_summary = {
            "n": len(recover_rows),
            "n_success": n_ok,
            "success_rate": (n_ok / len(recover_rows)) if recover_rows else None,
            "mean_retries_on_success": float(
                np.mean([r["n_retries"] for r in recover_rows if r["success"]])
            )
            if n_ok
            else None,
            "mean_retries_all": float(np.mean([r["n_retries"] for r in recover_rows]))
            if recover_rows
            else None,
        }
        recover_path = args.output_dir / "recover_results.json"
        recover_path.write_text(
            json.dumps({"summary": recover_summary, "episodes": recover_rows}, indent=2)
        )
        print("\n=== RECOVER SUMMARY ===", flush=True)
        print(json.dumps(recover_summary, indent=2), flush=True)
        print(f"wrote {recover_path}", flush=True)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "constraint": "critic uses cam+EE+gripper only; GT for scoring only",
                "detect": _summarize_detect(detect_rows) if detect_rows else None,
                "recover": {
                    "n": len(recover_rows),
                    "n_success": sum(1 for r in recover_rows if r["success"]),
                    "success_rate": (
                        sum(1 for r in recover_rows if r["success"]) / len(recover_rows)
                        if recover_rows
                        else None
                    ),
                }
                if recover_rows
                else None,
            },
            indent=2,
        )
    )
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
