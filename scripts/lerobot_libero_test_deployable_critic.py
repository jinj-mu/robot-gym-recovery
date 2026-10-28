#!/usr/bin/env python3
"""Test LeaveEmptyDeployableCritic on one baseline fail (action replay).

Compares deployable fire time vs GT leave_empty / holding from GraspRetryController
(GT used only for evaluation, not inside the critic).
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

from lerobot_libero_deployable_critic import LeaveEmptyDeployableCritic  # noqa: E402
from lerobot_libero_grasp_retry import GraspRetryController  # noqa: E402
from lerobot_libero_gt_label_fails import load_actions, make_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument(
        "--traj",
        type=Path,
        default=ROBOT_GYM_ROOT
        / "outputs/libero_inference/paper_oracle_v1/pi0_baseline/episodes"
        / "libero_object_task2_ep2/trajectory.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROBOT_GYM_ROOT
        / "outputs/libero_inference/paper_oracle_v1/deployable_critic_test_seed1202.json",
    )
    args = parser.parse_args()

    actions = load_actions(args.traj)
    env = make_env("libero_object", args.task_id)
    obs, _ = env.reset(seed=args.seed)
    gt = GraspRetryController(env)
    critic = LeaveEmptyDeployableCritic()
    critic.reset()

    rows = []
    fire_step = None
    gt_leave_step = None
    leave_streak = 0

    for step, action in enumerate(actions):
        image = obs["pixels"]["image"]
        ee = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        grip = np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float64)

        fired = critic.step(image=image, ee_pos=ee, grip_qpos=grip)
        dbg = critic.last_debug

        holding = bool(gt.holding())
        leaving = bool(gt.leaving())
        if leaving and not holding:
            leave_streak += 1
        else:
            leave_streak = 0
        if gt_leave_step is None and leave_streak >= 8:
            gt_leave_step = step + 1
        if fire_step is None and fired:
            fire_step = step + 1

        rows.append(
            {
                "step": step + 1,
                "fire": fired,
                "grasp_attempted": dbg.grasp_attempted,
                "ee_left_pick": dbg.ee_left_pick,
                "object_on_table": dbg.object_on_table,
                "grip_gap": dbg.grip_gap,
                "ee_lift": dbg.ee_lift,
                "ee_xy": dbg.ee_xy,
                "vision_mad": dbg.vision_mad,
                "gt_holding": holding,
                "gt_leaving": leaving,
            }
        )

        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        del reward, terminated, truncated, info
        if fire_step is not None and step > fire_step + 30:
            # Keep a little context after first fire, then stop early.
            break

    env.close()

    # Summaries
    first_grasp = next((r["step"] for r in rows if r["grasp_attempted"]), None)
    first_ee_left = next((r["step"] for r in rows if r["ee_left_pick"]), None)
    first_vision = next(
        (r["step"] for r in rows if r["grasp_attempted"] and r["object_on_table"]), None
    )

    out = {
        "traj": str(args.traj),
        "task_id": args.task_id,
        "seed": args.seed,
        "n_steps_logged": len(rows),
        "deployable_fire_step": fire_step,
        "gt_leave_empty_step": gt_leave_step,
        "first_grasp_attempted_step": first_grasp,
        "first_ee_left_step": first_ee_left,
        "first_object_on_table_true_step": first_vision,
        "critic_config": {
            "close_gap_thresh": critic.close_gap_thresh,
            "lift_thresh": critic.lift_thresh,
            "xy_thresh": critic.xy_thresh,
            "vision_mad_thresh": critic.vision_mad_thresh,
            "left_streak_need": critic.left_streak_need,
        },
        "note": (
            "Critic uses only pixels['image'] + eef.pos + gripper.qpos. "
            "GT holding/leaving are evaluation-only."
        ),
        "trace": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print("=== deployable leave-empty critic test ===")
    print(f"task={args.task_id} seed={args.seed}")
    print(f"grasp_attempted @ {first_grasp}")
    print(f"ee_left_pick    @ {first_ee_left}")
    print(f"object_on_table @ {first_vision} (first True after grasp)")
    print(f"DEPLOYABLE FIRE @ {fire_step}")
    print(f"GT leave_empty  @ {gt_leave_step}")
    if fire_step and gt_leave_step:
        print(f"delta (deployable - gt) = {fire_step - gt_leave_step} steps")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
