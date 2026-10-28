#!/usr/bin/env python3
"""Offline GT failure labeling for baseline LIBERO fails.

Replays recorded actions with the same seed, logs privileged object/basket/EE
geometry via GraspRetryController helpers, and assigns a failure mode.

GT is used ONLY for the atlas (annotation). Deployable critics must not call
sim body_xpos at runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT_GYM_ROOT / "scripts"))

from lerobot_libero_grasp_retry import GraspRetryController, unwrap_bddl_env  # noqa: E402


def make_env(task_suite_name: str, task_id: int):
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(task_suite_name)

    class LabelEnv(LiberoEnv):
        def _ensure_env(self) -> None:
            if self._env is not None:
                return
            from libero.libero.envs import OffScreenRenderEnv

            env = OffScreenRenderEnv(
                bddl_file_name=self._task_bddl_file,
                camera_heights=self.observation_height,
                camera_widths=self.observation_width,
                control_freq=self.control_freq,
                hard_reset=self.hard_reset,
            )
            env.reset()
            self._env = env

    return LabelEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=task_suite_name,
        obs_type="pixels_agent_pos",
        observation_height=128,
        observation_width=128,
        init_states=True,
        episode_index=0,
        n_envs=1,
        control_mode="relative",
    )


def load_actions(traj_csv: Path) -> list[np.ndarray]:
    rows = list(csv.DictReader(traj_csv.open()))
    return [
        np.array([float(r[f"action_{j}"]) for j in range(7)], dtype=np.float32)
        for r in rows
    ]


def obj_in_basket(ctrl: GraspRetryController) -> bool:
    obj = ctrl.obj_pos()
    basket = ctrl.basket_pos()
    xy = float(np.linalg.norm(obj[:2] - basket[:2]))
    return xy < ctrl.place_xy_ok() and float(obj[2]) < float(ctrl.aabb_of(ctrl.basket_name)[1][2]) + 0.05


def classify(trace: list[dict], *, success: bool) -> dict:
    if success:
        return {"mode_gt": "success", "reason": "env reported success during replay"}

    held = [t["holding"] for t in trace]
    left = [t["leaving"] for t in trace]
    in_b = [t["obj_in_basket"] for t in trace]
    obj_z = [t["obj_z"] for t in trace]
    init_z = trace[0]["obj_z"] if trace else 0.0
    max_z = max(obj_z) if obj_z else init_z
    lift = max_z - init_z

    ever_held = any(held)
    first_hold = next((i for i, h in enumerate(held) if h), None)
    last_hold = next((i for i, h in enumerate(reversed(held)) if h), None)
    last_hold = (len(held) - 1 - last_hold) if last_hold is not None else None

    # sustained leave-without-object
    leave_streak = 0
    first_leave = None
    for i, (h, lv) in enumerate(zip(held, left)):
        if lv and not h:
            leave_streak += 1
            if leave_streak >= 8 and first_leave is None:
                first_leave = i - 7
        else:
            leave_streak = 0

    ended_in_basket = bool(in_b[-1]) if in_b else False
    held_near_end = any(held[-40:]) if held else False

    if ever_held:
        dropped_after = last_hold is not None and last_hold < len(held) - 20 and not held[-1]
        if dropped_after and not ended_in_basket:
            # Was holding, then lost it away from success
            near_basket_at_drop = False
            if last_hold is not None:
                near_basket_at_drop = bool(trace[last_hold]["near_basket"])
            if near_basket_at_drop or any(in_b[first_hold or 0 :]):
                mode = "place_fail_or_drop_at_basket"
            else:
                mode = "drop_in_transit"
        elif held_near_end and not ended_in_basket:
            mode = "held_but_no_place"
        else:
            mode = "held_then_lost"
    elif first_leave is not None:
        mode = "leave_empty"
    elif lift < 0.02:
        mode = "no_lift"
    else:
        mode = "other_no_hold"

    return {
        "mode_gt": mode,
        "ever_held": ever_held,
        "first_hold_step": first_hold,
        "last_hold_step": last_hold,
        "first_leave_empty_step": first_leave,
        "max_obj_lift_m": float(lift),
        "ended_in_basket": ended_in_basket,
        "n_steps_replayed": len(trace),
    }


def label_one(env, *, seed: int, traj_csv: Path) -> dict:
    actions = load_actions(traj_csv)
    env.reset(seed=seed)
    ctrl = GraspRetryController(env)
    # Silence takeover — we only use geometry helpers
    ctrl.mode = "policy"

    trace: list[dict] = []
    success = False
    for step, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        del obs, reward, terminated, truncated
        success = bool(info.get("is_success", False))
        holding = bool(ctrl.holding())
        leaving = bool(ctrl.leaving())
        in_b = bool(obj_in_basket(ctrl))
        obj = ctrl.obj_pos()
        ee = ctrl.eef_pos()
        basket = ctrl.basket_pos()
        near_basket = float(np.linalg.norm(ee[:2] - basket[:2])) < ctrl.place_xy_ok()
        trace.append(
            {
                "step": step + 1,
                "holding": holding,
                "leaving": leaving,
                "obj_in_basket": in_b,
                "near_basket": near_basket,
                "obj_z": float(obj[2]),
                "ee_z": float(ee[2]),
                "obj_xy_dist_ee": float(np.linalg.norm(obj[:2] - ee[:2])),
                "grip": float(action[6]),
            }
        )
        if success:
            break

    stats = classify(trace, success=success)
    stats.update(
        {
            "target": ctrl.target_name,
            "init_obj_z": float(ctrl.init_obj_z),
            "replay_success": success,
        }
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=ROBOT_GYM_ROOT
        / "outputs/libero_inference/paper_oracle_v1/baseline_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROBOT_GYM_ROOT
        / "outputs/libero_inference/paper_oracle_v1/failure_atlas_gt.json",
    )
    parser.add_argument("--env-task", default="libero_object")
    args = parser.parse_args()

    summary = json.loads(args.baseline_summary.read_text())
    fails = summary["fail_list"]
    print(f"GT-labeling {len(fails)} fails from {args.baseline_summary}")

    # Cache one env per task_id
    envs: dict[int, object] = {}
    episodes = []
    counts: Counter[str] = Counter()

    for i, item in enumerate(fails):
        tid = int(item["task_id"])
        seed = int(item["seed"])
        traj = Path(item["trajectory_csv"])
        print(f"\n[{i+1}/{len(fails)}] task={tid} seed={seed} {traj.name}")
        if tid not in envs:
            envs[tid] = make_env(args.env_task, tid)
        stats = label_one(envs[tid], seed=seed, traj_csv=traj)
        rec = {**item, **stats}
        episodes.append(rec)
        counts[stats["mode_gt"]] += 1
        print(
            f"  -> {stats['mode_gt']}  ever_held={stats.get('ever_held')} "
            f"lift={stats.get('max_obj_lift_m', 0):.3f}m "
            f"leave@{stats.get('first_leave_empty_step')}"
        )

    for env in envs.values():
        try:
            env.close()
        except Exception:
            pass

    by_task: dict[str, Counter] = {}
    for ep in episodes:
        by_task.setdefault(str(ep["task_id"]), Counter())[ep["mode_gt"]] += 1

    out = {
        "source": "pi0_baseline 10x10 libero_object seed_base=1000",
        "method": "action replay + GraspRetryController GT geometry (offline labeling only)",
        "deployable_constraint": (
            "These GT labels use sim body_xpos/AABB. Deployable critics must NOT "
            "use that at runtime — vision+proprio only."
        ),
        "baseline_success_rate": summary["success_rate"],
        "n_fail": len(episodes),
        "mode_counts": dict(counts),
        "by_task": {k: dict(v) for k, v in sorted(by_task.items())},
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("\n=== GT mode counts ===")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
