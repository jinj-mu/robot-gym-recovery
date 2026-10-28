#!/usr/bin/env python3
"""Compare two hybrid recoveries on the same leave-empty fail.

A) low_safe scripted place (no sky-drop)
B) scripted regrasp → return to π₀ for place (grip locked until clearly placing)

Detect is always LeaveEmptyDeployableCritic (vision+proprio).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT_GYM_ROOT / "scripts"))

from lerobot_libero_collect import batch_observation, load_policy_bundle  # noqa: E402
from lerobot_libero_deployable_critic import LeaveEmptyDeployableCritic  # noqa: E402
from lerobot_libero_grasp_retry import GraspRetryController, clear_policy_queue  # noqa: E402
from lerobot_libero_gt_label_fails import load_actions  # noqa: E402
from lerobot_libero_hybrid_recover import flip_view, make_env_hd, write_mp4  # noqa: E402


def thin_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_variant(
    *,
    name: str,
    task_id: int,
    seed: int,
    actions: list[np.ndarray],
    place_mode: str,
    handback_on_hold: bool,
    policy_bundle,
    max_steps: int = 450,
    collision_lift: bool = True,
) -> dict:
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    env = make_env_hd(task_id, 256, 256)
    # Policy path expects 360 obs in bundle; critic/replay use env pixels. For π₀
    # inference we rebuild obs through env at its resolution — make_env_hd is 256.
    # Reload env at 360 if handback needs policy.
    if handback_on_hold:
        env.close()
        env = make_env_hd(task_id, 360, 360)

    obs, _ = env.reset(seed=seed)
    critic = LeaveEmptyDeployableCritic()
    critic.reset()
    recovery = GraspRetryController(
        env,
        place_mode=place_mode,
        handback_on_hold=handback_on_hold,
        collision_lift=collision_lift,
    )

    policy = preprocessor = postprocessor = env_pre = env_post = None
    if handback_on_hold:
        policy, preprocessor, postprocessor, env_pre, env_post, _, _ = policy_bundle
        policy.reset()
        clear_policy_queue(policy)

    frames: list[np.ndarray] = []
    fire_step = None
    hold_step = None
    handback_step = None
    success = False
    phase = "replay"
    release_ee_z = None
    height_too_low_n = 0
    height_too_high_n = 0
    height_band = None

    for step in range(max_steps):
        image = obs["pixels"]["image"]
        ee = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        grip = np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float64)
        view = flip_view(image)

        if phase == "replay":
            if critic.step(image=image, ee_pos=ee, grip_qpos=grip):
                fire_step = step + 1
                recovery.takeover_step = step
                recovery.start_retry(new_sequence=True)
                phase = "regrasp"
                action = recovery.scripted_action(image=image)
                label = f"{name}: regrasp"
            else:
                if step >= len(actions):
                    break
                action = actions[step]
                label = f"{name}: replay fail"
        elif phase == "regrasp":
            action = recovery.scripted_action(image=image)
            if recovery.mode == "handback":
                if handback_step is None:
                    handback_step = step + 1
                    hold_step = step + 1
                    clear_policy_queue(policy)
                    policy.reset()
                    print(f"  [{name}] handback to π₀ at step {handback_step}")
                phase = "pi0_place"
                # Compute π₀ action immediately this step.
                observation = preprocess_observation(batch_observation(obs))
                observation["task"] = [env.task_description]
                observation = env_pre(observation)
                proc = preprocessor(observation)
                with torch.inference_mode():
                    act = policy.select_action(proc)
                act = postprocessor(act)
                action_np = env_post({ACTION: act})[ACTION].to("cpu").numpy().reshape(-1)
                action = recovery.lock_grip_for_policy_action(action_np)
                label = f"{name}: → π₀ place"
            elif recovery.mode == "place":
                if hold_step is None:
                    hold_step = step + 1
                phase = "scripted_place"
                label = f"{name}: scripted place"
            else:
                label = f"{name}: regrasp ({recovery.phase})"
        elif phase == "scripted_place":
            action = recovery.scripted_action(image=image)
            hz = ""
            hc = getattr(recovery, "height_critic", None)
            if hc is not None and getattr(hc, "last_debug", None) is not None:
                d = hc.last_debug
                if height_band is None and getattr(hc, "armed", False):
                    height_band = [float(hc.z_lo), float(hc.z_hi)]
                if getattr(d, "collision_raise", False):
                    height_too_low_n += 1
                    hz = " ⇑push"
                elif d.too_low:
                    height_too_low_n += 1
                    hz = " ↑"
                elif d.too_high:
                    height_too_high_n += 1
                    hz = " ↓"
            label = f"{name}: place z={ee[2]:.2f}{hz} ({recovery.phase})"
            if recovery.phase == "place_release" and release_ee_z is None:
                release_ee_z = float(ee[2])
            if recovery.mode == "done":
                label = f"{name}: done"
        elif phase == "pi0_place":
            observation = preprocess_observation(batch_observation(obs))
            observation["task"] = [env.task_description]
            observation = env_pre(observation)
            proc = preprocessor(observation)
            with torch.inference_mode():
                act = policy.select_action(proc)
            act = postprocessor(act)
            action_np = env_post({ACTION: act})[ACTION].to("cpu").numpy().reshape(-1)
            action = recovery.lock_grip_for_policy_action(action_np)
            label = f"{name}: π₀ place (grip_lock={not recovery.clearly_placing()})"
            if recovery.mode == "done" and release_ee_z is None:
                release_ee_z = float(ee[2])
        else:
            break
        frames.append(thin_label(view, label[:60]))
        obs, _, _, _, info = env.step(np.asarray(action, dtype=np.float32).reshape(-1))
        success = bool(info.get("is_success", False))
        if success:
            for _ in range(20):
                frames.append(thin_label(flip_view(obs["pixels"]["image"]), f"{name}: SUCCESS"))
            break
        if recovery.mode == "done" and step > (fire_step or 0) + 10:
            break

    env.close()
    return {
        "name": name,
        "success": success,
        "fire_step": fire_step,
        "hold_step": hold_step,
        "handback_step": handback_step,
        "release_ee_z": release_ee_z,
        "height_band": height_band,
        "height_too_low_n": height_too_low_n,
        "height_too_high_n": height_too_high_n,
        "n_frames": len(frames),
        "frames": frames,
    }


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
        "--output-dir",
        type=Path,
        default=ROBOT_GYM_ROOT
        / "outputs/libero_inference/paper_oracle_v1/recover_ab_seed1202",
    )
    parser.add_argument("--policy.path", dest="policy_path", default="lerobot/pi0_libero_finetuned")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actions = load_actions(args.traj)
    print("Loading π₀ once for handback variant...")
    bundle = load_policy_bundle(args.policy_path, args.device, "libero_object", args.task_id)

    print("\n=== A: low_safe scripted place ===")
    a = run_variant(
        name="A low_safe",
        task_id=args.task_id,
        seed=args.seed,
        actions=actions,
        place_mode="low_safe",
        handback_on_hold=False,
        policy_bundle=None,
    )
    write_mp4(args.output_dir / "A_low_safe.mp4", a["frames"])

    print("\n=== B: regrasp → return π₀ ===")
    b = run_variant(
        name="B return_pi0",
        task_id=args.task_id,
        seed=args.seed,
        actions=actions,
        place_mode="low_safe",  # unused when handback
        handback_on_hold=True,
        policy_bundle=bundle,
    )
    write_mp4(args.output_dir / "B_return_pi0.mp4", b["frames"])

    # Side-by-side A | B
    n = max(len(a["frames"]), len(b["frames"]))
    h = min(a["frames"][0].shape[0], b["frames"][0].shape[0])
    w = min(a["frames"][0].shape[1], b["frames"][0].shape[1])

    def prep(f):
        return cv2.resize(f, (w, h))

    side = []
    for i in range(n):
        L = prep(a["frames"][i] if i < len(a["frames"]) else a["frames"][-1])
        R = prep(b["frames"][i] if i < len(b["frames"]) else b["frames"][-1])
        side.append(np.hstack([L, R]))
    write_mp4(args.output_dir / "A_vs_B.mp4", side)

    summary = {
        "seed": args.seed,
        "task_id": args.task_id,
        "A_low_safe": {k: v for k, v in a.items() if k != "frames"},
        "B_return_pi0": {k: v for k, v in b.items() if k != "frames"},
        "videos": {
            "A": str(args.output_dir / "A_low_safe.mp4"),
            "B": str(args.output_dir / "B_return_pi0.mp4"),
            "A_vs_B": str(args.output_dir / "A_vs_B.mp4"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== compare ===")
    print(json.dumps(summary, indent=2))
    print("ALL_OK")


if __name__ == "__main__":
    main()
