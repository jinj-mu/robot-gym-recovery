#!/usr/bin/env python3
"""Hybrid recovery demo: deployable leave-empty critic → oracle scripted recovery.

Detect: vision + proprio only (LeaveEmptyDeployableCritic).
Fix: existing GraspRetryController pick/place (uses sim poses — oracle recovery).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
from fractions import Fraction

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT_GYM_ROOT / "scripts"))

from lerobot_libero_deployable_critic import LeaveEmptyDeployableCritic  # noqa: E402
from lerobot_libero_grasp_retry import GraspRetryController  # noqa: E402
from lerobot_libero_gt_label_fails import load_actions  # noqa: E402


def make_env_hd(
    task_id: int,
    h: int = 256,
    w: int = 256,
    *,
    horizon: int = 3000,
    ignore_done: bool = True,
):
    """HD LIBERO env. Longer horizon so detect→retry can finish (not GT cheating)."""
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite("libero_object")

    class E(LiberoEnv):
        def _ensure_env(self):
            if self._env is not None:
                return
            from libero.libero.envs import OffScreenRenderEnv

            env = OffScreenRenderEnv(
                bddl_file_name=self._task_bddl_file,
                camera_heights=self.observation_height,
                camera_widths=self.observation_width,
                control_freq=self.control_freq,
                hard_reset=self.hard_reset,
                horizon=horizon,
                ignore_done=ignore_done,
            )
            env.reset()
            self._env = env

    return E(
        task_suite=suite,
        task_id=task_id,
        task_suite_name="libero_object",
        obs_type="pixels_agent_pos",
        observation_height=h,
        observation_width=w,
        init_states=True,
        episode_index=0,
        n_envs=1,
        control_mode="relative",
    )


def flip_view(image: np.ndarray) -> np.ndarray:
    view = np.asarray(image)
    if view.dtype != np.uint8:
        view = np.clip(view, 0, 255).astype(np.uint8)
    return view[::-1, ::-1].copy()


def banner(img: np.ndarray, lines: list[str], color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 20 + 18 * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(
            out, t, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )
    return out


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    h, w = frames[0].shape[:2]
    H, W = h + h % 2, w + w % 2
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=Fraction(fps))
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {
        "preset": "medium",
        "crf": "18",
        "profile": "baseline",
        "level": "3.1",
        "movflags": "+faststart",
    }
    for f in frames:
        pad = np.zeros((H, W, 3), dtype=np.uint8)
        pad[: f.shape[0], : f.shape[1]] = f
        rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


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
        / "outputs/libero_inference/paper_oracle_v1/hybrid_recover_seed1202",
    )
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument(
        "--place-mode",
        choices=("low_safe", "legacy_high"),
        default="low_safe",
        help="Scripted place after HOLD. low_safe enables CarryHeightCritic.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actions = load_actions(args.traj)
    env = make_env_hd(args.task_id)
    obs, _ = env.reset(seed=args.seed)

    critic = LeaveEmptyDeployableCritic()
    critic.reset()
    recovery = GraspRetryController(env, place_mode=args.place_mode)

    frames: list[np.ndarray] = []
    fire_step = None
    success = False
    phase = "replay_fail"

    for step in range(args.max_steps):
        image = obs["pixels"]["image"]
        ee = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        grip = np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float64)

        if phase == "replay_fail":
            fired = critic.step(image=image, ee_pos=ee, grip_qpos=grip)
            dbg = critic.last_debug
            if fired:
                fire_step = step + 1
                recovery.takeover_step = step
                recovery.start_retry(new_sequence=True)
                phase = "oracle_recovery"
                print(
                    f"step {fire_step}: DEPLOYABLE critic FIRE → start oracle scripted recovery "
                    f"(grasp={dbg.grasp_attempted} ee_left={dbg.ee_left_pick} "
                    f"on_table={dbg.object_on_table} mad={dbg.vision_mad:.3f})"
                )
                action = recovery.scripted_action(image=image)
                control = "retry"
            else:
                if step >= len(actions):
                    print("ran out of replay actions without critic fire")
                    break
                action = actions[step]
                control = "replay_fail"
                dbg = critic.last_debug
        else:
            action = recovery.scripted_action(image=image)
            control = {"place": "place", "done": "done"}.get(recovery.mode, "retry")
            dbg = critic.last_debug
            if recovery.mode == "done":
                # one more open step already produced; check success after step
                pass

        view = flip_view(image)
        if phase == "replay_fail":
            col = (255, 255, 255)
            tag = "REPLAY fail (π₀ actions)"
        elif recovery.mode == "place":
            col = (255, 200, 0)
            tag = "ORACLE recovery PLACE"
        elif recovery.mode == "done":
            col = (0, 255, 0)
            tag = "ORACLE recovery DONE"
        else:
            col = (0, 255, 255)
            tag = "ORACLE recovery PICK"

        vis = banner(
            view,
            [
                f"HYBRID: deployable detect → oracle fix   step={step+1}",
                f"phase={tag}  control={control}  mode={recovery.mode}",
                f"critic fire@{fire_step}  grasp={int(dbg.grasp_attempted)} "
                f"ee_left={int(dbg.ee_left_pick)} on_table={int(dbg.object_on_table)}",
                f"holding_GT={int(recovery.holding())}  success={int(success)}",
            ],
            color=col,
        )
        if fire_step and step + 1 >= fire_step:
            cv2.rectangle(vis, (3, 3), (vis.shape[1] - 4, vis.shape[0] - 4), col, 3)
        frames.append(vis)

        obs, reward, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.float32).reshape(-1)
        )
        del reward, terminated, truncated
        success = bool(info.get("is_success", False))
        if success:
            print(f"step {step+1}: SUCCESS after hybrid recovery")
            # pad a few hold frames
            for _ in range(15):
                frames.append(
                    banner(
                        flip_view(obs["pixels"]["image"]),
                        [
                            f"SUCCESS  fire@{fire_step}  total_steps={step+1}",
                            "deployable critic + oracle scripted recovery",
                        ],
                        color=(0, 255, 0),
                    )
                )
            break
        if phase == "oracle_recovery" and recovery.mode == "done" and step > (fire_step or 0) + 5:
            print(f"step {step+1}: recovery marked done, success={success}")
            break

    env.close()

    video_path = args.output_dir / "hybrid_recover.mp4"
    write_mp4(video_path, frames)
    summary = {
        "task_id": args.task_id,
        "seed": args.seed,
        "traj": str(args.traj),
        "deployable_fire_step": fire_step,
        "success": success,
        "n_frames": len(frames),
        "video": str(video_path),
        "stack": "LeaveEmptyDeployableCritic (vision+proprio) → GraspRetryController (oracle poses)",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("ALL_OK" if success else "DONE_NO_SUCCESS")


if __name__ == "__main__":
    main()
