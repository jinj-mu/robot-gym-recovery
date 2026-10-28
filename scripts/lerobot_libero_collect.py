#!/usr/bin/env python3
"""Offline LIBERO inference collection for Phase 1 (compare to human demos in Phase 2).

Runs policy rollouts headlessly, logs per-step trajectories compatible with
lerobot/libero human demos (7D action + 8D observation.state), plus summary metrics.

Outputs (under --output-dir):
  summary.json          — aggregate metrics
  runs.csv              — one row per episode (spreadsheet-friendly)
  episodes/<suite>_task<id>_ep<n>/
    trajectory.csv      — step-wise action + state (Phase 2 curves)
    episode.json        — metadata for this episode
    video.mp4           — optional (--save-video)

Examples
--------
bash scripts/lerobot_libero_collect.sh

POLICY_PATH=lerobot/pi0_libero_finetuned TASK_IDS='[0,1]' \\
  bash scripts/lerobot_libero_collect.sh

python scripts/lerobot_libero_collect.py \\
  --policy.path lerobot/pi05_libero_finetuned \\
  --env.task libero_object --task-ids 0 1 \\
  --n-episodes 2 --seed 1000 --save-video
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_DIR = Path(os.environ.get("LEROBOT_DIR", ROBOT_GYM_ROOT / "third_party" / "lerobot"))
LIBERO_DIR = Path(os.environ.get("LIBERO_DIR", ROBOT_GYM_ROOT / "third_party" / "LIBERO"))

CONTROL_FREQ_HZ = 20


def setup_paths() -> None:
    libero_repo = str(LIBERO_DIR)
    if libero_repo not in sys.path:
        sys.path.insert(0, libero_repo)
    os.environ.setdefault("LIBERO_DIR", libero_repo)
    os.environ.setdefault("MUJOCO_GL", "cgl")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_robot_gym"))


def patch_robosuite() -> None:
    patch_script = ROBOT_GYM_ROOT / "scripts" / "patch_robosuite_joint_type.py"
    if patch_script.is_file():
        subprocess.run([sys.executable, str(patch_script)], check=False)


def check_hf_auth() -> None:
    from huggingface_hub import get_token

    if not (get_token() or os.environ.get("HF_TOKEN")):
        print(
            "ERROR: Hugging Face auth required for google/paligemma-3b-pt-224 (gated).\n"
            "  hf auth login",
            file=sys.stderr,
        )
        sys.exit(1)


def batch_observation(obs: dict) -> dict:
    batched: dict = {}
    for key, value in obs.items():
        if key == "pixels" and isinstance(value, dict):
            batched[key] = {
                cam: (img[np.newaxis, ...] if isinstance(img, np.ndarray) and img.ndim == 3 else img)
                for cam, img in value.items()
            }
        elif key == "robot_state" and isinstance(value, dict):
            batched[key] = _add_batch_dim(value)
        elif isinstance(value, np.ndarray):
            batched[key] = value[np.newaxis, ...] if value.ndim >= 1 else value
        else:
            batched[key] = value
    return batched


def _add_batch_dim(tree):
    if isinstance(tree, dict):
        return {k: _add_batch_dim(v) for k, v in tree.items()}
    arr = np.asarray(tree)
    if arr.ndim == 1:
        return arr[np.newaxis, ...]
    return arr


@dataclass
class EpisodeMetrics:
    policy_path: str
    policy_type: str
    env_task: str
    task_id: int
    task_description: str
    episode_index: int
    seed: int | None
    success: bool
    num_steps: int
    num_inference_calls: int
    wall_time_s: float
    mean_step_time_s: float
    n_action_steps: int
    chunk_size: int
    grasp_retry: bool
    n_retry_sequences: int
    held_object: bool
    trajectory_csv: str
    episode_json: str
    video_path: str | None


def load_policy_bundle(policy_path: str, device: str, task_suite: str, task_id: int):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors

    env_cfg = LiberoEnvConfig(
        task=task_suite,
        task_ids=[task_id],
        obs_type="pixels_agent_pos",
        observation_height=360,
        observation_width=360,
    )

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = device
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False
    if hasattr(policy_cfg, "gradient_checkpointing"):
        policy_cfg.gradient_checkpointing = False

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor_overrides = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor, policy_cfg, env_cfg


def run_collect_episode(
    *,
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    policy_cfg,
    policy_path: str,
    env_task: str,
    task_id: int,
    env,
    episode_index: int,
    seed: int | None,
    save_video: bool,
    episode_dir: Path,
    grasp_retry: bool = False,
    replay_fail_csv: Path | None = None,
) -> tuple[list[dict], list[np.ndarray], EpisodeMetrics]:
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION, OBS_STATE
    from lerobot.utils.io_utils import write_video

    episode_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = episode_dir / "trajectory.csv"
    video_path = episode_dir / "video.mp4" if save_video else None

    max_steps = TASK_SUITE_MAX_STEPS.get(env_task, 500)
    if grasp_retry:
        # Extra time for detect-leave + several descend/close attempts.
        max_steps = max(max_steps, 450)

    replay_actions: list[np.ndarray] | None = None
    if replay_fail_csv is not None:
        import pandas as pd

        rdf = pd.read_csv(replay_fail_csv)
        replay_actions = [
            rdf.loc[i, [f"action_{j}" for j in range(7)]].to_numpy(dtype=np.float32)
            for i in range(len(rdf))
        ]
        print(f"  replaying {len(replay_actions)} actions from {replay_fail_csv}")

    rows: list[dict] = []
    frames: list[np.ndarray] = []

    policy.reset()
    obs, info = env.reset(seed=seed)
    task_desc = env.task_description

    retry = None
    if grasp_retry:
        from lerobot_libero_grasp_retry import GraspRetryController, clear_policy_queue

        retry = GraspRetryController(env)
        print(f"  grasp-retry on  target={retry.target_name} init_z={retry.init_obj_z:.3f}")

    step = 0
    inference_calls = 0
    success = False
    t0 = time.perf_counter()
    prev_retry_mode = "policy"

    while step < max_steps:
        if save_video:
            frames.append(env.render().copy())

        observation = preprocess_observation(batch_observation(obs))
        observation["task"] = [task_desc]
        observation = env_preprocessor(observation)

        state_vec = observation.get(OBS_STATE)
        if state_vec is not None:
            state_np = state_vec.detach().cpu().numpy().reshape(-1)
        else:
            state_np = np.zeros(8, dtype=np.float32)

        queue_len = len(getattr(policy, "_action_queue", []))
        is_inference = queue_len == 0
        use_retry = False
        control = "policy"
        if retry is not None:
            use_retry = retry.maybe_takeover(step=step, is_new_chunk=is_inference and step > 0)
            if use_retry:
                clear_policy_queue(policy)
                is_inference = 0
            prev_retry_mode = retry.mode if retry is not None else "policy"

        if use_retry:
            action_np = retry.scripted_action()
            control = {"place": "place", "done": "done"}.get(retry.mode, "retry")
            if retry.phase_t == 1 and retry.phase == "approach":
                obj = retry.obj_pos()
                ee = retry.eef_pos()
                print(
                    f"  step {step}: retry#{retry.n_sequences} offset={retry.offset_i} "
                    f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                    f"obj=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})"
                )
        elif replay_actions is not None and step < len(replay_actions):
            # Lock the same failure path as the original video until leave→retry.
            action_np = replay_actions[step]
            control = "replay_fail"
            is_inference = 0
        else:
            if is_inference:
                inference_calls += 1
            proc_obs = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(proc_obs)
            action = postprocessor(action)
            action_transition = env_postprocessor({ACTION: action})
            action_np = action_transition[ACTION].to("cpu").numpy().reshape(-1)
            control = "policy"

        obs, reward, terminated, truncated, info = env.step(action_np.astype(np.float32))
        step += 1
        success = bool(info.get("is_success", False))

        row = {
            "step": step,
            "time_s": step / CONTROL_FREQ_HZ,
            "reward": float(reward),
            "success": success,
            "is_inference": int(is_inference) if control == "policy" else 0,
            "control": control,
            "holding": int(retry.holding()) if retry is not None else -1,
        }
        for i in range(7):
            row[f"action_{i}"] = float(action_np[i]) if i < len(action_np) else 0.0
        for i in range(8):
            row[f"observation.state_{i}"] = float(state_np[i]) if i < len(state_np) else 0.0
        rows.append(row)

        if terminated or truncated:
            break
        if retry is not None and retry.mode == "done" and success:
            break
        if retry is not None and retry.mode == "done" and retry.phase_t >= max(1, int(0.75 * CONTROL_FREQ_HZ)):
            break

    wall = time.perf_counter() - t0

    fieldnames = list(rows[0].keys())
    with trajectory_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if save_video and frames:
        write_video(str(video_path), np.stack(frames, axis=0), fps=CONTROL_FREQ_HZ)

    metrics = EpisodeMetrics(
        policy_path=policy_path,
        policy_type=policy_cfg.type,
        env_task=env_task,
        task_id=task_id,
        task_description=task_desc,
        episode_index=episode_index,
        seed=seed,
        success=success,
        num_steps=step,
        num_inference_calls=inference_calls,
        wall_time_s=wall,
        mean_step_time_s=wall / step if step else 0.0,
        n_action_steps=getattr(policy_cfg, "n_action_steps", -1),
        chunk_size=getattr(policy_cfg, "chunk_size", -1),
        grasp_retry=bool(grasp_retry),
        n_retry_sequences=int(retry.n_sequences) if retry is not None else 0,
        held_object=bool(retry.holding()) if retry is not None else False,
        trajectory_csv=str(trajectory_path),
        episode_json=str(episode_dir / "episode.json"),
        video_path=str(video_path) if save_video and frames else None,
    )

    with (episode_dir / "episode.json").open("w") as f:
        json.dump(asdict(metrics), f, indent=2)

    return rows, frames, metrics


def parse_task_ids(values: list[str]) -> list[int]:
    if len(values) == 1 and values[0].startswith("["):
        return [int(x) for x in json.loads(values[0])]
    return [int(x) for x in values]


def main() -> None:
    sys.path.insert(0, str(ROBOT_GYM_ROOT / "scripts"))
    from lerobot_libero_live import make_live_libero_env

    parser = argparse.ArgumentParser(description="Offline LIBERO inference data collection")
    parser.add_argument("--policy.path", dest="policy_path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--env.task", dest="env_task", default="libero_object")
    parser.add_argument("--task-ids", nargs="+", default=["0"])
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: outputs/libero_inference/<timestamp>_<policy_slug>",
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--grasp-retry",
        action="store_true",
        help="If object is not held, interrupt place-like motion and offset-retry the grasp",
    )
    args = parser.parse_args()

    setup_paths()
    patch_robosuite()

    if not Path(os.path.expanduser("~/.libero/config.yaml")).is_file():
        print("Missing ~/.libero/config.yaml — run: bash scripts/lerobot_libero_mac_setup.sh", file=sys.stderr)
        sys.exit(1)

    check_hf_auth()

    task_ids = parse_task_ids(args.task_ids)
    policy_slug = args.policy_path.replace("/", "_").replace(":", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(args.output_dir or ROBOT_GYM_ROOT / "outputs" / "libero_inference" / f"{stamp}_{policy_slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[EpisodeMetrics] = []

    # Load once — LIBERO obs/action dims are shared across task_ids in a suite.
    print(f"\n=== Loading policy once: {args.policy_path} ({args.env_task}) ===")
    policy, preprocessor, postprocessor, env_pre, env_post, policy_cfg, _ = load_policy_bundle(
        args.policy_path, args.device, args.env_task, task_ids[0]
    )

    for task_id in task_ids:
        print(f"\n=== {args.env_task} task_id={task_id} ===")
        for ep in range(args.n_episodes):
            # Unique seed per (task, episode) so inits aren't identical across tasks
            ep_seed = (
                args.seed + task_id * 100 + ep if args.seed is not None else None
            )
            ep_dir = out_dir / "episodes" / f"{args.env_task}_task{task_id}_ep{ep}"
            print(f"Episode {ep + 1}/{args.n_episodes} seed={ep_seed}")

            env = make_live_libero_env(
                task_suite_name=args.env_task,
                task_id=task_id,
                render_mode="opencv",
            )
            try:
                _, _, metrics = run_collect_episode(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    env_preprocessor=env_pre,
                    env_postprocessor=env_post,
                    policy_cfg=policy_cfg,
                    policy_path=args.policy_path,
                    env_task=args.env_task,
                    task_id=task_id,
                    env=env,
                    episode_index=ep,
                    seed=ep_seed,
                    save_video=args.save_video,
                    episode_dir=ep_dir,
                    grasp_retry=args.grasp_retry,
                )
                all_metrics.append(metrics)
                # Append incrementally so a long run is still usable if interrupted
                runs_path = out_dir / "runs.csv"
                run_fields = list(asdict(metrics).keys())
                write_header = not runs_path.is_file()
                with runs_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=run_fields)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(asdict(metrics))
                print(
                    f"  success={metrics.success} steps={metrics.num_steps} "
                    f"inference_calls={metrics.num_inference_calls} wall={metrics.wall_time_s:.1f}s"
                )
            finally:
                env.close()

    # Rewrite runs.csv cleanly (dedupe / consistent order)
    runs_path = out_dir / "runs.csv"
    if all_metrics:
        run_fields = list(asdict(all_metrics[0]).keys())
        with runs_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_fields)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(asdict(m))

    successes = sum(1 for m in all_metrics if m.success)
    per_task: dict[int, dict] = {}
    for m in all_metrics:
        d = per_task.setdefault(m.task_id, {"n": 0, "success": 0, "steps": []})
        d["n"] += 1
        d["success"] += int(m.success)
        d["steps"].append(m.num_steps)
    per_task_summary = {
        str(tid): {
            "n": v["n"],
            "success_count": v["success"],
            "success_rate": v["success"] / v["n"] if v["n"] else 0.0,
            "mean_steps": float(sum(v["steps"]) / len(v["steps"])) if v["steps"] else None,
        }
        for tid, v in sorted(per_task.items())
    }
    summary = {
        "created_at": stamp,
        "policy_path": args.policy_path,
        "env_task": args.env_task,
        "task_ids": task_ids,
        "n_episodes_per_task": args.n_episodes,
        "seed_base": args.seed,
        "device": args.device,
        "control_freq_hz": CONTROL_FREQ_HZ,
        "trajectory_format": {
            "action_dims": 7,
            "observation_state_dims": 8,
            "note": "Matches lerobot/libero — compare in Phase 2 with visualize_dataset curves",
        },
        "episodes_total": len(all_metrics),
        "success_count": successes,
        "success_rate": successes / len(all_metrics) if all_metrics else 0.0,
        "per_task": per_task_summary,
        "episodes": [asdict(m) for m in all_metrics],
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Output: {out_dir}")
    print(f"  summary.json, runs.csv, episodes/*/trajectory.csv")
    print(f"  overall success_rate={summary['success_rate']:.3f} ({successes}/{len(all_metrics)})")
    if args.save_video:
        print("  episodes/*/video.mp4")


if __name__ == "__main__":
    main()
