#!/usr/bin/env python3
"""Live on-screen LIBERO + LeRobot policy demo (Mac MPS).

Runs a trained π₀ / π₀.₅ policy in LIBERO with real-time visualization.
Uses off-screen cameras for policy observations and either an OpenCV window
(default on Mac) or the robosuite MuJoCo viewer for display.

Usage examples
--------------
# π₀.₅ (default) on libero_object task 0
bash scripts/lerobot_libero_live.sh

# π₀ policy
POLICY_PATH=lerobot/pi0_libero_finetuned bash scripts/lerobot_libero_live.sh

# Direct Python — π₀.₅
python scripts/lerobot_libero_live.py \\
    --policy.path lerobot/pi05_libero_finetuned \\
    --env.task libero_object \\
    --task_id 0 \\
    --device mps

# Direct Python — π₀
python scripts/lerobot_libero_live.py \\
    --policy.path lerobot/pi0_libero_finetuned \\
    --env.task libero_spatial \\
    --task_id 2 \\
    --device mps

# Robosuite native viewer (needs GLFW; may fail on headless Mac)
python scripts/lerobot_libero_live.py --render-mode viewer

# OpenCV window (recommended on Mac) — agentview | wrist, task + action overlay
python scripts/lerobot_libero_live.py --render-mode opencv

# Quick env/render smoke test (no policy download)
python scripts/lerobot_libero_live.py --smoke-test

# Interactive: pick model + task from menus, rerun without full reload
bash scripts/lerobot_libero_play.sh

Mac notes
---------
- Set MUJOCO_GL=cgl (default in the shell wrapper).
- MPS inference is slow (~1–5 s/step for π₀.₅); expect low FPS.
- HF auth required for google/paligemma-3b-pt-224 (gated): hf auth login
- If robosuite viewer fails, use --render-mode opencv (default).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dataclasses import dataclass

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_DIR = Path(os.environ.get("LEROBOT_DIR", ROBOT_GYM_ROOT / "third_party" / "lerobot"))
LIBERO_DIR = Path(os.environ.get("LIBERO_DIR", ROBOT_GYM_ROOT / "third_party" / "LIBERO"))

POLICY_PRESETS = {
    "pi05": "lerobot/pi05_libero_finetuned",
    "pi0": "lerobot/pi0_libero_finetuned",
}

SUITE_PRESETS = [
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_10",
]


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
            "  1. Accept license: https://huggingface.co/google/paligemma-3b-pt-224\n"
            "  2. hf auth login   OR   export HF_TOKEN=<your token>",
            file=sys.stderr,
        )
        sys.exit(1)


def make_on_screen_env_class():
    """OnScreenRenderEnv is not in upstream LIBERO; mirror OffScreenRenderEnv pattern."""
    from libero.libero.envs.env_wrapper import ControlEnv

    class OnScreenRenderEnv(ControlEnv):
        """MuJoCo on-screen viewer plus off-screen cameras for policy obs."""

        def __init__(self, **kwargs):
            kwargs["has_renderer"] = True
            kwargs["has_offscreen_renderer"] = True
            kwargs.setdefault("use_camera_obs", True)
            super().__init__(**kwargs)

    return OnScreenRenderEnv


def make_live_libero_env(
    *,
    task_suite_name: str,
    task_id: int,
    render_mode: str,
    observation_height: int = 360,
    observation_width: int = 360,
):
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(task_suite_name)

    class LiveLiberoEnv(LiberoEnv):
        def _ensure_env(self) -> None:
            if self._env is not None:
                return
            if render_mode == "viewer":
                env_cls = make_on_screen_env_class()
            else:
                from libero.libero.envs import OffScreenRenderEnv

                env_cls = OffScreenRenderEnv

            env = env_cls(
                bddl_file_name=self._task_bddl_file,
                camera_heights=self.observation_height,
                camera_widths=self.observation_width,
                control_freq=self.control_freq,
                hard_reset=self.hard_reset,
            )
            env.reset()
            self._env = env

    return LiveLiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=task_suite_name,
        obs_type="pixels_agent_pos",
        observation_height=observation_height,
        observation_width=observation_width,
        init_states=True,
        episode_index=0,
        n_envs=1,
        control_mode="relative",
    )


def _flip_cam(img: np.ndarray) -> np.ndarray:
    """Match LeRobot LIBERO render() orientation for display."""
    return img[::-1, ::-1]


def format_action(action: np.ndarray) -> str:
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return "[]"
    xyz = ", ".join(f"{x:+.3f}" for x in a[: min(3, a.size)])
    if a.size <= 3:
        return f"[{xyz}]"
    rpy = ", ".join(f"{x:+.3f}" for x in a[3:6])
    grip = f", g={a[6]:.2f}" if a.size > 6 else ""
    return f"[{xyz} | {rpy}{grip}]"


def _wrap_text(text: str, max_chars: int = 56) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def build_display_frame(
    *,
    obs: dict | None = None,
    env=None,
    task_desc: str = "",
    step: int | None = None,
    action: np.ndarray | None = None,
    success: bool | None = None,
) -> np.ndarray:
    """Agentview + wrist side-by-side with task / step / action overlays."""
    import cv2

    panels: list[tuple[str, np.ndarray]] = []
    if obs is not None and isinstance(obs.get("pixels"), dict):
        for key, label in (("image", "agentview"), ("image2", "wrist")):
            if key in obs["pixels"]:
                panels.append((label, _flip_cam(np.asarray(obs["pixels"][key], dtype=np.uint8))))

    if panels:
        target_h = max(img.shape[0] for _, img in panels)
        resized: list[np.ndarray] = []
        for label, img in panels:
            if img.shape[0] != target_h:
                scale = target_h / img.shape[0]
                new_w = max(1, int(img.shape[1] * scale))
                img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
            resized.append(img)
            # camera label bar
            bar = np.zeros((24, img.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                bar,
                label,
                (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            resized[-1] = np.vstack([bar, img])
        frame = np.hstack(resized) if len(resized) > 1 else resized[0]
    elif env is not None:
        frame = env.render()
    else:
        frame = np.zeros((360, 360, 3), dtype=np.uint8)

    overlay_lines: list[str] = []
    if task_desc:
        overlay_lines.extend(_wrap_text(task_desc))
    if step is not None:
        status = "?" if success is None else str(bool(success))
        line = f"step {step}"
        if action is not None:
            line += f"  action {format_action(action)}"
        line += f"  success={status}"
        overlay_lines.append(line)

    if overlay_lines:
        line_h = 22
        pad = 8
        box_h = pad * 2 + line_h * len(overlay_lines)
        box = np.zeros((box_h, frame.shape[1], 3), dtype=np.uint8)
        box[:] = (24, 24, 24)
        y = pad + 16
        for line in overlay_lines:
            cv2.putText(
                box,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            y += line_h
        frame = np.vstack([frame, box])

    return frame


class LiveViewer:
    def __init__(self, render_mode: str, window_name: str = "LIBERO live"):
        self.render_mode = render_mode
        self.window_name = window_name
        self._cv2 = None
        if render_mode == "opencv":
            import cv2

            self._cv2 = cv2

    def show(
        self,
        env,
        *,
        obs: dict | None = None,
        task_desc: str = "",
        step: int | None = None,
        action: np.ndarray | None = None,
        success: bool | None = None,
    ) -> bool:
        """Update display. Returns False if user requested quit (q/Esc)."""
        if self.render_mode == "viewer":
            env._env.env.render()
            return True

        frame = build_display_frame(
            obs=obs,
            env=env,
            task_desc=task_desc,
            step=step,
            action=action,
            success=success,
        )
        if self._cv2 is None:
            return True
        bgr = frame[:, :, ::-1]
        self._cv2.imshow(self.window_name, bgr)
        key = self._cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))

    def close(self) -> None:
        if self._cv2 is not None:
            self._cv2.destroyAllWindows()


def smoke_test(render_mode: str, task_suite: str, task_id: int) -> None:
    """Verify env reset, stepping, and on-screen render without loading a policy."""
    print(f"Smoke test: render_mode={render_mode} suite={task_suite} task_id={task_id}")
    env = make_live_libero_env(
        task_suite_name=task_suite,
        task_id=task_id,
        render_mode=render_mode,
        observation_height=128,
        observation_width=128,
    )
    viewer = LiveViewer(render_mode)
    try:
        obs, info = env.reset()
        print("reset OK — obs keys:", list(obs.keys()))
        viewer.show(env)

        from lerobot.envs.libero import get_libero_dummy_action

        for step in range(5):
            action = np.array(get_libero_dummy_action(), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if not viewer.show(env):
                print("User quit")
                break
            if terminated or truncated:
                print(f"Episode ended at step {step}: success={info.get('is_success')}")
                break
            time.sleep(0.05)
        print("Smoke test passed.")
    finally:
        viewer.close()
        env.close()


@dataclass
class PolicyBundle:
    policy_path: str
    device: str
    policy: object
    preprocessor: object
    postprocessor: object
    env_preprocessor: object
    env_postprocessor: object
    env_cfg: object


def list_suite_tasks(suite_name: str) -> list[tuple[int, str]]:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    tasks: list[tuple[int, str]] = []
    for i in range(len(suite.tasks)):
        task = suite.get_task(i)
        label = getattr(task, "language", None) or getattr(task, "name", str(i))
        tasks.append((i, str(label).replace("_", " ")))
    return tasks


def load_policy_bundle(policy_path: str, device: str, task_suite: str, task_id: int) -> PolicyBundle:
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

    print(f"Loading policy: {policy_path} (type={policy_cfg.type}, device={device})")
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

    return PolicyBundle(
        policy_path=policy_path,
        device=device,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        env_cfg=env_cfg,
    )


def run_episode(
    bundle: PolicyBundle,
    *,
    task_suite: str,
    task_id: int,
    render_mode: str,
    n_episodes: int = 1,
    seed: int | None = 1000,
) -> None:
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    env = make_live_libero_env(
        task_suite_name=task_suite,
        task_id=task_id,
        render_mode=render_mode,
    )
    viewer = LiveViewer(render_mode, window_name=f"LIBERO — {bundle.policy_path}")
    max_steps = TASK_SUITE_MAX_STEPS.get(task_suite, 500)

    try:
        for episode in range(n_episodes):
            if seed is not None:
                episode_seed = seed + episode
            else:
                episode_seed = None

            bundle.policy.reset()
            obs, info = env.reset(seed=episode_seed)
            task_desc = env.task_description
            print(f"\nEpisode {episode + 1}/{n_episodes}: {task_desc}")
            print("  >>> OpenCV: agentview | wrist + task/action overlay (q/Esc to quit) <<<")

            if not viewer.show(env, obs=obs, task_desc=task_desc):
                break

            step = 0
            success = False
            last_action: np.ndarray | None = None
            t0 = time.perf_counter()
            while step < max_steps:
                observation = preprocess_observation(batch_observation(obs))
                observation["task"] = [task_desc]

                observation = bundle.env_preprocessor(observation)
                observation = bundle.preprocessor(observation)

                with torch.inference_mode():
                    action = bundle.policy.select_action(observation)
                action = bundle.postprocessor(action)
                action_transition = bundle.env_postprocessor({ACTION: action})
                action_np = action_transition[ACTION].to("cpu").numpy().squeeze(0)
                last_action = action_np

                obs, reward, terminated, truncated, info = env.step(action_np)
                step += 1

                elapsed = time.perf_counter() - t0
                fps = step / elapsed if elapsed > 0 else 0.0
                print(
                    f"  step {step:4d}  action={format_action(action_np)}  "
                    f"success={info.get('is_success')}  ~{fps:.2f} steps/s"
                )

                if not viewer.show(
                    env,
                    obs=obs,
                    task_desc=task_desc,
                    step=step,
                    action=last_action,
                    success=bool(info.get("is_success", False)),
                ):
                    print("User quit")
                    return

                if terminated or truncated:
                    success = bool(info.get("is_success", False))
                    break

            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                print(
                    f"Episode {episode + 1} done: steps={step} success={success} "
                    f"wall={elapsed:.1f}s avg={step / elapsed:.2f} steps/s"
                )
            else:
                print(f"Episode {episode + 1} done: steps={step} success={success}")
    finally:
        viewer.close()
        env.close()


def prompt_choice(prompt: str, options: list[str], default: int = 0) -> int:
    print(prompt)
    for i, label in enumerate(options):
        mark = " (default)" if i == default else ""
        print(f"  [{i}] {label}{mark}")
    while True:
        raw = input("> ").strip()
        if raw == "":
            return default
        try:
            idx = int(raw)
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("Invalid — enter a number from the list.")


def interactive_play(device: str, render_mode: str) -> None:
    print("\n=== LIBERO Play (pick model → task → watch sim) ===\n")

    policy_options = [
        f"π₀.₅ — {POLICY_PRESETS['pi05']}",
        f"π₀ — {POLICY_PRESETS['pi0']}",
        "Custom Hugging Face path (lerobot/… or local)",
    ]
    pi = prompt_choice("Policy:", policy_options, default=0)
    if pi == 2:
        policy_path = input("HF repo or path: ").strip()
        if not policy_path:
            print("Empty path.")
            return
    else:
        policy_path = POLICY_PRESETS["pi05"] if pi == 0 else POLICY_PRESETS["pi0"]

    suite_idx = prompt_choice("Task suite:", SUITE_PRESETS, default=0)
    task_suite = SUITE_PRESETS[suite_idx]

    tasks = list_suite_tasks(task_suite)
    print(f"\nTasks in {task_suite}:")
    for tid, label in tasks:
        print(f"  [{tid}] {label}")
    while True:
        raw = input("Task id> ").strip()
        if raw == "":
            task_id = 0
            break
        try:
            task_id = int(raw)
            if any(tid == task_id for tid, _ in tasks):
                break
        except ValueError:
            pass
        print("Invalid task id.")

    bundle = load_policy_bundle(policy_path, device, task_suite, task_id)

    while True:
        run_episode(bundle, task_suite=task_suite, task_id=task_id, render_mode=render_mode)

        print("\n--- Next ---")
        print("  [r] same task again")
        print("  [t] pick another task (same model, ~fast)")
        print("  [m] change model (~10 min reload)")
        print("  [q] quit")
        choice = input("> ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice == "m":
            pi = prompt_choice("Policy:", policy_options, default=0)
            if pi == 2:
                policy_path = input("HF repo or path: ").strip()
            else:
                policy_path = POLICY_PRESETS["pi05"] if pi == 0 else POLICY_PRESETS["pi0"]
            bundle = load_policy_bundle(policy_path, device, task_suite, task_id)
            continue
        if choice == "t":
            tasks = list_suite_tasks(task_suite)
            for tid, label in tasks:
                print(f"  [{tid}] {label}")
            while True:
                raw = input("Task id> ").strip()
                try:
                    task_id = int(raw)
                    if any(tid == task_id for tid, _ in tasks):
                        break
                except ValueError:
                    pass
                print("Invalid task id.")
            continue
        if choice == "r" or choice == "":
            continue
        print("Unknown option — running same task again.")


def run_live_rollout(
    *,
    policy_path: str,
    task_suite: str,
    task_id: int,
    device: str,
    render_mode: str,
    n_episodes: int,
    seed: int | None,
) -> None:
    bundle = load_policy_bundle(policy_path, device, task_suite, task_id)
    run_episode(
        bundle,
        task_suite=task_suite,
        task_id=task_id,
        render_mode=render_mode,
        n_episodes=n_episodes,
        seed=seed,
    )


def batch_observation(obs: dict) -> dict:
    """Add batch dim (B=1) for single-env rollout, matching vector-env eval format."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live LIBERO + LeRobot policy demo")
    parser.add_argument(
        "--policy.path",
        dest="policy_path",
        default=None,
        help="HF repo or local checkpoint (default: lerobot/pi05_libero_finetuned)",
    )
    parser.add_argument(
        "--policy",
        choices=sorted(POLICY_PRESETS),
        default=None,
        help="Shortcut preset: pi05 or pi0 (overridden by --policy.path)",
    )
    parser.add_argument("--env.task", dest="env_task", default="libero_object")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--render-mode",
        choices=("opencv", "viewer"),
        default="opencv",
        help="opencv: cv2.imshow from agentview (Mac default). viewer: robosuite GLFW window.",
    )
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Env/render check only; skip policy load",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Menu: pick policy + task, rerun without reloading model",
    )
    return parser.parse_args()


def resolve_policy_path(args: argparse.Namespace) -> str:
    if args.policy_path:
        return args.policy_path
    if args.policy:
        return POLICY_PRESETS[args.policy]
    env_policy = os.environ.get("POLICY_PATH")
    if env_policy:
        return env_policy
    return POLICY_PRESETS["pi05"]


def main() -> None:
    setup_paths()
    patch_robosuite()

    args = parse_args()

    if not args.smoke_test and not Path(os.path.expanduser("~/.libero/config.yaml")).is_file():
        print("Missing ~/.libero/config.yaml — run: bash scripts/lerobot_libero_mac_setup.sh", file=sys.stderr)
        sys.exit(1)

    if args.smoke_test:
        smoke_test(args.render_mode, args.env_task, args.task_id)
        return

    check_hf_auth()

    if args.interactive:
        interactive_play(device=args.device, render_mode=args.render_mode)
        return

    policy_path = resolve_policy_path(args)
    print(f"Suite: {args.env_task}  task_id={args.task_id}  render={args.render_mode}")
    run_live_rollout(
        policy_path=policy_path,
        task_suite=args.env_task,
        task_id=args.task_id,
        device=args.device,
        render_mode=args.render_mode,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
