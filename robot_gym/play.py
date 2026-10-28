"""Replay a trained PPO humanoid. Falls back to a random policy if nothing is saved yet."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot_gym import DEFAULT_RUN_DIR, ENV_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained Humanoid-v5 policy")
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN_DIR))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    policy_path = args.run_dir / "ppo_humanoid_final.zip"
    vec_path = args.run_dir / "vecnormalize.pkl"

    env = gym.make(ENV_ID, render_mode="human")

    if not policy_path.exists():
        print(f"No trained policy at {policy_path}")
        print("Showing random actions. Train first: uv run robot-gym-train")
        obs, _ = env.reset(seed=0)
        try:
            for _ in range(args.steps):
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    obs, _ = env.reset()
        except KeyboardInterrupt:
            pass
        env.close()
        return

    inner = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="human")])
    if vec_path.exists():
        inner = VecNormalize.load(str(vec_path), inner)
        inner.training = False
        inner.norm_reward = False

    model = PPO.load(str(policy_path), env=inner)
    obs = inner.reset()
    print(f"Loaded {policy_path}")
    try:
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, dones, _ = inner.step(action)
            if dones.any():
                obs = inner.reset()
    except KeyboardInterrupt:
        pass
    inner.close()


if __name__ == "__main__":
    main()
