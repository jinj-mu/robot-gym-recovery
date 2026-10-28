"""Train a PPO walking policy on Humanoid-v5. Expect hours, not minutes, for a stable gait."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot_gym import DEFAULT_RUN_DIR, ENV_ID


def make_env(seed: int):
    def _init():
        import gymnasium as gym

        env = gym.make(ENV_ID)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO train Humanoid-v5")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN_DIR))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log=str(args.run_dir / "tb"),
    )

    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(args.run_dir / "ckpts"),
        name_prefix="ppo_humanoid",
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint)
    model.save(args.run_dir / "ppo_humanoid_final")
    env.save(args.run_dir / "vecnormalize.pkl")
    env.close()
    print(f"Saved policy to {args.run_dir / 'ppo_humanoid_final.zip'}")


if __name__ == "__main__":
    main()
