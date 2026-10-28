"""Open the humanoid in MuJoCo. Physics-only: small random actions so you can confirm the sim."""

from __future__ import annotations

import argparse

import gymnasium as gym

from robot_gym import ENV_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="View the MuJoCo humanoid on Mac")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--scale", type=float, default=0.15, help="Random action scale")
    args = parser.parse_args()

    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset(seed=0)
    print(f"{ENV_ID}  obs={env.observation_space.shape}  act={env.action_space.shape}")
    print("Close the window or Ctrl+C to stop.")

    try:
        for _ in range(args.steps):
            action = env.action_space.sample() * args.scale
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
