"""Play a BC toy-pickup policy (or the scripted expert)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from robot_gym.envs import PickToysFloorEnv
from robot_gym.envs.expert import ToyPickExpert
from robot_gym.pick_train_bc import BCPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Play pick-toys policy")
    parser.add_argument("--policy", type=Path, default=Path("runs/pick_toys/bc_policy.pt"))
    parser.add_argument("--expert", action="store_true", help="Use scripted expert instead")
    parser.add_argument("--toy", default="toy_block", choices=["toy_block", "toy_ball", "toy_can", "random"])
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = PickToysFloorEnv(render_mode="human", toy=args.toy)
    expert = ToyPickExpert(env)
    policy = None
    if not args.expert:
        if not args.policy.exists():
            print(f"No policy at {args.policy}; falling back to expert.")
            args.expert = True
        else:
            ckpt = torch.load(args.policy, map_location="cpu", weights_only=True)
            policy = BCPolicy(obs_dim=ckpt["obs_dim"])
            policy.load_state_dict(ckpt["state_dict"])
            policy.eval()
            print(f"Loaded {args.policy}")

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=100 + ep)
            expert.reset()
            print(f"episode {ep + 1}: {info['active_toy']}")
            for _ in range(env.max_episode_steps):
                if args.expert or policy is None:
                    action = expert.act(obs)
                else:
                    with torch.no_grad():
                        action = policy(torch.from_numpy(obs.astype(np.float32))).numpy()
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"  success={info.get('success')}")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
