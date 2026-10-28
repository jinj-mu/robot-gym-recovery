"""Collect scripted expert demonstrations for toy pickup."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_gym.envs import PickToysFloorEnv
from robot_gym.envs.expert import ToyPickExpert


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect pick-toys demos")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--toy", default="random", choices=["toy_block", "toy_ball", "toy_can", "random"])
    parser.add_argument("--out", type=Path, default=Path("runs/pick_toys/demos.npz"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = PickToysFloorEnv(render_mode="human" if args.render else None, toy=args.toy)
    expert = ToyPickExpert(env)

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    successes = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        expert.reset()
        ep_obs: list[np.ndarray] = []
        ep_act: list[np.ndarray] = []
        ep_success = False
        for _ in range(env.max_episode_steps):
            action = expert.act(obs)
            ep_obs.append(obs.copy())
            ep_act.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            if info.get("success"):
                ep_success = True
            if terminated or truncated or expert.phase == "done":
                break
        if ep_success:
            obs_list.extend(ep_obs)
            act_list.extend(ep_act)
            successes += 1
        print(f"episode {ep + 1}/{args.episodes} success={ep_success}")

    env.close()
    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.float32)
    np.savez_compressed(args.out, obs=obs_arr, actions=act_arr)
    print(f"saved {len(obs_arr)} transitions ({successes}/{args.episodes} success) → {args.out}")


if __name__ == "__main__":
    main()
