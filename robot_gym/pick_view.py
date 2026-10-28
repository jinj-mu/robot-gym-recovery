"""Watch the Panda + floor toys scene (random actions or expert)."""

from __future__ import annotations

import argparse

from robot_gym.envs import PickToysFloorEnv
from robot_gym.envs.expert import ToyPickExpert


def main() -> None:
    parser = argparse.ArgumentParser(description="View pick-toys-floor scene")
    parser.add_argument("--expert", action="store_true", help="Run scripted pick expert")
    parser.add_argument("--toy", default="toy_block", choices=["toy_block", "toy_ball", "toy_can", "random"])
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = PickToysFloorEnv(render_mode="human", toy=args.toy)
    expert = ToyPickExpert(env)
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=ep)
            expert.reset()
            print(f"episode {ep + 1}: picking {info['active_toy']}")
            for _ in range(env.max_episode_steps):
                if args.expert:
                    action = expert.act(obs)
                else:
                    action = env.action_space.sample() * 0.3
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"  done success={info.get('success')} reward_last={reward:.2f}")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
