"""Watch G1 walk the playroom and put 5 toys in the bin."""

from __future__ import annotations

import argparse

from robot_gym.envs.room_cleanup import (
    TOY_LABELS,
    RoomCleanupEnv,
    RoomCleanupExpert,
    register_room_env,
)


def main() -> None:
    register_room_env()
    parser = argparse.ArgumentParser(description="G1 room cleanup — 5 toys")
    parser.add_argument("--expert", action="store_true", default=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Mission: pick up these 5 toys and put them in the bin:")
    for i, (key, label) in enumerate(TOY_LABELS.items(), 1):
        print(f"  {i}. {label}  [{key}]")

    env = RoomCleanupEnv(render_mode="human")
    expert = RoomCleanupExpert(env)
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            expert.reset()
            print(f"\nepisode {ep + 1}  first target={info.get('target')}")
            last_phase = None
            for _ in range(env.max_episode_steps):
                action = expert.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                phase = info.get("pick_phase")
                if phase and phase != last_phase:
                    print(f"  pick_phase → {phase}")
                    last_phase = phase
                if info.get("held") and info["held"] not in getattr(main, "_last_held", [None]):
                    print(f"  picked {TOY_LABELS.get(info['held'], info['held'])}")
                    main._last_held = [info["held"]]
                if len(info.get("deposited", [])) > getattr(main, "_n_dep", 0):
                    print(f"  deposited → {info['deposited']}")
                    main._n_dep = len(info["deposited"])
                if terminated or truncated:
                    print(f"  done success={info.get('success')} deposited={info.get('deposited')}")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
