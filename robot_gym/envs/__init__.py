"""Package init — register custom envs on import."""

from robot_gym.envs.pick_toys import PickToysFloorEnv, register_envs
from robot_gym.envs.room_cleanup import register_room_env

register_envs()
register_room_env()

__all__ = ["PickToysFloorEnv", "register_envs", "register_room_env"]

