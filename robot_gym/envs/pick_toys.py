"""Pick toys with a Franka Panda (mocap EE + gripper). Toys on a low play table."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ASSET_DIR = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "mujoco_menagerie"
    / "franka_emika_panda"
)
SCENE_XML = ASSET_DIR / "pick_toys_floor.xml"

TOY_BODIES = ("toy_block", "toy_ball", "toy_can")
GRIPPER_OPEN = 255.0
HOME_QUAT = np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float64)


class PickToysFloorEnv(gym.Env):
    """Action: dx, dy, dz, gripper ∈ [-1, 1]. Goal: put a toy in the bin."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        toy: str = "toy_block",
        max_episode_steps: int = 400,
        control_hz: int = 20,
    ):
        super().__init__()
        if toy not in TOY_BODIES and toy != "random":
            raise ValueError(f"toy must be one of {TOY_BODIES} or 'random'")
        self.toy_choice = toy
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        self.model.actuator_gainprm[:7] = 0.0
        self.model.actuator_biasprm[:7] = 0.0
        self.n_substeps = max(1, int(round(1.0 / (control_hz * self.model.opt.timestep))))

        self.hand_id = self.model.body("hand").id
        self.mocap_id = int(self.model.body("ee_mocap").mocapid[0])
        self.bin_id = self.model.site("bin_center").id
        self.toy_ids = {name: self.model.body(name).id for name in TOY_BODIES}
        self.active_toy = "toy_block"
        self._mocap_offset = np.zeros(3)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64)

        self._step_count = 0
        self._viewer = None
        self.mujoco_renderer = None
        # Gymnasium GLFW renderer (works without mjpython, but Mac scroll callbacks can be flaky).
        if render_mode in ("human", "rgb_array"):
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

            self.mujoco_renderer = MujocoRenderer(
                self.model, self.data, width=640, height=480
            )

    def _ee_pos(self) -> np.ndarray:
        return self.data.xpos[self.hand_id].copy()

    def _toy_pos(self) -> np.ndarray:
        return self.data.xpos[self.toy_ids[self.active_toy]].copy()

    def _bin_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.bin_id].copy()

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[:7].copy()
        qd = self.data.qvel[:7].copy()
        grip = np.array([self.data.ctrl[7] / GRIPPER_OPEN], dtype=np.float64)
        ee = self._ee_pos()
        toy = self._toy_pos()
        bin_p = self._bin_pos()
        return np.concatenate([q, qd, grip, ee, toy, bin_p, toy - ee, bin_p - toy])

    def _set_mocap_for_ee(self, ee_pos: np.ndarray) -> None:
        """Command mocap so the welded hand tracks ``ee_pos`` (accounts for weld offset)."""
        self.data.mocap_pos[self.mocap_id] = ee_pos - self._mocap_offset
        self.data.mocap_quat[self.mocap_id] = HOME_QUAT

    def _calibrate_offset(self) -> None:
        self._mocap_offset = self._ee_pos() - self.data.mocap_pos[self.mocap_id].copy()

    def _scatter_toys(self, rng: np.random.Generator) -> None:
        placements = {
            "toy_block": (0.45, 0.65, -0.08, 0.15),
            "toy_ball": (0.45, 0.62, -0.12, 0.08),
            "toy_can": (0.48, 0.65, -0.06, 0.12),
        }
        heights = {"toy_block": 0.255, "toy_ball": 0.255, "toy_can": 0.26}
        for name in TOY_BODIES:
            jnt = self.model.body(name).jntadr[0]
            qadr = self.model.jnt_qposadr[jnt]
            x0, x1, y0, y1 = placements[name]
            self.data.qpos[qadr : qadr + 3] = [
                rng.uniform(x0, x1),
                rng.uniform(y0, y1),
                heights[name],
            ]
            self.data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
            vadr = self.model.jnt_dofadr[jnt]
            self.data.qvel[vadr : vadr + 6] = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        self._scatter_toys(rng)
        self.active_toy = (
            TOY_BODIES[int(rng.integers(0, len(TOY_BODIES)))]
            if self.toy_choice == "random"
            else self.toy_choice
        )

        self.data.ctrl[:] = 0.0
        self.data.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)

        # Start mocap at current hand, calibrate weld offset, then move to hover.
        self.data.mocap_pos[self.mocap_id] = self._ee_pos()
        self.data.mocap_quat[self.mocap_id] = HOME_QUAT
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self._calibrate_offset()
        self._set_mocap_for_ee(np.array([0.55, 0.0, 0.50]))
        for _ in range(120):
            mujoco.mj_step(self.model, self.data)
        self._calibrate_offset()

        self._step_count = 0
        if self.render_mode == "human":
            self.render()
        return self._obs(), {"active_toy": self.active_toy}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)
        desired = self._ee_pos() + action[:3] * 0.03
        desired = np.clip(desired, [0.35, -0.40, 0.26], [0.75, 0.40, 0.70])
        self._set_mocap_for_ee(desired)
        self.data.ctrl[:7] = 0.0
        self.data.ctrl[7] = GRIPPER_OPEN * (0.5 * (1.0 - action[3]))

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._obs()
        toy = self._toy_pos()
        ee = self._ee_pos()
        bin_p = self._bin_pos()
        dist_ee = float(np.linalg.norm(toy - ee))
        dist_bin = float(np.linalg.norm(toy[:2] - bin_p[:2]))
        lifted = toy[2] > 0.32
        in_bin = dist_bin < 0.10 and 0.22 < toy[2] < 0.40
        reward = -0.1 * dist_ee - 0.05 * dist_bin + (0.5 if lifted else 0.0) + (10.0 if in_bin else 0.0)
        terminated = bool(in_bin)
        truncated = self._step_count >= self.max_episode_steps
        info = {
            "active_toy": self.active_toy,
            "lifted": bool(lifted),
            "in_bin": bool(in_bin),
            "dist_ee": dist_ee,
            "success": terminated,
        }
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self.mujoco_renderer is None:
            return None
        try:
            return self.mujoco_renderer.render(self.render_mode)
        except TypeError:
            # Ignore known gymnasium/mujoco scroll-callback mismatch on some Mac builds.
            return None

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
            self.mujoco_renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def register_envs() -> None:
    try:
        gym.spec("PickToysFloor-v0")
    except gym.error.Error:
        gym.register(
            id="PickToysFloor-v0",
            entry_point="robot_gym.envs.pick_toys:PickToysFloorEnv",
            max_episode_steps=400,
        )
