"""G1 playroom cleanup: walk to 5 toys, visibly reach/grasp, carry to bin."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE_XML = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "room_toys.xml"
)

TOY_NAMES = (
    "ruby_block",
    "ocean_ball",
    "sunny_duck",
    "race_car",
    "cozy_bear",
)

TOY_LABELS = {
    "ruby_block": "Ruby Block (red cube)",
    "ocean_ball": "Ocean Ball (blue sphere)",
    "sunny_duck": "Sunny Duck (yellow)",
    "race_car": "Race Car (green)",
    "cozy_bear": "Cozy Bear (brown plush)",
}

# Right-arm actuator indices in g1.xml (after legs 0-11, waist 12-14, left arm 15-21).
R_SHOULDER_PITCH = 22
R_SHOULDER_ROLL = 23
R_SHOULDER_YAW = 24
R_ELBOW = 25
R_WRIST_ROLL = 26
R_WRIST_PITCH = 27
R_WRIST_YAW = 28
WAIST_PITCH = 14

# Visible reach / carry joint targets (absolute ctrl values).
REACH_ARM = {
    R_SHOULDER_PITCH: 0.85,
    R_SHOULDER_ROLL: -0.55,
    R_SHOULDER_YAW: 0.15,
    R_ELBOW: 0.95,
    R_WRIST_ROLL: 0.0,
    R_WRIST_PITCH: 0.35,
    R_WRIST_YAW: 0.0,
    WAIST_PITCH: 0.35,
}
CARRY_ARM = {
    R_SHOULDER_PITCH: 0.25,
    R_SHOULDER_ROLL: -0.35,
    R_SHOULDER_YAW: 0.0,
    R_ELBOW: 1.15,
    R_WRIST_ROLL: 0.0,
    R_WRIST_PITCH: 0.1,
    R_WRIST_YAW: 0.0,
    WAIST_PITCH: 0.0,
}
PLACE_ARM = {
    R_SHOULDER_PITCH: 0.7,
    R_SHOULDER_ROLL: -0.4,
    R_SHOULDER_YAW: 0.1,
    R_ELBOW: 1.0,
    R_WRIST_ROLL: 0.0,
    R_WRIST_PITCH: 0.25,
    R_WRIST_YAW: 0.0,
    WAIST_PITCH: 0.25,
}

DEFAULT_CAM = {
    "distance": 7.5,
    "azimuth": 140.0,
    "elevation": -22.0,
    "lookat": np.array([0.0, 0.0, 0.85]),
}


class RoomCleanupEnv(gym.Env):
    """Unitree G1 walks a large playroom and puts 5 toys in the bin.

    Pick is an explicit animated reach → grasp → lift so the hand motion is visible.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 6000,
        walk_speed: float = 0.5,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.walk_speed = walk_speed

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        self.n_substeps = 10

        self.pelvis_id = self.model.body("pelvis").id
        self.hand_id = self.model.body("right_wrist_yaw_link").id
        self.bin_id = self.model.site("bin_center").id
        self.halo_id = self.model.site("target_halo").id
        self.toy_ids = {n: self.model.body(n).id for n in TOY_NAMES}
        self.toy_jnt = {n: self.model.body(n).jntadr[0] for n in TOY_NAMES}

        self.stand_qpos: np.ndarray | None = None
        self.stand_ctrl: np.ndarray | None = None
        self._load_stand_pose()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + 1 + 15 + 3 + 5 + 5,), dtype=np.float64
        )

        self._step_count = 0
        self.held_toy: str | None = None
        self.deposited: set[str] = set()
        self.mission_order = list(TOY_NAMES)
        self.mission_idx = 0
        self.pick_phase: str | None = None  # reach | grasp | lift | place | retract
        self.pick_t = 0
        self.pick_target: str | None = None
        self._arm_blend = 0.0

        self.mujoco_renderer = None
        if render_mode in ("human", "rgb_array"):
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

            self.mujoco_renderer = MujocoRenderer(
                self.model,
                self.data,
                default_cam_config=DEFAULT_CAM,
                width=1280,
                height=720,
            )

    def _load_stand_pose(self) -> None:
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key < 0:
            raise RuntimeError("G1 stand keyframe missing")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        self.stand_qpos = self.data.qpos.copy()
        self.stand_ctrl = self.data.ctrl.copy()

    def _pelvis_xy(self) -> np.ndarray:
        return self.data.qpos[0:2].copy()

    def _pelvis_yaw(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _set_yaw(self, yaw: float) -> None:
        half = 0.5 * yaw
        self.data.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]

    def _toy_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.toy_ids[name]].copy()

    def _bin_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.bin_id].copy()

    def _hand_pos(self) -> np.ndarray:
        return self.data.xpos[self.hand_id].copy()

    def busy_picking(self) -> bool:
        return self.pick_phase is not None

    def _scatter_toys(self, rng: np.random.Generator) -> None:
        slots = [
            (1.6, 2.4, 1.0, 2.0),
            (1.4, 2.3, -2.2, -1.2),
            (-1.8, -0.6, 1.4, 2.4),
            (2.0, 2.8, -0.6, 0.6),
            (-2.4, -1.4, 0.6, 1.6),
        ]
        heights = {
            "ruby_block": 0.07,
            "ocean_ball": 0.08,
            "sunny_duck": 0.08,
            "race_car": 0.06,
            "cozy_bear": 0.1,
        }
        for name, (x0, x1, y0, y1) in zip(TOY_NAMES, slots):
            j = self.toy_jnt[name]
            qadr = self.model.jnt_qposadr[j]
            self.data.qpos[qadr : qadr + 3] = [
                rng.uniform(x0, x1),
                rng.uniform(y0, y1),
                heights[name],
            ]
            self.data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
            vadr = self.model.jnt_dofadr[j]
            self.data.qvel[vadr : vadr + 6] = 0

    def _current_target(self) -> str | None:
        if self.held_toy is not None or self.pick_phase in ("reach", "grasp", "lift"):
            if self.pick_target and self.held_toy is None:
                return self.pick_target
            return None
        while self.mission_idx < len(self.mission_order):
            name = self.mission_order[self.mission_idx]
            if name not in self.deposited:
                return name
            self.mission_idx += 1
        return None

    def _apply_arm_pose(self, pose: dict[int, float], alpha: float) -> None:
        assert self.stand_ctrl is not None
        a = float(np.clip(alpha, 0.0, 1.0))
        for idx, target in pose.items():
            self.data.ctrl[idx] = (1 - a) * self.stand_ctrl[idx] + a * target

    def _set_toy_pose(self, name: str, pos: np.ndarray) -> None:
        j = self.toy_jnt[name]
        qadr = self.model.jnt_qposadr[j]
        self.data.qpos[qadr : qadr + 3] = pos
        self.data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
        vadr = self.model.jnt_dofadr[j]
        self.data.qvel[vadr : vadr + 6] = 0

    def _attach_held_toy(self) -> None:
        if self.held_toy is None:
            return
        # Sit clearly in the open hand / in front of the wrist.
        hand = self._hand_pos()
        yaw = self._pelvis_yaw()
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self._set_toy_pose(self.held_toy, hand + 0.08 * forward + np.array([0.0, 0.0, -0.04]))

    def _update_halo(self) -> None:
        # Hide by default under the floor; show on active floor toy.
        self.model.site_pos[self.halo_id] = np.array([0.0, 0.0, -1.0])
        name = self.pick_target or self._current_target()
        if name and name not in self.deposited and self.held_toy is None:
            p = self._toy_pos(name)
            # site_pos is in parent (world) frame for worldbody sites — use data
            self.data.site_xpos  # ensure fwd
            # For a worldbody site, set model.site_pos
            self.model.site_pos[self.halo_id] = np.array([p[0], p[1], 0.02])

    def _obs(self) -> np.ndarray:
        xy = self._pelvis_xy()
        yaw = np.array([self._pelvis_yaw()])
        toy_rel = []
        held = np.zeros(5)
        done = np.zeros(5)
        for i, name in enumerate(TOY_NAMES):
            toy_rel.append(self._toy_pos(name) - np.array([xy[0], xy[1], 0.79]))
            if self.held_toy == name:
                held[i] = 1.0
            if name in self.deposited:
                done[i] = 1.0
        bin_rel = self._bin_pos() - np.array([xy[0], xy[1], 0.79])
        return np.concatenate([xy, [0.79], yaw, *toy_rel, bin_rel, held, done])

    def _track_camera(self) -> None:
        if self.mujoco_renderer is None or self.mujoco_renderer.viewer is None:
            return
        cam = self.mujoco_renderer.viewer.cam
        xy = self._pelvis_xy()
        cam.lookat[:] = [xy[0], xy[1], 0.9]
        cam.distance = 6.8
        cam.elevation = -20.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        assert self.stand_qpos is not None and self.stand_ctrl is not None
        self.data.qpos[:] = self.stand_qpos
        self.data.ctrl[:] = self.stand_ctrl
        self.data.qpos[0:3] = [0.0, 0.0, 0.79]
        self._set_yaw(0.0)
        self._scatter_toys(self.np_random)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(40):
            self.data.ctrl[:] = self.stand_ctrl
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self.held_toy = None
        self.deposited = set()
        self.mission_idx = 0
        self.pick_phase = None
        self.pick_t = 0
        self.pick_target = None
        self._arm_blend = 0.0
        self._update_halo()
        if self.render_mode == "human":
            self.render()
        return self._obs(), {
            "toys": TOY_LABELS,
            "target": self._current_target(),
            "deposited": [],
            "pick_phase": None,
        }

    def _start_pick(self, name: str) -> None:
        self.pick_phase = "reach"
        self.pick_t = 0
        self.pick_target = name
        self._arm_blend = 0.0

    def _start_place(self) -> None:
        self.pick_phase = "place"
        self.pick_t = 0
        self._arm_blend = 0.0

    def _advance_pick_anim(self) -> None:
        """Animate reach/grasp/lift/place so pickup is visually precise."""
        assert self.stand_ctrl is not None
        self.pick_t += 1

        if self.pick_phase == "reach":
            self._arm_blend = min(1.0, self.pick_t / 50.0)
            self._apply_arm_pose(REACH_ARM, self._arm_blend)
            # After arm is mostly down, pull toy toward the hand (visible contact), then grasp.
            if self._arm_blend >= 0.95 and self.pick_target is not None:
                hand = self._hand_pos()
                toy = self._toy_pos(self.pick_target)
                # Slide toy toward hand over a few frames for a clear grab.
                self.pick_phase = "grasp"
                self.pick_t = 0

        elif self.pick_phase == "grasp":
            self._apply_arm_pose(REACH_ARM, 1.0)
            assert self.pick_target is not None
            hand = self._hand_pos()
            toy = self._toy_pos(self.pick_target)
            alpha = min(1.0, self.pick_t / 18.0)
            blended = (1 - alpha) * toy + alpha * (hand + np.array([0.0, 0.0, -0.03]))
            self._set_toy_pose(self.pick_target, blended)
            if alpha >= 1.0:
                self.held_toy = self.pick_target
                self.pick_phase = "lift"
                self.pick_t = 0

        elif self.pick_phase == "lift":
            self._arm_blend = min(1.0, self.pick_t / 40.0)
            # Blend reach → carry
            assert self.stand_ctrl is not None
            for idx in CARRY_ARM:
                reach_v = REACH_ARM[idx]
                carry_v = CARRY_ARM[idx]
                self.data.ctrl[idx] = (1 - self._arm_blend) * reach_v + self._arm_blend * carry_v
            self._attach_held_toy()
            if self._arm_blend >= 1.0:
                self.pick_phase = None
                self.pick_target = None

        elif self.pick_phase == "place":
            self._arm_blend = min(1.0, self.pick_t / 40.0)
            for idx in PLACE_ARM:
                carry_v = CARRY_ARM[idx]
                place_v = PLACE_ARM[idx]
                self.data.ctrl[idx] = (1 - self._arm_blend) * carry_v + self._arm_blend * place_v
            self._attach_held_toy()
            if self._arm_blend >= 1.0 and self.held_toy is not None:
                name = self.held_toy
                self.held_toy = None
                bp = self._bin_pos()
                self._set_toy_pose(name, np.array([bp[0], bp[1], 0.18]))
                self.deposited.add(name)
                self.mission_idx += 1
                self.pick_phase = "retract"
                self.pick_t = 0

        elif self.pick_phase == "retract":
            self._arm_blend = min(1.0, self.pick_t / 35.0)
            for idx, place_v in PLACE_ARM.items():
                stand_v = self.stand_ctrl[idx]
                self.data.ctrl[idx] = (1 - self._arm_blend) * place_v + self._arm_blend * stand_v
            if self._arm_blend >= 1.0:
                self.pick_phase = None
                self.pick_target = None

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)
        forward, strafe, yaw_rate, grip = action

        assert self.stand_ctrl is not None
        self.data.ctrl[:] = self.stand_ctrl

        animating = self.pick_phase is not None
        if not animating:
            t = self._step_count * 0.12
            walk_amp = 0.22 * max(abs(forward), abs(strafe))
            self.data.ctrl[0] = self.stand_ctrl[0] + walk_amp * np.sin(t)
            self.data.ctrl[3] = self.stand_ctrl[3] + walk_amp * (1 - np.cos(t)) * 0.5
            self.data.ctrl[6] = self.stand_ctrl[6] + walk_amp * np.sin(t + np.pi)
            self.data.ctrl[9] = self.stand_ctrl[9] + walk_amp * (1 - np.cos(t + np.pi)) * 0.5

            yaw = self._pelvis_yaw() + yaw_rate * 0.08
            c, s = np.cos(yaw), np.sin(yaw)
            vx = (forward * c - strafe * s) * self.walk_speed
            vy = (forward * s + strafe * c) * self.walk_speed
            dt = self.model.opt.timestep * self.n_substeps
            self.data.qpos[0] += vx * dt
            self.data.qpos[1] += vy * dt
            self.data.qpos[2] = 0.79
            self._set_yaw(yaw)
            self.data.qvel[0:6] = 0
            self.data.qpos[0] = float(np.clip(self.data.qpos[0], -3.5, 3.5))
            self.data.qpos[1] = float(np.clip(self.data.qpos[1], -3.5, 3.5))

            # Trigger animated pick / place
            target = self._current_target()
            if grip > 0.5 and self.held_toy is None and target is not None:
                if np.linalg.norm(self._toy_pos(target)[:2] - self._pelvis_xy()) < 0.7:
                    self._start_pick(target)
            if grip < -0.5 and self.held_toy is not None and self.pick_phase is None:
                if np.linalg.norm(self._bin_pos()[:2] - self._pelvis_xy()) < 0.85:
                    self._start_place()
        else:
            # Hold root still while arm moves
            self.data.qpos[2] = 0.79
            self.data.qvel[0:6] = 0

        if self.pick_phase is not None:
            self._advance_pick_anim()
        elif self.held_toy is not None:
            self._apply_arm_pose(CARRY_ARM, 1.0)
            self._attach_held_toy()

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Keep grasp/carry poses locked after physics (toys are free joints).
        if self.pick_phase == "grasp" and self.pick_target is not None:
            hand = self._hand_pos()
            self._set_toy_pose(self.pick_target, hand + np.array([0.0, 0.0, -0.03]))
            mujoco.mj_forward(self.model, self.data)
        elif self.held_toy is not None and self.pick_phase != "retract":
            if self.pick_phase == "place":
                self._attach_held_toy()
            elif self.pick_phase in (None, "lift"):
                self._attach_held_toy()
            mujoco.mj_forward(self.model, self.data)

        self._update_halo()
        self._step_count += 1
        n_done = len(self.deposited)
        reward = 0.1 * n_done + (0.05 if self.held_toy else 0.0)
        terminated = n_done >= len(TOY_NAMES)
        if terminated:
            reward += 20.0
        truncated = self._step_count >= self.max_episode_steps
        info = {
            "target": self._current_target(),
            "held": self.held_toy,
            "deposited": sorted(self.deposited),
            "n_done": n_done,
            "success": terminated,
            "toys": TOY_LABELS,
            "pick_phase": self.pick_phase,
        }
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.mujoco_renderer is None:
            return None
        try:
            # Create viewer on first call, then keep a wide tracking shot on the robot.
            if self.mujoco_renderer.viewer is None:
                self.mujoco_renderer.render(self.render_mode)
            self._track_camera()
            return self.mujoco_renderer.render(self.render_mode)
        except TypeError:
            return None

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
            self.mujoco_renderer = None


class RoomCleanupExpert:
    """Walk to each toy, wait for visible reach/grasp, deliver to bin."""

    def __init__(self, env: RoomCleanupEnv):
        self.env = env
        self.phase = "go_toy"

    def reset(self) -> None:
        self.phase = "go_toy"

    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs
        action = np.zeros(4, dtype=np.float64)

        # Freeze locomotion while the arm animates.
        if self.env.busy_picking():
            return action.astype(np.float32)

        xy = self.env._pelvis_xy()
        yaw = self.env._pelvis_yaw()

        if self.env.held_toy is None:
            target_name = self.env._current_target()
            if target_name is None:
                return action.astype(np.float32)
            goal = self.env._toy_pos(target_name)[:2]
            self.phase = "go_toy"
            want_pick = True
        else:
            goal = self.env._bin_pos()[:2]
            self.phase = "go_bin"
            want_pick = False

        delta = goal - xy
        dist = float(np.linalg.norm(delta))
        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

        action[2] = np.clip(yaw_err / 0.35, -1.0, 1.0)
        if abs(yaw_err) < 0.4:
            action[0] = np.clip(dist / 0.45, 0.0, 1.0)
        else:
            action[0] = 0.12

        if want_pick and dist < 0.55 and abs(yaw_err) < 0.35:
            action[0] = 0.0
            action[3] = 1.0
        elif (not want_pick) and dist < 0.7 and abs(yaw_err) < 0.4:
            action[0] = 0.0
            action[3] = -1.0

        return action.astype(np.float32)


def register_room_env() -> None:
    try:
        gym.spec("G1RoomCleanup-v0")
    except gym.error.Error:
        gym.register(
            id="G1RoomCleanup-v0",
            entry_point="robot_gym.envs.room_cleanup:RoomCleanupEnv",
            max_episode_steps=6000,
        )
