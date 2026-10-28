"""Industrial grasp-fail → offset retry for LIBERO + π₀ / π₀.₅.

If the target object is not held, do not keep executing place-like policy chunks.
Instead: open, stay near the object, shift 1–3 cm toward/around it, grasp, lift-test.

Oracle hold check uses sim object pose (valid for this LIBERO experiment).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]

try:
    from lerobot_libero_deployable_critic import CarryHeightCritic, CollideDeployableCritic
except ImportError:  # pragma: no cover
    CarryHeightCritic = None  # type: ignore
    CollideDeployableCritic = None  # type: ignore

# (task_id, episode_index, seed, label, original_fail_trajectory.csv)
RETRY_SPECS = [
    (
        8,
        2,
        1802,
        "pudding_seed1802",
        ROBOT_GYM_ROOT
        / "outputs/libero_inference/failure_videos/pi0/pudding_seed1802/trajectory.csv",
    ),
    (
        2,
        0,
        1200,
        "dressing_seed1200",
        ROBOT_GYM_ROOT
        / "outputs/libero_inference/failure_videos/pi0/dressing_seed1200/trajectory.csv",
    ),
    (
        9,
        0,
        1900,
        "oj_seed1900",
        ROBOT_GYM_ROOT
        / "outputs/libero_inference/failure_videos/pi0/oj_seed1900/trajectory.csv",
    ),
]

# Empirically on this LIBERO/Panda setup (gripper qpos):
#   action_6 = +1 → jaws CLOSE (qpos → ~0)
#   action_6 = -1 → jaws OPEN  (qpos → ~±0.04)
GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0
CONTROL_HZ = 20

# Dimensionless / time-based only — distances come from object/basket AABB.
TALL_ASPECT = 1.5  # height / max(xy) → pinch on the body, not the rim
PINCH_HEIGHT_FRAC_TALL = 0.30  # fraction of object height above COM
PINCH_HEIGHT_FRAC_SHORT = 0.30
LEAVE_XY_FRACS = 2.0  # × object XY radius
LEAVE_Z_FRACS = 1.5  # × object height
HOLD_LIFT_FRAC = 0.25  # × object height
HOLD_XY_FRACS = 1.5  # × object XY radius
PLACE_IN_BASKET_FRAC = 0.35  # × basket XY size
DEEPEN_FRAC = 0.40  # × object height per empty-close
LEAVE_S = 0.4
DROP_S = 0.4
MIN_POLICY_S = 2.25


def _steps(seconds: float) -> int:
    return max(1, int(round(seconds * CONTROL_HZ)))


def unwrap_bddl_env(env):
    inner = env
    if getattr(inner, "_env", None) is not None:
        inner = inner._env
    if getattr(inner, "env", None) is not None and hasattr(inner.env, "sim"):
        inner = inner.env
    return inner


def clear_policy_queue(policy) -> None:
    if hasattr(policy, "_action_queue"):
        policy._action_queue.clear()
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict):
        for q in queues.values():
            if isinstance(q, deque):
                q.clear()


def _clip_action(vec: np.ndarray) -> np.ndarray:
    out = np.zeros(7, dtype=np.float32)
    out[: len(vec)] = vec[:7]
    return np.clip(out, -1.0, 1.0)


class GraspRetryController:
    """After leave-without-object: offset-retry grasp, then scripted place with grip locked closed.

    place_mode:
      - "legacy_high": high carry + release (can sky-drop into basket)
      - "low_safe": modest clearance + only open after a real lower over basket
    handback_on_hold:
      - if True, after lift-test HOLD go to mode "handback" (caller returns to π₀ for place)
    """

    def __init__(
        self,
        env,
        *,
        place_mode: str = "legacy_high",
        handback_on_hold: bool = False,
        collision_lift: bool = True,
    ):
        self.env = env
        self.sim_env = unwrap_bddl_env(env)
        self.target_name = self.sim_env.obj_of_interest[0]
        interest = list(self.sim_env.obj_of_interest)
        self.basket_name = interest[1] if len(interest) > 1 else "basket_1"
        self.place_mode = place_mode
        self.handback_on_hold = handback_on_hold
        self.collision_lift = collision_lift
        self.init_obj_z = float(self.obj_pos()[2])
        self.eef_z0 = float(self.eef_pos()[2])
        self.grip_open_gap = max(self.gripper_gap(), 1e-3)
        self.mode = "policy"  # policy | retry | place | handback | done
        self.n_sequences = 0
        self.offset_i = 0
        self.phase = "approach"
        self.phase_t = 0
        self.leave_streak = 0
        self.drop_streak = 0
        self.takeover_step: int | None = None
        self.z_offset = self._pinch_height_offset()
        self.n_deepen = 0
        self._last_image: np.ndarray | None = None
        self.height_critic = CarryHeightCritic() if CarryHeightCritic is not None else None
        self.collide_critic = (
            CollideDeployableCritic() if CollideDeployableCritic is not None else None
        )
        self._place_basket0: np.ndarray | None = None
        self._clear_z: float = 0.0
        self._push_streak: int = 0
        self._last_lateral_cmd: float = 0.0
        size = self.obj_extent()
        print(
            f"  geom {self.target_name} size={size} tall={self.is_tall()} "
            f"pinch_off={self.z_offset:.3f} carry_z={self.carry_z():.3f} "
            f"place_xy={self.place_xy_ok():.3f} place_mode={self.place_mode} "
            f"handback_on_hold={self.handback_on_hold} collision_lift={self.collision_lift}"
        )

    def obj_pos(self) -> np.ndarray:
        body_id = self.sim_env.obj_body_id[self.target_name]
        return np.asarray(self.sim_env.sim.data.body_xpos[body_id], dtype=np.float64)

    def basket_pos(self) -> np.ndarray:
        body_id = self.sim_env.obj_body_id[self.basket_name]
        return np.asarray(self.sim_env.sim.data.body_xpos[body_id], dtype=np.float64)

    def eef_pos(self) -> np.ndarray:
        robot = self.sim_env.robots[0]
        return np.asarray(self.sim_env.sim.data.site_xpos[robot.eef_site_id], dtype=np.float64)

    def eef_rot(self) -> np.ndarray:
        """EE rotation matrix (3x3) from MuJoCo site — proprio, not object pose."""
        robot = self.sim_env.robots[0]
        return np.asarray(
            self.sim_env.sim.data.site_xmat[robot.eef_site_id], dtype=np.float64
        ).reshape(3, 3)

    def _subtree_body_ids(self, body_id: int) -> set[int]:
        model = self.sim_env.sim.model
        ids = {int(body_id)}
        changed = True
        while changed:
            changed = False
            for i in range(model.nbody):
                if int(model.body_parentid[i]) in ids and i not in ids:
                    ids.add(i)
                    changed = True
        return ids

    def aabb_of(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World AABB from collision geoms (box/cyl/capsule), not bounding-sphere rbound."""
        sim = self.sim_env.sim
        model = sim.model
        body_id = int(self.sim_env.obj_body_id[name])
        bodies = self._subtree_body_ids(body_id)
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        # mjGEOM: sphere=2 capsule=3 ellipsoid=4 cylinder=5 box=6
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) not in bodies:
                continue
            if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
                continue
            pos = np.asarray(sim.data.geom_xpos[g], dtype=np.float64)
            mat = np.asarray(sim.data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
            size = np.asarray(model.geom_size[g], dtype=np.float64)
            typ = int(model.geom_type[g])
            if typ == 2:
                half = np.array([size[0], size[0], size[0]])
            elif typ == 3:
                half = np.array([size[0], size[0], size[1] + size[0]])
            elif typ == 4:
                half = size[:3].copy()
            elif typ == 5:
                half = np.array([size[0], size[0], size[1]])
            elif typ == 6:
                half = size[:3].copy()
            else:
                r = float(model.geom_rbound[g])
                if r <= 0:
                    continue
                half = np.array([r, r, r])
            ext = np.abs(mat) @ half
            lo = np.minimum(lo, pos - ext)
            hi = np.maximum(hi, pos + ext)
        if not np.isfinite(lo).all():
            p = self.obj_pos() if name == self.target_name else self.basket_pos()
            return p.copy(), p.copy()
        return lo, hi

    def obj_extent(self) -> np.ndarray:
        lo, hi = self.aabb_of(self.target_name)
        return np.maximum(hi - lo, 1e-3)

    def basket_extent(self) -> np.ndarray:
        lo, hi = self.aabb_of(self.basket_name)
        return np.maximum(hi - lo, 1e-3)

    def obj_xy_radius(self) -> float:
        e = self.obj_extent()
        return 0.5 * float(max(e[0], e[1]))

    def obj_pinch_width(self) -> float:
        e = self.obj_extent()
        return float(min(e[0], e[1]))

    def is_tall(self) -> bool:
        e = self.obj_extent()
        return float(e[2]) > TALL_ASPECT * float(max(e[0], e[1]))

    def _pinch_height_offset(self) -> float:
        h = float(self.obj_extent()[2])
        frac = PINCH_HEIGHT_FRAC_TALL if self.is_tall() else PINCH_HEIGHT_FRAC_SHORT
        return frac * h

    def xy_offsets(self) -> tuple[tuple[float, float], ...]:
        r = self.obj_xy_radius()
        return (
            (0.0, 0.0),
            (r, 0.0),
            (-r, 0.0),
            (0.0, r),
            (0.0, -r),
            (r, r),
            (-r, r),
            (r, -r),
        )

    def carry_z(self) -> float:
        h = float(self.obj_extent()[2])
        _, b_hi = self.aabb_of(self.basket_name)
        basket_top = float(b_hi[2])
        if self.place_mode == "low_safe":
            # Modest default carry — only raise floor after a detected collision.
            base = float(max(self.init_obj_z + 1.15 * h, basket_top + 0.65 * h, 0.14))
            return float(max(base, self._clear_z))
        obj_top = float(self.obj_pos()[2]) + 0.5 * h
        return max(self.eef_z0, obj_top + h, basket_top + h)

    def clearance_z(self) -> float:
        """Lift-to height after a basket collision (not used as a proactive carry target)."""
        h = float(self.obj_extent()[2])
        _, b_hi = self.aabb_of(self.basket_name)
        basket_top = float(b_hi[2])
        return float(max(basket_top + (1.05 * h if self.is_tall() else 0.7 * h), 0.22))

    def release_z(self) -> float:
        """EE height at which opening is allowed (low_safe)."""
        h = float(self.obj_extent()[2])
        _, b_hi = self.aabb_of(self.basket_name)
        basket_top = float(b_hi[2])
        return basket_top + 0.35 * h + 0.02

    def clearly_placing(self, *, ee: np.ndarray | None = None) -> bool:
        """Near basket and low enough that opening is a place, not a mid-air drop."""
        ee = self.eef_pos() if ee is None else np.asarray(ee, dtype=np.float64)
        basket = self.basket_pos()
        xy = float(np.linalg.norm(ee[:2] - basket[:2]))
        return xy < 1.5 * self.place_xy_ok() and float(ee[2]) < self.release_z() + 0.05

    def place_xy_ok(self) -> float:
        b = self.basket_extent()
        return PLACE_IN_BASKET_FRAC * float(max(b[0], b[1]))

    def _robosuite_grasp(self) -> bool:
        try:
            robot = self.sim_env.robots[0]
            obj = self.sim_env.get_object(self.target_name)
            return bool(robot._check_grasp(gripper=robot.gripper, object_geoms=obj))
        except Exception:
            return False

    def holding(self) -> bool:
        obj = self.obj_pos()
        ee = self.eef_pos()
        h = float(self.obj_extent()[2])
        lifted = float(obj[2]) > self.init_obj_z + HOLD_LIFT_FRAC * h
        near = float(np.linalg.norm(obj[:2] - ee[:2])) < HOLD_XY_FRACS * self.obj_xy_radius()
        return bool(self._robosuite_grasp() or (lifted and near))

    def leaving(self) -> bool:
        obj = self.obj_pos()
        ee = self.eef_pos()
        xy = float(np.linalg.norm(obj[:2] - ee[:2]))
        return (
            xy > LEAVE_XY_FRACS * self.obj_xy_radius()
            and float(ee[2]) > float(obj[2]) + LEAVE_Z_FRACS * float(self.obj_extent()[2])
        )

    def grasp_z_target(self) -> float:
        z = float(self.obj_pos()[2])
        h = float(self.obj_extent()[2])
        if self.is_tall():
            zmin, zmax = z + 0.15 * h, z + 0.7 * h
        else:
            zmin, zmax = z - 0.2 * h, z + 0.6 * h
        return float(np.clip(z + self.z_offset, zmin, zmax))

    def gripper_gap(self) -> float:
        return float(np.mean(np.abs(self.gripper_qpos())))

    def pinch_missed_high(self) -> bool:
        """Jaws closed (or nearly) while the object is still on the table → pinch too high."""
        if self.holding():
            return False
        h = float(self.obj_extent()[2])
        obj_still_down = float(self.obj_pos()[2]) < self.init_obj_z + 0.2 * h
        wrap_gap = max(0.5 * self.obj_pinch_width(), 0.2 * self.grip_open_gap)
        empty = self.gripper_gap() < 0.45 * wrap_gap
        return obj_still_down and empty

    def deepen_pinch(self, *, why: str) -> bool:
        """Open a little, drop pinch height, close again. Same XY."""
        h = float(self.obj_extent()[2])
        floor = (0.15 * h) if self.is_tall() else (-0.5 * h)
        if self.n_deepen >= 6 or self.z_offset <= floor:
            return False
        self.n_deepen += 1
        self.z_offset -= DEEPEN_FRAC * h
        self.phase = "nibble_open"
        self.phase_t = 0
        print(
            f"    -> {why} empty close gap={self.gripper_gap():.4f} "
            f"lower pinch z_offset={self.z_offset:.3f} deepen={self.n_deepen}"
        )
        return True

    def start_retry(self, *, new_sequence: bool = True) -> None:
        self.mode = "retry"
        self.phase = "approach"
        self.phase_t = 0
        if new_sequence:
            self.n_sequences += 1

    def observe_image(self, image: np.ndarray | None) -> None:
        if image is not None:
            self._last_image = np.asarray(image)

    def start_place(self) -> None:
        self.drop_streak = 0
        self._push_streak = 0
        self._clear_z = 0.0  # raised only after deployable collide detect
        self._last_lateral_cmd = 0.0
        ee = self.eef_pos()
        rot = self.eef_rot()
        # Arm deployable critics at HOLD — image + EE pose only.
        if self.height_critic is not None and self._last_image is not None:
            self.height_critic.arm(ee_z=float(ee[2]), image=self._last_image)
            print(
                f"    -> height critic armed  "
                f"band=[{self.height_critic.z_lo:.3f},{self.height_critic.z_hi:.3f}]"
            )
        if self.collide_critic is not None and self._last_image is not None:
            self.collide_critic.arm(image=self._last_image, ee_pos=ee, ee_rot=rot)
            print("    -> collide critic armed (vision + EE xyz/rot only; no basket pose)")
        if self.handback_on_hold:
            self.mode = "handback"
            self.phase = "handback"
            self.phase_t = 0
            print(
                f"    -> HOLD OK — hand back to π₀ for place "
                f"(grip locked until clearly placing) "
                f"obj_z={self.obj_pos()[2]:.3f} basket={self.basket_pos()}"
            )
            return
        self.mode = "place"
        self.phase = "place_lift"
        self.phase_t = 0
        print(
            f"    -> HOLD OK — lift then carry (gripper stays CLOSED) "
            f"place_mode={self.place_mode} "
            f"obj_z={self.obj_pos()[2]:.3f} basket={self.basket_pos()} "
            f"carry_z={self.carry_z():.3f}"
        )

    def _carry_target_z(self) -> float:
        return float(self.carry_z())

    def _deployable_collide(self, *, ee: np.ndarray, active: bool) -> bool:
        """Vision + EE proprio collide bit. No basket/object poses."""
        if (
            not self.collision_lift
            or self.collide_critic is None
            or self._last_image is None
            or self.place_mode != "low_safe"
        ):
            return False
        dbg = self.collide_critic.step(
            image=self._last_image,
            ee_pos=ee,
            ee_rot=self.eef_rot(),
            lateral_cmd=self._last_lateral_cmd,
            active=active,
        )
        return bool(dbg.collide)

    def _abort_place_if_dropped(self) -> bool:
        if self.holding():
            self.drop_streak = 0
            return False
        self.drop_streak += 1
        if self.drop_streak < _steps(DROP_S):
            return False
        print("    -> dropped during place — regrasp")
        self.offset_i += 1
        self.start_retry(new_sequence=False)
        return True

    def failed_pick_confirmed(self) -> bool:
        if self.leaving():
            self.leave_streak += 1
        else:
            self.leave_streak = 0
        return self.leave_streak >= _steps(LEAVE_S)

    def maybe_takeover(self, *, step: int, is_new_chunk: bool) -> bool:
        del is_new_chunk
        if self.mode in ("place", "retry", "done", "handback"):
            # handback: caller drives π₀; we stay "in control" only for grip lock helpers
            return self.mode != "handback"
        if step < _steps(MIN_POLICY_S):
            return False
        if self.failed_pick_confirmed():
            self.takeover_step = step
            self.start_retry()
            print(
                f"  step {step}: DETECTED same failure (left without {self.target_name}) "
                f"— start offset retry"
            )
            return True
        return False

    def lock_grip_for_policy_action(self, action: np.ndarray) -> np.ndarray:
        """While handback: force CLOSE unless EE is clearly placing over the basket."""
        out = _clip_action(np.asarray(action, dtype=np.float64).reshape(-1))
        if self.mode != "handback":
            return out
        if self.clearly_placing():
            # Allow policy open/close; if it opens, treat as place attempt finishing.
            if float(out[6]) < 0.0:
                self.phase_t += 1
                if self.phase_t >= _steps(0.6):
                    self.mode = "done"
                    print("    -> π₀ place open accepted — done")
            return out
        out[6] = GRIP_CLOSE
        return out

    def gripper_qpos(self) -> np.ndarray:
        robot = self.sim_env.robots[0]
        try:
            names = robot.gripper.joints
            return np.array(
                [self.sim_env.sim.data.get_joint_qpos(n) for n in names],
                dtype=np.float64,
            )
        except Exception:
            return np.zeros(2)

    def scripted_action(self, image: np.ndarray | None = None) -> np.ndarray:
        self.observe_image(image)
        if self.mode == "done":
            return _clip_action(np.array([0.0, 0.0, 0.0, 0, 0, 0, GRIP_OPEN]))
        if self.mode == "handback":
            return _clip_action(np.array([0.0, 0.0, 0.0, 0, 0, 0, GRIP_CLOSE]))
        if self.mode == "place":
            return self._place_action()
        return self._pick_action()

    def _height_az(
        self, *, ee_z: float, allow_descent: bool, force_raise: bool = False
    ) -> float:
        if self.height_critic is None or self._last_image is None or self.place_mode != "low_safe":
            return 0.85 if force_raise else 0.0
        dbg = self.height_critic.step(
            ee_z=ee_z,
            image=self._last_image,
            allow_descent=allow_descent,
            force_raise=force_raise and self.collision_lift,
            # Scene MAD false-fires on arm motion; lift only on real basket collide.
            use_scene_disturb=False,
        )
        if dbg.collision_raise and self.phase_t % 5 == 1:
            print(
                f"    -> height critic COLLISION ee_z={dbg.ee_z:.3f} → lift (pause XY)"
            )
        elif dbg.too_low and self.phase_t % 10 == 1:
            print(
                f"    -> height critic TOO LOW ee_z={dbg.ee_z:.3f} "
                f"< {dbg.z_lo:.3f} → raise"
            )
        if dbg.too_high and self.phase_t % 10 == 1:
            print(
                f"    -> height critic TOO HIGH ee_z={dbg.ee_z:.3f} "
                f"> {dbg.z_hi:.3f} → lower"
            )
        return float(dbg.az_correction)

    def _place_action(self) -> np.ndarray:
        """Carry low by default; lift only on deployable collide (vision+EE).

        Note: scripted XY still aims with basket pose (oracle recovery actuator).
        Collide *detection* does not use basket/object poses.
        """
        if self.phase != "place_release" and self._abort_place_if_dropped():
            return self._pick_action()

        ee = self.eef_pos()
        basket = self.basket_pos()
        xy_err = basket[:2] - ee[:2]
        xy_dist = float(np.linalg.norm(xy_err))
        xy_ok = self.place_xy_ok()
        target = self._carry_target_z()
        obj_h = float(self.obj_extent()[2])
        basket_h = float(self.basket_extent()[2])
        self.phase_t += 1

        def closed(ax: float, ay: float, az: float) -> np.ndarray:
            return _clip_action(np.array([ax, ay, az, 0, 0, 0, GRIP_CLOSE]))

        # Critic intent from phase/FSM only — not from basket error.
        if self.phase == "place_goto":
            self._last_lateral_cmd = 0.5
        elif self.phase == "place_lower":
            self._last_lateral_cmd = 0.25
        else:
            self._last_lateral_cmd = 0.0

        active_collide = self.phase in ("place_goto", "place_lower")
        collided = self._deployable_collide(ee=ee, active=active_collide)
        if collided:
            self._push_streak += 1
        else:
            self._push_streak = max(0, self._push_streak - 1)

        need_lift = self._push_streak >= 1  # critic already has internal streak
        if need_lift and self.collision_lift and self.place_mode == "low_safe":
            # Lift target from EE proprio only (+fixed delta) — no basket AABB.
            if self._clear_z <= 0.0:
                self._clear_z = float(ee[2]) + 0.10
                if self.height_critic is not None:
                    self.height_critic.raise_floor(min_z=self._clear_z)
                cd = self.collide_critic.last_debug if self.collide_critic else None
                print(
                    f"    -> COLLIDE (vision+EE) scene_mad={getattr(cd,'scene_mad',0):.3f} "
                    f"stall={getattr(cd,'stalled',False)} "
                    f"xy_spd={getattr(cd,'ee_xy_speed',0):.4f} "
                    f"— lift once to z≥{self._clear_z:.3f}"
                )
                self.phase = "place_lift"
                self.phase_t = 0

            if float(ee[2]) >= self._clear_z - 0.01:
                self._push_streak = 0
                if self.collide_critic is not None and self._last_image is not None:
                    self.collide_critic.reanchor(
                        image=self._last_image, ee_pos=ee, ee_rot=self.eef_rot()
                    )
                print(
                    f"    -> clear after collide ee_z={ee[2]:.3f} "
                    f"— resume carry floor={self._clear_z:.3f}"
                )
            else:
                az = float(np.clip(8.0 * (self._clear_z - float(ee[2])), 0.35, 1.0))
                return closed(0.0, 0.0, az)

        if self.phase == "place_lift":
            az = float(np.clip(8.0 * (target - float(ee[2])), -0.2, 0.9))
            action = closed(0.0, 0.0, az)
            if float(ee[2]) >= target - 0.01 or self.phase_t >= _steps(2.5):
                self.phase = "place_goto"
                self.phase_t = 0
                print(
                    f"    -> carry to basket at z={ee[2]:.3f} "
                    f"xy_dist={xy_dist:.3f} holding={self.holding()}"
                )
            return action

        if self.phase == "place_goto":
            ax = float(np.clip(10.0 * xy_err[0], -0.8, 0.8))
            ay = float(np.clip(10.0 * xy_err[1], -0.8, 0.8))
            az = float(np.clip(6.0 * (target - float(ee[2])), -0.25, 0.55))
            action = closed(ax, ay, float(np.clip(az, -0.35, 0.7)))
            if xy_dist < xy_ok and float(ee[2]) >= self.release_z() + 0.03:
                self.phase = "place_lower"
                self.phase_t = 0
                print(f"    -> over basket, lower  xy_dist={xy_dist:.3f} ee_z={ee[2]:.3f}")
            elif self.phase_t >= _steps(8.0):
                print(f"    -> still not over basket xy_dist={xy_dist:.3f} (keep carrying)")
                self.phase_t = _steps(4.0)
            return action

        if self.phase == "place_lower":
            if xy_dist > xy_ok * 1.5:
                self.phase = "place_goto"
                self.phase_t = 0
                return closed(0.0, 0.0, 0.4)
            if self.place_mode == "low_safe":
                drop_z = self.release_z()
            else:
                drop_z = float(basket[2]) + 0.5 * obj_h + 0.25 * basket_h
            drop = np.array([basket[0], basket[1], drop_z], dtype=np.float64)
            err = drop - ee
            az = float(np.clip(6.0 * err[2], -0.35, 0.2))
            action = closed(
                float(np.clip(8.0 * err[0], -0.4, 0.4)),
                float(np.clip(8.0 * err[1], -0.4, 0.4)),
                az,
            )
            low_enough = float(ee[2]) <= drop_z + 0.02
            if self.place_mode == "low_safe":
                if xy_dist < xy_ok and low_enough:
                    self.phase = "place_release"
                    self.phase_t = 0
                    print(f"    -> place release (low_safe) xy_dist={xy_dist:.3f} ee_z={ee[2]:.3f}")
                elif self.phase_t >= _steps(6.0):
                    print(f"    -> still lowering xy_dist={xy_dist:.3f} ee_z={ee[2]:.3f}")
                    self.phase_t = _steps(3.0)
            else:
                if (xy_dist < xy_ok and low_enough) or self.phase_t >= _steps(2.5):
                    self.phase = "place_release"
                    self.phase_t = 0
                    print(f"    -> place release  xy_dist={xy_dist:.3f} ee_z={ee[2]:.3f}")
            return action

        if xy_dist > xy_ok * 1.5:
            self.phase = "place_goto"
            self.phase_t = 0
            return closed(
                float(np.clip(10.0 * xy_err[0], -0.8, 0.8)),
                float(np.clip(10.0 * xy_err[1], -0.8, 0.8)),
                0.4,
            )
        if self.place_mode == "low_safe" and float(ee[2]) > self.release_z() + 0.04:
            self.phase = "place_lower"
            self.phase_t = 0
            return closed(0.0, 0.0, -0.25)
        action = _clip_action(np.array([0.0, 0.0, 0.0, 0, 0, 0, GRIP_OPEN]))
        if self.phase_t >= _steps(1.0):
            self.mode = "done"
            print("    -> place done (gate stays on; no π₀ reopen)")
        return action

    def _pick_action(self) -> np.ndarray:
        obj = self.obj_pos()
        ee = self.eef_pos()
        offsets = self.xy_offsets()
        ox, oy = offsets[self.offset_i % len(offsets)]
        target_xy = obj[:2] + np.array([ox, oy], dtype=np.float64)
        xy_err = target_xy - ee[:2]
        xy_dist = float(np.linalg.norm(xy_err))
        grasp_z = self.grasp_z_target()
        z_err = grasp_z - float(ee[2])
        h = float(self.obj_extent()[2])
        xy_tol = 0.4 * self.obj_xy_radius()

        def toward_xy(scale: float = 8.0) -> tuple[float, float]:
            return (
                float(np.clip(scale * xy_err[0], -0.8, 0.8)),
                float(np.clip(scale * xy_err[1], -0.8, 0.8)),
            )

        self.phase_t += 1

        if self.phase == "approach":
            ax, ay = toward_xy(12.0)
            hover_z = grasp_z + h
            az = float(np.clip(6.0 * (hover_z - float(ee[2])), -0.5, 0.5))
            action = _clip_action(np.array([ax, ay, az, 0, 0, 0, GRIP_OPEN]))
            if xy_dist < xy_tol or self.phase_t >= _steps(2.0):
                self.phase = "descend"
                self.phase_t = 0
                print(
                    f"    -> descend  xy_dist={xy_dist:.3f} ee_z={ee[2]:.3f} "
                    f"grasp_z={grasp_z:.3f} tall={self.is_tall()}"
                )
            return action

        if self.phase == "descend":
            ax, ay = toward_xy(8.0)
            az = float(np.clip(12.0 * z_err, -1.0, 0.2))
            action = _clip_action(np.array([ax, ay, az, 0, 0, 0, GRIP_OPEN]))
            slack = 0.15 * h
            near_z = float(ee[2]) <= grasp_z + slack
            if near_z or self.phase_t >= _steps(3.5):
                self.phase = "close"
                self.phase_t = 0
                print(
                    f"    -> CLOSE gripper  ee_z={ee[2]:.3f} obj_z={obj[2]:.3f} "
                    f"grasp_z={grasp_z:.3f} z_off={self.z_offset:.3f} gq={self.gripper_qpos()}"
                )
            return action

        if self.phase == "nibble_open":
            ax, ay = toward_xy(4.0)
            action = _clip_action(np.array([ax, ay, -0.15, 0, 0, 0, GRIP_OPEN]))
            if self.phase_t >= _steps(0.6):
                self.phase = "descend"
                self.phase_t = 0
                print(f"    -> re-descend to grasp_z={grasp_z:.3f}")
            return action

        if self.phase == "close":
            ax, ay = toward_xy(3.0)
            az = float(np.clip(8.0 * z_err, -0.6, 0.05))
            action = _clip_action(np.array([ax, ay, az, 0, 0, 0, GRIP_CLOSE]))
            close_needed = _steps(2.5 if self.is_tall() else 1.75)
            if self.phase_t >= close_needed or self.holding():
                self.phase = "lift"
                self.phase_t = 0
                print(
                    f"    -> lift test  holding={self.holding()} ee_z={ee[2]:.3f} "
                    f"gap={self.gripper_gap():.4f} gq={self.gripper_qpos()}"
                )
            return action

        action = _clip_action(np.array([0.0, 0.0, 0.5, 0, 0, 0, GRIP_CLOSE]))
        obj_now = float(self.obj_pos()[2])
        lifted_obj = obj_now > self.init_obj_z + HOLD_LIFT_FRAC * h
        if self.holding() and lifted_obj:
            self.start_place()
            return action
        contact = self.gripper_gap() < 0.85 * self.grip_open_gap
        if self.phase_t >= _steps(1.0) and self.pinch_missed_high() and not contact:
            if self.deepen_pinch(why="lift"):
                return action
        lift_budget = _steps(3.0) if contact else _steps(2.0)
        if self.phase_t >= lift_budget:
            self.offset_i += 1
            print(
                f"    -> miss, next offset={self.offset_i} "
                f"gap={self.gripper_gap():.4f} z_off={self.z_offset:.3f}"
            )
            self.start_retry(new_sequence=False)
        return action


def run_retry_eval() -> None:
    sys.path.insert(0, str(ROBOT_GYM_ROOT / "scripts"))
    from lerobot_libero_collect import (
        check_hf_auth,
        load_policy_bundle,
        patch_robosuite,
        run_collect_episode,
        setup_paths,
    )
    from lerobot_libero_live import make_live_libero_env

    setup_paths()
    patch_robosuite()
    check_hf_auth()

    policy_path = "lerobot/pi0_libero_finetuned"
    out_root = ROBOT_GYM_ROOT / "outputs" / "libero_inference" / "grasp_retry" / "pi0_same_fail"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== grasp-retry eval (replay same fail, then retry): {policy_path} ===")
    policy, pre, post, env_pre, env_post, policy_cfg, _ = load_policy_bundle(
        policy_path, "mps", "libero_object", RETRY_SPECS[0][0]
    )

    log_rows: list[dict] = []
    only = {a for a in sys.argv[1:] if not a.startswith("-")}
    for task_id, ep, seed, label, fail_csv in RETRY_SPECS:
        if only and label not in only:
            continue
        ep_dir = out_root / label
        print(f"  {label} task={task_id} seed={seed} replay={fail_csv.name}")
        env = make_live_libero_env(
            task_suite_name="libero_object",
            task_id=task_id,
            render_mode="opencv",
        )
        try:
            _, _, metrics = run_collect_episode(
                policy=policy,
                preprocessor=pre,
                postprocessor=post,
                env_preprocessor=env_pre,
                env_postprocessor=env_post,
                policy_cfg=policy_cfg,
                policy_path=policy_path,
                env_task="libero_object",
                task_id=task_id,
                env=env,
                episode_index=ep,
                seed=seed,
                save_video=os.environ.get("GRASP_RETRY_SAVE_VIDEO", "1") != "0",
                episode_dir=ep_dir,
                grasp_retry=True,
                replay_fail_csv=fail_csv,
            )
            print(
                f"    success={metrics.success} held={metrics.held_object} "
                f"retries={metrics.n_retry_sequences} steps={metrics.num_steps} "
                f"video={ep_dir / 'video.mp4'}"
            )
            log_rows.append(
                {
                    "policy": "pi0",
                    "label": label,
                    "task_id": task_id,
                    "seed": seed,
                    "success": metrics.success,
                    "held_object": metrics.held_object,
                    "n_retry_sequences": metrics.n_retry_sequences,
                    "num_steps": metrics.num_steps,
                    "video": str(ep_dir / "video.mp4"),
                }
            )
        finally:
            env.close()

    with (out_root / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)
    with (out_root / "summary.json").open("w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"\nDone. {out_root}")


if __name__ == "__main__":
    run_retry_eval()
