"""Deployable leave-empty critic (vision + proprio, no sim object/basket poses).

v0 logic (all three required to fire):
  1) grasp_attempted  — gripper closed (proprio)
  2) ee_left_pick     — EE rose / moved away from grasp pose (proprio)
  3) object_on_table  — agentview table crop still matches grasp-time crop (vision)

GT body_xpos is NEVER used inside this critic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CriticDebug:
    grasp_attempted: bool = False
    ee_left_pick: bool = False
    object_on_table: bool = False
    fire: bool = False
    grip_gap: float = 0.0
    ee_lift: float = 0.0
    ee_xy: float = 0.0
    vision_mad: float = 0.0
    vision_ready: bool = False


@dataclass
class LeaveEmptyDeployableCritic:
    """v0 leave-without-object detector from images + proprio only."""

    # Gripper qpos: open ~±0.04, closed ~0 (same polarity as grasp-retry comments).
    close_gap_thresh: float = 0.02
    empty_gap_thresh: float = 0.007  # nearly fully closed → likely empty miss
    wrap_gap_lo: float = 0.008
    wrap_gap_hi: float = 0.035  # jaws around an object
    close_streak_need: int = 4
    # EE motion relative to pose at first sustained close.
    lift_thresh: float = 0.04  # meters
    xy_thresh: float = 0.06  # meters
    # Vision: mean abs diff on [0,1] table crop vs grasp reference.
    vision_mad_thresh: float = 0.08
    # Object left table (HOLD) — MAD above this.
    hold_mad_thresh: float = 0.07
    # Require EE-left for this many steps before allowing fire (debounce).
    left_streak_need: int = 6

    _close_streak: int = 0
    _left_streak: int = 0
    grasp_attempted: bool = False
    ee_at_grasp: np.ndarray | None = None
    table_ref: np.ndarray | None = None  # float HxWx3 in [0,1]
    last_debug: CriticDebug = field(default_factory=CriticDebug)

    def reset(self) -> None:
        self._close_streak = 0
        self._left_streak = 0
        self.grasp_attempted = False
        self.ee_at_grasp = None
        self.table_ref = None
        self.last_debug = CriticDebug()

    @staticmethod
    def _table_crop(image: np.ndarray) -> np.ndarray:
        """Lower-center crop where LIBERO tabletop objects usually sit."""
        img = np.asarray(image)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if img.max() > 1.5:
                img = img / 255.0
        h, w = img.shape[:2]
        y0, y1 = int(0.35 * h), int(0.95 * h)
        x0, x1 = int(0.15 * w), int(0.85 * w)
        return img[y0:y1, x0:x1]

    def _gap(self, grip_qpos: np.ndarray) -> float:
        return float(np.mean(np.abs(np.asarray(grip_qpos, dtype=np.float64))))

    def jaws_wrapping(self, grip_qpos: np.ndarray) -> bool:
        g = self._gap(grip_qpos)
        return self.wrap_gap_lo < g < self.wrap_gap_hi

    def jaws_empty_closed(self, grip_qpos: np.ndarray) -> bool:
        return self._gap(grip_qpos) < self.empty_gap_thresh

    def likely_holding(self, *, image: np.ndarray, grip_qpos: np.ndarray) -> bool:
        """Cam+EE HOLD proxy: jaws wrapping + table changed vs grasp ref."""
        if self.table_ref is None:
            return False
        if not self.jaws_wrapping(grip_qpos):
            return False
        cur = self._table_crop(image)
        if cur.shape != self.table_ref.shape:
            return False
        mad = float(np.mean(np.abs(cur - self.table_ref)))
        return mad > self.hold_mad_thresh

    def _update_grasp(self, grip_qpos: np.ndarray, ee_pos: np.ndarray, image: np.ndarray) -> None:
        gap = self._gap(grip_qpos)
        if gap < self.close_gap_thresh:
            self._close_streak += 1
        else:
            self._close_streak = 0
        if (not self.grasp_attempted) and self._close_streak >= self.close_streak_need:
            self.grasp_attempted = True
            self.ee_at_grasp = np.asarray(ee_pos, dtype=np.float64).reshape(3).copy()
            self.table_ref = self._table_crop(image).copy()

    def _ee_left(self, ee_pos: np.ndarray) -> bool:
        if self.ee_at_grasp is None:
            return False
        ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        lift = float(ee[2] - self.ee_at_grasp[2])
        xy = float(np.linalg.norm(ee[:2] - self.ee_at_grasp[:2]))
        self.last_debug.ee_lift = lift
        self.last_debug.ee_xy = xy
        return lift > self.lift_thresh or xy > self.xy_thresh

    def _object_on_table(self, image: np.ndarray) -> tuple[bool, float]:
        """If table crop still looks like grasp-time crop, object likely did not move."""
        if self.table_ref is None:
            return False, 0.0
        cur = self._table_crop(image)
        if cur.shape != self.table_ref.shape:
            return False, 1.0
        mad = float(np.mean(np.abs(cur - self.table_ref)))
        return mad < self.vision_mad_thresh, mad

    def step(
        self,
        *,
        image: np.ndarray,
        ee_pos: np.ndarray,
        grip_qpos: np.ndarray,
    ) -> bool:
        """Return True when leave-empty is detected (takeover signal)."""
        gap = self._gap(grip_qpos)
        self._update_grasp(grip_qpos, ee_pos, image)

        # Never treat a wrapped hold as miss-pick.
        if self.jaws_wrapping(grip_qpos) or self.likely_holding(image=image, grip_qpos=grip_qpos):
            on_table, mad = self._object_on_table(image) if self.grasp_attempted else (False, 0.0)
            self.last_debug = CriticDebug(
                grasp_attempted=self.grasp_attempted,
                ee_left_pick=False,
                object_on_table=on_table,
                fire=False,
                grip_gap=gap,
                ee_lift=self.last_debug.ee_lift,
                ee_xy=self.last_debug.ee_xy,
                vision_mad=mad,
                vision_ready=self.table_ref is not None,
            )
            return False

        ee_left = self._ee_left(ee_pos) if self.grasp_attempted else False
        if ee_left:
            self._left_streak += 1
        else:
            self._left_streak = 0
        ee_left_ok = self._left_streak >= self.left_streak_need

        on_table, mad = self._object_on_table(image) if self.grasp_attempted else (False, 0.0)
        empty = self.jaws_empty_closed(grip_qpos)

        fire = bool(self.grasp_attempted and ee_left_ok and on_table and empty)
        self.last_debug = CriticDebug(
            grasp_attempted=self.grasp_attempted,
            ee_left_pick=ee_left_ok,
            object_on_table=on_table,
            fire=fire,
            grip_gap=gap,
            ee_lift=self.last_debug.ee_lift,
            ee_xy=self.last_debug.ee_xy,
            vision_mad=mad,
            vision_ready=self.table_ref is not None,
        )
        return fire


@dataclass
class HeightDebug:
    too_low: bool = False
    too_high: bool = False
    collision_raise: bool = False
    ee_z: float = 0.0
    z_lo: float = 0.0
    z_hi: float = 0.0
    vision_near_table: bool = False
    vision_score: float = 0.0
    scene_mad: float = 0.0
    az_correction: float = 0.0


@dataclass
class CarryHeightCritic:
    """Pre-basket carry-height monitor (vision + proprio, no sim poses).

    Armed at HOLD with a carry-height anchor. During transit:
      - too_low  → suggest positive az (raise)
      - too_high → suggest negative az (lower)
      - collision_raise / scene disturbance → raise and pause lateral motion
    When allow_descent=True (over receptacle / placing), too_low is suppressed
    so the arm may lower to release height.
    """

    band_below: float = 0.02  # meters under carry anchor → too low
    band_above: float = 0.14  # meters over carry anchor → too high
    vision_near_thresh: float = 0.12  # bottom-crop bright / change cue
    scene_mad_thresh: float = 0.08  # mid-frame disturbance vs hold (push/scrape)
    raise_az: float = 0.55
    collision_raise_az: float = 0.85
    lower_az: float = -0.2

    armed: bool = False
    anchor_z: float = 0.0
    z_lo: float = 0.0
    z_hi: float = 0.0
    bottom_ref: np.ndarray | None = None
    scene_ref: np.ndarray | None = None
    last_debug: HeightDebug = field(default_factory=HeightDebug)

    def reset(self) -> None:
        self.armed = False
        self.anchor_z = 0.0
        self.z_lo = 0.0
        self.z_hi = 0.0
        self.bottom_ref = None
        self.scene_ref = None
        self.last_debug = HeightDebug()

    @staticmethod
    def _as_float_img(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        return img

    @classmethod
    def _bottom_crop(cls, image: np.ndarray) -> np.ndarray:
        img = cls._as_float_img(image)
        h, w = img.shape[:2]
        return img[int(0.72 * h) : h, int(0.2 * w) : int(0.8 * w)]

    @classmethod
    def _scene_crop(cls, image: np.ndarray) -> np.ndarray:
        """Lower-mid agentview: basket + table where a push shows up."""
        img = cls._as_float_img(image)
        h, w = img.shape[:2]
        return img[int(0.45 * h) : int(0.92 * h), int(0.15 * w) : int(0.85 * w)]

    def arm(self, *, ee_z: float, image: np.ndarray) -> None:
        self.armed = True
        self.anchor_z = float(ee_z)
        self.z_lo = self.anchor_z - self.band_below
        self.z_hi = self.anchor_z + self.band_above
        self.bottom_ref = self._bottom_crop(image).copy()
        self.scene_ref = self._scene_crop(image).copy()
        self.last_debug = HeightDebug(
            ee_z=self.anchor_z, z_lo=self.z_lo, z_hi=self.z_hi
        )

    def raise_floor(self, *, min_z: float) -> None:
        """After a collision/push: never carry below this height again."""
        floor = float(min_z)
        if floor > self.anchor_z:
            self.anchor_z = floor
            self.z_lo = self.anchor_z - self.band_below
            self.z_hi = self.anchor_z + self.band_above

    def _vision_near_table(self, image: np.ndarray) -> tuple[bool, float]:
        """Heuristic: bottom of agentview gets brighter / more filled when EE is too low."""
        if self.bottom_ref is None:
            return False, 0.0
        cur = self._bottom_crop(image)
        if cur.shape != self.bottom_ref.shape:
            return False, 0.0
        score = float(np.mean(cur) - np.mean(self.bottom_ref))
        return score > self.vision_near_thresh, score

    def _scene_disturbance(self, image: np.ndarray) -> tuple[bool, float]:
        """Large mid-frame change vs HOLD → something was shoved (often basket)."""
        if self.scene_ref is None:
            return False, 0.0
        cur = self._scene_crop(image)
        if cur.shape != self.scene_ref.shape:
            return False, 0.0
        mad = float(np.mean(np.abs(cur - self.scene_ref)))
        return mad > self.scene_mad_thresh, mad

    def step(
        self,
        *,
        ee_z: float,
        image: np.ndarray,
        allow_descent: bool = False,
        force_raise: bool = False,
        use_scene_disturb: bool = True,
    ) -> HeightDebug:
        if not self.armed:
            self.last_debug = HeightDebug(ee_z=float(ee_z))
            return self.last_debug

        near, vscore = self._vision_near_table(image)
        if use_scene_disturb:
            disturb, smad = self._scene_disturbance(image)
        else:
            disturb, smad = False, 0.0
        proprio_low = float(ee_z) < self.z_lo
        proprio_high = float(ee_z) > self.z_hi
        vision_low = near and float(ee_z) < self.anchor_z
        collision = (not allow_descent) and (force_raise or disturb)

        too_low = (not allow_descent) and (proprio_low or vision_low or collision)
        too_high = (not allow_descent) and proprio_high and not collision

        az = 0.0
        if collision:
            az = self.collision_raise_az
        elif too_low:
            az = self.raise_az
        elif too_high:
            az = self.lower_az

        self.last_debug = HeightDebug(
            too_low=too_low,
            too_high=too_high,
            collision_raise=collision,
            ee_z=float(ee_z),
            z_lo=self.z_lo,
            z_hi=self.z_hi,
            vision_near_table=near,
            vision_score=vscore,
            scene_mad=smad,
            az_correction=az,
        )
        return self.last_debug


@dataclass
class CollideDebug:
    collide: bool = False
    stalled: bool = False
    scene_mad: float = 0.0
    frame_mad: float = 0.0
    ee_xy_speed: float = 0.0
    rot_delta: float = 0.0
    ee_z: float = 0.0


@dataclass
class CollideDeployableCritic:
    """Detect scrape/push during carry from camera + EE pose only.

    Inputs allowed: agentview image, EE xyz, EE rotation (3x3 or quat).
    NEVER uses object/basket body poses.

    Fires when the arm commands lateral motion but XY progress stalls while
    the lower scene changes a lot (something being shoved in view).
    """

    scene_mad_thresh: float = 0.055
    frame_mad_thresh: float = 0.025
    stall_xy_speed: float = 0.0025  # m/step
    lateral_cmd_thresh: float = 0.12
    rot_delta_thresh: float = 0.08
    streak_need: int = 3
    warmup_steps: int = 8

    armed: bool = False
    scene_ref: np.ndarray | None = None
    prev_crop: np.ndarray | None = None
    ee_prev: np.ndarray | None = None
    rot_ref: np.ndarray | None = None
    rot_prev: np.ndarray | None = None
    _streak: int = 0
    _t: int = 0
    last_debug: CollideDebug = field(default_factory=CollideDebug)

    def reset(self) -> None:
        self.armed = False
        self.scene_ref = None
        self.prev_crop = None
        self.ee_prev = None
        self.rot_ref = None
        self.rot_prev = None
        self._streak = 0
        self._t = 0
        self.last_debug = CollideDebug()

    @staticmethod
    def _as_float_img(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        return img

    @classmethod
    def _basket_roi(cls, image: np.ndarray) -> np.ndarray:
        """Lower agentview ROI (table/basket); excludes most of the arm mast."""
        img = cls._as_float_img(image)
        h, w = img.shape[:2]
        return img[int(0.55 * h) : int(0.95 * h), int(0.1 * w) : int(0.9 * w)]

    @staticmethod
    def _as_rot(rot: np.ndarray | None) -> np.ndarray | None:
        if rot is None:
            return None
        r = np.asarray(rot, dtype=np.float64).reshape(-1)
        if r.size == 4:
            return r.copy()
        if r.size == 9:
            return r.reshape(3, 3).copy()
        return r.copy()

    @staticmethod
    def _rot_delta(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))

    def arm(
        self,
        *,
        image: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray | None = None,
    ) -> None:
        self.armed = True
        crop = self._basket_roi(image)
        self.scene_ref = crop.copy()
        self.prev_crop = crop.copy()
        self.ee_prev = np.asarray(ee_pos, dtype=np.float64).copy()
        self.rot_ref = self._as_rot(ee_rot)
        self.rot_prev = None if self.rot_ref is None else self.rot_ref.copy()
        self._streak = 0
        self._t = 0
        self.last_debug = CollideDebug(ee_z=float(ee_pos[2]))

    def step(
        self,
        *,
        image: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray | None = None,
        lateral_cmd: float = 0.0,
        active: bool = True,
    ) -> CollideDebug:
        ee = np.asarray(ee_pos, dtype=np.float64)
        if not self.armed or self.scene_ref is None:
            self.last_debug = CollideDebug(ee_z=float(ee[2]))
            return self.last_debug

        self._t += 1
        cur = self._basket_roi(image)
        if cur.shape != self.scene_ref.shape:
            self.last_debug = CollideDebug(ee_z=float(ee[2]))
            return self.last_debug

        scene_mad = float(np.mean(np.abs(cur - self.scene_ref)))
        frame_mad = (
            float(np.mean(np.abs(cur - self.prev_crop))) if self.prev_crop is not None else 0.0
        )
        ee_xy_speed = (
            float(np.linalg.norm(ee[:2] - self.ee_prev[:2])) if self.ee_prev is not None else 0.0
        )
        rot = self._as_rot(ee_rot)
        rot_delta = self._rot_delta(rot, self.rot_prev)

        commanding = float(lateral_cmd) >= self.lateral_cmd_thresh
        stalled = commanding and ee_xy_speed < self.stall_xy_speed
        vision_hit = scene_mad > self.scene_mad_thresh and frame_mad > self.frame_mad_thresh
        twist_hit = stalled and rot_delta > self.rot_delta_thresh
        # Push while still sliding: large scene change during lateral carry.
        moving_push = (
            commanding
            and ee_xy_speed >= self.stall_xy_speed
            and scene_mad > self.scene_mad_thresh
            and frame_mad > self.frame_mad_thresh
        )

        raw = False
        if active and self._t >= self.warmup_steps:
            raw = (stalled and vision_hit) or moving_push or (
                stalled and twist_hit and scene_mad > 0.5 * self.scene_mad_thresh
            )

        if raw:
            self._streak += 1
        else:
            self._streak = max(0, self._streak - 1)

        collide = self._streak >= self.streak_need
        self.last_debug = CollideDebug(
            collide=collide,
            stalled=stalled,
            scene_mad=scene_mad,
            frame_mad=frame_mad,
            ee_xy_speed=ee_xy_speed,
            rot_delta=rot_delta,
            ee_z=float(ee[2]),
        )

        self.prev_crop = cur.copy()
        self.ee_prev = ee.copy()
        if rot is not None:
            self.rot_prev = rot.copy()
        return self.last_debug

    def reanchor(
        self,
        *,
        image: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray | None = None,
    ) -> None:
        """After a lift correction, reset vision refs so we don't keep firing."""
        if not self.armed:
            return
        crop = self._basket_roi(image)
        self.scene_ref = crop.copy()
        self.prev_crop = crop.copy()
        self.ee_prev = np.asarray(ee_pos, dtype=np.float64).copy()
        self.rot_ref = self._as_rot(ee_rot)
        self.rot_prev = None if self.rot_ref is None else self.rot_ref.copy()
        self._streak = 0
        self._t = 0
