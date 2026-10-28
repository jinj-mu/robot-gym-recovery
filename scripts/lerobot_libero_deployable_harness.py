#!/usr/bin/env python3
"""Detect miss-pick / collide → retry (hand back to frozen π₀). Cam+EE only for critics.

Contribution: detect failure and keep retrying. On miss-pick: open → return to
remembered grasp EE (proprio) with spiral/vision XY search → lower → hand back to π₀.
No object/basket body poses in the loop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT_GYM_ROOT / "scripts"))

from lerobot_libero_collect import batch_observation, load_policy_bundle  # noqa: E402
from lerobot_libero_deployable_critic import (  # noqa: E402
    CollideDeployableCritic,
    LeaveEmptyDeployableCritic,
)
from lerobot_libero_grasp_retry import clear_policy_queue  # noqa: E402
from lerobot_libero_gt_label_fails import load_actions  # noqa: E402
from lerobot_libero_hybrid_recover import flip_view, make_env_hd, write_mp4  # noqa: E402

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0
CONTROL_HZ = 20


def _steps(seconds: float) -> int:
    return max(1, int(round(seconds * CONTROL_HZ)))


def thin_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _ee_from_obs(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eef = obs["robot_state"]["eef"]
    pos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
    quat = np.asarray(eef["quat"], dtype=np.float64).reshape(4)
    grip = np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float64)
    return pos, quat, grip


def _policy_action(obs, *, env, policy, preprocessor, postprocessor, env_pre, env_post) -> np.ndarray:
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    observation = preprocess_observation(batch_observation(obs))
    observation["task"] = [env.task_description]
    observation = env_pre(observation)
    proc = preprocessor(observation)
    with torch.inference_mode():
        act = policy.select_action(proc)
    act = postprocessor(act)
    return env_post({ACTION: act})[ACTION].to("cpu").numpy().reshape(-1)


def _clip_action(a: list[float] | np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).reshape(-1).copy()
    out[:3] = np.clip(out[:3], -1.0, 1.0)
    out[6] = float(np.clip(out[6], -1.0, 1.0))
    return out


def _toward(err: float, scale: float, lim: float) -> float:
    return float(np.clip(scale * err, -lim, lim))


# Near grid first (keeps the 12 recoveries). Far cardinals next so a ~14cm
# empty-close (BBQ 1307) is reached before retry 20. No GT object poses.
_NEAR_XY: list[tuple[float, float]] = [
    (0.00, 0.00),
    (0.020, 0.00),
    (-0.020, 0.00),
    (0.00, 0.020),
    (0.00, -0.020),
    (0.020, 0.020),
    (0.020, -0.020),
    (-0.020, 0.020),
    (-0.020, -0.020),
    (0.040, 0.00),
    (-0.040, 0.00),
    (0.00, 0.040),
    (0.00, -0.040),
    (0.035, 0.035),
    (0.035, -0.035),
    (-0.035, 0.035),
    (-0.035, -0.035),
]
_FAR_XY: list[tuple[float, float]] = [
    (0.080, 0.00),
    (-0.080, 0.00),
    (0.00, 0.080),
    (0.00, -0.080),
    (0.120, 0.00),
    (-0.120, 0.00),
    (0.00, 0.120),
    (0.00, -0.120),
    (0.140, 0.00),
    (-0.140, 0.00),
    (0.00, 0.140),
    (0.00, -0.140),
    (0.100, 0.00),
    (-0.100, 0.00),
    (0.00, 0.100),
    (0.00, -0.100),
    (0.060, 0.00),
    (-0.060, 0.00),
    (0.00, 0.060),
    (0.00, -0.060),
    (0.090, 0.090),
    (-0.090, 0.090),
    (0.090, -0.090),
    (-0.090, -0.090),
]
_SEARCH_XY = _NEAR_XY + _FAR_XY
_SEARCH_Z = (0.072, 0.088, 0.100, 0.060, 0.115, 0.050, 0.130, 0.080)
# libero_object basket is a suite fixture (same pose across object tasks).
# Used only as a place prior — not episode object GT.
BASKET_XY_PRIOR = np.array([0.0, 0.25], dtype=np.float64)
PLACE_CARRY_Z = 0.22
PLACE_RELEASE_Z = 0.12


def _lattice_xy_offset(n_retries: int, *, far_first: bool = False) -> np.ndarray:
    seq = (_FAR_XY + _NEAR_XY) if far_first else _SEARCH_XY
    i = max(0, int(n_retries) - 1)
    xy = seq[i % len(seq)]
    ring = i // len(seq)
    scale = 1.0 + 0.25 * ring
    return np.array([xy[0] * scale, xy[1] * scale], dtype=np.float64)


def _pick_z(n_retries: int, anchor_z: float) -> float:
    """Alternate pick heights. Tall bottles need ~6–8cm, not hover at 11cm."""
    if n_retries <= 1 and float(anchor_z) >= 0.075:
        return float(np.clip(anchor_z, 0.070, 0.120))
    return float(_SEARCH_Z[(int(n_retries) - 1) % len(_SEARCH_Z)])


def _vision_object_xy(
    image: np.ndarray,
    table_ref: np.ndarray | None,
    *,
    meters_per_px: float = 0.00125,
    max_nudge_m: float = 0.06,
) -> tuple[np.ndarray, float]:
    """Foreground blob in table crop → EE xy nudge + confidence in [0,1].

    Border pixels estimate table background; interior contrast ≈ object.
    Deployable (pixels only). Axis map is a LIBERO agentview heuristic.
    """
    if table_ref is not None:
        crop = np.asarray(table_ref, dtype=np.float32)
    else:
        crop = LeaveEmptyDeployableCritic._table_crop(image)
    if crop.max() > 1.5:
        crop = crop / 255.0
    if crop.ndim != 3 or min(crop.shape[:2]) < 16:
        return np.zeros(2, dtype=np.float64), 0.0

    h, w = crop.shape[:2]
    border = np.concatenate(
        [
            crop[0, :].reshape(-1, 3),
            crop[-1, :].reshape(-1, 3),
            crop[:, 0].reshape(-1, 3),
            crop[:, -1].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(crop - bg.reshape(1, 1, 3), axis=2)
    thr = float(np.percentile(dist, 70))
    mask = (dist > max(thr, float(np.median(dist)) + 0.08)).astype(np.uint8)
    # Drop a thin border so background edges don't dominate.
    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0
    # Keep largest connected component only.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        n_fg = 0
        mask[:] = 0
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        mask = (labels == best).astype(np.uint8)
        n_fg = int(mask.sum())
        # Reject giant "whole table" components.
        if n_fg > 0.25 * h * w:
            n_fg = 0
            mask[:] = 0
    if n_fg < 30:
        # Fallback: high-frequency salience.
        gray = crop.mean(axis=2)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
        sal = np.clip(np.abs(gray - blur) - float(np.median(np.abs(gray - blur))), 0, None)
        if float(sal.sum()) < 1e-6:
            return np.zeros(2, dtype=np.float64), 0.0
        ys, xs = np.indices(sal.shape)
        wsum = sal + 1e-8
        cx = float((xs * wsum).sum() / wsum.sum())
        cy = float((ys * wsum).sum() / wsum.sum())
        conf = 0.35
    else:
        ys, xs = np.where(mask > 0)
        cx = float(xs.mean())
        cy = float(ys.mean())
        conf = float(np.clip(n_fg / (0.15 * h * w), 0.0, 1.0))

    du = cx - 0.5 * (w - 1)
    dv = cy - 0.5 * (h - 1)
    dx = meters_per_px * dv
    dy = -meters_per_px * du
    out = np.array([dx, dy], dtype=np.float64)
    nrm = float(np.linalg.norm(out))
    if nrm > max_nudge_m:
        out *= max_nudge_m / nrm
    return out, conf


def _retry_target_ee(
    *,
    anchor_ee: np.ndarray,
    n_retries: int,
    image: np.ndarray,
    table_ref: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Anchor + small lattice (+ tiny vision). No GT object poses.

    Retry #1 stays on the proprio miss site (this recovered many seeds).
    Later retries only nudge a few cm — large vision pulls were hurting.
    """
    far_first = float(anchor_ee[2]) < 0.075
    lattice = _lattice_xy_offset(n_retries, far_first=far_first)
    vision, vconf = _vision_object_xy(
        image, table_ref, meters_per_px=0.0008, max_nudge_m=0.02
    )
    # Alternate camera-axis hypotheses only on later retries, and keep them small.
    axis_mode = (int(n_retries) - 1) % 4
    if axis_mode == 1:
        vision = np.array([vision[1], -vision[0]], dtype=np.float64)
    elif axis_mode == 2:
        vision = -vision
    elif axis_mode == 3:
        vision = np.array([-vision[1], vision[0]], dtype=np.float64)
    # Shrink vision so it never dominates the lattice.
    vision = 0.35 * vision
    if n_retries <= 1:
        # Still try a tiny vision nudge on #1 if confident — helps offset empty-closes.
        if vconf >= 0.55:
            xy = 0.5 * vision
        else:
            xy = np.zeros(2, dtype=np.float64)
    else:
        xy = lattice + vision
    target = np.asarray(anchor_ee, dtype=np.float64).reshape(3).copy()
    target[0] += float(xy[0])
    target[1] += float(xy[1])
    target[2] = _pick_z(n_retries, float(anchor_ee[2]))
    return target, lattice, vision, float(vconf)


def _begin_miss_retry(
    *,
    leave: LeaveEmptyDeployableCritic,
    ee: np.ndarray,
    image: np.ndarray,
    n_retries: int,
    name: str,
    step: int,
    miss_anchor_ee: np.ndarray | None,
    miss_table_ref: np.ndarray | None,
) -> tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Build retry target from fixed miss-anchor + lattice/vision; start OPEN."""
    if leave.ee_at_grasp is not None:
        grasp_ee = np.asarray(leave.ee_at_grasp, dtype=np.float64).reshape(3).copy()
    else:
        grasp_ee = np.asarray(ee, dtype=np.float64).reshape(3).copy()

    if miss_anchor_ee is None:
        anchor = grasp_ee.copy()
        # First empty-close: keep only if it looks like a table pick.
        if not (0.075 <= float(anchor[2]) <= 0.170):
            anchor[2] = 0.115
    else:
        # Never re-center on later empty-closes (often mid-air after drop).
        anchor = np.asarray(miss_anchor_ee, dtype=np.float64).reshape(3).copy()

    # Persist first grasp table crop for vision across leave.reset() handbacks.
    table_ref = miss_table_ref
    if table_ref is None and leave.table_ref is not None:
        table_ref = np.asarray(leave.table_ref, dtype=np.float32).copy()

    target, lattice, vision, vconf = _retry_target_ee(
        anchor_ee=anchor,
        n_retries=n_retries,
        image=image,
        table_ref=table_ref,
    )
    print(
        f"  [{name}] DETECT miss-pick @{step+1} → retry #{n_retries} "
        f"(open→lattice/vision→lower→π₀) "
        f"anchor={anchor} target={target} lattice={lattice} "
        f"vision={vision} vconf={vconf:.2f}"
    )
    action = _clip_action([0.0, 0.0, 0.05, 0, 0, 0, GRIP_OPEN])
    return "open", 0, target, action, anchor, table_ref


def run_pi0_fail(*, task_id: int, seed: int, actions: list[np.ndarray]) -> dict:
    env = make_env_hd(task_id, 256, 256)
    obs, _ = env.reset(seed=seed)
    frames = []
    success = False
    for step, action in enumerate(actions):
        frames.append(thin_label(flip_view(obs["pixels"]["image"]), f"π₀ fail  step={step+1}"))
        obs, _, _, _, info = env.step(np.asarray(action, dtype=np.float32).reshape(-1))
        success = bool(info.get("is_success", False))
        if success:
            break
    for _ in range(20):
        frames.append(thin_label(flip_view(obs["pixels"]["image"]), "π₀ FAIL"))
    env.close()
    return {"name": "π₀", "success": success, "n_frames": len(frames), "frames": frames}


def run_detect_retry(
    *,
    name: str,
    task_id: int,
    seed: int,
    actions: list[np.ndarray],
    policy_bundle,
    enable_collide: bool,
    max_retries: int = 10,
    max_steps: int = 2500,
    record_frames: bool = True,
) -> dict:
    """Replay fail until miss-pick; then keep detect→open/lift→π₀ until success or budget."""
    policy, preprocessor, postprocessor, env_pre, env_post, _, _ = policy_bundle
    env = make_env_hd(task_id, 360, 360, horizon=3000, ignore_done=True)
    obs, _ = env.reset(seed=seed)

    leave = LeaveEmptyDeployableCritic()
    leave.reset()
    collide = CollideDeployableCritic()
    collide.reset()

    policy.reset()
    clear_policy_queue(policy)

    frames: list[np.ndarray] = []
    fire_steps: list[int] = []
    handback_steps: list[int] = []
    collide_steps: list[int] = []
    n_retries = 0
    success = False
    hold_ok = False  # after good pick, let π₀ place; only watch drop/collide
    hold_since = 0

    phase = "replay"  # replay | open | reapproach | lower | close_try | lift_check | place_* | pi0
    phase_t = 0
    collide_target_z: float | None = None
    leave_cooldown = 0
    first_handback = True
    retry_grasp_ee: np.ndarray | None = None
    miss_anchor_ee: np.ndarray | None = None  # fixed lattice center across retries
    miss_table_ref: np.ndarray | None = None  # first grasp table crop for vision

    def _handback_to_pi0(step: int, *, keep_hold: bool = False) -> np.ndarray:
        nonlocal phase, phase_t, hold_ok, leave_cooldown, first_handback, hold_since
        clear_policy_queue(policy)
        # Fresh action chunk every handback — new entry state needs new plan.
        policy.reset()
        first_handback = False
        leave.reset()
        collide.reset()
        handback_steps.append(step + 1)
        leave_cooldown = _steps(2.0)
        phase = "pi0"
        phase_t = 0
        hold_ok = bool(keep_hold)
        hold_since = 0 if not keep_hold else hold_since
        print(f"  [{name}] near grasp pose — hand back to π₀ @{step+1} (retry #{n_retries})")
        return _policy_action(
            obs,
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_pre=env_pre,
            env_post=env_post,
        )

    for step in range(max_steps):
        image = obs["pixels"]["image"]
        ee, ee_quat, grip = _ee_from_obs(obs)
        gap = float(np.mean(np.abs(grip)))
        view = flip_view(image)

        if phase == "replay":
            fired = leave.step(image=image, ee_pos=ee, grip_qpos=grip)
            if fired:
                fire_steps.append(step + 1)
                n_retries += 1
                hold_ok = False
                phase, phase_t, retry_grasp_ee, action, miss_anchor_ee, miss_table_ref = (
                    _begin_miss_retry(
                        leave=leave,
                        ee=ee,
                        image=image,
                        n_retries=n_retries,
                        name=name,
                        step=step,
                        miss_anchor_ee=miss_anchor_ee,
                        miss_table_ref=miss_table_ref,
                    )
                )
                label = f"{name}: DETECT→retry#{n_retries}"
            else:
                if step >= len(actions):
                    phase = "pi0"
                    phase_t = 0
                    clear_policy_queue(policy)
                    action = _policy_action(
                        obs, env=env, policy=policy, preprocessor=preprocessor,
                        postprocessor=postprocessor, env_pre=env_pre, env_post=env_post,
                    )
                    label = f"{name}: π₀"
                else:
                    action = actions[step]
                    label = f"{name}: replay"

        elif phase == "open":
            phase_t += 1
            action = _clip_action([0.0, 0.0, 0.08, 0, 0, 0, GRIP_OPEN])
            label = f"{name}: OPEN #{n_retries}"
            if phase_t >= _steps(0.30):
                phase = "reapproach"
                phase_t = 0

        elif phase == "reapproach":
            phase_t += 1
            assert retry_grasp_ee is not None
            # Hover above spiral/vision target (proprio+cam), open gripper.
            hover_z = float(retry_grasp_ee[2]) + 0.08
            xy_err = retry_grasp_ee[:2] - ee[:2]
            xy_dist = float(np.linalg.norm(xy_err))
            action = _clip_action(
                [
                    _toward(float(xy_err[0]), 10.0, 0.8),
                    _toward(float(xy_err[1]), 10.0, 0.8),
                    _toward(hover_z - float(ee[2]), 6.0, 0.6),
                    0,
                    0,
                    0,
                    GRIP_OPEN,
                ]
            )
            label = f"{name}: →target #{n_retries}"
            if xy_dist < 0.025 or phase_t >= _steps(2.5):
                phase = "lower"
                phase_t = 0

        elif phase == "lower":
            phase_t += 1
            assert retry_grasp_ee is not None
            # Mid-body height for tall bottles (BBQ), not hover at grasp-EE z.
            deep_z = min(float(retry_grasp_ee[2]), 0.072)
            xy_err = retry_grasp_ee[:2] - ee[:2]
            action = _clip_action(
                [
                    _toward(float(xy_err[0]), 5.0, 0.35),
                    _toward(float(xy_err[1]), 5.0, 0.35),
                    _toward(deep_z - float(ee[2]), 8.0, 0.55),
                    0,
                    0,
                    0,
                    GRIP_OPEN,
                ]
            )
            label = f"{name}: LOWER #{n_retries}"
            if float(ee[2]) <= deep_z + 0.02 or phase_t >= _steps(1.8):
                # Retry #1: π₀ from near pose (recovered most of the 12).
                # Later: scripted pinch at lattice site (BBQ never wraps from hover).
                if n_retries >= 2:
                    phase = "close_try"
                    phase_t = 0
                    action = _clip_action([0.0, 0.0, -0.25, 0, 0, 0, GRIP_CLOSE])
                    label = f"{name}: CLOSE #{n_retries}"
                else:
                    action = _handback_to_pi0(step)
                    label = f"{name}: →π₀ #{n_retries}"

        elif phase == "close_try":
            phase_t += 1
            # Keep pressing down while closing so fingers meet the object body.
            action = _clip_action([0.0, 0.0, -0.35, 0, 0, 0, GRIP_CLOSE])
            label = f"{name}: CLOSE #{n_retries}"
            if phase_t >= _steps(1.2):
                phase = "lift_check"
                phase_t = 0

        elif phase == "lift_check":
            phase_t += 1
            action = _clip_action([0.0, 0.0, 0.45, 0, 0, 0, GRIP_CLOSE])
            label = f"{name}: LIFTCHK #{n_retries}"
            if phase_t >= _steps(0.9):
                if miss_table_ref is not None:
                    leave.table_ref = np.asarray(miss_table_ref, dtype=np.float32).copy()
                    leave.grasp_attempted = True
                    if miss_anchor_ee is not None:
                        leave.ee_at_grasp = np.asarray(miss_anchor_ee, dtype=np.float64).copy()
                leave.step(image=image, ee_pos=ee, grip_qpos=grip)
                gap_now = float(np.mean(np.abs(grip)))
                got = (
                    leave.jaws_wrapping(grip)
                    or leave.likely_holding(image=image, grip_qpos=grip)
                    # Tall bottles: partial wrap + lifted EE vs anchor.
                    or (
                        0.005 < gap_now < 0.040
                        and miss_anchor_ee is not None
                        and float(ee[2]) > float(miss_anchor_ee[2]) + 0.02
                    )
                )
                print(
                    f"  [{name}] lift_check gap={gap_now:.4f} wrap={leave.jaws_wrapping(grip)} "
                    f"hold={leave.likely_holding(image=image, grip_qpos=grip)} got={got}"
                )
                if got:
                    print(
                        f"  [{name}] SCRIPTED HOLD @{step+1} retry #{n_retries} "
                        f"— scripted place (basket prior)"
                    )
                    hold_ok = True
                    phase = "place_lift"
                    phase_t = 0
                    action = _clip_action([0.0, 0.0, 0.7, 0, 0, 0, GRIP_CLOSE])
                    label = f"{name}: PLACE↑ #{n_retries}"
                elif n_retries < max_retries:
                    n_retries += 1
                    fire_steps.append(step + 1)
                    phase, phase_t, retry_grasp_ee, action, miss_anchor_ee, miss_table_ref = (
                        _begin_miss_retry(
                            leave=leave,
                            ee=ee,
                            image=image,
                            n_retries=n_retries,
                            name=name,
                            step=step,
                            miss_anchor_ee=miss_anchor_ee,
                            miss_table_ref=miss_table_ref,
                        )
                    )
                    label = f"{name}: NEXT lattice #{n_retries}"
                else:
                    action = _handback_to_pi0(step)
                    label = f"{name}: →π₀ exhausted"

        elif phase == "place_lift":
            phase_t += 1
            action = _clip_action([0.0, 0.0, 0.85, 0, 0, 0, GRIP_CLOSE])
            label = f"{name}: PLACE↑"
            if float(ee[2]) >= PLACE_CARRY_Z - 0.02 or phase_t >= _steps(2.5):
                phase = "place_xy"
                phase_t = 0

        elif phase == "place_xy":
            phase_t += 1
            xy_err = BASKET_XY_PRIOR - ee[:2]
            action = _clip_action(
                [
                    _toward(float(xy_err[0]), 6.0, 0.7),
                    _toward(float(xy_err[1]), 6.0, 0.7),
                    _toward(PLACE_CARRY_Z - float(ee[2]), 4.0, 0.4),
                    0,
                    0,
                    0,
                    GRIP_CLOSE,
                ]
            )
            label = f"{name}: PLACE→basket"
            if float(np.linalg.norm(xy_err)) < 0.04 or phase_t >= _steps(4.0):
                phase = "place_lower"
                phase_t = 0

        elif phase == "place_lower":
            phase_t += 1
            xy_err = BASKET_XY_PRIOR - ee[:2]
            action = _clip_action(
                [
                    _toward(float(xy_err[0]), 4.0, 0.35),
                    _toward(float(xy_err[1]), 4.0, 0.35),
                    _toward(PLACE_RELEASE_Z - float(ee[2]), 6.0, 0.5),
                    0,
                    0,
                    0,
                    GRIP_CLOSE,
                ]
            )
            label = f"{name}: PLACE↓"
            if float(ee[2]) <= PLACE_RELEASE_Z + 0.03 or phase_t >= _steps(2.0):
                phase = "place_open"
                phase_t = 0

        elif phase == "place_open":
            phase_t += 1
            action = _clip_action([0.0, 0.0, 0.05, 0, 0, 0, GRIP_OPEN])
            label = f"{name}: PLACE open"
            if phase_t >= _steps(1.2):
                phase = "pi0"
                phase_t = 0
                print(f"  [{name}] scripted place done @{step+1} → π₀")

        elif phase == "collide_lift":
            phase_t += 1
            gcmd = GRIP_CLOSE if gap < 0.03 else GRIP_OPEN
            action = _clip_action([0.0, 0.0, 0.85, 0, 0, 0, gcmd])
            label = f"{name}: COLLIDE↑"
            done_lift = (
                collide_target_z is not None and float(ee[2]) >= collide_target_z - 0.01
            ) or phase_t >= _steps(2.0)
            if done_lift:
                collide.reanchor(image=image, ee_pos=ee, ee_rot=ee_quat)
                phase = "pi0"
                phase_t = 0
                print(f"  [{name}] collide clear ee_z={ee[2]:.3f} → π₀")

        else:  # pi0
            action = _policy_action(
                obs, env=env, policy=policy, preprocessor=preprocessor,
                postprocessor=postprocessor, env_pre=env_pre, env_post=env_post,
            )
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            fired = False

            leave.step(image=image, ee_pos=ee, grip_qpos=grip)

            if leave.likely_holding(image=image, grip_qpos=grip):
                if not hold_ok:
                    hold_ok = True
                    hold_since = 0
                    print(f"  [{name}] HOLD OK @{step+1} — suppress miss-pick, π₀ places")
                leave_cooldown = max(leave_cooldown, 1)
                hold_since += 1
            elif hold_ok:
                hold_since += 1

            if hold_ok and hold_since >= _steps(4.0):
                print(f"  [{name}] π₀ place stalled @{step+1} — scripted place")
                phase = "place_lift"
                phase_t = 0
                action = _clip_action([0.0, 0.0, 0.7, 0, 0, 0, GRIP_CLOSE])
                label = f"{name}: PLACE↑"
                fired = False

            elif leave_cooldown > 0:
                leave_cooldown -= 1
                fired = False
            elif hold_ok:
                fired = False
                if leave.jaws_empty_closed(grip) and leave.last_debug.object_on_table:
                    hold_ok = False
                    leave.reset()
                    # Keep anchors so we return to the object, not the drop pose.
                    print(f"  [{name}] lost hold @{step+1} — miss-pick armed again")
            else:
                fired = bool(leave.last_debug.fire)

            if fired and n_retries < max_retries:
                if leave.jaws_wrapping(grip) or leave.likely_holding(image=image, grip_qpos=grip):
                    hold_ok = True
                    label = f"{name}: π₀ (hold)"
                else:
                    fire_steps.append(step + 1)
                    n_retries += 1
                    hold_ok = False
                    phase, phase_t, retry_grasp_ee, action, miss_anchor_ee, miss_table_ref = (
                        _begin_miss_retry(
                            leave=leave,
                            ee=ee,
                            image=image,
                            n_retries=n_retries,
                            name=name,
                            step=step,
                            miss_anchor_ee=miss_anchor_ee,
                            miss_table_ref=miss_table_ref,
                        )
                    )
                    label = f"{name}: DETECT→retry#{n_retries}"
            elif enable_collide and hold_ok:
                if not collide.armed:
                    collide.arm(image=image, ee_pos=ee, ee_rot=ee_quat)
                lateral = float(max(abs(float(action[0])), abs(float(action[1]))))
                dbg = collide.step(
                    image=image,
                    ee_pos=ee,
                    ee_rot=ee_quat,
                    lateral_cmd=lateral,
                    active=True,
                )
                if dbg.collide:
                    collide_steps.append(step + 1)
                    collide_target_z = float(ee[2]) + 0.10
                    phase = "collide_lift"
                    phase_t = 0
                    print(
                        f"  [{name}] DETECT collide @{step+1} "
                        f"mad={dbg.scene_mad:.3f} → lift→π₀"
                    )
                    action = _clip_action([0.0, 0.0, 0.85, 0, 0, 0, GRIP_CLOSE])
                    label = f"{name}: COLLIDE↑"
                else:
                    label = f"{name}: π₀ place (hold)"
            else:
                label = f"{name}: π₀ (r={n_retries})" + (" hold" if hold_ok else "")

        if record_frames:
            frames.append(thin_label(view, label[:58]))
        try:
            obs, _, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32).reshape(-1)
            )
        except ValueError as exc:
            # Robosuite raises if we step after horizon/done; treat as episode end.
            if "terminated episode" in str(exc).lower():
                print(f"  [{name}] env already terminated @{step+1} — stop")
                break
            raise
        success = bool(info.get("is_success", False))
        done_flag = bool(terminated) or bool(truncated) or bool(info.get("done", False))
        if success:
            if record_frames:
                for _ in range(20):
                    frames.append(
                        thin_label(
                            flip_view(obs["pixels"]["image"]),
                            f"{name}: SUCCESS after {n_retries} retries",
                        )
                    )
            break
        if done_flag:
            print(f"  [{name}] env ended @{step+1} success={success}")
            break

    env.close()
    return {
        "name": name,
        "success": success,
        "n_retries": n_retries,
        "fire_steps": fire_steps,
        "handback_steps": handback_steps,
        "collide_steps": collide_steps,
        "n_frames": len(frames),
        "frames": frames,
        "stack": "detect miss-pick/collide → keep retrying via π₀; cam+EE critics only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument(
        "--traj",
        type=Path,
        default=ROBOT_GYM_ROOT
        / "outputs/libero_inference/paper_oracle_v1/pi0_baseline/episodes"
        / "libero_object_task2_ep2/trajectory.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROBOT_GYM_ROOT
        / "outputs/libero_inference/paper_oracle_v1/three_way_keep_retry_seed1202",
    )
    parser.add_argument("--policy", default="lerobot/pi0_libero_finetuned")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-retries", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actions = load_actions(args.traj)
    print("Loading π₀…")
    bundle = load_policy_bundle(args.policy, args.device, "libero_object", args.task_id)

    print("=== 1 π₀ fail ===")
    v1 = run_pi0_fail(task_id=args.task_id, seed=args.seed, actions=actions)
    print("=== 2 detect→keep retrying π₀ ===")
    v2 = run_detect_retry(
        name="RETRY",
        task_id=args.task_id,
        seed=args.seed,
        actions=actions,
        policy_bundle=bundle,
        enable_collide=False,
        max_retries=args.max_retries,
    )

    write_mp4(args.output_dir / "1_pi0_fail.mp4", v1["frames"])
    write_mp4(args.output_dir / "2_keep_retry.mp4", v2["frames"])

    def banner(img, text, col):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), col, -1)
        cv2.putText(out, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    n = max(len(v1["frames"]), len(v2["frames"]))
    side = []
    for i in range(n):
        L = v1["frames"][min(i, len(v1["frames"]) - 1)]
        M = v2["frames"][min(i, len(v2["frames"]) - 1)]
        h, w = L.shape[:2]
        M = cv2.resize(M, (w, h)) if M.shape[:2] != (h, w) else M
        side.append(
            np.hstack(
                [
                    banner(L, "1) π₀ fail once", (60, 60, 200)),
                    banner(M, "2) detect→keep retry π₀", (40, 140, 200)),
                ]
            )
        )
    write_mp4(args.output_dir / "two_way.mp4", side)
    # Keep legacy filename pointing at the same 2-panel cut.
    write_mp4(args.output_dir / "three_way.mp4", side)

    summary = {
        "contribution": "detect failure → keep retrying (hand back to π₀)",
        "1": {k: v for k, v in v1.items() if k != "frames"},
        "2": {k: v for k, v in v2.items() if k != "frames"},
        "videos": {
            "two_way": str(args.output_dir / "two_way.mp4"),
            "three_way": str(args.output_dir / "three_way.mp4"),
            "1": str(args.output_dir / "1_pi0_fail.mp4"),
            "2": str(args.output_dir / "2_keep_retry.mp4"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
