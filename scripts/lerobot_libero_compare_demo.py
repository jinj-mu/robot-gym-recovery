#!/usr/bin/env python3
"""Phase 2: compare policy inference trajectories to a human demo (lerobot/libero).

All trajectories are resampled to a common rate (default **20 Hz**) before plotting
so human (often 10 Hz on Hub) and policy (20 Hz collect) share the same time base.

Expects Phase 1 CSVs from lerobot_libero_collect.py:
  action_0..6, observation.state_0..7, step, time_s

Examples
--------
python scripts/lerobot_libero_compare_demo.py --auto --target-hz 20
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]
TARGET_HZ_DEFAULT = 20.0

ACTION_LABELS = [
    "Δx (action_0)",
    "Δy (action_1)",
    "Δz (action_2)",
    "Δroll (action_3)",
    "Δpitch (action_4)",
    "Δyaw (action_5)",
    "gripper (action_6)",
]


def load_traj(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty trajectory: {path}")
    out: dict[str, np.ndarray] = {}
    for k in rows[0].keys():
        try:
            out[k] = np.array([float(r[k]) for r in rows], dtype=np.float64)
        except (TypeError, ValueError):
            continue
    return out


def infer_hz(traj: dict[str, np.ndarray]) -> float:
    """Estimate sample rate from time_s (or fall back from step count / duration)."""
    if "time_s" in traj and len(traj["time_s"]) >= 2:
        dt = np.diff(traj["time_s"])
        dt = dt[dt > 1e-6]
        if len(dt):
            return float(1.0 / np.median(dt))
    if "time_s" in traj and len(traj["time_s"]) >= 1 and traj["time_s"][-1] > 0:
        return float(len(traj["step"]) / traj["time_s"][-1])
    return TARGET_HZ_DEFAULT


def retime_frames_as_hz(traj: dict[str, np.ndarray], hz: float) -> dict[str, np.ndarray]:
    """Keep frame count; set time_s = (i+1)/hz so 10 Hz Hub demos match 20 Hz policy length."""
    n = len(traj["step"]) if "step" in traj else len(next(iter(traj.values())))
    out = dict(traj)
    out["time_s"] = (np.arange(1, n + 1, dtype=np.float64)) / hz
    out["step"] = np.arange(1, n + 1, dtype=np.float64)
    return out


def resample_traj(traj: dict[str, np.ndarray], target_hz: float) -> dict[str, np.ndarray]:
    """Linear-resample numeric columns onto a uniform target_hz grid (preserves duration)."""
    if "time_s" not in traj:
        raise ValueError("trajectory missing time_s")
    t_src = traj["time_s"]
    t0, t1 = float(t_src[0]), float(t_src[-1])
    if t1 <= t0:
        t1 = t0 + (len(t_src) - 1) / target_hz
        t_src = np.arange(len(t_src), dtype=np.float64) / target_hz
    n = max(2, int(round((t1 - t0) * target_hz)) + 1)
    t_new = np.linspace(t0, t1, n)
    out: dict[str, np.ndarray] = {"time_s": t_new, "step": np.arange(1, n + 1, dtype=np.float64)}
    skip = {"time_s", "step"}
    for k, v in traj.items():
        if k in skip or v.ndim != 1 or len(v) != len(traj["time_s"]):
            continue
        # Gripper: nearest-neighbor to keep ±1 edges sharp; others linear
        if k == "action_6":
            idx = np.searchsorted(t_src, t_new, side="left")
            idx = np.clip(idx, 0, len(v) - 1)
            # pick closer of idx-1 and idx
            idx0 = np.clip(idx - 1, 0, len(v) - 1)
            choose = np.where(np.abs(t_new - t_src[idx0]) <= np.abs(t_new - t_src[idx]), idx0, idx)
            out[k] = v[choose]
        else:
            out[k] = np.interp(t_new, t_src, v)
    out["_source_hz"] = np.array([infer_hz(traj)])
    out["_target_hz"] = np.array([target_hz])
    return out


def summarize(name: str, traj: dict[str, np.ndarray], source_hz: float, target_hz: float) -> dict:
    n = len(traj["step"])
    g = traj.get("action_6")
    first_close = None
    if g is not None:
        for i, v in enumerate(g):
            if v > 0.5:
                first_close = i + 1
                break
    success = None
    if "success" in traj:
        success = bool(traj["success"][-1] > 0.5)
    return {
        "name": name,
        "source_hz": source_hz,
        "target_hz": target_hz,
        "num_steps_resampled": n,
        "duration_s": float(traj["time_s"][-1]) if "time_s" in traj else None,
        "success": success,
        "first_gripper_close_step": first_close,
        "first_gripper_close_s": float(traj["time_s"][first_close - 1]) if first_close else None,
        "action_xyz_rms": float(
            np.sqrt(np.mean(traj["action_0"] ** 2 + traj["action_1"] ** 2 + traj["action_2"] ** 2))
        )
        if all(k in traj for k in ("action_0", "action_1", "action_2"))
        else None,
        "gripper_frac_closed": float(np.mean(g > 0.5)) if g is not None else None,
    }


def plot_compare(series: list[tuple[str, dict[str, np.ndarray]]], out_dir: Path, target_hz: float) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def t_of(traj: dict[str, np.ndarray]) -> np.ndarray:
        return traj["time_s"]

    # 1) Gripper @ common Hz
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for name, traj in series:
        ax.plot(t_of(traj), traj["action_6"], label=name, linewidth=1.5)
    ax.set_title(f"Gripper action (action_6) — all @ {target_hz:g} Hz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("g")
    ax.set_ylim(-1.2, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "gripper_action.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 2) Continuous arm actions 0..5 (HF-style continuous curves)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for dim in range(6):
        ax = axes[dim]
        key = f"action_{dim}"
        for name, traj in series:
            ax.plot(t_of(traj), traj[key], label=name, linewidth=1.1)
        ax.set_title(ACTION_LABELS[dim])
        ax.set_ylabel("action")
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize=8)
    axes[-2].set_xlabel("time (s)")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Continuous arm actions (Δpose) — all @ {target_hz:g} Hz", y=1.01)
    p = out_dir / "action_continuous_6panel.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 3) HF-like overlay: observation.state_0 + action_0 on one axes (per run in subplots)
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.8 * len(series)), sharex=False)
    if len(series) == 1:
        axes = [axes]
    for ax, (name, traj) in zip(axes, series):
        ax.plot(t_of(traj), traj["observation.state_0"], color="#f4a261", label="observation.state_0", linewidth=1.5)
        ax.plot(t_of(traj), traj["action_0"], color="#4ea8de", label="action_0 (Δx)", linewidth=1.2, alpha=0.9)
        ax.set_title(f"{name} — observation.state_0 + action_0 @ {target_hz:g} Hz")
        ax.set_ylabel("value")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    p = out_dir / "state_action_overlay.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 4) All continuous actions overlaid for one channel comparison (action_0 across runs)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for name, traj in series:
        ax.plot(t_of(traj), traj["action_0"], label=name, linewidth=1.3)
    ax.set_title(f"action_0 (Δx) continuous — all @ {target_hz:g} Hz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Δx")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "action_dx.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 5) EE XY path (state space — Hz-independent)
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, traj in series:
        ax.plot(traj["observation.state_0"], traj["observation.state_1"], label=name, linewidth=1.5)
        ax.scatter(traj["observation.state_0"][0], traj["observation.state_1"][0], s=40, marker="o")
        ax.scatter(traj["observation.state_0"][-1], traj["observation.state_1"][-1], s=40, marker="x")
    ax.set_title("EE XY path (observation.state_0/1)  o=start  x=end")
    ax.set_xlabel("state_0 (x)")
    ax.set_ylabel("state_1 (y)")
    ax.axis("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "ee_xy_path.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 6) Δz
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for name, traj in series:
        ax.plot(t_of(traj), traj["action_2"], label=name, linewidth=1.2)
    ax.set_title(f"action_2 (Δz) continuous — all @ {target_hz:g} Hz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Δz")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "action_dz.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 7) state_0 over time
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for name, traj in series:
        ax.plot(t_of(traj), traj["observation.state_0"], label=name, linewidth=1.2)
    ax.set_title(f"observation.state_0 — all @ {target_hz:g} Hz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("state_0")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "state0_over_time.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    return paths


def resolve_auto() -> tuple[Path, list[Path], list[str]]:
    root = ROBOT_GYM_ROOT / "outputs" / "libero_inference"
    human = root / "human_demo_alphabet_soup" / "trajectory.csv"
    policies = []
    labels = ["human"]
    for slug, label in (
        ("phase1_pi05_task0", "pi05"),
        ("phase1_pi0_task0", "pi0"),
    ):
        cand = root / slug / "episodes" / "libero_object_task0_ep0" / "trajectory.csv"
        if cand.is_file():
            policies.append(cand)
            labels.append(label)
    return human, policies, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policy trajectories to human demo")
    parser.add_argument("--human", type=Path, default=None)
    parser.add_argument("--policy-csv", type=Path, nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=None, help="Labels: human then each policy")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--auto", action="store_true", help="Use default phase1 + human paths")
    parser.add_argument("--target-hz", type=float, default=TARGET_HZ_DEFAULT, help="Common plot rate (default 20)")
    parser.add_argument(
        "--time-mode",
        choices=("wall", "frames_as_target_hz"),
        default="frames_as_target_hz",
        help=(
            "wall: use CSV time_s (human ~15s at 10 Hz metadata). "
            "frames_as_target_hz: time=(frame)/target_hz — halves human length so it matches policy (~7–8s)."
        ),
    )
    args = parser.parse_args()

    if args.auto:
        human, policies, labels = resolve_auto()
        if not human.is_file():
            raise SystemExit(f"Missing human demo CSV: {human}")
        if not policies:
            raise SystemExit(
                "No Phase 1 policy CSVs yet. Wait for collect to finish, then re-run with --auto.\n"
                "Expected e.g. outputs/libero_inference/phase1_pi05_task0/episodes/.../trajectory.csv"
            )
    else:
        if not args.human or not args.policy_csv:
            raise SystemExit("Provide --human and --policy-csv, or use --auto")
        human, policies = args.human, list(args.policy_csv)
        labels = args.labels or (["human"] + [f"policy{i}" for i in range(len(policies))])

    out_dir = args.out_dir or (ROBOT_GYM_ROOT / "outputs" / "libero_inference" / "phase2_compare_alphabet_soup")
    out_dir.mkdir(parents=True, exist_ok=True)
    target_hz = float(args.target_hz)

    raw_series: list[tuple[str, dict[str, np.ndarray]]] = []
    raw_series.append((labels[0], load_traj(human)))
    for i, p in enumerate(policies):
        lab = labels[i + 1] if i + 1 < len(labels) else p.parent.parent.name
        raw_series.append((lab, load_traj(p)))

    series: list[tuple[str, dict[str, np.ndarray]]] = []
    summaries = []
    for name, traj in raw_series:
        src_hz = infer_hz(traj)
        if args.time_mode == "frames_as_target_hz":
            # Same frame count; assign 20 Hz timestamps → human 155 frames ≈ 7.75 s (≈ half of 15.5 s)
            aligned = retime_frames_as_hz(traj, target_hz)
            note = f"frames@{src_hz:.1f}Hz metadata → timed as {target_hz:g} Hz"
        else:
            aligned = resample_traj(traj, target_hz)
            note = f"wall-clock resample {src_hz:.1f}→{target_hz:g} Hz"
        series.append((name, aligned))
        summaries.append(summarize(name, aligned, src_hz, target_hz))
        print(f"{name}: {note} ({len(aligned['step'])} samples, {aligned['time_s'][-1]:.2f}s)")

    with (out_dir / "compare_summary.json").open("w") as f:
        json.dump(
            {"target_hz": target_hz, "time_mode": args.time_mode, "trajectories": summaries},
            f,
            indent=2,
        )

    lines = [
        "# Phase 2: human demo vs policy",
        "",
        f"Time mode: **`{args.time_mode}`** @ **{target_hz:g} Hz**.",
        (
            "Human Hub CSV is labeled 10 Hz (~15.5 s). "
            "`frames_as_target_hz` retimes each frame as 1/20 s so human length ≈ policy (~7–8 s)."
            if args.time_mode == "frames_as_target_hz"
            else "Using wall-clock `time_s` from CSV (human stays ~15.5 s)."
        ),
        "",
        "| run | source_hz | frames | duration_s | first close (s) | gripper closed frac | xyz action rms |",
        "|-----|----------:|-------:|-----------:|----------------:|--------------------:|---------------:|",
    ]
    for s in summaries:
        gcf = f"{s['gripper_frac_closed']:.3f}" if s["gripper_frac_closed"] is not None else "n/a"
        rms = f"{s['action_xyz_rms']:.4f}" if s["action_xyz_rms"] is not None else "n/a"
        fcs = f"{s['first_gripper_close_s']:.2f}" if s["first_gripper_close_s"] is not None else "n/a"
        lines.append(
            f"| {s['name']} | {s['source_hz']:.1f} | {s['num_steps_resampled']} | {s['duration_s']:.2f} | "
            f"{fcs} | {gcf} | {rms} |"
        )
    lines += [
        "",
        "Plots:",
        "- `gripper_action.png` — binary gripper (±1)",
        "- `action_continuous_6panel.png` — continuous Δpose actions (like HF arm curves)",
        "- `state_action_overlay.png` — HF-style state_0 + action_0 per run",
        "- `action_dx.png`, `action_dz.png`, `state0_over_time.png`, `ee_xy_path.png`",
    ]
    (out_dir / "compare.md").write_text("\n".join(lines) + "\n")

    plots = plot_compare(series, out_dir, target_hz)
    print(f"Wrote {out_dir}/compare_summary.json")
    print(f"Wrote {out_dir}/compare.md")
    for p in plots:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
