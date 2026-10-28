#!/usr/bin/env python3
"""Side-by-side π₀ fail vs deployable recover for the 12 recovered leave-empty seeds.

Left: original fail trajectory. Right: detect miss-pick → retry → π₀ (cam+EE only).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT_GYM_ROOT / "scripts"))

from lerobot_libero_collect import load_policy_bundle  # noqa: E402
from lerobot_libero_deployable_harness import (  # noqa: E402
    run_detect_retry,
    run_pi0_fail,
)
from lerobot_libero_gt_label_fails import load_actions  # noqa: E402
from lerobot_libero_hybrid_recover import write_mp4  # noqa: E402

ATLAS = (
    ROBOT_GYM_ROOT
    / "outputs/libero_inference/paper_oracle_v1/failure_atlas_gt.json"
)
EXISTING_1202 = (
    ROBOT_GYM_ROOT
    / "outputs/libero_inference/paper_oracle_v1/three_way_keep_retry_seed1202"
    / "two_way.mp4"
)
RECOVERED_SEEDS = (1101, 1202, 1207, 1209, 1302, 1502, 1505, 1608, 1703, 1707, 1900, 1906)
OUT_DEFAULT = (
    ROBOT_GYM_ROOT
    / "outputs/libero_inference/paper_oracle_v1/recovered_sxs_12"
)
SHORT = {
    1101: "cream cheese",
    1202: "salad dressing",
    1207: "salad dressing",
    1209: "salad dressing",
    1302: "bbq sauce",
    1502: "tomato sauce",
    1505: "tomato sauce",
    1608: "butter",
    1703: "milk",
    1707: "milk",
    1900: "orange juice",
    1906: "orange juice",
}


def _banner(img: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), color, -1)
    cv2.putText(
        out, text[:56], (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
    )
    return out


def _side_by_side(
    fail_frames: list[np.ndarray],
    retry_frames: list[np.ndarray],
    *,
    seed: int,
    obj: str,
    n_retries: int,
    recovered: bool,
) -> list[np.ndarray]:
    n = max(len(fail_frames), len(retry_frames))
    side = []
    for i in range(n):
        left = fail_frames[min(i, len(fail_frames) - 1)]
        right = retry_frames[min(i, len(retry_frames) - 1)]
        h, w = left.shape[:2]
        if right.shape[:2] != (h, w):
            right = cv2.resize(right, (w, h))
        tag = "recovered" if recovered else "STILL FAIL"
        side.append(
            np.hstack(
                [
                    _banner(left, f"π₀ fail  seed {seed}  {obj}", (60, 60, 180)),
                    _banner(
                        right,
                        f"detect→retry  r={n_retries}  {tag}",
                        (40, 140, 80) if recovered else (40, 40, 180),
                    ),
                ]
            )
        )
    return side


def _write_status(out_dir: Path, rows: list[dict]) -> None:
    n_ok = sum(1 for r in rows if r.get("success"))
    (out_dir / "gallery.json").write_text(
        json.dumps(
            {
                "n": len(rows),
                "n_success": n_ok,
                "episodes": rows,
            },
            indent=2,
        )
    )


def _ffmpeg_bin() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _concat_ffmpeg(paths: list[Path], dest: Path) -> bool:
    if not paths:
        return False
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        return False
    lst = dest.parent / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in paths))
    import subprocess

    r = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c",
            "copy",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr[-800:] if r.stderr else "ffmpeg concat failed", flush=True)
        return False
    return dest.is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--policy", default="lerobot/pi0_libero_finetuned")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in RECOVERED_SEEDS),
    )
    parser.add_argument(
        "--reuse-1202",
        action="store_true",
        default=True,
        help="Copy the existing seed-1202 two_way.mp4 instead of re-rendering",
    )
    parser.add_argument(
        "--concat-only",
        action="store_true",
        help="Skip rendering; concatenate existing two_way.mp4 files.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    want = [int(s) for s in args.seeds.split(",") if s.strip()]
    atlas = {int(e["seed"]): e for e in json.loads(ATLAS.read_text())["episodes"]}

    bundle = None

    def _ensure_bundle():
        nonlocal bundle
        if bundle is None:
            print("Loading π₀…", flush=True)
            bundle = load_policy_bundle(args.policy, args.device, "libero_object", 2)
        return bundle

    rows: list[dict] = []
    two_ways: list[Path] = []
    if args.concat_only:
        for seed in want:
            ep = atlas[seed]
            obj = SHORT.get(seed, f"task{ep['task_id']}")
            two_way = args.output_dir / f"seed{seed}_{obj.replace(' ', '_')}" / "two_way.mp4"
            if two_way.is_file():
                two_ways.append(two_way)
        compilation = args.output_dir / "all_12_two_way.mp4"
        if _concat_ffmpeg(two_ways, compilation):
            print(f"wrote {compilation} from {len(two_ways)} clips", flush=True)
        else:
            print("ffmpeg concat failed", flush=True)
        return

    for i, seed in enumerate(want):
        ep = atlas[seed]
        obj = SHORT.get(seed, f"task{ep['task_id']}")
        tag = f"seed{seed}_{obj.replace(' ', '_')}"
        seed_dir = args.output_dir / tag
        seed_dir.mkdir(parents=True, exist_ok=True)
        two_way = seed_dir / "two_way.mp4"
        print(f"\n[{i+1}/{len(want)}] {tag}", flush=True)

        if two_way.is_file():
            row = json.loads((seed_dir / "episode.json").read_text()) if (seed_dir / "episode.json").is_file() else {
                "seed": seed,
                "task_id": int(ep["task_id"]),
                "object": obj,
                "success": True,
                "n_retries": None,
                "skipped_existing": True,
                "two_way": str(two_way),
            }
            print(f"  skip existing {two_way}", flush=True)
        elif seed == 1202 and args.reuse_1202 and EXISTING_1202.is_file():
            shutil.copy2(EXISTING_1202, two_way)
            shutil.copy2(
                EXISTING_1202.parent / "1_pi0_fail.mp4",
                seed_dir / "1_pi0_fail.mp4",
            )
            shutil.copy2(
                EXISTING_1202.parent / "2_keep_retry.mp4",
                seed_dir / "2_keep_retry.mp4",
            )
            row = {
                "seed": seed,
                "task_id": int(ep["task_id"]),
                "object": obj,
                "success": True,
                "n_retries": 1,
                "reused": True,
                "two_way": str(two_way),
            }
            print(f"  reused {EXISTING_1202}", flush=True)
        else:
            actions = load_actions(Path(ep["trajectory_csv"]))
            print("  replay π₀ fail…", flush=True)
            v1 = run_pi0_fail(task_id=int(ep["task_id"]), seed=seed, actions=actions)
            print("  recover…", flush=True)
            v2 = run_detect_retry(
                name=tag,
                task_id=int(ep["task_id"]),
                seed=seed,
                actions=actions,
                policy_bundle=_ensure_bundle(),
                enable_collide=False,
                max_retries=args.max_retries,
                max_steps=2500,
                record_frames=True,
            )
            write_mp4(seed_dir / "1_pi0_fail.mp4", v1["frames"])
            write_mp4(seed_dir / "2_keep_retry.mp4", v2["frames"])
            side = _side_by_side(
                v1["frames"],
                v2["frames"],
                seed=seed,
                obj=obj,
                n_retries=int(v2.get("n_retries", 0)),
                recovered=bool(v2.get("success")),
            )
            write_mp4(two_way, side)
            row = {
                "seed": seed,
                "task_id": int(ep["task_id"]),
                "object": obj,
                "success": bool(v2.get("success")),
                "n_retries": int(v2.get("n_retries", 0)),
                "fire_steps": list(v2.get("fire_steps", [])),
                "n_fail_frames": int(v1.get("n_frames", 0)),
                "n_retry_frames": int(v2.get("n_frames", 0)),
                "two_way": str(two_way),
            }
            print(
                f"  → success={row['success']} retries={row['n_retries']} wrote {two_way}",
                flush=True,
            )
            del v1, v2, side

        (seed_dir / "episode.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        if two_way.is_file():
            two_ways.append(two_way)
        _write_status(args.output_dir, rows)

    compilation = args.output_dir / "all_12_two_way.mp4"
    if _concat_ffmpeg(two_ways, compilation):
        print(f"wrote {compilation}", flush=True)
    else:
        print("ffmpeg concat skipped (ffmpeg missing or failed)", flush=True)
    print(json.dumps({"n": len(rows), "n_success": sum(1 for r in rows if r['success'])}, indent=2))


if __name__ == "__main__":
    main()
