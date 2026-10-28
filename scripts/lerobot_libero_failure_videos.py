#!/usr/bin/env python3
"""Replay matrix failure episodes with video for visual debugging.

Usage:
  bash scripts/lerobot_libero_failure_videos.sh
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROBOT_GYM_ROOT = Path(__file__).resolve().parents[1]

# Episodes that failed in phase1_matrix (task_id, ep, seed, label)
FAILURE_SPECS = [
    (7, 2, 1702, "milk_seed1702"),
    (2, 0, 1200, "dressing_seed1200"),
    (2, 1, 1201, "dressing_seed1201"),
    (3, 1, 1301, "bbq_seed1301"),
    (0, 0, 1000, "soup_seed1000"),
    (8, 2, 1802, "pudding_seed1802"),
    (9, 0, 1900, "oj_seed1900"),
]

POLICIES = [
    ("pi05", "lerobot/pi05_libero_finetuned"),
    ("pi0", "lerobot/pi0_libero_finetuned"),
]

# Which specs each policy failed on in the matrix
POLICY_FAIL_LABELS = {
    "pi05": {"milk_seed1702", "bbq_seed1301"},
    "pi0": {
        "milk_seed1702",
        "dressing_seed1200",
        "dressing_seed1201",
        "soup_seed1000",
        "pudding_seed1802",
        "oj_seed1900",
    },
}


def main() -> None:
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

    out_root = ROBOT_GYM_ROOT / "outputs" / "libero_inference" / "failure_videos"
    out_root.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict] = []

    for slug, policy_path in POLICIES:
        labels = POLICY_FAIL_LABELS[slug]
        specs = [s for s in FAILURE_SPECS if s[3] in labels]
        if not specs:
            continue

        print(f"\n=== {slug}: {policy_path} ({len(specs)} episodes) ===")
        policy, pre, post, env_pre, env_post, policy_cfg, _ = load_policy_bundle(
            policy_path, "mps", "libero_object", specs[0][0]
        )

        for task_id, ep, seed, label in specs:
            ep_dir = out_root / slug / label
            print(f"  {label} task={task_id} ep={ep} seed={seed}")
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
                    save_video=True,
                    episode_dir=ep_dir,
                )
                print(
                    f"    success={metrics.success} steps={metrics.num_steps} "
                    f"video={ep_dir / 'video.mp4'}"
                )
                log_rows.append(
                    {
                        "policy": slug,
                        "label": label,
                        "task_id": task_id,
                        "episode_index": ep,
                        "seed": seed,
                        "success": metrics.success,
                        "num_steps": metrics.num_steps,
                        "video": str(ep_dir / "video.mp4"),
                    }
                )
            finally:
                env.close()

    runs_path = out_root / "runs.csv"
    if log_rows:
        with runs_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    with (out_root / "summary.json").open("w") as f:
        json.dump(log_rows, f, indent=2)

    print(f"\nDone. Videos under {out_root}")


if __name__ == "__main__":
    main()
