#!/usr/bin/env python3
"""Patch robosuite 1.4 binding_utils for numpy int32 jnt_type (MuJoCo 3.x + NumPy 2)."""
from __future__ import annotations

import sys
from pathlib import Path


def find_binding_utils() -> Path | None:
    try:
        import robosuite  # noqa: F401
        import robosuite.utils.binding_utils as bu

        return Path(bu.__file__)
    except ImportError:
        return None


OLD = "assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)"
NEW = (
    "assert joint_type == mujoco.mjtJoint.mjJNT_HINGE "
    "or joint_type == mujoco.mjtJoint.mjJNT_SLIDE"
)


def patch(path: Path) -> bool:
    text = path.read_text()
    if NEW in text:
        return False
    if OLD not in text:
        raise SystemExit(f"Expected pattern not found in {path}")
    path.write_text(text.replace(OLD, NEW))
    return True


def main() -> None:
    path = find_binding_utils()
    if path is None:
        print("robosuite not installed — skip patch")
        return
    changed = patch(path)
    print(f"{'Patched' if changed else 'Already patched'}: {path}")


if __name__ == "__main__":
    main()
