#!/usr/bin/env bash
# LeRobot + π₀.₅ + LIBERO on Apple Silicon (MPS). Official LIBERO (not hf-libero).
set -euo pipefail

ROBOT_GYM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$ROBOT_GYM_ROOT/third_party/lerobot}"
LIBERO_DIR="${LIBERO_DIR:-$ROBOT_GYM_ROOT/third_party/LIBERO}"
UV="${UV:-$HOME/.local/bin/uv}"
PY="${LEROBOT_DIR}/.venv/bin/python"
PIP="${LEROBOT_DIR}/.venv/bin/pip"

if [[ ! -x "$UV" ]]; then
  echo "Missing uv at $UV"
  exit 1
fi

if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
  echo "Cloning LeRobot into $LEROBOT_DIR ..."
  mkdir -p "$(dirname "$LEROBOT_DIR")"
  git clone --depth 1 https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi

if [[ ! -d "$LIBERO_DIR/.git" ]]; then
  echo "Cloning LIBERO into $LIBERO_DIR ..."
  mkdir -p "$(dirname "$LIBERO_DIR")"
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi

if [[ ! -d "$LEROBOT_DIR/.venv" ]]; then
  echo "Creating Python 3.12 venv ..."
  "$UV" venv --python 3.12 "$LEROBOT_DIR/.venv"
fi

echo "Installing LeRobot + pi extras ..."
"$UV" pip install --python "$PY" -U pip
"$UV" pip install --python "$PY" -e "$LEROBOT_DIR"
"$UV" pip install --python "$PY" -e "$LEROBOT_DIR[pi]"
"$UV" pip install --python "$PY" -e "$LEROBOT_DIR[dataset]"

echo "Installing official LIBERO + MuJoCo (skip hf-libero on Mac) ..."
# robosuite 1.4: numpy int32 jnt_type breaks `in (mjJNT_HINGE, ...)` on MuJoCo 3.x — patch below.
"$UV" pip install --python "$PY" \
  "mujoco>=3.1.6,<3.10" "robosuite==1.4.0" "bddl==1.0.1" imageio imageio-ffmpeg easydict cloudpickle gym pytest matplotlib

"$PY" "$ROBOT_GYM_ROOT/scripts/patch_robosuite_joint_type.py"

# pip editable is more reliable than uv editable for LIBERO on Mac
"$PY" -m pip install -e "$LIBERO_DIR"

echo "Writing ~/.libero/config.yaml ..."
"$PY" - <<PY
import os
from pathlib import Path
import yaml

libero_root = Path("${LIBERO_DIR}/libero/libero").resolve()
paths = {
    "benchmark_root": str(libero_root),
    "bddl_files": str(libero_root / "bddl_files"),
    "init_states": str(libero_root / "init_files"),
    "datasets": str(libero_root.parent / "datasets"),
    "assets": str(libero_root / "assets"),
}
config_dir = Path(os.path.expanduser("~/.libero"))
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / "config.yaml"
with open(config_file, "w") as f:
    yaml.dump(paths, f)
for k, v in paths.items():
    print(f"  {k}: {v}")
print(f"Wrote {config_file}")
PY

export PYTHONPATH="${LIBERO_DIR}:${PYTHONPATH:-}"
export LIBERO_DIR

echo "Verifying LIBERO env reset (off-screen) ..."
export MUJOCO_GL=cgl
LIBERO_DIR="$LIBERO_DIR" MUJOCO_GL=cgl NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$HOME/.cache/numba_robot_gym}" "$PY" - <<'PY'
import os
import sys
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_robot_gym"))
libero_repo = os.environ.get("LIBERO_DIR", "")
if libero_repo and libero_repo not in sys.path:
    sys.path.insert(0, libero_repo)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()["libero_object"]()
task = bench.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
obs = env.reset()
print("libero reset OK, obs keys:", list(obs.keys())[:5])
env.close()
PY

echo "Verifying MPS + MuJoCo + LIBERO ..."
LIBERO_DIR="$LIBERO_DIR" "$PY" - <<'PY'
import os
import sys
libero_repo = os.environ.get("LIBERO_DIR", "")
if libero_repo and libero_repo not in sys.path:
    sys.path.insert(0, libero_repo)

import torch
import mujoco
from libero.libero import benchmark

print("torch:", torch.__version__)
print("mps:", torch.backends.mps.is_available())
print("mujoco:", mujoco.__version__)
print("libero suites:", sorted(benchmark.get_benchmark_dict().keys())[:4], "...")
if not torch.backends.mps.is_available():
    raise SystemExit("MPS not available")
PY

echo "Setup OK. Run: bash scripts/lerobot_libero_eval_one.sh"
echo ""
echo "Before eval: accept https://huggingface.co/google/paligemma-3b-pt-224 license"
echo "and run: hf auth login   (or export HF_TOKEN=...)"
