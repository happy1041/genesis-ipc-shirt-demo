#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM1_ROOT="${SIM1_ROOT:-/home/happy1041/Workspace/SIM1}"
GENESIS_ENV="${GENESIS_ENV:-/home/happy1041/work/genesis-ipc-env}"
EPISODE_ROOT="${EPISODE_ROOT:-/home/happy1041/work/Sim1SameTrajectory/scale_fold_shirt_in_domain/episode_000000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
SHIRT_OBJ="${SHIRT_OBJ:-$OUTPUT_ROOT/assets/short-shirt.obj}"
TRAJECTORY="${TRAJECTORY:-$EPISODE_ROOT/episode_000000_sim1_replay.npz}"
ROBOT_URDF="${ROBOT_URDF:-}"

# Genesis/CUDA teardown can block inside the NVIDIA driver if a second scene is
# started before the first one has fully released its context.  Keep every
# wrapper in this project behind one process-lifetime lock.
GENESIS_RUN_LOCK="${GENESIS_RUN_LOCK:-/tmp/genesis_ipc_shirt_demo_${UID}.lock}"
exec 9>"$GENESIS_RUN_LOCK"
if ! flock -n 9; then
  echo "Another Genesis shirt-demo process still owns $GENESIS_RUN_LOCK" >&2
  echo "Wait for it to exit completely before starting another run." >&2
  exit 75
fi

mkdir -p "$OUTPUT_ROOT/assets"
if [[ ! -f "$SHIRT_OBJ" ]]; then
  /home/happy1041/work/sim1-linux-env/bin/python \
    "$PROJECT_ROOT/export_usdc_mesh.py" \
    "$SIM1_ROOT/assets/cloth/short-shirt.usdc" \
    "$SHIRT_OBJ"
fi

# Pin by UUID because CUDA and nvidia-smi enumerate these two GPUs differently.
# This UUID is the desktop NVIDIA GeForce RTX 5070 Ti on the current machine.
export CUDA_VISIBLE_DEVICES="${GENESIS_GPU:-GPU-573ea767-dda1-d1a3-84e3-aa0b5697120c}"
NVIDIA_WHEEL_ROOT="$GENESIS_ENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_WHEEL_ROOT/cu13/lib:$NVIDIA_WHEEL_ROOT/cuda_nvrtc/lib:$NVIDIA_WHEEL_ROOT/cublas/lib:$NVIDIA_WHEEL_ROOT/cusparse/lib:$NVIDIA_WHEEL_ROOT/cusolver/lib:$NVIDIA_WHEEL_ROOT/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

# The host account cannot install python3-tk system-wide without an interactive
# sudo password.  Use the Ubuntu packages extracted under the user's work
# directory so the Genesis viewer can open its recording save dialog.
LOCAL_TK_ROOT="${LOCAL_TK_ROOT:-/home/happy1041/work/local-debs/tkinter/root}"
if [[ -f "$LOCAL_TK_ROOT/usr/lib/python3.12/lib-dynload/_tkinter.cpython-312-x86_64-linux-gnu.so" ]]; then
  export PYTHONPATH="$LOCAL_TK_ROOT/usr/lib/python3.12:$LOCAL_TK_ROOT/usr/lib/python3.12/lib-dynload:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="$LOCAL_TK_ROOT/usr/lib:$LOCAL_TK_ROOT/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
  export TK_LIBRARY="$LOCAL_TK_ROOT/usr/share/tcltk/tk8.6"
fi

robot_urdf_args=()
if [[ -n "$ROBOT_URDF" ]]; then
  robot_urdf_args+=(--robot-urdf "$ROBOT_URDF")
fi

"$GENESIS_ENV/bin/python" "$PROJECT_ROOT/run_genesis_ipc.py" \
  --sim1-root "$SIM1_ROOT" \
  --trajectory "$TRAJECTORY" \
  --shirt-obj "$SHIRT_OBJ" \
  --output "$OUTPUT_ROOT/genesis_ipc_episode_000000.mp4" \
  "${robot_urdf_args[@]}" \
  "$@"
