#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SIM1_ROOT:?Set SIM1_ROOT to a clean SIM1 checkout}"

if [[ -z "${GENESIS_PYTHON:-}" ]]; then
  : "${GENESIS_ENV:?Set GENESIS_ENV or GENESIS_PYTHON}"
  GENESIS_PYTHON="$GENESIS_ENV/bin/python"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
SHIRT_OBJ="${SHIRT_OBJ:-$OUTPUT_ROOT/assets/short-shirt.obj}"
ROBOT_URDF="${ROBOT_URDF:-$OUTPUT_ROOT/assets/acone_ipc/acone_ipc.urdf}"
if [[ -z "${TRAJECTORY:-}" ]]; then
  : "${EPISODE_ROOT:?Set EPISODE_ROOT or TRAJECTORY}"
  TRAJECTORY="$EPISODE_ROOT/episode_000000_sim1_replay.npz"
fi

export GENESIS_PYTHON OUTPUT_ROOT SHIRT_OBJ ROBOT_URDF
if [[ ! -f "$SHIRT_OBJ" || ! -f "$ROBOT_URDF" ]]; then
  "$PROJECT_ROOT/scripts/prepare_assets.sh"
fi
for required in "$GENESIS_PYTHON" "$TRAJECTORY" "$SHIRT_OBJ" "$ROBOT_URDF"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path does not exist: $required" >&2
    exit 2
  fi
done

GENESIS_RUN_LOCK="${GENESIS_RUN_LOCK:-/tmp/genesis_ipc_shirt_demo_${UID}.lock}"
exec 9>"$GENESIS_RUN_LOCK"
if ! flock -n 9; then
  echo "Another Genesis shirt-demo process owns $GENESIS_RUN_LOCK" >&2
  exit 75
fi

if [[ -n "${GENESIS_GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GENESIS_GPU"
fi

purelib="$($GENESIS_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
nvidia_root="$purelib/nvidia"
cuda_libs=()
for relative in cu13/lib cuda_nvrtc/lib cublas/lib cusparse/lib cusolver/lib nvjitlink/lib; do
  [[ -d "$nvidia_root/$relative" ]] && cuda_libs+=("$nvidia_root/$relative")
done
if ((${#cuda_libs[@]})); then
  cuda_path="$(IFS=:; echo "${cuda_libs[*]}")"
  export LD_LIBRARY_PATH="$cuda_path:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$OUTPUT_ROOT"
exec "$GENESIS_PYTHON" "$PROJECT_ROOT/src/run_genesis_ipc.py" \
  --sim1-root "$SIM1_ROOT" \
  --trajectory "$TRAJECTORY" \
  --shirt-obj "$SHIRT_OBJ" \
  --robot-urdf "$ROBOT_URDF" \
  "$@"
