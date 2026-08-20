#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 LABEL {fast|slow} SECOND_FOLD_STACK_OVERLAP [SECOND_FOLD_PLACEMENT_RELAX] [SECOND_FOLD_PLACEMENT_LIFT]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
label="$1"
solver="$2"
overlap="$3"
placement_relax="${4:-0.060}"
placement_lift="${5:-0.000}"
shirt_obj="$OUTPUT_ROOT/assets/short-shirt.obj"
robot_urdf="/home/happy1041/Workspace/SIM1/assets/acone/acone.urdf"

case "$solver" in
  fast) fast_preview=1 ;;
  slow) fast_preview=0 ;;
  *) echo "solver must be fast or slow" >&2; exit 2 ;;
esac

gpu_power_control="/sys/bus/pci/devices/0000:08:00.0/power/control"
gpu_runtime_status="/sys/bus/pci/devices/0000:08:00.0/power/runtime_status"
gpu_suspended_time="/sys/bus/pci/devices/0000:08:00.0/power/runtime_suspended_time"
if [[ -r "$gpu_power_control" && "$(<"$gpu_power_control")" != "on" ]]; then
  if [[ ! -r "$gpu_runtime_status" || ! -r "$gpu_suspended_time" \
        || "$(<"$gpu_runtime_status")" != "active" \
        || "$(<"$gpu_suspended_time")" != "0" ]]; then
    echo "refusing run: RTX 5070 Ti is not continuously active" >&2
    exit 78
  fi
fi

FAST_PREVIEW="$fast_preview" \
FAST_OBJ="$shirt_obj" \
GENESIS_GPU="${GENESIS_GPU:-GPU-573ea767-dda1-d1a3-84e3-aa0b5697120c}" \
SHIRT_OBJ="$shirt_obj" \
ROBOT_URDF="$robot_urdf" \
CHECKPOINT_MODE=save \
CHECKPOINT_FILE="$OUTPUT_ROOT/checkpoints/${label}_pre_third.pkl" \
RUN_LABEL="$label" \
ARC_MESH_FACES=13767 \
DEMO_FRAMES="${DEMO_FRAMES:-946}" \
SHOW_VIEWER=0 \
TABLE_FRICTION=1.0 \
FIRST_FOLD_STACK_OVERLAP=-0.015 \
SECOND_FOLD_TRANSPORT_LIFT=0.000 \
SECOND_FOLD_ROLL_ARC_HEIGHT=0.100 \
SECOND_FOLD_ROLL_PATH=staged \
SECOND_FOLD_PLACEMENT_RELAX="$placement_relax" \
SECOND_FOLD_PLACEMENT_LIFT="$placement_lift" \
SECOND_FOLD_STACK_OVERLAP="$overlap" \
THIRD_FOLD_GRASP_DEPTH="${THIRD_FOLD_GRASP_DEPTH:-0.005}" \
THIRD_FOLD_GRASP_LATERAL="${THIRD_FOLD_GRASP_LATERAL:-0.012}" \
THIRD_FOLD_GRASP_WORLD_X="${THIRD_FOLD_GRASP_WORLD_X:--0.095}" \
THIRD_FOLD_GRASP_WORLD_Y="${THIRD_FOLD_GRASP_WORLD_Y:--0.055}" \
THIRD_FOLD_LIFT="${THIRD_FOLD_LIFT:-0.080}" \
THIRD_FOLD_OUTWARD_PULL_CANCEL="${THIRD_FOLD_OUTWARD_PULL_CANCEL:-0.000}" \
THIRD_FOLD_PLACEMENT_DEPTH="${THIRD_FOLD_PLACEMENT_DEPTH:-0.000}" \
THIRD_FOLD_TOP_OFFSET="${THIRD_FOLD_TOP_OFFSET:-0.100}" \
THIRD_FOLD_PLACEMENT_LEVEL="${THIRD_FOLD_PLACEMENT_LEVEL:-0.800}" \
THIRD_FOLD_FRONT_PLANE_ROLL_DEG="${THIRD_FOLD_FRONT_PLANE_ROLL_DEG:-0.000}" \
THIRD_FOLD_RELEASE_HOLD="${THIRD_FOLD_RELEASE_HOLD:-1}" \
POST_RELEASE_OPEN_HOLD_FRAMES="${POST_RELEASE_OPEN_HOLD_FRAMES:-10}" \
POST_RELEASE_RETREAT_FRAMES="${POST_RELEASE_RETREAT_FRAMES:-40}" \
POST_RELEASE_RETREAT_HEIGHT="${POST_RELEASE_RETREAT_HEIGHT:-0.090}" \
POST_RELEASE_RETREAT_TOP_OFFSET="${POST_RELEASE_RETREAT_TOP_OFFSET:-0.000}" \
POST_RELEASE_SETTLE_FRAMES="${POST_RELEASE_SETTLE_FRAMES:-120}" \
  "$PROJECT_ROOT/run_three_folds_arc.sh" --settle-frames 30

"/home/happy1041/work/genesis-ipc-env/bin/python" \
  "$PROJECT_ROOT/analyze_second_fold_quality.py" \
  "$OUTPUT_ROOT/${label}_second_fold_state" \
  --reference "$OUTPUT_ROOT/ablation_8000f_slow_original_urdf_full_second_fold_state"
