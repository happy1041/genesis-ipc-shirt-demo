#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {8000|13767} {fast|slow}" >&2
  exit 2
fi

faces="$1"
solver="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
ORIGINAL_URDF="/home/happy1041/Workspace/SIM1/assets/acone/acone.urdf"

# Lets an already-running matrix honor a one-shot request to omit its final,
# expensive 13767/slow variant without changing the other controlled runs.
skip_13767_slow_marker="$PROJECT_ROOT/.skip_13767_slow_once"
if [[ "$faces" == "13767" && "$solver" == "slow" && -f "$skip_13767_slow_marker" ]]; then
  echo "SKIP ablation_13767f_slow_original_urdf_full (one-shot user request)"
  rm -f -- "$skip_13767_slow_marker"
  exit 0
fi

case "$faces" in
  8000) shirt_obj="$OUTPUT_ROOT/assets/short-shirt-8000f.obj" ;;
  13767) shirt_obj="$OUTPUT_ROOT/assets/short-shirt.obj" ;;
  *) echo "faces must be 8000 or 13767" >&2; exit 2 ;;
esac
case "$solver" in
  fast) fast_preview=1 ;;
  slow) fast_preview=0 ;;
  *) echo "solver must be fast or slow" >&2; exit 2 ;;
esac

for required in "$shirt_obj" "$ORIGINAL_URDF"; do
  [[ -f "$required" ]] || { echo "missing input: $required" >&2; exit 66; }
done

gpu_power_control="/sys/bus/pci/devices/0000:08:00.0/power/control"
if [[ -r "$gpu_power_control" ]] && [[ "$(<"$gpu_power_control")" != "on" ]]; then
  echo "refusing run: RTX 5070 Ti PCI runtime power control is not 'on'" >&2
  exit 78
fi

label="ablation_${faces}f_${solver}_original_urdf_full"

echo "ablation_config label=$label faces=$faces solver=$solver viewer=off frames=946"
echo "ablation_inputs shirt_obj=$shirt_obj robot_urdf=$ORIGINAL_URDF"

# Control-variable experiment.  All four variants use the same 30-frame
# initial settle, original robot URDF, accepted G3 V2 fold corrections, and
# current release/retreat tail.  Only cloth tessellation and IPC solver mode
# differ between invocations.
FAST_PREVIEW="$fast_preview" \
FAST_OBJ="$shirt_obj" \
GENESIS_GPU="${GENESIS_GPU:-GPU-573ea767-dda1-d1a3-84e3-aa0b5697120c}" \
SHIRT_OBJ="$shirt_obj" \
ROBOT_URDF="$ORIGINAL_URDF" \
CHECKPOINT_MODE=save \
CHECKPOINT_FILE="$OUTPUT_ROOT/checkpoints/${label}_pre_third.pkl" \
RUN_LABEL="$label" \
ARC_MESH_FACES="$faces" \
DEMO_FRAMES=946 \
SHOW_VIEWER=0 \
TABLE_FRICTION=1.0 \
FIRST_FOLD_STACK_OVERLAP=-0.015 \
SECOND_FOLD_TRANSPORT_LIFT=0.000 \
SECOND_FOLD_ROLL_ARC_HEIGHT=0.100 \
SECOND_FOLD_ROLL_PATH=staged \
SECOND_FOLD_PLACEMENT_RELAX=0.060 \
SECOND_FOLD_STACK_OVERLAP=0.100 \
THIRD_FOLD_GRASP_DEPTH=0.005 \
THIRD_FOLD_GRASP_LATERAL=0.012 \
THIRD_FOLD_GRASP_WORLD_X=-0.095 \
THIRD_FOLD_GRASP_WORLD_Y=-0.055 \
THIRD_FOLD_LIFT=0.040 \
THIRD_FOLD_OUTWARD_PULL_CANCEL=0.000 \
THIRD_FOLD_PLACEMENT_DEPTH=0.000 \
THIRD_FOLD_TOP_OFFSET=0.110 \
THIRD_FOLD_PLACEMENT_LEVEL=0.000 \
THIRD_FOLD_FRONT_PLANE_ROLL_DEG=0.000 \
THIRD_FOLD_RELEASE_HOLD=1 \
POST_RELEASE_OPEN_HOLD_FRAMES=10 \
POST_RELEASE_RETREAT_FRAMES=40 \
POST_RELEASE_RETREAT_HEIGHT=0.090 \
POST_RELEASE_RETREAT_TOP_OFFSET=0.000 \
POST_RELEASE_SETTLE_FRAMES=120 \
  "$PROJECT_ROOT/run_three_folds_arc.sh" --settle-frames 30
