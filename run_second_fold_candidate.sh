#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 LABEL SECOND_FOLD_STACK_OVERLAP [SECOND_FOLD_ROLL_ARC_HEIGHT] [SECOND_FOLD_PLACEMENT_RELAX] [SECOND_FOLD_PLACEMENT_LIFT]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
label="$1"
overlap="$2"
arc_height="${3:-0.100}"
placement_relax="${4:-0.060}"
placement_lift="${5:-0.000}"
shirt_obj="$OUTPUT_ROOT/assets/short-shirt.obj"
robot_urdf="/home/happy1041/Workspace/SIM1/assets/acone/acone.urdf"

for required in "$shirt_obj" "$robot_urdf"; do
  [[ -f "$required" ]] || { echo "missing input: $required" >&2; exit 66; }
done

gpu_power_control="/sys/bus/pci/devices/0000:08:00.0/power/control"
gpu_runtime_status="/sys/bus/pci/devices/0000:08:00.0/power/runtime_status"
gpu_suspended_time="/sys/bus/pci/devices/0000:08:00.0/power/runtime_suspended_time"
if [[ -r "$gpu_power_control" && "$(<"$gpu_power_control")" != "on" ]]; then
  # After the latest reboot the 5070 Ti remained P0/active continuously even
  # though the generic PCI policy says auto.  This is safe to use because no
  # resume is required.  Reject as soon as this boot has recorded any suspend.
  if [[ ! -r "$gpu_runtime_status" || ! -r "$gpu_suspended_time" \
        || "$(<"$gpu_runtime_status")" != "active" \
        || "$(<"$gpu_suspended_time")" != "0" ]]; then
    echo "refusing run: RTX 5070 Ti is not continuously active" >&2
    exit 78
  fi
fi

echo "candidate_config label=$label overlap=$overlap arc_height=$arc_height placement_relax=$placement_relax placement_lift=$placement_lift faces=13767 solver=fast viewer=off frames=620"

FAST_PREVIEW=1 \
FAST_OBJ="$shirt_obj" \
GENESIS_GPU="${GENESIS_GPU:-GPU-573ea767-dda1-d1a3-84e3-aa0b5697120c}" \
SHIRT_OBJ="$shirt_obj" \
ROBOT_URDF="$robot_urdf" \
CHECKPOINT_MODE=save \
CHECKPOINT_FILE="$OUTPUT_ROOT/checkpoints/${label}_pre_third.pkl" \
RUN_LABEL="$label" \
ARC_MESH_FACES=13767 \
DEMO_FRAMES=620 \
SHOW_VIEWER=0 \
TABLE_FRICTION=1.0 \
FIRST_FOLD_STACK_OVERLAP=-0.015 \
SECOND_FOLD_TRANSPORT_LIFT=0.000 \
SECOND_FOLD_ROLL_ARC_HEIGHT="$arc_height" \
SECOND_FOLD_ROLL_PATH=staged \
SECOND_FOLD_PLACEMENT_RELAX="$placement_relax" \
SECOND_FOLD_PLACEMENT_LIFT="$placement_lift" \
SECOND_FOLD_STACK_OVERLAP="$overlap" \
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
POST_RELEASE_OPEN_HOLD_FRAMES=0 \
POST_RELEASE_RETREAT_FRAMES=0 \
POST_RELEASE_SETTLE_FRAMES=0 \
  "$PROJECT_ROOT/run_three_folds_arc.sh" --settle-frames 30

"/home/happy1041/work/genesis-ipc-env/bin/python" \
  "$PROJECT_ROOT/analyze_second_fold_quality.py" \
  "$OUTPUT_ROOT/${label}_second_fold_state" \
  --reference "$OUTPUT_ROOT/ablation_8000f_slow_original_urdf_full_second_fold_state"
