#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
ROBOT_PROXY_URDF="${ROBOT_PROXY_URDF:-$OUTPUT_ROOT/assets/acone_collision4000/acone_collision4000.urdf}"
TARGET_GPU_UUID="${GENESIS_GPU:-GPU-573ea767-dda1-d1a3-84e3-aa0b5697120c}"

# A previous full-quality run filled the disk, and a later run wedged inside
# the NVIDIA runtime-PM resume path.  Fail before allocating the IPC scene
# instead of discovering either condition midway through a 15k-face run.
available_kib="$(df -Pk "$OUTPUT_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}')"
minimum_kib=$((20 * 1024 * 1024))
if [[ -n "$available_kib" && "$available_kib" -lt "$minimum_kib" ]]; then
  echo "Refusing high-quality run: less than 20 GiB is free under $OUTPUT_ROOT" >&2
  exit 78
fi

if [[ "${SKIP_GPU_RUNTIME_PM_GUARD:-0}" != "1" ]]; then
  gpu_bus_raw="$(
    nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv,noheader,nounits |
      awk -F',' -v uuid="$TARGET_GPU_UUID" '{gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $2); if ($1 == uuid) print $2}'
  )"
  if [[ -z "$gpu_bus_raw" ]]; then
    echo "Cannot resolve target GPU UUID $TARGET_GPU_UUID" >&2
    exit 78
  fi
  # nvidia-smi commonly prints 00000000:08:00.0; sysfs uses 0000:08:00.0.
  gpu_bus_short="${gpu_bus_raw#00000000:}"
  gpu_bus="0000:${gpu_bus_short}"
  gpu_power_control="/sys/bus/pci/devices/$gpu_bus/power/control"
  if [[ -r "$gpu_power_control" ]]; then
    gpu_power_mode="$(<"$gpu_power_control")"
    if [[ "$gpu_power_mode" != "on" ]]; then
      echo "Refusing high-quality run: $TARGET_GPU_UUID ($gpu_bus) runtime power mode is '$gpu_power_mode'." >&2
      echo "The NVIDIA 580.173.02 driver previously deadlocked in rpm_resume under this workload." >&2
      echo "Before rerunning, execute:" >&2
      echo "  echo on | sudo tee $gpu_power_control" >&2
      echo "Set SKIP_GPU_RUNTIME_PM_GUARD=1 only if you intentionally accept that risk." >&2
      exit 78
    fi
  fi
fi

if [[ ! -f "$ROBOT_PROXY_URDF" ]]; then
  /home/happy1041/work/genesis-ipc-env/bin/python \
    "$PROJECT_ROOT/prepare_acone_collision_proxy.py" \
    /home/happy1041/Workspace/SIM1/assets/acone/acone.urdf \
    "$ROBOT_PROXY_URDF" \
    --faces 4000
fi

# Full-process validation of the accepted V2 folds and the current third-fold
# trajectory.  Unlike the interactive tuning entrypoint, this uses the
# original exported SIM1 shirt mesh (13,767 faces) and the complete IPC solver.
FAST_PREVIEW=0 \
GENESIS_GPU="$TARGET_GPU_UUID" \
SHIRT_OBJ="${SHIRT_OBJ:-$OUTPUT_ROOT/assets/short-shirt.obj}" \
ROBOT_URDF="$ROBOT_PROXY_URDF" \
CHECKPOINT_MODE=save \
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$OUTPUT_ROOT/checkpoints/current_best_slow_original_pre_third.pkl}" \
RUN_LABEL="${RUN_LABEL:-current_best_slow_original_collision4k_full_trajectory}" \
ARC_MESH_FACES=13767 \
DEMO_FRAMES="${DEMO_FRAMES:-946}" \
SHOW_VIEWER="${SHOW_VIEWER:-1}" \
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
  "$PROJECT_ROOT/run_three_folds_arc.sh" "$@"
