#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
CONTACT_MESH_FACES="${CONTACT_MESH_FACES:-8000}"
FAST_PREVIEW="${FAST_PREVIEW:-1}"

# No virtual anchors are enabled here. The complete gripper pose replays the
# public SIM1 trajectory as a prescribed rigid boundary. The packed official
# action is a negatively scaled openness: 0=closed and -3.24=open. run.sh
# decodes it to normalized openness where 1=open and 0=closed. Its recorded
# closed target (~1 mm) is advanced only to the URDF lower limit (0 mm), never
# through the limit or past the opposing finger. IPC contact supplies normal
# pressure and friction while preventing cloth penetration.
runner=("$PROJECT_ROOT/run_fast.sh")
runner_env=(FAST_FACES="$CONTACT_MESH_FACES")
if [[ "$FAST_PREVIEW" == "0" ]]; then
  # Preserve the exported SIM1 mesh and enable the complete IPC solver.  This
  # path intentionally does not add --fast-preview or decimate the cloth.
  runner=("$PROJECT_ROOT/run.sh")
  runner_env=()
fi

grasp_args=(--contact-grasp-test)
if [[ "${USE_SOFT_VIRTUAL_GRASP:-0}" == "1" ]]; then
  grasp_args=(
    --virtual-grasp
    --grasp-mode soft
    --prepin-first-grasp
    --grasp-strength "${SOFT_VIRTUAL_GRASP_STRENGTH:-20}"
  )
fi

env "${runner_env[@]}" "${runner[@]}" \
  --frames 240 \
  --drive-mode direct \
  "${grasp_args[@]}" \
  --initial-shirt-x 0.667 \
  --initial-shirt-y 0.015 \
  --contact-d-hat 0.005 \
  --finger-overclose 0.001 \
  --right-finger-overclose-extra 0.00025 \
  --finger-kp 1000 \
  --finger-kv 50 \
  --robot-friction 2.0 \
  --cloth-friction 1.0 \
  --exact-finger-collision \
  --camera-view oblique \
  --output "$OUTPUT_ROOT/genesis_ipc_contact_grasp_first_lift_contact_aligned.mp4" \
  "$@"
