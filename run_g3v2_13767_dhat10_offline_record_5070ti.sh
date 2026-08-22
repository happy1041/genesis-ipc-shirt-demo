#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
FINAL_LABEL="${RUN_LABEL:-g3v2_13767f_dhat10_tcp_rotation_5070ti}"
PHYSICS_LABEL="${FINAL_LABEL}_physics"
REPLAY_STATES="$OUTPUT_ROOT/${PHYSICS_LABEL}.replay_states.npz"

export GENESIS_GPU="${GENESIS_GPU:-GPU-573ea767-dda1-d1a3-84e3-aa0b5697120c}"
export USE_SOFT_VIRTUAL_GRASP=1
export SOFT_VIRTUAL_GRASP_STRENGTH=20
export VISUALIZE_IPC_PROXIES=0
export RENDER_IPC_ACTUAL_VISUALS=1
export SHOW_VIEWER=0
export SUBSTEPS=2
export DEMO_FRAMES=980

common_args=(
  --shirt-obj "$OUTPUT_ROOT/assets/short-shirt.obj"
  --cloth-thickness 0.0001
  --contact-d-hat 0.001
  --cloth-friction 1
  --table-friction 1
  --robot-friction 2
  --cloth-E 20000
  --grasp-points 6
  --final-right-grasp-points 6
  --soft-grasp-first-capture-frames 20
  --first-grasp-right-depth 0.003
  --soft-grasp-second-translation-only
  --soft-grasp-final-right-rotate-about-tcp
  --no-soft-grasp-final-right-translation-only
  --post-release-settle-frames 0
  --no-fast-preview
)

# Real-time GPU rendering changes CUDA scheduling enough to send the marginal
# dHat=1 mm contact solve down a different branch. First compute and persist the
# authoritative physics trajectory with all camera recording disabled.
RUN_LABEL="$PHYSICS_LABEL" CHECKPOINT_MODE=off \
  "$PROJECT_ROOT/run_g3v2_majorfix_8000.sh" \
  "${common_args[@]}" \
  --no-record \
  --dump-replay-states "$REPLAY_STATES" \
  "$@"

# Render the saved trajectory without advancing IPC. Four camera views can now
# be generated without changing cloth contact or virtual-grasp convergence.
RUN_LABEL="$FINAL_LABEL" CHECKPOINT_MODE=off RECORD_MULTI_VIEW=1 \
  "$PROJECT_ROOT/run_g3v2_majorfix_8000.sh" \
  "${common_args[@]}" \
  --no-record \
  --replay-states "$REPLAY_STATES" \
  "$@"

echo "Physics metrics: $OUTPUT_ROOT/${PHYSICS_LABEL}.json"
echo "Replay manifest: $OUTPUT_ROOT/${FINAL_LABEL}.replay.json"

