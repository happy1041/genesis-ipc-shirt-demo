#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
FINAL_LABEL="${RUN_LABEL:-g3v2_13767f_dhat15_purefriction_staged_slow}"
PHYSICS_LABEL="${FINAL_LABEL}_physics"
REPLAY_STATES="$OUTPUT_ROOT/${PHYSICS_LABEL}.replay_states.npz"

export USE_SOFT_VIRTUAL_GRASP=0
export VISUALIZE_IPC_PROXIES="${VISUALIZE_IPC_PROXIES:-0}"
export RENDER_IPC_ACTUAL_VISUALS="${RENDER_IPC_ACTUAL_VISUALS:-1}"
export SHOW_VIEWER="${SHOW_VIEWER:-0}"
export SUBSTEPS="${SUBSTEPS:-2}"
export DEMO_FRAMES="${DEMO_FRAMES:-980}"

common_args=(
  --shirt-obj "$OUTPUT_ROOT/assets/short-shirt.obj"
  --cloth-thickness 0.0001
  --contact-d-hat 0.0015
  --cloth-friction 1
  --table-friction 1
  --robot-friction 2
  --cloth-E 20000
  --first-grasp-left-depth 0.008
  --first-grasp-right-depth 0.003
  --first-left-staged-close
  --post-release-settle-frames 0
  --no-fast-preview
)

# Rendering during IPC stepping changes CUDA scheduling and can send marginal
# contact solves down a different branch. Persist physics first, then render it.
RUN_LABEL="$PHYSICS_LABEL" CHECKPOINT_MODE=off RECORD_MULTI_VIEW=0 \
  "$PROJECT_ROOT/run_g3v2_majorfix_8000.sh" \
  "${common_args[@]}" \
  --no-record \
  --dump-replay-states "$REPLAY_STATES" \
  "$@"

RUN_LABEL="$FINAL_LABEL" CHECKPOINT_MODE=off RECORD_MULTI_VIEW=1 \
  "$PROJECT_ROOT/run_g3v2_majorfix_8000.sh" \
  "${common_args[@]}" \
  --no-record \
  --replay-states "$REPLAY_STATES" \
  "$@"

echo "Physics metrics: $OUTPUT_ROOT/${PHYSICS_LABEL}.json"
echo "Multi-view video: $OUTPUT_ROOT/${FINAL_LABEL}_multiview.mp4"
echo "Replay manifest: $OUTPUT_ROOT/${FINAL_LABEL}.replay.json"
