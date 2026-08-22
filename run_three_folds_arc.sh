#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
ARC_MESH_FACES="${ARC_MESH_FACES:-8000}"
DEMO_FRAMES="${DEMO_FRAMES:-980}"
THIRD_FOLD_LIFT="${THIRD_FOLD_LIFT:-0.080}"
THIRD_FOLD_OUTWARD_PULL_CANCEL="${THIRD_FOLD_OUTWARD_PULL_CANCEL:-0.000}"
THIRD_FOLD_TOP_OFFSET="${THIRD_FOLD_TOP_OFFSET:-0.040}"
FIRST_FOLD_STACK_OVERLAP="${FIRST_FOLD_STACK_OVERLAP:--0.015}"
FIRST_FOLD_TCP_LIFT="${FIRST_FOLD_TCP_LIFT:-0.015}"
FIRST_FOLD_TRANSFER_LIFT="${FIRST_FOLD_TRANSFER_LIFT:-0.070}"
FIRST_GRASP_CLEARANCE_LIFT="${FIRST_GRASP_CLEARANCE_LIFT:-0.035}"
SECOND_FOLD_TRANSPORT_LIFT="${SECOND_FOLD_TRANSPORT_LIFT:-0.000}"
SECOND_FOLD_ROLL_ARC_HEIGHT="${SECOND_FOLD_ROLL_ARC_HEIGHT:-0.060}"
SECOND_FOLD_ROLL_PATH="${SECOND_FOLD_ROLL_PATH:-staged}"
SECOND_FOLD_PLACEMENT_RELAX="${SECOND_FOLD_PLACEMENT_RELAX:-0.060}"
SECOND_FOLD_PLACEMENT_LIFT="${SECOND_FOLD_PLACEMENT_LIFT:-0.000}"
SECOND_FOLD_STACK_OVERLAP="${SECOND_FOLD_STACK_OVERLAP:-0.000}"
SECOND_FOLD_CORRECTION_RELEASE_START="${SECOND_FOLD_CORRECTION_RELEASE_START:-600}"
SECOND_FOLD_CORRECTION_RELEASE_END="${SECOND_FOLD_CORRECTION_RELEASE_END:-619}"
TABLE_FRICTION="${TABLE_FRICTION:-0.6}"
THIRD_FOLD_GRASP_DEPTH="${THIRD_FOLD_GRASP_DEPTH:-0.000}"
THIRD_FOLD_GRASP_LATERAL="${THIRD_FOLD_GRASP_LATERAL:-0.012}"
THIRD_FOLD_GRASP_WORLD_X="${THIRD_FOLD_GRASP_WORLD_X:--0.095}"
THIRD_FOLD_GRASP_WORLD_Y="${THIRD_FOLD_GRASP_WORLD_Y:-0.000}"
THIRD_FOLD_PLACEMENT_LEVEL="${THIRD_FOLD_PLACEMENT_LEVEL:-0.000}"
THIRD_FOLD_FRONT_PLANE_ROLL_DEG="${THIRD_FOLD_FRONT_PLANE_ROLL_DEG:-0.000}"
THIRD_FOLD_SMOOTH_ROTATION="${THIRD_FOLD_SMOOTH_ROTATION:-0}"
THIRD_FOLD_RELEASE_HOLD="${THIRD_FOLD_RELEASE_HOLD:-0}"
POST_RELEASE_SETTLE_FRAMES="${POST_RELEASE_SETTLE_FRAMES:-0}"
POST_RELEASE_OPEN_HOLD_FRAMES="${POST_RELEASE_OPEN_HOLD_FRAMES:-0}"
POST_RELEASE_RETREAT_FRAMES="${POST_RELEASE_RETREAT_FRAMES:-0}"
POST_RELEASE_RETREAT_HEIGHT="${POST_RELEASE_RETREAT_HEIGHT:-0.0}"
POST_RELEASE_RETREAT_TOP_OFFSET="${POST_RELEASE_RETREAT_TOP_OFFSET:-0.0}"
CONTACT_D_HAT="${CONTACT_D_HAT:-0.005}"
CONTACT_CONSTITUTION="${CONTACT_CONSTITUTION:-ipc}"
SETTLE_FRAMES="${SETTLE_FRAMES:-30}"
SUBSTEPS="${SUBSTEPS:-1}"
SHOW_VIEWER="${SHOW_VIEWER:-1}"
RECORD_MULTI_VIEW="${RECORD_MULTI_VIEW:-1}"
IPC_RIGID_RIGID_CONTACT="${IPC_RIGID_RIGID_CONTACT:-1}"
VISUALIZE_IPC_PROXIES="${VISUALIZE_IPC_PROXIES:-0}"
RENDER_IPC_ACTUAL_VISUALS="${RENDER_IPC_ACTUAL_VISUALS:-0}"
IPC_CONSTRAINT_STRENGTH_TRANSLATION="${IPC_CONSTRAINT_STRENGTH_TRANSLATION:-100}"
IPC_CONSTRAINT_STRENGTH_ROTATION="${IPC_CONSTRAINT_STRENGTH_ROTATION:-500}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-save}"
CHECKPOINT_SOURCE_FRAME="${CHECKPOINT_SOURCE_FRAME:-583}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$OUTPUT_ROOT/checkpoints/pre_third_fold_${ARC_MESH_FACES}f.pkl}"
RUN_LABEL="${RUN_LABEL:-genesis_ipc_three_folds_arc_lift80_top40_${ARC_MESH_FACES}f_${CHECKPOINT_MODE}}"
mkdir -p "$OUTPUT_ROOT"

viewer_args=()
if [[ "$SHOW_VIEWER" != "0" ]]; then
  viewer_args+=(--viewer)
fi

ipc_debug_args=()
if [[ "$IPC_RIGID_RIGID_CONTACT" != "0" ]]; then
  ipc_debug_args+=(--ipc-rigid-rigid-contact)
else
  ipc_debug_args+=(--no-ipc-rigid-rigid-contact)
fi

record_args=()
if [[ "$RECORD_MULTI_VIEW" != "0" ]]; then
  record_args+=(--record-multi-view)
fi
if [[ "$VISUALIZE_IPC_PROXIES" != "0" ]]; then
  ipc_debug_args+=(--visualize-ipc-proxies)
fi
if [[ "$RENDER_IPC_ACTUAL_VISUALS" != "0" ]]; then
  ipc_debug_args+=(--render-ipc-actual-visuals)
fi

checkpoint_args=(--third-fold-checkpoint-source-frame "$CHECKPOINT_SOURCE_FRAME")
if [[ "$THIRD_FOLD_RELEASE_HOLD" != "0" ]]; then
  checkpoint_args+=(--third-fold-release-hold)
fi
if [[ "$THIRD_FOLD_SMOOTH_ROTATION" != "0" ]]; then
  checkpoint_args+=(--third-fold-smooth-rotation)
fi
case "$CHECKPOINT_MODE" in
  save)
    checkpoint_args+=(--save-third-fold-checkpoint "$CHECKPOINT_FILE")
    ;;
  load)
    checkpoint_args+=(--load-third-fold-checkpoint "$CHECKPOINT_FILE")
    ;;
  off)
    ;;
  *)
    echo "CHECKPOINT_MODE must be save, load, or off" >&2
    exit 2
    ;;
esac

# Shirt-centric frame for episode 0:
#   +Y: waist/bottom -> collar/top, +Z: away from the table.
# Fold three keeps the calibrated centered grasp, lifts the waist panel before
# transporting it, lands 40 mm farther toward the collar, and removes the old
# 30 mm downward placement correction that compressed the folded stack.
CONTACT_MESH_FACES="$ARC_MESH_FACES" \
  "$PROJECT_ROOT/run_contact_grasp_test.sh" \
  --frames "$DEMO_FRAMES" \
  --substeps "$SUBSTEPS" \
  --seed 0 \
  --camera-view oblique \
  "${record_args[@]}" \
  --cloth-rho 800 \
  --table-friction "$TABLE_FRICTION" \
  --contact-d-hat "$CONTACT_D_HAT" \
  --contact-constitution "$CONTACT_CONSTITUTION" \
  --settle-frames "$SETTLE_FRAMES" \
  --first-grasp-clearance-lift "$FIRST_GRASP_CLEARANCE_LIFT" \
  --first-fold-tcp-lift "$FIRST_FOLD_TCP_LIFT" \
  --first-fold-transfer-lift "$FIRST_FOLD_TRANSFER_LIFT" \
  --first-fold-stack-overlap "$FIRST_FOLD_STACK_OVERLAP" \
  --second-fold-left-approach-lift 0.06 \
  --second-fold-right-approach-lift 0.08 \
  --second-fold-transport-lift "$SECOND_FOLD_TRANSPORT_LIFT" \
  --second-fold-roll-arc-height "$SECOND_FOLD_ROLL_ARC_HEIGHT" \
  --second-fold-roll-path "$SECOND_FOLD_ROLL_PATH" \
  --second-fold-placement-relax "$SECOND_FOLD_PLACEMENT_RELAX" \
  --second-fold-placement-lift "$SECOND_FOLD_PLACEMENT_LIFT" \
  --second-fold-stack-overlap "$SECOND_FOLD_STACK_OVERLAP" \
  --second-fold-correction-release-start "$SECOND_FOLD_CORRECTION_RELEASE_START" \
  --second-fold-correction-release-end "$SECOND_FOLD_CORRECTION_RELEASE_END" \
  --third-fold-right-grasp-lift 0.04 \
  --third-fold-right-grasp-depth "$THIRD_FOLD_GRASP_DEPTH" \
  --third-fold-right-grasp-lateral "$THIRD_FOLD_GRASP_LATERAL" \
  --third-fold-right-grasp-world-x "$THIRD_FOLD_GRASP_WORLD_X" \
  --third-fold-right-grasp-world-y "$THIRD_FOLD_GRASP_WORLD_Y" \
  --third-fold-placement-depth "${THIRD_FOLD_PLACEMENT_DEPTH:-0.0}" \
  --third-fold-post-close-lift "$THIRD_FOLD_LIFT" \
  --third-fold-outward-pull-cancel "$THIRD_FOLD_OUTWARD_PULL_CANCEL" \
  --third-fold-shirt-top-offset "$THIRD_FOLD_TOP_OFFSET" \
  --third-fold-placement-level "$THIRD_FOLD_PLACEMENT_LEVEL" \
  --third-fold-front-plane-roll-deg "$THIRD_FOLD_FRONT_PLANE_ROLL_DEG" \
  --post-release-settle-frames "$POST_RELEASE_SETTLE_FRAMES" \
  --post-release-open-hold-frames "$POST_RELEASE_OPEN_HOLD_FRAMES" \
  --post-release-retreat-frames "$POST_RELEASE_RETREAT_FRAMES" \
  --post-release-retreat-height "$POST_RELEASE_RETREAT_HEIGHT" \
  --post-release-retreat-top-offset "$POST_RELEASE_RETREAT_TOP_OFFSET" \
  --debug-third-fold-dir "$OUTPUT_ROOT/${RUN_LABEL}_third_fold_state" \
  --debug-second-fold-dir "$OUTPUT_ROOT/${RUN_LABEL}_second_fold_state" \
  --keyframe-diagnostics-dir "$OUTPUT_ROOT/${RUN_LABEL}_keyframes" \
  --ipc-constraint-strength-translation "$IPC_CONSTRAINT_STRENGTH_TRANSLATION" \
  --ipc-constraint-strength-rotation "$IPC_CONSTRAINT_STRENGTH_ROTATION" \
  "${ipc_debug_args[@]}" \
  "${checkpoint_args[@]}" \
  --output "$OUTPUT_ROOT/${RUN_LABEL}.mp4" \
  "${viewer_args[@]}" \
  "$@"
