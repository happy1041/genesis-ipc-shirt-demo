#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${DEMO_CONFIG:-$PROJECT_ROOT/configs/dhat_1p5_pure_friction.args}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
RUN_LABEL="${RUN_LABEL:-g3v2_13767f_dhat15_purefriction_staged_slow}"
PHYSICS_LABEL="${RUN_LABEL}_physics"
REPLAY_STATES="$OUTPUT_ROOT/${PHYSICS_LABEL}.replay_states.npz"
mode=all
extra_args=()

usage() {
  cat <<'EOF'
Usage: scripts/run_demo.sh [--physics-only|--render-only] [extra runner args]

Required environment: SIM1_ROOT and either GENESIS_ENV or GENESIS_PYTHON.
Set TRAJECTORY directly, or set EPISODE_ROOT. Asset generation additionally
requires SIM1_PYTHON when the exported shirt OBJ is absent.
EOF
}

while (($#)); do
  case "$1" in
    --physics-only) mode=physics ;;
    --render-only) mode=render ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra_args+=("$@"); break ;;
    *) extra_args+=("$1") ;;
  esac
  shift
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Preset not found: $CONFIG" >&2
  exit 2
fi
mapfile -t preset_args < <(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "$CONFIG")
common_args=(
  "${preset_args[@]}"
  --debug-third-fold-dir "$OUTPUT_ROOT/${PHYSICS_LABEL}_third_fold_state"
  --debug-second-fold-dir "$OUTPUT_ROOT/${PHYSICS_LABEL}_second_fold_state"
  --keyframe-diagnostics-dir "$OUTPUT_ROOT/${PHYSICS_LABEL}_keyframes"
)

if [[ "$mode" != render ]]; then
  "$PROJECT_ROOT/scripts/_launch.sh" \
    "${common_args[@]}" "${extra_args[@]}" \
    --no-record \
    --dump-replay-states "$REPLAY_STATES" \
    --output "$OUTPUT_ROOT/${PHYSICS_LABEL}.mp4"
fi

if [[ "$mode" != physics ]]; then
  if [[ ! -f "$REPLAY_STATES" ]]; then
    echo "Replay states not found: $REPLAY_STATES" >&2
    exit 2
  fi
  "$PROJECT_ROOT/scripts/_launch.sh" \
    "${common_args[@]}" "${extra_args[@]}" \
    --record-multi-view \
    --no-record \
    --replay-states "$REPLAY_STATES" \
    --output "$OUTPUT_ROOT/${RUN_LABEL}.mp4"
fi

if [[ "$mode" != render ]]; then
  printf 'physics_metrics=%s\n' "$OUTPUT_ROOT/${PHYSICS_LABEL}.json"
fi
if [[ "$mode" != physics ]]; then
  printf 'video=%s\n' "$OUTPUT_ROOT/${RUN_LABEL}_multiview.mp4"
fi
