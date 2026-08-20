#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
QUEUE_LOG="$OUTPUT_ROOT/ablation_remaining_background.status.log"

run_variant() {
  local faces="$1"
  local solver="$2"
  local label="ablation_${faces}f_${solver}_original_urdf_full"
  local log="$OUTPUT_ROOT/${label}.log"

  printf 'START %s %s\n' "$label" "$(date --iso-8601=seconds)" | tee -a "$QUEUE_LOG"
  "$PROJECT_ROOT/run_ablation_full.sh" "$faces" "$solver" >"$log" 2>&1
  printf 'DONE  %s %s\n' "$label" "$(date --iso-8601=seconds)" | tee -a "$QUEUE_LOG"
}

# Deliberately excludes 13767/slow: the user accepted the earlier high-quality
# run as sufficient and asked us not to repeat that expensive variant.
run_variant 8000 slow
run_variant 13767 fast

printf 'QUEUE COMPLETE %s\n' "$(date --iso-8601=seconds)" | tee -a "$QUEUE_LOG"
