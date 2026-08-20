#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
LOG_ROOT="$OUTPUT_ROOT/tuning_logs"
STATUS="$LOG_ROOT/13767_autotune.status.log"
POWER_CONTROL="/sys/bus/pci/devices/0000:08:00.0/power/control"
RUNTIME_STATUS="/sys/bus/pci/devices/0000:08:00.0/power/runtime_status"
SUSPENDED_TIME="/sys/bus/pci/devices/0000:08:00.0/power/runtime_suspended_time"
mkdir -p "$LOG_ROOT"

status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "$STATUS"
}

gpu_ready() {
  [[ ! -r "$POWER_CONTROL" ]] && return 0
  [[ "$(<"$POWER_CONTROL")" == "on" ]] && return 0
  [[ -r "$RUNTIME_STATUS" && -r "$SUSPENDED_TIME" \
     && "$(<"$RUNTIME_STATUS")" == "active" \
     && "$(<"$SUSPENDED_TIME")" == "0" ]]
}

status "queue_started pid=$$ viewer=off"
while ! gpu_ready; do
  status "waiting_for_safe_gpu_state control=$(<"$POWER_CONTROL") status=$(<"$RUNTIME_STATUS") suspended_ms=$(<"$SUSPENDED_TIME")"
  sleep 30
done
status "gpu_power_ready control=$(<"$POWER_CONTROL") status=$(<"$RUNTIME_STATUS") suspended_ms=$(<"$SUSPENDED_TIME")"

declare -A overlaps=(
  [tune13767_overlap115]=0.115
  [tune13767_overlap125]=0.125
)

for label in tune13767_overlap115 tune13767_overlap125; do
  summary="$OUTPUT_ROOT/${label}_second_fold_state/second_fold_quality_summary.json"
  if [[ -s "$summary" ]]; then
    status "candidate_cached label=$label"
    continue
  fi
  status "candidate_start label=$label overlap=${overlaps[$label]}"
  set +e
  timeout --signal=TERM --kill-after=30s 30m \
    "$PROJECT_ROOT/run_second_fold_candidate.sh" "$label" "${overlaps[$label]}" 0.100 \
    > "$LOG_ROOT/${label}.log" 2>&1
  candidate_exit=$?
  set -e
  if [[ "$candidate_exit" -ne 0 ]]; then
    status "candidate_failed label=$label exit=$candidate_exit"
    exit 1
  fi
  status "candidate_done label=$label"
done

selection="$LOG_ROOT/13767_autotune_selection.tsv"
: > "$selection"
for label in tune13767_overlap115 tune13767_overlap125; do
  summary="$OUTPUT_ROOT/${label}_second_fold_state/second_fold_quality_summary.json"
  score="$(jq -r '.comparison.score_lower_is_better' "$summary")"
  printf '%s\t%s\t%s\n' "$score" "$label" "${overlaps[$label]}" >> "$selection"
done
read -r best_score best_label best_overlap < <(sort -g "$selection" | head -1)
status "selected label=$best_label overlap=$best_overlap score=$best_score"

fast_label="tuned13767_overlap${best_overlap/./}_fast_full"
status "full_fast_start label=$fast_label"
timeout --signal=TERM --kill-after=30s 45m \
  "$PROJECT_ROOT/run_13767_tuned_full.sh" "$fast_label" fast "$best_overlap" \
  > "$LOG_ROOT/${fast_label}.log" 2>&1
status "full_fast_done label=$fast_label"

# The complete solver is only worth the cost after the same trajectory has
# survived a full fast-solver run and produced all mandatory keyframes.
if [[ -s "$OUTPUT_ROOT/${fast_label}_keyframes/third_fold_contact_sheet.png" ]]; then
  slow_label="tuned13767_overlap${best_overlap/./}_slow_full"
  status "full_slow_start label=$slow_label"
  timeout --signal=TERM --kill-after=30s 90m \
    "$PROJECT_ROOT/run_13767_tuned_full.sh" "$slow_label" slow "$best_overlap" \
    > "$LOG_ROOT/${slow_label}.log" 2>&1
  status "full_slow_done label=$slow_label"
else
  status "full_slow_skipped missing_fast_visual_diagnostics"
  exit 1
fi

status "queue_complete"
