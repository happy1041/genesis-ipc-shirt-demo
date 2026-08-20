#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"

variants=(
  "8000 fast"
  "8000 slow"
  "13767 fast"
  "13767 slow"
)

for variant in "${variants[@]}"; do
  read -r faces solver <<<"$variant"
  label="ablation_${faces}f_${solver}_original_urdf_full"
  echo "===== START $label $(date --iso-8601=seconds) ====="
  "$PROJECT_ROOT/run_ablation_full.sh" "$faces" "$solver" \
    2>&1 | tee "$OUTPUT_ROOT/${label}.log"
  echo "===== DONE $label $(date --iso-8601=seconds) ====="
done
