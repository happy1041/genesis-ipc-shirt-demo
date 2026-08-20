#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"

# 8000 faces preserve the two-layer hem geometry around the public first-grasp
# locations while remaining substantially cheaper than the 13767-face source.
CONTACT_MESH_FACES="${CONTACT_MESH_FACES:-8000}" \
  "$PROJECT_ROOT/run_contact_grasp_test.sh" \
  --frames 380 \
  --seed 0 \
  --camera-view overhead \
  --output "$OUTPUT_ROOT/genesis_ipc_contact_first_fold_8000f_380_overhead.mp4" \
  "$@"
