#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"

# Preserve the double-layer grasp topology, keep the shirt body from sliding as
# freely as the first baseline, and turn the diagonal placement into a high
# transfer followed by a near-vertical descent at the fold destination.
CONTACT_MESH_FACES=8000 \
  "$PROJECT_ROOT/run_contact_grasp_test.sh" \
  --frames 380 \
  --seed 0 \
  --camera-view overhead \
  --table-friction 0.6 \
  --first-fold-tcp-lift 0.015 \
  --first-fold-transfer-lift 0.07 \
  --output "$OUTPUT_ROOT/genesis_ipc_first_fold_arc_improved_8000f.mp4" \
  "$@"
