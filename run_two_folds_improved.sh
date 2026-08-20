#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"

# Shirt-centric intent:
# 1) first fold: high transfer, then near-vertical placement;
# 2) second fold setup: both robot arms clear the first folded flap during their
#    staggered approaches, then return to their respective public grasp poses.
CONTACT_MESH_FACES=8000 \
  "$PROJECT_ROOT/run_contact_grasp_test.sh" \
  --frames 640 \
  --seed 0 \
  --camera-view overhead \
  --table-friction 0.6 \
  --first-fold-tcp-lift 0.015 \
  --first-fold-transfer-lift 0.07 \
  --second-fold-left-approach-lift 0.06 \
  --second-fold-right-approach-lift 0.08 \
  --output "$OUTPUT_ROOT/genesis_ipc_two_folds_clear_approaches_8000f.mp4" \
  "$@"
