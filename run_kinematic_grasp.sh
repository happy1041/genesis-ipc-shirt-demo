#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"

# Deterministic attachment smoke test. This deliberately disables IPC contact
# and gravity; it validates trajectory, TCP transforms, vertex selection and
# release timing, but is not a physical folding-quality result.
"$PROJECT_ROOT/run_fast.sh" \
  --frames 240 \
  --virtual-grasp \
  --grasp-mode hard \
  --prepin-first-grasp \
  --grasp-radius 0.15 \
  --grasp-points 6 \
  --final-right-grasp-points 6 \
  --camera-view oblique \
  --output "$OUTPUT_ROOT/genesis_kinematic_grasp_first_lift_oblique.mp4" \
  "$@"
