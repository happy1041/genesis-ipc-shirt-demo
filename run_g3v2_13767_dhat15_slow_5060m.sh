#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

: "${GENESIS_GPU:=GPU-898ae6e2-1630-0a74-1d76-1653e9ffe972}"
: "${RUN_LABEL:=g3v2_13767f_dhat15_softgrasp_slow_full}"
export GENESIS_GPU RUN_LABEL
export USE_SOFT_VIRTUAL_GRASP=1
export SOFT_VIRTUAL_GRASP_STRENGTH=20
export VISUALIZE_IPC_PROXIES=0
export RENDER_IPC_ACTUAL_VISUALS=1
export SHOW_VIEWER=0
export SUBSTEPS=2
export DEMO_FRAMES=980

# dHat=1.5 mm needs the complete IPC solve on the 13,767-face mesh.  The
# fast solver can reach a near-zero negative TOI during the second fold.
exec ./run_g3v2_majorfix_8000.sh \
  --shirt-obj outputs/episode_000000/assets/short-shirt.obj \
  --cloth-thickness 0.0001 \
  --contact-d-hat 0.0015 \
  --cloth-friction 1 \
  --table-friction 1 \
  --robot-friction 2 \
  --cloth-E 20000 \
  --grasp-points 6 \
  --final-right-grasp-points 6 \
  --soft-grasp-first-capture-frames 20 \
  --first-grasp-right-depth 0.003 \
  --post-release-settle-frames 0 \
  --no-fast-preview \
  "$@"
