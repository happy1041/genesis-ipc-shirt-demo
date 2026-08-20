#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

: "${GENESIS_GPU:=GPU-898ae6e2-1630-0a74-1d76-1653e9ffe972}"
: "${RUN_LABEL:=g3v2_cloth_t01_fabric_render_8000}"

export GENESIS_GPU RUN_LABEL
export VISUALIZE_IPC_PROXIES=0
export RENDER_IPC_ACTUAL_VISUALS=1
export SHOW_VIEWER=0
export SUBSTEPS=2

exec ./run_g3v2_majorfix_8000.sh \
  --cloth-thickness 0.0001 \
  --contact-d-hat 0.002 \
  --cloth-friction 1 \
  --table-friction 1 \
  --robot-friction 2 \
  --cloth-E 20000 \
  --post-release-settle-frames 0 \
  --first-grasp-right-depth 0 \
  "$@"

