#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
FAST_OBJ="$OUTPUT_ROOT/assets/short-shirt-2500f.obj"

if [[ ! -f "$FAST_OBJ" ]]; then
  "$PROJECT_ROOT/run_fast.sh" --frames 1 --settle-frames 0 --no-record
fi

# Stable scripted-grasp preset: low-resolution shirt with the strict IPC solve,
# an IPC-internal anchor, and one vertex per gripper.
SHIRT_OBJ="$FAST_OBJ" "$PROJECT_ROOT/run.sh" \
  --settle-frames 30 \
  --virtual-grasp \
  --grasp-mode soft \
  --prepin-first-grasp \
  --grasp-points 1 \
  --final-right-grasp-points 1 \
  --grasp-strength 10000 \
  "$@"
