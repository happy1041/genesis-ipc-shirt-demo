#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
SOURCE_OBJ="$OUTPUT_ROOT/assets/short-shirt.obj"
FAST_FACES="${FAST_FACES:-2500}"
FAST_OBJ="${FAST_OBJ:-$OUTPUT_ROOT/assets/short-shirt-${FAST_FACES}f.obj}"

if [[ ! -f "$SOURCE_OBJ" ]]; then
  "$PROJECT_ROOT/run.sh" --frames 1 --no-record --settle-frames 0
fi
if [[ ! -f "$FAST_OBJ" ]]; then
  /home/happy1041/work/genesis-ipc-env/bin/python \
    "$PROJECT_ROOT/prepare_lowres_shirt.py" "$SOURCE_OBJ" "$FAST_OBJ" --faces "$FAST_FACES"
fi

SHIRT_OBJ="$FAST_OBJ" "$PROJECT_ROOT/run.sh" \
  --fast-preview \
  --settle-frames 30 \
  "$@"
