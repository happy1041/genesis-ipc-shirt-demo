#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SIM1_ROOT:?Set SIM1_ROOT to a clean SIM1 checkout}"
: "${GENESIS_PYTHON:?Set GENESIS_PYTHON to the patched Genesis environment Python}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
SHIRT_OBJ="${SHIRT_OBJ:-$OUTPUT_ROOT/assets/short-shirt.obj}"
ROBOT_URDF="${ROBOT_URDF:-$OUTPUT_ROOT/assets/acone_ipc/acone_ipc.urdf}"

if [[ ! -f "$SHIRT_OBJ" ]]; then
  : "${SIM1_PYTHON:?Set SIM1_PYTHON to a Python environment containing pxr}"
  "$SIM1_PYTHON" "$PROJECT_ROOT/tools/assets/export_usdc_mesh.py" \
    "$SIM1_ROOT/assets/cloth/short-shirt.usdc" "$SHIRT_OBJ"
fi

if [[ ! -f "$ROBOT_URDF" ]]; then
  "$GENESIS_PYTHON" "$PROJECT_ROOT/tools/assets/build_acone_ipc_proxy_urdf.py" \
    --source "$SIM1_ROOT/assets/acone/acone.urdf" \
    --output-dir "$(dirname "$ROBOT_URDF")"
fi

printf 'shirt_obj=%s\nrobot_urdf=%s\n' "$SHIRT_OBJ" "$ROBOT_URDF"
