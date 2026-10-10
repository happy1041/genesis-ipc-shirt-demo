#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
args=()
if [[ -n "${TRAJECTORY_WORKBENCH_CONFIG:-}" ]]; then
  args+=(--config "$TRAJECTORY_WORKBENCH_CONFIG")
fi
if [[ -n "${TRAJECTORY_IK_OUTPUT_DIR:-}" ]]; then
  args+=(--output-dir "$TRAJECTORY_IK_OUTPUT_DIR")
fi
exec "${GENESIS_PYTHON:-python}" "$PROJECT_ROOT/tools/trajectory_workbench/ik_preflight.py" "${args[@]}" "$@"
