#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${GENESIS_PYTHON:-python}" "$PROJECT_ROOT/tools/trajectory_workbench/ik_preflight.py" "$@"
