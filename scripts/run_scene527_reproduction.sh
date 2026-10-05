#!/usr/bin/env bash
set -euo pipefail
REPRO_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPRO_PYTHON="${GENESIS_PYTHON:-${GENESIS_ENV:+$GENESIS_ENV/bin/python}}"
REPRO_PYTHON="${REPRO_PYTHON:-python3}"
exec "$REPRO_PYTHON" "$REPRO_PROJECT_ROOT/tools/run_scene527_reproduction.py" "$@"
