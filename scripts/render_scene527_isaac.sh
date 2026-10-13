#!/usr/bin/env bash
# Prepare an offline Omniverse job; pass --run to render it as well.
set -euo pipefail
RENDER_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RENDER_PYTHON="${GENESIS_PYTHON:-${GENESIS_ENV:+$GENESIS_ENV/bin/python}}"
RENDER_PYTHON="${RENDER_PYTHON:-python3}"
exec "$RENDER_PYTHON" "$RENDER_PROJECT_ROOT/tools/prepare_scene527_isaac.py" "$@"
