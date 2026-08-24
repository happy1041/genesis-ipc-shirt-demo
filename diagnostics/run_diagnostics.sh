#!/usr/bin/env bash
set -euo pipefail

DIAGNOSTICS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTICS_ROOT/.." && pwd)"
: "${GENESIS_PYTHON:?Set GENESIS_PYTHON to the patched Genesis environment Python}"
: "${GENESIS_ROOT:?Set GENESIS_ROOT to the patched Genesis checkout}"
PYTHON_BIN="$GENESIS_PYTHON"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_JSON [MANUAL_REVIEW_JSON]" >&2
  exit 2
fi

run_json="$1"
manual_args=()
if [[ $# -ge 2 ]]; then
  manual_args=(--manual-review "$2")
fi

"$PYTHON_BIN" "$DIAGNOSTICS_ROOT/capture_provenance.py" "$run_json"
if ! "$PYTHON_BIN" "$DIAGNOSTICS_ROOT/analyze_semantic_states.py" "$run_json"; then
  echo "semantic analysis unavailable; evaluator will mark the evidence INCOMPLETE" >&2
fi
"$PYTHON_BIN" "$DIAGNOSTICS_ROOT/evaluate_run.py" "$run_json" "${manual_args[@]}"
"$PYTHON_BIN" "$DIAGNOSTICS_ROOT/index_runs.py" \
  "$OUTPUT_ROOT"
