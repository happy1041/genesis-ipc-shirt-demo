#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

failures=0
check_path() {
  local label="$1" path="${2:-}" kind="${3:-file}"
  if [[ -z "$path" ]]; then
    printf 'MISSING  %-18s not configured\n' "$label"
    failures=$((failures + 1))
  elif [[ "$kind" == dir && -d "$path" ]] || [[ "$kind" == file && -f "$path" ]]; then
    printf 'OK       %-18s %s\n' "$label" "$path"
  else
    printf 'MISSING  %-18s %s\n' "$label" "$path"
    failures=$((failures + 1))
  fi
}

GENESIS_PYTHON="${GENESIS_PYTHON:-${GENESIS_ENV:-}/bin/python}"
TRAJECTORY="${TRAJECTORY:-${EPISODE_ROOT:-}/episode_000000_sim1_replay.npz}"

check_path SIM1_ROOT "${SIM1_ROOT:-}" dir
check_path shirt_usdc "${SIM1_ROOT:-}/assets/cloth/short-shirt.usdc"
check_path acone_urdf "${SIM1_ROOT:-}/assets/acone/acone.urdf"
check_path trajectory "$TRAJECTORY"
check_path genesis_python "$GENESIS_PYTHON"
check_path sim1_python "${SIM1_PYTHON:-}"

if [[ $failures -ne 0 ]]; then
  printf '\nSetup is incomplete: %d required item(s) missing.\n' "$failures" >&2
  printf 'Copy .env.example to .env, edit the paths, and read docs/ASSETS_CN.md.\n' >&2
  exit 2
fi

if ! "$GENESIS_PYTHON" -c 'import genesis, numpy, trimesh' >/dev/null 2>&1; then
  printf '\nMISSING  Python packages     genesis, numpy, or trimesh cannot be imported\n' >&2
  exit 2
fi

printf '\nSetup check passed. Run: ./scripts/run_demo.sh\n'
