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
warnings=0
ok() { printf 'OK       %-22s %s\n' "$1" "$2"; }
fail() { printf 'MISSING  %-22s %s\n' "$1" "$2" >&2; failures=$((failures + 1)); }
warn() { printf 'WARNING  %-22s %s\n' "$1" "$2" >&2; warnings=$((warnings + 1)); }

check_path() {
  local label="$1" path="${2:-}" kind="${3:-file}"
  if [[ -z "$path" ]]; then
    fail "$label" "not configured"
  elif [[ "$kind" == dir && -d "$path" ]] || [[ "$kind" == file && -f "$path" ]]; then
    ok "$label" "$path"
  else
    fail "$label" "$path"
  fi
}

GENESIS_PYTHON="${GENESIS_PYTHON:-${GENESIS_ENV:-}/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
SHIRT_OBJ="${SHIRT_OBJ:-$OUTPUT_ROOT/assets/short-shirt.obj}"
TRAJECTORY="${TRAJECTORY:-${EPISODE_ROOT:-}/episode_000000_sim1_replay.npz}"

check_path SIM1_ROOT "${SIM1_ROOT:-}" dir
check_path GENESIS_ROOT "${GENESIS_ROOT:-}" dir
check_path trajectory "$TRAJECTORY"
check_path genesis_python "$GENESIS_PYTHON"

if [[ -x "$GENESIS_PYTHON" && -d "${GENESIS_ROOT:-}" ]]; then
  if report="$(GENESIS_ROOT="$GENESIS_ROOT" "$GENESIS_PYTHON" - <<'PY' 2>&1
import importlib.metadata as md
import os
from pathlib import Path
import sys

import genesis
import numpy
import torch
import trimesh
import uipc

root = Path(os.environ["GENESIS_ROOT"]).resolve()
loaded = Path(genesis.__file__).resolve()
try:
    loaded.relative_to(root)
except ValueError:
    raise SystemExit(f"Genesis imported from {loaded}, outside GENESIS_ROOT={root}")

uipc_version = md.version("pyuipc")
if uipc_version != "0.0.25":
    raise SystemExit(f"pyuipc must be 0.0.25 for the verified baseline, got {uipc_version}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")

major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
arches = torch.cuda.get_arch_list()
# PyTorch wheels may omit the device's exact minor architecture while still
# shipping compatible cubins/PTX (for example sm_86 code running on sm_89).
# A real CUDA kernel is a stronger compatibility check than string equality.
probe = torch.arange(1, 1025, device="cuda", dtype=torch.float32)
probe_result = (probe * probe).sum()
torch.cuda.synchronize()
if not torch.isfinite(probe_result):
    raise SystemExit(f"CUDA compatibility probe returned {probe_result.item()}")

print(f"python={sys.version.split()[0]}")
print(f"genesis={loaded}")
print(f"pyuipc={uipc_version}")
print(f"torch={torch.__version__} runtime_cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)} capability={major}.{minor}")
print(f"torch_arches={','.join(arches)}")
print(f"cuda_probe={probe_result.item():.1f} exact_arch_listed={arch in arches}")
PY
)"; then
    ok runtime "$report"
    if ! grep -q '^python=3\.12\.' <<<"$report"; then
      warn python_version "verified with Python 3.12.3"
    fi
    if ! grep -q '^torch=2\.8\.0+cu128 ' <<<"$report"; then
      warn torch_version "different from the recorded RTX4090 torch 2.8.0+cu128 environment; rerun GPU validation"
    fi
  else
    fail runtime "$report"
  fi
fi

expected_genesis_commit=8b1dba2fc99d0eff9ab7cc4b4bbc87685688fa44
if [[ -d "${GENESIS_ROOT:-}/.git" ]]; then
  actual_commit="$(git -C "$GENESIS_ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$actual_commit" == "$expected_genesis_commit" ]]; then
    ok genesis_commit "$actual_commit"
  else
    fail genesis_commit "expected $expected_genesis_commit, got ${actual_commit:-unknown}"
  fi
else
  warn genesis_git "source snapshot has no .git; verifying patched files by SHA256"
fi

verify_hash() {
  local label="$1" path="$2" expected="$3"
  if [[ ! -f "$path" ]]; then fail "$label" "$path"; return; fi
  local actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" == "$expected" ]]; then
    ok "$label" "$actual"
  else
    fail "$label" "SHA256 expected=$expected actual=$actual path=$path"
  fi
}

if [[ -d "${GENESIS_ROOT:-}" ]]; then
  verify_hash genesis_coupler "$GENESIS_ROOT/genesis/engine/couplers/ipc_coupler/coupler.py" c403be43580383ccd018df1386984ea5ebb2683511431711747856407012065b
  verify_hash genesis_fem_entity "$GENESIS_ROOT/genesis/engine/entities/fem_entity.py" 3f130b06605f3f2d611733ac0f228349ee56b37139ad4653e0195833adaec64a
  verify_hash genesis_cloth_material "$GENESIS_ROOT/genesis/engine/materials/FEM/cloth.py" 091ff861144cb6ba509f9620bae7a549d726274dbc66c1d073217dd9380541ee
  verify_hash genesis_fem_solver "$GENESIS_ROOT/genesis/engine/solvers/fem_solver.py" 6419127645bd414201bd0c9eae12955fe2148ddf6cce0778f335f848a8fb4915
  verify_hash genesis_scene "$GENESIS_ROOT/genesis/engine/scene.py" 493f948fe9283be97fc13ec70622c5be53d47cf4e221c43482229ecb306dc759
  verify_hash genesis_solvers "$GENESIS_ROOT/genesis/options/solvers.py" b78a3bc31e6ab770d3e02dffb1b8e105f02aca4b2fef05a716775d618d35402e
  verify_hash genesis_visualizer "$GENESIS_ROOT/genesis/vis/visualizer.py" 2ace526f3ccb499557a77f381b17aff18cbf842e977eddc207e38c8180028a54
fi
verify_hash demo_patch "$PROJECT_ROOT/patches/genesis-world-8b1dba2-current.patch" 355102f4710081590a080fd0f94bbc64cab4b1adf7324439df70a5a8d77793ae

manifest="$PROJECT_ROOT/configs/verified_inputs.sha256"
if [[ -f "$manifest" && -d "${SIM1_ROOT:-}" ]]; then
  while read -r expected logical; do
    [[ -z "$expected" || "$expected" == \#* ]] && continue
    case "$logical" in
      SIM1/*) path="$SIM1_ROOT/${logical#SIM1/}" ;;
      EPISODE/*) path="$TRAJECTORY" ;;
      *) fail input_manifest "unknown logical path $logical"; continue ;;
    esac
    verify_hash "input:${logical##*/}" "$path" "$expected"
  done < "$manifest"
else
  warn input_manifest "optional legacy input hash manifest not present; configured input paths are checked but their hashes are not pinned"
fi

if [[ -f "$SHIRT_OBJ" ]]; then
  ok derived_shirt_obj "SHA256=$(sha256sum "$SHIRT_OBJ" | cut -d' ' -f1); verify that its topology matches the selected trajectory"
else
  check_path SIM1_PYTHON "${SIM1_PYTHON:-}"
  if [[ -x "${SIM1_PYTHON:-}" ]]; then
    if "$SIM1_PYTHON" -c 'from pxr import Usd, UsdGeom' >/dev/null 2>&1; then
      ok pxr "available in $SIM1_PYTHON"
    else
      fail pxr "cannot import pxr from $SIM1_PYTHON"
    fi
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  ok nvidia_smi "$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | head -1)"
else
  warn nvidia_smi "not found"
fi
if ! command -v ffmpeg >/dev/null 2>&1 && ! "$GENESIS_PYTHON" -c 'import imageio_ffmpeg' >/dev/null 2>&1; then
  warn video_encoder "neither system ffmpeg nor imageio-ffmpeg is available"
fi

printf '\nSummary: failures=%d warnings=%d\n' "$failures" "$warnings"
if [[ $failures -ne 0 ]]; then
  printf 'Setup does not match the verified baseline.\n' >&2
  exit 2
fi
printf 'Setup matches required baseline inputs. Warnings require a GPU smoke test.\n'
