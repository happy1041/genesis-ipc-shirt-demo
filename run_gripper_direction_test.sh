#!/usr/bin/env bash
set -euo pipefail

GENESIS_ENV="${GENESIS_ENV:-/home/happy1041/work/genesis-ipc-env}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${GENESIS_GPU:-GPU-573ea767-dda1-d1a3-84e3-aa0b5697120c}"
NVIDIA_WHEEL_ROOT="$GENESIS_ENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_WHEEL_ROOT/cublas/lib:$NVIDIA_WHEEL_ROOT/cusparse/lib:$NVIDIA_WHEEL_ROOT/cusolver/lib:$NVIDIA_WHEEL_ROOT/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

"$GENESIS_ENV/bin/python" "$PROJECT_ROOT/test_gripper_direction.py" "$@"
