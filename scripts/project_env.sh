#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Keep the Python environment and disposable caches with the checkout. They are
# ignored by Git. Reserve external_artifacts for checkpoints, generated data,
# profiler traces, and experiment outputs that can accumulate.
export WORM_POSE_EXTERNAL_ROOT="/temp_data4/alex/external_artifacts"
export UV_PROJECT="$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.uv-python"
export UV_PYTHON="3.13"

# Keep libraries from trying to write beneath the read-only home directory.
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export MPLBACKEND="Agg"

# Physical CUDA device 0 is the only GPU this project may expose. It becomes
# logical cuda:0 inside PyTorch.
export CUDA_VISIBLE_DEVICES="0"

mkdir -p \
  "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$MPLCONFIGDIR" \
  "$TORCH_HOME"

if (( $# == 0 )); then
  printf 'usage: %s COMMAND [ARG ...]\n' "$0" >&2
  exit 64
fi

exec "$@"
