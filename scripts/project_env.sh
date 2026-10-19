#!/usr/bin/env bash

set -euo pipefail

export PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Keep the Python environment and disposable caches with the checkout. They are
# ignored by Git. What the user makes goes to the data directory instead
# (settings.json, src/worm_pose_gen/settings.py).
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

# Number GPUs as nvidia-smi does, so a GPU index means the same card to torch,
# to the app's job runner and to the person reading nvidia-smi.
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

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
