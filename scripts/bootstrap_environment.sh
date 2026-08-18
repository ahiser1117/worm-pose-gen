#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUN_IN_ENV="$PROJECT_ROOT/scripts/project_env.sh"
EXTERNAL_ROOT="/temp_data4/alex/external_artifacts"
REQUIRE_CUDA=0

if (( $# > 1 )); then
  printf 'usage: %s [--require-cuda]\n' "$0" >&2
  exit 64
fi
if (( $# == 1 )); then
  if [[ "$1" != "--require-cuda" ]]; then
    printf 'unknown argument: %s\n' "$1" >&2
    exit 64
  fi
  REQUIRE_CUDA=1
fi

mkdir -p \
  "$EXTERNAL_ROOT/artifacts/worm_pose_gen/checkpoints" \
  "$EXTERNAL_ROOT/artifacts/worm_pose_gen/profiler" \
  "$EXTERNAL_ROOT/datasets/worm_pose_gen" \
  "$EXTERNAL_ROOT/experiments/worm_pose_gen"

write_probe="$(mktemp "$EXTERNAL_ROOT/.worm-pose-write-test.XXXXXX")"
cleanup() {
  rm -f -- "$write_probe"
}
trap cleanup EXIT
printf 'worm-pose environment write test\n' >"$write_probe"
test -s "$write_probe"

if ! "$RUN_IN_ENV" uv python find --no-python-downloads 3.13; then
  "$RUN_IN_ENV" uv python install 3.13
fi
"$RUN_IN_ENV" uv python find --no-python-downloads 3.13
"$RUN_IN_ENV" uv sync --frozen --python 3.13

preflight_args=()
if (( REQUIRE_CUDA == 1 )); then
  preflight_args+=(--require-cuda)
fi
"$RUN_IN_ENV" uv run --no-sync --frozen python \
  "$PROJECT_ROOT/scripts/preflight.py" "${preflight_args[@]}"

printf 'Environment bootstrap completed successfully.\n'
