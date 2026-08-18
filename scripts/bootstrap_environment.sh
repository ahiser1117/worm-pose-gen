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

# UV_PYTHON_INSTALL_DIR is set by project_env.sh, so this installs into the
# checkout rather than uv's user-wide interpreter directory. Do not accept an
# otherwise compatible interpreter found through an existing .venv symlink.
"$RUN_IN_ENV" uv python install 3.13
"$RUN_IN_ENV" uv python pin 3.13
managed_python="$(
  "$RUN_IN_ENV" uv python find \
    --no-project --managed-python --no-python-downloads 3.13
)"
managed_python="$(realpath -- "$managed_python")"
case "$managed_python" in
  "$PROJECT_ROOT/.uv-python"/*) ;;
  *)
    printf 'uv selected interpreter outside %s: %s\n' \
      "$PROJECT_ROOT/.uv-python" "$managed_python" >&2
    exit 1
    ;;
esac

venv_python="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$venv_python" ]] || \
   [[ "$(realpath -- "$venv_python")" != "$PROJECT_ROOT/.uv-python/"* ]]; then
  if [[ -L "$PROJECT_ROOT/.venv" ]]; then
    printf 'refusing to clear symbolic-link environment path: %s\n' \
      "$PROJECT_ROOT/.venv" >&2
    exit 1
  fi
  "$RUN_IN_ENV" uv venv --clear --python "$managed_python" "$PROJECT_ROOT/.venv"
fi
"$RUN_IN_ENV" uv sync --frozen --python "$managed_python"

preflight_args=()
if (( REQUIRE_CUDA == 1 )); then
  preflight_args+=(--require-cuda)
fi
"$RUN_IN_ENV" uv run --no-sync --frozen python \
  "$PROJECT_ROOT/scripts/preflight.py" "${preflight_args[@]}"

printf 'Environment bootstrap completed successfully.\n'
