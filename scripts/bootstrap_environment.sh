#!/usr/bin/env bash
# Install the project into this checkout and choose where your data goes.
#
#   scripts/bootstrap_environment.sh [--data-dir DIR]
#
# Installs Python 3.13 and the locked dependencies into the checkout (.uv-python,
# .venv), writes settings.json with the data directory (default: user_data/ in
# the checkout, or the one settings.json already names), and runs
# scripts/preflight.py. Run it again to update the environment after a pull, or
# with --data-dir to move where new data goes (existing data is not moved).

set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUN_IN_ENV="$PROJECT_ROOT/scripts/project_env.sh"
SETTINGS="$PROJECT_ROOT/settings.json"
DATA_DIR=""

usage() { printf 'usage: %s [--data-dir DIR]\n' "$0" >&2; exit 64; }
while (( $# > 0 )); do
  case "$1" in
    --data-dir) (( $# >= 2 )) || usage; DATA_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is not installed. Install it with\n\n  curl -LsSf https://astral.sh/uv/install.sh | sh\n\nthen open a new shell and run this script again.\n' >&2
  exit 1
fi

# The data directory: --data-dir, else the one settings.json names, else user_data/.
if [[ -z "$DATA_DIR" && -f "$SETTINGS" ]]; then
  DATA_DIR="$(sed -n 's/.*"data_dir": *"\([^"]*\)".*/\1/p' "$SETTINGS")"
fi
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/user_data}"
mkdir -p -- "$DATA_DIR"
DATA_DIR="$(CDPATH= cd -- "$DATA_DIR" && pwd -P)"
if [[ ! -w "$DATA_DIR" ]]; then
  printf 'cannot write to the data directory %s; pass --data-dir with a folder you own\n' "$DATA_DIR" >&2
  exit 1
fi
printf '{\n "data_dir": "%s"\n}\n' "$DATA_DIR" >"$SETTINGS"
printf 'Data directory: %s (in %s)\n' "$DATA_DIR" "$SETTINGS"

# UV_PYTHON_INSTALL_DIR is set by project_env.sh, so this installs into the
# checkout rather than uv's user-wide interpreter directory. Do not accept an
# otherwise compatible interpreter found through an existing .venv symlink.
"$RUN_IN_ENV" uv python install 3.13
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
# The downloaded wheels are copies of what is now in .venv (about 5 GB); a later
# sync downloads them again.
"$RUN_IN_ENV" uv cache prune --ci

"$RUN_IN_ENV" uv run --no-sync --frozen python "$PROJECT_ROOT/scripts/preflight.py"
printf '\nInstalled. Start the app with ./worm-pose-app\n'
