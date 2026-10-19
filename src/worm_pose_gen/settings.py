"""Where this checkout keeps the user's data.

One directory holds everything a user makes, so a new user has one place to
choose and nothing else to configure::

    <data dir>/
        library/       the personal library: labels, datasets, models, setups (:mod:`.library`)
        workspaces/    one workspace per analysed recording, the job records and Labeling queues
        cache/         flat fields and recording priors, rebuilt when missing
        runs/          outputs of the command-line scripts

``settings.json`` at the root of the checkout (ignored by Git, written by
``scripts/bootstrap_environment.sh``) names the data directory as
``{"data_dir": "/path"}``; without it the data directory is ``user_data/``
inside the checkout.  The file is read once, at import, so a change takes
effect when the app or a script starts.

What the lab shares (models, published labels, setups) is not here: the lab
library is read-only and found by host name (:mod:`.library.roots`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = REPO_ROOT / "settings.json"
DEFAULT_DATA_DIR = REPO_ROOT / "user_data"


def read_settings(path: Path = SETTINGS_FILE) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def write_settings(values: dict[str, Any], path: Path = SETTINGS_FILE) -> None:
    Path(path).write_text(json.dumps(values, indent=1, sort_keys=True) + "\n")


def data_dir(path: Path = SETTINGS_FILE) -> Path:
    value = read_settings(path).get("data_dir")
    return Path(value).expanduser() if value else DEFAULT_DATA_DIR


DATA_DIR = data_dir()
LIBRARY_DIR = DATA_DIR / "library"
WORKSPACES_DIR = DATA_DIR / "workspaces"
CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = DATA_DIR / "runs"
