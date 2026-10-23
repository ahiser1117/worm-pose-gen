#!/usr/bin/env python3
"""Check an installed checkout: the environment, the lab library, the data directory and the GPUs.

    scripts/project_env.sh uv run --no-sync --frozen python scripts/preflight.py

``scripts/bootstrap_environment.sh`` runs it last.  Each line is a check;
``FAIL`` lines are what stops the app from working and make the exit status
1, ``warn`` lines are what it works without (no GPU, the lab library not
reachable from this machine, little free space).  The last line is ``OK``
or ``FAILED``.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import sys

from worm_pose_gen import library
from worm_pose_gen.compute import detect_compute
from worm_pose_gen.library.roots import Libraries, short_hostname
from worm_pose_gen.settings import DATA_DIR, REPO_ROOT, SETTINGS_FILE

# Below this the data directory fills after a few analysed recordings (about 2 GB each).
LOW_SPACE_GB = 50


def check(failures: list[str], ok: bool, message: str, *, fatal: bool = True) -> None:
    print(f"{'ok  ' if ok else 'FAIL' if fatal else 'warn'}  {message}", flush=True)
    if not ok and fatal:
        failures.append(message)


def default_problem(libraries: Libraries, model: str) -> str:
    """What a default model lacks: its weights, or the pseudo-label sets its recipe trains with; empty when nothing."""

    try:
        library.weights_path(libraries, model)
        card = library.get_card(libraries, model)
    except LookupError:
        return "weights missing"
    missing = []
    for ref in card.hparams.get("pseudo_labels") or ():
        try:
            library.get_pseudo_labels(libraries, ref)
        except LookupError:
            missing.append(ref)
    return f"pseudo-labels of its recipe missing ({', '.join(missing)})" if missing else ""


def lab_defaults(libraries: Libraries) -> list[tuple[str, str, str]]:
    """``(setup, model, problem)`` for every default model of the lab setups (:func:`default_problem`)."""

    return [(setup.ref, model, default_problem(libraries, model))
            for setup in library.list_setups(libraries) for model in sorted(set(setup.defaults.values()))]


def main() -> int:
    failures: list[str] = []
    environment = (REPO_ROOT / ".venv").resolve()
    check(failures, sys.version_info[:2] == (3, 13), f"Python {platform.python_version()} (3.13 required)")
    check(failures, Path(sys.prefix).resolve() == environment, f"environment {Path(sys.prefix).resolve()} (expected {environment})")
    check(failures, os.environ.get("UV_PROJECT_ENVIRONMENT") == str(REPO_ROOT / ".venv"), "run through scripts/project_env.sh")

    host = short_hostname()
    libraries = Libraries.for_host()
    if libraries.lab is None:
        check(failures, False, f"no lab library is known for host {host} (flv-c* and Engaging are); only your own models", fatal=False)
    else:
        readable = libraries.lab.is_dir() and os.access(libraries.lab, os.R_OK | os.X_OK)
        check(failures, readable, f"lab library {libraries.lab}" + ("" if readable else " is not readable from here"), fatal=False)
        if readable:
            for setup, model, problem in lab_defaults(libraries):
                check(failures, not problem, f"{setup} default {model}" + (f": {problem}" if problem else ""))

    data = DATA_DIR
    problem = "" if data.is_dir() and os.access(data, os.W_OK) else " is not writable" if data.is_dir() else " does not exist (scripts/bootstrap_environment.sh makes it)"
    check(failures, not problem, f"data directory {data}{problem} ({'from ' + str(SETTINGS_FILE) if SETTINGS_FILE.exists() else 'default'})")
    if data.is_dir():
        free_gb = shutil.disk_usage(data).free / 2**30
        check(failures, free_gb >= LOW_SPACE_GB, f"{free_gb:.0f} GB free for the data directory (an analysed recording takes about 2 GB)", fatal=False)

    compute = detect_compute()
    gpus = ", ".join(f"{gpu.index}: {gpu.name}" for gpu in compute.gpus)
    check(failures, bool(compute.gpus), f"GPUs: {gpus}" if gpus else "no GPU on this machine: analysis and training need one (or SLURM)", fatal=False)
    slurm = compute.slurm
    print(f"info  SLURM {'available' if slurm.available else 'not available'}{'' if slurm.available else f' ({slurm.reason})'}", flush=True)

    print("FAILED" if failures else "OK", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
