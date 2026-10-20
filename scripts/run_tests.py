#!/usr/bin/env python3
"""Run the unit tests on the CPU, one process per test class, in parallel.

Each class runs in its own interpreter with this checkout's ``src`` first on
``PYTHONPATH`` (so a worktree's copy of this script tests the worktree's code),
no CUDA device, and ``--threads`` OpenMP/MKL threads: torch otherwise starts
128 threads per process on the flv machines and parallel runs slow each other
down several times over.  Classes start slowest first, by the times of earlier
runs (kept in ``$XDG_CACHE_HOME/worm_pose_gen_test_times.json``).  ``--fast``
skips the tests marked ``@slow`` (``tests/slow.py``).

Examples:

    scripts/project_env.sh uv run --no-sync --frozen python scripts/run_tests.py --fast
    scripts/project_env.sh uv run --no-sync --frozen python scripts/run_tests.py tests.test_pipeline tests.test_fixes
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMES_PATH = Path(os.environ.get("XDG_CACHE_HOME", PROJECT_ROOT / ".cache")) / "worm_pose_gen_test_times.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modules", nargs="*", help="test modules, e.g. tests.test_pipeline (default: every tests/test_*.py)")
    parser.add_argument("--fast", action="store_true", help="skip the tests marked @slow")
    parser.add_argument("-j", "--jobs", type=int, default=12, help="classes run at once")
    parser.add_argument("--threads", type=int, default=4, help="OpenMP/MKL threads per process")
    return parser.parse_args()


def test_classes(module: str) -> list[str]:
    """``module.Class`` for every class in the file with ``test_`` methods of its own or from a base in the file."""

    tree = ast.parse((PROJECT_ROOT / (module.replace(".", "/") + ".py")).read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def has_tests(name: str) -> bool:
        node = classes[name]
        if any(isinstance(item, ast.FunctionDef) and item.name.startswith("test") for item in node.body):
            return True
        return any(isinstance(base, ast.Name) and base.id in classes and has_tests(base.id) for base in node.bases)

    return [f"{module}.{name}" for name in classes if has_tests(name)]


def run(unit: str, env: dict[str, str]) -> tuple[str, float, int, str]:
    start = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", "unittest", unit], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    return unit, time.perf_counter() - start, result.returncode, result.stderr


def main() -> int:
    args = parse_args()
    modules = args.modules or sorted(f"tests.{path.stem}" for path in (PROJECT_ROOT / "tests").glob("test_*.py"))
    units = [unit for module in modules for unit in test_classes(module)]
    times = json.loads(TIMES_PATH.read_text()) if TIMES_PATH.exists() else {}
    mode = "fast" if args.fast else "full"
    units.sort(key=lambda unit: -times.get(mode, {}).get(unit, 0.0))
    env = {
        **os.environ,
        "PYTHONPATH": f"{PROJECT_ROOT / 'src'}{os.pathsep}{PROJECT_ROOT}",
        "CUDA_VISIBLE_DEVICES": "",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "WORM_POSE_FAST_TESTS": "1" if args.fast else "0",
    }

    start = time.perf_counter()
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for future in as_completed([pool.submit(run, unit, env) for unit in units]):
            unit, seconds, code, output = future.result()
            times.setdefault(mode, {})[unit] = seconds
            summary = output.strip().splitlines()[-1] if output.strip() else ""
            print(f"{seconds:7.1f}s  {'ok  ' if code == 0 else 'FAIL'}  {unit}  {summary}", flush=True)
            if code != 0:
                failed.append((unit, output))

    TIMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = TIMES_PATH.with_suffix(f".{os.getpid()}.tmp")
    partial.write_text(json.dumps(times, indent=1, sort_keys=True))
    partial.replace(TIMES_PATH)

    for unit, output in failed:
        print(f"\n{'=' * 70}\n{unit}\n{'=' * 70}\n{output}")
    print(f"\n{len(units) - len(failed)}/{len(units)} classes passed ({mode}) in {time.perf_counter() - start:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
