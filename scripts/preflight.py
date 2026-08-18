from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import h5py
import lightning
import matplotlib
import numpy
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the worm-pose runtime.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless exactly one CUDA device is visible.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    expected_environment = (project_root / ".venv").resolve()
    external_root = Path(
        os.environ.get(
            "WORM_POSE_EXTERNAL_ROOT", "/temp_data4/alex/external_artifacts"
        )
    )

    failures: list[str] = []
    if sys.version_info[:2] != (3, 13):
        failures.append(f"expected Python 3.13, found {platform.python_version()}")
    if Path(sys.prefix).resolve() != expected_environment:
        failures.append(
            f"expected environment {expected_environment}, found {Path(sys.prefix).resolve()}"
        )

    required_environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "MPLBACKEND": "Agg",
        "UV_PROJECT": str(project_root),
        "UV_PROJECT_ENVIRONMENT": str(project_root / ".venv"),
        "WORM_POSE_EXTERNAL_ROOT": "/temp_data4/alex/external_artifacts",
    }
    for name, expected in required_environment.items():
        actual = os.environ.get(name)
        if actual != expected:
            failures.append(f"{name}={actual!r}; expected {expected!r}")

    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    if args.require_cuda and not cuda_available:
        failures.append("CUDA is required but unavailable")
    if args.require_cuda and cuda_count != 1:
        failures.append(f"expected exactly one CUDA device, found {cuda_count}")

    cuda_details: dict[str, object] = {
        "available": cuda_available,
        "device_count": cuda_count,
        "runtime": torch.version.cuda,
    }
    if cuda_available and cuda_count:
        properties = torch.cuda.get_device_properties(0)
        cuda_details.update(
            {
                "logical_index": torch.cuda.current_device(),
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )

    report = {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "project_root": str(project_root),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "packages": {
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
            "lightning": lightning.__version__,
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "torch": torch.__version__,
        },
        "cuda": cuda_details,
        "paths": {
            "external_configured": str(external_root),
            "external_resolved": str(external_root.resolve()),
            "external_writable": os.access(external_root, os.W_OK),
            "matplotlib_config": matplotlib.get_configdir(),
            "uv_cache": os.environ.get("UV_CACHE_DIR"),
            "uv_project": os.environ.get("UV_PROJECT"),
            "uv_python_install": os.environ.get("UV_PYTHON_INSTALL_DIR"),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
