from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import h5py
import lightning
import matplotlib
import numpy
import torch


# Phase 0 is tied to physical GPU 0 on the approved node. These values are
# intentionally explicit rather than inferred from CUDA_VISIBLE_DEVICES.
EXPECTED_PHYSICAL_GPU_INDEX = 0
EXPECTED_PHYSICAL_GPU_UUID = "GPU-f72d2ba7-8334-183e-e368-2c527e8a39e6"
EXPECTED_PHYSICAL_GPU_PCI_BUS_ID = "00000000:01:00.0"


def parse_nvidia_smi_row(output: str) -> dict[str, object]:
    """Parse the single physical-GPU row requested by the preflight command."""
    rows = [
        row for row in csv.reader(io.StringIO(output), skipinitialspace=True) if row
    ]
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError(
            "expected one nvidia-smi row with 6 fields, "
            f"found {len(rows)} row(s): {rows!r}"
        )
    index, uuid, pci_bus_id, name, memory_total, driver_version = (
        value.strip() for value in rows[0]
    )
    try:
        physical_index = int(index)
    except ValueError as exc:
        raise ValueError(f"invalid physical GPU index: {index!r}") from exc
    return {
        "physical_index": physical_index,
        "uuid": uuid,
        "pci_bus_id": pci_bus_id,
        "name": name,
        "total_memory": memory_total,
        "driver_version": driver_version,
    }


def query_physical_gpu_zero() -> dict[str, object]:
    fields = "index,uuid,pci.bus_id,name,memory.total,driver_version"
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={EXPECTED_PHYSICAL_GPU_INDEX}",
            f"--query-gpu={fields}",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_row(result.stdout)


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
    expected_python_install = (project_root / ".uv-python").resolve()
    executable = Path(sys.executable)
    executable_resolved = executable.resolve()
    base_executable = Path(getattr(sys, "_base_executable", sys.executable))
    base_executable_resolved = base_executable.resolve()
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
    if not base_executable_resolved.is_relative_to(expected_python_install):
        failures.append(
            "expected base interpreter beneath "
            f"{expected_python_install}, found {base_executable_resolved}"
        )

    required_environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "MPLBACKEND": "Agg",
        "PROJECT_ROOT": str(project_root),
        "UV_PROJECT": str(project_root),
        "UV_PROJECT_ENVIRONMENT": str(project_root / ".venv"),
        "UV_PYTHON_INSTALL_DIR": str(project_root / ".uv-python"),
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
        "expected_physical": {
            "index": EXPECTED_PHYSICAL_GPU_INDEX,
            "uuid": EXPECTED_PHYSICAL_GPU_UUID,
            "pci_bus_id": EXPECTED_PHYSICAL_GPU_PCI_BUS_ID,
        },
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
        if args.require_cuda and torch.cuda.current_device() != 0:
            failures.append(
                f"expected visible logical CUDA index 0, found {torch.cuda.current_device()}"
            )

    if args.require_cuda:
        try:
            physical_gpu = query_physical_gpu_zero()
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
            failures.append(f"could not verify physical CUDA device 0: {exc}")
        else:
            cuda_details["physical_device"] = physical_gpu
            cuda_details["mapping"] = {
                "physical_index": physical_gpu["physical_index"],
                "visible_logical_index": 0,
            }
            if physical_gpu["physical_index"] != EXPECTED_PHYSICAL_GPU_INDEX:
                failures.append(
                    "nvidia-smi returned physical GPU index "
                    f"{physical_gpu['physical_index']}; expected {EXPECTED_PHYSICAL_GPU_INDEX}"
                )
            if physical_gpu["uuid"] != EXPECTED_PHYSICAL_GPU_UUID:
                failures.append(
                    f"physical GPU UUID {physical_gpu['uuid']!r}; "
                    f"expected {EXPECTED_PHYSICAL_GPU_UUID!r}"
                )
            if str(physical_gpu["pci_bus_id"]).lower() != (
                EXPECTED_PHYSICAL_GPU_PCI_BUS_ID.lower()
            ):
                failures.append(
                    f"physical GPU PCI identity {physical_gpu['pci_bus_id']!r}; "
                    f"expected {EXPECTED_PHYSICAL_GPU_PCI_BUS_ID!r}"
                )

    report = {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "project_root": str(project_root),
        "python": {
            "version": platform.python_version(),
            "executable": str(executable),
            "executable_resolved": str(executable_resolved),
            "base_executable": str(base_executable),
            "base_executable_resolved": str(base_executable_resolved),
            "prefix": sys.prefix,
            "prefix_resolved": str(Path(sys.prefix).resolve()),
            "base_prefix": sys.base_prefix,
            "base_prefix_resolved": str(Path(sys.base_prefix).resolve()),
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
