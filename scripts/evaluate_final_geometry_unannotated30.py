#!/usr/bin/env python3
"""Run the frozen A1--A6 geometry on 30 frames from readable raw recordings.

This is an annotation-free operational stress test.  It deliberately reports
coverage and geometry diagnostics only; it cannot report anatomical error or
replace evaluation against the 30 primary manual traces.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

import build_smooth_body_prior_experiment as smooth
from evaluate_final_geometry_primary30 import CALLOUT_INDICES, fit_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "final_algorithm_unannotated30"
DEFAULT_RECORDINGS = (
    Path(
        "/store1/shared/all_data_raw/prj_aversion/"
        "2024-01-31/2024-01-31-02.h5"
    ),
    Path(
        "/store1/shared/all_data_raw/prj_aversion/"
        "2023-08-22/2023-08-22-01.h5"
    ),
    Path(
        "/store1/shared/all_data_raw/prj_aversion/"
        "2023-06-23/2023-06-23-01.h5"
    ),
)
DATASET_PATH = "/img_nir"
FRAMES_PER_RECORDING = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        action="append",
        type=Path,
        dest="recordings",
        help="raw HDF5 recording; specify exactly three times",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    sample = np.asarray(list(values), dtype=np.float64)
    if not len(sample):
        return {"n": 0, "median": None, "mean": None, "p95": None}
    return {
        "n": int(len(sample)),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "p95": float(np.percentile(sample, 95)),
    }


def recording_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate each HDF5 and define ten uniformly spaced frame positions."""

    cases: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for recording_number, path in enumerate(paths):
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        with h5py.File(resolved, "r") as handle:
            dataset = handle[DATASET_PATH]
            if dataset.ndim != 3 or tuple(dataset.shape[1:]) != (732, 968):
                raise RuntimeError(f"unexpected image shape in {resolved}: {dataset.shape}")
            if dataset.dtype != np.dtype("uint8"):
                raise RuntimeError(f"unexpected image dtype in {resolved}: {dataset.dtype}")
            frame_indices = np.linspace(
                0, int(dataset.shape[0]) - 1, FRAMES_PER_RECORDING, dtype=np.int64
            )
            # A file only qualifies after bounded reads from all three regions.
            verification_indices = [
                int(frame_indices[0]),
                int(frame_indices[len(frame_indices) // 2]),
                int(frame_indices[-1]),
            ]
            verification = []
            for frame_index in verification_indices:
                frame = np.asarray(dataset[frame_index])
                verification.append(
                    {
                        "frame_index": frame_index,
                        "minimum": int(frame.min()),
                        "maximum": int(frame.max()),
                        "mean": float(frame.mean()),
                    }
                )
            frame_count = int(dataset.shape[0])
        provenance.append(
            {
                "input_path": str(path),
                "resolved_path": str(resolved),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "dataset_path": DATASET_PATH,
                "shape": [frame_count, 732, 968],
                "dtype": "uint8",
                "readability_verification": verification,
                "selected_frame_indices": [int(value) for value in frame_indices],
            }
        )
        for within_recording, frame_index in enumerate(frame_indices):
            sample_index = recording_number * FRAMES_PER_RECORDING + within_recording
            cases.append(
                {
                    # fit_case retains the historical key name, but this run has
                    # no annotations.  sample_index is the public interpretation.
                    "annotation_index": sample_index,
                    "sample_index": sample_index,
                    "sample_id": f"stress-{recording_number}-{int(frame_index)}",
                    "recording": path.stem,
                    "resolved_source_path": str(resolved),
                    "source_dataset_path": DATASET_PATH,
                    "source_size_bytes": int(stat.st_size),
                    "source_mtime_ns": int(stat.st_mtime_ns),
                    "frame_index": int(frame_index),
                    "image_height": 732,
                    "image_width": 968,
                    "selection_stratum": "uniform_time_10_per_recording",
                }
            )
    return cases, provenance


def fit_recording(
    sources: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any], dict[str, np.ndarray]]]:
    """Read and fit one recording serially with a single read-only handle."""

    first = sources[0]
    path = Path(first["resolved_source_path"])
    stat = path.stat()
    if stat.st_size != int(first["source_size_bytes"]):
        raise RuntimeError(f"source size changed: {path}")
    if stat.st_mtime_ns != int(first["source_mtime_ns"]):
        raise RuntimeError(f"source mtime changed: {path}")
    output: list[tuple[int, dict[str, Any], dict[str, np.ndarray]]] = []
    with h5py.File(path, "r") as handle:
        dataset = handle[str(first["source_dataset_path"])]
        for source in sources:
            index = int(source["sample_index"])
            frame = np.asarray(dataset[int(source["frame_index"])], dtype=np.uint8)
            result, arrays = fit_case((source, frame))
            result["sample_index"] = index
            result["annotation_index"] = None
            result["frame_read_source"] = "verified_raw_source"
            if index in CALLOUT_INDICES:
                arrays["frame"] = frame
            output.append((index, result, arrays))
            print(
                json.dumps(
                    {
                        "sample_index": index,
                        "recording": result["recording"],
                        "frame_index": result["frame_index"],
                        "accepted": result["accepted"],
                        "failure_stage": result["failure_stage"],
                    }
                ),
                flush=True,
            )
    return output


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [case for case in cases if case["accepted"]]
    failures = Counter(
        str(case["failure_stage"]) for case in cases if not case["accepted"]
    )
    radii = Counter(
        int(case["selected_repair"]["radius_px"])
        for case in cases
        if "selected_repair" in case
    )
    by_recording: dict[str, dict[str, int]] = {}
    for recording in sorted({str(case["recording"]) for case in cases}):
        members = [case for case in cases if case["recording"] == recording]
        by_recording[recording] = {
            "accepted": sum(bool(case["accepted"]) for case in members),
            "total": len(members),
        }
    return {
        "requested_frames": len(cases),
        "accepted_frames": len(accepted),
        "accepted_fraction": len(accepted) / len(cases),
        "stage_coverage": {
            "section3_component": sum("section3" in case for case in cases),
            "A1_geometry_selected": sum("selected_repair" in case for case in cases),
            "A5_modeled_body_passed": sum(
                bool((case.get("a5_body") or {}).get("accepted")) for case in cases
            ),
            "A6_endpoint_extension_and_length_gate_passed": len(accepted),
        },
        "failure_stage_counts": dict(sorted(failures.items())),
        "selected_radius_counts": {
            str(radius): count for radius, count in sorted(radii.items())
        },
        "coverage_by_recording": by_recording,
        "accepted_extension_gain_px": numeric_summary(
            case["a6_extension"]["a6_resampled_length_gain_px"]
            for case in accepted
        ),
        "accepted_a5_length_px": numeric_summary(
            case["a6_extension"]["a5_centerline_length_px"] for case in accepted
        ),
        "accepted_a6_length_px": numeric_summary(
            case["a6_extension"]["a6_centerline_length_px"] for case in accepted
        ),
        "runtime_seconds": numeric_summary(case["runtime_seconds"] for case in cases),
    }


def plot_summary(cases: list[dict[str, Any]], metrics: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    coverage = metrics["stage_coverage"]
    labels = ["requested", "section 3", "A1", "A5", "A6 final"]
    values = [
        len(cases),
        coverage["section3_component"],
        coverage["A1_geometry_selected"],
        coverage["A5_modeled_body_passed"],
        coverage["A6_endpoint_extension_and_length_gate_passed"],
    ]
    axes[0].bar(
        labels,
        values,
        color=[smooth.GRAY, smooth.CYAN, smooth.CYAN, smooth.ORANGE, smooth.GREEN],
    )
    axes[0].set_ylim(0, 32)
    axes[0].set_ylabel("frames")
    axes[0].set_title("Fail-closed stage coverage")
    axes[0].tick_params(axis="x", labelrotation=20)
    for position, value in enumerate(values):
        axes[0].text(position, value + 0.6, str(value), ha="center")

    recording_names = list(metrics["coverage_by_recording"])
    accepted_by_recording = [
        metrics["coverage_by_recording"][name]["accepted"] for name in recording_names
    ]
    axes[1].bar(recording_names, accepted_by_recording, color=smooth.GREEN)
    axes[1].set_ylim(0, 10.8)
    axes[1].set_ylabel("accepted of 10")
    axes[1].set_title("Final coverage by recording")
    axes[1].tick_params(axis="x", labelrotation=20)
    for position, value in enumerate(accepted_by_recording):
        axes[1].text(position, value + 0.25, str(value), ha="center")

    accepted = [case for case in cases if case["accepted"]]
    gain = [case["a6_extension"]["a6_resampled_length_gain_px"] for case in accepted]
    if gain:
        axes[2].hist(gain, bins=min(10, len(gain)), color=smooth.ORANGE, edgecolor="white")
    axes[2].set_xlabel("A6 length gain over A5 (px)")
    axes[2].set_ylabel("accepted frames")
    axes[2].set_title("Endpoint-extension magnitude")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.suptitle(
        f"Frozen A1--A6 unannotated stress test: {metrics['accepted_frames']}/30 final outputs"
    )
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_callouts(
    cases: list[dict[str, Any]], arrays: dict[int, dict[str, np.ndarray]], path: Path
) -> None:
    by_index = {int(case["sample_index"]): case for case in cases}
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3), constrained_layout=True)
    for axis, index in zip(axes, sorted(CALLOUT_INDICES), strict=True):
        case = by_index[index]
        data = arrays[index]
        frame = data["frame"]
        lower, upper = np.percentile(frame, [1, 99])
        axis.imshow(frame, cmap="gray", vmin=lower, vmax=upper)
        if "section3_component" in data:
            mask = data["section3_component"]
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[..., :3] = to_rgb(smooth.CYAN)
            rgba[..., 3] = mask.astype(np.float32) * 0.18
            axis.imshow(rgba, interpolation="nearest")
        if "a5_centerline_xy" in data:
            a5 = data["a5_centerline_xy"]
            a6 = data["a6_centerline_xy"]
            axis.plot(a5[:, 0], a5[:, 1], color="white", linewidth=2.1, label="A5")
            axis.plot(a6[:, 0], a6[:, 1], color=smooth.ORANGE, linewidth=2.3, label="A6")
            outcome = (
                f"accepted; A6 gain "
                f"{case['a6_extension']['a6_resampled_length_gain_px']:.2f} px"
            )
        else:
            outcome = f"rejected at {case['failure_stage']}"
        axis.set_title(
            f"stress-test position {index}: {case['recording']} frame {case['frame_index']}\n"
            f"{outcome}"
        )
        axis.set_xlim(0, frame.shape[1] - 1)
        axis.set_ylim(frame.shape[0] - 1, 0)
        axis.set_axis_off()
        if "a5_centerline_xy" in data:
            axis.legend(loc="lower right", framealpha=0.78)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    recordings = list(args.recordings or DEFAULT_RECORDINGS)
    if len(recordings) != 3:
        raise ValueError("exactly three recordings are required")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources, provenance = recording_records(recordings)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        grouped.setdefault(str(source["resolved_source_path"]), []).append(source)

    fitted: dict[int, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    if args.workers == 1:
        result_groups = [fit_recording(group) for group in grouped.values()]
    else:
        result_groups = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(grouped))) as executor:
            futures = [executor.submit(fit_recording, group) for group in grouped.values()]
            for future in as_completed(futures):
                result_groups.append(future.result())
    for group in result_groups:
        for index, result, case_arrays in group:
            fitted[index] = (result, case_arrays)

    cases = [fitted[index][0] for index in range(30)]
    arrays = {index: fitted[index][1] for index in range(30) if fitted[index][1]}
    summary = summarize(cases)
    payload = {
        "status": "frozen_A1_through_A6_unannotated_operational_stress_test",
        "evidence_boundary": {
            "manual_annotations_available": False,
            "anatomical_accuracy_claim": False,
            "substitute_for_primary30_annotation_audit": False,
            "protected_2025_holdout_opened": False,
            "parameters_frozen_from_annotation_index_5_frame_3420": True,
            "selection": "10 uniformly spaced frames from each of 3 readable recordings",
        },
        "inputs": provenance,
        "summary": summary,
        "per_case": cases,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip(),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    predictions: dict[str, np.ndarray] = {}
    for index, data in arrays.items():
        if "a6_centerline_xy" not in data:
            continue
        predictions[f"sample_{index:02d}_a5_centerline_xy"] = data[
            "a5_centerline_xy"
        ]
        predictions[f"sample_{index:02d}_a6_centerline_xy"] = data[
            "a6_centerline_xy"
        ]
    np.savez_compressed(args.output_dir / "predictions.npz", **predictions)
    plot_summary(cases, summary, args.output_dir / "summary.png")
    plot_callouts(cases, arrays, args.output_dir / "positions_2_22.png")
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
