#!/usr/bin/env python3
"""Generate conservative classical proxy labels from development records only."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.classical import ClassicalConfig, ClassicalResult, extract_centerline
from worm_pose_gen.data import HDF5FrameSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL = Path("/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/proxy_v1")
EXPECTED_DEVELOPMENT = ("2023-09-19-01", "2023-09-27-01", "2023-10-11-01")
FORBIDDEN_RECORD = "2025-03-06-01"


def _uniform_indices(frame_range: list[int], count: int) -> np.ndarray:
    start, stop = map(int, frame_range)
    if stop <= start:
        raise ValueError(f"empty frame range {frame_range}")
    return np.linspace(start, stop - 1, min(count, stop - start), dtype=np.int64)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot_overlay(ax: Any, image: np.ndarray, result: ClassicalResult, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    if result.centerline_xy is not None:
        line = result.centerline_xy
        color = "#36f1cd" if result.accepted else "#ffb000"
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.3)
        ax.scatter(line[[0, -1], 0], line[[0, -1], 1], c=["#ff3b6b", "#3b82ff"], s=9)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def _montage(cases: list[dict[str, Any]], path: Path, *, columns: int = 6) -> None:
    rows = max(1, math.ceil(len(cases) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.35 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, case in zip(axes.flat, cases):
        _plot_overlay(ax, case["image"], case["result"], case["title"])
    figure.tight_layout(pad=0.5)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _angle_profiles(cases: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey=True)
    for ax, case in zip(axes.flat, cases):
        angle = np.unwrap(case["result"].tangent_angle)
        ax.plot(np.linspace(0, 1, len(angle)), np.degrees(angle), linewidth=1.3)
        ax.set_title(case["title"], fontsize=7)
        ax.grid(alpha=0.25)
    figure.supxlabel("normalized body coordinate (export order; anatomical orientation uncertain)")
    figure.supylabel("unwrapped image-coordinate tangent angle (degrees)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_hdf5(
    path: Path, records: list[dict[str, Any]], config: ClassicalConfig, manifest_path: Path
) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite existing project output: {path} or {partial}")
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(partial, "x") as output:
        output.attrs["schema_version"] = 1
        output.attrs["evidence_tier"] = "Tier B candidate proxy labels; not ground truth"
        output.attrs["source_dataset_path"] = "/img_nir"
        output.attrs["split_manifest"] = str(manifest_path)
        output.attrs["split_manifest_sha256"] = _sha256(manifest_path)
        output.attrs["classical_config_json"] = json.dumps(asdict(config), sort_keys=True)
        output.attrs["orientation_note"] = "static endpoint appearance heuristic; unvalidated and capped at 0.65"
        output.attrs["geometry_convention"] = "xy pixels; x right, y down; angles atan2(dy,dx)"
        for record in records:
            group = output.create_group(record["record_id"])
            group.attrs["configured_source_path"] = record["configured_path"]
            group.attrs["resolved_source_path"] = record["resolved_path"]
            group.attrs["source_size_bytes"] = record["size_bytes"]
            group.attrs["source_mtime_ns"] = record["mtime_ns"]
            indices = record["indices"]
            results = record["results"]
            group.create_dataset("sample_frame_index", data=indices)
            group.create_dataset("accepted", data=np.asarray([item.accepted for item in results], dtype=bool))
            group.create_dataset("quality_score", data=np.asarray([item.quality_score for item in results], dtype=np.float32))
            group.create_dataset("head_tail_probability", data=np.asarray([item.head_tail_probability for item in results], dtype=np.float32))
            group.create_dataset("rejection_reasons", data=np.asarray([
                ";".join(item.rejection_reasons) for item in results
            ], dtype=object), dtype=string_dtype)
            centerlines = np.full((len(results), config.n_points, 2), np.nan, dtype=np.float32)
            angles = np.full((len(results), config.n_points), np.nan, dtype=np.float32)
            for index, result in enumerate(results):
                if result.accepted:
                    centerlines[index] = result.centerline_xy
                    angles[index] = result.tangent_angle
            group.create_dataset("centerline_xy", data=centerlines)
            group.create_dataset("tangent_angle", data=angles)
            qc_keys = sorted({key for result in results for key in result.qc})
            qc_group = group.create_group("qc")
            for key in qc_keys:
                values = [result.qc.get(key, np.nan) for result in results]
                qc_group.create_dataset(key, data=np.asarray(values))
            accepted_positions = np.flatnonzero([item.accepted for item in results])
            group.create_dataset("accepted_sample_position", data=accepted_positions.astype(np.int64))
            group.create_dataset("accepted_frame_index", data=indices[accepted_positions])
            if len(accepted_positions):
                images = np.stack([record["images"][int(index)] for index in accepted_positions])
                group.create_dataset("accepted_image", data=images, chunks=(1, *images.shape[1:]),
                                     compression="gzip", compression_opts=4, shuffle=True)
        output.attrs["complete"] = True
        output.flush()
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs/split_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--experiment-dir", type=Path, default=PROJECT_ROOT / "experiments/exp_0001_classical_proxy")
    parser.add_argument("--samples-per-recording", type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.samples_per_recording <= 48:
        raise ValueError("samples-per-recording must be in [1, 48]")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("this experiment requires split manifest schema version 2")
    development = tuple(manifest.get("development_records", ()))
    if development != EXPECTED_DEVELOPMENT or FORBIDDEN_RECORD in development:
        raise ValueError(f"unexpected development records: {development}")
    if manifest.get("audited_holdout", {}).get("record") != FORBIDDEN_RECORD:
        raise ValueError("audited holdout declaration is missing or changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    config = ClassicalConfig()
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for record_id in development:
        metadata = manifest["records"][record_id]
        configured = PROJECT_ROOT / metadata["configured_path"]
        if configured.name == FORBIDDEN_RECORD + ".h5":
            raise RuntimeError("refusing to open audited holdout")
        stat = configured.resolve(strict=True).stat()
        if stat.st_size != metadata["size_bytes"] or stat.st_mtime_ns != metadata["mtime_ns"]:
            raise RuntimeError(f"source identity mismatch for {record_id}")
        indices = _uniform_indices(metadata["frame_range"], args.samples_per_recording)
        images: list[np.ndarray] = []
        results: list[ClassicalResult] = []
        record_started = time.perf_counter()
        # Exactly one read-only HDF5 source is open, and frames are individual reads.
        with HDF5FrameSource(configured, manifest["source_dataset_path"],
                             expected_frame_shape=(732, 968), allowed_dtypes=["uint8"],
                             expected_ndim=3, max_frames_per_read=1) as source:
            if len(source) != metadata["frame_count"]:
                raise RuntimeError(f"frame-count mismatch for {record_id}")
            for frame_index in indices:
                image = source.read_frame(int(frame_index))
                images.append(image)
                results.append(extract_centerline(image, config))
        records.append({
            "record_id": record_id,
            "configured_path": metadata["configured_path"],
            "resolved_path": metadata["resolved_path"],
            "size_bytes": metadata["size_bytes"],
            "mtime_ns": metadata["mtime_ns"],
            "indices": indices,
            "images": images,
            "results": results,
            "runtime_seconds": time.perf_counter() - record_started,
        })

    accepted_cases: list[dict[str, Any]] = []
    rejected_cases: list[dict[str, Any]] = []
    per_recording: dict[str, Any] = {}
    for record in records:
        accepted = sum(result.accepted for result in record["results"])
        rejection_counts = Counter(reason for result in record["results"] for reason in result.rejection_reasons)
        per_recording[record["record_id"]] = {
            "sampled": len(record["results"]),
            "accepted": accepted,
            "yield": accepted / len(record["results"]),
            "yield_wilson_95": _wilson(accepted, len(record["results"])),
            "rejection_reason_counts_nonexclusive": dict(sorted(rejection_counts.items())),
            "runtime_seconds": record["runtime_seconds"],
        }
        for position, (frame_index, image, result) in enumerate(zip(record["indices"], record["images"], record["results"])):
            case = {"record": record["record_id"], "frame": int(frame_index), "position": position,
                    "image": image, "result": result,
                    "title": f'{record["record_id"]} f={int(frame_index)} q={result.quality_score:.2f}'}
            (accepted_cases if result.accepted else rejected_cases).append(case)

    if len(accepted_cases) < 24:
        raise RuntimeError(f"only {len(accepted_cases)} accepted cases; cannot perform frozen 24-case audit")
    rng = np.random.default_rng(int(manifest["seed"]))
    random_cases = [accepted_cases[index] for index in sorted(rng.choice(len(accepted_cases), 24, replace=False))]
    worst_cases = []
    for record_id in development:
        candidates = [case for case in accepted_cases if case["record"] == record_id]
        minimum = min(case["result"].quality_score for case in candidates)
        worst_cases.extend(case for case in candidates if case["result"].quality_score == minimum)
    rejection_selection = rejected_cases if len(rejected_cases) <= 24 else [
        rejected_cases[index] for index in np.linspace(0, len(rejected_cases) - 1, 24, dtype=int)
    ]
    _montage(random_cases, figures_dir / "random_accepted_overlays.png")
    _montage(worst_cases, figures_dir / "worst_quality_overlays.png", columns=3)
    _montage(rejection_selection, figures_dir / "rejected_cases.png")
    _angle_profiles(random_cases[:12], figures_dir / "angle_profiles.png")

    output_h5 = args.output_dir / "proxy_labels.h5"
    _write_hdf5(output_h5, records, config, args.manifest)
    elapsed = time.perf_counter() - started
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                text=True, capture_output=True, check=True).stdout.strip()
    metrics = {
        "experiment": "EXP-0001",
        "status": "completed_pending_visual_audit",
        "evidence_tier": "Tier B candidate proxies; no manual ground truth",
        "seed": int(manifest["seed"]),
        "development_records": list(development),
        "audited_holdout_opened": False,
        "samples_per_recording": args.samples_per_recording,
        "config": asdict(config),
        "per_recording": per_recording,
        "total_sampled": len(accepted_cases) + len(rejected_cases),
        "total_accepted": len(accepted_cases),
        "internal_qc_note": "foreground-tube and dark-ridge support share extractor preprocessing and are correlated internal QC",
        "visual_overlay_audit": {"sample_count": 24, "grossly_off_midline": None, "review_status": "pending human review"},
        "worst_quality_cases": [{"record": case["record"], "frame": case["frame"],
                                 "quality_score": case["result"].quality_score} for case in worst_cases],
        "runtime_seconds": elapsed,
        "output_hdf5": str(output_h5),
        "output_hdf5_bytes": output_h5.stat().st_size,
        "output_hdf5_sha256": _sha256(output_h5),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "h5py": h5py.__version__, "git_commit": git_commit, "device": "CPU"},
    }
    (args.experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    stdout_lines = [
        f"EXP-0001 completed in {elapsed:.3f} s on CPU",
        f"development records only: {', '.join(development)}",
        "audited holdout opened: false",
    ]
    for record_id, values in per_recording.items():
        stdout_lines.append(f"{record_id}: {values['accepted']}/{values['sampled']} accepted ({values['yield']:.3f}), "
                            f"Wilson95={values['yield_wilson_95']}, rejections={values['rejection_reason_counts_nonexclusive']}")
    stdout_lines.append(f"proxy output: {output_h5} ({output_h5.stat().st_size} bytes, sha256={metrics['output_hdf5_sha256']})")
    (args.experiment_dir / "stdout.log").write_text("\n".join(stdout_lines) + "\n")
    print("\n".join(stdout_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
