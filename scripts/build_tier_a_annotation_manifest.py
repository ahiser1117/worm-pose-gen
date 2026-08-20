#!/usr/bin/env python3
"""Build the blind, development-only EXP-001 annotation selection manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.annotation_selection import (
    SELECTION_SCHEMA_VERSION,
    canonical_json_sha256,
    proxy_rows,
    select_recording_frames,
)
from worm_pose_gen.data import HDF5FrameSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "configs/split_manifest.json")
    parser.add_argument(
        "--proxy-hdf5",
        type=Path,
        default=Path("/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/proxy_v1/proxy_labels.h5"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "experiments/scientific_exp_001_annotation",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def timestamp_values(path: Path, frame_count: int, indices: list[int]) -> tuple[list[float | None], str]:
    with h5py.File(path, "r") as handle:
        timestamp_path = "/img_metadata/img_timestamp"
        if timestamp_path not in handle:
            return [None] * len(indices), "unavailable"
        timestamp = handle[timestamp_path]
        if len(timestamp) == frame_count:
            return [float(timestamp[index]) for index in indices], "one_timestamp_per_img_nir_frame"
        if len(timestamp) == 2 * frame_count:
            return [float(timestamp[2 * index]) for index in indices], "every_other_timestamp_length_2x_img_nir"
        if "/img_metadata/q_iter_save" in handle:
            saved = np.asarray(handle["/img_metadata/q_iter_save"]).astype(bool)
            if len(saved) == len(timestamp) and int(saved.sum()) == frame_count:
                positions = np.flatnonzero(saved)
                return [float(timestamp[int(positions[index])]) for index in indices], "q_iter_save_true"
    return [None] * len(indices), "unmapped"


def plot_preview(records: list[dict[str, Any]], output: Path) -> None:
    # A deterministic subset verifies identities and visual diversity without
    # showing a proxy/model overlay.  Source recordings are opened read-only.
    subset_positions = np.linspace(0, len(records) - 1, 36, dtype=np.int64)
    subset = [records[int(index)] for index in subset_positions]
    fig, axes = plt.subplots(6, 6, figsize=(15, 12), constrained_layout=True)
    sources: dict[str, HDF5FrameSource] = {}
    try:
        for axis, record in zip(axes.flat, subset, strict=True):
            recording = record["recording"]
            if recording not in sources:
                sources[recording] = HDF5FrameSource(
                    PROJECT_ROOT / record["configured_source_path"],
                    record["source_dataset_path"], expected_ndim=3, max_frames_per_read=11,
                )
            image = sources[recording].read_frame(record["frame_index"])
            axis.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
            axis.set_title(f"{recording}\nf{record['frame_index']} {record['selection_stratum']}", fontsize=7)
            axis.axis("off")
    finally:
        for source in sources.values():
            source.close()
    fig.suptitle("EXP-001 blind selection preview (no pose overlays)")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_manifest.read_text())
    records: list[dict[str, Any]] = []
    allocation = [86, 85, 85]
    for record_position, (recording, target_count) in enumerate(
        zip(split["development_records"], allocation, strict=True)
    ):
        source = split["records"][recording]
        selected = select_recording_frames(
            recording=recording,
            frame_count=int(source["frame_count"]),
            target_count=target_count,
            proxy_rows=proxy_rows(args.proxy_hdf5, recording),
            seed=args.seed + record_position,
        )
        indices = [value.frame_index for value in selected]
        configured = PROJECT_ROOT / source["configured_path"]
        timestamp, timestamp_mapping = timestamp_values(configured, int(source["frame_count"]), indices)
        stat = configured.stat()
        with h5py.File(configured, "r") as handle:
            frame_shape = tuple(int(value) for value in handle[split["source_dataset_path"]].shape[1:])
        if len(frame_shape) != 2:
            raise RuntimeError(f"expected grayscale frames for {recording}, got {frame_shape}")
        if int(stat.st_size) != int(source["size_bytes"]) or int(stat.st_mtime_ns) != int(source["mtime_ns"]):
            raise RuntimeError(f"source identity changed for {recording}")
        for value, raw_timestamp in zip(selected, timestamp, strict=True):
            row = value.as_dict()
            row.update(
                {
                    "split_role": "development_tier_a",
                    "configured_source_path": source["configured_path"],
                    "resolved_source_path": str(configured.resolve(strict=True)),
                    "source_size_bytes": int(stat.st_size),
                    "source_mtime_ns": int(stat.st_mtime_ns),
                    "source_dataset_path": split["source_dataset_path"],
                    "image_height": frame_shape[0],
                    "image_width": frame_shape[1],
                    "timestamp_raw": raw_timestamp,
                    "timestamp_mapping": timestamp_mapping,
                    "annotation_overlays": [],
                    "primary_annotation_view": "raw_temporal_context_without_pose_overlay",
                }
            )
            records.append(row)
    if len(records) != 256 or sum(record["double_annotate"] for record in records) < 64:
        raise RuntimeError("EXP-001 requires 256 frames and at least 64 double annotations")
    if {record["recording"] for record in records} & {split["audited_holdout"]["record"]}:
        raise RuntimeError("protected holdout entered the annotation selection")

    manifest: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "experiment": "EXP-001",
        "seed": args.seed,
        "selection_policy": (
            "development-only; two complete blind 11-frame double-annotation windows per session; "
            "classical accept/reject outcomes enrich difficulty but centerlines/overlays remain hidden"
        ),
        "protected_holdout_opened": False,
        "source_dataset_path": split["source_dataset_path"],
        "total_frames": len(records),
        "double_annotation_frames": sum(record["double_annotate"] for record in records),
        "records": records,
    }
    manifest["records_sha256"] = canonical_json_sha256(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "selection_manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_preview:
        plot_preview(records, args.output_dir / "selection_preview.png")
    print(json.dumps({key: manifest[key] for key in (
        "experiment", "total_frames", "double_annotation_frames",
        "protected_holdout_opened", "records_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
