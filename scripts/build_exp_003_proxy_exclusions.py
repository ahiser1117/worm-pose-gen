#!/usr/bin/env python3
"""Freeze proxy-training exclusions around the 30 Tier-A primary frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_tier_a_primary import _verified_inputs
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from evaluate_tier_a_primary import _verified_inputs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_intervals(values: set[int]) -> list[list[int]]:
    if not values:
        return []
    ordered = sorted(values)
    result: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            result.append([start, previous])
            start = value
        previous = value
    result.append([start, previous])
    return result


def build_exclusions(
    manifest_path: Path, annotation_path: Path, *, temporal_radius: int
) -> dict[str, Any]:
    if temporal_radius < 0:
        raise ValueError("temporal_radius must be nonnegative")
    payload, rows = _verified_inputs(manifest_path, annotation_path)
    recordings = sorted({str(source["recording"]) for source, _ in rows})
    primary: dict[str, list[int]] = {recording: [] for recording in recordings}
    exclusion: dict[str, set[int]] = {recording: set() for recording in recordings}
    for source, _ in rows:
        recording = str(source["recording"])
        frame = int(source["frame_index"])
        primary[recording].append(frame)
        exclusion[recording].update(
            range(max(0, frame - temporal_radius), frame + temporal_radius + 1)
        )
    return {
        "schema_version": 1,
        "experiment": "EXP-003",
        "purpose": "exclude_Tier_A_primary_targets_and_temporal_neighbors_from_proxy_training",
        "temporal_radius_frames": temporal_radius,
        "inputs": {
            "manifest": str(manifest_path.resolve(strict=True)),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_records_sha256": payload["manifest_records_sha256"],
            "annotations": str(annotation_path.resolve(strict=True)),
            "annotations_sha256": sha256_file(annotation_path),
        },
        "recordings": {
            recording: {
                "primary_frame_indices": sorted(primary[recording]),
                "excluded_frame_indices": sorted(exclusion[recording]),
                "excluded_intervals_inclusive": merge_intervals(exclusion[recording]),
                "primary_frame_count": len(primary[recording]),
                "excluded_frame_count": len(exclusion[recording]),
            }
            for recording in recordings
        },
        "total_primary_frames": sum(len(value) for value in primary.values()),
        "total_excluded_recording_frame_pairs": sum(len(value) for value in exclusion.values()),
        "protected_holdout_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--temporal-radius", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a frozen exclusion manifest")
    result = build_exclusions(
        args.manifest, args.annotations, temporal_radius=args.temporal_radius
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "total_primary_frames": result["total_primary_frames"],
        "total_excluded_recording_frame_pairs": result[
            "total_excluded_recording_frame_pairs"
        ],
        "protected_holdout_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
