#!/usr/bin/env python3
"""Evaluate independently locked Tier-A annotations for EXP-001."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.annotation import (
    annotation_pair_metrics,
    annotation_semantic_pair_metrics,
    bootstrap_interval,
    resample_polyline,
    validate_annotation,
)
from worm_pose_gen.data import HDF5FrameSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--comparison-mode",
        choices=("inter-annotator", "intra-annotator"),
        default="inter-annotator",
        help="Use intra-annotator only for delayed blind repeats by one person",
    )
    parser.add_argument(
        "--minimum-pairs",
        type=int,
        default=None,
        help="Defaults to 64 for inter-annotator agreement and 10 for repeatability",
    )
    return parser.parse_args()


def load_annotations(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text()
        if path.suffix == ".jsonl":
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            values = payload.get("annotations", []) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError(f"annotations in {path} must be a list")
        records.extend(values)
    return records


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def verify_manifest_binding(annotation: dict[str, Any], row: dict[str, Any]) -> None:
    """Reject annotations whose copied source/selection identity drifted."""

    for key in (
        "configured_source_path", "resolved_source_path", "source_size_bytes",
        "source_mtime_ns", "source_dataset_path", "frame_index", "split_role",
        "selection_stratum", "timestamp_raw",
        "timestamp_mapping",
    ):
        if annotation.get(key) != row.get(key):
            raise ValueError(f"annotation {annotation.get('annotation_id')!r} mismatches manifest {key}")
    if list(annotation.get("temporal_window_indices", [])) != list(row["temporal_window_indices"]):
        raise ValueError("annotation temporal_window_indices mismatch the manifest")
    if annotation.get("annotation_view") != "temporal_window":
        raise ValueError("EXP-001 primary annotations must declare temporal_window view")
    if annotation.get("annotation_overlays") != []:
        raise ValueError("EXP-001 primary annotations must be overlay-blind")


def aggregate(pair_metrics: list[dict[str, Any]], *, resamples: int, seed: int) -> dict[str, Any]:
    frame_median = np.asarray([value["median_point_distance_px"] for value in pair_metrics])
    all_points = np.concatenate([value["point_distance_px"] for value in pair_metrics])
    angle = np.asarray([value["mean_tangent_angle_error_deg"] for value in pair_metrics])
    endpoints = np.stack([value["endpoint_distance_px"] for value in pair_metrics])
    length = np.asarray([value["relative_body_length_disagreement"] for value in pair_metrics])
    normalized = np.asarray(
        [value["median_point_distance_body_widths"] for value in pair_metrics
         if value["median_point_distance_body_widths"] is not None]
    )
    return {
        "complete_pair_count": len(pair_metrics),
        "point_distance_px": {
            "median_all_points": float(np.median(all_points)),
            "p95_all_points": float(np.percentile(all_points, 95)),
            "median_of_frame_medians": float(np.median(frame_median)),
            "median_frame_bootstrap_95_interval": bootstrap_interval(
                frame_median, resamples=resamples, seed=seed
            ),
        },
        "tangent_angle_error_deg": {
            "mean_of_frame_means": float(angle.mean()),
            "median_of_frame_means": float(np.median(angle)),
            "p95_of_frame_means": float(np.percentile(angle, 95)),
        },
        "endpoint_distance_px": {
            "start_median": float(np.median(endpoints[:, 0])),
            "end_median": float(np.median(endpoints[:, 1])),
            "either_endpoint_p95": float(np.percentile(endpoints, 95)),
        },
        "relative_body_length_disagreement": {
            "median": float(np.median(length)),
            "p95": float(np.percentile(length, 95)),
        },
        "median_point_distance_body_widths": (
            {
                "frame_count": len(normalized),
                "median": float(np.median(normalized)),
                "p95": float(np.percentile(normalized, 95)),
            }
            if len(normalized) else None
        ),
    }


def plot_agreement(
    pair_metrics: list[dict[str, Any]], path: Path, *, comparison_label: str
) -> None:
    point = np.concatenate([value["point_distance_px"] for value in pair_metrics])
    angle = np.concatenate([value["tangent_angle_error_deg"] for value in pair_metrics])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(point, bins=40, color="#31688e")
    axes[0].axvline(np.median(point), color="black", linestyle="--", label=f"median {np.median(point):.2f} px")
    axes[0].axvline(np.percentile(point, 95), color="#b40426", linestyle=":", label=f"p95 {np.percentile(point, 95):.2f} px")
    axes[0].set(xlabel="point disagreement (px)", ylabel="body samples", title="Centerline disagreement")
    axes[0].legend()
    axes[1].hist(angle, bins=40, color="#35b779")
    axes[1].axvline(np.mean(angle), color="black", linestyle="--", label=f"mean {np.mean(angle):.2f}°")
    axes[1].set(xlabel="absolute tangent disagreement (degrees)", ylabel="body samples", title="Tangent-angle disagreement")
    axes[1].legend()
    fig.suptitle(f"EXP-001 {comparison_label}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_body_position(
    pair_metrics: list[dict[str, Any]], path: Path, *, comparison_label: str
) -> None:
    point = np.stack([value["point_distance_px"] for value in pair_metrics])
    angle = np.stack([value["tangent_angle_error_deg"] for value in pair_metrics])
    position = np.linspace(0, 1, point.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for axis, values, label in (
        (axes[0], point, "point disagreement (px)"),
        (axes[1], angle, "tangent disagreement (degrees)"),
    ):
        axis.plot(position, np.median(values, axis=0), color="#31688e", label="median")
        axis.fill_between(position, np.percentile(values, 25, axis=0), np.percentile(values, 75, axis=0), color="#31688e", alpha=0.25, label="IQR")
        axis.plot(position, np.percentile(values, 95, axis=0), color="#b40426", linestyle=":", label="p95")
        axis.set_ylabel(label)
        axis.legend(ncol=3)
    axes[-1].set_xlabel("normalized body position (orientation-symmetric)")
    fig.suptitle(f"EXP-001 {comparison_label} by body position")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_overlay_montage(
    pair_metrics: list[dict[str, Any]], annotations: dict[str, list[Any]],
    manifest: dict[str, dict[str, Any]], path: Path, *,
    comparison_label: str,
    annotation_pass_by_id: dict[str, str] | None = None,
) -> None:
    worst = sorted(pair_metrics, key=lambda value: value["median_point_distance_px"], reverse=True)[:12]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)
    sources: dict[str, HDF5FrameSource] = {}
    try:
        for axis, values in zip(axes.flat, worst, strict=False):
            sample_id = values["sample_id"]
            row = manifest[sample_id]
            recording = row["recording"]
            if recording not in sources:
                source_path = PROJECT_ROOT / row["configured_source_path"]
                sources[recording] = HDF5FrameSource(
                    source_path, row["source_dataset_path"], expected_ndim=3,
                    max_frames_per_read=11,
                )
            image = sources[recording].read_frame(row["frame_index"])
            first, second = sorted(annotations[sample_id], key=lambda value: value.annotator_id)[:2]
            a = resample_polyline(first.points_xy, 100)
            b = resample_polyline(second.points_xy, 100)
            if values["first_trace_reversed"]:
                a = a[::-1]
            if values["second_trace_reversed"]:
                b = b[::-1]
            axis.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
            first_label = (
                annotation_pass_by_id.get(first.annotation_id, first.annotator_id)
                if annotation_pass_by_id else first.annotator_id
            )
            second_label = (
                annotation_pass_by_id.get(second.annotation_id, second.annotator_id)
                if annotation_pass_by_id else second.annotator_id
            )
            axis.plot(a[:, 0], a[:, 1], color="#00ffff", linewidth=1.2, label=first_label)
            axis.plot(b[:, 0], b[:, 1], color="#ff00ff", linewidth=1.2, label=second_label)
            axis.set_title(f"{sample_id}\nmedian {values['median_point_distance_px']:.2f} px", fontsize=8)
            axis.axis("off")
        for axis in axes.flat[len(worst):]:
            axis.axis("off")
    finally:
        for source in sources.values():
            source.close()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(f"Largest complete-trace {comparison_label} disagreements")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    manifest_payload = json.loads(args.manifest.read_text())
    manifest = {row["sample_id"]: row for row in manifest_payload["records"]}
    grouped: dict[str, list[Any]] = defaultdict(list)
    raw_by_annotation_id: dict[str, dict[str, Any]] = {}
    for raw in load_annotations(args.annotations):
        sample_id = raw.get("sample_id")
        if sample_id not in manifest:
            raise ValueError(f"annotation refers to unknown sample_id {sample_id!r}")
        row = manifest[sample_id]
        verify_manifest_binding(raw, row)
        validated = validate_annotation(
            raw, image_height=int(row["image_height"]), image_width=int(row["image_width"])
        )
        grouped[sample_id].append(validated)
        raw_by_annotation_id[validated.annotation_id] = raw
    pair_metrics: list[dict[str, Any]] = []
    semantic_metrics: list[dict[str, Any]] = []
    incomplete: list[str] = []
    expected_double_ids = {
        sample_id for sample_id, row in manifest.items() if row["double_annotate"]
    }
    intra_annotator = args.comparison_mode == "intra-annotator"
    required_pairs = args.minimum_pairs if args.minimum_pairs is not None else (10 if intra_annotator else 64)
    if required_pairs < 1:
        raise ValueError("minimum-pairs must be positive")
    for sample_id, values in sorted(grouped.items()):
        if intra_annotator:
            values = sorted(
                values,
                key=lambda value: str(raw_by_annotation_id[value.annotation_id].get("annotation_pass", "")),
            )
        else:
            values = sorted(values, key=lambda value: value.annotator_id)
        annotator_ids = [value.annotator_id for value in values]
        if not intra_annotator and len(set(annotator_ids)) != len(annotator_ids):
            raise ValueError(f"duplicate annotator record for {sample_id}")
        if len(values) < 2:
            continue
        if not intra_annotator and sample_id not in expected_double_ids:
            continue
        if len(values) != 2:
            raise ValueError(f"{sample_id} must have exactly two independent annotations")
        if intra_annotator:
            passes = [raw_by_annotation_id[value.annotation_id].get("annotation_pass") for value in values]
            if set(passes) != {"primary", "repeat"}:
                raise ValueError(f"{sample_id} intra-annotator pair must contain primary and repeat passes")
            if len(set(annotator_ids)) != 1:
                raise ValueError(f"{sample_id} intra-annotator pair must come from one annotator")
        semantic_metrics.append(annotation_semantic_pair_metrics(
            values[0], values[1], allow_same_annotator=intra_annotator
        ))
        complete_pairs = [pair for pair in combinations(values, 2) if pair[0].is_complete and pair[1].is_complete]
        if not complete_pairs:
            incomplete.append(sample_id)
            continue
        if len(complete_pairs) > 1:
            raise ValueError(f"{sample_id} has more than one independent annotation pair")
        pair_metrics.append(annotation_pair_metrics(
            *complete_pairs[0], allow_same_annotator=intra_annotator
        ))
    if not pair_metrics:
        raise ValueError("no complete independent annotation pairs are available")
    args.output_dir.mkdir(parents=True)
    summary = {
        "experiment": "EXP-001",
        "comparison_mode": args.comparison_mode,
        "interpretation": (
            "intra-annotator repeatability; does not estimate inter-annotator variability"
            if intra_annotator else "inter-annotator agreement"
        ),
        "conclusion": "INCONCLUSIVE" if len(pair_metrics) < required_pairs else "MEASURED",
        "required_complete_pairs": required_pairs,
        "independent_pair_count": len(semantic_metrics),
        "semantic_agreement": {
            "trace_state_fraction": float(np.mean([value["trace_state_agreement"] for value in semantic_metrics])),
            "head_tail_state_fraction": float(np.mean([value["head_tail_state_agreement"] for value in semantic_metrics])),
            "truncation_end_fraction": float(np.mean([value["truncation_end_agreement"] for value in semantic_metrics])),
        },
        "incomplete_pair_sample_ids": incomplete,
        "aggregate": aggregate(pair_metrics, resamples=args.bootstrap_resamples, seed=args.seed),
        "per_pair": pair_metrics,
        "per_pair_semantics": semantic_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n")
    comparison_label = "intra-annotator repeatability" if intra_annotator else "inter-annotator precision"
    plot_agreement(
        pair_metrics, args.output_dir / "human_annotation_agreement.png",
        comparison_label=comparison_label,
    )
    plot_body_position(
        pair_metrics, args.output_dir / "human_error_by_body_position.png",
        comparison_label=comparison_label,
    )
    plot_overlay_montage(
        pair_metrics, grouped, manifest, args.output_dir / "human_annotation_overlay_montage.png",
        comparison_label=comparison_label,
        annotation_pass_by_id=(
            {
                annotation_id: str(raw.get("annotation_pass", "annotation"))
                for annotation_id, raw in raw_by_annotation_id.items()
            }
            if intra_annotator else None
        ),
    )
    print(json.dumps(json_safe(summary["aggregate"]), indent=2))


if __name__ == "__main__":
    main()
