#!/usr/bin/env python3
"""Score frozen baselines on a completed single-annotator Tier-A primary pass.

This is directional development evidence, not an annotation-noise estimate.  It
scores complete traces pointwise and reports one-way visible-trace coverage for
naturally truncated traces.  The latter deliberately makes no claim about
hidden anatomy or matched anatomical body positions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.annotation import (
    ValidatedAnnotation,
    resample_polyline,
    validate_annotation,
)
from worm_pose_gen.annotation_tool import AnnotationProtocol, AnnotationSession
from worm_pose_gen.classical import ClassicalResult, extract_centerline
from worm_pose_gen.data import HDF5FrameSource
from worm_pose_gen.inference import ExploratoryPoseInference, VALIDATION_STATUS


SEED = 20260819


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.remainder(value + np.pi, 2 * np.pi) - np.pi


def _tangent(points: np.ndarray) -> np.ndarray:
    derivative = np.empty_like(points)
    derivative[0] = points[1] - points[0]
    derivative[-1] = points[-1] - points[-2]
    derivative[1:-1] = points[2:] - points[:-2]
    return np.arctan2(derivative[:, 1], derivative[:, 0])


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def complete_curve_metrics(
    prediction_xy: np.ndarray, annotation_xy: np.ndarray, *, num_points: int = 100
) -> dict[str, Any]:
    """Orientation-symmetric pointwise metrics for a fully visible trace."""

    prediction = resample_polyline(prediction_xy, num_points)
    target = resample_polyline(annotation_xy, num_points)
    forward = np.linalg.norm(prediction - target, axis=1)
    reverse = np.linalg.norm(prediction - target[::-1], axis=1)
    reversed_target = bool(reverse.mean() < forward.mean())
    if reversed_target:
        target = target[::-1]
    distance = np.linalg.norm(prediction - target, axis=1)
    angle = np.abs(_wrap_angle(_tangent(prediction) - _tangent(target)))
    endpoint = distance[[0, -1]]
    return {
        "target_reversed": reversed_target,
        "point_distance_px": distance.tolist(),
        "tangent_error_deg": np.rad2deg(angle).tolist(),
        "median_point_distance_px": float(np.median(distance)),
        "mean_point_distance_px": float(np.mean(distance)),
        "p95_point_distance_px": float(np.percentile(distance, 95)),
        "mean_tangent_error_deg": float(np.rad2deg(angle).mean()),
        "p95_tangent_error_deg": float(np.percentile(np.rad2deg(angle), 95)),
        "mean_endpoint_error_px": float(endpoint.mean()),
        "body_length_error_px": abs(_length(prediction) - _length(target)),
        "body_length_error_fraction": abs(_length(prediction) - _length(target))
        / max(_length(target), np.finfo(float).eps),
    }


def visible_trace_metrics(
    prediction_xy: np.ndarray, annotation_xy: np.ndarray, *, num_points: int = 100
) -> dict[str, Any]:
    """One-way geometric coverage of a visible, anatomically unmatched trace."""

    prediction = resample_polyline(prediction_xy, num_points)
    target = resample_polyline(annotation_xy, num_points)
    segment = prediction[1:] - prediction[:-1]
    offset = target[:, None, :] - prediction[None, :-1, :]
    fraction = np.clip(
        np.einsum("tse,se->ts", offset, segment)
        / np.maximum(np.einsum("se,se->s", segment, segment), np.finfo(float).eps)[None, :],
        0.0,
        1.0,
    )
    projection = prediction[None, :-1, :] + fraction[..., None] * segment[None, :, :]
    pairwise = np.linalg.norm(target[:, None, :] - projection, axis=2)
    nearest_index = np.argmin(pairwise, axis=1)
    distance = pairwise[np.arange(num_points), nearest_index]
    target_tangent = _tangent(target)
    prediction_tangent = np.arctan2(segment[:, 1], segment[:, 0])[nearest_index]
    angle = np.abs(_wrap_angle(target_tangent - prediction_tangent))
    # A visible curve has no anatomical orientation; compare local axes in [0, pi/2].
    angle = np.minimum(angle, np.pi - angle)
    return {
        "visible_trace_distance_px": distance.tolist(),
        "visible_trace_axis_error_deg": np.rad2deg(angle).tolist(),
        "median_visible_trace_distance_px": float(np.median(distance)),
        "mean_visible_trace_distance_px": float(np.mean(distance)),
        "p95_visible_trace_distance_px": float(np.percentile(distance, 95)),
        "mean_visible_trace_axis_error_deg": float(np.rad2deg(angle).mean()),
        "metric_scope": "annotated_visible_trace_to_nearest_predicted_curve_point",
    }


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    sample = np.asarray(list(values), dtype=np.float64)
    if not len(sample):
        return {"n": 0, "median": None, "mean": None, "p95": None}
    return {
        "n": int(len(sample)),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "p95": float(np.percentile(sample, 95)),
    }


def summarize_method(cases: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [case for case in cases if case.get("complete_metrics") is not None]
    truncated = [case for case in cases if case.get("visible_metrics") is not None]
    available = [case for case in cases if case["prediction_available"]]
    accepted = [case for case in cases if case["accepted_output"]]
    complete_metrics = [case["complete_metrics"] for case in complete]
    visible_metrics = [case["visible_metrics"] for case in truncated]
    return {
        "requested_frames": len(cases),
        "prediction_available_frames": len(available),
        "prediction_available_fraction": len(available) / len(cases),
        "accepted_output_frames": len(accepted),
        "accepted_output_fraction": len(accepted) / len(cases),
        "complete_trace_scored_frames": len(complete),
        "truncated_visible_trace_scored_frames": len(truncated),
        "complete_trace": {
            "per_frame_median_point_distance_px": _summary(
                value["median_point_distance_px"] for value in complete_metrics
            ),
            "all_body_positions_point_distance_px": _summary(
                point
                for value in complete_metrics
                for point in value["point_distance_px"]
            ),
            "per_frame_mean_tangent_error_deg": _summary(
                value["mean_tangent_error_deg"] for value in complete_metrics
            ),
            "per_frame_mean_endpoint_error_px": _summary(
                value["mean_endpoint_error_px"] for value in complete_metrics
            ),
            "per_frame_body_length_error_fraction": _summary(
                value["body_length_error_fraction"] for value in complete_metrics
            ),
        },
        "truncated_visible_trace_diagnostic": {
            "per_frame_median_distance_px": _summary(
                value["median_visible_trace_distance_px"] for value in visible_metrics
            ),
            "all_visible_positions_distance_px": _summary(
                point
                for value in visible_metrics
                for point in value["visible_trace_distance_px"]
            ),
            "per_frame_mean_axis_error_deg": _summary(
                value["mean_visible_trace_axis_error_deg"] for value in visible_metrics
            ),
            "interpretation": (
                "one-way visible-trace coverage only; no hidden-body or matched-"
                "anatomical-position claim"
            ),
        },
    }


def _verified_inputs(
    manifest_path: Path, annotation_path: Path
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], ValidatedAnnotation]]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protected_holdout_opened") is not False:
        raise RuntimeError("refusing a manifest that opens the protected holdout")
    payload = json.loads(annotation_path.read_text())
    annotator_id = payload.get("annotator_id")
    if not isinstance(annotator_id, str) or not annotator_id.strip():
        raise RuntimeError("annotation session has no stable annotator_id")
    session = AnnotationSession(
        manifest_path, annotation_path, annotator_id, AnnotationProtocol()
    )
    state = session.state()
    if state["primary_complete"] != 30 or state["repeat_complete"] != 0:
        raise RuntimeError("this frozen analysis requires exactly 30 primary and 0 repeat traces")
    records = {str(row["sample_id"]): row for row in manifest["records"]}
    result: list[tuple[dict[str, Any], ValidatedAnnotation]] = []
    for raw in payload["annotations"]:
        if raw.get("annotation_pass") != "primary":
            raise RuntimeError("primary analysis encountered a non-primary annotation")
        source = records[str(raw["sample_id"])]
        for field in (
            "frame_index", "configured_source_path", "resolved_source_path",
            "source_size_bytes", "source_mtime_ns", "source_dataset_path",
            "split_role", "selection_stratum",
        ):
            if raw.get(field) != source.get(field):
                raise RuntimeError(f"annotation/manifest mismatch for {raw['sample_id']}:{field}")
        if raw.get("annotation_overlays") != []:
            raise RuntimeError("primary annotation contains an overlay")
        validated = validate_annotation(
            raw, image_height=int(source["image_height"]), image_width=int(source["image_width"])
        )
        result.append((source, validated))
    if len(result) != 30 or len({value.sample_id for _, value in result}) != 30:
        raise RuntimeError("expected 30 unique primary samples")
    return payload, result


def _read_frames(
    rows: list[tuple[dict[str, Any], ValidatedAnnotation]]
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[dict[str, Any], ValidatedAnnotation]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[0]["resolved_source_path"])].append(row)
    frames: dict[str, np.ndarray] = {}
    for raw_path, group in grouped.items():
        path = Path(raw_path)
        expected_size = int(group[0][0]["source_size_bytes"])
        expected_mtime = int(group[0][0]["source_mtime_ns"])
        stat = path.stat()
        if stat.st_size != expected_size or stat.st_mtime_ns != expected_mtime:
            raise RuntimeError(f"source identity changed: {path}")
        source = HDF5FrameSource(
            path,
            str(group[0][0]["source_dataset_path"]),
            expected_frame_shape=(732, 968),
            expected_ndim=3,
            max_frames_per_read=len(group),
        )
        indices = [int(item[0]["frame_index"]) for item in group]
        batch = source.read_indices(indices)
        for item, frame in zip(group, batch, strict=True):
            frames[item[1].sample_id] = frame
        source.close()
    return frames


def _case(
    source: dict[str, Any],
    annotation: ValidatedAnnotation,
    prediction: np.ndarray | None,
    *,
    accepted: bool,
    reason: list[str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "sample_id": annotation.sample_id,
        "recording": source["recording"],
        "frame_index": int(source["frame_index"]),
        "selection_stratum": source["selection_stratum"],
        "trace_state": annotation.trace_state,
        "prediction_available": prediction is not None,
        "accepted_output": bool(accepted),
        "rejection_reasons": reason or [],
        "complete_metrics": None,
        "visible_metrics": None,
    }
    if prediction is not None and annotation.is_complete:
        value["complete_metrics"] = complete_curve_metrics(prediction, annotation.points_xy)
    elif prediction is not None and annotation.trace_state == "truncated":
        value["visible_metrics"] = visible_trace_metrics(prediction, annotation.points_xy)
    return value


def _plot_summary(metrics: dict[str, Any], path: Path) -> None:
    names = ["classical accepted", "classical ungated\ndiagnostic", "rejected global\nmodel"]
    keys = ["classical_accepted", "classical_ungated_diagnostic", "rejected_global_model"]
    colors = ["#e69f00", "#999999", "#cc79a7"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    coverage = [metrics[key]["prediction_available_fraction"] * 100 for key in keys]
    axes[0].bar(names, coverage, color=colors)
    axes[0].set_ylabel("candidate output available (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Coverage on all 30 primary frames")
    complete = [
        metrics[key]["complete_trace"]["per_frame_median_point_distance_px"]["median"]
        for key in keys
    ]
    axes[1].bar(names, [np.nan if value is None else value for value in complete], color=colors)
    axes[1].set_ylabel("median point distance (px)")
    axes[1].set_title("Conditional complete-trace accuracy")
    visible = [
        metrics[key]["truncated_visible_trace_diagnostic"]["per_frame_median_distance_px"]["median"]
        for key in keys
    ]
    axes[2].bar(names, [np.nan if value is None else value for value in visible], color=colors)
    axes[2].set_ylabel("nearest-curve distance (px)")
    axes[2].set_title("Conditional truncated-trace coverage")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=15)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tier A primary-30 baseline triage (single annotator; directional)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_overlays(
    rows: list[tuple[dict[str, Any], ValidatedAnnotation]],
    frames: dict[str, np.ndarray],
    classical: dict[str, ClassicalResult],
    model: dict[str, np.ndarray],
    path: Path,
) -> None:
    # Deterministic coverage: each trace state and recording, with the worst model cases.
    def score(row: tuple[dict[str, Any], ValidatedAnnotation]) -> float:
        annotation = row[1]
        prediction = model[annotation.sample_id]
        if annotation.is_complete:
            return complete_curve_metrics(prediction, annotation.points_xy)["median_point_distance_px"]
        if annotation.trace_state == "truncated":
            return visible_trace_metrics(prediction, annotation.points_xy)["median_visible_trace_distance_px"]
        return float("inf")

    selected = sorted(rows, key=score, reverse=True)[:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9.5))
    for axis, (source, annotation) in zip(axes.flat, selected, strict=True):
        frame = frames[annotation.sample_id]
        lo, hi = np.percentile(frame, (1, 99))
        axis.imshow(frame, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
        if len(annotation.points_xy):
            axis.plot(
                annotation.points_xy[:, 0], annotation.points_xy[:, 1],
                color="#00ffff", lw=1.7, label="manual visible",
            )
        result = classical[annotation.sample_id]
        if result.centerline_xy is not None:
            axis.plot(
                result.centerline_xy[:, 0], result.centerline_xy[:, 1],
                color="#e69f00", lw=1.2, ls="-" if result.accepted else ":",
                label="classical" if result.accepted else "classical rejected candidate",
            )
        axis.plot(
            model[annotation.sample_id][:, 0], model[annotation.sample_id][:, 1],
            color="#ff4fb3", lw=1.2, label="rejected global model",
        )
        axis.set_title(
            f"{annotation.sample_id}\n{annotation.trace_state}; {source['selection_stratum']}",
            fontsize=8,
        )
        axis.set_xlim(0, frame.shape[1])
        axis.set_ylim(frame.shape[0], 0)
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Worst rejected-model Tier A primary cases (not agreement evidence)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        metrics_path = args.output_dir / "metrics.json"
        if not args.overwrite:
            raise FileExistsError("refusing to overwrite an existing Tier-A evaluation")
        if not metrics_path.exists() or json.loads(metrics_path.read_text()).get(
            "experiment"
        ) != "EXP-002-primary30-directional":
            raise RuntimeError("overwrite target is not this experiment's output directory")
    payload, rows = _verified_inputs(args.manifest, args.annotations)
    frames = _read_frames(rows)

    classical_results: dict[str, ClassicalResult] = {}
    start = time.perf_counter()
    for _, annotation in rows:
        classical_results[annotation.sample_id] = extract_centerline(
            frames[annotation.sample_id]
        )
    classical_seconds = time.perf_counter() - start

    inference = ExploratoryPoseInference.from_checkpoint(
        args.checkpoint,
        original_height=732,
        original_width=968,
        config_path=args.inference_config,
        device="cpu",
        allow_exploratory=True,
    )
    image_batch = np.stack([frames[annotation.sample_id] for _, annotation in rows])
    start = time.perf_counter()
    prediction_batch = inference.predict_batch(image_batch)
    model_seconds = time.perf_counter() - start
    model_predictions = {
        annotation.sample_id: prediction_batch.centerline_xy[index].numpy()
        for index, (_, annotation) in enumerate(rows)
    }

    classical_accepted: list[dict[str, Any]] = []
    classical_ungated: list[dict[str, Any]] = []
    model_cases: list[dict[str, Any]] = []
    for source, annotation in rows:
        result = classical_results[annotation.sample_id]
        classical_accepted.append(_case(
            source,
            annotation,
            result.centerline_xy if result.accepted else None,
            accepted=result.accepted,
            reason=list(result.rejection_reasons),
        ))
        classical_ungated.append(_case(
            source,
            annotation,
            result.centerline_xy,
            accepted=result.centerline_xy is not None,
            reason=list(result.rejection_reasons),
        ))
        model_cases.append(_case(
            source,
            annotation,
            model_predictions[annotation.sample_id],
            # The exploratory checkpoint emits a candidate on every frame, but
            # its frozen quality sentinel is zero: none is quality-accepted.
            accepted=False,
        ))

    method_cases = {
        "classical_accepted": classical_accepted,
        "classical_ungated_diagnostic": classical_ungated,
        "rejected_global_model": model_cases,
    }
    methods = {name: summarize_method(cases) for name, cases in method_cases.items()}
    methods["classical_accepted"]["runtime"] = {
        "seconds_30_frames": classical_seconds,
        "frames_per_second": len(rows) / classical_seconds,
        "scope": "in_memory_frame_to_result_serial_cpu",
    }
    methods["classical_ungated_diagnostic"]["runtime"] = methods["classical_accepted"]["runtime"]
    methods["rejected_global_model"]["runtime"] = {
        "seconds_30_frames": model_seconds,
        "frames_per_second": len(rows) / model_seconds,
        "scope": "in_memory_batch_preprocess_and_forward_cpu_excludes_checkpoint_load",
    }

    trace_counts = Counter(annotation.trace_state for _, annotation in rows)
    recording_counts = Counter(str(source["recording"]) for source, _ in rows)
    stratum_counts = Counter(str(source["selection_stratum"]) for source, _ in rows)
    rejection_counts = Counter(
        reason
        for result in classical_results.values()
        for reason in result.rejection_reasons
    )
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "EXP-002-primary30-directional",
        "result_scope": (
            "single-annotator development Tier A; primary pass only; not inter- or "
            "intra-annotator precision"
        ),
        "seed": SEED,
        "inputs": {
            "manifest": str(args.manifest.resolve(strict=True)),
            "manifest_sha256": sha256_file(args.manifest),
            "manifest_records_sha256": payload["manifest_records_sha256"],
            "annotations": str(args.annotations.resolve(strict=True)),
            "annotations_sha256": sha256_file(args.annotations),
            "checkpoint": str(args.checkpoint.resolve(strict=True)),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "inference_config": str(args.inference_config.resolve(strict=True)),
            "inference_config_sha256": sha256_file(args.inference_config),
        },
        "annotation_tranche": {
            "primary_frames": len(rows),
            "repeat_frames": 0,
            "trace_state_counts": dict(sorted(trace_counts.items())),
            "recording_counts": dict(sorted(recording_counts.items())),
            "selection_stratum_counts": dict(sorted(stratum_counts.items())),
        },
        "methods": methods,
        "classical_rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "per_case": method_cases,
        "evidence_boundary": {
            "protected_holdout_opened": False,
            "source_recordings": sorted(recording_counts),
            "old_model_validation_status": VALIDATION_STATUS,
            "truncated_metric": (
                "one-way visible geometric coverage; hidden anatomy and matched body "
                "positions are not scored"
            ),
            "not_identifiable_frames": int(trace_counts["not_identifiable"]),
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=args.overwrite)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _plot_summary(methods, args.output_dir / "baseline_comparison.png")
    _plot_overlays(
        rows,
        frames,
        classical_results,
        model_predictions,
        args.output_dir / "baseline_worst_overlays.png",
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "trace_state_counts": dict(trace_counts),
        "methods": methods,
        "protected_holdout_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
