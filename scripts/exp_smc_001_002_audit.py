#!/usr/bin/env python3
"""Preregistered CPU audit for classical soft foreground and mask anchors.

The real audit is intentionally fail-closed: it verifies the frozen primary-30
source identities, rejects protected recordings, reads HDF5 in read-only mode,
and refuses to replace any generated result.  Pure summarization and gate
functions are kept importable without the mask-native runtime modules so they
can be unit tested before those APIs land.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.annotation import ValidatedAnnotation, resample_polyline, validate_annotation
from worm_pose_gen.data import HDF5FrameSource


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/smc_exp_001_002_audit.json"


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    sample = np.asarray(list(values), dtype=np.float64)
    if not len(sample):
        return {"n": 0, "median": None, "mean": None, "p10": None, "p95": None}
    return {
        "n": int(len(sample)),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "p10": float(np.percentile(sample, 10)),
        "p95": float(np.percentile(sample, 95)),
    }


def evaluate_exp_smc_001_gate(
    summary: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the frozen trace-proxy diagnostic gate.

    Passing is capped at ``PARTIALLY_SUPPORTED`` because the audited method is
    classical and the reference is a trace rather than a manual mask.
    """

    observed = {
        "median_cleaned_mask_trace_containment": summary.get(
            "cleaned_mask_trace_containment", {}
        ).get("median"),
        "p10_cleaned_mask_trace_containment": summary.get(
            "cleaned_mask_trace_containment", {}
        ).get("p10"),
        "median_terminal_containment": summary.get("terminal_containment", {}).get(
            "median"
        ),
        "median_soft_probability_on_trace": summary.get(
            "soft_probability_on_trace", {}
        ).get("median"),
        "non_identifiable_evidence_used": int(
            summary.get("non_identifiable_evidence_used", 0)
        ),
    }
    soft_minimum = gate.get("median_soft_probability_on_trace_min")
    checks = {
        "median_cleaned_mask_trace_containment": _at_least(
            observed["median_cleaned_mask_trace_containment"],
            gate["median_cleaned_mask_trace_containment_min"],
        ),
        "p10_cleaned_mask_trace_containment": _at_least(
            observed["p10_cleaned_mask_trace_containment"],
            gate["p10_cleaned_mask_trace_containment_min"],
        ),
        "median_terminal_containment": _at_least(
            observed["median_terminal_containment"],
            gate["median_terminal_containment_min"],
        ),
        # A later structural cleanup experiment may deliberately leave the
        # uncalibrated classical score untouched.  ``null`` means report this
        # diagnostic without pretending that rescaling a logistic score is a
        # scientific segmentation improvement.
        "median_soft_probability_on_trace": (
            True
            if soft_minimum is None
            else _at_least(
                observed["median_soft_probability_on_trace"], soft_minimum
            )
        ),
        "non_identifiable_evidence_used": (
            observed["non_identifiable_evidence_used"]
            <= int(gate["non_identifiable_evidence_used_max"])
        ),
    }
    optional_maxima = {
        "median_hysteresis_recovered_area_fraction": (
            "hysteresis_recovered_area_fraction",
            "median",
            "median_hysteresis_recovered_area_fraction_max",
        ),
        "p95_hysteresis_recovered_area_fraction": (
            "hysteresis_recovered_area_fraction",
            "p95",
            "p95_hysteresis_recovered_area_fraction_max",
        ),
        "p95_adjacent_area_relative_change": (
            "adjacent_area_relative_change",
            "p95",
            "p95_adjacent_area_relative_change_max",
        ),
    }
    for check_name, (summary_name, statistic, gate_name) in optional_maxima.items():
        if gate_name not in gate:
            continue
        value = summary.get(summary_name, {}).get(statistic)
        observed[check_name] = value
        checks[check_name] = _at_most(value, gate[gate_name])
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": "PARTIALLY_SUPPORTED" if passed else "NOT_SUPPORTED",
        "evidence_ceiling": "PARTIALLY_SUPPORTED",
        "observed": observed,
        "checks": checks,
    }


def evaluate_exp_smc_002_gate(
    summary: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the frozen conditional accepted-anchor gate."""

    observed = {
        "accepted_complete_count": int(summary.get("accepted_complete_count", 0)),
        "median_frame_point_error_px": summary.get("accepted_complete_point_error_px", {}).get(
            "median"
        ),
        "p95_frame_point_error_px": summary.get("accepted_complete_point_error_px", {}).get(
            "p95"
        ),
        "median_frame_tangent_error_deg": summary.get(
            "accepted_complete_tangent_error_deg", {}
        ).get("median"),
        "median_frame_endpoint_error_px": summary.get(
            "accepted_complete_endpoint_error_px", {}
        ).get("median"),
        "median_frame_length_error_fraction": summary.get(
            "accepted_complete_length_error_fraction", {}
        ).get("median"),
        "fraction_individually_at_most_8px": summary.get(
            "accepted_complete_fraction_individually_at_most_8px"
        ),
        "frames_at_or_above_20px": int(
            summary.get("accepted_complete_frames_at_or_above_20px", 0)
        ),
        "truncated_rejection_fraction": summary.get("truncated_rejection_fraction"),
    }
    checks = {
        "has_accepted_complete": observed["accepted_complete_count"] > 0,
        "median_frame_point_error_px": _at_most(
            observed["median_frame_point_error_px"],
            gate["accepted_complete_median_frame_point_error_px_max"],
        ),
        "p95_frame_point_error_px": _at_most(
            observed["p95_frame_point_error_px"],
            gate["accepted_complete_p95_frame_point_error_px_max"],
        ),
        "median_frame_tangent_error_deg": _at_most(
            observed["median_frame_tangent_error_deg"],
            gate["accepted_complete_median_frame_tangent_error_deg_max"],
        ),
        "median_frame_endpoint_error_px": _at_most(
            observed["median_frame_endpoint_error_px"],
            gate["accepted_complete_median_frame_endpoint_error_px_max"],
        ),
        "median_frame_length_error_fraction": _at_most(
            observed["median_frame_length_error_fraction"],
            gate["accepted_complete_median_frame_length_error_fraction_max"],
        ),
        "fraction_individually_at_most_8px": _at_least(
            observed["fraction_individually_at_most_8px"],
            gate["accepted_complete_fraction_individually_at_most_8px_min"],
        ),
        "frames_at_or_above_20px": (
            observed["frames_at_or_above_20px"]
            <= int(gate["accepted_complete_frames_at_or_above_20px_max"])
        ),
        "truncated_rejection_fraction": _at_least(
            observed["truncated_rejection_fraction"],
            gate["truncated_rejection_fraction_min"],
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": "SUPPORTED" if passed else "NOT_SUPPORTED",
        "observed": observed,
        "checks": checks,
    }


def _at_least(value: Any, threshold: Any) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) >= float(threshold)


def _at_most(value: Any, threshold: Any) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) <= float(threshold)


def summarize_segmentation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identifiable = [row for row in rows if row.get("trace_state") != "not_identifiable"]
    used_non_identifiable = [
        row
        for row in rows
        if row.get("trace_state") == "not_identifiable" and row.get("used_as_evidence", False)
    ]
    recovered_fractions: list[float] = []
    for row in rows:
        qc = row.get("segmentation_qc", {})
        seed_area = int(qc.get("largest_high_confidence_component_area", 0) or 0)
        recovered_area = int(qc.get("hysteresis_recovered_area", 0) or 0)
        if bool(qc.get("hysteresis_enabled", False)) and seed_area > 0:
            recovered_fractions.append(recovered_area / seed_area)
    return {
        "total_primary_frames": len(rows),
        "identifiable_trace_frames": len(identifiable),
        "not_identifiable_frames": len(rows) - len(identifiable),
        "non_identifiable_evidence_used": len(used_non_identifiable),
        "cleaned_mask_trace_containment": _summary(
            row["cleaned_mask_trace_containment"] for row in identifiable
        ),
        "terminal_containment": _summary(row["terminal_containment"] for row in identifiable),
        "soft_probability_on_trace": _summary(
            row["soft_probability_on_trace"] for row in identifiable
        ),
        "nearest_cleaned_mask_distance_px": _summary(
            row["median_nearest_cleaned_mask_distance_px"] for row in identifiable
        ),
        "terminal_omission_fraction": _summary(
            row["terminal_omission_fraction"] for row in identifiable
        ),
        "cleaned_mask_area_px": _summary(row["cleaned_mask_area_px"] for row in rows),
        "component_count": _summary(row["component_count"] for row in rows),
        "hole_count": _summary(row["hole_count"] for row in rows),
        "boundary_contact_pixels": _summary(row["boundary_contact_pixels"] for row in rows),
        "adjacent_area_relative_change": _summary(
            value for row in rows for value in row.get("adjacent_area_relative_change", [])
        ),
        "adjacent_centroid_displacement_px": _summary(
            value for row in rows for value in row.get("adjacent_centroid_displacement_px", [])
        ),
        "hysteresis_recovered_area_fraction": _summary(recovered_fractions),
    }


def summarize_anchor_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    accepted_complete = [
        row for row in accepted if row.get("trace_state") == "complete" and row.get("complete_metrics")
    ]
    truncated = [row for row in rows if row.get("trace_state") == "truncated"]
    errors = [float(row["complete_metrics"]["median_point_distance_px"]) for row in accepted_complete]
    def coverage_by(field: str) -> dict[str, dict[str, int | float]]:
        result: dict[str, dict[str, int | float]] = {}
        values = sorted({str(row.get(field, "unknown")) for row in rows})
        for name in values:
            group = [row for row in rows if str(row.get(field, "unknown")) == name]
            count = sum(bool(row["accepted"]) for row in group)
            result[name] = {
                "total": len(group),
                "accepted": count,
                "coverage": count / len(group) if group else 0.0,
            }
        return result
    rejection_reasons = Counter(
        reason for row in rows if not row["accepted"] for reason in row.get("rejection_reasons", [])
    )
    return {
        "total_frames": len(rows),
        "accepted_frames": len(accepted),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "coverage_by_stratum": coverage_by("selection_stratum"),
        "coverage_by_recording": coverage_by("recording"),
        "coverage_by_trace_state": coverage_by("trace_state"),
        "accepted_complete_count": len(accepted_complete),
        "accepted_complete_point_error_px": _summary(errors),
        "accepted_complete_tangent_error_deg": _summary(
            row["complete_metrics"]["mean_tangent_error_deg"] for row in accepted_complete
        ),
        "accepted_complete_endpoint_error_px": _summary(
            row["complete_metrics"]["mean_endpoint_error_px"] for row in accepted_complete
        ),
        "accepted_complete_length_error_fraction": _summary(
            row["complete_metrics"]["body_length_error_fraction"] for row in accepted_complete
        ),
        "accepted_complete_fraction_individually_at_most_8px": (
            sum(value <= 8.0 for value in errors) / len(errors) if errors else None
        ),
        "conditional_accepted_precision_proxy": {
            "definition": "fraction of accepted complete frames individually at most 8 px median point error",
            "numerator": sum(value <= 8.0 for value in errors),
            "denominator": len(errors),
            "fraction": sum(value <= 8.0 for value in errors) / len(errors) if errors else None,
            "evidence_boundary": "single-annotator development traces; not human precision",
        },
        "accepted_complete_frames_at_or_above_20px": sum(value >= 20.0 for value in errors),
        "truncated_frames": len(truncated),
        "truncated_rejected_frames": sum(not row["accepted"] for row in truncated),
        "truncated_rejection_fraction": (
            sum(not row["accepted"] for row in truncated) / len(truncated) if truncated else None
        ),
        "accepted_truncated_visible_distance_px": _summary(
            row["visible_metrics"]["median_visible_trace_distance_px"]
            for row in accepted
            if row.get("trace_state") == "truncated" and row.get("visible_metrics")
        ),
        "mask_render_iou_trace_width_proxy": _summary(
            row["mask_render_iou_trace_width_proxy"]
            for row in accepted
            if row.get("mask_render_iou_trace_width_proxy") is not None
        ),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
    }


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.remainder(value + np.pi, 2 * np.pi) - np.pi


def _tangent(points: np.ndarray) -> np.ndarray:
    derivative = np.empty_like(points)
    derivative[0] = points[1] - points[0]
    derivative[-1] = points[-1] - points[-2]
    derivative[1:-1] = points[2:] - points[:-2]
    return np.arctan2(derivative[:, 1], derivative[:, 0])


def _curve_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def complete_curve_metrics(
    prediction_xy: np.ndarray, annotation_xy: np.ndarray, *, num_points: int = 100
) -> dict[str, Any]:
    prediction = resample_polyline(prediction_xy, num_points)
    target = resample_polyline(annotation_xy, num_points)
    forward = np.linalg.norm(prediction - target, axis=1)
    reverse = np.linalg.norm(prediction - target[::-1], axis=1)
    if reverse.mean() < forward.mean():
        target = target[::-1]
    distance = np.linalg.norm(prediction - target, axis=1)
    angle = np.rad2deg(np.abs(_wrap_angle(_tangent(prediction) - _tangent(target))))
    return {
        "metric_orientation": "symmetric",
        "point_distance_px": distance.tolist(),
        "median_point_distance_px": float(np.median(distance)),
        "p95_point_distance_px": float(np.percentile(distance, 95)),
        "mean_tangent_error_deg": float(np.mean(angle)),
        "mean_endpoint_error_px": float(distance[[0, -1]].mean()),
        "body_length_error_fraction": abs(_curve_length(prediction) - _curve_length(target))
        / max(_curve_length(target), np.finfo(float).eps),
    }


def visible_trace_metrics(
    prediction_xy: np.ndarray, annotation_xy: np.ndarray, *, num_points: int = 100
) -> dict[str, Any]:
    prediction = resample_polyline(prediction_xy, num_points)
    target = resample_polyline(annotation_xy, num_points)
    segment = prediction[1:] - prediction[:-1]
    offset = target[:, None, :] - prediction[None, :-1, :]
    denominator = np.maximum(np.einsum("se,se->s", segment, segment), np.finfo(float).eps)
    fraction = np.clip(np.einsum("tse,se->ts", offset, segment) / denominator[None, :], 0, 1)
    projection = prediction[None, :-1, :] + fraction[..., None] * segment[None, :, :]
    distance = np.linalg.norm(target[:, None, :] - projection, axis=2).min(axis=1)
    return {
        "median_visible_trace_distance_px": float(np.median(distance)),
        "p95_visible_trace_distance_px": float(np.percentile(distance, 95)),
        "metric_scope": "one_way_visible_trace_to_predicted_curve; no hidden-body claim",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"result has no named fields: {type(value).__name__}")


def _field(value: Any, *names: str, default: Any = None) -> Any:
    mapping = _as_mapping(value)
    for name in names:
        if name in mapping:
            return mapping[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _construct_config(cls: type[Any], values: Mapping[str, Any]) -> Any:
    try:
        return cls(**dict(values))
    except TypeError as error:
        raise RuntimeError(f"configuration does not match {cls.__name__}: {error}") from error


def _verified_annotations(
    manifest_path: Path, annotation_path: Path, config: Mapping[str, Any]
) -> list[tuple[dict[str, Any], ValidatedAnnotation]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protected_holdout_opened") is not False:
        raise RuntimeError("manifest does not certify the protected holdout closed")
    annotations = json.loads(annotation_path.read_text())
    manifest_rows = {str(row["sample_id"]): row for row in manifest["records"]}
    primary = [row for row in annotations["annotations"] if row.get("annotation_pass") == "primary"]
    repeats = [row for row in annotations["annotations"] if row.get("annotation_pass") != "primary"]
    inputs = config["inputs"]
    if len(primary) != int(inputs["required_primary_annotations"]):
        raise RuntimeError("audit requires exactly the frozen 30 primary annotations")
    if len(repeats) != int(inputs["required_repeat_annotations"]):
        raise RuntimeError("audit refuses repeat annotations")
    result: list[tuple[dict[str, Any], ValidatedAnnotation]] = []
    for raw in primary:
        source = manifest_rows.get(str(raw["sample_id"]))
        if source is None:
            raise RuntimeError(f"annotation missing from manifest: {raw['sample_id']}")
        for field in (
            "frame_index", "resolved_source_path", "source_size_bytes", "source_mtime_ns",
            "source_dataset_path", "split_role", "selection_stratum",
        ):
            if raw.get(field) != source.get(field):
                raise RuntimeError(f"annotation/manifest mismatch: {raw['sample_id']}:{field}")
        if source["source_dataset_path"] != inputs["required_dataset_path"]:
            raise RuntimeError("audit is preregistered only for explicit /img_nir")
        recording = str(source["recording"])
        if any(recording.startswith(prefix) for prefix in inputs["forbidden_recording_prefixes"]):
            raise RuntimeError(f"protected recording encountered: {recording}")
        validated = validate_annotation(
            raw, image_height=int(source["image_height"]), image_width=int(source["image_width"])
        )
        result.append((source, validated))
    if len({annotation.sample_id for _, annotation in result}) != len(result):
        raise RuntimeError("duplicate primary sample IDs")
    return result


def _read_windows(
    rows: Sequence[tuple[dict[str, Any], ValidatedAnnotation]], half_window: int
) -> dict[tuple[str, int], np.ndarray]:
    grouped: dict[str, list[tuple[dict[str, Any], ValidatedAnnotation]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[0]["resolved_source_path"])].append(row)
    frames: dict[tuple[str, int], np.ndarray] = {}
    for raw_path, group in grouped.items():
        path = Path(raw_path)
        stat = path.stat()
        if stat.st_size != int(group[0][0]["source_size_bytes"]) or stat.st_mtime_ns != int(
            group[0][0]["source_mtime_ns"]
        ):
            raise RuntimeError(f"source identity changed: {path}")
        dataset_path = str(group[0][0]["source_dataset_path"])
        source = HDF5FrameSource(
            path, dataset_path, expected_frame_shape=(732, 968), expected_ndim=3,
            allowed_dtypes=(np.uint8,), max_frames_per_read=3,
        )
        recording = str(group[0][0]["recording"])
        for item, _ in group:
            center = int(item["frame_index"])
            window = source.read_window(center, before=half_window, after=half_window, padding="edge")
            for index, frame, valid in zip(window.source_indices, window.frames, window.valid_mask, strict=True):
                if valid:
                    frames[(recording, int(index))] = frame
        source.close()
    return frames


def _sample_at_points(array: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    x = np.clip(np.rint(points_xy[:, 0]).astype(int), 0, array.shape[1] - 1)
    y = np.clip(np.rint(points_xy[:, 1]).astype(int), 0, array.shape[0] - 1)
    return np.asarray(array[y, x])


def _nearest_mask_distances(mask: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    yx = np.argwhere(mask)
    if not len(yx):
        return np.full(len(points_xy), math.hypot(*mask.shape))
    foreground_xy = yx[:, ::-1].astype(np.float64)
    result = []
    for start in range(0, len(points_xy), 16):
        chunk = points_xy[start : start + 16]
        squared = ((chunk[:, None, :] - foreground_xy[None, :, :]) ** 2).sum(axis=2)
        result.extend(np.sqrt(squared.min(axis=1)).tolist())
    return np.asarray(result)


def _mask_qc(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    coords = {tuple(value) for value in np.argwhere(mask)}
    sizes: list[int] = []
    while coords:
        seed = coords.pop()
        queue = [seed]
        size = 0
        while queue:
            y, x = queue.pop()
            size += 1
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in coords:
                    coords.remove(neighbor)
                    queue.append(neighbor)
        sizes.append(size)
    area = int(mask.sum())
    if area:
        yx = np.argwhere(mask)
        centroid = [float(yx[:, 1].mean()), float(yx[:, 0].mean())]
        y0, x0 = np.maximum(yx.min(axis=0) - 1, 0)
        y1, x1 = np.minimum(yx.max(axis=0) + 2, mask.shape)
        crop = mask[y0:y1, x0:x1]
        background = ~crop
        outside = np.zeros_like(background)
        queue: deque[tuple[int, int]] = deque()
        for y in range(background.shape[0]):
            for x in (0, background.shape[1] - 1):
                if background[y, x] and not outside[y, x]:
                    outside[y, x] = True; queue.append((y, x))
        for x in range(background.shape[1]):
            for y in (0, background.shape[0] - 1):
                if background[y, x] and not outside[y, x]:
                    outside[y, x] = True; queue.append((y, x))
        while queue:
            y, x = queue.popleft()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < background.shape[0] and 0 <= xx < background.shape[1] and background[yy, xx] and not outside[yy, xx]:
                    outside[yy, xx] = True; queue.append((yy, xx))
        hole_pixels = background & ~outside
        hole_count = _component_count(hole_pixels)
    else:
        centroid = [None, None]
        hole_count = 0
    boundary_contact = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    return {
        "cleaned_mask_area_px": area,
        "component_count": len(sizes),
        "component_sizes_px": sorted(sizes, reverse=True),
        "hole_count": hole_count,
        "boundary_contact_pixels": boundary_contact,
        "centroid_xy": centroid,
    }


def _component_count(mask: np.ndarray) -> int:
    coords = {tuple(value) for value in np.argwhere(mask)}
    count = 0
    while coords:
        count += 1
        queue = [coords.pop()]
        while queue:
            y, x = queue.pop()
            for value in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if value in coords:
                    coords.remove(value); queue.append(value)
    return count


def _iou(first: np.ndarray, second: np.ndarray) -> float | None:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else None


def _safe_output_dir(path: Path, allow_existing_empty: bool) -> None:
    """Validate a destination without mutating it."""

    generated = {"metrics.json", "stdout.json", "figures"}
    if path.exists():
        collisions = [child for child in path.iterdir() if child.name in generated]
        if collisions:
            raise FileExistsError(f"refusing to overwrite generated audit output in {path}")
        allowed = {"config.json", "notes.md"}
        unexpected = [child.name for child in path.iterdir() if child.name not in allowed]
        if unexpected or not allow_existing_empty:
            raise FileExistsError(
                f"existing output scaffold requires --allow-existing-empty: {path}"
            )


def _initialize_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir()


def _render_anchor_mask(
    renderer: Any, centerline: np.ndarray, width: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    signature = inspect.signature(renderer)
    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if name in {"centerline_xy", "centerline"}:
            kwargs[name] = centerline
        elif name in {"width", "width_profile", "width_px", "widths"}:
            kwargs[name] = width
        elif name in {"image_shape", "shape"}:
            kwargs[name] = image_shape
    if len(kwargs) != len(signature.parameters):
        raise RuntimeError(f"unsupported render_centerline_mask signature: {signature}")
    return np.asarray(renderer(**kwargs), dtype=bool)


def _segmentation_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = _field(result, "probability", "probability_map", "soft_probability")
    raw_mask = _field(result, "raw_mask", "threshold_mask", "binary_mask")
    cleaned_mask = _field(result, "cleaned_mask", "mask")
    if probability is None or raw_mask is None or cleaned_mask is None:
        raise RuntimeError("segmentation result must expose probability, raw_mask, cleaned_mask")
    return (
        np.asarray(probability, dtype=np.float32),
        np.asarray(raw_mask, dtype=bool),
        np.asarray(cleaned_mask, dtype=bool),
    )


def _anchor_fields(result: Any) -> tuple[bool, np.ndarray | None, list[str], np.ndarray | None]:
    accepted = bool(_field(result, "accepted", default=False))
    centerline = _field(result, "centerline_xy", "centerline")
    reasons = _field(result, "rejection_reasons", "reasons", default=[])
    width = _field(
        result, "width_profile", "estimated_width", "width_profile_px", "width_px"
    )
    return (
        accepted,
        None if centerline is None else np.asarray(centerline, dtype=np.float64),
        [str(value) for value in reasons],
        None if width is None else np.asarray(width, dtype=np.float64),
    )


def _save_segmentation_montage(
    cases: Sequence[Mapping[str, Any]], path: Path, title: str
) -> None:
    fig, axes = plt.subplots(len(cases), 5, figsize=(15, max(2.2 * len(cases), 3)), squeeze=False)
    headings = ["raw", "probability", "raw mask", "cleaned mask", "boundary + manual"]
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)
    for row_axes, case in zip(axes, cases, strict=True):
        raw = case["_frame"]
        lo, hi = np.percentile(raw, (1, 99))
        row_axes[0].imshow(raw, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
        row_axes[1].imshow(case["_probability"], cmap="magma", vmin=0, vmax=1)
        row_axes[2].imshow(case["_raw_mask"], cmap="gray", vmin=0, vmax=1)
        row_axes[3].imshow(case["_cleaned_mask"], cmap="gray", vmin=0, vmax=1)
        row_axes[4].imshow(raw, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
        row_axes[4].contour(case["_cleaned_mask"], levels=[0.5], colors=["#e69f00"], linewidths=0.8)
        if len(case["_manual_xy"]):
            row_axes[4].plot(case["_manual_xy"][:, 0], case["_manual_xy"][:, 1], color="#00ffff", lw=1)
        for axis in row_axes:
            axis.axis("off")
        row_axes[0].set_ylabel(case["sample_id"], fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_anchor_montage(
    cases: Sequence[Mapping[str, Any]], path: Path, title: str
) -> None:
    columns = 4
    rows = max(1, math.ceil(len(cases) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.2 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, case in zip(axes.flat, cases):
        raw = case["_frame"]
        lo, hi = np.percentile(raw, (1, 99))
        axis.imshow(raw, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
        manual = case["_manual_xy"]
        if len(manual):
            axis.plot(manual[:, 0], manual[:, 1], color="#00ffff", lw=1.3)
        if case.get("_centerline") is not None:
            curve = case["_centerline"]
            axis.plot(curve[:, 0], curve[:, 1], color="#e69f00" if case["accepted"] else "#d55e00", lw=1.1)
        metric = case.get("complete_metrics") or {}
        suffix = f"; {metric['median_point_distance_px']:.1f}px" if metric else ""
        axis.set_title(f"{case['sample_id']}\n{'E' if case['accepted'] else 'H'}{suffix}", fontsize=8)
        axis.set_xlim(0, raw.shape[1]); axis.set_ylim(raw.shape[0], 0); axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_timelines(cases: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, max(0.35 * len(cases), 4)), squeeze=False)
    for axis, case in zip(axes.flat, cases, strict=True):
        states = case["timeline_states"]
        colors = ["#009e73" if value == "E" else "#d55e00" for value in states]
        axis.scatter([-1, 0, 1], [0, 0, 0], c=colors, s=35)
        axis.set_xlim(-1.4, 1.4); axis.set_ylim(-0.5, 0.5); axis.axis("off")
        axis.text(-1.35, 0, case["sample_id"], ha="left", va="center", fontsize=6)
    fig.suptitle("Annotated ±1 mask-anchor timeline (E accepted; H rejected)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _public_case(case: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--exp-001-dir", type=Path)
    parser.add_argument("--exp-002-dir", type=Path)
    parser.add_argument("--allow-existing-empty", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text())
    annotation_path = (args.annotations or Path(config["inputs"]["annotations"])).resolve(strict=True)
    manifest_value = args.manifest or REPO_ROOT / config["inputs"]["manifest"]
    manifest_path = manifest_value.resolve(strict=True)
    exp_001_dir = args.exp_001_dir or REPO_ROOT / config["outputs"]["exp_smc_001_dir"]
    exp_002_dir = args.exp_002_dir or REPO_ROOT / config["outputs"]["exp_smc_002_dir"]
    _safe_output_dir(exp_001_dir, args.allow_existing_empty)
    _safe_output_dir(exp_002_dir, args.allow_existing_empty)

    # Runtime-only imports keep pure gate tests independent of API landing time.
    from worm_pose_gen.segmentation import SoftForegroundConfig, segment_soft_foreground
    from worm_pose_gen.anchors import AnchorConfig, extract_mask_anchor, render_centerline_mask

    segment_config = _construct_config(SoftForegroundConfig, config["soft_foreground_config"])
    anchor_config = _construct_config(AnchorConfig, config["anchor_config"])
    rows = _verified_annotations(manifest_path, annotation_path, config)
    half_window = int(config["evaluation"]["timeline_half_window"])
    frames = _read_windows(rows, half_window)
    # No destination is mutated until API/config/source preflight has passed.
    _initialize_output_dir(exp_001_dir)
    _initialize_output_dir(exp_002_dir)

    segmentation: dict[tuple[str, int], tuple[Any, float]] = {}
    for key, frame in frames.items():
        start = time.perf_counter()
        result = segment_soft_foreground(frame, segment_config)
        segmentation[key] = (result, time.perf_counter() - start)

    exp_001_cases: list[dict[str, Any]] = []
    exp_002_cases: list[dict[str, Any]] = []
    central_anchor_runtime = 0.0
    for source, annotation in rows:
        recording = str(source["recording"]); index = int(source["frame_index"])
        frame = frames[(recording, index)]
        seg_result, segment_seconds = segmentation[(recording, index)]
        probability, raw_mask, cleaned_mask = _segmentation_arrays(seg_result)
        qc = _mask_qc(cleaned_mask)
        trace = (
            resample_polyline(annotation.points_xy, int(config["evaluation"]["trace_resample_points"]))
            if annotation.trace_state != "not_identifiable" else np.empty((0, 2))
        )
        evidence = annotation.trace_state != "not_identifiable"
        terminal_points = int(config["evaluation"]["terminal_points"])
        if evidence:
            contained = _sample_at_points(cleaned_mask, trace).astype(float)
            soft = _sample_at_points(probability, trace).astype(float)
            terminal = np.concatenate((contained[:terminal_points], contained[-terminal_points:]))
            distance = _nearest_mask_distances(cleaned_mask, trace)
            containment = float(contained.mean())
            terminal_containment = float(terminal.mean())
            soft_mean = float(soft.mean())
            median_distance = float(np.median(distance))
        else:
            containment = terminal_containment = soft_mean = median_distance = None
        neighbor_area_change: list[float] = []
        neighbor_centroid_change: list[float] = []
        for neighbor in (index - 1, index + 1):
            if (recording, neighbor) not in segmentation:
                continue
            _, _, neighbor_mask = _segmentation_arrays(segmentation[(recording, neighbor)][0])
            neighbor_qc = _mask_qc(neighbor_mask)
            denominator = max(qc["cleaned_mask_area_px"], 1)
            neighbor_area_change.append(abs(neighbor_qc["cleaned_mask_area_px"] - qc["cleaned_mask_area_px"]) / denominator)
            if qc["centroid_xy"][0] is not None and neighbor_qc["centroid_xy"][0] is not None:
                neighbor_centroid_change.append(float(np.linalg.norm(np.asarray(qc["centroid_xy"]) - np.asarray(neighbor_qc["centroid_xy"]))))
        exp_001_cases.append({
            "sample_id": annotation.sample_id, "recording": recording, "frame_index": index,
            "selection_stratum": source["selection_stratum"], "trace_state": annotation.trace_state,
            "used_as_evidence": evidence, "cleaned_mask_trace_containment": containment,
            "terminal_containment": terminal_containment,
            "terminal_omission_fraction": None if terminal_containment is None else 1.0 - terminal_containment,
            "soft_probability_on_trace": soft_mean,
            "median_nearest_cleaned_mask_distance_px": median_distance,
            **qc, "adjacent_area_relative_change": neighbor_area_change,
            "adjacent_centroid_displacement_px": neighbor_centroid_change,
            "segmentation_runtime_seconds": segment_seconds,
            "segmentation_qc": _field(seg_result, "qc", default={}),
            "proxy_metric_label": (
                "manual-visible-trace-derived, including terminal samples; "
                "not manual-mask Dice/IoU or anatomical hidden-end evidence"
            ),
            "_frame": frame, "_probability": probability, "_raw_mask": raw_mask,
            "_cleaned_mask": cleaned_mask, "_manual_xy": annotation.points_xy,
        })

        start = time.perf_counter()
        anchor_result = extract_mask_anchor(
            cleaned_mask, probability=probability, config=anchor_config
        )
        central_anchor_runtime += time.perf_counter() - start
        accepted, centerline, reasons, width = _anchor_fields(anchor_result)
        complete_metrics = None; visible_metrics = None; render_iou = None
        if accepted and centerline is None:
            raise RuntimeError("accepted anchor has no centerline")
        if centerline is not None and annotation.is_complete:
            complete_metrics = complete_curve_metrics(centerline, annotation.points_xy)
        elif centerline is not None and annotation.trace_state == "truncated":
            visible_metrics = visible_trace_metrics(centerline, annotation.points_xy)
        if accepted and centerline is not None and width is not None:
            rendered_value = _field(anchor_result, "rendered_mask", default=None)
            rendered = (
                np.asarray(rendered_value, dtype=bool)
                if rendered_value is not None
                else _render_anchor_mask(
                    render_centerline_mask, centerline, width, cleaned_mask.shape
                )
            )
            render_iou = _iou(rendered, cleaned_mask)
        exp_002_cases.append({
            "sample_id": annotation.sample_id, "recording": recording, "frame_index": index,
            "selection_stratum": source["selection_stratum"], "trace_state": annotation.trace_state,
            "accepted": accepted, "rejection_reasons": reasons,
            "complete_metrics": complete_metrics, "visible_metrics": visible_metrics,
            "width_profile_px": None if width is None else width.tolist(),
            "mask_render_iou_trace_width_proxy": render_iou,
            "topology_qc": _field(anchor_result, "topology_qc", "qc", default={}),
            "quality_score": _field(anchor_result, "quality_score", default=None),
            "_frame": frame, "_manual_xy": annotation.points_xy, "_centerline": centerline,
        })

    # Timelines evaluate all declared ±1 frames, not only annotated centers.
    timeline_anchor_start = time.perf_counter()
    for case in exp_002_cases:
        states: list[str] = []
        for neighbor in (case["frame_index"] - 1, case["frame_index"], case["frame_index"] + 1):
            _, _, mask = _segmentation_arrays(segmentation[(case["recording"], neighbor)][0])
            neighbor_probability, _, _ = _segmentation_arrays(
                segmentation[(case["recording"], neighbor)][0]
            )
            neighbor_anchor = extract_mask_anchor(
                mask, probability=neighbor_probability, config=anchor_config
            )
            states.append(
                "E" if bool(_field(neighbor_anchor, "accepted", default=False)) else "H"
            )
        case["timeline_states"] = states
    timeline_anchor_seconds = time.perf_counter() - timeline_anchor_start

    summary_001 = summarize_segmentation_rows(exp_001_cases)
    summary_002 = summarize_anchor_rows(exp_002_cases)
    gate_001 = evaluate_exp_smc_001_gate(summary_001, config["exp_smc_001_gate"])
    gate_002 = evaluate_exp_smc_002_gate(summary_002, config["exp_smc_002_gate"])
    shared_inputs = {
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "annotations": str(annotation_path), "annotations_sha256": _sha256(annotation_path),
        "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "protected_2025_holdout_opened": False,
    }
    metrics_001 = {
        "schema_version": 1, "experiment": "EXP-SMC-001", "method": "classical_soft_dark_ridge_baseline",
        "manual_mask_truth_available": False, "inputs": shared_inputs, "summary": summary_001,
        "gate": gate_001, "runtime": {
            "serial_cpu_seconds_central_30": float(sum(case["segmentation_runtime_seconds"] for case in exp_001_cases)),
            "serial_cpu_seconds_all_unique_central_and_adjacent_frames": float(
                sum(value[1] for value in segmentation.values())
            ),
            "unique_central_and_adjacent_frames": len(segmentation),
            "scope": "in_memory_serial_cpu_excludes_HDF5_read_and_plotting",
        }, "per_case": [_public_case(case) for case in exp_001_cases],
    }
    metrics_002 = {
        "schema_version": 1, "experiment": "EXP-SMC-002", "input": "EXP-SMC-001 cleaned masks",
        "inputs": shared_inputs, "summary": summary_002, "gate": gate_002,
        "runtime": {
            "serial_cpu_seconds_central_30": central_anchor_runtime,
            "serial_cpu_seconds_timeline_reextraction_90": timeline_anchor_seconds,
            "scope": "serial_CPU_cleaned_mask_to_anchor",
        },
        "per_case": [_public_case(case) for case in exp_002_cases],
        "evidence_boundary": {"complete_orientation": "symmetric", "truncated": "visible-trace diagnostic only", "coverage_role": "secondary"},
    }
    (exp_001_dir / "metrics.json").write_text(json.dumps(metrics_001, indent=2) + "\n")
    (exp_002_dir / "metrics.json").write_text(json.dumps(metrics_002, indent=2) + "\n")

    identifiable = [case for case in exp_001_cases if case["used_as_evidence"]]
    worst = sorted(identifiable, key=lambda case: (case["cleaned_mask_trace_containment"], -case["median_nearest_cleaned_mask_distance_px"]))[: int(config["evaluation"]["worst_review_cases"])]
    rng = np.random.default_rng(int(config["seed"]))
    random_cases = [identifiable[index] for index in sorted(rng.choice(len(identifiable), size=min(int(config["evaluation"]["random_review_cases"]), len(identifiable)), replace=False))]
    _save_segmentation_montage(random_cases, exp_001_dir / "figures/random_cases.png", "EXP-SMC-001 deterministic random trace-proxy cases")
    _save_segmentation_montage(worst, exp_001_dir / "figures/worst_cases.png", "EXP-SMC-001 worst trace containment cases")

    accepted_cases = [case for case in exp_002_cases if case["accepted"]]
    rejected_cases = [case for case in exp_002_cases if not case["accepted"]]
    worst_accepted = sorted(accepted_cases, key=lambda case: (case.get("complete_metrics") or {}).get("median_point_distance_px", -1), reverse=True)[:8]
    review_count = int(config["evaluation"]["threshold_review_cases"])
    threshold_nearest = (
        sorted(accepted_cases, key=lambda case: float(case["quality_score"] or 0.0))[
            : review_count // 2
        ]
        + sorted(
            rejected_cases,
            key=lambda case: float(case["quality_score"] or 0.0),
            reverse=True,
        )[: review_count - review_count // 2]
    )
    _save_anchor_montage(accepted_cases[:12], exp_002_dir / "figures/accepted.png", "EXP-SMC-002 accepted mask anchors")
    _save_anchor_montage(rejected_cases[:12], exp_002_dir / "figures/rejected.png", "EXP-SMC-002 rejected mask anchors")
    _save_anchor_montage(worst_accepted, exp_002_dir / "figures/worst_accepted.png", "EXP-SMC-002 worst accepted anchors")
    _save_anchor_montage(threshold_nearest, exp_002_dir / "figures/threshold_nearest.png", "EXP-SMC-002 quality-score threshold-nearest review")
    _save_timelines(exp_002_cases, exp_002_dir / "figures/annotated_windows_timeline.png")
    print(json.dumps({"EXP-SMC-001": gate_001, "EXP-SMC-002": gate_002}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
