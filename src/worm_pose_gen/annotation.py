"""Tier-A annotation validation and inter-annotator agreement metrics.

The annotation format deliberately keeps human evidence independent of model
outputs.  Traces are variable-density polylines in original-image pixel-center
coordinates.  Complete traces are resampled only for evaluation; the locked
source vertices remain unchanged in the annotation record.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

SCHEMA_VERSION = "1.0.0"
SUPPORT_STATES = frozenset(
    {"supported", "occluded_in_fov", "outside_fov", "not_identifiable"}
)
HEAD_TAIL_STATES = frozenset({"start_is_head", "start_is_tail", "ambiguous"})
COMPLETE_STATES = frozenset({"complete", "truncated", "not_identifiable"})


@dataclass(frozen=True)
class ValidatedAnnotation:
    """Validated human trace used by EXP-001 agreement evaluation."""

    sample_id: str
    annotation_id: str
    annotator_id: str
    points_xy: FloatArray
    support_state: tuple[str, ...]
    head_tail_state: str
    trace_state: str
    worm_width_px: float | None
    difficulty: tuple[str, ...]
    outside_fov_at_start: bool
    outside_fov_at_end: bool

    @property
    def is_complete(self) -> bool:
        return self.trace_state == "complete" and all(
            value in {"supported", "occluded_in_fov"} for value in self.support_state
        )


def _required_text(record: Mapping[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def validate_annotation(
    record: Mapping[str, Any], *, image_height: int, image_width: int
) -> ValidatedAnnotation:
    """Validate one immutable human annotation record.

    Supported and in-FOV occluded vertices must have finite coordinates inside
    the half-open image bounds.  ``outside_fov`` is a semantic terminal marker,
    not an invented hidden coordinate, and therefore uses ``[null, null]``.
    """

    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")
    sample_id = _required_text(record, "sample_id")
    annotation_id = _required_text(record, "annotation_id")
    annotator_id = _required_text(record, "annotator_id")
    _required_text(record, "tool_name")
    _required_text(record, "tool_version")
    _required_text(record, "started_at_utc")
    _required_text(record, "completed_at_utc")
    for field in (
        "configured_source_path", "resolved_source_path", "source_dataset_path",
        "split_role", "selection_stratum", "annotation_view", "timestamp_mapping",
    ):
        _required_text(record, field)
    for field in ("source_size_bytes", "source_mtime_ns", "frame_index"):
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    parent = record.get("parent_annotation_id")
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        raise ValueError("parent_annotation_id must be null or a nonempty string")
    temporal = record.get("temporal_window_indices")
    if (
        not isinstance(temporal, Sequence) or isinstance(temporal, (str, bytes))
        or not temporal or any(not isinstance(value, int) or value < 0 for value in temporal)
    ):
        raise ValueError("temporal_window_indices must contain nonnegative integers")
    overlays = record.get("annotation_overlays")
    if not isinstance(overlays, Sequence) or isinstance(overlays, (str, bytes)):
        raise ValueError("annotation_overlays must be a sequence")
    timestamp = record.get("timestamp_raw")
    if timestamp is not None and (
        not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
    ):
        raise ValueError("timestamp_raw must be null or a finite number")

    head_tail_state = record.get("head_tail_state")
    if head_tail_state not in HEAD_TAIL_STATES:
        raise ValueError(f"head_tail_state must be one of {sorted(HEAD_TAIL_STATES)}")
    trace_state = record.get("trace_state")
    if trace_state not in COMPLETE_STATES:
        raise ValueError(f"trace_state must be one of {sorted(COMPLETE_STATES)}")

    raw_vertices = record.get("vertices")
    if not isinstance(raw_vertices, Sequence) or isinstance(raw_vertices, (str, bytes)):
        raise ValueError("vertices must be a sequence")
    if trace_state != "not_identifiable" and len(raw_vertices) < 2:
        raise ValueError("vertices must contain at least two entries")
    if trace_state == "not_identifiable" and len(raw_vertices):
        raise ValueError("a not_identifiable annotation must not invent vertices")
    points: list[tuple[float, float]] = []
    states: list[str] = []
    for index, vertex in enumerate(raw_vertices):
        if not isinstance(vertex, Mapping):
            raise ValueError(f"vertex {index} must be an object")
        state = vertex.get("support_state")
        if state not in SUPPORT_STATES:
            raise ValueError(
                f"vertex {index} support_state must be one of {sorted(SUPPORT_STATES)}"
            )
        raw_xy = vertex.get("xy")
        if state == "outside_fov":
            if raw_xy not in (None, [None, None], (None, None)):
                raise ValueError("outside_fov vertices must not invent hidden coordinates")
            if index not in (0, len(raw_vertices) - 1):
                raise ValueError("outside_fov may only be a terminal marker")
            states.append(state)
            continue
        try:
            xy = np.asarray(raw_xy, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"vertex {index} xy must contain two finite numbers") from error
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError(f"vertex {index} xy must contain two finite numbers")
        x, y = float(xy[0]), float(xy[1])
        if not (0 <= x < image_width and 0 <= y < image_height):
            raise ValueError(f"vertex {index} coordinate lies outside the image")
        points.append((x, y))
        states.append(str(state))

    points_xy = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    if trace_state != "not_identifiable":
        if len(points_xy) < 2:
            raise ValueError("an identifiable annotation needs at least two coordinate vertices")
        segment = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
        if np.any(segment <= 0):
            raise ValueError("consecutive coordinate vertices must be distinct")
    if trace_state == "complete" and any(value == "outside_fov" for value in states):
        raise ValueError("a complete trace cannot contain outside_fov markers")
    if trace_state == "truncated" and not any(value == "outside_fov" for value in states):
        raise ValueError("a truncated trace requires an outside_fov terminal marker")

    raw_width = record.get("worm_width_px")
    worm_width_px: float | None
    if raw_width is None:
        worm_width_px = None
    else:
        worm_width_px = float(raw_width)
        if not math.isfinite(worm_width_px) or worm_width_px <= 0:
            raise ValueError("worm_width_px must be finite and positive")
    raw_difficulty = record.get("difficulty", [])
    if not isinstance(raw_difficulty, Sequence) or isinstance(raw_difficulty, (str, bytes)):
        raise ValueError("difficulty must be a sequence of strings")
    difficulty = tuple(sorted({_required_text({"value": value}, "value") for value in raw_difficulty}))

    return ValidatedAnnotation(
        sample_id=sample_id,
        annotation_id=annotation_id,
        annotator_id=annotator_id,
        points_xy=points_xy,
        support_state=tuple(states),
        head_tail_state=str(head_tail_state),
        trace_state=str(trace_state),
        worm_width_px=worm_width_px,
        difficulty=difficulty,
        outside_fov_at_start=bool(states and states[0] == "outside_fov"),
        outside_fov_at_end=bool(states and states[-1] == "outside_fov"),
    )


def resample_polyline(points_xy: NDArray[np.generic], num_points: int = 100) -> FloatArray:
    """Arc-length-resample a nondegenerate polyline without moving endpoints."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("points_xy must have shape [N>=2, 2]")
    if num_points < 2 or not np.isfinite(points).all():
        raise ValueError("num_points must be >=2 and points must be finite")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segment <= 0):
        raise ValueError("consecutive points must be distinct")
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    target = np.linspace(0.0, arc[-1], num_points)
    result = np.column_stack(
        (np.interp(target, arc, points[:, 0]), np.interp(target, arc, points[:, 1]))
    )
    result[0], result[-1] = points[0], points[-1]
    return result


def _tangent_angle(points_xy: FloatArray) -> FloatArray:
    difference = np.empty_like(points_xy)
    difference[0] = points_xy[1] - points_xy[0]
    difference[-1] = points_xy[-1] - points_xy[-2]
    difference[1:-1] = points_xy[2:] - points_xy[:-2]
    return np.arctan2(difference[:, 1], difference[:, 0])


def _wrap_angle(value: FloatArray) -> FloatArray:
    return np.remainder(value + np.pi, 2 * np.pi) - np.pi


def _length(points_xy: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points_xy, axis=0), axis=1).sum())


def annotation_pair_metrics(
    first: ValidatedAnnotation,
    second: ValidatedAnnotation,
    *,
    num_points: int = 100,
    symmetric_orientation: bool = True,
    allow_same_annotator: bool = False,
) -> dict[str, Any]:
    """Compute metrics for two independently produced complete traces.

    The normal EXP-001 comparison requires different annotators. A delayed,
    overlay-blind repeat by one annotator can opt into ``allow_same_annotator``
    to measure intra-annotator repeatability instead. These quantities must be
    named and reported separately.
    """

    if first.sample_id != second.sample_id:
        raise ValueError("annotations must refer to the same sample_id")
    if first.annotator_id == second.annotator_id and not allow_same_annotator:
        raise ValueError("agreement requires two different annotators")
    if not first.is_complete or not second.is_complete:
        raise ValueError("pointwise agreement requires two complete traces")
    a = resample_polyline(first.points_xy, num_points)
    b = resample_polyline(second.points_xy, num_points)
    first_reversed = False
    reversed_second = False
    anatomy_known = (
        first.head_tail_state != "ambiguous" and second.head_tail_state != "ambiguous"
    )
    if anatomy_known:
        first_reversed = first.head_tail_state == "start_is_tail"
        reversed_second = second.head_tail_state == "start_is_tail"
        if first_reversed:
            a = a[::-1]
        if reversed_second:
            b = b[::-1]
    elif symmetric_orientation:
        forward = np.linalg.norm(a - b, axis=1)
        reverse = np.linalg.norm(a - b[::-1], axis=1)
        reversed_second = bool(reverse.mean() < forward.mean())
    if reversed_second and not anatomy_known:
        b = b[::-1]
    point_distance = np.linalg.norm(a - b, axis=1)
    angle_difference = np.abs(_wrap_angle(_tangent_angle(a) - _tangent_angle(b)))
    endpoint_distance = np.asarray(
        (np.linalg.norm(a[0] - b[0]), np.linalg.norm(a[-1] - b[-1])), dtype=np.float64
    )
    length_a, length_b = _length(a), _length(b)
    widths = [value for value in (first.worm_width_px, second.worm_width_px) if value]
    width = float(np.mean(widths)) if widths else None
    first_states = tuple(value for value in first.support_state if value != "outside_fov")
    second_states = tuple(value for value in second.support_state if value != "outside_fov")
    support_a = _resample_categorical(first.points_xy, first_states, num_points)
    support_b = _resample_categorical(second.points_xy, second_states, num_points)
    if first_reversed:
        support_a = support_a[::-1]
    if reversed_second:
        support_b = support_b[::-1]
    return {
        "sample_id": first.sample_id,
        "annotators": [first.annotator_id, second.annotator_id],
        "first_trace_reversed": first_reversed,
        "second_trace_reversed": reversed_second,
        "point_distance_px": point_distance,
        "median_point_distance_px": float(np.median(point_distance)),
        "p95_point_distance_px": float(np.percentile(point_distance, 95)),
        "mean_tangent_angle_error_deg": float(np.rad2deg(angle_difference).mean()),
        "tangent_angle_error_deg": np.rad2deg(angle_difference),
        "endpoint_distance_px": endpoint_distance,
        "body_length_px": [length_a, length_b],
        "absolute_body_length_disagreement_px": abs(length_a - length_b),
        "relative_body_length_disagreement": abs(length_a - length_b)
        / max(0.5 * (length_a + length_b), np.finfo(np.float64).eps),
        "worm_width_px": width,
        "support_state_agreement_fraction": float(np.mean(support_a == support_b)),
        "median_point_distance_body_widths": (
            float(np.median(point_distance) / width) if width is not None else None
        ),
        "difficulty": sorted(set(first.difficulty) | set(second.difficulty)),
    }


def _resample_categorical(
    points_xy: FloatArray, states: Sequence[str], num_points: int
) -> NDArray[np.str_]:
    if len(points_xy) != len(states):
        raise ValueError("coordinate support states must align with coordinate vertices")
    segment = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    target = np.linspace(0.0, arc[-1], num_points)
    right = np.searchsorted(arc, target, side="left").clip(0, len(arc) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(arc[right] - target) < np.abs(target - arc[left])
    nearest = np.where(choose_right, right, left)
    return np.asarray(states, dtype=str)[nearest]


def annotation_semantic_pair_metrics(
    first: ValidatedAnnotation,
    second: ValidatedAnnotation,
    *,
    allow_same_annotator: bool = False,
) -> dict[str, Any]:
    """Agreement available even when hidden anatomy has no coordinates."""

    if first.sample_id != second.sample_id:
        raise ValueError("semantic agreement requires one sample")
    if first.annotator_id == second.annotator_id and not allow_same_annotator:
        raise ValueError("semantic agreement requires two annotators")
    if len(first.points_xy) < 2 or len(second.points_xy) < 2:
        return {
            "sample_id": first.sample_id,
            "trace_state_agreement": first.trace_state == second.trace_state,
            "head_tail_state_agreement": first.head_tail_state == second.head_tail_state,
            "first_head_tail_state": first.head_tail_state,
            "second_head_tail_state_aligned": second.head_tail_state,
            "truncation_end_agreement": (
                first.outside_fov_at_start, first.outside_fov_at_end
            ) == (second.outside_fov_at_start, second.outside_fov_at_end),
            "first_truncation_ends": [first.outside_fov_at_start, first.outside_fov_at_end],
            "second_truncation_ends_aligned": [
                second.outside_fov_at_start, second.outside_fov_at_end
            ],
            "second_trace_reversed_for_semantics": False,
        }
    a = resample_polyline(first.points_xy, 100)
    b = resample_polyline(second.points_xy, 100)
    reverse = bool(np.linalg.norm(a - b[::-1], axis=1).mean() < np.linalg.norm(a - b, axis=1).mean())
    second_orientation = second.head_tail_state
    if reverse:
        second_orientation = {
            "start_is_head": "start_is_tail",
            "start_is_tail": "start_is_head",
            "ambiguous": "ambiguous",
        }[second_orientation]
    second_truncation = (second.outside_fov_at_start, second.outside_fov_at_end)
    if reverse:
        second_truncation = second_truncation[::-1]
    return {
        "sample_id": first.sample_id,
        "trace_state_agreement": first.trace_state == second.trace_state,
        "head_tail_state_agreement": first.head_tail_state == second_orientation,
        "first_head_tail_state": first.head_tail_state,
        "second_head_tail_state_aligned": second_orientation,
        "truncation_end_agreement": (
            first.outside_fov_at_start, first.outside_fov_at_end
        ) == second_truncation,
        "first_truncation_ends": [first.outside_fov_at_start, first.outside_fov_at_end],
        "second_truncation_ends_aligned": list(second_truncation),
        "second_trace_reversed_for_semantics": reverse,
    }


def bootstrap_interval(
    values: NDArray[np.generic],
    *,
    statistic: str = "median",
    resamples: int = 2_000,
    seed: int = 20260818,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Deterministic frame-level percentile bootstrap interval."""

    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or not len(sample) or not np.isfinite(sample).all():
        raise ValueError("values must be a nonempty finite one-dimensional array")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must lie in (0,1)")
    if statistic == "median":
        reducer = np.median
    elif statistic == "mean":
        reducer = np.mean
    else:
        raise ValueError("statistic must be 'median' or 'mean'")
    generator = np.random.default_rng(seed)
    draw = generator.integers(0, len(sample), size=(resamples, len(sample)))
    distribution = reducer(sample[draw], axis=1)
    alpha = (1 - confidence) / 2
    return tuple(float(value) for value in np.quantile(distribution, (alpha, 1 - alpha)))
