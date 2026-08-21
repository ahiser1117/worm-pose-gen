#!/usr/bin/env python3
"""Run the frozen A1--A6 geometric pipeline on all 30 primary annotations.

The frame set and its order come from the already-materialized EXP-002 baseline
case list.  Geometry is completed before the annotation JSON is opened.  The
manual traces are then used only for orientation-symmetric complete-curve
metrics or one-way visible-trace diagnostics on naturally truncated cases.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

import build_boundary_notch_repair_experiment as boundary
import build_smooth_body_prior_experiment as smooth
from worm_pose_gen.anchors import extend_centerline_to_mask_boundary, skeleton_topology
from worm_pose_gen.annotation import resample_polyline, validate_annotation
from worm_pose_gen.classical import (
    ClassicalConfig,
    _prune_skeleton_endpoints,
    _thin,
    resample_centerline,
    robust_dark_ridge,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "experiments" / "scientific_exp_001_annotation" / "selection_manifest.json"
)
DEFAULT_BASELINE_METRICS = (
    PROJECT_ROOT / "experiments" / "scientific_exp_002_primary30_baselines" / "metrics.json"
)
DEFAULT_ANNOTATIONS = Path(
    "/temp_data4/alex/external_artifacts/annotations/worm_pose_tier_a_alex.json"
)
DEFAULT_PROXY_HDF5 = Path(
    "/temp_data4/alex/external_artifacts/datasets/"
    "worm_pose_gen/proxy_v1/proxy_labels.h5"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "final_algorithm_primary30"

CONTEXT_POINTS = 7
STEP_PX = 0.25
MAX_EXTENSION_PX = 80.0
CALLOUT_INDICES = frozenset({2, 22})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--proxy-hdf5", type=Path, default=DEFAULT_PROXY_HDF5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def curve_length(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def square_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Exact square dilation, using separable NumPy window reductions."""

    binary = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return binary.copy()
    size = 2 * radius + 1
    horizontal = np.lib.stride_tricks.sliding_window_view(
        np.pad(binary, ((0, 0), (radius, radius)), mode="constant"),
        size,
        axis=1,
    ).any(axis=-1)
    return np.lib.stride_tricks.sliding_window_view(
        np.pad(horizontal, ((radius, radius), (0, 0)), mode="constant"),
        size,
        axis=0,
    ).any(axis=-1)


def square_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Exact square erosion with the same false padding as the frozen code."""

    binary = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return binary.copy()
    size = 2 * radius + 1
    horizontal = np.lib.stride_tricks.sliding_window_view(
        np.pad(binary, ((0, 0), (radius, radius)), mode="constant"),
        size,
        axis=1,
    ).all(axis=-1)
    return np.lib.stride_tricks.sliding_window_view(
        np.pad(horizontal, ((radius, radius), (0, 0)), mode="constant"),
        size,
        axis=0,
    ).all(axis=-1)


def largest_component_rle(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return the largest 8-connected component using row-run union-find.

    This is algorithmically equivalent to the frozen pixel BFS but avoids
    visiting every pixel in Python for each candidate radius.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    runs: list[tuple[int, int, int, int]] = []
    parent: list[int] = []
    component_size: list[int] = []
    previous: list[tuple[int, int, int]] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            following = parent[node]
            parent[node] = root
            node = following
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        # Retaining the earlier run as root reproduces row-major tie behavior.
        if second_root < first_root:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        component_size[first_root] += component_size[second_root]

    for y, row in enumerate(binary):
        padded = np.pad(row, (1, 1), mode="constant")
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        stops = np.flatnonzero(padded[:-1] & ~padded[1:])
        current: list[tuple[int, int, int]] = []
        previous_cursor = 0
        for start, stop in zip(starts, stops, strict=True):
            node = len(parent)
            parent.append(node)
            component_size.append(int(stop - start))
            runs.append((y, int(start), int(stop), node))
            while (
                previous_cursor < len(previous)
                and previous[previous_cursor][1] < start
            ):
                previous_cursor += 1
            cursor = previous_cursor
            while cursor < len(previous) and previous[cursor][0] <= stop:
                previous_start, previous_stop, previous_node = previous[cursor]
                if previous_stop >= start and previous_start <= stop:
                    union(node, previous_node)
                cursor += 1
            current.append((int(start), int(stop), node))
        previous = current

    if not runs:
        return np.zeros_like(binary), 0, 0
    roots = sorted({find(node) for node in range(len(parent))})
    largest_root = max(roots, key=lambda root: (component_size[root], -root))
    result = np.zeros_like(binary)
    for y, start, stop, node in runs:
        if find(node) == largest_root:
            result[y, start:stop] = True
    return result, int(component_size[largest_root]), len(roots)


def initialization_skeleton(mask: np.ndarray) -> np.ndarray:
    """Reproduce the frozen initialization skeleton without finding its path."""

    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise RuntimeError("initialization mask is empty")
    pad = 3
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(mask.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(mask.shape[1], int(xx.max()) + pad + 1)
    skeleton_crop = _prune_skeleton_endpoints(_thin(mask[y0:y1, x0:x1]))
    skeleton = np.zeros_like(mask, dtype=bool)
    skeleton[y0:y1, x0:x1] = skeleton_crop
    return skeleton


def candidate_repair_for_sweep(
    baseline_mask: np.ndarray,
    original_component: np.ndarray,
    score: np.ndarray,
    radius: int,
    before_exterior: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen seal-then-fill rule without an unused longest-path fit."""

    closed = square_erode(square_dilate(baseline_mask, radius), radius)
    sealed = baseline_mask | closed
    bridge = sealed & ~baseline_mask
    sealed_exterior, _, _ = largest_component_rle(~sealed)
    pocket = (~sealed) & ~sealed_exterior
    repaired = sealed | pocket
    added = repaired & ~baseline_mask
    # Filling every non-exterior pocket leaves the same exterior component.
    after_exterior = sealed_exterior
    newly_enclosed = (~repaired) & ~after_exterior
    retained_exterior = float(
        np.logical_and(before_exterior, after_exterior).sum() / before_exterior.sum()
    )
    topology = skeleton_topology(initialization_skeleton(repaired))
    added_fraction = float(added.sum() / original_component.sum())
    reasons: list[str] = []
    if int(newly_enclosed.sum()):
        reasons.append("seals_exterior_background_into_a_hole")
    if retained_exterior < boundary.MIN_EXTERIOR_RETENTION:
        reasons.append("removes_too_much_exterior_background")
    if added_fraction > boundary.MAX_ADDED_FRACTION:
        reasons.append("adds_more_than_10_percent_of_original_component")
    if topology["endpoint_count"] != 2:
        reasons.append("not_two_endpoints")
    if topology["branch_pixels"] != 0:
        reasons.append("branches_remain")
    if topology["has_cycle"]:
        reasons.append("cycle_remains")
    metrics: dict[str, Any] = {
        "arm": "seal_then_fill",
        "radius_px": int(radius),
        "accepted_by_geometry_only_rule": not reasons,
        "rejection_reasons": reasons,
        "added_pixels": int(added.sum()),
        "bridge_pixels": int(bridge.sum()),
        "sealed_pocket_pixels": int(pocket.sum()),
        "bridge_local_darkness_z0_fraction": (
            float(np.mean(score[bridge] >= 0.0)) if np.any(bridge) else 0.0
        ),
        "bridge_original_threshold_z2_6_fraction": (
            float(np.mean(score[bridge] >= 2.6)) if np.any(bridge) else 0.0
        ),
        "pocket_local_darkness_z0_fraction": (
            float(np.mean(score[pocket] >= 0.0)) if np.any(pocket) else 0.0
        ),
        "pocket_original_threshold_z2_6_fraction": (
            float(np.mean(score[pocket] >= 2.6)) if np.any(pocket) else 0.0
        ),
        "added_fraction_of_original_component": added_fraction,
        "added_local_darkness_z0_fraction": (
            float(np.mean(score[added] >= 0.0)) if np.any(added) else 0.0
        ),
        "added_original_threshold_z2_6_fraction": (
            float(np.mean(score[added] >= 2.6)) if np.any(added) else 0.0
        ),
        "newly_enclosed_background_pixels": int(newly_enclosed.sum()),
        "exterior_background_retained_fraction": retained_exterior,
        "topology": topology,
    }
    return repaired, metrics


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.remainder(value + np.pi, 2 * np.pi) - np.pi


def _tangent(points: np.ndarray) -> np.ndarray:
    derivative = np.empty_like(points)
    derivative[0] = points[1] - points[0]
    derivative[-1] = points[-1] - points[-2]
    derivative[1:-1] = points[2:] - points[:-2]
    return np.arctan2(derivative[:, 1], derivative[:, 0])


def visible_trace_metrics(
    prediction_xy: np.ndarray, annotation_xy: np.ndarray, *, num_points: int = 100
) -> dict[str, Any]:
    """One-way coverage of a visible trace with unmatched anatomical positions."""

    prediction = resample_polyline(prediction_xy, num_points)
    target = resample_polyline(annotation_xy, num_points)
    segment = prediction[1:] - prediction[:-1]
    offset = target[:, None, :] - prediction[None, :-1, :]
    fraction = np.clip(
        np.einsum("tse,se->ts", offset, segment)
        / np.maximum(
            np.einsum("se,se->s", segment, segment), np.finfo(float).eps
        )[None, :],
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


def summary(values: Iterable[float]) -> dict[str, float | int | None]:
    sample = np.asarray(list(values), dtype=np.float64)
    if not len(sample):
        return {"n": 0, "median": None, "mean": None, "p95": None}
    return {
        "n": int(len(sample)),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "p95": float(np.percentile(sample, 95)),
    }


def load_case_records(
    manifest_path: Path, baseline_metrics_path: Path
) -> list[dict[str, Any]]:
    """Load the frozen 30-frame order without opening manual trace vertices."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protected_holdout_opened") is not False:
        raise RuntimeError("refusing a manifest that opens the protected holdout")
    records = {str(row["sample_id"]): row for row in manifest["records"]}
    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    cases = baseline["per_case"]["classical_ungated_diagnostic"]
    if len(cases) != 30:
        raise RuntimeError("expected exactly 30 frozen primary cases")
    ordered: list[dict[str, Any]] = []
    for annotation_index, case in enumerate(cases):
        sample_id = str(case["sample_id"])
        source = dict(records[sample_id])
        source["annotation_index"] = annotation_index
        ordered.append(source)
    if len({row["sample_id"] for row in ordered}) != 30:
        raise RuntimeError("primary case list contains duplicate sample IDs")
    return ordered


def read_frame(source: dict[str, Any], proxy_hdf5: Path) -> tuple[np.ndarray, str]:
    """Use the provenance-preserving cache when present, otherwise the raw source."""

    with h5py.File(proxy_hdf5, "r") as proxy:
        recording = str(source["recording"])
        if recording in proxy:
            group = proxy[recording]
            indices = np.asarray(group["accepted_frame_index"], dtype=np.int64)
            positions = np.flatnonzero(indices == int(source["frame_index"]))
            if len(positions) == 1:
                frame = np.asarray(group["accepted_image"][int(positions[0])], dtype=np.uint8)
                expected_shape = (
                    int(source["image_height"]), int(source["image_width"])
                )
                if frame.shape != expected_shape:
                    raise RuntimeError(
                        f"unexpected cached frame shape for {source['sample_id']}: {frame.shape}"
                    )
                return frame, "provenance_preserving_proxy_cache"

    path = Path(source["resolved_source_path"])
    stat = path.stat()
    if stat.st_size != int(source["source_size_bytes"]):
        raise RuntimeError(f"source size changed: {path}")
    if stat.st_mtime_ns != int(source["source_mtime_ns"]):
        raise RuntimeError(f"source mtime changed: {path}")
    with h5py.File(path, "r") as handle:
        frame = np.asarray(
            handle[str(source["source_dataset_path"])][int(source["frame_index"])],
            dtype=np.uint8,
        )
    expected_shape = (int(source["image_height"]), int(source["image_width"]))
    if frame.shape != expected_shape:
        raise RuntimeError(f"unexpected frame shape for {source['sample_id']}: {frame.shape}")
    return frame, "verified_raw_source"


def _failure(
    result: dict[str, Any], stage: str, reasons: list[str], started: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result.update(
        {
            "accepted": False,
            "failure_stage": stage,
            "failure_reasons": reasons,
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    return result, {}


def fit_case(
    payload: tuple[dict[str, Any], np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit A1--A6 using image geometry only; annotations are unavailable here."""

    # The frozen builders use square morphology and 8-connectivity.  These
    # replacements preserve those operations exactly while avoiding repeated
    # full-frame Python pixel scans during the 30-frame audit.
    boundary._dilate = square_dilate
    boundary._erode = square_erode
    boundary._largest_component = largest_component_rle
    smooth._largest_component = largest_component_rle

    source, frame = payload
    started = time.perf_counter()
    result: dict[str, Any] = {
        "annotation_index": int(source["annotation_index"]),
        "sample_id": str(source["sample_id"]),
        "recording": str(source["recording"]),
        "frame_index": int(source["frame_index"]),
        "selection_stratum": str(source["selection_stratum"]),
    }
    cfg = ClassicalConfig()

    try:
        score = robust_dark_ridge(frame, cfg)
        raw = score >= cfg.foreground_z
        closed = square_erode(square_dilate(raw, cfg.close_radius), cfg.close_radius)
        component, component_area, component_count = largest_component_rle(closed)
        if not component_area:
            return _failure(result, "section3", ["empty_largest_component"], started)
        baseline_mask, enclosed, enclosed_count = smooth.fill_enclosed_cavities(component)
    except Exception as error:  # pragma: no cover - recorded batch failure path
        return _failure(result, "section3", [f"{type(error).__name__}: {error}"], started)

    result["section3"] = {
        "component_area_px": int(component_area),
        "pre_keep_component_count": int(component_count),
        "enclosed_pixels_filled": int(enclosed.sum()),
        "enclosed_cavity_count": int(enclosed_count),
    }

    sweep: list[dict[str, Any]] = []
    accepted_candidates: dict[int, np.ndarray] = {}
    before_exterior, _, _ = largest_component_rle(~baseline_mask)
    for radius in boundary.SEARCH_RADII:
        try:
            repaired, metrics = candidate_repair_for_sweep(
                baseline_mask,
                component,
                score,
                radius,
                before_exterior,
            )
            sweep.append(metrics)
            if metrics["accepted_by_geometry_only_rule"]:
                accepted_candidates[int(radius)] = repaired
                # Radii are ascending and the frozen selection rule chooses the
                # first passing candidate, so later candidates cannot affect
                # the result.
                break
        except Exception as error:  # pragma: no cover - recorded batch failure path
            sweep.append(
                {
                    "radius_px": int(radius),
                    "accepted_by_geometry_only_rule": False,
                    "rejection_reasons": [f"{type(error).__name__}: {error}"],
                }
            )
    result["seal_then_fill_candidate_sweep"] = sweep
    if not accepted_candidates:
        reasons = Counter(
            reason
            for candidate in sweep
            for reason in candidate.get("rejection_reasons", [])
        )
        result["candidate_rejection_reason_counts"] = dict(sorted(reasons.items()))
        return _failure(
            result,
            "A1_geometry_selection",
            ["no_radius_3_through_12_passed_every_geometry_guard"],
            started,
        )

    selected_radius = min(accepted_candidates)
    selected_metrics = next(
        row for row in sweep if int(row["radius_px"]) == selected_radius
    )
    result["selected_repair"] = {
        "radius_px": selected_radius,
        "added_pixels": int(selected_metrics["added_pixels"]),
        "added_fraction_of_original_component": float(
            selected_metrics["added_fraction_of_original_component"]
        ),
        "exterior_background_retained_fraction": float(
            selected_metrics["exterior_background_retained_fraction"]
        ),
        "topology": selected_metrics["topology"],
    }

    try:
        body_metrics, body_arrays = boundary.evaluate_body_fit(
            accepted_candidates[selected_radius], component, score, cfg
        )
    except Exception as error:  # pragma: no cover - recorded batch failure path
        return _failure(result, "A4_A5_body_fit", [f"{type(error).__name__}: {error}"], started)
    result["a5_body"] = body_metrics
    if not body_metrics["accepted"]:
        return _failure(
            result,
            "A5_modeled_body_gate",
            list(body_metrics["rejection_reasons"]),
            started,
        )

    a5 = np.asarray(body_arrays["centerline"], dtype=np.float64)
    body_mask = np.asarray(body_arrays["body_mask"], dtype=bool)
    try:
        dense = extend_centerline_to_mask_boundary(
            a5,
            body_mask,
            context_points=CONTEXT_POINTS,
            step=STEP_PX,
            max_extension=MAX_EXTENSION_PX,
        )
        a6 = resample_centerline(dense, cfg.n_points)
        a5_length = curve_length(a5)
        dense_length = curve_length(dense)
        a6_length = curve_length(a6)
    except Exception as error:  # pragma: no cover - recorded batch failure path
        return _failure(
            result,
            "A6_endpoint_extension",
            [f"{type(error).__name__}: {error}"],
            started,
        )

    length_pass = bool(cfg.min_length <= a6_length <= cfg.max_length)
    result["a6_extension"] = {
        "context_points": CONTEXT_POINTS,
        "integration_step_px": STEP_PX,
        "maximum_extension_per_end_px": MAX_EXTENSION_PX,
        "a5_centerline_length_px": a5_length,
        "dense_curve_length_px": dense_length,
        "dense_extension_gain_px": dense_length - a5_length,
        "a6_centerline_length_px": a6_length,
        "a6_resampled_length_gain_px": a6_length - a5_length,
        "length_gate_allowed_px": [cfg.min_length, cfg.max_length],
        "length_gate_passed": length_pass,
    }
    if not length_pass:
        return _failure(
            result,
            "A6_length_gate",
            ["extended_centerline_outside_250_to_750_px"],
            started,
        )

    result.update(
        {
            "accepted": True,
            "failure_stage": None,
            "failure_reasons": [],
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    arrays = {
        "section3_component": component,
        "a5_body_mask": body_mask,
        "a5_centerline_xy": a5,
        "a6_centerline_xy": a6,
    }
    if int(source["annotation_index"]) in CALLOUT_INDICES:
        arrays["frame"] = frame
    return result, arrays


def fit_source(
    payload: tuple[dict[str, Any], Path]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read one verified frame in the worker, then run the annotation-blind fit."""

    source, proxy_hdf5 = payload
    frame, frame_source = read_frame(source, proxy_hdf5)
    result, arrays = fit_case((source, frame))
    result["frame_read_source"] = frame_source
    if int(source["annotation_index"]) in CALLOUT_INDICES:
        arrays["frame"] = frame
    return result, arrays


def fit_recording(
    payload: tuple[list[dict[str, Any]], Path]
) -> list[tuple[int, dict[str, Any], dict[str, np.ndarray]]]:
    """Read and fit one recording serially, keeping one raw HDF5 handle open."""

    sources, proxy_hdf5 = payload
    if not sources:
        return []
    first = sources[0]
    raw_path = Path(first["resolved_source_path"])
    stat = raw_path.stat()
    if stat.st_size != int(first["source_size_bytes"]):
        raise RuntimeError(f"source size changed: {raw_path}")
    if stat.st_mtime_ns != int(first["source_mtime_ns"]):
        raise RuntimeError(f"source mtime changed: {raw_path}")
    recording = str(first["recording"])
    results: list[tuple[int, dict[str, Any], dict[str, np.ndarray]]] = []
    print(json.dumps({"recording": recording, "stage": "proxy_open_start"}), flush=True)
    with h5py.File(proxy_hdf5, "r") as proxy:
        print(json.dumps({"recording": recording, "stage": "proxy_open_complete"}), flush=True)
        proxy_group = proxy.get(recording)
        proxy_positions: dict[int, int] = {}
        if proxy_group is not None:
            proxy_positions = {
                int(frame_index): position
                for position, frame_index in enumerate(
                    np.asarray(proxy_group["accepted_frame_index"], dtype=np.int64)
                )
            }
        raw: h5py.File | None = None
        try:
            for source in sources:
                if str(source["resolved_source_path"]) != str(raw_path):
                    raise RuntimeError("recording group mixes source paths")
                frame_index = int(source["frame_index"])
                index = int(source["annotation_index"])
                print(
                    json.dumps(
                        {
                            "recording": recording,
                            "annotation_index": index,
                            "frame_index": frame_index,
                            "stage": "frame_read_start",
                        }
                    ),
                    flush=True,
                )
                if frame_index in proxy_positions:
                    assert proxy_group is not None
                    frame = np.asarray(
                        proxy_group["accepted_image"][proxy_positions[frame_index]],
                        dtype=np.uint8,
                    )
                    frame_source = "provenance_preserving_proxy_cache"
                else:
                    if raw is None:
                        print(
                            json.dumps({"recording": recording, "stage": "raw_open_start"}),
                            flush=True,
                        )
                        raw = h5py.File(raw_path, "r")
                        print(
                            json.dumps({"recording": recording, "stage": "raw_open_complete"}),
                            flush=True,
                        )
                    dataset = raw[str(first["source_dataset_path"])]
                    frame = np.asarray(dataset[frame_index], dtype=np.uint8)
                    frame_source = "verified_raw_source"
                print(
                    json.dumps(
                        {
                            "recording": recording,
                            "annotation_index": index,
                            "stage": "frame_read_complete",
                            "source": frame_source,
                        }
                    ),
                    flush=True,
                )
                expected_shape = (
                    int(source["image_height"]), int(source["image_width"])
                )
                if frame.shape != expected_shape:
                    raise RuntimeError(
                        f"unexpected frame shape for {source['sample_id']}: {frame.shape}"
                    )
                result, arrays = fit_case((source, frame))
                result["frame_read_source"] = frame_source
                if index in CALLOUT_INDICES:
                    arrays["frame"] = frame
                results.append((index, result, arrays))
                print(
                    json.dumps(
                        {
                            "recording": recording,
                            "annotation_index": index,
                            "stage": "fit_complete",
                            "accepted": result["accepted"],
                            "failure_stage": result["failure_stage"],
                        }
                    ),
                    flush=True,
                )
        finally:
            if raw is not None:
                raw.close()
    return results


def score_predictions(
    cases: list[dict[str, Any]],
    arrays: dict[int, dict[str, np.ndarray]],
    annotation_path: Path,
) -> dict[int, np.ndarray]:
    """Open manual annotations only after every geometric prediction is fixed."""

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    raw_by_sample = {
        str(row["sample_id"]): row
        for row in payload["annotations"]
        if row.get("annotation_pass") == "primary"
    }
    annotation_points: dict[int, np.ndarray] = {}
    for case in cases:
        raw = raw_by_sample[case["sample_id"]]
        annotation = validate_annotation(raw, image_height=732, image_width=968)
        case["annotation"] = {
            "annotation_id": annotation.annotation_id,
            "trace_state": annotation.trace_state,
            "single_annotator_protocol": bool(raw.get("single_annotator_protocol")),
            "used_during_geometry_fit": False,
        }
        index = int(case["annotation_index"])
        annotation_points[index] = annotation.points_xy
        if index not in arrays or "a6_centerline_xy" not in arrays[index]:
            case["postfit_manual_audit"] = None
            continue
        a5 = arrays[index]["a5_centerline_xy"]
        a6 = arrays[index]["a6_centerline_xy"]
        if annotation.is_complete:
            case["postfit_manual_audit"] = {
                "metric_scope": "orientation_symmetric_complete_curve",
                "a5": smooth.complete_curve_metrics(a5, annotation.points_xy),
                "a6": smooth.complete_curve_metrics(a6, annotation.points_xy),
            }
        elif annotation.trace_state == "truncated":
            case["postfit_manual_audit"] = {
                "metric_scope": "one_way_visible_trace_to_predicted_curve",
                "a5": visible_trace_metrics(a5, annotation.points_xy),
                "a6": visible_trace_metrics(a6, annotation.points_xy),
            }
        else:
            case["postfit_manual_audit"] = None
    return annotation_points


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [case for case in cases if case["accepted"]]
    complete = [
        case for case in accepted
        if (case.get("postfit_manual_audit") or {}).get("metric_scope")
        == "orientation_symmetric_complete_curve"
    ]
    truncated = [
        case for case in accepted
        if (case.get("postfit_manual_audit") or {}).get("metric_scope")
        == "one_way_visible_trace_to_predicted_curve"
    ]
    failures = Counter(
        str(case["failure_stage"]) for case in cases if not case["accepted"]
    )
    radius_counts = Counter(
        int(case["selected_repair"]["radius_px"])
        for case in cases
        if "selected_repair" in case
    )
    accepted_by_trace = Counter(case["annotation"]["trace_state"] for case in accepted)
    total_by_trace = Counter(case["annotation"]["trace_state"] for case in cases)
    accepted_by_stratum = Counter(case["selection_stratum"] for case in accepted)
    total_by_stratum = Counter(case["selection_stratum"] for case in cases)

    def complete_stage(stage: str, metric: str) -> dict[str, float | int | None]:
        return summary(
            case["postfit_manual_audit"][stage][metric] for case in complete
        )

    def truncated_stage(stage: str, metric: str) -> dict[str, float | int | None]:
        return summary(
            case["postfit_manual_audit"][stage][metric] for case in truncated
        )

    return {
        "requested_frames": len(cases),
        "accepted_frames": len(accepted),
        "accepted_fraction": len(accepted) / len(cases),
        "failure_stage_counts": dict(sorted(failures.items())),
        "selected_radius_counts": {str(key): value for key, value in sorted(radius_counts.items())},
        "coverage_by_trace_state": {
            key: {"accepted": accepted_by_trace[key], "total": total_by_trace[key]}
            for key in sorted(total_by_trace)
        },
        "coverage_by_selection_stratum": {
            key: {"accepted": accepted_by_stratum[key], "total": total_by_stratum[key]}
            for key in sorted(total_by_stratum)
        },
        "complete_trace_matched_accepted_cases": {
            "frames": len(complete),
            "a5": {
                "per_frame_median_point_error_px": complete_stage(
                    "a5", "median_point_distance_px"
                ),
                "per_frame_mean_tangent_error_deg": complete_stage(
                    "a5", "mean_tangent_error_deg"
                ),
                "per_frame_mean_endpoint_error_px": complete_stage(
                    "a5", "mean_endpoint_error_px"
                ),
                "per_frame_body_length_error_px": complete_stage(
                    "a5", "body_length_error_px"
                ),
            },
            "a6": {
                "per_frame_median_point_error_px": complete_stage(
                    "a6", "median_point_distance_px"
                ),
                "per_frame_mean_tangent_error_deg": complete_stage(
                    "a6", "mean_tangent_error_deg"
                ),
                "per_frame_mean_endpoint_error_px": complete_stage(
                    "a6", "mean_endpoint_error_px"
                ),
                "per_frame_body_length_error_px": complete_stage(
                    "a6", "body_length_error_px"
                ),
            },
        },
        "truncated_trace_matched_accepted_cases": {
            "frames": len(truncated),
            "metric_scope": (
                "one-way visible-trace coverage only; no hidden anatomy or "
                "matched anatomical-position claim"
            ),
            "a5": {
                "per_frame_median_visible_distance_px": truncated_stage(
                    "a5", "median_visible_trace_distance_px"
                ),
                "per_frame_mean_visible_axis_error_deg": truncated_stage(
                    "a5", "mean_visible_trace_axis_error_deg"
                ),
            },
            "a6": {
                "per_frame_median_visible_distance_px": truncated_stage(
                    "a6", "median_visible_trace_distance_px"
                ),
                "per_frame_mean_visible_axis_error_deg": truncated_stage(
                    "a6", "mean_visible_trace_axis_error_deg"
                ),
            },
        },
        "runtime_seconds": summary(case["runtime_seconds"] for case in cases),
    }


def plot_summary(cases: list[dict[str, Any]], summary_metrics: dict[str, Any], path: Path) -> None:
    accepted = [case for case in cases if case["accepted"]]
    complete = [
        case for case in accepted
        if (case.get("postfit_manual_audit") or {}).get("metric_scope")
        == "orientation_symmetric_complete_curve"
    ]
    truncated = [
        case for case in accepted
        if (case.get("postfit_manual_audit") or {}).get("metric_scope")
        == "one_way_visible_trace_to_predicted_curve"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    stages = ["requested", "A1 selected", "A5 gate", "A6 final"]
    counts = [
        len(cases),
        sum("selected_repair" in case for case in cases),
        sum((case.get("a5_body") or {}).get("accepted", False) for case in cases),
        len(accepted),
    ]
    axes[0].bar(stages, counts, color=[smooth.GRAY, smooth.CYAN, smooth.ORANGE, smooth.GREEN])
    axes[0].set_ylim(0, 32)
    axes[0].set_ylabel("frames")
    axes[0].set_title("Fail-closed pipeline coverage")
    axes[0].tick_params(axis="x", labelrotation=18)
    for index, count in enumerate(counts):
        axes[0].text(index, count + 0.6, str(count), ha="center")

    for case in complete:
        audit = case["postfit_manual_audit"]
        axes[1].plot(
            [audit["a5"]["median_point_distance_px"], audit["a6"]["median_point_distance_px"]],
            [audit["a5"]["mean_endpoint_error_px"], audit["a6"]["mean_endpoint_error_px"]],
            marker="o",
            alpha=0.72,
        )
    axes[1].set_xlabel("median point error (px)")
    axes[1].set_ylabel("mean endpoint error (px)")
    axes[1].set_title(f"A5 to A6 on {len(complete)} accepted complete traces")

    for case in truncated:
        audit = case["postfit_manual_audit"]
        axes[2].scatter(
            audit["a5"]["median_visible_trace_distance_px"],
            audit["a6"]["median_visible_trace_distance_px"],
            color=smooth.ORANGE,
            alpha=0.8,
        )
    if truncated:
        maximum = max(
            max(
                case["postfit_manual_audit"][stage]["median_visible_trace_distance_px"]
                for case in truncated
                for stage in ("a5", "a6")
            ),
            1.0,
        )
        axes[2].plot([0, maximum], [0, maximum], linestyle="--", color=smooth.GRAY)
    axes[2].set_xlabel("A5 median visible distance (px)")
    axes[2].set_ylabel("A6 median visible distance (px)")
    axes[2].set_title(f"Endpoint effect on {len(truncated)} accepted truncated traces")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.suptitle(
        f"Frozen A1--A6 primary-30 audit: {summary_metrics['accepted_frames']}/30 final outputs"
    )
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_callouts(
    cases: list[dict[str, Any]],
    arrays: dict[int, dict[str, np.ndarray]],
    annotation_points: dict[int, np.ndarray],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3), constrained_layout=True)
    by_index = {int(case["annotation_index"]): case for case in cases}
    for axis, index in zip(axes, sorted(CALLOUT_INDICES), strict=True):
        case = by_index[index]
        data = arrays[index]
        frame = data["frame"]
        axis.imshow(frame, cmap="gray", vmin=80, vmax=225)
        if "section3_component" in data:
            mask = data["section3_component"]
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[..., :3] = to_rgb(smooth.CYAN)
            rgba[..., 3] = mask.astype(np.float32) * 0.22
            axis.imshow(rgba, interpolation="nearest")
            a5 = data["a5_centerline_xy"]
            a6 = data["a6_centerline_xy"]
            axis.plot(a5[:, 0], a5[:, 1], color="white", linewidth=2.1, label="A5")
            axis.plot(a6[:, 0], a6[:, 1], color=smooth.ORANGE, linewidth=2.3, label="A6")
        trace = annotation_points[index]
        if len(trace):
            axis.plot(trace[:, 0], trace[:, 1], color=smooth.MAGENTA, linewidth=2.1, label="visible trace")
        if case["accepted"]:
            audit = case["postfit_manual_audit"]["a6"]
            metric = f"visible median={audit['median_visible_trace_distance_px']:.2f} px"
        else:
            metric = f"rejected at {case['failure_stage']}"
        axis.set_title(
            f"annotation index {index}: {case['sample_id']}\n{case['annotation']['trace_state']}; {metric}"
        )
        axis.set_xlim(0, frame.shape[1] - 1)
        axis.set_ylim(frame.shape[0] - 1, 0)
        axis.set_axis_off()
        axis.legend(loc="lower right", framealpha=0.78)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_case_records(args.manifest, args.baseline_metrics)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        grouped.setdefault(str(source["recording"]), []).append(source)

    fitted: dict[int, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    if args.workers == 1:
        recording_results = [
            fit_recording((recording_sources, args.proxy_hdf5))
            for recording_sources in grouped.values()
        ]
    else:
        recording_results = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(grouped))) as executor:
            futures = [
                executor.submit(fit_recording, (recording_sources, args.proxy_hdf5))
                for recording_sources in grouped.values()
            ]
            for future in as_completed(futures):
                recording_results.append(future.result())
    for group_results in recording_results:
        for index, result, case_arrays in group_results:
            fitted[index] = (result, case_arrays)
            print(
                json.dumps(
                    {
                        "annotation_index": index,
                        "sample_id": result["sample_id"],
                        "accepted": result["accepted"],
                        "failure_stage": result["failure_stage"],
                    }
                ),
                flush=True,
            )

    cases = [fitted[index][0] for index in range(30)]
    arrays = {
        index: fitted[index][1] for index in range(30) if fitted[index][1]
    }
    annotation_points = score_predictions(cases, arrays, args.annotations)
    summary_metrics = summarize(cases)

    metrics = {
        "status": "frozen_A1_through_A6_primary30_development_audit",
        "evidence_boundary": {
            "single_annotator_primary_pass": True,
            "independent_validation": False,
            "parameters_frozen_from_annotation_index_5_frame_3420": True,
            "manual_trace_used_during_geometry_fit_or_selection": False,
            "protected_2025_holdout_opened": False,
            "complete_trace_metric": "orientation-symmetric 100-point correspondence",
            "truncated_trace_metric": (
                "one-way visible-trace coverage; no hidden anatomy or matched "
                "anatomical-position claim"
            ),
        },
        "inputs": {
            "manifest": str(args.manifest.resolve(strict=True)),
            "manifest_sha256": sha256_file(args.manifest),
            "baseline_case_order": str(args.baseline_metrics.resolve(strict=True)),
            "baseline_case_order_sha256": sha256_file(args.baseline_metrics),
            "provenance_preserving_proxy_hdf5": str(
                args.proxy_hdf5.resolve(strict=True)
            ),
            "provenance_preserving_proxy_hdf5_sha256": sha256_file(args.proxy_hdf5),
            "annotations": str(args.annotations.resolve(strict=True)),
            "annotations_sha256": sha256_file(args.annotations),
        },
        "frozen_parameters": {
            "foreground_z": ClassicalConfig().foreground_z,
            "raw_closing_radius_px": ClassicalConfig().close_radius,
            "candidate_close_radii_px": list(boundary.SEARCH_RADII),
            "max_added_fraction_of_original_component": boundary.MAX_ADDED_FRACTION,
            "minimum_exterior_background_retention": boundary.MIN_EXTERIOR_RETENTION,
            "required_initialization_topology": "2 endpoints, 0 branches, no cycle",
            "latent_angle_coefficients": 16,
            "containment_margin_px": 0.75,
            "radius_max_adjacent_change_px": 1.0,
            "max_full_width_px": 80.0,
            "modeled_body_area_allowed_px": [
                ClassicalConfig().min_area,
                boundary.MODELED_BODY_MAX_AREA,
            ],
            "centerline_length_allowed_px": [
                ClassicalConfig().min_length,
                ClassicalConfig().max_length,
            ],
            "endpoint_context_points": CONTEXT_POINTS,
            "endpoint_integration_step_px": STEP_PX,
            "maximum_extension_per_end_px": MAX_EXTENSION_PX,
        },
        "summary": summary_metrics,
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
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    prediction_arrays: dict[str, np.ndarray] = {}
    for index, data in arrays.items():
        if "a6_centerline_xy" not in data:
            continue
        prediction_arrays[f"annotation_{index:02d}_a5_centerline_xy"] = data[
            "a5_centerline_xy"
        ]
        prediction_arrays[f"annotation_{index:02d}_a6_centerline_xy"] = data[
            "a6_centerline_xy"
        ]
    np.savez_compressed(args.output_dir / "predictions.npz", **prediction_arrays)
    plot_summary(cases, summary_metrics, args.output_dir / "summary.png")
    plot_callouts(
        cases,
        arrays,
        annotation_points,
        args.output_dir / "annotations_2_22.png",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
