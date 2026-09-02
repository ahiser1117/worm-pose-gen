#!/usr/bin/env python3
"""Compare the frozen and edge-aware pipelines on the same 30 raw frames.

This remains an annotation-free operational stress test.  Flat-fielding and
field-of-view completion may improve coverage, but this script deliberately
does not claim anatomical accuracy.  Samples 09, 19, and 29 are known empty
frames and are always rejected, even if either geometric pipeline produces a
worm-like false positive.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import gc
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
from evaluate_final_geometry_primary30 import fit_case
from evaluate_final_geometry_unannotated30 import (
    DATASET_PATH,
    DEFAULT_RECORDINGS,
    FRAMES_PER_RECORDING,
    diagnostic_filename,
    numeric_summary,
    recording_records,
)
from worm_pose_gen.classical import ClassicalConfig, robust_dark_ridge
from worm_pose_gen.flat_field import FlatField, apply_flat_field, estimate_flat_field
from worm_pose_gen.fov_completion import (
    BoundaryTruncationResult,
    build_boundary_stable_skeleton,
    classify_boundary_truncation,
    complete_centerline_to_length,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "final_algorithm_edge_aware_unannotated30"
DEFAULT_BASELINE_METRICS = (
    PROJECT_ROOT / "docs" / "final_algorithm_unannotated30" / "metrics.json"
)
EXPECTED_NO_WORM_INDICES = frozenset({9, 19, 29})
DEFAULT_LENGTH_PRIORS_PX = {
    "2024-01-31-02": 697.6862355900387,
    "2023-08-22-01": 734.8633082624273,
    "2023-06-23-01": 733.4126642937758,
}
LENGTH_PRIOR_PROVENANCE = (
    "Per-recording longest frozen accepted A6 pose below 750 px. These are "
    "annotation-free lower-bound proxies, not independently measured anatomical truth."
)
FLAT_FIELD_SAMPLE_COUNT = 24


def _release_large_image_buffers() -> None:
    """Return per-sheet NumPy/Matplotlib buffers in long-running workers."""

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):  # pragma: no cover - non-glibc fallback
        pass


def _plot_in_disposable_child(
    baseline: dict[str, Any],
    baseline_arrays: dict[str, np.ndarray],
    edge: dict[str, Any],
    edge_arrays: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Isolate Matplotlib's large raster buffers from long-running fit workers."""

    if not hasattr(os, "fork"):  # pragma: no cover - non-POSIX fallback
        plot_comparison_sheet(baseline, baseline_arrays, edge, edge_arrays, path)
        return
    child = os.fork()
    if child == 0:  # pragma: no cover - exercised by the real batch
        try:
            plot_comparison_sheet(baseline, baseline_arrays, edge, edge_arrays, path)
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(child, 0)
    if status != 0:
        raise RuntimeError(f"comparison-sheet subprocess failed for {path.name}")


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
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--flat-field-frames", type=int, default=FLAT_FIELD_SAMPLE_COUNT)
    parser.add_argument(
        "--plot-range",
        default="0:30",
        metavar="START:STOP",
        help="half-open sample range whose diagnostic sheets are regenerated",
    )
    parser.add_argument(
        "--length-prior",
        action="append",
        default=[],
        metavar="RECORDING=PIXELS",
        help="override a per-recording full-length proxy",
    )
    return parser.parse_args()


def parse_length_priors(overrides: list[str]) -> dict[str, float]:
    result = dict(DEFAULT_LENGTH_PRIORS_PX)
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"invalid --length-prior {raw!r}; expected RECORDING=PIXELS")
        recording, value = raw.rsplit("=", 1)
        pixels = float(value)
        if not recording or not np.isfinite(pixels) or pixels <= 0:
            raise ValueError(f"invalid --length-prior {raw!r}")
        result[recording] = pixels
    return result


def parse_plot_range(raw: str) -> frozenset[int]:
    try:
        start_raw, stop_raw = raw.split(":", 1)
        start, stop = int(start_raw), int(stop_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("--plot-range must be START:STOP") from error
    if not 0 <= start <= stop <= 30:
        raise ValueError("--plot-range must satisfy 0 <= START <= STOP <= 30")
    return frozenset(range(start, stop))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def curve_length(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _truncation_summary(classification: BoundaryTruncationResult) -> dict[str, Any]:
    return {
        "state": classification.state,
        "edge_band_px": int(classification.edge_band_px),
        "contact_ends": list(classification.contact_ends),
        "contacts": [_jsonable(contact) for contact in classification.contacts],
        "diagnostics": _jsonable(classification.diagnostics),
    }


def _skeleton_summary(result: Any) -> dict[str, Any]:
    centerline = getattr(result, "centerline_xy", None)
    visible = getattr(result, "visible_centerline_xy", None)
    return {
        "centerline_available": centerline is not None,
        "full_path_length_px": None if centerline is None else curve_length(centerline),
        "visible_path_length_px": None if visible is None else curve_length(visible),
        "offset_xy": list(result.offset_xy),
        "diagnostics": _jsonable(result.diagnostics),
    }


def _field_diagnostics(field: FlatField, sample_indices: np.ndarray) -> dict[str, Any]:
    illumination = np.asarray(field.illumination, dtype=np.float64)
    gain = np.asarray(field.gain, dtype=np.float64)
    return {
        "sample_frame_indices": [int(value) for value in sample_indices],
        "temporal_quantile": 0.8,
        "spatial_radius_px": 31,
        "smoothing_passes": 2,
        "dark_level": float(field.dark_level),
        "reference_level": float(field.reference_level),
        "illumination": {
            "minimum": float(illumination.min()),
            "median": float(np.median(illumination)),
            "maximum": float(illumination.max()),
        },
        "gain": {
            "minimum": float(gain.min()),
            "median": float(np.median(gain)),
            "maximum": float(gain.max()),
            "p99": float(np.percentile(gain, 99)),
        },
    }


def _force_expected_no_worm(result: dict[str, Any]) -> None:
    result.update(
        {
            "accepted": False,
            "outcome": "no_worm_expected",
            "failure_stage": "expected_no_worm_rejection",
            "failure_reasons": ["sample_is_in_explicit_expected_no_worm_index_set"],
            "expected_no_worm": True,
        }
    )


def _curve_for_completion(arrays: dict[str, np.ndarray]) -> tuple[str, np.ndarray] | None:
    for name in (
        "a6_centerline_xy",
        "a5_centerline_xy",
        "a4_latent_midline_xy",
        "boundary_stable_centerline_xy",
    ):
        if name in arrays:
            return name, np.asarray(arrays[name], dtype=np.float64)
    return None


def _is_truncated(classification: BoundaryTruncationResult) -> bool:
    value = getattr(classification, "is_truncated", None)
    if value is not None:
        return bool(value)
    label = str(getattr(classification, "state", classification)).lower()
    return "truncated" in label and "not_truncated" not in label


def _is_completion_candidate(classification: BoundaryTruncationResult) -> bool:
    if _is_truncated(classification):
        return True
    return classification.state == "boundary_uncertain" and 1 <= len(
        classification.contact_ends
    ) <= 2


def fit_edge_case(
    source: dict[str, Any],
    raw_frame: np.ndarray,
    field: FlatField,
    target_length_px: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Apply correction, classify raw support, fit, then complete if censored."""

    cfg = ClassicalConfig()
    corrected = apply_flat_field(raw_frame, field, clip=(0.0, 255.0))
    score = robust_dark_ridge(corrected, cfg)
    raw_threshold = score >= cfg.foreground_z
    truncation = classify_boundary_truncation(
        raw_threshold,
        close_radius=cfg.close_radius,
        worm_radius_px=7.0,
    )
    stable_skeleton = None
    stable_skeleton_error = None
    try:
        stable_skeleton = build_boundary_stable_skeleton(raw_threshold, truncation)
    except (RuntimeError, ValueError) as error:
        stable_skeleton_error = f"{type(error).__name__}: {error}"
    result, arrays = fit_case((source, corrected), retain_diagnostics=True)
    result["flat_field_applied"] = True
    result["truncation"] = _truncation_summary(truncation)
    result["boundary_stable_skeleton"] = (
        {"error": stable_skeleton_error}
        if stable_skeleton is None
        else _skeleton_summary(stable_skeleton)
    )
    result["target_length_px"] = float(target_length_px)
    result["target_length_provenance"] = LENGTH_PRIOR_PROVENANCE
    arrays["corrected_frame"] = corrected
    arrays["edge_raw_threshold_mask"] = raw_threshold
    if (
        stable_skeleton is not None
        and stable_skeleton.visible_centerline_xy is not None
        and len(stable_skeleton.visible_centerline_xy) >= 2
    ):
        arrays["boundary_stable_centerline_xy"] = np.asarray(
            stable_skeleton.visible_centerline_xy, dtype=np.float64
        )

    curve_item = _curve_for_completion(arrays)
    if _is_completion_candidate(truncation) and curve_item is not None:
        source_stage, visible_curve = curve_item
        try:
            completion = complete_centerline_to_length(
                visible_curve,
                raw_frame.shape,
                target_length_px,
                truncation,
            )
        except (RuntimeError, ValueError) as error:
            result["fov_completion_error"] = f"{type(error).__name__}: {error}"
            completion = None
        if completion is None:
            index = int(source["sample_index"])
            result["sample_index"] = index
            result["annotation_index"] = None
            result["expected_no_worm"] = index in EXPECTED_NO_WORM_INDICES
            if index in EXPECTED_NO_WORM_INDICES:
                _force_expected_no_worm(result)
            else:
                result.setdefault(
                    "outcome", "success" if result["accepted"] else "rejected"
                )
            return result, arrays
        completed = np.asarray(completion.centerline_xy, dtype=np.float64)
        arrays["fov_completed_centerline_xy"] = completed
        arrays["fov_completion_in_fov"] = np.asarray(completion.in_fov, dtype=bool)
        arrays["fov_completion_observed_support"] = np.asarray(
            completion.observed_support, dtype=bool
        )
        result["fov_completion"] = {
            "source_stage": source_stage,
            "visible_length_px": curve_length(visible_curve),
            "completed_length_px": curve_length(completed),
            "complete": bool(completion.complete),
            "ambiguous": bool(completion.ambiguous),
            "classification_confident": _is_truncated(truncation),
            "diagnostics": _jsonable(completion.diagnostics),
        }
        # Completion is an inferred full curve. It can rescue a length-only A6
        # failure but must not erase unrelated segmentation/body-model failures.
        if (
            completion.complete
            and _is_truncated(truncation)
            and result.get("failure_stage")
            in {None, "A6_endpoint_extension", "A6_length_gate"}
        ):
            result.update(
                {
                    "accepted": True,
                    "failure_stage": None,
                    "failure_reasons": [],
                    "outcome": "success_with_fov_completion",
                }
            )
        elif completion.complete and not _is_truncated(truncation):
            result["outcome"] = (
                "success_with_uncertain_fov_completion"
                if result["accepted"]
                else "inferred_fov_completion_uncertain"
            )
        elif completion.complete:
            result["outcome"] = "partial_boundary_initialization"

    index = int(source["sample_index"])
    result["sample_index"] = index
    result["annotation_index"] = None
    result["expected_no_worm"] = index in EXPECTED_NO_WORM_INDICES
    if index in EXPECTED_NO_WORM_INDICES:
        _force_expected_no_worm(result)
    else:
        result.setdefault("outcome", "success" if result["accepted"] else "rejected")
    return result, arrays


def _fit_recording(
    payload: tuple[
        list[dict[str, Any]], int, dict[str, float], Path, frozenset[int]
    ],
) -> tuple[
    list[tuple[int, dict[str, Any], dict[str, Any], dict[str, np.ndarray]]],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    sources, flat_field_frames, length_priors, visual_dir, plot_indices = payload
    first = sources[0]
    path = Path(first["resolved_source_path"])
    stat = path.stat()
    if stat.st_size != int(first["source_size_bytes"]):
        raise RuntimeError(f"source size changed: {path}")
    if stat.st_mtime_ns != int(first["source_mtime_ns"]):
        raise RuntimeError(f"source mtime changed: {path}")
    recording = str(first["recording"])
    if recording not in length_priors:
        raise ValueError(f"no target length configured for {recording}")
    output = []
    with h5py.File(path, "r") as handle:
        dataset = handle[str(first["source_dataset_path"])]
        sample_indices = np.linspace(
            0,
            int(dataset.shape[0]) - 1,
            min(flat_field_frames, int(dataset.shape[0])),
            dtype=np.int64,
        )
        calibration = np.stack(
            [np.asarray(dataset[int(index)], dtype=np.uint8) for index in sample_indices]
        )
        field = estimate_flat_field(
            calibration,
            temporal_quantile=0.8,
            spatial_radius=31,
            smoothing_passes=2,
            min_gain=0.5,
            max_gain=2.5,
        )
        del calibration
        _release_large_image_buffers()
        field_metrics = _field_diagnostics(field, sample_indices)
        for source in sources:
            index = int(source["sample_index"])
            frame = np.asarray(dataset[int(source["frame_index"])], dtype=np.uint8)
            baseline, baseline_arrays = fit_case((source, frame), retain_diagnostics=True)
            baseline["sample_index"] = index
            baseline["annotation_index"] = None
            baseline["expected_no_worm"] = index in EXPECTED_NO_WORM_INDICES
            if index in EXPECTED_NO_WORM_INDICES:
                _force_expected_no_worm(baseline)
            else:
                baseline["outcome"] = "success" if baseline["accepted"] else "rejected"
            edge, edge_arrays = fit_edge_case(
                source, frame, field, float(length_priors[recording])
            )
            visual_name = diagnostic_filename(edge)
            if index in plot_indices:
                _plot_in_disposable_child(
                    baseline,
                    baseline_arrays,
                    edge,
                    edge_arrays,
                    visual_dir / visual_name,
                )
            edge["visual_artifact"] = f"frame_steps/{visual_name}"
            retained = {
                key: np.asarray(edge_arrays[key])
                for key in (
                    "a5_centerline_xy",
                    "a6_centerline_xy",
                    "fov_completed_centerline_xy",
                )
                if key in edge_arrays and index not in EXPECTED_NO_WORM_INDICES
            }
            output.append((index, baseline, edge, retained))
            del baseline_arrays, edge_arrays
            _release_large_image_buffers()
            print(
                json.dumps(
                    {
                        "sample_index": index,
                        "recording": recording,
                        "baseline": baseline["outcome"],
                        "edge_aware": edge["outcome"],
                    }
                ),
                flush=True,
            )
    return output, field_metrics, np.asarray(field.illumination), np.asarray(field.gain)


def _show_frame(axis: plt.Axes, frame: np.ndarray) -> None:
    lower, upper = np.percentile(frame, [1, 99])
    axis.imshow(frame, cmap="gray", vmin=lower, vmax=upper)
    axis.set_xlim(0, frame.shape[1] - 1)
    axis.set_ylim(frame.shape[0] - 1, 0)
    axis.set_axis_off()


def _overlay_mask(axis: plt.Axes, mask: np.ndarray, color: str, alpha: float) -> None:
    binary = np.asarray(mask, dtype=bool)
    rgba = np.zeros((*binary.shape, 4), dtype=np.float32)
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = binary.astype(np.float32) * alpha
    axis.imshow(rgba, interpolation="nearest")


def _plot_curve(axis: plt.Axes, arrays: dict[str, np.ndarray], color: str) -> None:
    if "fov_completed_centerline_xy" in arrays:
        curve = np.asarray(arrays["fov_completed_centerline_xy"], dtype=np.float64)
        observed_support = np.asarray(
            arrays.get(
                "fov_completion_observed_support",
                np.ones(len(curve), dtype=bool),
            ),
            dtype=bool,
        )
        for index in range(len(curve) - 1):
            observed = bool(observed_support[index] and observed_support[index + 1])
            axis.plot(
                curve[index : index + 2, 0],
                curve[index : index + 2, 1],
                color="white" if observed else color,
                linewidth=2.0 if observed else 2.4,
            )
        return
    for key in ("fov_completed_centerline_xy", "a6_centerline_xy", "a5_centerline_xy"):
        if key in arrays:
            curve = arrays[key]
            axis.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.0)
            return


def plot_comparison_sheet(
    baseline: dict[str, Any],
    baseline_arrays: dict[str, np.ndarray],
    edge: dict[str, Any],
    edge_arrays: dict[str, np.ndarray],
    path: Path,
) -> None:
    frame = np.asarray(baseline_arrays["frame"])
    corrected = np.asarray(edge_arrays["corrected_frame"])
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.2), constrained_layout=False)
    titles = (
        "Frozen: raw frame",
        "Frozen: threshold",
        "Frozen: component",
        "Frozen: final available curve",
        "Edge-aware: flat-fielded",
        "Edge-aware: pre-morphology threshold",
        "Edge-aware: component + boundary state",
        "Edge-aware: final/inferred curve",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title, fontsize=10)
    for axis in axes[0]:
        _show_frame(axis, frame)
    for axis in axes[1]:
        _show_frame(axis, corrected)
    if "raw_threshold_mask" in baseline_arrays:
        _overlay_mask(axes[0, 1], baseline_arrays["raw_threshold_mask"], smooth.CYAN, 0.45)
    if "section3_component" in baseline_arrays:
        _overlay_mask(axes[0, 2], baseline_arrays["section3_component"], smooth.GREEN, 0.45)
    _plot_curve(axes[0, 3], baseline_arrays, smooth.ORANGE)
    _overlay_mask(axes[1, 1], edge_arrays["edge_raw_threshold_mask"], smooth.CYAN, 0.45)
    if "section3_component" in edge_arrays:
        _overlay_mask(axes[1, 2], edge_arrays["section3_component"], smooth.GREEN, 0.45)
    _plot_curve(axes[1, 3], edge_arrays, smooth.ORANGE)
    if "fov_completed_centerline_xy" in edge_arrays:
        curve = np.asarray(edge_arrays["fov_completed_centerline_xy"])
        height, width = frame.shape
        margin = 10.0
        x_min = min(-margin, float(curve[:, 0].min()) - margin)
        x_max = max(width - 1 + margin, float(curve[:, 0].max()) + margin)
        y_min = min(-margin, float(curve[:, 1].min()) - margin)
        y_max = max(height - 1 + margin, float(curve[:, 1].max()) + margin)
        axes[1, 3].plot(
            [0, width - 1, width - 1, 0, 0],
            [0, 0, height - 1, height - 1, 0],
            color=smooth.CYAN,
            linestyle="--",
            linewidth=1.2,
            label="camera FOV",
        )
        axes[1, 3].set_xlim(x_min, x_max)
        axes[1, 3].set_ylim(y_max, y_min)
    axes[0, 3].text(
        0.02, 0.03, baseline["outcome"], transform=axes[0, 3].transAxes,
        color="white", bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )
    axes[1, 2].text(
        0.02,
        0.03,
        (
            f"state: {(edge.get('truncation') or {}).get('state', 'unavailable')}\n"
            f"contacts: {len((edge.get('truncation') or {}).get('contacts', []))}; "
            f"ends: {(edge.get('truncation') or {}).get('contact_ends', [])}\n"
            "raw skeleton: "
            f"{((edge.get('truncation') or {}).get('diagnostics') or {}).get('raw_skeleton_endpoint_count', 'n/a')} endpoints, "
            f"{((edge.get('truncation') or {}).get('diagnostics') or {}).get('raw_skeleton_branch_pixels', 'n/a')} branch px"
        ),
        transform=axes[1, 2].transAxes,
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )
    axes[1, 3].text(
        0.02, 0.03, edge["outcome"], transform=axes[1, 3].transAxes,
        color="white", bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )
    fig.suptitle(
        f"Sample {edge['sample_index']:02d}: frozen versus edge-aware — "
        f"{edge['recording']} frame {edge['frame_index']}",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.025, top=0.91,
                        wspace=0.045, hspace=0.18)
    fig.savefig(path, dpi=90, pil_kwargs={"quality": 86})
    plt.close(fig)


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [case for case in cases if not case["expected_no_worm"]]
    accepted = [case for case in eligible if case["accepted"]]
    return {
        "requested_frames": len(cases),
        "eligible_worm_frames": len(eligible),
        "expected_no_worm_frames": len(cases) - len(eligible),
        "expected_no_worm_indices": sorted(EXPECTED_NO_WORM_INDICES),
        "accepted_eligible_frames": len(accepted),
        "accepted_fraction_of_eligible": len(accepted) / len(eligible),
        "outcome_counts": dict(sorted(Counter(str(case["outcome"]) for case in cases).items())),
        "failure_stage_counts": dict(
            sorted(Counter(str(case["failure_stage"]) for case in cases if not case["accepted"]).items())
        ),
        "truncated_frames": sum(
            (case.get("truncation") or {}).get("state")
            in {"one_end_truncated", "two_sided_truncated"}
            for case in eligible
        ),
        "boundary_uncertain_frames": sum(
            (case.get("truncation") or {}).get("state") == "boundary_uncertain"
            for case in eligible
        ),
        "uncertain_censored_candidates": sum(
            (case.get("truncation") or {}).get("state") == "boundary_uncertain"
            and 1 <= len((case.get("truncation") or {}).get("contact_ends", [])) <= 2
            for case in eligible
        ),
        "fov_completed_frames": sum("fov_completion" in case for case in eligible),
        "completed_length_px": numeric_summary(
            case["fov_completion"]["completed_length_px"]
            for case in eligible if "fov_completion" in case
        ),
        "runtime_seconds": numeric_summary(case["runtime_seconds"] for case in cases),
    }


def plot_summary(comparison: dict[str, Any], path: Path) -> None:
    baseline = comparison["baseline"]
    edge = comparison["edge_aware"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    axes[0].bar(
        ["frozen", "edge-aware"],
        [baseline["accepted_eligible_frames"], edge["accepted_eligible_frames"]],
        color=[smooth.GRAY, smooth.GREEN],
    )
    axes[0].set_ylim(0, 29)
    axes[0].set_ylabel("accepted eligible frames (of 27)")
    axes[0].set_title("Operational coverage")
    labels = ["confident truncation", "uncertain candidate", "FOV completed"]
    axes[1].bar(
        labels,
        [
            edge["truncated_frames"],
            edge["uncertain_censored_candidates"],
            edge["fov_completed_frames"],
        ],
        color=[smooth.CYAN, smooth.GRAY, smooth.ORANGE],
    )
    axes[1].tick_params(axis="x", labelrotation=15)
    axes[1].set_ylim(0, 29)
    axes[1].set_ylabel("frames")
    axes[1].set_title("Edge-aware actions")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.suptitle("Same 30 frames; samples 09, 19, 29 excluded as known empty")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_visual_index(cases: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Frozen versus edge-aware per-frame diagnostics",
        "",
        "These sheets compare the same 30 annotation-free archive frames. The edge-aware side applies one recording-level flat field before segmentation, classifies boundary contact before morphology, and shows fixed-length FOV completion when supported. This is a coverage comparison, not an anatomical-accuracy evaluation.",
        "",
        "Samples 09, 19, and 29 are known empty frames and are hard-rejected as `no_worm_expected`.",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## Sample {case['sample_index']:02d}: `{case['recording']}` frame {case['frame_index']}",
                "",
                f"Frozen: **{case['baseline']['outcome']}**. Edge-aware: **{case['edge_aware']['outcome']}**.",
                "",
                f"![Comparison for sample {case['sample_index']:02d}]({case['visual_artifact']})",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    recordings = list(args.recordings or DEFAULT_RECORDINGS)
    if len(recordings) != 3:
        raise ValueError("exactly three recordings are required")
    if args.workers < 1 or args.flat_field_frames < 3:
        raise ValueError("workers must be >=1 and flat-field-frames must be >=3")
    length_priors = parse_length_priors(args.length_prior)
    plot_indices = parse_plot_range(args.plot_range)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "frame_steps"
    visual_dir.mkdir(parents=True, exist_ok=True)

    sources, provenance = recording_records(recordings)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        grouped.setdefault(str(source["resolved_source_path"]), []).append(source)
    payloads = [
        (group, args.flat_field_frames, length_priors, visual_dir, plot_indices)
        for group in grouped.values()
    ]
    groups = []
    if args.workers == 1:
        groups = [_fit_recording(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(payloads))) as executor:
            futures = [executor.submit(_fit_recording, payload) for payload in payloads]
            for future in as_completed(futures):
                groups.append(future.result())

    fitted = {}
    flat_fields = {}
    flat_arrays = {}
    for rows, field_metrics, illumination, gain in groups:
        recording = str(rows[0][2]["recording"])
        flat_fields[recording] = field_metrics
        flat_arrays[f"{recording}_illumination"] = illumination
        flat_arrays[f"{recording}_gain"] = gain
        for index, baseline, edge, edge_arrays in rows:
            fitted[index] = (baseline, edge, edge_arrays)

    report_cases = []
    baseline_cases = []
    edge_cases = []
    predictions = dict(flat_arrays)
    for index in range(30):
        baseline, edge, edge_arrays = fitted[index]
        baseline_cases.append(baseline)
        edge_cases.append(edge)
        report_cases.append(
            {
                "sample_index": index,
                "recording": edge["recording"],
                "frame_index": edge["frame_index"],
                "expected_no_worm": index in EXPECTED_NO_WORM_INDICES,
                "baseline": baseline,
                "edge_aware": edge,
                "visual_artifact": edge["visual_artifact"],
            }
        )
        for key in ("a5_centerline_xy", "a6_centerline_xy", "fov_completed_centerline_xy"):
            if key in edge_arrays and index not in EXPECTED_NO_WORM_INDICES:
                predictions[f"sample_{index:02d}_{key}"] = edge_arrays[key]

    comparison = {"baseline": summarize(baseline_cases), "edge_aware": summarize(edge_cases)}
    frozen_payload = None
    if args.baseline_metrics.exists():
        frozen_payload = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))["summary"]
    payload = {
        "status": "edge_aware_unannotated_operational_comparison",
        "evidence_boundary": {
            "manual_annotations_available": False,
            "anatomical_accuracy_claim": False,
            "protected_2025_holdout_opened": False,
            "expected_no_worm_indices": sorted(EXPECTED_NO_WORM_INDICES),
        },
        "method": {
            "flat_field": "one temporal-upper-quantile field per recording",
            "truncation": "classified from threshold support before morphology",
            "fov_completion": "fixed-length curvature extrapolation outside the FOV",
            "length_priors_px": length_priors,
            "length_prior_provenance": LENGTH_PRIOR_PROVENANCE,
        },
        "inputs": provenance,
        "flat_fields": flat_fields,
        "comparison": comparison,
        "previous_frozen_summary_for_reference": frozen_payload,
        "per_case": report_cases,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, cwd=PROJECT_ROOT,
        ).stdout.strip(),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "predictions_and_flat_fields.npz", **predictions)
    plot_summary(comparison, args.output_dir / "summary.png")
    write_visual_index(report_cases, args.output_dir / "FRAME_STEPS.md")
    print(json.dumps({"output_dir": str(args.output_dir), "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
