#!/usr/bin/env python3
"""Evaluate visible-only edge-censored geometry on the frozen 30-frame set.

This script deliberately does not infer anatomy outside the camera field of
view.  It estimates one recording-level flat field, runs the existing A1--A6
pipeline, classifies boundary contact from the pre-morphology threshold mask,
and repairs a boundary-affected A3 path by fitting only its reliable interior
core.  The fitted curve stops at the pixel-center camera rectangle.

The archive frames have no anatomical centerline annotations.  The generated
metrics are therefore operational coverage and diagnostic evidence, not an
accuracy benchmark.  Samples 09, 19, and 29 are known empty and are always
hard-rejected.
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
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

import build_smooth_body_prior_experiment as smooth
from evaluate_final_geometry_primary30 import fit_case
from evaluate_final_geometry_unannotated30 import (
    DEFAULT_RECORDINGS,
    diagnostic_filename,
    numeric_summary,
    recording_records,
)
from worm_pose_gen.classical import ClassicalConfig, robust_dark_ridge
from worm_pose_gen.edge_censored import repair_edge_censored_centerline
from worm_pose_gen.flat_field import FlatField, apply_flat_field, estimate_flat_field
from worm_pose_gen.fov_completion import (
    BoundaryTruncationResult,
    classify_boundary_truncation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "docs" / "final_algorithm_edge_censored_unannotated30"
)
DEFAULT_FROZEN_METRICS = (
    PROJECT_ROOT / "docs" / "final_algorithm_unannotated30" / "metrics.json"
)
DEFAULT_EDGE_AWARE_METRICS = (
    PROJECT_ROOT / "docs" / "final_algorithm_edge_aware_unannotated30" / "metrics.json"
)
EXPECTED_NO_WORM_INDICES = frozenset({9, 19, 29})
FLAT_FIELD_SAMPLE_COUNT = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording", action="append", type=Path, dest="recordings",
        help="raw HDF5 recording; specify exactly three times",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--flat-field-frames", type=int, default=FLAT_FIELD_SAMPLE_COUNT)
    parser.add_argument(
        "--plot-range", default="0:30", metavar="START:STOP",
        help="half-open sample range whose diagnostic sheets are regenerated",
    )
    parser.add_argument("--frozen-metrics", type=Path, default=DEFAULT_FROZEN_METRICS)
    parser.add_argument(
        "--edge-aware-metrics", type=Path, default=DEFAULT_EDGE_AWARE_METRICS
    )
    parser.add_argument("--smoothness", type=float, default=2.0)
    parser.add_argument("--context-points", type=int, default=8)
    parser.add_argument("--min-core-points", type=int, default=8)
    parser.add_argument("--min-core-length-px", type=float, default=20.0)
    return parser.parse_args()


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
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    return value


def curve_length(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _release_buffers() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):  # pragma: no cover - non-glibc fallback
        pass


def _force_expected_no_worm(case: dict[str, Any]) -> None:
    case.update(
        accepted=False,
        expected_no_worm=True,
        outcome="no_worm_expected",
        failure_stage="expected_no_worm_rejection",
        failure_reasons=["sample_is_in_explicit_expected_no_worm_index_set"],
        final_visible_pose_available=False,
    )


def _truncation_summary(value: BoundaryTruncationResult) -> dict[str, Any]:
    return {
        "state": value.state,
        "edge_band_px": int(value.edge_band_px),
        "contact_ends": list(value.contact_ends),
        "contacts": [_jsonable(item) for item in value.contacts],
        "diagnostics": _jsonable(value.diagnostics),
    }


def _field_summary(field: FlatField, indices: np.ndarray) -> dict[str, Any]:
    gain = np.asarray(field.gain, dtype=np.float64)
    illumination = np.asarray(field.illumination, dtype=np.float64)
    return {
        "sample_frame_indices": [int(item) for item in indices],
        "dark_level": float(field.dark_level),
        "reference_level": float(field.reference_level),
        "illumination_min_median_max": [
            float(illumination.min()), float(np.median(illumination)),
            float(illumination.max()),
        ],
        "gain_min_median_p99_max": [
            float(gain.min()), float(np.median(gain)),
            float(np.percentile(gain, 99)), float(gain.max()),
        ],
    }


def _is_censored_candidate(value: BoundaryTruncationResult) -> bool:
    return value.state in {"one_end_truncated", "two_sided_truncated"} or (
        value.state == "boundary_uncertain" and bool(value.contact_ends)
    )


def _fit_edge_censored_case(
    source: dict[str, Any],
    raw_frame: np.ndarray,
    field: FlatField,
    *,
    smoothness: float,
    context_points: int,
    min_core_points: int,
    min_core_length_px: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    corrected = apply_flat_field(raw_frame, field, clip=(0.0, 255.0))
    case, arrays = fit_case((source, corrected), retain_diagnostics=True)
    cfg = ClassicalConfig()
    score = robust_dark_ridge(corrected, cfg)
    raw_mask = score >= cfg.foreground_z
    truncation = classify_boundary_truncation(
        raw_mask, close_radius=cfg.close_radius, worm_radius_px=7.0
    )
    arrays["raw_frame"] = np.asarray(raw_frame)
    arrays["corrected_frame"] = np.asarray(corrected)
    arrays["edge_raw_threshold_mask"] = np.asarray(raw_mask, dtype=bool)
    arrays["boundary_component_mask"] = np.asarray(
        truncation.component_mask, dtype=bool
    )
    case["flat_field_applied"] = True
    case["truncation"] = _truncation_summary(truncation)
    case["edge_censored_attempted"] = False
    case["edge_censored_success"] = False

    selected_mask = arrays.get("a1_selected_repair_mask")
    if selected_mask is not None:
        try:
            original_path, original_skeleton = smooth.initial_path(selected_mask)
            arrays["original_a3_path_xy"] = np.asarray(original_path, dtype=np.float64)
            arrays["original_a3_skeleton"] = np.asarray(original_skeleton, dtype=bool)
        except (RuntimeError, ValueError) as error:
            case["edge_censored_failure_reason"] = (
                f"original_A3_path_unavailable: {type(error).__name__}: {error}"
            )

    if _is_censored_candidate(truncation) and "original_a3_path_xy" in arrays:
        case["edge_censored_attempted"] = True
        try:
            repaired = repair_edge_censored_centerline(
                arrays["original_a3_path_xy"],
                raw_frame.shape,
                truncation,
                raw_mask=raw_mask,
                smoothness=smoothness,
                context_points=context_points,
                min_core_points=min_core_points,
                min_core_length_px=min_core_length_px,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            repaired = None
            case["edge_censored_failure_reason"] = (
                f"{type(error).__name__}: {error}"
            )
        if repaired is None:
            case["edge_censored"] = {
                "success": False,
                "censored_ends": [],
                "core_slice": None,
                "boundary_crossings_xy": {},
                "diagnostics": {},
                "failure_reason": case["edge_censored_failure_reason"],
            }
        else:
            case["edge_censored"] = {
                "success": bool(repaired.success),
                "censored_ends": list(repaired.censored_ends),
                "core_slice": _jsonable(repaired.core_slice),
                "boundary_crossings_xy": _jsonable(repaired.boundary_crossings_xy),
                "diagnostics": _jsonable(repaired.diagnostics),
                "failure_reason": repaired.failure_reason,
            }
        if repaired is not None and repaired.reliable_core_mask is not None:
            arrays["edge_reliable_core_mask"] = np.asarray(
                repaired.reliable_core_mask, dtype=bool
            )
        if repaired is not None and repaired.observed_support is not None:
            arrays["edge_observed_support"] = np.asarray(
                repaired.observed_support, dtype=bool
            )
        if repaired is not None and repaired.censored_endpoint_mask is not None:
            arrays["edge_censored_endpoint_mask"] = np.asarray(
                repaired.censored_endpoint_mask, dtype=bool
            )
        if repaired is not None and repaired.success and repaired.centerline_xy is not None:
            visible = np.asarray(repaired.centerline_xy, dtype=np.float64)
            height, width = raw_frame.shape
            if not (
                np.all((0.0 <= visible[:, 0]) & (visible[:, 0] <= width - 1))
                and np.all((0.0 <= visible[:, 1]) & (visible[:, 1] <= height - 1))
            ):
                raise RuntimeError("edge-censored core returned coordinates outside FOV")
            arrays["edge_visible_centerline_xy"] = visible
            case["edge_censored_success"] = True
            case["final_visible_pose_available"] = True
            case["visible_pose_source"] = "edge_censored_A3_core_fit"
            case["visible_pose_length_px"] = curve_length(visible)
            case["outcome"] = "visible_edge_censored_pose"
        elif repaired is not None:
            case["edge_censored_failure_reason"] = repaired.failure_reason

        if not case["edge_censored_success"]:
            case["outcome"] = "edge_censored_repair_failed"

    if not case.get("final_visible_pose_available", False):
        for key, source_name in (
            ("a6_centerline_xy", "flat_fielded_A6"),
            ("a5_centerline_xy", "flat_fielded_A5"),
            ("a4_latent_midline_xy", "flat_fielded_A4"),
        ):
            if key in arrays and not _is_censored_candidate(truncation):
                case["final_visible_pose_available"] = True
                case["visible_pose_source"] = source_name
                case["visible_pose_length_px"] = curve_length(arrays[key])
                case.setdefault("outcome", "success" if case["accepted"] else "partial")
                break
    case.setdefault("final_visible_pose_available", False)
    case.setdefault("outcome", "success" if case["accepted"] else "rejected")

    index = int(source["sample_index"])
    case["sample_index"] = index
    case["annotation_index"] = None
    case["expected_no_worm"] = index in EXPECTED_NO_WORM_INDICES
    if index in EXPECTED_NO_WORM_INDICES:
        _force_expected_no_worm(case)
    return case, arrays


def _show_frame(axis: plt.Axes, frame: np.ndarray) -> None:
    lower, upper = np.percentile(frame, [1, 99])
    axis.imshow(frame, cmap="gray", vmin=lower, vmax=upper)
    axis.set_xlim(0, frame.shape[1] - 1)
    axis.set_ylim(frame.shape[0] - 1, 0)
    axis.set_axis_off()


def _overlay_mask(
    axis: plt.Axes, mask: np.ndarray, color: str, alpha: float = 0.35
) -> None:
    binary = np.asarray(mask, dtype=bool)
    rgba = np.zeros((*binary.shape, 4), dtype=np.float32)
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = binary.astype(np.float32) * alpha
    axis.imshow(rgba, interpolation="nearest")


def _edge_band_mask(shape: tuple[int, int], band: int) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    return np.minimum.reduce((yy, xx, height - 1 - yy, width - 1 - xx)) < band


def _plot_path(axis: plt.Axes, path: np.ndarray, **kwargs: Any) -> None:
    points = np.asarray(path, dtype=np.float64)
    if len(points):
        axis.plot(points[:, 0], points[:, 1], **kwargs)


def plot_frame_steps(case: dict[str, Any], arrays: dict[str, np.ndarray], path: Path) -> None:
    raw = np.asarray(arrays["raw_frame"])
    corrected = np.asarray(arrays["corrected_frame"])
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 12.2), constrained_layout=True)
    titles = (
        "0. Raw NIR frame",
        "1. Recording flat-field correction",
        "2. Local darkness / z ≥ 2.6",
        "3. Component + boundary state",
        "A3. Original skeleton and ordered path",
        "A3c. Edge exclusion + reliable core",
        "A3c. Visible-only fit to FOV boundary",
        "Downstream / final visible pose",
        "Outcome and evidence boundary",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title, fontsize=10.5)

    _show_frame(axes[0, 0], raw)
    _show_frame(axes[0, 1], corrected)
    score = arrays.get("local_darkness_score")
    if score is not None:
        limit = max(3.5, float(np.percentile(score, 99.5)))
        image = axes[0, 2].imshow(score, cmap="magma", vmin=-1.0, vmax=limit)
        _overlay_mask(axes[0, 2], arrays["edge_raw_threshold_mask"], smooth.CYAN, 0.28)
        axes[0, 2].set_axis_off()
        fig.colorbar(image, ax=axes[0, 2], fraction=0.045, pad=0.02)

    for axis in (axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0]):
        _show_frame(axis, corrected)
    component = arrays.get("section3_component", arrays["boundary_component_mask"])
    _overlay_mask(axes[1, 0], component, smooth.GREEN, 0.40)
    truncation = case["truncation"]
    axes[1, 0].text(
        0.02, 0.03,
        f"{truncation['state']}\nband {truncation['edge_band_px']} px; "
        f"contacts {len(truncation['contacts'])}",
        transform=axes[1, 0].transAxes, color="white", fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.68, "edgecolor": "none"},
    )

    if "original_a3_skeleton" in arrays:
        _overlay_mask(axes[1, 1], arrays["original_a3_skeleton"], smooth.ORANGE, 0.9)
    if "original_a3_path_xy" in arrays:
        original = arrays["original_a3_path_xy"]
        _plot_path(axes[1, 1], original, color="white", linewidth=1.4)
        band_mask = _edge_band_mask(raw.shape, int(truncation["edge_band_px"]))
        _overlay_mask(axes[1, 2], band_mask, "crimson", 0.20)
        _plot_path(axes[1, 2], original, color=smooth.GRAY, linewidth=1.4)
        core = arrays.get("edge_reliable_core_mask")
        if core is not None and len(core) == len(original):
            _plot_path(axes[1, 2], original[core], color=smooth.GREEN, linewidth=2.8)
    else:
        axes[1, 1].text(0.5, 0.5, "A3 path unavailable", ha="center", va="center",
                        transform=axes[1, 1].transAxes, color="white")

    if "edge_visible_centerline_xy" in arrays:
        visible = arrays["edge_visible_centerline_xy"]
        _plot_path(axes[2, 0], visible, color=smooth.ORANGE, linewidth=2.5)
        observed = arrays.get("edge_observed_support")
        if observed is not None and len(observed) == len(visible):
            _plot_path(axes[2, 0], visible[observed], color="white", linewidth=1.5)
        for crossing in (case.get("edge_censored") or {}).get(
            "boundary_crossings_xy", {}
        ).values():
            axes[2, 0].scatter(crossing[0], crossing[1], color=smooth.CYAN, s=38)
    else:
        axes[2, 0].text(0.5, 0.5, "No edge-censored fit", ha="center", va="center",
                        transform=axes[2, 0].transAxes, color="white")

    _show_frame(axes[2, 1], corrected)
    if case.get("visible_pose_source") == "edge_censored_A3_core_fit":
        final_key = "edge_visible_centerline_xy"
    else:
        final_key = next(
            (key for key in ("a6_centerline_xy", "a5_centerline_xy", "a4_latent_midline_xy")
             if key in arrays), None,
        )
    if final_key is not None and case.get("final_visible_pose_available"):
        _plot_path(axes[2, 1], arrays[final_key], color=smooth.ORANGE, linewidth=2.5)
    else:
        axes[2, 1].text(0.5, 0.5, "No final visible pose", ha="center", va="center",
                        transform=axes[2, 1].transAxes, color="white")

    axes[2, 2].set_axis_off()
    failure = case.get("edge_censored_failure_reason") or "; ".join(
        case.get("failure_reasons") or []
    )
    axes[2, 2].text(
        0.02, 0.98,
        f"Outcome: {case['outcome']}\n"
        f"Pipeline accepted: {bool(case['accepted'])}\n"
        f"Visible pose: {bool(case['final_visible_pose_available'])}\n"
        f"Source: {case.get('visible_pose_source', 'none')}\n"
        f"Failure: {failure or 'none'}\n\n"
        "No off-camera points are inferred.\n"
        "No centerline ground truth is available for this frame.",
        transform=axes[2, 2].transAxes, va="top", fontsize=10,
    )
    fig.suptitle(
        f"Sample {case['sample_index']:02d}: {case['recording']} frame "
        f"{case['frame_index']} — visible-only edge censoring",
        fontsize=14,
    )
    fig.savefig(path, dpi=95, pil_kwargs={"quality": 88})
    plt.close(fig)


def _fit_recording(payload: tuple[Any, ...]) -> tuple[Any, ...]:
    (
        sources, flat_field_frames, visual_dir, plot_indices, smoothness,
        context_points, min_core_points, min_core_length_px,
    ) = payload
    first = sources[0]
    source_path = Path(first["resolved_source_path"])
    stat = source_path.stat()
    if stat.st_size != int(first["source_size_bytes"]) or stat.st_mtime_ns != int(
        first["source_mtime_ns"]
    ):
        raise RuntimeError(f"source changed: {source_path}")
    rows = []
    with h5py.File(source_path, "r") as handle:
        dataset = handle[str(first["source_dataset_path"])]
        indices = np.linspace(
            0, int(dataset.shape[0]) - 1,
            min(flat_field_frames, int(dataset.shape[0])), dtype=np.int64,
        )
        calibration = np.stack(
            [np.asarray(dataset[int(index)], dtype=np.uint8) for index in indices]
        )
        field = estimate_flat_field(
            calibration, temporal_quantile=0.8, spatial_radius=31,
            smoothing_passes=2, min_gain=0.5, max_gain=2.5,
        )
        del calibration
        _release_buffers()
        for source in sources:
            index = int(source["sample_index"])
            frame = np.asarray(dataset[int(source["frame_index"])], dtype=np.uint8)
            case, arrays = _fit_edge_censored_case(
                source, frame, field, smoothness=smoothness,
                context_points=context_points, min_core_points=min_core_points,
                min_core_length_px=min_core_length_px,
            )
            visual_name = diagnostic_filename(case)
            if index in plot_indices:
                plot_frame_steps(case, arrays, visual_dir / visual_name)
            case["visual_artifact"] = f"frame_steps/{visual_name}"
            retained = {
                key: np.asarray(arrays[key])
                for key in (
                    "edge_visible_centerline_xy", "edge_reliable_core_mask",
                    "original_a3_path_xy", "a5_centerline_xy", "a6_centerline_xy",
                )
                if key in arrays and index not in EXPECTED_NO_WORM_INDICES
            }
            rows.append((index, case, retained))
            print(json.dumps({"sample_index": index, "outcome": case["outcome"]}), flush=True)
            del arrays
            _release_buffers()
    return (
        rows, str(first["recording"]), _field_summary(field, indices),
        np.asarray(field.illumination), np.asarray(field.gain),
    )


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [case for case in cases if not case["expected_no_worm"]]
    return {
        "requested_frames": len(cases),
        "eligible_worm_frames": len(eligible),
        "expected_no_worm_indices": sorted(EXPECTED_NO_WORM_INDICES),
        "pipeline_accepted_eligible": sum(bool(case["accepted"]) for case in eligible),
        "final_visible_pose_available": sum(
            bool(case["final_visible_pose_available"]) for case in eligible
        ),
        "edge_censored_attempted": sum(
            bool(case["edge_censored_attempted"]) for case in eligible
        ),
        "edge_censored_succeeded": sum(
            bool(case["edge_censored_success"]) for case in eligible
        ),
        "boundary_state_counts": dict(sorted(Counter(
            case["truncation"]["state"] for case in eligible
        ).items())),
        "outcome_counts": dict(sorted(Counter(case["outcome"] for case in cases).items())),
        "failure_stage_counts": dict(sorted(Counter(
            str(case.get("failure_stage")) for case in eligible if not case["accepted"]
        ).items())),
        "visible_pose_length_px": numeric_summary(
            case["visible_pose_length_px"] for case in eligible
            if "visible_pose_length_px" in case
        ),
        "runtime_seconds": numeric_summary(case["runtime_seconds"] for case in cases),
    }


def _prior_summary(path: Path, key: str | None = None) -> Any:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("summary") or payload.get("comparison")
    return value if key is None or value is None else value.get(key)


def plot_summary(summary: dict[str, Any], prior: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    frozen = prior.get("frozen_accepted")
    flat = prior.get("flat_field_accepted")
    labels, values = [], []
    if frozen is not None:
        labels.append("frozen")
        values.append(frozen)
    if flat is not None:
        labels.append("prior edge-aware")
        values.append(flat)
    labels.extend(("corrected pipeline", "visible pose"))
    values.extend((summary["pipeline_accepted_eligible"], summary["final_visible_pose_available"]))
    axes[0].bar(labels, values, color=[smooth.GRAY, smooth.CYAN, smooth.GREEN, smooth.ORANGE][-len(labels):])
    axes[0].set_ylim(0, 29)
    axes[0].set_ylabel("eligible frames (of 27)")
    axes[0].tick_params(axis="x", labelrotation=18)
    axes[0].set_title("Operational coverage (not anatomical accuracy)")
    axes[1].bar(
        ["attempted", "succeeded"],
        [summary["edge_censored_attempted"], summary["edge_censored_succeeded"]],
        color=[smooth.CYAN, smooth.GREEN],
    )
    axes[1].set_ylim(0, 29)
    axes[1].set_title("Visible-only edge-censored repair")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_visual_index(cases: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Flat-fielded, visible-only edge-censored frame diagnostics", "",
        "Each sheet shows raw input, flat-field correction, segmentation, the original A3 path, the excluded edge band and retained core, the curve fitted only through the visible field, and the final visible pose or failure.", "",
        "No points outside the camera FOV are generated. These frames have no anatomical centerline labels, so the sheets demonstrate operational behavior rather than accuracy. Samples 09, 19, and 29 are known empty and hard-rejected.", "",
    ]
    for case in cases:
        lines.extend((
            f"## Sample {case['sample_index']:02d}: `{case['recording']}` frame {case['frame_index']}", "",
            f"Outcome: **{case['outcome']}**. Boundary state: **{case['truncation']['state']}**.", "",
            f"![Step sheet for sample {case['sample_index']:02d}]({case['visual_artifact']})", "",
        ))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    recordings = list(args.recordings or DEFAULT_RECORDINGS)
    if len(recordings) != 3:
        raise ValueError("exactly three recordings are required")
    if args.workers < 1 or args.flat_field_frames < 3:
        raise ValueError("workers must be >= 1 and flat-field-frames must be >= 3")
    plot_indices = parse_plot_range(args.plot_range)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "frame_steps"
    visual_dir.mkdir(parents=True, exist_ok=True)
    sources, provenance = recording_records(recordings)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        grouped.setdefault(str(source["resolved_source_path"]), []).append(source)
    payloads = [
        (
            group, args.flat_field_frames, visual_dir, plot_indices, args.smoothness,
            args.context_points, args.min_core_points, args.min_core_length_px,
        )
        for group in grouped.values()
    ]
    if args.workers == 1:
        groups = [_fit_recording(payload) for payload in payloads]
    else:
        groups = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(payloads))) as pool:
            futures = [pool.submit(_fit_recording, payload) for payload in payloads]
            for future in as_completed(futures):
                groups.append(future.result())

    fitted: dict[int, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    flat_fields, predictions = {}, {}
    for rows, recording, field_summary, illumination, gain in groups:
        flat_fields[recording] = field_summary
        predictions[f"{recording}_illumination"] = illumination
        predictions[f"{recording}_gain"] = gain
        for index, case, retained in rows:
            fitted[index] = (case, retained)
    cases = []
    for index in range(30):
        case, retained = fitted[index]
        cases.append(case)
        for key, value in retained.items():
            predictions[f"sample_{index:02d}_{key}"] = value
    summary = summarize(cases)
    frozen = _prior_summary(args.frozen_metrics)
    edge_aware = _prior_summary(args.edge_aware_metrics, "edge_aware")
    prior = {
        "frozen_accepted": None if frozen is None else frozen.get("accepted_eligible_frames", frozen.get("accepted_frames")),
        "flat_field_accepted": None if edge_aware is None else edge_aware.get("accepted_eligible_frames"),
    }
    payload = {
        "status": "edge_censored_visible_only_unannotated_operational_comparison",
        "evidence_boundary": {
            "manual_centerline_annotations_available": False,
            "anatomical_accuracy_claim": False,
            "off_fov_inference_performed": False,
            "protected_2025_holdout_opened": False,
            "expected_no_worm_indices": sorted(EXPECTED_NO_WORM_INDICES),
            "important_limitation": "A repaired visible curve is an initialization result; operational success does not establish anatomical correctness near the boundary.",
        },
        "method": {
            "flat_field": "recording temporal upper quantile, spatially smoothed, divisive gain capped to [0.5, 2.5]",
            "boundary_classification": "pre-morphology threshold support",
            "edge_censoring": "remove edge-influenced A3 stations, fit the reliable interior core, and continue only to the pixel-center FOV rectangle",
            "off_camera_curve": "never generated",
            "parameters": {
                "smoothness": args.smoothness,
                "context_points": args.context_points,
                "min_core_points": args.min_core_points,
                "min_core_length_px": args.min_core_length_px,
            },
        },
        "inputs": provenance,
        "flat_fields": flat_fields,
        "prior_comparison_for_reference": prior,
        "summary": summary,
        "per_case": cases,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, cwd=PROJECT_ROOT,
        ).stdout.strip(),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "predictions_and_flat_fields.npz", **predictions)
    plot_summary(summary, prior, args.output_dir / "summary.png")
    write_visual_index(cases, args.output_dir / "FRAME_STEPS.md")
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
