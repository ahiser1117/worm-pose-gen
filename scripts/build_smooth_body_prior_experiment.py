#!/usr/bin/env python3
"""Audit a smooth-midline, slowly-varying-width completion of explainer step 3.

This is deliberately a one-frame geometry experiment, not a production method.
It starts from the exact largest connected component shown in section 3 of
``docs/POSE_ESTIMATION_EXPLAINER.md`` and writes reproducible audit figures plus
machine-readable metrics.  It does not modify the original explainer assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from worm_pose_gen.anchors import render_centerline_mask, skeleton_topology
from worm_pose_gen.annotation import resample_polyline
from worm_pose_gen.classical import (
    ClassicalConfig,
    _dilate,
    _erode,
    _largest_component,
    _prune_skeleton_endpoints,
    _skeleton_longest_path,
    _thin,
    resample_centerline,
    robust_dark_ridge,
    tangent_angles,
)
from worm_pose_gen.latent import decode_centerline, encode_centerline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROXY_HDF5 = Path(
    "/temp_data4/alex/external_artifacts/datasets/"
    "worm_pose_gen/proxy_v1/proxy_labels.h5"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "smooth_body_prior_experiment"
MODELED_BODY_MAX_AREA = 40_000
DEFAULT_ANNOTATIONS = Path(
    "/temp_data4/alex/external_artifacts/annotations/worm_pose_tier_a_alex.json"
)
RECORDING = "2023-09-19-01"
FRAME_INDEX = 3420
SOURCE_DATASET = "/img_nir"

CYAN = "#2ec4b6"
PALE_CYAN = "#7ee8fa"
MAGENTA = "#ff4fa3"
ORANGE = "#ff9f1c"
GREEN = "#57d68d"
RED = "#ff5d5d"
GRAY = "#a7adb4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-hdf5", type=Path, default=DEFAULT_PROXY_HDF5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="one-annotator trace used only for post-fit auditing",
    )
    parser.add_argument(
        "--latent-coefficients",
        type=int,
        default=16,
        help="cubic B-spline angle coefficients in the smooth midline",
    )
    parser.add_argument(
        "--radius-slope-limit",
        type=float,
        default=1.0,
        help="maximum radius change in pixels between adjacent body stations",
    )
    parser.add_argument(
        "--containment-margin",
        type=float,
        default=0.75,
        help="extra radius in pixels around every detected positive pixel center",
    )
    parser.add_argument(
        "--max-full-width",
        type=float,
        default=80.0,
        help="repo-consistent full-width feasibility ceiling in pixels",
    )
    return parser.parse_args()


def load_real_frame(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with h5py.File(path, "r") as handle:
        if handle.attrs["source_dataset_path"] != SOURCE_DATASET:
            raise RuntimeError("unexpected source dataset path")
        group = handle[RECORDING]
        source_indices = np.asarray(group["accepted_frame_index"], dtype=np.int64)
        positions = np.flatnonzero(source_indices == FRAME_INDEX)
        if len(positions) != 1:
            raise RuntimeError(f"expected one cached copy of frame {FRAME_INDEX}")
        frame = np.asarray(group["accepted_image"][int(positions[0])], dtype=np.uint8)
        provenance = {
            "proxy_hdf5": str(path),
            "configured_source_path": str(group.attrs["configured_source_path"]),
            "resolved_source_path": str(group.attrs["resolved_source_path"]),
            "source_dataset_path": str(handle.attrs["source_dataset_path"]),
            "recording": RECORDING,
            "frame_index": FRAME_INDEX,
        }
    return frame, provenance


def crop_bounds(mask: np.ndarray, padding: int = 35) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    height, width = mask.shape
    return (
        max(0, int(xx.min()) - padding),
        min(width, int(xx.max()) + padding + 1),
        max(0, int(yy.min()) - padding),
        min(height, int(yy.max()) + padding + 1),
    )


def set_crop(ax: plt.Axes, bounds: tuple[int, int, int, int]) -> None:
    x0, x1, y0, y1 = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_axis_off()


def overlay_mask(ax: plt.Axes, mask: np.ndarray, color: str, alpha: float) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = np.asarray(mask, dtype=np.float32) * alpha
    ax.imshow(rgba, interpolation="nearest")


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def initial_path(component: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the section-4/5 path used only to initialize the prior."""

    yy, xx = np.nonzero(component)
    pad = 3
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(component.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(component.shape[1], int(xx.max()) + pad + 1)
    skeleton_crop = _prune_skeleton_endpoints(_thin(component[y0:y1, x0:x1]))
    path, _, _ = _skeleton_longest_path(skeleton_crop)
    if path is None:
        raise RuntimeError("section-3 component has no initial skeleton path")
    path[:, 0] += x0
    path[:, 1] += y0
    skeleton = np.zeros_like(component)
    skeleton[y0:y1, x0:x1] = skeleton_crop
    return path, skeleton


def fill_enclosed_cavities(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill background components not connected to the image exterior.

    The largest background component is the exterior for this fully in-frame
    object.  It includes the open center of the U, so that anatomically
    important negative space is preserved.  Only isolated interior cavities
    become initialization foreground.
    """

    exterior, _, background_component_count = _largest_component(~np.asarray(mask, dtype=bool))
    enclosed = ~mask & ~exterior
    filled = mask | enclosed
    return filled, enclosed, max(0, background_component_count - 1)


def nearest_station_requirements(
    positive_mask: np.ndarray,
    centerline_xy: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign every positive pixel center to its nearest midline station."""

    yy, xx = np.nonzero(positive_mask)
    positive_xy = np.column_stack((xx, yy)).astype(np.float64)
    best_distance2 = np.full(len(positive_xy), np.inf, dtype=np.float64)
    nearest = np.zeros(len(positive_xy), dtype=np.int64)
    for index, point in enumerate(centerline_xy):
        distance2 = np.square(positive_xy - point).sum(axis=1)
        update = distance2 < best_distance2
        best_distance2[update] = distance2[update]
        nearest[update] = index
    required = np.full(len(centerline_xy), margin, dtype=np.float64)
    np.maximum.at(required, nearest, np.sqrt(best_distance2) + margin)
    return required, positive_xy, nearest


def slowly_varying_majorant(required: np.ndarray, slope_limit: float) -> np.ndarray:
    """Return a smooth upper envelope with bounded adjacent width changes.

    ``max_j(required[j] - slope_limit * abs(i-j))`` is the smallest sequence
    above ``required`` whose adjacent changes are at most ``slope_limit``.
    A short binomial blur rounds its corners; taking the maximum preserves the
    containment guarantee and the same Lipschitz bound.
    """

    values = np.asarray(required, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("required radii must be a finite nonempty vector")
    if not np.isfinite(slope_limit) or slope_limit <= 0:
        raise ValueError("slope_limit must be finite and positive")
    station = np.arange(len(values), dtype=np.float64)
    envelope = np.max(
        values[None, :] - slope_limit * np.abs(station[:, None] - station[None, :]),
        axis=1,
    )
    kernel = np.asarray([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    rounded = np.convolve(np.pad(envelope, 2, mode="edge"), kernel, mode="valid")
    result = np.maximum(envelope, rounded)
    if np.any(result + 1e-10 < values):
        raise RuntimeError("width envelope lost containment")
    if np.max(np.abs(np.diff(result))) > slope_limit + 1e-10:
        raise RuntimeError("width envelope violated slope limit")
    return result


def rerun_backbone(mask: np.ndarray, n_points: int) -> dict[str, object]:
    """Run thinning, endpoint peeling, longest path, and resampling on a mask."""

    yy, xx = np.nonzero(mask)
    pad = 3
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(mask.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(mask.shape[1], int(xx.max()) + pad + 1)
    thinned_crop = _thin(mask[y0:y1, x0:x1])
    skeleton_crop = _prune_skeleton_endpoints(thinned_crop)
    path, endpoint_count, branch_pixels = _skeleton_longest_path(skeleton_crop)
    if path is None:
        raise RuntimeError("modeled mask has no endpoint-to-endpoint skeleton path")
    path[:, 0] += x0
    path[:, 1] += y0
    centerline = resample_centerline(path, n_points)
    thinned = np.zeros_like(mask)
    thinned[y0:y1, x0:x1] = thinned_crop
    skeleton = np.zeros_like(mask)
    skeleton[y0:y1, x0:x1] = skeleton_crop
    return {
        "thinned": thinned,
        "skeleton": skeleton,
        "path": path,
        "centerline": centerline,
        "endpoint_count": endpoint_count,
        "branch_pixels": branch_pixels,
    }


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def load_postfit_annotation(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load the complete one-annotator trace without exposing it to the fit."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    sample_id = f"{RECORDING}-f{FRAME_INDEX:06d}"
    matches = [row for row in payload["annotations"] if row.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one audit annotation for {sample_id}")
    row = matches[0]
    if row.get("trace_state") != "complete" or row.get("annotation_pass") != "primary":
        raise RuntimeError("audit annotation must be a complete primary trace")
    if row.get("single_annotator_protocol") is not True:
        raise RuntimeError("audit annotation is not marked as single-annotator protocol")
    points = np.asarray([vertex["xy"] for vertex in row["vertices"]], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise RuntimeError("audit annotation has invalid vertices")
    metadata = {
        "path": str(path),
        "sample_id": sample_id,
        "annotation_id": str(row["annotation_id"]),
        "annotator_id": str(row["annotator_id"]),
        "trace_state": str(row["trace_state"]),
        "annotation_pass": str(row["annotation_pass"]),
        "used_during_fit": False,
    }
    return points, metadata


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.remainder(value + np.pi, 2 * np.pi) - np.pi


def _curve_tangent(points: np.ndarray) -> np.ndarray:
    derivative = np.empty_like(points)
    derivative[0] = points[1] - points[0]
    derivative[-1] = points[-1] - points[-2]
    derivative[1:-1] = points[2:] - points[:-2]
    return np.arctan2(derivative[:, 1], derivative[:, 0])


def complete_curve_metrics(
    prediction_xy: np.ndarray, annotation_xy: np.ndarray, *, num_points: int = 100
) -> dict[str, object]:
    """Repo-consistent, orientation-symmetric metrics for a complete trace."""

    prediction = resample_polyline(prediction_xy, num_points)
    target = resample_polyline(annotation_xy, num_points)
    forward = np.linalg.norm(prediction - target, axis=1)
    reverse = np.linalg.norm(prediction - target[::-1], axis=1)
    target_reversed = bool(reverse.mean() < forward.mean())
    if target_reversed:
        target = target[::-1]
    distance = np.linalg.norm(prediction - target, axis=1)
    tangent_error = np.rad2deg(
        np.abs(_wrap_angle(_curve_tangent(prediction) - _curve_tangent(target)))
    )
    prediction_length = float(np.linalg.norm(np.diff(prediction, axis=0), axis=1).sum())
    target_length = float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum())
    return {
        "target_reversed": target_reversed,
        "median_point_distance_px": float(np.median(distance)),
        "mean_point_distance_px": float(np.mean(distance)),
        "p95_point_distance_px": float(np.percentile(distance, 95)),
        "mean_tangent_error_deg": float(tangent_error.mean()),
        "p95_tangent_error_deg": float(np.percentile(tangent_error, 95)),
        "mean_endpoint_error_px": float(distance[[0, -1]].mean()),
        "body_length_error_px": abs(prediction_length - target_length),
        "body_length_error_fraction": abs(prediction_length - target_length)
        / max(target_length, np.finfo(float).eps),
    }


def main() -> None:
    args = parse_args()
    if args.latent_coefficients < 4:
        raise ValueError("latent-coefficients must be at least 4")
    if args.containment_margin < 0 or not np.isfinite(args.containment_margin):
        raise ValueError("containment-margin must be finite and non-negative")
    if not np.isfinite(args.max_full_width) or args.max_full_width <= 0:
        raise ValueError("max-full-width must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, provenance = load_real_frame(args.proxy_hdf5)
    cfg = ClassicalConfig()
    score = robust_dark_ridge(frame, cfg)
    raw = score >= cfg.foreground_z
    closed = _erode(_dilate(raw, cfg.close_radius), cfg.close_radius)
    component, _, component_count = _largest_component(closed)
    bounds = crop_bounds(component)

    original_path, original_skeleton = initial_path(component)
    original_centerline = resample_centerline(original_path, cfg.n_points)
    initialization_mask, enclosed_cavities, enclosed_cavity_count = fill_enclosed_cavities(
        component
    )
    path0, initial_skeleton = initial_path(initialization_mask)
    initial_centerline = resample_centerline(path0, cfg.n_points)
    latent = encode_centerline(initial_centerline, args.latent_coefficients)
    smooth_midline = decode_centerline(latent, args.latent_coefficients)

    required_radius, positive_xy, assignment = nearest_station_requirements(
        component, smooth_midline, args.containment_margin
    )
    fitted_radius = slowly_varying_majorant(required_radius, args.radius_slope_limit)
    fitted_width = 2.0 * fitted_radius
    modeled_mask = render_centerline_mask(smooth_midline, fitted_width, component.shape)
    missed_positives = int(np.logical_and(component, ~modeled_mask).sum())
    if missed_positives:
        raise RuntimeError(f"modeled mask missed {missed_positives} detected positives")

    recovered = rerun_backbone(modeled_mask, cfg.n_points)
    recovered_centerline = np.asarray(recovered["centerline"])
    recovered_skeleton = np.asarray(recovered["skeleton"])
    recovered_path = np.asarray(recovered["path"])
    original_topology = skeleton_topology(original_skeleton)
    filled_initial_topology = skeleton_topology(initial_skeleton)
    topology = skeleton_topology(recovered_skeleton)
    annotation_xy, annotation_metadata = load_postfit_annotation(args.annotations)
    audit_before = complete_curve_metrics(original_centerline, annotation_xy)
    audit_filled_initialization = complete_curve_metrics(initial_centerline, annotation_xy)
    audit_after = complete_curve_metrics(recovered_centerline, annotation_xy)

    length = float(np.linalg.norm(np.diff(recovered_centerline, axis=0), axis=1).sum())
    height, width = frame.shape
    boundary_distance = float(
        np.min(
            np.column_stack(
                (
                    recovered_centerline[:, 0],
                    recovered_centerline[:, 1],
                    (width - 1) - recovered_centerline[:, 0],
                    (height - 1) - recovered_centerline[:, 1],
                )
            )
        )
    )
    sample_x = np.clip(np.rint(recovered_centerline[:, 0]).astype(int), 0, width - 1)
    sample_y = np.clip(np.rint(recovered_centerline[:, 1]).astype(int), 0, height - 1)
    tube_support = float(np.mean(_dilate(modeled_mask, 1)[sample_y, sample_x]))
    boundary_contact = bool(
        np.any(modeled_mask[0])
        or np.any(modeled_mask[-1])
        or np.any(modeled_mask[:, 0])
        or np.any(modeled_mask[:, -1])
    )
    reasons: list[str] = []
    modeled_area = int(modeled_mask.sum())
    width_feasible = bool(float(fitted_width.max()) <= args.max_full_width + 1e-10)
    capped_width = np.minimum(fitted_width, args.max_full_width)
    capped_mask = render_centerline_mask(smooth_midline, capped_width, component.shape)
    capped_missed_positives = int(np.logical_and(component, ~capped_mask).sum())
    newly_filled = np.logical_and(modeled_mask, ~component)
    added_local_darkness_z0 = float(np.mean(score[newly_filled] >= 0.0))
    added_original_threshold_z26 = float(
        np.mean(score[newly_filled] >= cfg.foreground_z)
    )
    if not cfg.min_area <= modeled_area <= MODELED_BODY_MAX_AREA:
        reasons.append("implausible_area")
    if not cfg.min_length <= length <= cfg.max_length:
        reasons.append("implausible_length")
    if boundary_distance < cfg.boundary_margin or boundary_contact:
        reasons.append("boundary_contact")
    endpoint_count = int(recovered["endpoint_count"])
    branch_pixels = int(recovered["branch_pixels"])
    if (
        endpoint_count < 2
        or endpoint_count > cfg.max_raw_endpoints
        or branch_pixels > cfg.max_branch_pixels
    ):
        reasons.append("unstable_endpoints")
    if tube_support < cfg.min_tube_support:
        reasons.append("low_tube_support")
    declared_reasons = list(reasons)
    if not width_feasible:
        declared_reasons.append("full_width_exceeds_80_px_feasibility_ceiling")
    length_score = max(0.0, 1.0 - abs(length - 450.0) / 300.0)
    topology_score = max(0.0, 1.0 - branch_pixels / max(cfg.max_branch_pixels, 1))
    boundary_score = min(1.0, boundary_distance / 20.0)
    quality = float(
        np.clip(
            0.4 * tube_support
            + 0.2 * length_score
            + 0.2 * topology_score
            + 0.2 * boundary_score,
            0,
            1,
        )
    )

    metrics: dict[str, object] = {
        "status": "one_frame_auditable_geometry_experiment",
        "provenance": provenance,
        "parameters": {
            "local_radius": cfg.local_radius,
            "smooth_radius": cfg.smooth_radius,
            "foreground_z": cfg.foreground_z,
            "close_radius": cfg.close_radius,
            "n_body_stations": cfg.n_points,
            "latent_angle_coefficients": args.latent_coefficients,
            "width_assignment": "nearest smooth-midline station",
            "containment_margin_px": args.containment_margin,
            "radius_max_adjacent_change_px": args.radius_slope_limit,
            "max_full_width_px": args.max_full_width,
            "modeled_body_max_area_px": MODELED_BODY_MAX_AREA,
            "skeleton_endpoint_peel_iterations": 8,
        },
        "section_3_input": {
            "positive_pixel_count": int(component.sum()),
            "connected_component_count_before_keep_largest": int(component_count),
        },
        "topology_prior_initialization": {
            "rule": "fill only background cavities disconnected from the image-border exterior",
            "open_u_center_preserved": True,
            "enclosed_cavity_count": int(enclosed_cavity_count),
            "enclosed_pixels_filled_for_initialization": int(enclosed_cavities.sum()),
            "initialization_area_px": int(initialization_mask.sum()),
            "width_fit_target_remains_original_section3_component": True,
        },
        "body_prior_fit": {
            "latent_dimension": int(len(latent)),
            "required_radius_min_px": float(required_radius.min()),
            "required_radius_median_px": float(np.median(required_radius)),
            "required_radius_max_px": float(required_radius.max()),
            "fitted_full_width_min_px": float(fitted_width.min()),
            "fitted_full_width_median_px": float(np.median(fitted_width)),
            "fitted_full_width_max_px": float(fitted_width.max()),
            "fitted_radius_max_adjacent_change_px": float(np.abs(np.diff(fitted_radius)).max()),
            "full_width_feasible": width_feasible,
            "width_feasibility_excess_px": max(0.0, float(fitted_width.max() - args.max_full_width)),
            "fit_status": "feasible" if width_feasible else "infeasible_exact_containment_fit",
            "detected_positive_pixels_missed": missed_positives,
            "detected_positive_recall": float(np.mean(modeled_mask[component])),
            "width_capped_diagnostic_positive_pixels_missed": capped_missed_positives,
            "width_capped_diagnostic_positive_recall": float(np.mean(capped_mask[component])),
            "modeled_area_px": modeled_area,
            "newly_filled_pixel_count": int(newly_filled.sum()),
            "newly_filled_local_darkness_z0_fraction": added_local_darkness_z0,
            "newly_filled_original_threshold_z2_6_fraction": added_original_threshold_z26,
            "mask_iou_with_section_3_component": mask_iou(modeled_mask, component),
        },
        "rerun_result": {
            "accepted_by_revised_geometric_gates": not reasons,
            "rejection_reasons": reasons,
            "accepted_by_all_declared_constraints": not declared_reasons,
            "all_declared_rejection_reasons": declared_reasons,
            "area_gate": {
                "passed": bool(cfg.min_area <= modeled_area <= MODELED_BODY_MAX_AREA),
                "value_px": modeled_area,
                "allowed_px": [cfg.min_area, MODELED_BODY_MAX_AREA],
            },
            "full_width_feasibility_gate": {
                "passed": width_feasible,
                "value_max_px": float(fitted_width.max()),
                "maximum_px": args.max_full_width,
            },
            "centerline_length_px": length,
            "boundary_distance_px": boundary_distance,
            "tube_support_fraction_on_modeled_mask": tube_support,
            "longest_path_endpoint_count": endpoint_count,
            "longest_path_branch_pixels": branch_pixels,
            "skeleton_topology": topology,
            "geometric_quality_score": quality,
        },
        "topology_before_after": {
            "original_section3_pruned_skeleton": original_topology,
            "filled_initialization_pruned_skeleton": filled_initial_topology,
            "after_completed_body_pruned_skeleton": topology,
        },
        "manual_trace_postfit_audit": {
            "annotation": annotation_metadata,
            "role": "post_fit_audit_only; never used by initialization, midline fit, width fit, or parameter selection",
            "original_pipeline_pose": audit_before,
            "filled_initialization_pose": audit_filled_initialization,
            "completed_body_pose": audit_after,
        },
        "limitations": [
            "The initial path comes from the section-3 evidence after enclosed-cavity filling, so this does not rescue a missing or grossly wrong backbone.",
            "Containment prevents false negatives relative to section 3 but can add false-positive background pixels.",
            "The parameters were tried on this single frame and are not validated on other frames or manual traces.",
            "The one-annotator trace is an after-the-fact diagnostic and was excluded from every fitting and parameter choice.",
            "The centerline is defined geometrically from the body border; no centerline darkness metric is used.",
        ],
    }

    # 0 — exact section-3 component, now isolated as the experiment input.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    overlay_mask(ax, component, CYAN, 0.68)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "00_section3_input.png")

    # 1 — make the topology prior explicit: fill enclosed cavities, not the open U.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    overlay_mask(ax, component, CYAN, 0.50)
    overlay_mask(ax, enclosed_cavities, ORANGE, 0.88)
    ax.text(
        0.015, 0.025,
        f"orange: {int(enclosed_cavities.sum()):,} enclosed pixels filled only for initialization\n"
        "open U-shaped background is connected to the exterior and stays unfilled",
        transform=ax.transAxes, color="white", fontsize=10.5,
        bbox={"facecolor": "black", "alpha": 0.66, "edgecolor": "none", "pad": 6},
    )
    set_crop(ax, bounds)
    save(fig, args.output_dir / "01_enclosed_hole_initialization.png")

    # 2 — filled-mask initialization and its low-dimensional smooth reconstruction.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    overlay_mask(ax, initialization_mask, CYAN, 0.35)
    ax.plot(initial_centerline[:, 0], initial_centerline[:, 1], color=GRAY, linewidth=4.0,
            label="skeleton initialization")
    ax.plot(smooth_midline[:, 0], smooth_midline[:, 1], color=ORANGE, linewidth=2.5,
            label=f"smooth {args.latent_coefficients}-coefficient midline")
    ax.legend(loc="upper right", framealpha=0.85)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "02_smooth_latent_midline.png")

    # 3 — raw containment radii and the slowly varying upper envelope.
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    station = np.arange(cfg.n_points)
    ax.plot(station, 2 * required_radius, color=MAGENTA, linewidth=1.3,
            label="minimum full width needed to contain positives")
    ax.plot(station, fitted_width, color=ORANGE, linewidth=3.0,
            label="slowly varying fitted full width")
    ax.fill_between(station, 0, fitted_width, color=ORANGE, alpha=0.12)
    ax.set(xlabel="body station (0–99)", ylabel="full width (pixels)")
    ax.grid(alpha=0.2)
    ax.legend(framealpha=0.9)
    save(fig, args.output_dir / "03_width_profile.png")

    # 4 — fitted tube. Green is retained evidence; orange is prior-filled area.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    overlay_mask(ax, modeled_mask & ~component, ORANGE, 0.62)
    overlay_mask(ax, component, GREEN, 0.68)
    ax.plot(smooth_midline[:, 0], smooth_midline[:, 1], color=PALE_CYAN, linewidth=2.0)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "04_completed_body_model.png")

    # 5 — rerun thinning and endpoint peeling on the completed body.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(modeled_mask, cmap="gray", vmin=0, vmax=1)
    overlay_mask(ax, np.asarray(recovered["thinned"]) & ~recovered_skeleton, RED, 0.9)
    overlay_mask(ax, recovered_skeleton, PALE_CYAN, 1.0)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "05_rerun_skeleton.png")

    # 6 — rerun longest path.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    overlay_mask(ax, recovered_skeleton, GRAY, 0.72)
    ax.plot(recovered_path[:, 0], recovered_path[:, 1], color=ORANGE, linewidth=3.0)
    ax.scatter(recovered_path[[0, -1], 0], recovered_path[[0, -1], 1], s=75,
               c=GREEN, edgecolors="black")
    set_crop(ax, bounds)
    save(fig, args.output_dir / "06_rerun_longest_path.png")

    # 7 — rerun resampling and tangent output.
    angles = tangent_angles(recovered_centerline)
    arrow_indices = np.arange(4, len(recovered_centerline) - 4, 9)
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(recovered_centerline[:, 0], recovered_centerline[:, 1], color=ORANGE,
            linewidth=3.0)
    ax.scatter(recovered_centerline[:, 0], recovered_centerline[:, 1], s=14,
               c=PALE_CYAN, linewidths=0)
    ax.quiver(
        recovered_centerline[arrow_indices, 0], recovered_centerline[arrow_indices, 1],
        np.cos(angles[arrow_indices]), np.sin(angles[arrow_indices]), color=GREEN,
        angles="xy", scale_units="xy", scale=0.045, width=0.005,
    )
    set_crop(ax, bounds)
    save(fig, args.output_dir / "07_rerun_pose.png")

    # 8 — direct before/after centerline comparison and gate outcome.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(original_centerline[:, 0], original_centerline[:, 1], color=MAGENTA,
            linewidth=4.5, alpha=0.78, label="original pipeline pose")
    ax.plot(recovered_centerline[:, 0], recovered_centerline[:, 1], color=ORANGE,
            linewidth=2.5, label="pose after smooth-body completion")
    gate_text = (
        f"{'PASS' if not declared_reasons else 'REJECT'}\n"
        f"area {modeled_area:,} px²\nlength {length:.1f} px\n"
        f"inside body mask {100 * tube_support:.0f}%\n"
        f"positive recall {100 * np.mean(modeled_mask[component]):.0f}%"
    )
    ax.text(0.73, 0.97, gate_text, transform=ax.transAxes, va="top", color="white",
            fontsize=10.5, linespacing=1.3,
            bbox={"facecolor": "#12643b" if not declared_reasons else "#8a2222", "alpha": 0.9,
                  "edgecolor": GREEN if not declared_reasons else RED, "pad": 7})
    ax.legend(loc="upper left", framealpha=0.86)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "08_before_after_and_gate.png")

    # 9 — the manual trace appears only after all fitting and gate decisions.
    audit_target = resample_polyline(annotation_xy, cfg.n_points)
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(audit_target[:, 0], audit_target[:, 1], color=GREEN, linewidth=4.5,
            label="one-annotator trace (post-fit audit only)")
    ax.plot(original_centerline[:, 0], original_centerline[:, 1], color=MAGENTA,
            linewidth=3.0, label=f"before: median {audit_before['median_point_distance_px']:.1f} px")
    ax.plot(recovered_centerline[:, 0], recovered_centerline[:, 1], color=ORANGE,
            linewidth=2.4, label=f"after: median {audit_after['median_point_distance_px']:.1f} px")
    ax.legend(loc="upper left", framealpha=0.88)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "09_manual_trace_postfit_audit.png")

    np.savez_compressed(
        args.output_dir / "experiment_arrays.npz",
        section3_component=component,
        enclosed_cavities=enclosed_cavities,
        filled_initialization_mask=initialization_mask,
        original_centerline_xy=original_centerline,
        initial_centerline_xy=initial_centerline,
        smooth_midline_xy=smooth_midline,
        required_radius_px=required_radius,
        fitted_radius_px=fitted_radius,
        width_capped_mask=capped_mask,
        modeled_mask=modeled_mask,
        rerun_skeleton=recovered_skeleton,
        rerun_longest_path_xy=recovered_path,
        rerun_centerline_xy=recovered_centerline,
        postfit_annotation_xy=annotation_xy,
        positive_xy=positive_xy,
        positive_nearest_station=assignment,
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
