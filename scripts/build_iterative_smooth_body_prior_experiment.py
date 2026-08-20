#!/usr/bin/env python3
"""Iteratively recompute the medial path of the modeled worm body.

The manual trace is loaded only after the deterministic fit has stopped.  This
script writes to a separate output directory and does not alter the one-pass
smooth-body experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_smooth_body_prior_experiment import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_PROXY_HDF5,
    FRAME_INDEX,
    GREEN,
    MAGENTA,
    ORANGE,
    PALE_CYAN,
    RECORDING,
    complete_curve_metrics,
    crop_bounds,
    fill_enclosed_cavities,
    initial_path,
    load_postfit_annotation,
    load_real_frame,
    mask_iou,
    nearest_station_requirements,
    overlay_mask,
    rerun_backbone,
    save,
    set_crop,
    slowly_varying_majorant,
)
from worm_pose_gen.anchors import render_centerline_mask, skeleton_topology
from worm_pose_gen.annotation import resample_polyline
from worm_pose_gen.classical import (
    ClassicalConfig,
    _dilate,
    _erode,
    _largest_component,
    resample_centerline,
    robust_dark_ridge,
)
from worm_pose_gen.latent import decode_centerline, encode_centerline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "iterative_smooth_body_prior_experiment"
MODELED_BODY_MAX_AREA = 40_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-hdf5", type=Path, default=DEFAULT_PROXY_HDF5)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latent-coefficients", type=int, default=16)
    parser.add_argument("--radius-slope-limit", type=float, default=1.0)
    parser.add_argument("--containment-margin", type=float, default=0.75)
    parser.add_argument("--max-full-width", type=float, default=80.0)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--stable-mask-fraction",
        type=float,
        default=1e-4,
        help="maximum changed-frame-pixel fraction for a stable update",
    )
    parser.add_argument(
        "--stable-midline-displacement",
        type=float,
        default=0.25,
        help="maximum station displacement in pixels for a stable update",
    )
    parser.add_argument("--stable-updates", type=int, default=2)
    return parser.parse_args()


def orient_to_reference(curve: np.ndarray, reference: np.ndarray) -> np.ndarray:
    forward = np.linalg.norm(curve - reference, axis=1).mean()
    reverse = np.linalg.norm(curve[::-1] - reference, axis=1).mean()
    return curve[::-1].copy() if reverse < forward else curve.copy()


def curve_distance(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    aligned = orient_to_reference(np.asarray(first), np.asarray(second))
    distance = np.linalg.norm(aligned - second, axis=1)
    return {
        "mean_px": float(distance.mean()),
        "median_px": float(np.median(distance)),
        "max_px": float(distance.max()),
    }


def gate_and_measure(
    mask: np.ndarray,
    midline: np.ndarray,
    full_width: np.ndarray,
    component: np.ndarray,
    score: np.ndarray,
    cfg: ClassicalConfig,
    frozen_pose: np.ndarray | None,
    max_full_width: float,
) -> tuple[dict[str, object], dict[str, object]]:
    recovered = rerun_backbone(mask, cfg.n_points)
    pose = np.asarray(recovered["centerline"])
    if frozen_pose is not None:
        pose = orient_to_reference(pose, frozen_pose)
    skeleton = np.asarray(recovered["skeleton"])
    topology = skeleton_topology(skeleton)
    length = float(np.linalg.norm(np.diff(pose, axis=0), axis=1).sum())
    height, image_width = mask.shape
    boundary_distance = float(
        np.column_stack(
            (
                pose[:, 0], pose[:, 1],
                (image_width - 1) - pose[:, 0],
                (height - 1) - pose[:, 1],
            )
        ).min()
    )
    added = mask & ~component
    area = int(mask.sum())
    reasons: list[str] = []
    if not cfg.min_area <= area <= MODELED_BODY_MAX_AREA:
        reasons.append("implausible_area")
    if not cfg.min_length <= length <= cfg.max_length:
        reasons.append("implausible_length")
    if boundary_distance < cfg.boundary_margin:
        reasons.append("boundary_contact")
    if int(topology["endpoint_count"]) < 2 or int(topology["endpoint_count"]) > cfg.max_raw_endpoints:
        reasons.append("unstable_endpoints")
    if int(topology["branch_pixels"]) > cfg.max_branch_pixels:
        reasons.append("unstable_endpoints")
    if float(full_width.max()) > max_full_width:
        reasons.append("width_above_repo_ceiling")
    measurement: dict[str, object] = {
        "mask_area_px": area,
        "added_area_px": int(added.sum()),
        "section3_positive_recall": float(np.mean(mask[component])),
        "mask_iou_with_section3": mask_iou(mask, component),
        "added_local_darkness_z0_fraction": (
            float(np.mean(score[added] >= 0.0)) if np.any(added) else 1.0
        ),
        "added_original_threshold_z2_6_fraction": (
            float(np.mean(score[added] >= cfg.foreground_z)) if np.any(added) else 1.0
        ),
        "full_width_min_px": float(full_width.min()),
        "full_width_median_px": float(np.median(full_width)),
        "full_width_max_px": float(full_width.max()),
        "full_width_max_adjacent_change_px": float(np.abs(np.diff(full_width)).max()),
        "centerline_length_px": length,
        "boundary_distance_px": boundary_distance,
        "topology": topology,
        "accepted_by_revised_modeled_body_gates": not reasons,
        "modeled_body_area_allowed_px": [cfg.min_area, MODELED_BODY_MAX_AREA],
        "rejection_reasons": list(dict.fromkeys(reasons)),
    }
    if frozen_pose is not None:
        measurement["pose_displacement_from_frozen_completed_body"] = curve_distance(
            pose, frozen_pose
        )
    recovered["centerline"] = pose
    return measurement, recovered


def main() -> None:
    args = parse_args()
    if args.max_iterations < 1 or args.stable_updates < 1:
        raise ValueError("iteration counts must be positive")
    if not 0 <= args.stable_mask_fraction < 1:
        raise ValueError("stable-mask-fraction must lie in [0,1)")
    if args.stable_midline_displacement < 0:
        raise ValueError("stable-midline-displacement must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, provenance = load_real_frame(args.proxy_hdf5)
    cfg = ClassicalConfig()
    score = robust_dark_ridge(frame, cfg)
    raw = score >= cfg.foreground_z
    closed = _erode(_dilate(raw, cfg.close_radius), cfg.close_radius)
    component, _, component_count = _largest_component(closed)
    bounds = crop_bounds(component)

    original_path, _ = initial_path(component)
    original_pose = resample_centerline(original_path, cfg.n_points)

    initialization_mask, enclosed, enclosed_count = fill_enclosed_cavities(component)
    path0, _ = initial_path(initialization_mask)
    initial_pose = resample_centerline(path0, cfg.n_points)
    midline = decode_centerline(
        encode_centerline(initial_pose, args.latent_coefficients), args.latent_coefficients
    )
    required, _, _ = nearest_station_requirements(component, midline, args.containment_margin)
    radius = slowly_varying_majorant(required, args.radius_slope_limit)
    full_width = 2.0 * radius
    if float(full_width.max()) > args.max_full_width:
        raise RuntimeError("one-pass completed body violates the configured width ceiling")
    mask = render_centerline_mask(midline, full_width, component.shape)
    if not bool(np.all(mask[component])):
        raise RuntimeError("one-pass completed body lost Section-3 positives")

    frozen_metrics, frozen_recovered = gate_and_measure(
        mask, midline, full_width, component, score, cfg, None, args.max_full_width
    )
    frozen_pose = np.asarray(frozen_recovered["centerline"])
    frozen_mask = mask.copy()
    frozen_midline = midline.copy()
    frozen_width = full_width.copy()

    records: list[dict[str, object]] = []
    negative_candidate_poses: list[np.ndarray] = []
    states: list[dict[str, np.ndarray]] = [
        {"midline": midline.copy(), "width": full_width.copy(), "mask": mask.copy(),
         "pose": frozen_pose.copy()}
    ]
    stable_count = 0
    stop_reason = "maximum_iterations_reached"
    seen_masks = {np.packbits(mask).tobytes(): 0}
    previous_pose = frozen_pose.copy()

    for iteration in range(1, args.max_iterations + 1):
        recentered = orient_to_reference(previous_pose, midline)
        candidate_midline = decode_centerline(
            encode_centerline(recentered, args.latent_coefficients), args.latent_coefficients
        )
        candidate_midline = orient_to_reference(candidate_midline, midline)
        required, _, _ = nearest_station_requirements(
            component, candidate_midline, args.containment_margin
        )
        candidate_radius = slowly_varying_majorant(required, args.radius_slope_limit)
        candidate_width = 2.0 * candidate_radius
        if float(candidate_width.max()) > args.max_full_width:
            stop_reason = "infeasible_width_ceiling"
            break
        candidate_mask = render_centerline_mask(
            candidate_midline, candidate_width, component.shape
        )
        if not bool(np.all(candidate_mask[component])):
            raise RuntimeError("iterative refit lost Section-3 positives")
        measurement, recovered = gate_and_measure(
            candidate_mask, candidate_midline, candidate_width,
            component, score, cfg, frozen_pose, args.max_full_width,
        )
        candidate_pose = np.asarray(recovered["centerline"])
        negative_candidate_poses.append(candidate_pose.copy())
        mask_changed = int(np.logical_xor(candidate_mask, mask).sum())
        mask_changed_fraction = mask_changed / candidate_mask.size
        update_displacement = curve_distance(candidate_midline, midline)
        stable = bool(
            mask_changed_fraction <= args.stable_mask_fraction
            and update_displacement["max_px"] <= args.stable_midline_displacement
        )
        stable_count = stable_count + 1 if stable else 0
        key = np.packbits(candidate_mask).tobytes()
        cycle_from = seen_masks.get(key)
        record = {
            "iteration": iteration,
            **measurement,
            "changed_pixels_from_previous": mask_changed,
            "changed_frame_fraction_from_previous": mask_changed_fraction,
            "latent_midline_update_from_previous": update_displacement,
            "stable_update": stable,
            "consecutive_stable_updates": stable_count,
            "repeated_prior_mask_iteration": cycle_from,
        }
        area_improved = bool(int(candidate_mask.sum()) < int(mask.sum()))
        record["strict_area_decrease_from_previous"] = area_improved
        record["accepted_update"] = area_improved
        records.append(record)
        if not area_improved:
            stop_reason = "no_strict_area_decrease"
            break
        states.append(
            {"midline": candidate_midline.copy(), "width": candidate_width.copy(),
             "mask": candidate_mask.copy(), "pose": candidate_pose.copy()}
        )
        midline, full_width, mask = candidate_midline, candidate_width, candidate_mask
        previous_pose = candidate_pose
        if stable_count >= args.stable_updates:
            stop_reason = "two_consecutive_stable_updates"
            break
        if cycle_from is not None:
            stop_reason = "repeated_mask_cycle"
            break
        seen_masks[key] = iteration

    final_state = states[-1]
    final_pose = final_state["pose"]
    final_mask = final_state["mask"]
    final_width = final_state["width"]
    final_midline = final_state["midline"]

    # The annotation is deliberately unavailable until fitting and stopping end.
    annotation, annotation_metadata = load_postfit_annotation(args.annotations)
    trace_frozen = complete_curve_metrics(frozen_pose, annotation)
    trace_final = complete_curve_metrics(final_pose, annotation)
    trace_original = complete_curve_metrics(original_pose, annotation)
    for record, candidate_pose in zip(records, negative_candidate_poses, strict=True):
        record["manual_trace_postfit_only"] = complete_curve_metrics(candidate_pose, annotation)

    metrics = {
        "status": "one_frame_iterative_geometry_audit",
        "provenance": provenance,
        "manual_trace_isolation": {
            "loaded_after_fit_stopped": True,
            "used_during_fit_or_parameter_selection": False,
            "annotation": annotation_metadata,
        },
        "parameters": {
            "latent_coefficients": args.latent_coefficients,
            "containment_margin_px": args.containment_margin,
            "radius_max_adjacent_change_px": args.radius_slope_limit,
            "max_full_width_px": args.max_full_width,
            "modeled_body_max_area_px": MODELED_BODY_MAX_AREA,
            "modeled_body_area_rationale": (
                "rendered completion includes prior-recovered pixels; the historical "
                "30000 px ceiling remains specific to the raw-component extractor"
            ),
            "max_iterations": args.max_iterations,
            "stable_mask_fraction": args.stable_mask_fraction,
            "stable_midline_max_displacement_px": args.stable_midline_displacement,
            "required_consecutive_stable_updates": args.stable_updates,
            "centerline_definition": (
                "the smoothed medial path of the modeled body: the geometric midpoint "
                "between its two sides; no image-darkness value is sampled at the midline"
            ),
            "stopping_rule": (
                "accept an update only when it preserves exact Section-3 containment, "
                "respects the width ceiling, and strictly decreases rendered area; stop "
                "at the first non-decreasing candidate. Also stop after two consecutive "
                "updates with changed-frame-pixel fraction <= "
                "stable_mask_fraction and maximum latent-midline station displacement <= "
                "stable_midline_max_displacement_px; otherwise stop on a repeated mask, "
                "width infeasibility, or the fixed iteration cap"
            ),
        },
        "section3_input": {
            "positive_pixel_count": int(component.sum()),
            "component_count_before_keep_largest": component_count,
            "enclosed_cavity_count_for_initialization_only": enclosed_count,
            "enclosed_pixels_for_initialization_only": int(enclosed.sum()),
        },
        "frozen_one_pass_completed_body": frozen_metrics,
        "boundary_midpoint_iteration": {
            "iterations": records,
            "stop": {
                "reason": stop_reason,
                "updates_attempted": len(records),
                "updates_accepted": len(states) - 1,
                "converged_by_stability_rule": stop_reason == "two_consecutive_stable_updates",
            },
            "final_vs_frozen": {
                "area_change_px": int(final_mask.sum()) - int(frozen_mask.sum()),
                "mask_iou": mask_iou(final_mask, frozen_mask),
                "pose_displacement": curve_distance(final_pose, frozen_pose),
                "latent_midline_displacement": curve_distance(final_midline, frozen_midline),
                "max_full_width_change_px": float(final_width.max() - frozen_width.max()),
            },
        },
        "manual_trace_postfit_only": {
            "original_classical_pose": trace_original,
            "frozen_one_pass_completed_body_pose": trace_frozen,
            "boundary_midpoint_iteration_pose": trace_final,
        },
        "limitations": [
            "This is a deterministic audit on one already accepted, fully visible frame.",
            "Exact containment preserves Section-3 positives but does not prove that added pixels are worm.",
            "The stopping thresholds are declared engineering tolerances, not validated biological constants.",
            "The iteration recomputes the midpoint of its own modeled boundaries; it adds no new border evidence.",
            "The manual trace was used only after fitting stopped and therefore cannot justify parameter choices.",
        ],
    }

    accepted_records = [row for row in records if bool(row["accepted_update"])]
    iteration_numbers = [0] + [int(row["iteration"]) for row in accepted_records]
    areas = [int(frozen_metrics["mask_area_px"])] + [
        int(row["mask_area_px"]) for row in accepted_records
    ]
    max_widths = [float(frozen_metrics["full_width_max_px"])] + [
        float(row["full_width_max_px"]) for row in accepted_records
    ]
    displacements = [0.0] + [
        float(row["pose_displacement_from_frozen_completed_body"]["median_px"])
        for row in accepted_records
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].plot(iteration_numbers, areas, marker="o", color=ORANGE)
    axes[0].axhline(
        MODELED_BODY_MAX_AREA, color=MAGENTA, linestyle="--", label="area gate"
    )
    axes[0].set(xlabel="iteration", ylabel="modeled area (px²)")
    axes[0].legend()
    axes[1].plot(iteration_numbers, max_widths, marker="o", color=GREEN)
    axes[1].axhline(
        args.max_full_width, color=MAGENTA, linestyle="--", label="width ceiling"
    )
    axes[1].set(xlabel="iteration", ylabel="maximum full width (px)")
    axes[1].legend()
    axes[2].plot(iteration_numbers, displacements, marker="o", color=PALE_CYAN)
    axes[2].set(xlabel="iteration", ylabel="median pose shift from frozen (px)")
    for ax in axes:
        ax.grid(alpha=0.2)
    save(fig, args.output_dir / "00_iteration_metrics.png")

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    overlay_mask(ax, final_mask & ~component, ORANGE, 0.55)
    overlay_mask(ax, component, GREEN, 0.55)
    ax.plot(frozen_pose[:, 0], frozen_pose[:, 1], color=MAGENTA, linewidth=3.5,
            label="frozen one-pass completed-body pose")
    ax.plot(final_pose[:, 0], final_pose[:, 1], color=PALE_CYAN, linewidth=2.3,
            label="boundary-midpoint iteration")
    ax.legend(loc="upper left", framealpha=0.88)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "01_final_vs_frozen.png")

    target = resample_polyline(annotation, cfg.n_points)
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(target[:, 0], target[:, 1], color=GREEN, linewidth=4,
            label="manual trace (post-fit only)")
    ax.plot(frozen_pose[:, 0], frozen_pose[:, 1], color=MAGENTA, linewidth=3,
            label=f"frozen: {trace_frozen['median_point_distance_px']:.2f} px median")
    ax.plot(final_pose[:, 0], final_pose[:, 1], color=ORANGE, linewidth=2.4,
            label=f"boundary midpoint: {trace_final['median_point_distance_px']:.2f} px median")
    ax.legend(loc="upper left", framealpha=0.88)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "02_manual_trace_postfit_only.png")

    np.savez_compressed(
        args.output_dir / "iteration_arrays.npz",
        section3_component=component,
        frozen_mask=frozen_mask,
        frozen_midline_xy=frozen_midline,
        frozen_width_px=frozen_width,
        frozen_pose_xy=frozen_pose,
        final_mask=final_mask,
        final_midline_xy=final_midline,
        final_width_px=final_width,
        final_pose_xy=final_pose,
        manual_trace_postfit_xy=annotation,
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
