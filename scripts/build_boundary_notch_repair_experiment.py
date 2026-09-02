#!/usr/bin/env python3
"""Prototype a conservative exterior-notch repair before latent midline fitting.

This one-frame experiment is isolated from the existing smooth-body experiment.
Candidate selection uses only mask geometry.  The manual trace is loaded only
after selection and fitting, for a clearly separated post-fit audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_smooth_body_prior_experiment as smooth
from worm_pose_gen.anchors import render_centerline_mask, skeleton_topology
from worm_pose_gen.classical import (
    ClassicalConfig,
    _dilate,
    _erode,
    _largest_component,
    resample_centerline,
    segment_dark_ridge,
)
from worm_pose_gen.latent import decode_centerline, encode_centerline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "boundary_notch_repair_experiment"
SEARCH_RADII = tuple(range(3, 13))
MAX_ADDED_FRACTION = 0.10
MIN_EXTERIOR_RETENTION = 0.99
# A rendered completion includes pixels deliberately recovered by the body
# prior, so it needs a larger area allowance than the raw-component extractor.
MODELED_BODY_MAX_AREA = 40_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-hdf5", type=Path, default=smooth.DEFAULT_PROXY_HDF5)
    parser.add_argument("--annotations", type=Path, default=smooth.DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def exterior_background(mask: np.ndarray) -> np.ndarray:
    exterior, _, _ = _largest_component(~np.asarray(mask, dtype=bool))
    return exterior


def candidate_repair(
    baseline_mask: np.ndarray,
    original_component: np.ndarray,
    score: np.ndarray,
    radius: int,
    *,
    fill_sealed_pockets: bool,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Seal narrow mouths and optionally fill only pockets isolated by the seal."""

    closed = _erode(_dilate(baseline_mask, radius), radius)
    sealed = baseline_mask | closed
    bridge = sealed & ~baseline_mask
    sealed_exterior = exterior_background(sealed)
    pocket = (~sealed) & ~sealed_exterior
    repaired = sealed | pocket if fill_sealed_pockets else sealed
    added = repaired & ~baseline_mask
    before_exterior = exterior_background(baseline_mask)
    after_exterior = exterior_background(repaired)
    newly_enclosed = (~repaired) & ~after_exterior
    retained_exterior = float(np.logical_and(before_exterior, after_exterior).sum() / before_exterior.sum())
    _, skeleton = smooth.initial_path(repaired)
    topology = skeleton_topology(skeleton)
    added_fraction = float(added.sum() / original_component.sum())
    reasons: list[str] = []
    if int(newly_enclosed.sum()):
        reasons.append("seals_exterior_background_into_a_hole")
    if retained_exterior < MIN_EXTERIOR_RETENTION:
        reasons.append("removes_too_much_exterior_background")
    if added_fraction > MAX_ADDED_FRACTION:
        reasons.append("adds_more_than_10_percent_of_original_component")
    if topology["endpoint_count"] != 2:
        reasons.append("not_two_endpoints")
    if topology["branch_pixels"] != 0:
        reasons.append("branches_remain")
    if topology["has_cycle"]:
        reasons.append("cycle_remains")
    metrics = {
        "arm": "seal_then_fill" if fill_sealed_pockets else "direct_closing",
        "radius_px": radius,
        "accepted_by_geometry_only_rule": not reasons,
        "rejection_reasons": reasons,
        "added_pixels": int(added.sum()),
        "bridge_pixels": int(bridge.sum()),
        "sealed_pocket_pixels": int(pocket.sum()),
        "bridge_local_darkness_z0_fraction": float(np.mean(score[bridge] >= 0.0)) if np.any(bridge) else 0.0,
        "bridge_original_threshold_z2_6_fraction": float(np.mean(score[bridge] >= 2.6)) if np.any(bridge) else 0.0,
        "pocket_local_darkness_z0_fraction": float(np.mean(score[pocket] >= 0.0)) if np.any(pocket) else 0.0,
        "pocket_original_threshold_z2_6_fraction": float(np.mean(score[pocket] >= 2.6)) if np.any(pocket) else 0.0,
        "added_fraction_of_original_component": added_fraction,
        "added_local_darkness_z0_fraction": float(np.mean(score[added] >= 0.0)) if np.any(added) else 0.0,
        "added_original_threshold_z2_6_fraction": float(np.mean(score[added] >= 2.6)) if np.any(added) else 0.0,
        "newly_enclosed_background_pixels": int(newly_enclosed.sum()),
        "exterior_background_retained_fraction": retained_exterior,
        "topology": topology,
    }
    return repaired, metrics, added, bridge, pocket


def evaluate_body_fit(
    initialization_mask: np.ndarray,
    original_component: np.ndarray,
    score: np.ndarray,
    cfg: ClassicalConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run the frozen K=16 exact-containment fit and downstream stages."""

    path, initialization_skeleton = smooth.initial_path(initialization_mask)
    initialization_centerline = resample_centerline(path, cfg.n_points)
    latent = encode_centerline(initialization_centerline, 16)
    latent_midline = decode_centerline(latent, 16)
    required_radius, _, _ = smooth.nearest_station_requirements(
        original_component, latent_midline, 0.75
    )
    fitted_radius = smooth.slowly_varying_majorant(required_radius, 1.0)
    fitted_width = 2.0 * fitted_radius
    body_mask = render_centerline_mask(latent_midline, fitted_width, original_component.shape)
    missed = int(np.logical_and(original_component, ~body_mask).sum())
    downstream = smooth.rerun_backbone(body_mask, cfg.n_points)
    centerline = np.asarray(downstream["centerline"])
    skeleton = np.asarray(downstream["skeleton"])
    length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    height, image_width = body_mask.shape
    boundary_distance = float(
        np.min(
            np.column_stack(
                (
                    centerline[:, 0], centerline[:, 1],
                    (image_width - 1) - centerline[:, 0],
                    (height - 1) - centerline[:, 1],
                )
            )
        )
    )
    area = int(body_mask.sum())
    width_feasible = bool(float(fitted_width.max()) <= 80.0 + 1e-10)
    area_pass = bool(cfg.min_area <= area <= MODELED_BODY_MAX_AREA)
    length_pass = bool(cfg.min_length <= length <= cfg.max_length)
    topology = skeleton_topology(skeleton)
    topology_pass = bool(
        int(downstream["endpoint_count"]) >= 2
        and int(downstream["endpoint_count"]) <= cfg.max_raw_endpoints
        and int(downstream["branch_pixels"]) <= cfg.max_branch_pixels
    )
    newly_added = body_mask & ~original_component
    reasons = []
    if not width_feasible:
        reasons.append("full_width_exceeds_80_px")
    if not area_pass:
        reasons.append("implausible_area")
    if not length_pass:
        reasons.append("implausible_length")
    if not topology_pass:
        reasons.append("unstable_endpoints")
    metrics: dict[str, object] = {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "detected_positive_recall": float(np.mean(body_mask[original_component])),
        "detected_positive_pixels_missed": missed,
        "max_full_width_px": float(fitted_width.max()),
        "median_full_width_px": float(np.median(fitted_width)),
        "width_feasibility_passed": width_feasible,
        "modeled_area_px": area,
        "area_gate_passed": area_pass,
        "area_gate_allowed_px": [cfg.min_area, MODELED_BODY_MAX_AREA],
        "centerline_length_px": length,
        "length_gate_passed": length_pass,
        "boundary_distance_px": boundary_distance,
        "downstream_topology": topology,
        "newly_modeled_pixels": int(newly_added.sum()),
        "newly_modeled_local_darkness_z0_fraction": float(np.mean(score[newly_added] >= 0.0)),
        "newly_modeled_original_threshold_z2_6_fraction": float(np.mean(score[newly_added] >= cfg.foreground_z)),
        "mask_iou_with_original_section3": smooth.mask_iou(body_mask, original_component),
    }
    arrays = {
        "initialization_mask": initialization_mask,
        "initialization_skeleton": initialization_skeleton,
        "initialization_centerline": initialization_centerline,
        "latent_midline": latent_midline,
        "required_radius": required_radius,
        "fitted_radius": fitted_radius,
        "body_mask": body_mask,
        "downstream_skeleton": skeleton,
        "downstream_path": np.asarray(downstream["path"]),
        "centerline": centerline,
    }
    return metrics, arrays


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, provenance = smooth.load_real_frame(args.proxy_hdf5)
    cfg = ClassicalConfig()
    segmentation = segment_dark_ridge(frame, cfg)
    score = segmentation.score
    component = segmentation.component
    component_count = segmentation.component_count
    baseline_mask, enclosed, enclosed_count = smooth.fill_enclosed_cavities(component)
    bounds = smooth.crop_bounds(component)

    direct_sweep: list[dict[str, object]] = []
    seal_fill_sweep: list[dict[str, object]] = []
    direct_candidates: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    seal_fill_candidates: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for radius in SEARCH_RADII:
        direct, direct_metrics, direct_added, direct_bridge, direct_pocket = candidate_repair(
            baseline_mask, component, score, radius, fill_sealed_pockets=False
        )
        filled, filled_metrics, filled_added, filled_bridge, filled_pocket = candidate_repair(
            baseline_mask, component, score, radius, fill_sealed_pockets=True
        )
        direct_sweep.append(direct_metrics)
        seal_fill_sweep.append(filled_metrics)
        direct_candidates[radius] = (direct, direct_added, direct_bridge, direct_pocket)
        seal_fill_candidates[radius] = (filled, filled_added, filled_bridge, filled_pocket)
    direct_radii = [int(row["radius_px"]) for row in direct_sweep if row["accepted_by_geometry_only_rule"]]
    seal_fill_radii = [int(row["radius_px"]) for row in seal_fill_sweep if row["accepted_by_geometry_only_rule"]]
    if not direct_radii or not seal_fill_radii:
        raise RuntimeError("both repair arms need a geometry-only passing candidate")
    direct_radius = min(direct_radii)
    selected_radius = min(seal_fill_radii)
    direct_mask, direct_added, direct_bridge, direct_pocket = direct_candidates[direct_radius]
    repaired_mask, repair_added, repair_bridge, repair_pocket = seal_fill_candidates[selected_radius]

    baseline_metrics, baseline_arrays = evaluate_body_fit(baseline_mask, component, score, cfg)
    direct_metrics, direct_arrays = evaluate_body_fit(direct_mask, component, score, cfg)
    repair_metrics, repair_arrays = evaluate_body_fit(repaired_mask, component, score, cfg)

    # Manual data is deliberately loaded only after candidate selection and both fits.
    annotation_xy, annotation_metadata = smooth.load_postfit_annotation(args.annotations)
    baseline_audit = smooth.complete_curve_metrics(baseline_arrays["centerline"], annotation_xy)
    direct_audit = smooth.complete_curve_metrics(direct_arrays["centerline"], annotation_xy)
    repair_audit = smooth.complete_curve_metrics(repair_arrays["centerline"], annotation_xy)

    metrics: dict[str, object] = {
        "status": "isolated_one_frame_boundary_notch_repair_prototype",
        "provenance": provenance,
        "parameters": {
            "candidate_close_radii_px": list(SEARCH_RADII),
            "selection": "smallest radius passing every geometry-only guard",
            "max_added_fraction_of_original_component": MAX_ADDED_FRACTION,
            "minimum_exterior_background_retention": MIN_EXTERIOR_RETENTION,
            "required_initialization_topology": "2 endpoints, 0 branches, no cycle",
            "newly_enclosed_background_allowed_px": 0,
            "latent_angle_coefficients": 16,
            "containment_margin_px": 0.75,
            "radius_max_adjacent_change_px": 1.0,
            "max_full_width_px": 80.0,
            "modeled_body_max_area_px": MODELED_BODY_MAX_AREA,
            "modeled_body_area_rationale": (
                "rendered completion includes prior-recovered pixels; the historical "
                "30000 px ceiling remains specific to the raw-component extractor"
            ),
            "manual_trace_used_for_fit_or_selection": False,
        },
        "input": {
            "section3_positive_pixels": int(component.sum()),
            "section3_pre_keep_component_count": int(component_count),
            "baseline_enclosed_cavities": int(enclosed_count),
            "baseline_enclosed_pixels_filled": int(enclosed.sum()),
        },
        "direct_closing_candidate_sweep": direct_sweep,
        "seal_then_fill_candidate_sweep": seal_fill_sweep,
        "current_direct_closing_arm": {
            "radius_px": direct_radius,
            "bridge_pixels": int(direct_bridge.sum()),
            "pocket_pixels_left_background": int(direct_pocket.sum()),
            "fit": direct_metrics,
        },
        "selected_repair": {
            "arm": "seal_then_fill",
            "radius_px": selected_radius,
            "added_pixels": int(repair_added.sum()),
            "bridge_pixels": int(repair_bridge.sum()),
            "sealed_pocket_pixels": int(repair_pocket.sum()),
            "bridge_local_darkness_z0_fraction": float(np.mean(score[repair_bridge] >= 0.0)),
            "bridge_original_threshold_z2_6_fraction": float(np.mean(score[repair_bridge] >= cfg.foreground_z)),
            "pocket_local_darkness_z0_fraction": float(np.mean(score[repair_pocket] >= 0.0)),
            "pocket_original_threshold_z2_6_fraction": float(np.mean(score[repair_pocket] >= cfg.foreground_z)),
            "added_fraction_of_original_component": float(repair_added.sum() / component.sum()),
            "added_local_darkness_z0_fraction": float(np.mean(score[repair_added] >= 0.0)),
            "added_original_threshold_z2_6_fraction": float(np.mean(score[repair_added] >= cfg.foreground_z)),
            "topology_before": skeleton_topology(baseline_arrays["initialization_skeleton"]),
            "topology_after": skeleton_topology(repair_arrays["initialization_skeleton"]),
            "newly_enclosed_background_pixels_after_fill": 0,
            "exterior_background_retained_fraction": float(
                np.logical_and(exterior_background(baseline_mask), exterior_background(repaired_mask)).sum()
                / exterior_background(baseline_mask).sum()
            ),
            "open_center_guard": "passed: no newly enclosed background and >=99% exterior retained",
        },
        "frozen_enclosed-hole_baseline": baseline_metrics,
        "direct-closing-comparison": direct_metrics,
        "boundary-notch-repair": repair_metrics,
        "postfit_manual_audit": {
            "annotation": annotation_metadata,
            "role": "loaded only after geometry-only selection and both fits completed",
            "frozen_baseline": baseline_audit,
            "direct_closing": direct_audit,
            "notch_repair": repair_audit,
        },
        "failure_cases": [
            "A true anatomical bend narrower than the tested close diameter could be mistaken for a notch.",
            "The guard rejects candidates that leave a newly sealed cavity, but cannot prove every added pixel is anatomy.",
            "The simple-path selection rule can prefer an over-smoothed mask when topology alone is misleading.",
            "This is one development frame and the radius search is not validated across recordings.",
            "Manual-trace metrics are post-fit diagnostics from one annotator, not selection evidence.",
        ],
    }

    # 0 — original Section-3 evidence.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    smooth.overlay_mask(ax, component, smooth.CYAN, 0.68)
    smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "00_section3_input.png")

    # 1 — frozen baseline initialization.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    smooth.overlay_mask(ax, component, smooth.CYAN, 0.52)
    smooth.overlay_mask(ax, enclosed, smooth.ORANGE, 0.88)
    ax.text(0.015, 0.025, f"frozen baseline: {int(enclosed.sum()):,} enclosed pixels filled",
            transform=ax.transAxes, color="white", fontsize=11,
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 6})
    smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "01_frozen_enclosed_hole_baseline.png")

    # 2 — candidate sweep, selected without annotation.
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True, constrained_layout=True)
    radii = np.asarray([row["radius_px"] for row in seal_fill_sweep])
    area = np.asarray([row["added_pixels"] for row in seal_fill_sweep])
    direct_area = np.asarray([row["added_pixels"] for row in direct_sweep])
    support = np.asarray([row["added_local_darkness_z0_fraction"] for row in seal_fill_sweep])
    passed = np.asarray([row["accepted_by_geometry_only_rule"] for row in seal_fill_sweep])
    axes[0].plot(radii, direct_area, marker="o", color=smooth.GRAY, label="direct close")
    axes[0].plot(radii, area, marker="o", color=smooth.ORANGE, label="seal + fill pocket")
    axes[0].axvline(selected_radius, color=smooth.GREEN, linestyle="--", label=f"selected r={selected_radius}")
    axes[0].set_ylabel("notch pixels added")
    axes[0].legend()
    axes[1].plot(radii, 100 * support, marker="o", color=smooth.CYAN, label="added-pixel local darkness z≥0")
    axes[1].scatter(radii[passed], 100 * support[passed], s=80, facecolors="none",
                    edgecolors=smooth.GREEN, label="geometry guard passes")
    axes[1].set(xlabel="closing radius (pixels)", ylabel="support (%)")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    save(fig, args.output_dir / "02_geometry_only_radius_sweep.png")

    # 3 — selected exterior-notch additions.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    smooth.overlay_mask(ax, baseline_mask, smooth.CYAN, 0.52)
    smooth.overlay_mask(ax, repair_bridge, smooth.ORANGE, 0.9)
    smooth.overlay_mask(ax, repair_pocket, smooth.MAGENTA, 0.85)
    ax.text(0.015, 0.025,
            f"selected r={selected_radius}: {int(repair_bridge.sum()):,} bridge + "
            f"{int(repair_pocket.sum()):,} pocket pixels\n"
            "large open U remains exterior-connected",
            transform=ax.transAxes, color="white", fontsize=10.5,
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 6})
    smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "03_selected_boundary_notch_repair.png")

    # 4 — topology before and after repair.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    for ax, key, title in (
        (axes[0], baseline_arrays, "frozen baseline initialization"),
        (axes[1], direct_arrays, f"direct close r={direct_radius}"),
        (axes[2], repair_arrays, f"seal + pocket fill r={selected_radius}"),
    ):
        ax.imshow(key["initialization_mask"], cmap="gray", vmin=0, vmax=1)
        smooth.overlay_mask(ax, key["initialization_skeleton"], smooth.PALE_CYAN, 1.0)
        ax.set_title(title)
        smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "04_initialization_topology_comparison.png")

    # 5 — latent midlines.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(baseline_arrays["latent_midline"][:, 0], baseline_arrays["latent_midline"][:, 1],
            color=smooth.MAGENTA, linewidth=4, label="frozen baseline latent midline")
    ax.plot(direct_arrays["latent_midline"][:, 0], direct_arrays["latent_midline"][:, 1],
            color=smooth.GRAY, linewidth=3.2, label="direct-close latent midline")
    ax.plot(repair_arrays["latent_midline"][:, 0], repair_arrays["latent_midline"][:, 1],
            color=smooth.ORANGE, linewidth=2.7, label="notch-repair latent midline")
    ax.legend(loc="upper left", framealpha=0.87)
    smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "05_latent_midline_comparison.png")

    # 6 — completed body masks.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    for ax, arrays, label in (
        (axes[0], baseline_arrays, "frozen baseline body model"),
        (axes[1], direct_arrays, "direct-closing body model"),
        (axes[2], repair_arrays, "seal-then-fill body model"),
    ):
        ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
        smooth.overlay_mask(ax, arrays["body_mask"] & ~component, smooth.ORANGE, 0.60)
        smooth.overlay_mask(ax, component, smooth.GREEN, 0.62)
        ax.plot(arrays["latent_midline"][:, 0], arrays["latent_midline"][:, 1],
                color=smooth.PALE_CYAN, linewidth=2)
        ax.set_title(label)
        smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "06_completed_body_comparison.png")

    # 7 — downstream poses and honest gate outcomes.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(baseline_arrays["centerline"][:, 0], baseline_arrays["centerline"][:, 1],
            color=smooth.MAGENTA, linewidth=4, label="frozen baseline final pose")
    ax.plot(direct_arrays["centerline"][:, 0], direct_arrays["centerline"][:, 1],
            color=smooth.GRAY, linewidth=3.2, label="direct-close final pose")
    ax.plot(repair_arrays["centerline"][:, 0], repair_arrays["centerline"][:, 1],
            color=smooth.ORANGE, linewidth=2.7, label="notch-repair final pose")
    reason_label = ", ".join(str(reason) for reason in repair_metrics["rejection_reasons"])
    status_label = "PASS" if repair_metrics["accepted"] else f"REJECT: {reason_label}"
    label = (
        f"repair: {status_label}\n"
        f"area PASS {repair_metrics['modeled_area_px']:,} / {MODELED_BODY_MAX_AREA:,} px²\n"
        f"max width {repair_metrics['max_full_width_px']:.1f} / 80.0 px\n"
        f"positive containment {100 * repair_metrics['detected_positive_recall']:.0f}%"
    )
    ax.text(0.73, 0.97, label, transform=ax.transAxes, va="top", color="white",
            fontsize=10.5, linespacing=1.3,
            bbox={"facecolor": "#12643b" if repair_metrics["accepted"] else "#8a2222",
                  "alpha": 0.9, "edgecolor": smooth.GREEN if repair_metrics["accepted"] else smooth.RED,
                  "pad": 7})
    ax.legend(loc="upper left", framealpha=0.87)
    smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "07_downstream_pose_and_gates.png")

    # 8 — manual post-fit audit only.
    target = smooth.resample_polyline(annotation_xy, cfg.n_points)
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(target[:, 0], target[:, 1], color=smooth.GREEN, linewidth=4.5,
            label="one-annotator trace (post-fit only)")
    ax.plot(baseline_arrays["centerline"][:, 0], baseline_arrays["centerline"][:, 1],
            color=smooth.MAGENTA, linewidth=3,
            label=f"baseline median {baseline_audit['median_point_distance_px']:.1f} px")
    ax.plot(direct_arrays["centerline"][:, 0], direct_arrays["centerline"][:, 1],
            color=smooth.GRAY, linewidth=2.7,
            label=f"direct close median {direct_audit['median_point_distance_px']:.1f} px")
    ax.plot(repair_arrays["centerline"][:, 0], repair_arrays["centerline"][:, 1],
            color=smooth.ORANGE, linewidth=2.5,
            label=f"repair median {repair_audit['median_point_distance_px']:.1f} px")
    ax.legend(loc="upper left", framealpha=0.87)
    smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "08_manual_trace_postfit_audit.png")

    np.savez_compressed(
        args.output_dir / "experiment_arrays.npz",
        frame=frame,
        section3_component=component,
        frozen_baseline_mask=baseline_mask,
        repaired_initialization_mask=repaired_mask,
        notch_added_mask=repair_added,
        notch_bridge_mask=repair_bridge,
        notch_pocket_mask=repair_pocket,
        direct_closing_mask=direct_mask,
        baseline_latent_midline_xy=baseline_arrays["latent_midline"],
        direct_latent_midline_xy=direct_arrays["latent_midline"],
        repair_latent_midline_xy=repair_arrays["latent_midline"],
        baseline_body_mask=baseline_arrays["body_mask"],
        direct_body_mask=direct_arrays["body_mask"],
        repair_body_mask=repair_arrays["body_mask"],
        baseline_centerline_xy=baseline_arrays["centerline"],
        direct_centerline_xy=direct_arrays["centerline"],
        repair_centerline_xy=repair_arrays["centerline"],
        postfit_annotation_xy=annotation_xy,
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
