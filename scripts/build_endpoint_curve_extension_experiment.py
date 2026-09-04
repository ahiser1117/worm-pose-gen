#!/usr/bin/env python3
"""Extend the frozen Experiment-A midline through both modeled end caps.

The extension uses only the A5 completed-body mask and final ordered midline.
Local terminal curvature is fitted before the manual trace is loaded.  The
trace remains a post-fit audit and cannot affect the extension geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_smooth_body_prior_experiment as smooth
from worm_pose_gen.anchors import (
    _inside,
    _terminal_curve,
    extend_centerline_to_mask_boundary,
)
from worm_pose_gen.classical import ClassicalConfig, resample_centerline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A5_ARRAYS = (
    PROJECT_ROOT / "docs" / "boundary_notch_repair_experiment" / "experiment_arrays.npz"
)
DEFAULT_A5_METRICS = (
    PROJECT_ROOT / "docs" / "boundary_notch_repair_experiment" / "metrics.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "endpoint_curve_extension_experiment"

CONTEXT_POINTS = 7
STEP_PX = 0.25
MAX_EXTENSION_PX = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5-arrays", type=Path, default=DEFAULT_A5_ARRAYS)
    parser.add_argument("--a5-metrics", type=Path, default=DEFAULT_A5_METRICS)
    parser.add_argument("--annotations", type=Path, default=smooth.DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-points", type=int, default=CONTEXT_POINTS)
    parser.add_argument("--step-px", type=float, default=STEP_PX)
    parser.add_argument("--max-extension-px", type=float, default=MAX_EXTENSION_PX)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curve_length(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def locate_original_curve(extended_xy: np.ndarray, original_xy: np.ndarray) -> int:
    """Locate the exactly retained original samples inside the dense result."""

    stop = len(extended_xy) - len(original_xy) + 1
    for start in range(stop):
        if np.array_equal(extended_xy[start : start + len(original_xy)], original_xy):
            return start
    raise RuntimeError("endpoint continuation did not retain the original centerline exactly")


def terminal_context_length(points_xy: np.ndarray, context_points: int) -> float:
    local = np.asarray(points_xy, dtype=np.float64)[-min(context_points, len(points_xy)) :]
    return curve_length(local)


def endpoint_metrics(
    oriented_original_xy: np.ndarray,
    outward_extension_xy: np.ndarray,
    context_points: int,
) -> dict[str, object]:
    angle, curvature = _terminal_curve(oriented_original_xy, context_points)
    extension_length = curve_length(outward_extension_xy)
    return {
        "context_points": min(context_points, len(oriented_original_xy)),
        "context_arc_length_px": terminal_context_length(oriented_original_xy, context_points),
        "fitted_outward_tangent_rad": angle,
        "fitted_outward_tangent_deg": float(np.degrees(angle)),
        "fitted_signed_curvature_rad_per_px": curvature,
        "fitted_radius_px": float(1.0 / abs(curvature)) if abs(curvature) > 1e-12 else None,
        "extension_arc_length_px": extension_length,
        "extension_chord_length_px": float(
            np.linalg.norm(outward_extension_xy[-1] - outward_extension_xy[0])
        ),
        "boundary_xy": outward_extension_xy[-1].tolist(),
        "first_boundary_hit": True,
    }


def validate_frozen_input(
    frame: np.ndarray,
    component: np.ndarray,
    body_mask: np.ndarray,
    centerline_xy: np.ndarray,
) -> None:
    if frame.ndim != 2 or component.shape != frame.shape or body_mask.shape != frame.shape:
        raise ValueError("A5 frame and masks must be aligned two-dimensional arrays")
    if centerline_xy.ndim != 2 or centerline_xy.shape != (100, 2):
        raise ValueError("A5 centerline must retain the frozen [100,2] representation")
    if np.any(component & ~body_mask):
        raise ValueError("A5 completed body no longer contains every Section-3 positive")
    if not _inside(body_mask, *centerline_xy[0]) or not _inside(body_mask, *centerline_xy[-1]):
        raise ValueError("A5 centerline endpoints must start inside its completed body")


def set_endpoint_crop(ax: plt.Axes, points_xy: np.ndarray, margin: float = 42.0) -> None:
    x0, y0 = np.min(points_xy, axis=0)
    x1, y1 = np.max(points_xy, axis=0)
    ax.set_xlim(x0 - margin, x1 + margin)
    ax.set_ylim(y1 + margin, y0 - margin)
    ax.set_axis_off()


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.context_points < 3:
        raise ValueError("context-points must be at least 3")
    if not np.isfinite(args.step_px) or args.step_px <= 0:
        raise ValueError("step-px must be finite and positive")
    if not np.isfinite(args.max_extension_px) or args.max_extension_px <= 0:
        raise ValueError("max-extension-px must be finite and positive")

    # Freeze the A5 geometry before the annotation is opened.  In particular,
    # do not read the postfit_annotation_xy field stored in the source NPZ.
    with np.load(args.a5_arrays) as source:
        frame = np.asarray(source["frame"])
        component = np.asarray(source["section3_component"], dtype=bool)
        body_mask = np.asarray(source["repair_body_mask"], dtype=bool)
        a5_centerline = np.asarray(source["repair_centerline_xy"], dtype=np.float64)
    validate_frozen_input(frame, component, body_mask, a5_centerline)
    frozen_body_mask = body_mask.copy()

    dense_extended = extend_centerline_to_mask_boundary(
        a5_centerline,
        body_mask,
        context_points=args.context_points,
        step=args.step_px,
        max_extension=args.max_extension_px,
    )
    original_start = locate_original_curve(dense_extended, a5_centerline)
    original_stop = original_start + len(a5_centerline)
    start_extension_boundary_to_join = dense_extended[: original_start + 1]
    end_extension_join_to_boundary = dense_extended[original_stop - 1 :]
    a6_centerline = resample_centerline(dense_extended, len(a5_centerline))

    if not all(_inside(body_mask, *point) for point in dense_extended):
        raise RuntimeError("a continuation sample left the body before its recorded first exit")
    if not np.array_equal(body_mask, frozen_body_mask):
        raise RuntimeError("completed-body mask changed during endpoint continuation")

    cfg = ClassicalConfig()
    a5_length = curve_length(a5_centerline)
    dense_length = curve_length(dense_extended)
    a6_length = curve_length(a6_centerline)
    length_pass = bool(cfg.min_length <= a6_length <= cfg.max_length)
    if not length_pass:
        raise RuntimeError("extended pose fails the frozen classical length gate")

    start_metrics = endpoint_metrics(
        a5_centerline[::-1], start_extension_boundary_to_join[::-1], args.context_points
    )
    end_metrics = endpoint_metrics(
        a5_centerline, end_extension_join_to_boundary, args.context_points
    )

    # Only after the continuation and all geometry gates are fixed do we read
    # the prior metrics (which contain their own post-fit audit) or open the
    # single-annotator trace for this disclosed post-fit comparison.
    a5_metrics = json.loads(args.a5_metrics.read_text(encoding="utf-8"))
    annotation_xy, annotation_metadata = smooth.load_postfit_annotation(args.annotations)
    a5_audit = smooth.complete_curve_metrics(a5_centerline, annotation_xy)
    a6_audit = smooth.complete_curve_metrics(a6_centerline, annotation_xy)

    source_body_metrics = a5_metrics["boundary-notch-repair"]
    metrics: dict[str, object] = {
        "status": "isolated_one_frame_curvature_aware_endpoint_extension",
        "provenance": {
            "a5_arrays": str(args.a5_arrays),
            "a5_arrays_sha256": sha256(args.a5_arrays),
            "a5_metrics": str(args.a5_metrics),
            "a5_metrics_sha256": sha256(args.a5_metrics),
            "source_experiment_status": a5_metrics["status"],
            "recording": a5_metrics["provenance"],
        },
        "parameters": {
            "boundary_target": "A5 repair_body_mask: orange model-added pixels union green Section-3 positives",
            "source_pose": "A5 repair_centerline_xy after thinning, endpoint peeling, longest path, and 100-point resampling",
            "terminal_model": "least-squares unwrapped tangent angle versus local arc length; constant signed-curvature continuation",
            "context_points": args.context_points,
            "integration_step_px": args.step_px,
            "maximum_extension_per_end_px": args.max_extension_px,
            "boundary_rule": "first foreground-to-background exit under nearest-pixel support, refined by curved-step bisection",
            "output_points": len(a5_centerline),
            "manual_trace_used_for_extension_or_parameter_selection": False,
        },
        "frozen_a5": {
            "body_mask_area_px": int(body_mask.sum()),
            "section3_positive_pixels": int(component.sum()),
            "centerline_length_px": a5_length,
            "centerline_endpoints_xy": a5_centerline[[0, -1]].tolist(),
            "body_fit": source_body_metrics,
        },
        "extension_geometry": {
            "start": start_metrics,
            "end": end_metrics,
            "both_first_boundaries_hit": True,
            "total_dense_extension_length_px": float(dense_length - a5_length),
            "dense_curve_length_px": dense_length,
            "dense_samples": int(len(dense_extended)),
            "original_samples_retained_exactly": True,
            "body_mask_changed": False,
        },
        "pose_after_extension": {
            "centerline_points": int(len(a6_centerline)),
            "centerline_length_px": a6_length,
            "length_gain_px": float(a6_length - a5_length),
            "length_gate_allowed_px": [cfg.min_length, cfg.max_length],
            "length_gate_passed": length_pass,
            "centerline_endpoints_xy": a6_centerline[[0, -1]].tolist(),
            "body_area_width_and_containment_gates": "unchanged because repair_body_mask is not modified",
            "modeled_area_px": int(source_body_metrics["modeled_area_px"]),
            "max_full_width_px": float(source_body_metrics["max_full_width_px"]),
            "detected_positive_recall": float(source_body_metrics["detected_positive_recall"]),
        },
        "postfit_manual_audit": {
            "annotation": annotation_metadata,
            "role": "loaded only after extension and geometry gates completed",
            "a5_unextended": a5_audit,
            "a6_curved_extension": a6_audit,
        },
        "failure_cases": [
            "The completed end caps come from the smooth-body model, so reaching them is geometric self-consistency rather than new image evidence.",
            "A constant-curvature continuation can miss a rapid curvature change beyond the observed terminal context.",
            "The first-exit rule prevents jumping across background to another arm, but a wrong completed-body cap still produces a wrong endpoint.",
            "Endpoint retreat is not the only length bias: fixed-count resampling can shorten a highly curved pixel path by replacing arcs with chords.",
            "This is one disclosed development frame; the context and step must be frozen before multi-frame evaluation.",
        ],
    }

    bounds = smooth.crop_bounds(body_mask)

    # 0 — reproduce the A5 rightmost body and expose its endpoint gaps.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    smooth.overlay_mask(ax, body_mask & ~component, smooth.ORANGE, 0.60)
    smooth.overlay_mask(ax, component, smooth.GREEN, 0.62)
    ax.plot(a5_centerline[:, 0], a5_centerline[:, 1], color=smooth.PALE_CYAN,
            linewidth=2.5, label="frozen A5 midline")
    ax.scatter(a5_centerline[[0, -1], 0], a5_centerline[[0, -1], 1], s=65,
               color=smooth.MAGENTA, edgecolor="white", linewidth=0.8,
               label="skeletonized endpoints")
    ax.legend(loc="upper left", framealpha=0.87)
    smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "00_frozen_a5_endpoint_gap.png")

    # 1 — show the fitted context and curved continuation at each cap.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    endpoint_panels = (
        (
            axes[0],
            a5_centerline[: args.context_points],
            start_extension_boundary_to_join,
            "index-0 cap",
            start_metrics,
        ),
        (
            axes[1],
            a5_centerline[-args.context_points :],
            end_extension_join_to_boundary,
            "index-99 cap",
            end_metrics,
        ),
    )
    for ax, context, extension, title, values in endpoint_panels:
        ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
        smooth.overlay_mask(ax, body_mask, smooth.GREEN, 0.42)
        ax.plot(a5_centerline[:, 0], a5_centerline[:, 1], color=smooth.GRAY,
                linewidth=2, label="A5 midline")
        ax.plot(context[:, 0], context[:, 1], color=smooth.MAGENTA,
                linewidth=4, label="curve-fit context")
        ax.plot(extension[:, 0], extension[:, 1], color=smooth.ORANGE,
                linewidth=3, label="curved continuation")
        ax.scatter(extension[-1 if title == "index-99 cap" else 0, 0],
                   extension[-1 if title == "index-99 cap" else 0, 1],
                   s=75, color=smooth.PALE_CYAN, edgecolor="black", linewidth=0.7,
                   label="first boundary")
        ax.set_title(
            f"{title}: {values['extension_arc_length_px']:.2f} px added, "
            f"curvature {values['fitted_signed_curvature_rad_per_px']:+.4f} rad/px"
        )
        set_endpoint_crop(ax, np.vstack((context, extension)))
    axes[0].legend(loc="upper left", framealpha=0.87)
    save(fig, args.output_dir / "01_curve_context_and_extension.png")

    # 2 — the only pose update: splice both continuations and resample to 100.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    smooth.overlay_mask(ax, body_mask, smooth.GREEN, 0.38)
    ax.plot(a5_centerline[:, 0], a5_centerline[:, 1], color=smooth.MAGENTA,
            linewidth=4, label=f"A5: {a5_length:.1f} px")
    ax.plot(a6_centerline[:, 0], a6_centerline[:, 1], color=smooth.PALE_CYAN,
            linewidth=2.6, label=f"A6 extended: {a6_length:.1f} px")
    ax.scatter(a6_centerline[[0, -1], 0], a6_centerline[[0, -1], 1],
               s=62, color=smooth.ORANGE, edgecolor="white", linewidth=0.8)
    ax.text(
        0.73,
        0.97,
        f"both boundaries HIT\nlength +{a6_length - a5_length:.1f} px\n"
        f"length gate PASS: {a6_length:.1f} / {cfg.max_length:.0f} px",
        transform=ax.transAxes,
        va="top",
        color="white",
        fontsize=10.5,
        linespacing=1.3,
        bbox={"facecolor": "#12643b", "alpha": 0.9, "edgecolor": smooth.GREEN, "pad": 7},
    )
    ax.legend(loc="upper left", framealpha=0.87)
    smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "02_extended_pose_comparison.png")

    # 3 — annotation remains a post-fit audit only.
    target = smooth.resample_polyline(annotation_xy, cfg.n_points)
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(target[:, 0], target[:, 1], color=smooth.GREEN, linewidth=4.5,
            label="one-annotator trace (post-fit only)")
    ax.plot(a5_centerline[:, 0], a5_centerline[:, 1], color=smooth.MAGENTA,
            linewidth=3, label=f"A5 median {a5_audit['median_point_distance_px']:.2f} px")
    ax.plot(a6_centerline[:, 0], a6_centerline[:, 1], color=smooth.ORANGE,
            linewidth=2.5, label=f"A6 median {a6_audit['median_point_distance_px']:.2f} px")
    ax.legend(loc="upper left", framealpha=0.87)
    smooth.set_crop(ax, bounds)
    save(fig, args.output_dir / "03_manual_trace_postfit_audit.png")

    np.savez_compressed(
        args.output_dir / "experiment_arrays.npz",
        frame=frame,
        section3_component=component,
        a5_body_mask=body_mask,
        a5_centerline_xy=a5_centerline,
        start_context_xy=a5_centerline[: args.context_points],
        end_context_xy=a5_centerline[-args.context_points :],
        start_extension_boundary_to_join_xy=start_extension_boundary_to_join,
        end_extension_join_to_boundary_xy=end_extension_join_to_boundary,
        dense_extended_centerline_xy=dense_extended,
        a6_centerline_xy=a6_centerline,
        postfit_annotation_xy=annotation_xy,
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
