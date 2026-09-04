#!/usr/bin/env python3
"""Build the worked visuals for docs/POSE_ESTIMATION_EXPLAINER.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import numpy as np

from worm_pose_gen.classical import (
    ClassicalConfig,
    _prune_skeleton_endpoints,
    _skeleton_longest_path,
    _thin,
    extract_centerline,
    resample_centerline,
    segment_dark_ridge,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROXY_HDF5 = Path(
    "/temp_data4/alex/external_artifacts/datasets/"
    "worm_pose_gen/proxy_v1/proxy_labels.h5"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "pose_estimation_explainer"
RECORDING = "2023-09-19-01"
FRAME_INDEX = 3420
SOURCE_DATASET = "/img_nir"

ORANGE = "#ff9f1c"
CYAN = "#2ec4b6"
PALE_CYAN = "#7ee8fa"
MAGENTA = "#ff4fa3"
GREEN = "#57d68d"
RED = "#ff5d5d"
GRAY = "#a7adb4"
AMBER = "#ffb142"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-hdf5", type=Path, default=DEFAULT_PROXY_HDF5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_real_frame(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Read the cached uint8 frame and its original-video provenance."""

    with h5py.File(path, "r") as handle:
        if handle.attrs["source_dataset_path"] != SOURCE_DATASET:
            raise RuntimeError("unexpected source dataset path")
        group = handle[RECORDING]
        frame_indices = np.asarray(group["accepted_frame_index"], dtype=np.int64)
        positions = np.flatnonzero(frame_indices == FRAME_INDEX)
        if len(positions) != 1:
            raise RuntimeError(f"expected one cached copy of frame {FRAME_INDEX}")
        frame = np.asarray(group["accepted_image"][int(positions[0])], dtype=np.uint8)
        provenance = {
            "configured_source_path": str(group.attrs["configured_source_path"]),
            "resolved_source_path": str(group.attrs["resolved_source_path"]),
            "source_dataset_path": str(handle.attrs["source_dataset_path"]),
            "frame_index": FRAME_INDEX,
        }
    if frame.shape != (732, 968):
        raise RuntimeError(f"unexpected frame shape {frame.shape}")
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


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def overlay_mask(ax: plt.Axes, mask: np.ndarray, color: str, alpha: float) -> None:
    rgb = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgb[..., :3] = to_rgb(color)
    rgb[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(rgb, interpolation="nearest")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, provenance = load_real_frame(args.proxy_hdf5)
    cfg = ClassicalConfig()

    segmentation = segment_dark_ridge(frame, cfg)
    score = segmentation.score
    raw = segmentation.high_threshold_mask
    faint = (
        segmentation.connected_threshold_mask & ~raw
        if segmentation.connected_threshold_mask is not None
        else np.zeros_like(raw)
    )
    closed = segmentation.closed_high_mask
    component = segmentation.component
    component_count = segmentation.component_count
    bounds = crop_bounds(component)

    yy, xx = np.nonzero(component)
    pad = 3
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(frame.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(frame.shape[1], int(xx.max()) + pad + 1)
    thinned_crop = _thin(component[y0:y1, x0:x1])
    skeleton_crop = _prune_skeleton_endpoints(thinned_crop)
    path_crop, endpoint_count, branch_pixels = _skeleton_longest_path(skeleton_crop)
    if path_crop is None:
        raise RuntimeError("worked example unexpectedly has no skeleton path")
    path = path_crop.copy()
    path[:, 0] += x0
    path[:, 1] += y0
    centerline_before_orientation = resample_centerline(path, cfg.n_points)

    thinned = np.zeros_like(component)
    thinned[y0:y1, x0:x1] = thinned_crop
    skeleton = np.zeros_like(component)
    skeleton[y0:y1, x0:x1] = skeleton_crop

    result = extract_centerline(frame, cfg, keep_mask=True)
    if not result.accepted or result.centerline_xy is None or result.tangent_angle is None:
        raise RuntimeError(f"worked example was rejected: {result.rejection_reasons}")
    if not (
        np.allclose(result.centerline_xy, centerline_before_orientation)
        or np.allclose(result.centerline_xy, centerline_before_orientation[::-1])
    ):
        raise RuntimeError("manual stage reconstruction disagrees with extractor output")

    # 0 — untouched real frame.
    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.set_axis_off()
    ax.text(
        0.015,
        0.02,
        f"{RECORDING}  /img_nir  frame {FRAME_INDEX}  |  732 x 968 uint8",
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 5},
    )
    save(fig, args.output_dir / "00_raw_frame.png")

    # 1 — robust local darkness score.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    vmax = float(np.percentile(score[y0:y1, x0:x1], 99))
    image = ax.imshow(score, cmap="magma", vmin=-2.0, vmax=vmax)
    ax.contour(score >= cfg.foreground_z, levels=[0.5], colors=[PALE_CYAN], linewidths=1.0)
    set_crop(ax, bounds)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.015)
    colorbar.set_label("local darkness score", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    save(fig, args.output_dir / "01_dark_ridge_score.png")

    # 2 — thresholded foreground candidates.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    overlay_mask(ax, faint, AMBER, 0.60)
    overlay_mask(ax, raw, MAGENTA, 0.72)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "02_threshold_mask.png")

    # 3 — closed mask and retained largest component.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    overlay_mask(ax, closed & ~component, RED, 0.48)
    overlay_mask(ax, component, CYAN, 0.68)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "03_cleaned_component.png")

    # 4 — one-pixel skeleton after endpoint peeling.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(component, cmap="gray", vmin=0, vmax=1)
    removed_skeleton = thinned & ~skeleton
    overlay_mask(ax, removed_skeleton, RED, 0.9)
    overlay_mask(ax, skeleton, PALE_CYAN, 1.0)
    set_crop(ax, bounds)
    save(fig, args.output_dir / "04_skeleton.png")

    # 5 — longest endpoint-to-endpoint path.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    overlay_mask(ax, skeleton, GRAY, 0.72)
    ax.plot(path[:, 0], path[:, 1], color=ORANGE, linewidth=3.0)
    ax.scatter(path[[0, -1], 0], path[[0, -1], 1], s=75, c=GREEN, edgecolors="black")
    set_crop(ax, bounds)
    save(fig, args.output_dir / "05_longest_path.png")

    # 6 — fixed-size pose.
    centerline = result.centerline_xy
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(centerline[:, 0], centerline[:, 1], color=ORANGE, linewidth=2.4)
    ax.scatter(centerline[:, 0], centerline[:, 1], s=18, c=PALE_CYAN, linewidths=0)
    ax.scatter(centerline[[0, -1], 0], centerline[[0, -1], 1], s=72, c=GREEN, edgecolors="black")
    set_crop(ax, bounds)
    save(fig, args.output_dir / "06_resampled_pose.png")

    # 7 — quality gate and actual values for this frame.
    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(centerline[:, 0], centerline[:, 1], color=ORANGE, linewidth=3.0)
    margin = cfg.boundary_margin
    ax.add_patch(
        Rectangle(
            (margin, margin),
            frame.shape[1] - 1 - 2 * margin,
            frame.shape[0] - 1 - 2 * margin,
            fill=False,
            edgecolor=RED,
            linestyle="--",
            linewidth=1.5,
        )
    )
    qc = result.qc
    label = (
        "PASS\n"
        f"area {int(qc['area']):,} px\u00b2\n"
        f"length {float(qc['length_px']):.1f} px\n"
        f"border clearance {float(qc['boundary_distance_px']):.0f} px\n"
        f"inside body mask {100 * float(qc['tube_support_fraction']):.0f}%"
    )
    ax.text(
        0.72,
        0.97,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=11,
        linespacing=1.35,
        bbox={"facecolor": "#12643b", "alpha": 0.90, "edgecolor": GREEN, "pad": 7},
    )
    ax.set_axis_off()
    save(fig, args.output_dir / "07_quality_gate.png")

    # 8 — ordered output and sparse tangent arrows.
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.imshow(frame, cmap="gray", vmin=80, vmax=225)
    ax.plot(centerline[:, 0], centerline[:, 1], color=ORANGE, linewidth=3.2)
    arrow_indices = np.arange(4, len(centerline) - 4, 9)
    angle = result.tangent_angle[arrow_indices]
    ax.quiver(
        centerline[arrow_indices, 0],
        centerline[arrow_indices, 1],
        np.cos(angle),
        np.sin(angle),
        color=PALE_CYAN,
        angles="xy",
        scale_units="xy",
        scale=0.045,
        width=0.006,
        headwidth=4.2,
        headlength=5.0,
        zorder=4,
    )
    ax.scatter(centerline[[0, -1], 0], centerline[[0, -1], 1], s=85, c=GREEN, edgecolors="black")
    for body_index in (0, 99):
        ax.annotate(
            f"index {body_index}",
            centerline[body_index],
            xytext=(8, -12 if body_index == 0 else 12),
            textcoords="offset points",
            color="white",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2},
        )
    set_crop(ax, bounds)
    save(fig, args.output_dir / "08_tangent_output.png")

    print(
        "Built explainer visuals for "
        f"{provenance['configured_source_path']}:{SOURCE_DATASET}[{FRAME_INDEX}]\n"
        f"accepted={result.accepted} components={component_count} "
        f"raw_endpoints={endpoint_count} branch_pixels={branch_pixels} "
        f"quality={result.quality_score:.3f}"
    )


if __name__ == "__main__":
    main()
