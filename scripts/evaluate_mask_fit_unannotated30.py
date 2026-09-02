#!/usr/bin/env python3
"""Fit the intrinsic tube model to the masks of the frozen 30-frame stress set.

This is the first mask-in, pose-out run: instead of thinning the Section 3
component into a skeleton and demanding simple topology, the 20-value body
model plus one width scale is optimized until its rendered tube matches the
observed component.  Every frame is fit from several starts (the frozen A6
pose where one exists, the longest skeleton path even when branched, and
moment-based arcs) and the best final overlap wins.

The archive frames have no manual centerline annotations.  The operating
metric is intersection-over-union between the rendered hard mask and the
observed component; it measures agreement with the segmentation, not
anatomical truth.  Samples 09, 19, and 29 are known empty frames and are
reported separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_final_geometry_unannotated30 import (
    DEFAULT_RECORDINGS,
    _overlay_mask,
    _show_frame,
    numeric_summary,
    recording_records,
)
from worm_pose_gen.classical import ClassicalConfig, segment_dark_ridge
from worm_pose_gen.mask_fit import (
    MaskFitConfig,
    MaskFitResult,
    fill_narrow_holes,
    fit_mask,
    hard_iou,
    measure_width_template,
    standard_initializations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "mask_fit_unannotated30"
DEFAULT_FROZEN_DIR = PROJECT_ROOT / "docs" / "final_algorithm_unannotated30"
EXPECTED_NO_WORM_INDICES = (9, 19, 29)
IOU_THRESHOLDS = (0.8, 0.9)
# Enclosed background narrower than a 17 px square is segmentation texture;
# a coil interior is far wider and is never filled.
HOLE_FILL_RADIUS_PX = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", action="append", type=Path, dest="recordings")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_frozen(frozen_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray]]:
    metrics = json.loads((frozen_dir / "metrics.json").read_text())
    outcomes = {int(case["sample_index"]): case for case in metrics["per_case"]}
    curves: dict[int, np.ndarray] = {}
    with np.load(frozen_dir / "predictions.npz") as archive:
        for key in archive.files:
            if key.endswith("_a6_centerline_xy"):
                curves[int(key.split("_")[1])] = np.asarray(archive[key], dtype=np.float64)
    return outcomes, curves


def frozen_group(case: dict[str, Any] | None, sample_index: int) -> str:
    if sample_index in EXPECTED_NO_WORM_INDICES:
        return "expected_no_worm"
    if case is None:
        return "unknown"
    if case.get("accepted"):
        return "frozen_accepted"
    stage = str(case.get("failure_stage"))
    if stage == "A1_geometry_selection":
        return "frozen_rejected_A1"
    return "frozen_rejected_A5_A6"


def read_frames(cases: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_path[case["resolved_source_path"]].append(case)
    for path, group in by_path.items():
        with h5py.File(path, "r") as handle:
            dataset = handle[group[0]["source_dataset_path"]]
            for case in group:
                frames[int(case["sample_index"])] = np.asarray(
                    dataset[int(case["frame_index"])], dtype=np.uint8
                )
    return frames


def crop_slices(result: MaskFitResult) -> tuple[slice, slice]:
    crop = result.crop
    return slice(crop.y0, crop.y1), slice(crop.x0, crop.x1)


def plot_case(
    frame: np.ndarray,
    component: np.ndarray,
    raw: np.ndarray,
    result: MaskFitResult,
    record: dict[str, Any],
    path: Path,
) -> None:
    from worm_pose_gen.latent import decode_centerline

    rows, cols = crop_slices(result)
    crop = result.crop
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    extent = (crop.x0 - 0.5, crop.x1 - 0.5, crop.y1 - 0.5, crop.y0 - 0.5)

    def show(ax: plt.Axes, title: str) -> None:
        lower, upper = np.percentile(frame, [1, 99])
        ax.imshow(frame[rows, cols], cmap="gray", vmin=lower, vmax=upper, extent=extent)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    show(axes[0, 0], f"Sample {record['sample_index']:02d}: {record['recording']} frame {record['frame_index']}")
    show(axes[0, 1], "Observed: raw threshold (amber) and hole-filled Section 3 component (magenta)")
    axes[0, 1].imshow(
        np.where(raw[rows, cols], 1.0, np.nan), cmap="autumn", alpha=0.35, extent=extent, interpolation="nearest"
    )
    axes[0, 1].imshow(
        np.where(component[rows, cols], 1.0, np.nan), cmap="spring", alpha=0.45, extent=extent, interpolation="nearest"
    )
    show(axes[0, 2], "Starting states (dashed) and best final centerline (green)")
    colors = plt.get_cmap("tab10")
    for index, start in enumerate(result.initializations):
        curve = decode_centerline(start.latent)
        axes[0, 2].plot(curve[:, 0], curve[:, 1], "--", lw=1.0, color=colors(index % 10), label=start.name)
    axes[0, 2].plot(result.centerline_xy[:, 0], result.centerline_xy[:, 1], "-", lw=2.0, color="#57d68d")
    axes[0, 2].legend(fontsize=7, loc="lower right")
    axes[0, 2].set_xlim(extent[0], extent[1])
    axes[0, 2].set_ylim(extent[2], extent[3])

    show(
        axes[1, 0],
        f"Best fit ({record['best_start']}): IoU {record['final_iou_target']:.3f}, "
        f"length {record['body_length_px']:.0f} px, width {record['width_px']:.1f} px",
    )
    axes[1, 0].contour(
        np.arange(crop.x0, crop.x1), np.arange(crop.y0, crop.y1),
        result.rendered_hard_mask.astype(float), levels=[0.5], colors=["#57d68d"], linewidths=1.2,
    )
    axes[1, 0].plot(result.centerline_xy[:, 0], result.centerline_xy[:, 1], "-", lw=1.4, color="#ff4fa3")
    axes[1, 0].plot(result.centerline_xy[[0, -1], 0], result.centerline_xy[[0, -1], 1], "o", ms=4, color="#ff4fa3")

    observed = component[rows, cols]
    residual = np.zeros((*observed.shape, 3), dtype=np.float32)
    residual[np.logical_and(observed, result.rendered_hard_mask)] = (0.35, 0.35, 0.35)
    residual[np.logical_and(observed, ~result.rendered_hard_mask)] = (1.0, 0.31, 0.64)
    residual[np.logical_and(~observed, result.rendered_hard_mask)] = (0.34, 0.84, 0.55)
    axes[1, 1].imshow(residual, extent=extent, interpolation="nearest")
    axes[1, 1].set_title("Residual: magenta = observed only, green = model only", fontsize=10)
    axes[1, 1].set_axis_off()

    history = result.energy_history
    for index, start in enumerate(result.initializations):
        axes[1, 2].plot(history[:, index], lw=1.0, color=colors(index % 10), label=start.name)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("optimization step (all stages)")
    axes[1, 2].set_ylabel("soft Dice energy at stage scale")
    axes[1, 2].set_title("Energy per start", fontsize=10)
    axes[1, 2].legend(fontsize=7)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_summary(records: list[dict[str, Any]], path: Path) -> None:
    groups = {
        "frozen_accepted": "#57d68d",
        "frozen_rejected_A1": "#ff4fa3",
        "frozen_rejected_A5_A6": "#ffb142",
        "expected_no_worm": "#a7adb4",
    }
    ordered = sorted(records, key=lambda r: r["final_iou_target"], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    axes[0].bar(
        range(len(ordered)),
        [r["final_iou_target"] for r in ordered],
        color=[groups[r["frozen_group"]] for r in ordered],
    )
    axes[0].scatter(
        range(len(ordered)),
        [r["best_start_initial_iou"] for r in ordered],
        marker="_", s=120, color="black", label="IoU of the winning start before optimization",
    )
    for threshold in IOU_THRESHOLDS:
        axes[0].axhline(threshold, ls=":", color="black", lw=0.8)
    axes[0].set_xticks(range(len(ordered)))
    axes[0].set_xticklabels([f"{r['sample_index']:02d}" for r in ordered], fontsize=7)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("sample index, sorted by final IoU")
    axes[0].set_ylabel("IoU of rendered tube vs hole-filled component")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in groups.values()]
    axes[0].legend(handles + [axes[0].collections[0]], list(groups) + ["winning start before optimization"], fontsize=8, loc="lower left")
    axes[0].set_title("Mask-fit overlap per frame, colored by frozen pipeline outcome")

    accepted = [r for r in records if r["frozen_group"] == "frozen_accepted"]
    axes[1].scatter(
        [r["frozen_a6_iou"] for r in accepted],
        [r["final_iou_target"] for r in accepted],
        color=groups["frozen_accepted"],
    )
    for r in accepted:
        axes[1].annotate(f"{r['sample_index']:02d}", (r["frozen_a6_iou"], r["final_iou_target"]), fontsize=7)
    axes[1].plot([0, 1], [0, 1], ls=":", color="black", lw=0.8)
    axes[1].set_xlim(0.5, 1)
    axes[1].set_ylim(0.5, 1)
    axes[1].set_xlabel("IoU of frozen A6 pose rendered with the same template")
    axes[1].set_ylabel("IoU after mask fit")
    axes[1].set_title("Frozen-accepted frames: overlap before and after fitting")
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_visual_index(records: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Mask-fit diagnostic sheets",
        "",
        "Each sheet shows the raw crop, the observed raw threshold and Section 3",
        "component, every starting state with the winning final centerline, the",
        "best fit's rendered boundary, the residual between rendered and observed",
        "masks, and the energy trajectory of every start. IoU is agreement with",
        "the hole-filled segmentation, not anatomical accuracy. Samples 09, 19, and 29 are",
        "known empty frames.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## Sample {record['sample_index']:02d}: `{record['recording']}` frame {record['frame_index']}",
                "",
                f"Frozen pipeline: **{record['frozen_group']}**. Mask fit: IoU "
                f"**{record['final_iou_target']:.3f}** from `{record['best_start']}`, "
                f"length {record['body_length_px']:.0f} px, width {record['width_px']:.1f} px, "
                f"{record['points_in_fov']}/100 points in view.",
                "",
                f"![Mask fit for sample {record['sample_index']:02d}]({record['visual_artifact']})",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, dict[str, Any]] = {}
    for group in ("frozen_accepted", "frozen_rejected_A1", "frozen_rejected_A5_A6", "expected_no_worm"):
        members = [r for r in records if r["frozen_group"] == group]
        entry: dict[str, Any] = {"frames": len(members)}
        if members:
            ious = [r["final_iou_target"] for r in members]
            entry["final_iou_target"] = numeric_summary(ious)
            entry["winning_start_initial_iou"] = numeric_summary(r["best_start_initial_iou"] for r in members)
            for threshold in IOU_THRESHOLDS:
                entry[f"frames_with_iou_at_least_{threshold:g}"] = int(sum(v >= threshold for v in ious))
            entry["best_start_counts"] = {
                name: sum(r["best_start"] == name for r in members)
                for name in sorted({r["best_start"] for r in members})
            }
        by_group[group] = entry
    worm = [r for r in records if r["frozen_group"] != "expected_no_worm"]
    return {
        "requested_frames": len(records),
        "eligible_worm_frames": len(worm),
        "by_frozen_group": by_group,
        "eligible_frames_with_iou_at_least": {
            f"{threshold:g}": int(sum(r["final_iou_target"] >= threshold for r in worm))
            for threshold in IOU_THRESHOLDS
        },
        "eligible_final_iou_target": numeric_summary(r["final_iou_target"] for r in worm),
        "eligible_final_iou_component": numeric_summary(r["final_iou_component"] for r in worm),
        "eligible_final_iou_raw_threshold": numeric_summary(r["final_iou_raw_threshold"] for r in worm),
        "eligible_body_length_px": numeric_summary(r["body_length_px"] for r in worm),
        "eligible_width_px": numeric_summary(r["width_px"] for r in worm),
        "eligible_points_in_fov": numeric_summary(r["points_in_fov"] for r in worm),
        "runtime_seconds": numeric_summary(r["runtime_seconds"] for r in records),
    }


def main() -> int:
    args = parse_args()
    recordings = list(args.recordings or DEFAULT_RECORDINGS)
    if len(recordings) != 3:
        raise SystemExit("exactly three --recording arguments are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "frame_steps"
    visual_dir.mkdir(exist_ok=True)

    cases, provenance = recording_records(recordings)
    frozen_cases, frozen_curves = load_frozen(args.frozen_dir)
    frames = read_frames(cases)
    config = MaskFitConfig()
    classical = ClassicalConfig()

    segmentations: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for case in cases:
        index = int(case["sample_index"])
        segmentation = segment_dark_ridge(frames[index], classical)
        target, filled_area = fill_narrow_holes(
            segmentation.component, HOLE_FILL_RADIUS_PX, device=args.device
        )
        segmentations[index] = (
            target, segmentation.component, segmentation.high_threshold_mask, filled_area
        )

    template_indices = sorted(
        index for index in frozen_curves
        if index in segmentations and frozen_cases.get(index, {}).get("accepted")
    )
    template, template_scales = measure_width_template(
        [segmentations[index][0] for index in template_indices],
        [frozen_curves[index] for index in template_indices],
        n_points=config.n_points,
    )

    records: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {"width_template": template}
    for case in cases:
        index = int(case["sample_index"])
        target, component, raw, filled_area = segmentations[index]
        frozen_case = frozen_cases.get(index)
        group = frozen_group(frozen_case, index)
        reference = frozen_curves.get(index) if group == "frozen_accepted" else None
        started = time.perf_counter()
        starts = standard_initializations(target, reference_centerline_xy=reference, config=config)
        result = fit_mask(
            target, starts, width_template=template, config=config, device=args.device,
            extra_masks={"component": component, "raw_threshold": raw},
        )
        runtime = time.perf_counter() - started
        best = result.records[result.best_index]
        reference_record = next((r for r in result.records if r["name"] == "reference"), None)
        record: dict[str, Any] = {
            "sample_index": index,
            "sample_id": case["sample_id"],
            "recording": case["recording"],
            "frame_index": int(case["frame_index"]),
            "expected_no_worm": index in EXPECTED_NO_WORM_INDICES,
            "frozen_group": group,
            "frozen_failure_stage": None if frozen_case is None else frozen_case.get("failure_stage"),
            "component_area_px": int(component.sum()),
            "hole_filled_area_px": int(filled_area),
            "target_area_px": int(target.sum()),
            "raw_threshold_area_px": int(raw.sum()),
            "starts": result.records,
            "best_start": best["name"],
            "best_start_initial_iou": best["initial_iou"],
            "final_iou_target": best["final_iou"],
            "final_iou_component": result.extra_iou["component"],
            "final_iou_raw_threshold": result.extra_iou["raw_threshold"],
            "final_soft_dice_energy": best["final_soft_dice_energy"],
            "frozen_a6_iou": None if reference_record is None else reference_record["initial_iou"],
            "body_length_px": result.body_length_px,
            "width_px": result.width_px,
            "points_in_fov": result.points_in_fov,
            "crop": asdict(result.crop),
            "runtime_seconds": runtime,
        }
        visual_name = f"sample_{index:02d}_{case['recording']}_frame_{int(case['frame_index']):05d}.jpg"
        record["visual_artifact"] = f"frame_steps/{visual_name}"
        plot_case(frames[index], target, raw, result, record, visual_dir / visual_name)
        predictions[f"sample_{index:02d}_centerline_xy"] = result.centerline_xy
        predictions[f"sample_{index:02d}_latent"] = result.latent
        predictions[f"sample_{index:02d}_width_profile"] = result.width_profile
        records.append(record)
        print(
            json.dumps(
                {
                    "sample_index": index,
                    "frozen_group": group,
                    "best_start": best["name"],
                    "final_iou": round(best["final_iou"], 4),
                    "length_px": round(result.body_length_px, 1),
                    "seconds": round(runtime, 1),
                }
            ),
            flush=True,
        )

    summary = summarize(records)
    metrics = {
        "status": "mask_fit_unannotated_operational_run",
        "evidence_boundary": {
            "manual_annotations_available": False,
            "anatomical_accuracy_claim": False,
            "iou_measures_agreement_with_segmentation_only": True,
            "protected_2025_holdout_opened": False,
            "expected_no_worm_indices": list(EXPECTED_NO_WORM_INDICES),
        },
        "method": {
            "observed_mask": "Section 3 largest component of the frozen local-darkness threshold, enclosed holes narrower than a 17 px square filled",
            "hole_fill_radius_px": HOLE_FILL_RADIUS_PX,
            "body_model": "16 cubic tangent coefficients, rotation, length, centroid, one width scale",
            "width_template_source": {
                "frozen_accepted_sample_indices": template_indices,
                "midbody_width_px": numeric_summary(template_scales),
            },
            "energy": "soft Dice between segment-rendered tube and signed-distance-blurred component, coarse to fine",
            "starts": "frozen A6 pose when accepted, longest skeleton path, moment-based straight and arc starts",
            "selection": "lowest final soft Dice energy among starts",
            "mask_fit_config": asdict(config),
            "classical_config": asdict(classical),
            "device": str(args.device or ("cuda" if torch.cuda.is_available() else "cpu")),
        },
        "inputs": provenance,
        "summary": summary,
        "per_case": records,
        "git_commit": git_commit(),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez_compressed(args.output_dir / "predictions.npz", **predictions)
    plot_summary(records, args.output_dir / "summary.png")
    write_visual_index(records, args.output_dir / "FRAME_STEPS.md")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
