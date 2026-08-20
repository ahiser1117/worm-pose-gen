#!/usr/bin/env python3
"""EXP-SMC-004 leave-one-frame-out width-capacity and anti-compensation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.exp_smc_001_002_audit import _read_windows, _verified_annotations
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from exp_smc_001_002_audit import _read_windows, _verified_annotations
from worm_pose_gen.annotation import resample_polyline
from worm_pose_gen.anchors import (
    estimate_width_along_normals,
    render_centerline_mask,
)
from worm_pose_gen.segmentation import SoftForegroundConfig, segment_soft_foreground
from worm_pose_gen.width import fit_profile_parameters, fit_width_profile_model


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/smc_exp_004_width_model.json"
DEFAULT_OUTPUT = REPO / "experiments/exp_smc_004_width_model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.logical_or(first, second).sum())
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def summary(values: list[float]) -> dict[str, float | int | None]:
    sample = np.asarray(values, dtype=np.float64)
    if not len(sample):
        return {"n": 0, "median": None, "mean": None, "p95": None, "std": None}
    return {
        "n": int(len(sample)),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "p95": float(np.percentile(sample, 95)),
        "std": float(np.std(sample)),
    }


def _safe_output(output: Path) -> None:
    allowed = {"config.json", "notes.md"}
    if output.exists():
        unexpected = [path.name for path in output.iterdir() if path.name not in allowed]
        if unexpected:
            raise FileExistsError(f"refusing existing generated output: {unexpected}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    output = args.output.resolve()
    config = json.loads(config_path.read_text())
    segmentation_path = (REPO / config["inputs"]["segmentation_config"]).resolve(strict=True)
    segmentation_config_document = json.loads(segmentation_path.read_text())
    annotation_path = Path(config["inputs"]["annotations"]).resolve(strict=True)
    manifest_path = (REPO / config["inputs"]["manifest"]).resolve(strict=True)
    rows = _verified_annotations(manifest_path, annotation_path, segmentation_config_document)
    excluded = set(config["inputs"]["excluded_expert_hard_samples"])
    selected = [
        (source, annotation)
        for source, annotation in rows
        if annotation.is_complete and annotation.sample_id not in excluded
    ]
    if len(selected) != 16:
        raise RuntimeError(f"expected 16 complete non-hard development traces, found {len(selected)}")
    _safe_output(output)
    frames = _read_windows(selected, 0)
    segment_config = SoftForegroundConfig(**segmentation_config_document["soft_foreground_config"])
    width_config = config["width"]
    n_points = int(width_config["n_points"])
    step = float(width_config["normal_walk_step_px"])
    minimum = float(width_config["minimum_px"])
    maximum = float(width_config["maximum_px"])
    scale_bounds = tuple(map(float, width_config["scale_bounds"]))
    shift = float(width_config["translation_stress_px"])
    grid_low, grid_high, grid_count = width_config["translation_scale_grid"]
    scale_grid = np.linspace(float(grid_low), float(grid_high), int(grid_count))

    cases: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for source, annotation in selected:
        recording = str(source["recording"])
        frame = frames[(recording, int(source["frame_index"]))]
        segmentation = segment_soft_foreground(frame, segment_config)
        centerline = resample_polyline(annotation.points_xy, n_points)
        measured = estimate_width_along_normals(
            segmentation.cleaned_mask, centerline, step=step
        )
        measured = 0.5 * (measured + measured[::-1])
        arrays[annotation.sample_id] = {
            "frame": frame,
            "mask": segmentation.cleaned_mask,
            "centerline": centerline,
            "measured": measured,
        }
        cases.append({
            "sample_id": annotation.sample_id,
            "recording": recording,
            "frame_index": int(source["frame_index"]),
            "selection_stratum": source["selection_stratum"],
        })

    by_recording: dict[str, list[str]] = {}
    for case in cases:
        by_recording.setdefault(case["recording"], []).append(case["sample_id"])
    if any(len(sample_ids) < 3 for sample_ids in by_recording.values()):
        raise RuntimeError("width LOO requires at least three selected frames per recording")

    methods = ("fixed_mean", "mean_times_scale", "width_pca_1", "width_pca_2", "per_point_oracle")
    for case in cases:
        sample_id = case["sample_id"]
        values = arrays[sample_id]
        train_ids = [
            other for other in by_recording[case["recording"]] if other != sample_id
        ]
        train = np.stack([arrays[other]["measured"] for other in train_ids])
        target = values["measured"]
        models = {
            0: fit_width_profile_model(
                train, components=0, minimum=minimum, maximum=maximum
            ),
            1: fit_width_profile_model(
                train, components=1, minimum=minimum, maximum=maximum
            ),
            2: fit_width_profile_model(
                train, components=2, minimum=minimum, maximum=maximum
            ),
        }
        fixed, _, _ = fit_profile_parameters(
            target, models[0], fit_scale=False, scale_bounds=scale_bounds
        )
        scaled, alpha, _ = fit_profile_parameters(
            target, models[0], fit_scale=True, scale_bounds=scale_bounds
        )
        pca1, _, coefficient1 = fit_profile_parameters(
            target, models[1], fit_scale=False, scale_bounds=scale_bounds
        )
        pca2, _, coefficient2 = fit_profile_parameters(
            target, models[2], fit_scale=False, scale_bounds=scale_bounds
        )
        profiles = {
            "fixed_mean": fixed,
            "mean_times_scale": scaled,
            "width_pca_1": pca1,
            "width_pca_2": pca2,
            "per_point_oracle": target,
        }
        method_values: dict[str, Any] = {}
        for name in methods:
            profile = profiles[name]
            rendered = render_centerline_mask(
                values["centerline"], profile, values["mask"].shape
            )
            method_values[name] = {
                "mask_iou": iou(rendered, values["mask"]),
                "width_rmse_px": float(np.sqrt(np.mean(np.square(profile - target)))),
            }
        shifted = values["centerline"] + np.asarray([shift, 0.0])
        shifted_ious = []
        for candidate_scale in scale_grid:
            shifted_render = render_centerline_mask(
                shifted,
                candidate_scale * models[0].mean,
                values["mask"].shape,
            )
            shifted_ious.append(iou(shifted_render, values["mask"]))
        best_index = int(np.argmax(shifted_ious))
        case.update({
            "width_scale": alpha,
            "width_pca_1_coefficient": coefficient1.tolist(),
            "width_pca_2_coefficients": coefficient2.tolist(),
            "methods": method_values,
            "translation_stress": {
                "shift_xy_px": [shift, 0.0],
                "best_refit_scale": float(scale_grid[best_index]),
                "best_mask_iou": float(shifted_ious[best_index]),
                "selected_correct_mask_iou": method_values["mean_times_scale"]["mask_iou"],
                "iou_drop": float(
                    method_values["mean_times_scale"]["mask_iou"] - shifted_ious[best_index]
                ),
            },
            "measured_width_summary_px": summary(target.tolist()),
            "_profiles": profiles,
        })

    method_summary = {
        method: {
            "mask_iou": summary([case["methods"][method]["mask_iou"] for case in cases]),
            "width_rmse_px": summary(
                [case["methods"][method]["width_rmse_px"] for case in cases]
            ),
        }
        for method in methods
    }
    scale_summary = summary([case["width_scale"] for case in cases])
    stress_summary = summary([case["translation_stress"]["iou_drop"] for case in cases])
    pca_gain = float(
        method_summary["width_pca_2"]["mask_iou"]["median"]
        - method_summary["mean_times_scale"]["mask_iou"]["median"]
    )
    gate = config["gate"]
    checks = {
        "selected_median_mask_iou": (
            method_summary["mean_times_scale"]["mask_iou"]["median"]
            >= float(gate["selected_median_mask_iou_min"])
        ),
        "pca2_median_iou_gain_over_scale": (
            pca_gain <= float(gate["pca2_median_iou_gain_over_scale_max"])
        ),
        "translation_stress_median_iou_drop": (
            stress_summary["median"]
            >= float(gate["translation_stress_median_iou_drop_min"])
        ),
        "scale_standard_deviation": (
            scale_summary["std"] <= float(gate["scale_standard_deviation_max"])
        ),
    }
    decision = {
        "passed": all(checks.values()),
        "decision": "SUPPORTED" if all(checks.values()) else "NOT_SUPPORTED",
        "selected_model": "recording_mean_times_bounded_frame_scale",
        "checks": checks,
        "observed": {
            "selected_median_mask_iou": method_summary["mean_times_scale"]["mask_iou"]["median"],
            "pca2_median_iou_gain_over_scale": pca_gain,
            "translation_stress_median_iou_drop": stress_summary["median"],
            "scale_standard_deviation": scale_summary["std"],
        },
    }
    public_cases = [
        {key: value for key, value in case.items() if not key.startswith("_")}
        for case in cases
    ]
    metrics = {
        "schema_version": 1,
        "experiment": "EXP-SMC-004",
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "segmentation_config": str(segmentation_path),
            "segmentation_config_sha256": sha256(segmentation_path),
            "annotations": str(annotation_path),
            "annotations_sha256": sha256(annotation_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "protected_2025_holdout_opened": False,
        },
        "evidence_boundary": config["evidence_boundary"],
        "selected_frames": len(cases),
        "selected_by_recording": {key: len(value) for key, value in by_recording.items()},
        "method_summary": method_summary,
        "width_scale": scale_summary,
        "translation_stress_iou_drop": stress_summary,
        "decision": decision,
        "per_case": public_cases,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    colors = {
        "fixed_mean": "#0072b2",
        "mean_times_scale": "#009e73",
        "width_pca_1": "#e69f00",
        "width_pca_2": "#cc79a7",
        "per_point_oracle": "#555555",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(methods)
    medians = [method_summary[name]["mask_iou"]["median"] for name in names]
    axes[0].bar(range(len(names)), medians, color=[colors[name] for name in names])
    axes[0].set_xticks(range(len(names)), [name.replace("_", "\n") for name in names], fontsize=8)
    axes[0].set_ylabel("median cleaned-mask IoU")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(float(gate["selected_median_mask_iou_min"]), color="#d55e00", ls="--")
    drops = [case["translation_stress"]["iou_drop"] for case in cases]
    axes[1].hist(drops, bins=8, color="#009e73", alpha=0.85)
    axes[1].axvline(float(gate["translation_stress_median_iou_drop_min"]), color="#d55e00", ls="--")
    axes[1].set_xlabel("IoU drop after 10 px shift and scale refit")
    axes[1].set_ylabel("frames")
    fig.suptitle("EXP-SMC-004 width capacity and anti-compensation")
    fig.tight_layout()
    fig.savefig(output / "figures/width_model_summary.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(len(by_recording), 1, figsize=(10, 3 * len(by_recording)), squeeze=False)
    s = np.linspace(0, 1, n_points)
    for axis, (recording, sample_ids) in zip(axes.flat, sorted(by_recording.items()), strict=True):
        for sample_id in sample_ids:
            axis.plot(s, arrays[sample_id]["measured"], alpha=0.35, color="#777777")
        mean = np.stack([arrays[sample_id]["measured"] for sample_id in sample_ids]).mean(0)
        axis.plot(s, mean, color="#0072b2", lw=2.5, label="recording mean")
        axis.set_title(recording)
        axis.set_ylabel("width (px)")
        axis.legend()
    axes[-1, 0].set_xlabel("normalized body position")
    fig.tight_layout()
    fig.savefig(output / "figures/width_profiles_by_recording.png", dpi=170)
    plt.close(fig)

    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
