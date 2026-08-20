#!/usr/bin/env python3
"""Evaluate one trained EXP-003 checkpoint on the frozen Tier-A primary snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

try:
    from scripts.evaluate_tier_a_primary import (
        _read_frames,
        _verified_inputs,
        complete_curve_metrics,
        sha256_file,
        visible_trace_metrics,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from evaluate_tier_a_primary import (
        _read_frames,
        _verified_inputs,
        complete_curve_metrics,
        sha256_file,
        visible_trace_metrics,
    )
from worm_pose_gen.model import WormProposalModule
from worm_pose_gen.spatial_model import SpatialPoseModule
from worm_pose_gen.training_data import normalize_image


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "p95": None}
    sample = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "p95": float(np.percentile(sample, 95)),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [value["complete_metrics"] for value in cases if value["complete_metrics"]]
    visible = [value["visible_metrics"] for value in cases if value["visible_metrics"]]
    return {
        "requested_frames": len(cases),
        "algorithmic_success_frames": sum(value["algorithmic_success"] for value in cases),
        "algorithmic_failure_frames": sum(not value["algorithmic_success"] for value in cases),
        "complete_trace_scored_frames": len(complete),
        "truncated_visible_trace_scored_frames": len(visible),
        "complete_trace": {
            "per_frame_median_point_distance_px": _summary(
                [value["median_point_distance_px"] for value in complete]
            ),
            "all_body_positions_point_distance_px": _summary(
                [point for value in complete for point in value["point_distance_px"]]
            ),
            "per_frame_mean_tangent_error_deg": _summary(
                [value["mean_tangent_error_deg"] for value in complete]
            ),
            "per_frame_mean_endpoint_error_px": _summary(
                [value["mean_endpoint_error_px"] for value in complete]
            ),
            "per_frame_body_length_error_fraction": _summary(
                [value["body_length_error_fraction"] for value in complete]
            ),
        },
        "truncated_visible_trace": {
            "per_frame_median_distance_px": _summary(
                [value["median_visible_trace_distance_px"] for value in visible]
            ),
            "all_visible_positions_distance_px": _summary(
                [point for value in visible for point in value["visible_trace_distance_px"]]
            ),
            "per_frame_mean_axis_error_deg": _summary(
                [value["mean_visible_trace_axis_error_deg"] for value in visible]
            ),
            "interpretation": "one-way annotated-visible-trace coverage only",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Tier-A evaluation: {args.output}")
    config = yaml.safe_load(args.config.read_text())
    run = json.loads(args.run_metrics.read_text())
    if (
        run.get("experiment") != "EXP-003"
        or run.get("status") != "TRAINED_PENDING_EVALUATION"
        or int(run.get("global_step", -1)) != int(config["training"]["maximum_steps"])
        or run.get("config_sha256") != sha256_file(args.config)
        or run.get("evidence_boundary", {}).get("protected_holdout_opened") is not False
        or run.get("evidence_boundary", {}).get("primary_Tier_A_used_for_gradients") is not False
    ):
        raise RuntimeError("run metrics do not identify a complete leakage-safe EXP-003 run")
    checkpoint = Path(run["checkpoint_path"])
    if sha256_file(checkpoint) != run["checkpoint_sha256"]:
        raise RuntimeError("EXP-003 checkpoint hash changed")
    variant = str(run["variant"])
    device = torch.device(args.device)
    if variant == "global_intrinsic_budget_matched":
        model = WormProposalModule.load_from_checkpoint(checkpoint, map_location=device)
    else:
        model = SpatialPoseModule.load_from_checkpoint(checkpoint, map_location=device)
    model = model.to(device).eval()
    if (
        int(model.hparams.model_seed) != int(run["model_seed"])
        or int(model.hparams.data_seed) != int(run["data_seed"])
        or model.hparams.training_order_sha256 != run["training_order_sha256"]
    ):
        raise RuntimeError("checkpoint training identity mismatches run metrics")
    _, rows = _verified_inputs(args.manifest, args.annotations)
    frames = _read_frames(rows)
    images = torch.stack([
        normalize_image(frames[annotation.sample_id])
        for _, annotation in rows
    ]).to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    scale = images.new_tensor((968 / 256, 732 / 192))
    predictions = (output["centerline_xy"] * scale).detach().cpu().numpy()
    selection = output.get("selection_score")
    selection_values = (
        [None] * len(rows)
        if selection is None
        else selection.detach().cpu().numpy().astype(float).tolist()
    )
    cases: list[dict[str, Any]] = []
    for index, (source, annotation) in enumerate(rows):
        prediction = predictions[index]
        success = bool(np.isfinite(prediction).all())
        complete = None
        visible = None
        if success and annotation.is_complete:
            complete = complete_curve_metrics(prediction, annotation.points_xy)
        elif success and annotation.trace_state == "truncated":
            visible = visible_trace_metrics(prediction, annotation.points_xy)
        cases.append({
            "sample_id": annotation.sample_id,
            "recording": source["recording"],
            "frame_index": int(source["frame_index"]),
            "selection_stratum": source["selection_stratum"],
            "trace_state": annotation.trace_state,
            "algorithmic_success": success,
            "selection_score_uncalibrated": selection_values[index],
            "prediction_centerline_xy": prediction.tolist() if success else None,
            "complete_metrics": complete,
            "visible_metrics": visible,
        })
    result = {
        "schema_version": 1,
        "experiment": "EXP-003",
        "evaluation": "Tier-A-primary30",
        "variant": variant,
        "model_seed": int(run["model_seed"]),
        "data_seed": int(run["data_seed"]),
        "global_step": int(run["global_step"]),
        "run_metrics": str(args.run_metrics.resolve(strict=True)),
        "run_metrics_sha256": sha256_file(args.run_metrics),
        "checkpoint_path": str(checkpoint.resolve(strict=True)),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "config_sha256": run["config_sha256"],
        "training_order_sha256": run["training_order_sha256"],
        "materialized_dataset_sha256": run["materialized_dataset_sha256"],
        "training_source_sha256": run["source_sha256"],
        "parameters": int(run["parameters"]),
        "training_throughput": run["throughput"],
        "training_gpu": run["gpu"],
        "annotations_sha256": sha256_file(args.annotations),
        "summary": summarize(cases),
        "per_case": cases,
        "runtime": {
            "device": str(device),
            "seconds_30_frames": elapsed,
            "frames_per_second": len(rows) / elapsed,
            "scope": "in_memory_batch_preprocess_excluded_forward_only",
        },
        "quality_boundary": {
            "algorithmic_failure": "nonfinite_or_inference_failure_only",
            "selection_score_calibrated": False,
            "quality_threshold_preregistered": False,
        },
        "evidence_boundary": {
            "protected_holdout_opened": False,
            "primary_Tier_A_used_for_gradients": False,
            "repeat_annotations_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "variant": variant,
        "model_seed": run["model_seed"],
        "summary": result["summary"],
        "protected_holdout_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
