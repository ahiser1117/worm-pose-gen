#!/usr/bin/env python3
"""Apply the frozen Tier-C gate to the EXP-004 analytic 5k primary run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
import yaml

try:
    from scripts.evaluate_exp_003_tier_c import _case_metrics, _summary
    from scripts.train_exp_004 import _validate, _validate_gpu
except ModuleNotFoundError:
    from evaluate_exp_003_tier_c import _case_metrics, _summary
    from train_exp_004 import _validate, _validate_gpu
from worm_pose_gen.topology_rescue_model import SoftAnchoredIntrinsicModule
from worm_pose_gen.training_data import (
    MaterializedPoseDataset,
    SyntheticTierCDataset,
    materialized_dataset_sha256,
    sha256_file,
)


def controlled_gate_decision(
    observed: Mapping[str, float], gate: Mapping[str, Any]
) -> tuple[bool, str]:
    names = (
        "median_full_latent_point_distance_px",
        "median_mean_tangent_error_degrees",
        "median_body_length_error_fraction",
    )
    if set(observed) != set(names) or any(
        not math.isfinite(float(observed[name])) for name in names
    ):
        raise RuntimeError("EXP-004 controlled-gate metrics are missing or non-finite")
    passed = (
        float(observed[names[0]])
        <= float(gate["median_full_latent_point_distance_px_at_most"])
        and float(observed[names[1]])
        <= float(gate["median_mean_tangent_error_degrees_at_most"])
        and float(observed[names[2]])
        <= float(gate["median_body_length_error_fraction_at_most"])
    )
    return passed, (
        "AUTHORIZE_REPEAT_SEEDS_ONLY"
        if passed
        else "PRIMARY_CONTROLLED_GATE_FAIL_KEEP_ALL_PROTECTED_EVIDENCE_CLOSED"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite EXP-004 gate: {args.output}")
    config = yaml.safe_load(args.config.read_text())
    _validate(config)
    config_sha256 = sha256_file(args.config)
    run = json.loads(args.run_metrics.read_text())
    boundary = run.get("evidence_boundary", {})
    if (
        run.get("experiment") != "EXP-004"
        or run.get("phase") != "analytic_5k_scale_control"
        or run.get("status") != "TRAINED_PENDING_TIER_C_GATE"
        or run.get("model_seed") != int(config["training"]["primary_model_seed"])
        or run.get("global_step") != int(config["training"]["maximum_steps"])
        or run.get("config_sha256") != config_sha256
        or run.get("training_samples") != 5053
        or run.get("analytic_training_samples") != 5000
        or run.get("excluded_proxy_rows") != 9
        or run.get("repeat_authorization") is not None
        or boundary.get("Tier_A_evaluated") is not False
        or boundary.get("protected_holdout_opened") is not False
        or boundary.get("repeat_annotations_used") is not False
    ):
        raise RuntimeError("run does not identify a complete protected EXP-004 primary model")
    checkpoint = Path(run["checkpoint_path"])
    if sha256_file(checkpoint) != run.get("checkpoint_sha256"):
        raise RuntimeError("EXP-004 checkpoint hash changed")
    materialization_path = Path(run["materialization_provenance_path"])
    if sha256_file(materialization_path) != run.get("materialization_provenance_sha256"):
        raise RuntimeError("EXP-004 materialization provenance hash changed")
    materialization = json.loads(materialization_path.read_text())
    if (
        materialization.get("status") != "FROZEN_BEFORE_OPTIMIZATION"
        or materialization.get("config_sha256") != config_sha256
        or materialization.get("model_seed") != run["model_seed"]
        or materialization.get("materialized_dataset_sha256")
        != run.get("materialized_dataset_sha256")
        or materialization.get("evidence_boundary") != boundary
    ):
        raise RuntimeError("EXP-004 materialization provenance identity changed")
    root = Path(__file__).resolve().parents[1]
    for name, expected in run.get("source_sha256", {}).items():
        if name != "scripts/train_exp_004.py" and sha256_file(root / name) != expected:
            raise RuntimeError(f"EXP-004 evaluated source changed: {name}")
    if sha256_file(root / "scripts/train_exp_004.py") != run["source_sha256"].get(
        "scripts/train_exp_004.py"
    ):
        raise RuntimeError("EXP-004 training script changed after the run")

    gpu = _validate_gpu(config)
    training = config["training"]
    dataset = MaterializedPoseDataset(
        SyntheticTierCDataset(
            int(training["synthetic_validation_samples"]),
            seed=int(training["data_seed"])
            + 5_000_000
            + int(training["fold"]) * 100_000,
            profile="held_out",
        )
    )
    dataset_hash = materialized_dataset_sha256(dataset)
    expected_dataset_hash = config["provenance"][
        "expected_tier_c_validation_materialized_sha256"
    ]
    if (
        dataset_hash != expected_dataset_hash
        or dataset_hash != run["materialized_dataset_sha256"]["tier_c_validation"]
    ):
        raise RuntimeError("frozen EXP-004 Tier-C validation tensors changed")

    device = torch.device("cuda")
    model = SoftAnchoredIntrinsicModule.load_from_checkpoint(checkpoint, map_location=device)
    model = model.to(device).eval()
    cases = []
    loader = DataLoader(
        dataset,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["centerline_xy"].to(device, non_blocking=True)
            support = batch["image_support_target"].to(device, non_blocking=True)
            cases.extend(_case_metrics(model(image)["centerline_xy"], target, support))
    summary = _summary(cases)
    fully_visible = summary["fully_visible_43"]
    observed = {
        "median_full_latent_point_distance_px": fully_visible[
            "median_full_latent_point_distance_px"
        ]["median"],
        "median_mean_tangent_error_degrees": fully_visible[
            "mean_full_latent_tangent_error_deg"
        ]["median"],
        "median_body_length_error_fraction": fully_visible[
            "body_length_error_fraction"
        ]["median"],
    }
    passed, decision = controlled_gate_decision(observed, config["controlled_gate"])
    result = {
        "schema_version": 1,
        "experiment": "EXP-004",
        "phase": "analytic_5k_scale_control",
        "evaluation": "frozen-Tier-C-128-controlled-gate",
        "model_seed": int(run["model_seed"]),
        "config_path": str(args.config.resolve(strict=True)),
        "config_sha256": config_sha256,
        "run_metrics": str(args.run_metrics.resolve(strict=True)),
        "run_metrics_sha256": sha256_file(args.run_metrics),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "materialization_provenance_sha256": run["materialization_provenance_sha256"],
        "materialized_tier_c_validation_sha256": dataset_hash,
        "summary": summary,
        "controlled_gate": {
            "thresholds": config["controlled_gate"],
            "observed_fully_visible_43": observed,
            "passed": passed,
            "decision": decision,
        },
        "gpu": gpu,
        "evidence_boundary": {
            "protected_holdout_opened": False,
            "Tier_A_evaluated": False,
            "repeat_annotations_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["controlled_gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
