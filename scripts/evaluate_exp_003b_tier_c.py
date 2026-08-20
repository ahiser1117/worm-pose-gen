#!/usr/bin/env python3
"""Apply the preregistered controlled Tier-C gate to one EXP-003B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

try:
    from scripts.evaluate_exp_003_tier_c import _case_metrics, _summary
except ModuleNotFoundError:
    from evaluate_exp_003_tier_c import _case_metrics, _summary
from worm_pose_gen.topology_rescue_model import SoftAnchoredIntrinsicModule
from worm_pose_gen.training_data import (
    MaterializedPoseDataset,
    SyntheticTierCDataset,
    materialized_dataset_sha256,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite EXP-003B gate: {args.output}")
    config = yaml.safe_load(args.config.read_text())
    run = json.loads(args.run_metrics.read_text())
    if (
        config.get("experiment") != "EXP-003B"
        or run.get("experiment") != "EXP-003B"
        or run.get("status") != "TRAINED_PENDING_TIER_C_GATE"
        or run.get("config_sha256") != sha256_file(args.config)
        or run.get("evidence_boundary", {}).get("Tier_A_evaluated") is not False
    ):
        raise RuntimeError("run does not identify a complete Tier-A-blind EXP-003B model")
    checkpoint = Path(run["checkpoint_path"])
    if sha256_file(checkpoint) != run["checkpoint_sha256"]:
        raise RuntimeError("EXP-003B checkpoint hash changed")
    training = config["training"]
    dataset = MaterializedPoseDataset(SyntheticTierCDataset(
        int(training["synthetic_validation_samples"]),
        seed=int(training["data_seed"]) + 5_000_000 + int(training["fold"]) * 100_000,
        profile="held_out",
    ))
    dataset_hash = materialized_dataset_sha256(dataset)
    if dataset_hash != run["materialized_dataset_sha256"]["tier_c_validation"]:
        raise RuntimeError("EXP-003B Tier-C validation tensors changed")
    device = torch.device(args.device)
    model = SoftAnchoredIntrinsicModule.load_from_checkpoint(checkpoint, map_location=device)
    model = model.to(device).eval()
    cases = []
    loader = DataLoader(
        dataset,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["centerline_xy"].to(device, non_blocking=True)
            support = batch["image_support_target"].to(device, non_blocking=True)
            cases.extend(_case_metrics(model(image)["centerline_xy"], target, support))
    summary = _summary(cases)
    fully_visible = summary["fully_visible_43"]
    gate = config["controlled_gate"]
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
    passed = (
        observed["median_full_latent_point_distance_px"]
        <= float(gate["median_full_latent_point_distance_px_at_most"])
        and observed["median_mean_tangent_error_degrees"]
        <= float(gate["median_mean_tangent_error_degrees_at_most"])
        and observed["median_body_length_error_fraction"]
        <= float(gate["median_body_length_error_fraction_at_most"])
    )
    result = {
        "schema_version": 1,
        "experiment": "EXP-003B",
        "evaluation": "frozen-Tier-C-128-controlled-gate",
        "model_seed": int(run["model_seed"]),
        "run_metrics": str(args.run_metrics.resolve(strict=True)),
        "run_metrics_sha256": sha256_file(args.run_metrics),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "materialized_tier_c_validation_sha256": dataset_hash,
        "summary": summary,
        "controlled_gate": {
            "thresholds": gate,
            "observed_fully_visible_43": observed,
            "passed": passed,
            "decision": "AUTHORIZE_REPEAT_SEEDS" if passed else "REJECT_WITHOUT_TIER_A",
        },
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
