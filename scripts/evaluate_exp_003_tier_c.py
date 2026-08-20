#!/usr/bin/env python3
"""Evaluate the complete EXP-003 matrix on frozen controlled Tier-C truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import yaml

from worm_pose_gen.geometry import tangent_angles, wrap_angle
from worm_pose_gen.model import WormProposalModule
from worm_pose_gen.spatial_model import SpatialPoseModule
from worm_pose_gen.training_data import (
    MaterializedPoseDataset,
    SyntheticTierCDataset,
    materialized_dataset_sha256,
    sha256_file,
)


VARIANTS = (
    "global_intrinsic_budget_matched",
    "dense_centerline_field",
    "anchored_intrinsic_grid",
)
SEEDS = (20260819, 20260820, 20260821)


def _heatmap_argmax(logits: Tensor, *, image_height: int = 192, image_width: int = 256) -> Tensor:
    """Decode each ordered point at the maximum-probability heatmap cell."""

    if logits.ndim != 4:
        raise ValueError("heatmap logits must be [B,N,H,W]")
    _, _, height, width = logits.shape
    index = logits.flatten(-2).argmax(-1)
    x = (index.remainder(width).to(logits.dtype) + 0.5) * (image_width / width)
    y = (index.div(width, rounding_mode="floor").to(logits.dtype) + 0.5) * (
        image_height / height
    )
    return torch.stack((x, y), dim=-1)


def _oracle_candidate(candidates: Tensor, target: Tensor) -> Tensor:
    """Select the candidate with lowest truth-aligned median distance per case."""

    if candidates.ndim != 4 or candidates.shape[0] != target.shape[0] or candidates.shape[2:] != target.shape[1:]:
        raise ValueError("candidates must be [B,C,N,2] matching target [B,N,2]")
    scale = target.new_tensor((968 / 256, 732 / 192))
    forward = torch.linalg.vector_norm(
        (candidates - target[:, None]) * scale, dim=-1
    ).median(-1).values
    reverse = torch.linalg.vector_norm(
        (candidates - target.flip(-2)[:, None]) * scale, dim=-1
    ).median(-1).values
    selected = torch.minimum(forward, reverse).argmin(-1)
    return candidates[torch.arange(len(candidates), device=candidates.device), selected]


def _case_metrics(prediction: Tensor, target: Tensor, support: Tensor) -> list[dict[str, Any]]:
    """Return orientation-symmetric native-coordinate metrics for one batch."""

    if prediction.shape != target.shape or target.shape[-2:] != (100, 2):
        raise ValueError("prediction and target must both be [B,100,2]")
    scale = target.new_tensor((968 / 256, 732 / 192))
    forward_distance = torch.linalg.vector_norm((prediction - target) * scale, dim=-1)
    reverse_distance = torch.linalg.vector_norm((prediction - target.flip(-2)) * scale, dim=-1)
    reverse = reverse_distance.median(-1).values < forward_distance.median(-1).values
    condition = reverse[:, None, None]
    aligned_target = torch.where(condition, target.flip(-2), target)
    aligned_support = torch.where(reverse[:, None], support.flip(-1), support)
    distance = torch.linalg.vector_norm((prediction - aligned_target) * scale, dim=-1)
    predicted_angle = tangent_angles(prediction * scale)
    target_angle = tangent_angles(aligned_target * scale)
    angle = wrap_angle(predicted_angle - target_angle).abs() * 180 / torch.pi
    predicted_length = torch.linalg.vector_norm(
        (prediction[:, 1:] - prediction[:, :-1]) * scale, dim=-1
    ).sum(-1)
    target_length = torch.linalg.vector_norm(
        (aligned_target[:, 1:] - aligned_target[:, :-1]) * scale, dim=-1
    ).sum(-1)
    results = []
    for index in range(len(prediction)):
        visible = distance[index][aligned_support[index].bool()]
        results.append({
            "fully_visible": bool(support[index].all()),
            "orientation_reversed": bool(reverse[index]),
            "median_full_latent_point_distance_px": float(distance[index].median()),
            "p95_full_latent_point_distance_px": float(
                torch.quantile(distance[index], 0.95)
            ),
            "mean_full_latent_tangent_error_deg": float(angle[index].mean()),
            "median_visible_correspondence_distance_px": float(visible.median()),
            "body_length_error_fraction": float(
                (predicted_length[index] - target_length[index]).abs()
                / target_length[index]
            ),
        })
    return results


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def group(values: list[dict[str, Any]]) -> dict[str, Any]:
        names = (
            "median_full_latent_point_distance_px",
            "p95_full_latent_point_distance_px",
            "mean_full_latent_tangent_error_deg",
            "median_visible_correspondence_distance_px",
            "body_length_error_fraction",
        )
        return {
            "n": len(values),
            **{
                name: {
                    "median": float(np.median([case[name] for case in values])),
                    "p95": float(np.percentile([case[name] for case in values], 95)),
                }
                for name in names
            },
        }

    return {
        "all_128": group(cases),
        "fully_visible_43": group([case for case in cases if case["fully_visible"]]),
        "truncated_85": group([case for case in cases if not case["fully_visible"]]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, nargs=9, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Tier-C diagnostic: {args.output}")
    config = yaml.safe_load(args.config.read_text())
    config_hash = sha256_file(args.config)
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for path in args.run_metrics:
        run = json.loads(path.read_text())
        key = (str(run["variant"]), int(run["model_seed"]))
        if (
            key in runs
            or key[0] not in VARIANTS
            or key[1] not in SEEDS
            or run.get("status") != "TRAINED_PENDING_EVALUATION"
            or run.get("config_sha256") != config_hash
        ):
            raise RuntimeError(f"invalid EXP-003 run identity: {key}")
        checkpoint = Path(run["checkpoint_path"])
        if sha256_file(checkpoint) != run["checkpoint_sha256"]:
            raise RuntimeError("EXP-003 checkpoint hash changed")
        run["run_metrics_path"] = str(path.resolve(strict=True))
        run["run_metrics_sha256"] = sha256_file(path)
        runs[key] = run
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if set(runs) != expected:
        raise RuntimeError("Tier-C diagnostic requires the exact 3x3 EXP-003 matrix")
    dataset_hashes = {
        json.dumps(run["materialized_dataset_sha256"], sort_keys=True)
        for run in runs.values()
    }
    source_hashes = {
        json.dumps(run["source_sha256"], sort_keys=True) for run in runs.values()
    }
    if len(dataset_hashes) != 1 or len(source_hashes) != 1:
        raise RuntimeError("EXP-003 runs do not share paired training provenance")

    data_seed = int(config["training"]["data_seed"])
    fold = int(config["training"]["fold"])
    dataset = MaterializedPoseDataset(SyntheticTierCDataset(
        128,
        seed=data_seed + 5_000_000 + fold * 100_000,
        profile="held_out",
    ))
    dataset_hash = materialized_dataset_sha256(dataset)
    expected_dataset_hash = next(iter(runs.values()))["materialized_dataset_sha256"][
        "tier_c_validation"
    ]
    if dataset_hash != expected_dataset_hash:
        raise RuntimeError("reconstructed Tier-C validation tensors changed")
    device = torch.device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["evaluation_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    results: dict[str, Any] = {}
    for variant in VARIANTS:
        results[variant] = {}
        for seed in SEEDS:
            run = runs[(variant, seed)]
            checkpoint = Path(run["checkpoint_path"])
            if variant == "global_intrinsic_budget_matched":
                model = WormProposalModule.load_from_checkpoint(checkpoint, map_location=device)
            else:
                model = SpatialPoseModule.load_from_checkpoint(checkpoint, map_location=device)
            model = model.to(device).eval()
            hard_cases: list[dict[str, Any]] = []
            soft_cases: list[dict[str, Any]] = []
            argmax_cases: list[dict[str, Any]] = []
            oracle_cases: list[dict[str, Any]] = []
            with torch.inference_mode():
                for batch in loader:
                    image = batch["image"].to(device, non_blocking=True)
                    target = batch["centerline_xy"].to(device, non_blocking=True)
                    support = batch["image_support_target"].to(device, non_blocking=True)
                    output = model(image)
                    hard_cases.extend(_case_metrics(output["centerline_xy"], target, support))
                    if variant == "dense_centerline_field":
                        argmax_cases.extend(_case_metrics(
                            _heatmap_argmax(output["dense_heatmap_logits"]), target, support
                        ))
                    if variant == "anchored_intrinsic_grid":
                        soft_cases.extend(
                            _case_metrics(output["soft_centerline_xy"], target, support)
                        )
                        oracle_cases.extend(_case_metrics(
                            _oracle_candidate(output["candidate_centerline_xy"], target),
                            target,
                            support,
                        ))
            results[variant][str(seed)] = {
                "checkpoint_sha256": run["checkpoint_sha256"],
                "hard_or_standard_inference": _summary(hard_cases),
                "soft_candidate_mixture_diagnostic": (
                    _summary(soft_cases) if soft_cases else None
                ),
                "hard_heatmap_argmax_diagnostic": (
                    _summary(argmax_cases) if argmax_cases else None
                ),
                "oracle_candidate_diagnostic": (
                    _summary(oracle_cases) if oracle_cases else None
                ),
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    output = {
        "schema_version": 1,
        "experiment": "EXP-003",
        "evaluation": "Tier-C-held-out-128-diagnostic",
        "role": "controlled_diagnostic_not_model_selection",
        "config_sha256": config_hash,
        "materialized_tier_c_validation_sha256": dataset_hash,
        "results": results,
        "evidence_boundary": {
            "protected_holdout_opened": False,
            "primary_Tier_A_used_for_gradients": False,
            "repeat_annotations_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "dataset_sha256": dataset_hash,
        "fully_visible_count": sum(dataset[index]["image_support_target"].all() for index in range(128)),
        "protected_holdout_opened": False,
    }, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
