#!/usr/bin/env python3
"""Evaluate candidate-proxy and held-out analytic Tier C strata separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from worm_pose_gen.geometry import in_fov_mask, tangent_angles, wrap_angle
from worm_pose_gen.model import WormProposalModule
from worm_pose_gen.training_data import ProxyDataset, SyntheticTierCDataset, sha256_file


def aligned_geometry_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support_probability: torch.Tensor,
    support_target: torch.Tensor,
    reported_in_fov: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Align orientation and measure geometry in original 968x732 pixels."""

    scale = target.new_tensor((968 / 256, 732 / 192))
    prediction_original = prediction * scale
    target_original = target * scale
    forward = torch.linalg.vector_norm(prediction_original - target_original, dim=-1)
    reverse = torch.linalg.vector_norm(prediction_original - target_original.flip(-2), dim=-1)
    choose_reverse = reverse.mean(-1) < forward.mean(-1)
    chosen_target = torch.where(choose_reverse[:, None, None], target.flip(-2), target)
    chosen_original = chosen_target * scale
    aligned_support = torch.where(
        choose_reverse[:, None], support_target.flip(-1), support_target
    ).bool()
    point = torch.linalg.vector_norm(prediction_original - chosen_original, dim=-1)
    angle = wrap_angle(
        tangent_angles(prediction_original) - tangent_angles(chosen_original)
    ).abs() * 180 / torch.pi
    proximity = torch.stack(
        (
            chosen_original[..., 0],
            968 - chosen_original[..., 0],
            chosen_original[..., 1],
            732 - chosen_original[..., 1],
        ),
        -1,
    ).amin(-1)
    predicted_length = torch.linalg.vector_norm(
        prediction_original[:, 1:] - prediction_original[:, :-1], dim=-1
    ).sum(-1)
    target_length = torch.linalg.vector_norm(
        chosen_original[:, 1:] - chosen_original[:, :-1], dim=-1
    ).sum(-1)
    recomputed_fov = in_fov_mask(prediction, 192, 256)
    return {
        "chosen_target": chosen_target,
        "point": point,
        "angle": angle,
        "proximity": proximity,
        "endpoint": point[:, [0, -1]],
        "body_length_error": (predicted_length - target_length).abs(),
        "body_length_error_fraction": (
            (predicted_length - target_length).abs() / target_length.clamp_min(1e-12)
        ),
        "aligned_support": aligned_support,
        "support_probability": support_probability,
        "support_squared_error": (
            support_probability - aligned_support.to(support_probability.dtype)
        ).square(),
        "reported_fov_agreement": reported_in_fov == recomputed_fov,
        "predicted_geometric_fov": recomputed_fov,
    }


def expected_calibration_error(probability: torch.Tensor, target: torch.Tensor) -> float:
    """Ten fixed-width bins, including probability exactly one in the last bin."""

    result = probability.new_zeros(())
    bin_index = torch.clamp((probability * 10).long(), min=0, max=9)
    for index in range(10):
        mask = bin_index == index
        if bool(mask.any()):
            result += mask.float().mean() * (
                probability[mask].mean() - target[mask].float().mean()
            ).abs()
    return float(result)


def summarize_cases(cases: list[dict], *, failed_inference_count: int = 0) -> dict:
    """Summarize one homogeneous evidence stratum without mixing crop regimes."""

    if not cases:
        raise ValueError("cannot summarize an empty evaluation stratum")
    all_point = torch.cat([case["point"] for case in cases])
    all_angle = torch.cat([case["angle"] for case in cases])
    all_probability = torch.cat([case["support_probability"] for case in cases])
    all_support = torch.cat([case["aligned_support"] for case in cases])
    visible_point = torch.cat([case["point"][case["aligned_support"]] for case in cases])
    hidden_point = torch.cat([case["point"][~case["aligned_support"]] for case in cases])
    visible_angle = torch.cat([case["angle"][case["aligned_support"]] for case in cases])
    hidden_angle = torch.cat([case["angle"][~case["aligned_support"]] for case in cases])
    return {
        "samples": len(cases), "median_point_px": float(all_point.median()),
        "p95_point_px": float(torch.quantile(all_point, .95)),
        "mean_angle_degrees": float(all_angle.mean()),
        "p95_frame_angle_degrees": float(torch.quantile(torch.tensor([float(case["angle"].mean()) for case in cases]), .95)),
        "mean_endpoint_error_px": float(torch.cat([case["endpoint"] for case in cases]).mean()),
        "mean_head_endpoint_error_px": float(
            torch.stack([case["endpoint"][0] for case in cases]).mean()
        ),
        "mean_tail_endpoint_error_px": float(
            torch.stack([case["endpoint"][1] for case in cases]).mean()
        ),
        "mean_endpoint_error_px_each": [
            float(torch.stack([case["endpoint"][0] for case in cases]).mean()),
            float(torch.stack([case["endpoint"][1] for case in cases]).mean()),
        ],
        "mean_body_length_error_px": float(torch.stack([case["body_length_error"] for case in cases]).mean()),
        "median_body_length_error_fraction": float(
            torch.stack([case["body_length_error_fraction"] for case in cases]).median()
        ),
        "visible_mean_point_px": float(visible_point.mean()) if len(visible_point) else None,
        "hidden_mean_point_px": float(hidden_point.mean()) if len(hidden_point) else None,
        "visible_mean_angle_degrees": float(visible_angle.mean()) if len(visible_angle) else None,
        "hidden_mean_angle_degrees": float(hidden_angle.mean()) if len(hidden_angle) else None,
        "support_brier": float(torch.cat([case["support_squared_error"] for case in cases]).mean()),
        "support_ece_10_bin": expected_calibration_error(all_probability, all_support),
        "reported_vs_recomputed_in_fov_exact_agreement": bool(
            torch.cat([case["reported_fov_agreement"] for case in cases]).all()
        ),
        "reported_vs_recomputed_in_fov_note": "contract consistency only; not FOV accuracy",
        "predicted_geometric_fov_accuracy": float(
            torch.cat(
                [case["predicted_geometric_fov"] == case["aligned_support"] for case in cases]
            ).float().mean()
        ),
        "failed_inference_count": failed_inference_count,
    }


def batch_case_id(batch: dict, index: int) -> str:
    return (
        f"seed:{int(batch['sample_seed'][index])}"
        if int(batch["sample_seed"][index]) >= 0
        else f"{batch['record'][index]}:frame:{int(batch['frame_index'][index])}"
    )


@torch.inference_mode()
def evaluate(module: WormProposalModule, dataset: torch.utils.data.Dataset, batch_size: int, device: torch.device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    cases = []
    failed_inference_cases: list[dict[str, object]] = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["centerline_xy"].to(device)
        try:
            output = module(image)
        except Exception:
            failed_inference_cases.extend(
                {
                    "case_id": batch_case_id(batch, index),
                    "fully_visible": bool(
                        batch["image_support_target"][index].bool().all()
                    ),
                    "reason": "inference_exception",
                }
                for index in range(len(image))
            )
            continue
        prediction = output["centerline_xy"]
        finite = torch.isfinite(prediction).all((-2, -1)) & torch.isfinite(
            output["image_support_probability"]
        ).all(-1)
        failed_inference_cases.extend(
            {
                "case_id": batch_case_id(batch, index),
                "fully_visible": bool(
                    batch["image_support_target"][index].bool().all()
                ),
                "reason": "non_finite_output",
            }
            for index in torch.nonzero(~finite, as_tuple=False).flatten().tolist()
        )
        if not bool(finite.any()):
            continue
        target_support = batch["image_support_target"].to(device)[finite]
        metrics = aligned_geometry_metrics(
            prediction[finite],
            target[finite],
            output["image_support_probability"][finite],
            target_support,
            output["in_fov_mask"][finite],
        )
        valid_indices = torch.nonzero(finite, as_tuple=False).flatten().tolist()
        for metric_index, index in enumerate(valid_indices):
            case_id = batch_case_id(batch, index)
            cases.append({
                "case_id": case_id,
                "image": image[index, 0].cpu(), "prediction": prediction[index].cpu(),
                "target": metrics["chosen_target"][metric_index].cpu(),
                **{name: value[metric_index].cpu() for name, value in metrics.items() if name != "chosen_target"},
            })
    if not cases:
        raise RuntimeError("evaluation produced no finite inference cases")
    if len(cases) + len(failed_inference_cases) != len(dataset):
        raise RuntimeError("evaluation accounting does not match requested dataset size")
    summary = summarize_cases(cases, failed_inference_count=len(failed_inference_cases))
    summary.update(
        {
            "requested_samples": len(dataset),
            "evaluated_samples": len(cases),
            "failed_inference_cases": failed_inference_cases,
        }
    )
    return cases, summary


def overlays(cases, indices, path: Path, title: str) -> None:
    columns = 4
    rows = int(np.ceil(len(indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
    for ax in axes.flat: ax.axis("off")
    for ax, index in zip(axes.flat, indices, strict=False):
        case = cases[index]
        ax.imshow(case["image"], cmap="gray", vmin=0, vmax=1)
        ax.plot(case["target"][:, 0], case["target"][:, 1], color="#36f1cd", lw=1)
        ax.plot(case["prediction"][:, 0], case["prediction"][:, 1], color="#ff3b6b", lw=1)
        ax.set_title(f"mean={case['point'].mean():.1f}px", fontsize=8)
    figure.suptitle(title); figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)


def diagnostics(results, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for tier, (cases, _) in results.items():
        point = torch.stack([case["point"] for case in cases])
        angle = torch.stack([case["angle"] for case in cases])
        proximity = torch.stack([case["proximity"] for case in cases])
        label = "candidate proxy" if tier == "B" else "Tier C analytic"
        axes[0].plot(torch.linspace(0, 1, 100), angle.mean(0), label=label)
        axes[1].hist(point.flatten().numpy(), bins=40, alpha=.45, label=label)
        axes[2].scatter(proximity.flatten().numpy(), point.flatten().numpy(), s=2, alpha=.12, label=label)
    axes[0].set(xlabel="body position", ylabel="angle error (degrees)")
    axes[1].set(xlabel="point error (original-image px)", ylabel="count")
    axes[2].set(xlabel="signed FOV-edge proximity (original-image px)", ylabel="point error (original-image px)")
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    figure.tight_layout(); figure.savefig(output_dir / "diagnostics.png", dpi=160); plt.close(figure)


def aggregate_results(
    paths: list[Path],
    config: dict,
    *,
    coordinate_throughput: float | None = None,
    intrinsic_throughput: float | None = None,
) -> dict:
    """Apply the predeclared all-fold rules and paired frame bootstrap."""

    documents = [json.loads(path.read_text()) for path in paths]
    indexed = {(item["variant"], int(item["fold"])): item for item in documents}
    expected = {(variant, fold) for variant in ("coordinate", "intrinsic") for fold in (0, 1, 2)}
    if set(indexed) != expected:
        raise ValueError(f"aggregation requires exactly six primary variant/fold files; got {sorted(indexed)}")
    decision = config["decision"]
    fold_results = []
    paired_by_fold = []
    for fold in (0, 1, 2):
        coordinate = indexed[("coordinate", fold)]
        intrinsic = indexed[("intrinsic", fold)]
        tier_c_coord = {case["case_id"]: case for case in coordinate["tier_C"]["cases"]}
        tier_c_intr = {case["case_id"]: case for case in intrinsic["tier_C"]["cases"]}
        if tier_c_coord.keys() != tier_c_intr.keys():
            raise ValueError(f"fold {fold} Tier C cases are not identically paired")
        pairs = np.asarray([
            (tier_c_coord[key]["mean_angle_degrees"], tier_c_intr[key]["mean_angle_degrees"])
            for key in sorted(tier_c_coord)
        ])
        paired_by_fold.append(pairs)
        angle_improvement = 1 - float(pairs[:, 1].mean() / pairs[:, 0].mean())
        if any(
            document["tier_B_candidate_proxy"].get("point_error_units")
            != "original_image_pixels_968x732"
            for document in (coordinate, intrinsic)
        ):
            raise ValueError(f"fold {fold} candidate-proxy guardrail is not in original pixels")
        proxy_coord = coordinate["tier_B_candidate_proxy"]["median_point_px"]
        proxy_intr = intrinsic["tier_B_candidate_proxy"]["median_point_px"]
        proxy_regression = float(proxy_intr / proxy_coord - 1)
        fold_results.append({"fold": fold, "angle_improvement_fraction": angle_improvement,
                             "candidate_proxy_point_regression_fraction": proxy_regression})
    pooled = np.concatenate(paired_by_fold)
    pooled_improvement = 1 - float(pooled[:, 1].mean() / pooled[:, 0].mean())
    rng = np.random.default_rng(int(config["seed"]))
    bootstrap = []
    for _ in range(int(decision["bootstrap_resamples"])):
        sampled = np.concatenate([pairs[rng.integers(0, len(pairs), len(pairs))] for pairs in paired_by_fold])
        bootstrap.append(1 - float(sampled[:, 1].mean() / sampled[:, 0].mean()))
    bootstrap_array = np.asarray(bootstrap)
    threshold = float(decision["minimum_angle_improvement_fraction"])
    bootstrap_changes_decision = bool(np.any(bootstrap_array < threshold) and np.any(bootstrap_array >= threshold))
    all_fold_angle = all(item["angle_improvement_fraction"] > 0 for item in fold_results)
    all_fold_proxy = all(item["candidate_proxy_point_regression_fraction"] <= float(decision["maximum_proxy_point_regression_fraction"]) for item in fold_results)
    throughput_ratio = None if coordinate_throughput is None or intrinsic_throughput is None else intrinsic_throughput / coordinate_throughput
    throughput_pass = throughput_ratio is not None and throughput_ratio >= float(decision["minimum_throughput_ratio"])
    reliability_by_variant: dict[str, dict] = {}
    reliability_gate_values = []
    for variant in ("coordinate", "intrinsic"):
        fold_gates = []
        for fold in (0, 1, 2):
            values = indexed[(variant, fold)]["tier_C"]
            if values.get("point_error_units") != "original_image_pixels_968x732":
                raise ValueError(f"{variant} fold {fold} does not declare original-pixel metrics")
            checks = {
                "median_point_px": {
                    "value": values["median_point_px"],
                    "maximum": decision["reliability_tier_c_median_point_px"],
                },
                "p95_point_px": {
                    "value": values["p95_point_px"],
                    "maximum": decision["reliability_tier_c_p95_point_px"],
                },
                "mean_angle_degrees": {
                    "value": values["mean_angle_degrees"],
                    "maximum": decision["reliability_tier_c_mean_angle_degrees"],
                },
                "p95_frame_angle_degrees": {
                    "value": values["p95_frame_angle_degrees"],
                    "maximum": decision["reliability_tier_c_p95_frame_angle_degrees"],
                },
            }
            numeric_pass = all(item["value"] <= item["maximum"] for item in checks.values())
            contract_pass = bool(values.get("reported_vs_recomputed_in_fov_exact_agreement", False))
            inference_pass = int(values.get("failed_inference_count", 1)) == 0
            required_diagnostics = {
                "mean_endpoint_error_px",
                "mean_body_length_error_px",
                "predicted_geometric_fov_accuracy",
                "support_brier",
                "support_ece_10_bin",
                "visible_mean_point_px",
                "hidden_mean_point_px",
                "visible_mean_angle_degrees",
                "hidden_mean_angle_degrees",
            }
            diagnostics_complete = required_diagnostics.issubset(values)
            fold_pass = numeric_pass and contract_pass and inference_pass and diagnostics_complete
            fold_gates.append(
                {
                    "fold": fold,
                    "metrics": checks,
                    "reported_vs_recomputed_in_fov_exact_agreement": contract_pass,
                    "predicted_geometric_fov_accuracy_diagnostic": values.get("predicted_geometric_fov_accuracy"),
                    "mean_endpoint_error_px_diagnostic": values.get("mean_endpoint_error_px"),
                    "mean_body_length_error_px_diagnostic": values.get("mean_body_length_error_px"),
                    "failed_inference_count": values.get("failed_inference_count"),
                    "required_diagnostics_reported": diagnostics_complete,
                    "pass": fold_pass,
                }
            )
            reliability_gate_values.extend(
                (item["value"], float(item["maximum"])) for item in checks.values()
            )
        reliability_by_variant[variant] = {
            "folds": fold_gates,
            "all_folds_pass": all(item["pass"] for item in fold_gates),
        }
    intrinsic_accepted = (
        pooled_improvement >= threshold
        and all_fold_angle
        and all_fold_proxy
        and throughput_pass
        and reliability_by_variant["intrinsic"]["all_folds_pass"]
        and not bootstrap_changes_decision
    )
    near_fraction = float(decision["near_gate_relative_band_fraction"])
    gate_values = [(pooled_improvement, threshold)]
    for fold in fold_results:
        gate_values.extend([(fold["angle_improvement_fraction"], 0.0),
                            (fold["candidate_proxy_point_regression_fraction"], float(decision["maximum_proxy_point_regression_fraction"]))])
    gate_values.extend(reliability_gate_values)
    if throughput_ratio is not None:
        gate_values.append((throughput_ratio, float(decision["minimum_throughput_ratio"])))
    repeat_required = any(
        abs(value - gate) <= near_fraction * max(abs(gate), threshold)
        for value, gate in gate_values
    )
    if intrinsic_accepted:
        final_decision = "ACCEPT_INTRINSIC"
    elif reliability_by_variant["coordinate"]["all_folds_pass"]:
        final_decision = "RETAIN_COORDINATE"
    else:
        final_decision = "REVISE_NO_RELIABLE_PROPOSAL"
    return {
        "folds": fold_results,
        "pooled_angle_improvement_fraction": pooled_improvement,
        "paired_within_recording_bootstrap": {
            "resamples": len(bootstrap),
            "ci95": np.quantile(bootstrap_array, [.025, .975]).tolist(),
            "changes_5_percent_decision": bootstrap_changes_decision,
        },
        "throughput_ratio": throughput_ratio,
        "reliability_gates_by_variant": reliability_by_variant,
        "repeat_required": repeat_required,
        "repeat_seeds": decision["near_gate_repeat_seeds"] if repeat_required else [],
        "decision": final_decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--baseline-metadata", type=Path)
    parser.add_argument("--aggregate", type=Path, nargs="+")
    parser.add_argument("--coordinate-throughput", type=float)
    parser.add_argument("--intrinsic-throughput", type=float)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("experiment") not in ("EXP-0004", "EXP-0007"):
        raise ValueError("evaluate.py supports only EXP-0004 and EXP-0007")
    if config["input"].get("audited_holdout_allowed", False): raise RuntimeError("audited holdout forbidden")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        if config["experiment"] != "EXP-0004":
            raise ValueError("the two-variant aggregate mode applies only to EXP-0004")
        aggregate = aggregate_results(
            args.aggregate,
            config,
            coordinate_throughput=args.coordinate_throughput,
            intrinsic_throughput=args.intrinsic_throughput,
        )
        (args.output_dir / "aggregate_decision.json").write_text(json.dumps(aggregate, indent=2) + "\n")
        print(json.dumps(aggregate, indent=2))
        return 0
    if args.checkpoint is None or args.fold is None:
        parser.error("evaluation requires --checkpoint and --fold unless --aggregate is used")
    baseline_identity = None
    baseline = None
    if config["experiment"] == "EXP-0007":
        if args.baseline_metadata is None:
            parser.error("EXP-0007 evaluation requires --baseline-metadata")
        baseline = json.loads(args.baseline_metadata.read_text())
        if (
            baseline.get("experiment") != "EXP-0007"
            or int(baseline.get("fold", -1)) != args.fold
            or int(baseline.get("data_seed", -1)) != int(config["data_seed"])
            or baseline.get("config_sha256") != sha256_file(args.config)
        ):
            raise RuntimeError("baseline metadata does not match evaluation identity")
        tensor_path = args.baseline_metadata.parent / baseline["tensor_file"]
        if sha256_file(tensor_path) != baseline["tensor_file_sha256"]:
            raise RuntimeError("baseline tensor-file hash mismatch")
        repository_root = Path(__file__).resolve().parents[1]
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        provenance = baseline.get("code_provenance", {})
        if (
            git_status
            or not provenance.get("source_sha256")
            or provenance.get("git_commit") != git_commit
            or provenance.get("git_tree") != git_tree
            or any(
                sha256_file(repository_root / name) != digest
                for name, digest in provenance.get("source_sha256", {}).items()
            )
        ):
            raise RuntimeError("baseline implementation provenance mismatch")
        baseline_identity = {
            "metadata_path": str(args.baseline_metadata.resolve(strict=True)),
            "metadata_sha256": sha256_file(args.baseline_metadata),
            "tensor_sha256": baseline["tensor_sha256"],
            "tensor_file_sha256": baseline["tensor_file_sha256"],
            "median_point_error_px": baseline["validation"]["median_point_error_px"],
            "code_provenance": provenance,
        }
    elif args.baseline_metadata is not None:
        parser.error("--baseline-metadata applies only to EXP-0007")
    device = torch.device(args.device)
    module = WormProposalModule.load_from_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    expected_variant = (
        config["model"]["variant"]
        if config["experiment"] == "EXP-0007"
        else module.variant
    )
    expected_pool = tuple(config["model"].get("encoder_pool_output", (2, 2)))
    if module.variant != expected_variant or tuple(module.encoder.pool_output) != expected_pool:
        raise RuntimeError("checkpoint variant/pool does not match the evaluation config")
    configured_model_seed = int(config.get("model_seed", config.get("seed", 20260818)))
    data_seed = int(config.get("data_seed", config.get("seed", 20260818)))
    evaluated_model_seed = args.model_seed if args.model_seed is not None else configured_model_seed
    if config["experiment"] == "EXP-0007" and (
        int(module.hparams.get("model_seed", -1)) != evaluated_model_seed
        or int(module.hparams.get("data_seed", -1)) != data_seed
    ):
        raise RuntimeError("checkpoint model/data seed does not match evaluation identity")
    datasets = {
        "B": ProxyDataset(config["input"]["proxy_hdf5"], expected_sha256=config["input"]["proxy_sha256"], fold=args.fold, split="validation"),
        "C": SyntheticTierCDataset(int(config["training"]["synthetic_validation_samples"]), seed=data_seed + 5_000_000 + args.fold * 100_000, profile=config["input"]["synthetic_validation_profile"]),
    }
    results = {tier: evaluate(module, dataset, int(config["training"]["evaluation_batch_size"]), device) for tier, dataset in datasets.items()}
    tier_c_cases, tier_c_all_summary = results["C"]
    tier_c_fully_visible = [
        case for case in tier_c_cases if bool(case["aligned_support"].all())
    ]
    tier_c_cropped = [
        case for case in tier_c_cases if not bool(case["aligned_support"].all())
    ]
    tier_c_failures = tier_c_all_summary["failed_inference_cases"]
    fully_visible_failures = [
        case for case in tier_c_failures if bool(case["fully_visible"])
    ]
    cropped_failures = [
        case for case in tier_c_failures if not bool(case["fully_visible"])
    ]
    if not tier_c_fully_visible or not tier_c_cropped:
        raise RuntimeError("Tier C evaluation must contain fully-visible and cropped strata")
    if (
        len(tier_c_fully_visible)
        + len(tier_c_cropped)
        + len(tier_c_failures)
        != len(datasets["C"])
    ):
        raise RuntimeError("Tier C visibility strata do not partition the cases")
    full_summary = summarize_cases(
        tier_c_fully_visible, failed_inference_count=len(fully_visible_failures)
    )
    full_summary.update(
        {
            "requested_samples": len(tier_c_fully_visible)
            + len(fully_visible_failures),
            "evaluated_samples": len(tier_c_fully_visible),
            "failed_inference_cases": fully_visible_failures,
        }
    )
    if config["experiment"] == "EXP-0007":
        expected_ids = baseline["validation"]["fully_visible_case_ids"]
        observed_ids = sorted(
            [case["case_id"] for case in tier_c_fully_visible]
            + [case["case_id"] for case in fully_visible_failures],
            key=lambda value: int(value.split(":", 1)[1]),
        )
        if observed_ids != expected_ids:
            raise RuntimeError("Tier C fully-visible identities differ from frozen baseline")
    results["C"] = (tier_c_fully_visible, full_summary)
    rng = np.random.default_rng(data_seed + args.fold)
    for tier, (cases, _) in results.items():
        count = min(12, len(cases))
        random_indices = sorted(rng.choice(len(cases), count, replace=False).tolist())
        worst = sorted(range(len(cases)), key=lambda i: float(cases[i]["point"].mean()), reverse=True)[:count]
        stratum = "candidate proxy" if tier == "B" else "Tier C analytic"
        overlays(cases, random_indices, args.output_dir / f"tier_{tier}_random.png", f"{stratum} random")
        overlays(cases, worst, args.output_dir / f"tier_{tier}_worst.png", f"{stratum} worst")
    diagnostics(results, args.output_dir)
    metrics = {
        "experiment": config["experiment"],
        "variant": module.variant,
        "encoder_pool_output": list(module.encoder.pool_output),
        "model_seed": evaluated_model_seed,
        "data_seed": data_seed,
        "config_sha256": sha256_file(args.config),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "baseline_identity": baseline_identity,
        "advancement_scope": (
            "geometry_only_rescue_no_temporal_authorization"
            if config["experiment"] == "EXP-0007"
            else "representation_ablation"
        ),
        "fold": args.fold,
        "checkpoint_path": str(args.checkpoint.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_global_step": int(
            torch.load(args.checkpoint, map_location="cpu", weights_only=False).get(
                "global_step", -1
            )
        ),
        "evidence": {
            "tier_B_candidate_proxy": "candidate-proxy engineering evidence; not manual-label truth",
            "tier_C": "controlled analytic synthetic evidence",
            "audited_holdout_opened": False,
        },
    }
    for tier, (cases, values) in results.items():
        namespace = "tier_B_candidate_proxy" if tier == "B" else "tier_C"
        metrics[namespace] = {
            **values,
            "point_error_units": "original_image_pixels_968x732",
            "cases": [
                {
                    "case_id": case["case_id"],
                    "mean_point_px": float(case["point"].mean()),
                    "median_point_px": float(case["point"].median()),
                    "p95_point_px": float(torch.quantile(case["point"], .95)),
                    "mean_endpoint_error_px": float(case["endpoint"].mean()),
                    "head_endpoint_error_px": float(case["endpoint"][0]),
                    "tail_endpoint_error_px": float(case["endpoint"][1]),
                    "body_length_error_px": float(case["body_length_error"]),
                    "body_length_error_fraction": float(
                        case["body_length_error_fraction"]
                    ),
                    "mean_angle_degrees": float(case["angle"].mean()),
                    "p95_angle_degrees": float(torch.quantile(case["angle"], .95)),
                }
                for case in cases
            ],
        }
    metrics["tier_C_cropped"] = {
        **summarize_cases(
            tier_c_cropped, failed_inference_count=len(cropped_failures)
        ),
        "requested_samples": len(tier_c_cropped) + len(cropped_failures),
        "evaluated_samples": len(tier_c_cropped),
        "failed_inference_cases": cropped_failures,
        "point_error_units": "original_image_pixels_968x732",
        "ordinary_reliability_gate_applicable": False,
        "case_partition_note": (
            "artificially truncated Tier C cases; evaluated separately from the "
            "frozen fully-visible ordinary-frame gate"
        ),
        "cases": [
            {
                "case_id": case["case_id"],
                "mean_point_px": float(case["point"].mean()),
                "median_point_px": float(case["point"].median()),
                "p95_point_px": float(torch.quantile(case["point"], .95)),
                "mean_endpoint_error_px": float(case["endpoint"].mean()),
                "head_endpoint_error_px": float(case["endpoint"][0]),
                "tail_endpoint_error_px": float(case["endpoint"][1]),
                "body_length_error_px": float(case["body_length_error"]),
                "body_length_error_fraction": float(
                    case["body_length_error_fraction"]
                ),
                "mean_angle_degrees": float(case["angle"].mean()),
                "p95_angle_degrees": float(torch.quantile(case["angle"], .95)),
            }
            for case in tier_c_cropped
        ],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
