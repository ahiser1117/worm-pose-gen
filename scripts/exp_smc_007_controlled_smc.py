#!/usr/bin/env python3
"""Bounded EXP-SMC-007 controlled known-pose particle-filter recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_tier_a_primary import _verified_inputs
from exp_005_representation_oracle import _canonical_complete_annotations
from worm_pose_gen.latent import decode_centerline_torch, encode_centerline
from worm_pose_gen.observation import soft_dice_energy
from worm_pose_gen.renderer import render_worm
from worm_pose_gen.smc import (
    effective_sample_size,
    normalize_log_weights,
    systematic_resample,
    trace_genealogy,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _summary(values: list[float]) -> dict[str, float | int]:
    sample = np.asarray(values, dtype=np.float64)
    return {
        "n": len(sample), "median": float(np.median(sample)),
        "mean": float(sample.mean()), "p95": float(np.percentile(sample, 95)),
    }


def _load_start_anchors(config: dict[str, Any]) -> list[dict[str, Any]]:
    strict = json.loads(_resolved(config["inputs"]["strict_anchor_metrics"]).read_text())
    accepted = {case["sample_id"]: case for case in strict["per_case"] if case["accepted"]}
    _, rows = _verified_inputs(
        _resolved(config["inputs"]["manifest"]), Path(config["inputs"]["annotations"])
    )
    identities, targets = _canonical_complete_annotations(rows)
    target_by_id = dict(zip((item["sample_id"] for item in identities), targets, strict=True))
    starts = []
    for sample_id, case in accepted.items():
        if sample_id not in target_by_id or case["width_profile_px"] is None:
            continue
        starts.append({
            "sample_id": sample_id,
            "recording": case["recording"],
            "latent": torch.tensor(encode_centerline(target_by_id[sample_id]), dtype=torch.float32),
            "width": torch.tensor(case["width_profile_px"], dtype=torch.float32),
            "source_role": "complete single-annotator trace associated with strict accepted anchor; strict mask width profile",
        })
    if len(starts) != 3:
        raise RuntimeError(f"expected three strict complete anchor starts, got {len(starts)}")
    return starts


def _process_std(config: dict[str, Any], device: torch.device) -> Tensor:
    source = config["state"]["process_std_per_frame_original_units"]
    return torch.tensor(
        [source["shape_coefficients_rad"]] * 16
        + [source["global_rotation_rad"], source["body_length_px"]]
        + [source["translation_xy_px"]] * 2,
        dtype=torch.float32, device=device,
    )


def _truth_sequence(
    start: Tensor, frames: int, std: Tensor, seed: int, process_scale: float,
    *, partial_fov: bool,
) -> Tensor:
    generator = torch.Generator(device=start.device).manual_seed(seed)
    initial = start.clone()
    if partial_fov:
        curve = decode_centerline_torch(initial)
        initial[-2] += -20.0 - curve[:, 0].min()
    states = [initial]
    for _ in range(1, frames):
        innovation = torch.randn(
            initial.shape, generator=generator, device=start.device
        ) * std * process_scale
        next_state = states[-1] + innovation
        next_state[-3] = next_state[-3].clamp_min(100.0)
        states.append(next_state)
    return torch.stack(states)


def _render_masks(latents: Tensor, width: Tensor, config: dict[str, Any]) -> Tensor:
    render = config["observation"]
    curves = decode_centerline_torch(latents) / render["downsample_factor"]
    result = render_worm(
        curves, width / render["downsample_factor"],
        render["height"], render["width"],
        edge_softness=render["edge_softness_px"],
    )
    return result["tube_mask"]


def _particle_energies(
    particles: Tensor, width: Tensor, target: Tensor, config: dict[str, Any]
) -> Tensor:
    chunk = config["observation"]["render_particle_chunk"]
    values = []
    for start in range(0, len(particles), chunk):
        rendered = _render_masks(particles[start : start + chunk], width, config)
        values.append(soft_dice_energy(rendered, target))
    return torch.cat(values)


def _curve_errors(path: Tensor, truth: Tensor) -> Tensor:
    prediction = decode_centerline_torch(path)
    target = decode_centerline_torch(truth)
    return torch.linalg.vector_norm(prediction - target, dim=-1).mean(-1)


def _run_case(
    start: dict[str, Any], scenario: str, seed: int, particles_count: int,
    temperature: float, config: dict[str, Any], *, keep_trajectories: bool,
) -> tuple[dict[str, Any], dict[str, Tensor] | None]:
    began = time.perf_counter()
    device = start["latent"].device
    frames = config["state"]["frames_per_sequence"]
    std = _process_std(config, device)
    process_scale = {"process_scale_0.5": 0.5, "process_scale_2.0": 2.0}.get(scenario, 1.0)
    truth = _truth_sequence(
        start["latent"], frames, std, seed * 17 + 3, process_scale,
        partial_fov=scenario == "partial_fov",
    )
    observation_width = start["width"] * (1.15 if scenario == "width_mismatch_1.15" else 1.0)
    targets = _render_masks(truth, observation_width, config)
    dropout = frames // 2 if scenario == "middle_observation_dropout" else None

    particles = truth[0].expand(particles_count, -1).clone()
    history = [particles.clone()]
    genealogy = []
    ess_values: list[float] = []
    survival: list[bool] = [True]
    minimum_errors: list[float] = [0.0]
    generator = torch.Generator(device=device).manual_seed(seed * 1009 + particles_count)
    final_log_weights = torch.full((particles_count,), -math.log(particles_count), device=device)
    for frame in range(1, frames):
        particles = particles + torch.randn(
            particles.shape, generator=generator, device=device
        ) * std
        particles[:, -3].clamp_(min=100.0)
        if frame == dropout:
            log_weights = torch.full_like(final_log_weights, -math.log(particles_count))
        else:
            energy = _particle_energies(particles, start["width"], targets[frame], config)
            log_weights = normalize_log_weights(-energy / temperature)
        ess_values.append(float(effective_sample_size(log_weights)))
        errors = _curve_errors(particles, truth[frame])
        minimum = float(errors.min())
        minimum_errors.append(minimum)
        survival.append(minimum <= 8.0)
        if frame < frames - 1:
            ancestors = systematic_resample(log_weights.exp(), generator=generator)
            particles = particles[ancestors]
            genealogy.append(ancestors)
            history.append(particles.clone())
        else:
            genealogy.append(torch.arange(particles_count, device=device))
            history.append(particles.clone())
            final_log_weights = log_weights
    history_tensor = torch.stack(history)
    genealogy_tensor = torch.stack(genealogy)
    forward_terminal = int(torch.argmax(final_log_weights))
    forward_indices = trace_genealogy(genealogy_tensor, forward_terminal)
    forward_path = history_tensor[torch.arange(frames, device=device), forward_indices]

    right_scale = (std * math.sqrt(frames - 1)).clamp_min(1e-3)
    terminal_compatibility = -0.5 * ((history_tensor[-1] - truth[-1]) / right_scale).square().sum(-1)
    right_log_weights = normalize_log_weights(final_log_weights + terminal_compatibility)
    right_terminal = int(torch.argmax(right_log_weights))
    right_indices = trace_genealogy(genealogy_tensor, right_terminal)
    right_path = history_tensor[torch.arange(frames, device=device), right_indices]

    hold_path = truth[0].expand_as(truth)
    alpha = torch.linspace(0, 1, frames, device=device)[:, None]
    interpolation = truth[0] + alpha * (truth[-1] - truth[0])
    errors = {
        "hold": _curve_errors(hold_path, truth),
        "two_anchor_latent_interpolation": _curve_errors(interpolation, truth),
        "forward_bootstrap_smc_map_genealogy": _curve_errors(forward_path, truth),
        "terminal_right_anchor_reweighted_genealogy": _curve_errors(right_path, truth),
    }
    roots = trace_genealogy(
        genealogy_tensor, torch.arange(particles_count, device=device)
    )[0]
    result: dict[str, Any] = {
        "scenario": scenario,
        "seed": seed,
        "start_sample_id": start["sample_id"],
        "particle_count": particles_count,
        "temperature": temperature,
        "method_errors_px": {
            name: {
                "per_frame": value.tolist(),
                "trajectory_mean": float(value.mean()),
                "final": float(value[-1]),
            } for name, value in errors.items()
        },
        "ess": {
            "per_observation": ess_values,
            "median": float(np.median(ess_values)),
            "median_fraction": float(np.median(ess_values) / particles_count),
            "minimum": float(min(ess_values)),
        },
        "resampling_events": frames - 2,
        "truth_survival": {
            "threshold_px": 8.0,
            "per_frame": survival,
            "fraction": float(np.mean(survival)),
            "minimum_particle_error_px": minimum_errors,
        },
        "diversity": {
            "unique_root_ancestors": int(torch.unique(roots).numel()),
            "final_shape_coefficient_std_mean": float(history_tensor[-1, :, :16].std(0).mean()),
        },
        "runtime_seconds": time.perf_counter() - began,
    }
    trajectories = None
    if keep_trajectories:
        trajectories = {
            "truth_latent": truth.cpu(), "hold_latent": hold_path.cpu(),
            "interpolation_latent": interpolation.cpu(),
            "forward_latent": forward_path.cpu(), "right_reweighted_latent": right_path.cpu(),
            "genealogy": genealogy_tensor.cpu(),
        }
    return result, trajectories


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    methods = next(iter(cases))["method_errors_px"]
    return {
        "cases": len(cases),
        "methods": {
            method: {
                "trajectory_mean_px": _summary([
                    case["method_errors_px"][method]["trajectory_mean"] for case in cases
                ]),
                "final_px": _summary([
                    case["method_errors_px"][method]["final"] for case in cases
                ]),
            } for method in methods
        },
        "truth_survival_fraction": _summary([
            case["truth_survival"]["fraction"] for case in cases
        ]),
        "ess_median_fraction": _summary([case["ess"]["median_fraction"] for case in cases]),
        "runtime_seconds": float(sum(case["runtime_seconds"] for case in cases)),
    }


def _plot_calibration(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for temperature in sorted({row["temperature"] for row in rows}):
        selected = [row for row in rows if row["temperature"] == temperature]
        axis.plot(
            [row["particle_count"] for row in selected],
            [row["method_errors_px"]["forward_bootstrap_smc_map_genealogy"]["trajectory_mean"] for row in selected],
            marker="o", label=f"temperature={temperature:g}",
        )
    axis.set(xscale="log", xlabel="particles", ylabel="forward genealogy mean error (px)")
    axis.grid(alpha=0.25)
    axis.legend()
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("EXP-SMC-007 nominal calibration")
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _plot_evaluation(rows: list[dict[str, Any]], path: Path) -> None:
    methods = [
        "hold", "two_anchor_latent_interpolation",
        "forward_bootstrap_smc_map_genealogy",
        "terminal_right_anchor_reweighted_genealogy",
    ]
    labels = ["hold", "two-anchor interp.", "forward SMC", "right-anchor reweighted"]
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    values = np.arange(len(scenarios)); width = 0.19
    figure, axis = plt.subplots(figsize=(11.2, 4.8))
    for index, (method, label) in enumerate(zip(methods, labels, strict=True)):
        medians = [np.median([
            row["method_errors_px"][method]["trajectory_mean"]
            for row in rows if row["scenario"] == scenario
        ]) for scenario in scenarios]
        axis.bar(values + (index - 1.5) * width, medians, width, label=label)
    axis.set_xticks(values, [value.replace("_", "\n") for value in scenarios])
    axis.set_ylabel("median trajectory mean error (px)")
    axis.grid(axis="y", alpha=0.25); axis.legend(fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Held-out controlled recovery; synthetic zero-drift random walks")
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _plot_trajectory(trajectory: dict[str, Tensor], path: Path) -> None:
    methods = [
        ("truth_latent", "truth", "#111111"),
        ("hold_latent", "hold", "#999999"),
        ("interpolation_latent", "two-anchor", "#d55e00"),
        ("forward_latent", "forward SMC", "#0072b2"),
        ("right_reweighted_latent", "right reweighted", "#009e73"),
    ]
    figure, axes = plt.subplots(1, len(trajectory["truth_latent"]), figsize=(14, 3.2))
    for time_index, axis in enumerate(axes):
        for key, label, color in methods:
            curve = decode_centerline_torch(trajectory[key][time_index]).numpy()
            axis.plot(curve[:, 0], curve[:, 1], color=color, linewidth=1.2,
                      label=label if time_index == 0 else None)
        axis.set_aspect("equal"); axis.invert_yaxis(); axis.set_title(f"t={time_index}")
        axis.axis("off")
    axes[0].legend(fontsize=7, loc="best")
    figure.suptitle("Representative nominal controlled trajectories")
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/smc_exp_007_controlled_smc.json")
    parser.add_argument("--experiment-dir", type=Path, default=PROJECT_ROOT / "experiments/exp_smc_007_controlled_smc")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("experiment") != "EXP-SMC-007" or config.get("frozen_before_run") is not True:
        raise ValueError("frozen EXP-SMC-007 config required")
    if config["inputs"]["protected_2025_holdout_opened"] is not False:
        raise RuntimeError("protected holdout must remain closed")
    for name, expected in config["inputs"].items():
        if name.endswith("_sha256"):
            source_name = name.removesuffix("_sha256")
            if sha256_file(_resolved(config["inputs"][source_name])) != expected:
                raise RuntimeError(f"input digest mismatch: {source_name}")
    dynamics = json.loads(_resolved(config["inputs"]["dynamics_addendum"]).read_text())
    if dynamics["selected_controlled_experiment_prior"] != "zero_drift_block_diagonal_random_walk":
        raise RuntimeError("EXP-SMC-006 synthetic-prior decision changed")
    representation = json.loads(_resolved(config["inputs"]["representation_metrics"]).read_text())
    observation = json.loads(_resolved(config["inputs"]["observation_metrics"]).read_text())
    if representation["decision"]["selected_shape_coefficients"] != 16:
        raise RuntimeError("selected latent representation changed")
    if observation["decision"]["selected_energy"] != "soft_dice":
        raise RuntimeError("selected observation energy changed")
    results_path = args.experiment_dir / "metrics.json"
    figures_dir = args.experiment_dir / "figures"
    trajectory_path = Path(config["storage"]["full_trajectories"])
    if results_path.exists() or figures_dir.exists() or trajectory_path.exists():
        raise FileExistsError("refusing to overwrite EXP-SMC-007 outputs")
    starts = _load_start_anchors(config)

    calibration: list[dict[str, Any]] = []
    for seed in config["calibration"]["seeds"]:
        for temperature in config["calibration"]["temperatures"]:
            for count in config["calibration"]["particle_counts"]:
                row, _ = _run_case(
                    starts[0], "nominal", seed, count, temperature, config,
                    keep_trajectories=False,
                )
                calibration.append(row)
    calibration.sort(key=lambda row: (
        row["method_errors_px"]["forward_bootstrap_smc_map_genealogy"]["trajectory_mean"],
        row["particle_count"], -row["ess"]["median_fraction"],
    ))
    selected = calibration[0]
    selected_count = selected["particle_count"]
    selected_temperature = selected["temperature"]

    evaluation: list[dict[str, Any]] = []
    trajectories: dict[str, dict[str, Tensor]] = {}
    for scenario_index, scenario in enumerate(config["evaluation"]["scenarios"]):
        for seed_index, seed in enumerate(config["evaluation"]["held_out_seeds"]):
            start = starts[(scenario_index + seed_index + 1) % len(starts)]
            row, stored = _run_case(
                start, scenario, seed, selected_count, selected_temperature, config,
                keep_trajectories=True,
            )
            evaluation.append(row)
            assert stored is not None
            trajectories[f"{scenario}_seed{seed}"] = stored
    aggregate = {
        scenario: _aggregate([row for row in evaluation if row["scenario"] == scenario])
        for scenario in config["evaluation"]["scenarios"]
    }
    nominal = aggregate["nominal"]
    gate = config["gate"]
    checks = {
        "nominal_forward": nominal["methods"]["forward_bootstrap_smc_map_genealogy"]["trajectory_mean_px"]["median"]
        <= gate["nominal_forward_median_trajectory_error_px_max"],
        "nominal_right_reweighted": nominal["methods"]["terminal_right_anchor_reweighted_genealogy"]["trajectory_mean_px"]["median"]
        <= gate["nominal_terminal_reweighted_median_trajectory_error_px_max"],
        "nominal_truth_survival": nominal["truth_survival_fraction"]["median"]
        >= gate["nominal_truth_survival_fraction_min"],
    }
    for scenario, summary in aggregate.items():
        checks[f"stress_{scenario}"] = (
            summary["methods"]["terminal_right_anchor_reweighted_genealogy"]["trajectory_mean_px"]["median"]
            <= gate["all_stress_terminal_reweighted_median_trajectory_error_px_max"]
        )
    decision = {
        "passed": all(checks.values()),
        "decision": "SUPPORTED_CONTROLLED_SYNTHETIC_ONLY" if all(checks.values()) else "NOT_SUPPORTED_CONTROLLED_RECOVERY",
        "checks": checks,
        "comparative_outcome": {
            "two_anchor_interpolation_lowest_median_trajectory_error_all_scenarios": all(
                summary["methods"]["two_anchor_latent_interpolation"]["trajectory_mean_px"]["median"]
                == min(
                    method["trajectory_mean_px"]["median"]
                    for method in summary["methods"].values()
                )
                for summary in aggregate.values()
            ),
            "terminal_reweighting_improves_forward_smc_scenarios": [
                scenario for scenario, summary in aggregate.items()
                if summary["methods"]["terminal_right_anchor_reweighted_genealogy"]["trajectory_mean_px"]["median"]
                < summary["methods"]["forward_bootstrap_smc_map_genealogy"]["trajectory_mean_px"]["median"]
            ],
            "terminal_reweighting_improvement_count": sum(
                summary["methods"]["terminal_right_anchor_reweighted_genealogy"]["trajectory_mean_px"]["median"]
                < summary["methods"]["forward_bootstrap_smc_map_genealogy"]["trajectory_mean_px"]["median"]
                for summary in aggregate.values()
            ),
            "scenario_count": len(aggregate),
        },
        "scientific_decisions": {
            "controlled_algorithm_execution_and_truth_survival": "SUPPORTED_SYNTHETIC_ONLY",
            "terminal_anchor_smoothing_benefit": "NOT_SUPPORTED",
            "smc_superiority_over_two_anchor_interpolation": "NOT_SUPPORTED",
            "H8_supported": False,
        },
        "natural_motion_claim_allowed": False,
        "natural_smc_authorized": False,
    }
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "trajectories": trajectories}, trajectory_path)
    output = {
        "schema_version": 1, "experiment": "EXP-SMC-007",
        "config": str(args.config.resolve(strict=True)), "config_sha256": sha256_file(args.config),
        "input_hashes_verified": True, "device": "cpu",
        "strict_anchor_starts": [{k: v for k, v in start.items() if k not in {"latent", "width"}} for start in starts],
        "calibration": {
            "rows": calibration,
            "selected_particle_count": selected_count,
            "selected_temperature": selected_temperature,
            "selection_rule": config["calibration"]["selection"],
        },
        "evaluation": {"per_case": evaluation, "aggregate_by_scenario": aggregate},
        "decision": decision,
        "omissions": {
            "particle_count_1024": "optional and omitted for bounded CPU runtime",
            "self_contact_bridge": config["evaluation"]["self_contact_bridge"],
        },
        "trajectory_artifact": {
            "configured_path": str(trajectory_path),
            "resolved_path": str(trajectory_path.resolve(strict=True)),
            "sha256": sha256_file(trajectory_path), "bytes": trajectory_path.stat().st_size,
        },
        "evidence_boundary": {
            "truth": "synthetic zero-drift latent random walk",
            "real_strict_anchors_role": "initial shape and width only",
            "empirical_process_scale_claim_allowed": False,
            "natural_motion_claim_allowed": False,
            "protected_2025_holdout_opened": False,
        },
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                     check=True, capture_output=True, text=True).stdout.strip(),
    }
    figures_dir.mkdir(parents=True)
    _plot_calibration(calibration, figures_dir / "calibration_particle_temperature.png")
    _plot_evaluation(evaluation, figures_dir / "controlled_recovery_summary.png")
    _plot_trajectory(trajectories["nominal_seed7101"], figures_dir / "nominal_trajectory.png")
    results_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"selected_particles": selected_count, "selected_temperature": selected_temperature,
                      "decision": decision, "trajectory_artifact": output["trajectory_artifact"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
