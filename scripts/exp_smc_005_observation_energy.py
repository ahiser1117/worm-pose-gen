#!/usr/bin/env python3
"""EXP-SMC-005 controlled mask-observation energy audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scripts.exp_005_representation_oracle import (
        cubic_bspline_basis,
        intrinsic_target,
        reconstruct_from_shape,
    )
    from scripts.exp_smc_001_002_audit import _read_windows, _verified_annotations
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from exp_005_representation_oracle import (
        cubic_bspline_basis,
        intrinsic_target,
        reconstruct_from_shape,
    )
    from exp_smc_001_002_audit import _read_windows, _verified_annotations

from worm_pose_gen.annotation import resample_polyline
from worm_pose_gen.anchors import estimate_width_along_normals
from worm_pose_gen.observation import (
    balanced_soft_bce_energy,
    hybrid_mask_energy,
    signed_distance_energy,
    signed_distance_from_mask,
    soft_dice_energy,
)
from worm_pose_gen.renderer import render_worm
from worm_pose_gen.segmentation import SoftForegroundConfig, segment_soft_foreground


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/smc_exp_005_observation_energy.json"
DEFAULT_OUTPUT = REPO / "experiments/exp_smc_005_observation_energy"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _downsample(values: np.ndarray, size: tuple[int, int], *, mode: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)[None, None]
    if mode == "nearest":
        result = F.interpolate(tensor, size=size, mode=mode)
    else:
        result = F.interpolate(tensor, size=size, mode=mode, align_corners=False)
    return result[0, 0]


def _fit_cubic(points: np.ndarray, coefficients: int) -> tuple[np.ndarray, np.ndarray]:
    shape, rotation, length = intrinsic_target(points)
    basis = cubic_bspline_basis(len(shape), coefficients)
    fitted = np.linalg.lstsq(basis, shape, rcond=None)[0]
    reconstruction = reconstruct_from_shape(basis @ fitted, rotation, length, points)
    latent = np.concatenate((fitted, [rotation, length], points.mean(axis=0)))
    return reconstruction, latent


def _reconstruct_torch_latent(latent: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coefficients = latent[:-4]
    rotation, length = latent[-4], latent[-3]
    translation = latent[-2:]
    angle = basis @ coefficients + rotation
    step = length / len(angle)
    difference = step * torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
    curve = torch.cat(
        (
            torch.zeros(1, 2, dtype=latent.dtype, device=latent.device),
            difference.cumsum(0),
        )
    )
    return curve - curve.mean(0, keepdim=True) + translation


def _perturbed_curves(
    base: torch.Tensor,
    name: str,
    levels: list[float],
    downsample_factor: float,
) -> torch.Tensor:
    candidates = base.unsqueeze(0).repeat(len(levels), 1, 1)
    values = torch.as_tensor(levels, dtype=base.dtype, device=base.device)
    centroid = base.mean(0, keepdim=True)
    if name == "translation_x":
        candidates[:, :, 0] += values[:, None] / downsample_factor
    elif name == "rotation":
        angle = torch.deg2rad(values)
        relative = base - centroid
        cosine, sine = torch.cos(angle), torch.sin(angle)
        candidates[:, :, 0] = (
            centroid[0, 0]
            + cosine[:, None] * relative[:, 0]
            - sine[:, None] * relative[:, 1]
        )
        candidates[:, :, 1] = (
            centroid[0, 1]
            + sine[:, None] * relative[:, 0]
            + cosine[:, None] * relative[:, 1]
        )
    elif name == "shape_normal_amplitude":
        tangent = torch.empty_like(base)
        tangent[1:-1] = base[2:] - base[:-2]
        tangent[0] = base[1] - base[0]
        tangent[-1] = base[-1] - base[-2]
        tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp_min(1e-6)
        normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
        envelope = torch.sin(
            torch.linspace(0, 2 * torch.pi, len(base), device=base.device)
        )
        candidates += (
            values[:, None, None]
            / downsample_factor
            * envelope[None, :, None]
            * normal[None]
        )
    elif name == "length_error":
        candidates = centroid + (1 + values[:, None, None] / 100.0) * (base - centroid)
    else:
        raise ValueError(f"unknown perturbation: {name}")
    return candidates


def _energies(
    rendered: torch.Tensor,
    probability: torch.Tensor,
    mask: torch.Tensor,
    sdf: torch.Tensor,
    render_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    softness = float(render_config["edge_softness_px"])
    clip = float(render_config["signed_distance_clip_px"])
    return {
        "balanced_bce": balanced_soft_bce_energy(rendered, probability),
        "soft_dice": soft_dice_energy(rendered, mask),
        "signed_distance": signed_distance_energy(
            rendered, sdf, edge_softness=softness, clip_distance=clip
        ),
        "hybrid": hybrid_mask_energy(
            rendered,
            mask,
            sdf,
            edge_softness=softness,
            clip_distance=clip,
            signed_distance_weight=float(render_config["hybrid_signed_distance_weight"]),
        ),
    }


def _gradient_diagnostics(
    latent_values: torch.Tensor,
    basis: torch.Tensor,
    width: torch.Tensor,
    probability: torch.Tensor,
    mask: torch.Tensor,
    sdf: torch.Tensor,
    render_config: dict[str, Any],
) -> dict[str, dict[str, float | bool]]:
    result: dict[str, dict[str, float | bool]] = {}
    for name in ("balanced_bce", "soft_dice", "signed_distance", "hybrid"):
        latent = latent_values.detach().clone().requires_grad_(True)
        points = _reconstruct_torch_latent(latent, basis)
        rendered = render_worm(
            points,
            width,
            int(render_config["image_height"]),
            int(render_config["image_width"]),
            edge_softness=float(render_config["edge_softness_px"]),
        )["tube_mask"]
        energy = _energies(rendered, probability, mask, sdf, render_config)[name]
        gradient = torch.autograd.grad(energy, latent)[0]
        finite = bool(torch.isfinite(gradient).all())
        norm = float(gradient.norm().detach().cpu()) if finite else float("nan")
        result[name] = {
            "finite": finite,
            "l2_norm": norm,
            "nonzero": norm > 0,
            "shape_coefficient_l2_norm": float(gradient[:-4].norm().detach().cpu()),
            "rotation_abs_gradient": float(gradient[-4].abs().detach().cpu()),
            "length_abs_gradient": float(gradient[-3].abs().detach().cpu()),
            "translation_l2_norm": float(gradient[-2:].norm().detach().cpu()),
        }
    return result


def _curve_gate_summary(
    cases: list[dict[str, Any]], energy_name: str, perturbations: list[str]
) -> dict[str, Any]:
    near_by_kind: dict[str, list[bool]] = {kind: [] for kind in perturbations}
    monotonic_by_kind: dict[str, list[bool]] = {kind: [] for kind in perturbations}
    endpoint_by_kind: dict[str, list[float]] = {kind: [] for kind in perturbations}
    for case in cases:
        for kind in perturbations:
            values = np.asarray(case["perturbations"][kind]["energies"][energy_name])
            center = len(values) // 2
            near_by_kind[kind].append(abs(int(np.argmin(values)) - center) <= 1)
            comparisons = [
                values[center - 1] >= values[center],
                values[center - 2] >= values[center - 1],
                values[center - 3] >= values[center - 2],
                values[center + 1] >= values[center],
                values[center + 2] >= values[center + 1],
                values[center + 3] >= values[center + 2],
            ]
            monotonic_by_kind[kind].extend(map(bool, comparisons))
            endpoint_by_kind[kind].append(
                float(0.5 * (values[0] + values[-1]) - values[center])
            )
    all_near = [value for values in near_by_kind.values() for value in values]
    all_monotonic = [value for values in monotonic_by_kind.values() for value in values]
    return {
        "overall_near_zero_minimum_fraction": float(np.mean(all_near)),
        "near_zero_minimum_fraction_by_perturbation": {
            key: float(np.mean(values)) for key, values in near_by_kind.items()
        },
        "overall_outward_monotonic_step_fraction": float(np.mean(all_monotonic)),
        "outward_monotonic_step_fraction_by_perturbation": {
            key: float(np.mean(values)) for key, values in monotonic_by_kind.items()
        },
        "endpoint_minus_zero_energy_by_perturbation": {
            key: summary(values) for key, values in endpoint_by_kind.items()
        },
    }


def _benchmark(
    base: torch.Tensor,
    width: torch.Tensor,
    probability: torch.Tensor,
    mask: torch.Tensor,
    sdf: torch.Tensor,
    selected_energy: str,
    render_config: dict[str, Any],
) -> dict[str, Any]:
    warmup = int(render_config["benchmark_warmup"])
    repeats = int(render_config["benchmark_repeats"])
    result: dict[str, Any] = {}
    for batch_size in render_config["benchmark_batch_sizes"]:
        batch_size = int(batch_size)
        points = base.unsqueeze(0).repeat(batch_size, 1, 1)
        timings: list[float] = []
        for repeat in range(warmup + repeats):
            torch.cuda.synchronize(base.device)
            start = time.perf_counter()
            rendered = render_worm(
                points,
                width,
                int(render_config["image_height"]),
                int(render_config["image_width"]),
                edge_softness=float(render_config["edge_softness_px"]),
            )["tube_mask"]
            value = _energies(rendered, probability, mask, sdf, render_config)[selected_energy]
            _ = float(value.sum().detach().cpu())
            torch.cuda.synchronize(base.device)
            elapsed = 1000 * (time.perf_counter() - start)
            if repeat >= warmup:
                timings.append(elapsed)
        result[str(batch_size)] = {
            "batch_size": batch_size,
            "total_ms": summary(timings),
            "ms_per_particle": summary([value / batch_size for value in timings]),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    output = args.output.resolve()
    config = json.loads(config_path.read_text())
    aggregation = config["aggregation"]
    if (
        aggregation["zero_level_index"] != 3
        or aggregation["near_zero_minimum_indices"] != [2, 3, 4]
        or aggregation["outward_monotonic_comparisons_per_curve"] != 6
    ):
        raise RuntimeError("runner and frozen aggregation contract disagree")
    input_config = config["inputs"]
    segmentation_path = (REPO / input_config["segmentation_config"]).resolve(strict=True)
    segmentation_document = json.loads(segmentation_path.read_text())
    representation_path = (REPO / input_config["representation_metrics"]).resolve(strict=True)
    representation = json.loads(representation_path.read_text())
    width_path = (REPO / input_config["width_metrics"]).resolve(strict=True)
    width_metrics = json.loads(width_path.read_text())
    adjudication_path = (REPO / input_config["expert_adjudication"]).resolve(strict=True)
    annotation_path = Path(input_config["annotations"]).resolve(strict=True)
    manifest_path = (REPO / input_config["manifest"]).resolve(strict=True)
    if representation["decision"]["selected_family"] != "cubic_tangent_spline":
        raise RuntimeError("EXP-SMC-003 did not select the preregistered cubic representation")
    coefficients = int(representation["decision"]["selected_shape_coefficients"])
    if coefficients != 16 or not width_metrics["decision"]["passed"]:
        raise RuntimeError("required representation or width evidence is unavailable")
    rows = _verified_annotations(manifest_path, annotation_path, segmentation_document)
    excluded = set(input_config["excluded_expert_hard_samples"])
    selected = [
        (source, annotation)
        for source, annotation in rows
        if annotation.is_complete and annotation.sample_id not in excluded
    ]
    if len(selected) != 16:
        raise RuntimeError(f"expected 16 complete non-hard traces, found {len(selected)}")
    if any(str(source["recording"]).startswith("2025-") for source, _ in selected):
        raise RuntimeError("protected holdout was selected")
    _safe_output(output)

    frames = _read_windows(selected, 0)
    segment_config = SoftForegroundConfig(**segmentation_document["soft_foreground_config"])
    render_config = config["render"]
    device = torch.device(render_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EXP-SMC-005 preregistered a CUDA run")
    height, width = int(render_config["image_height"]), int(render_config["image_width"])
    downsample_factor = float(render_config["downsample_factor"])
    cases: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, Any]] = {}
    by_recording: dict[str, list[str]] = {}
    for source, annotation in selected:
        recording = str(source["recording"])
        sample_id = annotation.sample_id
        frame = frames[(recording, int(source["frame_index"]))]
        segmentation = segment_soft_foreground(frame, segment_config)
        target = resample_polyline(annotation.points_xy, 100)
        reconstructed, latent = _fit_cubic(target, coefficients)
        measured_width = estimate_width_along_normals(
            segmentation.cleaned_mask, target, step=0.5
        )
        measured_width = 0.5 * (measured_width + measured_width[::-1])
        arrays[sample_id] = {
            "frame": frame,
            "probability": segmentation.probability,
            "mask": segmentation.cleaned_mask,
            "target": target,
            "reconstructed": reconstructed,
            "latent": latent,
            "measured_width": measured_width,
        }
        by_recording.setdefault(recording, []).append(sample_id)
        cases.append({
            "sample_id": sample_id,
            "recording": recording,
            "frame_index": int(source["frame_index"]),
            "selection_stratum": source["selection_stratum"],
        })

    perturbation_levels = {
        "translation_x": config["perturbations"]["translation_x_px_original"],
        "rotation": config["perturbations"]["rotation_deg"],
        "shape_normal_amplitude": config["perturbations"]["shape_normal_amplitude_px_original"],
        "length_error": config["perturbations"]["length_error_percent"],
    }
    base_render_overlays: list[dict[str, Any]] = []
    for case in cases:
        sample_id = case["sample_id"]
        values = arrays[sample_id]
        train_ids = [
            other for other in by_recording[case["recording"]] if other != sample_id
        ]
        profile = np.stack([arrays[other]["measured_width"] for other in train_ids]).mean(0)
        base = torch.as_tensor(
            values["reconstructed"] / downsample_factor,
            dtype=torch.float32,
            device=device,
        )
        profile_tensor = torch.as_tensor(
            profile / downsample_factor, dtype=torch.float32, device=device
        )
        latent_scaled = values["latent"].copy()
        latent_scaled[-3:] /= downsample_factor
        latent_tensor = torch.as_tensor(latent_scaled, dtype=torch.float32, device=device)
        basis_tensor = torch.as_tensor(
            cubic_bspline_basis(99, coefficients), dtype=torch.float32, device=device
        )
        probability = _downsample(values["probability"], (height, width), mode="bilinear").to(device)
        mask = _downsample(values["mask"], (height, width), mode="nearest").to(device)
        sdf = signed_distance_from_mask(mask.bool())
        case["loo_width_profile_summary_px_original"] = summary(profile.tolist())
        case["perturbations"] = {}
        with torch.no_grad():
            base_render = render_worm(
                base,
                profile_tensor,
                height,
                width,
                edge_softness=float(render_config["edge_softness_px"]),
            )["tube_mask"]
            base_dice = 1 - float(soft_dice_energy(base_render, mask).cpu())
        case["base_soft_dice"] = base_dice
        for kind, levels in perturbation_levels.items():
            candidates = _perturbed_curves(base, kind, list(map(float, levels)), downsample_factor)
            with torch.no_grad():
                rendered = render_worm(
                    candidates,
                    profile_tensor,
                    height,
                    width,
                    edge_softness=float(render_config["edge_softness_px"]),
                )["tube_mask"]
                energies = _energies(rendered, probability, mask, sdf, render_config)
            case["perturbations"][kind] = {
                "levels": list(map(float, levels)),
                "energies": {
                    name: tensor.detach().cpu().tolist() for name, tensor in energies.items()
                },
            }
        case["gradients"] = _gradient_diagnostics(
            latent_tensor,
            basis_tensor,
            profile_tensor,
            probability,
            mask,
            sdf,
            render_config,
        )
        base_render_overlays.append({
            "sample_id": sample_id,
            "frame": values["frame"],
            "mask": values["mask"],
            "reconstructed": values["reconstructed"],
            "base_soft_dice": base_dice,
        })

    kinds = list(perturbation_levels)
    candidate_summaries = {
        name: _curve_gate_summary(cases, name, kinds)
        for name in config["energies"]["candidates"]
    }
    gate = config["gate"]
    for name, values in candidate_summaries.items():
        gradient_ok = all(
            bool(case["gradients"][name]["finite"])
            and bool(case["gradients"][name]["nonzero"])
            and float(case["gradients"][name]["shape_coefficient_l2_norm"]) > 0
            and float(case["gradients"][name]["rotation_abs_gradient"]) > 0
            and float(case["gradients"][name]["length_abs_gradient"]) > 0
            and float(case["gradients"][name]["translation_l2_norm"]) > 0
            for case in cases
        )
        checks = {
            "overall_near_zero_minimum_fraction": (
                values["overall_near_zero_minimum_fraction"]
                >= float(gate["overall_near_zero_minimum_fraction_min"])
            ),
            "per_perturbation_near_zero_minimum_fraction": all(
                value >= float(gate["per_perturbation_near_zero_minimum_fraction_min"])
                for value in values["near_zero_minimum_fraction_by_perturbation"].values()
            ),
            "outward_monotonic_step_fraction": (
                values["overall_outward_monotonic_step_fraction"]
                >= float(gate["outward_monotonic_step_fraction_min"])
            ),
            "endpoint_minus_zero_energy": all(
                value["median"] > float(gate["median_endpoint_minus_zero_energy_min"])
                for value in values["endpoint_minus_zero_energy_by_perturbation"].values()
            ),
            "finite_nonzero_gradient": gradient_ok,
        }
        values["checks"] = checks
        values["passed"] = all(checks.values())
    passing = [
        name
        for name in config["energies"]["preference_order"]
        if candidate_summaries[name]["passed"]
    ]
    selected_energy = passing[0] if passing else None
    decision = {
        "passed": selected_energy is not None,
        "decision": "SUPPORTED" if selected_energy is not None else "NOT_SUPPORTED",
        "selected_energy": selected_energy,
        "passing_energies": passing,
    }

    first_case = cases[0]
    first_values = arrays[first_case["sample_id"]]
    first_training = [
        other
        for other in by_recording[first_case["recording"]]
        if other != first_case["sample_id"]
    ]
    first_profile = np.stack([arrays[other]["measured_width"] for other in first_training]).mean(0)
    first_base = torch.as_tensor(
        first_values["reconstructed"] / downsample_factor,
        dtype=torch.float32,
        device=device,
    )
    first_width = torch.as_tensor(first_profile / downsample_factor, dtype=torch.float32, device=device)
    first_probability = _downsample(first_values["probability"], (height, width), mode="bilinear").to(device)
    first_mask = _downsample(first_values["mask"], (height, width), mode="nearest").to(device)
    first_sdf = signed_distance_from_mask(first_mask.bool())
    benchmark = (
        _benchmark(
            first_base,
            first_width,
            first_probability,
            first_mask,
            first_sdf,
            selected_energy,
            render_config,
        )
        if selected_energy is not None
        else None
    )

    metrics = {
        "schema_version": 1,
        "experiment": "EXP-SMC-005",
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "segmentation_config": str(segmentation_path),
            "segmentation_config_sha256": sha256(segmentation_path),
            "representation_metrics": str(representation_path),
            "representation_metrics_sha256": sha256(representation_path),
            "width_metrics": str(width_path),
            "width_metrics_sha256": sha256(width_path),
            "expert_adjudication": str(adjudication_path),
            "expert_adjudication_sha256": sha256(adjudication_path),
            "annotations": str(annotation_path),
            "annotations_sha256": sha256(annotation_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "protected_2025_holdout_opened": False,
        },
        "evidence_boundary": config["evidence_boundary"],
        "device": {
            "requested": str(device),
            "name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        },
        "selected_frames": len(cases),
        "selected_by_recording": {key: len(value) for key, value in by_recording.items()},
        "base_soft_dice": summary([case["base_soft_dice"] for case in cases]),
        "candidate_summary": candidate_summaries,
        "decision": decision,
        "runtime_benchmark": benchmark,
        "per_case": cases,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    colors = {
        "balanced_bce": "#0072b2",
        "soft_dice": "#009e73",
        "signed_distance": "#e69f00",
        "hybrid": "#cc79a7",
    }
    labels = {
        "translation_x": "translation x (original px)",
        "rotation": "rotation (degrees)",
        "shape_normal_amplitude": "normal shape amplitude (original px)",
        "length_error": "length error (%)",
    }
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), squeeze=False)
    for row, energy_name in enumerate(config["energies"]["candidates"]):
        for column, kind in enumerate(kinds):
            axis = axes[row, column]
            levels = np.asarray(perturbation_levels[kind], dtype=np.float64)
            matrix = np.asarray(
                [case["perturbations"][kind]["energies"][energy_name] for case in cases]
            )
            delta = matrix - matrix[:, [len(levels) // 2]]
            median = np.median(delta, axis=0)
            low, high = np.percentile(delta, [25, 75], axis=0)
            axis.fill_between(levels, low, high, color=colors[energy_name], alpha=0.2)
            axis.plot(levels, median, marker="o", color=colors[energy_name])
            axis.axvline(0, color="#555555", lw=0.8, ls="--")
            axis.axhline(0, color="#555555", lw=0.8, ls=":")
            if row == 0:
                axis.set_title(labels[kind])
            if column == 0:
                axis.set_ylabel(f"{energy_name}\nenergy minus zero")
            if row == 3:
                axis.set_xlabel(labels[kind])
    fig.suptitle("EXP-SMC-005 controlled observation-energy basins (median and IQR)")
    fig.tight_layout()
    fig.savefig(output / "figures/energy_perturbation_curves.png", dpi=170)
    plt.close(fig)

    ordered = sorted(base_render_overlays, key=lambda item: item["base_soft_dice"])
    display = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), squeeze=False)
    for row, item in enumerate(display):
        axes[row, 0].imshow(item["frame"], cmap="gray")
        axes[row, 0].plot(item["reconstructed"][:, 0], item["reconstructed"][:, 1], color="#d55e00", lw=1.5)
        axes[row, 0].set_title(f"{item['sample_id']} reconstructed pose")
        axes[row, 1].imshow(item["mask"], cmap="gray")
        axes[row, 1].set_title("cleaned observation mask")
        axes[row, 2].imshow(item["frame"], cmap="gray")
        axes[row, 2].imshow(item["mask"], cmap="Blues", alpha=0.25)
        axes[row, 2].plot(item["reconstructed"][:, 0], item["reconstructed"][:, 1], color="#d55e00", lw=1.5)
        axes[row, 2].set_title(f"overlay; soft Dice {item['base_soft_dice']:.3f}")
        for axis in axes[row]:
            axis.set_axis_off()
    fig.suptitle("EXP-SMC-005 base-pose visual audit: worst, median, best")
    fig.tight_layout()
    fig.savefig(output / "figures/base_pose_overlays.png", dpi=170)
    plt.close(fig)

    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
