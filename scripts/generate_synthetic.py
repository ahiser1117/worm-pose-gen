#!/usr/bin/env python
"""Run the predeclared EXP-0002 exact synthetic/crop contract benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import matplotlib.pyplot as plt
import torch

from worm_pose_gen.geometry import in_fov_mask
from worm_pose_gen.renderer import render_worm
from worm_pose_gen.synthetic import (
    PARAMETER_PROFILES,
    SyntheticConfig,
    anatomical_crop_transform,
    generate_synthetic_pose,
    moving_crop_sequence,
    original_to_render,
    render_to_original,
)


RENDER_NUISANCE_SPEC = {
    "foreground_intensity": {"distribution": "uniform", "range": [0.12, 0.28]},
    "background_intensity": {"distribution": "uniform", "range": [0.68, 0.88]},
    "illumination_gradient_xy": {"distribution": "uniform independent", "range": [-0.07, 0.07]},
    "noise_standard_deviation": {"distribution": "uniform", "range": [0.005, 0.020]},
    "noise_samples": {"distribution": "normal", "mean": 0.0},
    "edge_softness_px": {"distribution": "uniform", "range": [0.45, 0.75]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-samples", type=int, default=512)
    parser.add_argument("--held-out-samples", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/exp_0002_synthetic_crop")
    )
    return parser.parse_args()


def render_example(pose: dict[str, object], config: SyntheticConfig) -> torch.Tensor:
    nuisance_generator = torch.Generator().manual_seed(int(pose["seed"]) + 17)

    def uniform(low: float, high: float, shape: tuple[int, ...] = ()) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, generator=nuisance_generator, dtype=torch.float64)

    foreground = uniform(0.12, 0.28)
    background = uniform(0.68, 0.88)
    illumination = uniform(-0.07, 0.07, (1, 2))
    noise_std = uniform(0.005, 0.020)
    edge_softness = float(uniform(0.45, 0.75))
    noise = noise_std * torch.randn(
        config.render_height, config.render_width, generator=nuisance_generator, dtype=torch.float64
    )
    result = render_worm(
        pose["centerline_render_xy"], pose["width_profile_render"],
        config.render_height, config.render_width,
        foreground=foreground, background=background, edge_softness=edge_softness,
        illumination_gradient=illumination, noise=noise,
    )
    return result["image"].detach()


def main() -> None:
    args = parse_args()
    if not 0 < args.dev_samples <= 512 or not 0 < args.held_out_samples <= 128:
        raise ValueError("sample counts exceed the predeclared budget")
    start = time.perf_counter()
    config = SyntheticConfig()
    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(exist_ok=True)
    dev_seeds = list(range(20260818, 20260818 + args.dev_samples))
    held_out_seeds = list(range(20270000, 20270000 + args.held_out_samples))
    seeds = dev_seeds + held_out_seeds
    fractions = (0.05, 0.10, 0.20, 0.30, 0.40)
    experiment_config = {
        "dev_seed_range": [dev_seeds[0], dev_seeds[-1]],
        "held_out_seed_range": [held_out_seeds[0], held_out_seeds[-1]],
        "parameter_profiles": {
            name: asdict(profile) for name, profile in PARAMETER_PROFILES.items()
        },
        "geometry_distributions_shared": {
            "phase_rad": {"distribution": "uniform independent", "range": [-torch.pi, torch.pi]},
            "global_orientation_rad": {"distribution": "uniform", "range": [-torch.pi, torch.pi]},
            "mode_direction": {
                "distribution": "uniform independent then normalized to exact bend amplitude",
                "raw_range": [-1.0, 1.0],
                "mode_weights": [1.0, 0.55, 0.25],
            },
            "mean_width_px": {"distribution": "uniform", "range": [10.0, 15.0]},
            "placement": "uniform over feasible fully-visible anchors with 18 px margin",
        },
        "render_nuisance_distributions": RENDER_NUISANCE_SPEC,
        "hidden_fractions": list(fractions),
        "hidden_ends": ["head", "tail"],
        "temporal_crop": {
            "frames": 21,
            "start_hidden_fraction": 0.05,
            "end_hidden_fraction": 0.40,
            "boundary_motion": "linear in longitudinal source-coordinate position",
        },
        "coordinate_convention": "pixel centers; x right, y down; half-open bounds",
        "original_canvas_hw": [732, 968],
        "render_canvas_hw": [192, 256],
        "mapping": "u=x*(render_size/original_size) per axis; preserves half-open FOV membership",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "data_persistence": "none; deterministic arrays generated on demand",
    }
    # The complete distribution contract is persisted before the first sample is generated.
    (args.output / "config.json").write_text(json.dumps(experiment_config, indent=2) + "\n")
    max_roundtrip = 0.0
    max_raster_roundtrip = 0.0
    lengths: dict[str, list[float]] = {"development": [], "held_out": []}
    bend_amplitudes: dict[str, list[float]] = {"development": [], "held_out": []}
    maximum_curvature: list[tuple[float, int, str]] = []
    crop_cases = 0
    poses: dict[int, dict[str, object]] = {}
    for seed in seeds:
        profile = "development" if seed in dev_seeds else "held_out"
        pose = generate_synthetic_pose(seed, config, profile=profile)
        centerline = pose["centerline_xy"]
        if not bool(torch.isfinite(centerline).all()) or not bool(pose["in_fov_mask"].all()):
            raise RuntimeError(f"geometry invariant failed for seed {seed}")
        length = float(pose["body_length"])
        lengths[profile].append(length)
        bend_amplitudes[profile].append(float(pose["bend_amplitude"]))
        maximum_curvature.append((float(pose["curvature"].abs().max()), seed, profile))
        raster_error = (render_to_original(original_to_render(centerline, config), config) - centerline).abs().max()
        max_raster_roundtrip = max(max_raster_roundtrip, float(raster_error))
        for fraction in fractions:
            for hidden_end in ("head", "tail"):
                transform, camera, support = anatomical_crop_transform(
                    centerline, fraction, hidden_end, config
                )
                if int((~support).sum()) != round(fraction * config.num_points):
                    raise RuntimeError("hidden count mismatch")
                if not torch.equal(support, in_fov_mask(camera, 732, 968)):
                    raise RuntimeError("half-open support mismatch")
                if not torch.equal(support, in_fov_mask(original_to_render(camera), 192, 256)):
                    raise RuntimeError("original/render support mapping mismatch")
                error = (transform.to_source(camera) - centerline).abs().max()
                max_roundtrip = max(max_roundtrip, float(error))
                crop_cases += 1
        if seed in seeds[:4]:
            poses[seed] = pose

    # Renderer gradient audit.
    audit_pose = generate_synthetic_pose(seeds[0], config, profile="development")
    audit_xy = audit_pose["centerline_render_xy"].clone().requires_grad_()
    audit_width = audit_pose["width_profile_render"].clone().requires_grad_()
    rendered = render_worm(audit_xy, audit_width, 192, 256, edge_softness=0.55)
    weighted = rendered["image"] * torch.linspace(0.1, 1.0, 256, dtype=torch.float64)[None, :]
    weighted.sum().backward()
    gradient_finite = bool(torch.isfinite(audit_xy.grad).all() and torch.isfinite(audit_width.grad).all())
    gradient_norm = float(audit_xy.grad.abs().sum() + audit_width.grad.abs().sum())
    if not gradient_finite or gradient_norm <= 0:
        raise RuntimeError("renderer gradient contract failed")

    # Deterministic random sample montage plus most-curved held-out example.
    _, curved_seed, curved_profile = max(maximum_curvature)
    montage_seeds = dev_seeds[:3] + held_out_seeds[:1] + [curved_seed]
    fig, axes = plt.subplots(len(montage_seeds), 1, figsize=(8, 7), constrained_layout=True)
    for axis, seed in zip(axes, montage_seeds, strict=True):
        profile = "development" if seed in dev_seeds else "held_out"
        pose = poses.get(seed) or generate_synthetic_pose(seed, config, profile=profile)
        axis.imshow(render_example(pose, config), cmap="gray", vmin=0, vmax=1)
        xy = pose["centerline_render_xy"]
        axis.plot(xy[:, 0], xy[:, 1], color="tab:orange", linewidth=0.8)
        suffix = " (most curved)" if seed == curved_seed else ""
        axis.set_title(f"{profile}: seed {seed}{suffix}", fontsize=8)
        axis.axis("off")
    fig.savefig(figure_dir / "generator_montage.png", dpi=150)
    plt.close(fig)

    source = generate_synthetic_pose(seeds[0], config, profile="development")["centerline_xy"]
    sequence_audits = {}
    fig, axes = plt.subplots(2, 5, figsize=(12, 4), constrained_layout=True)
    for row, hidden_end in enumerate(("head", "tail")):
        sequence = moving_crop_sequence(source, hidden_end=hidden_end, config=config)
        cameras = sequence["centerline_camera_xy"]
        supports = sequence["support_mask"]
        transforms = sequence["transforms"]
        offsets = torch.stack([transform.offset for transform in transforms])
        second_difference = offsets[2:] - 2 * offsets[1:-1] + offsets[:-2]
        sequence_error = max(
            float((transform.to_source(camera) - source).abs().max())
            for transform, camera in zip(transforms, cameras, strict=True)
        )
        sequence_audits[hidden_end] = {
            "frames": len(transforms),
            "hidden_counts": sequence["hidden_count"].tolist(),
            "max_transform_roundtrip_error_px": sequence_error,
            "max_offset_second_difference_px": float(second_difference.abs().max()),
            "exact_half_open_support": bool(torch.equal(
                supports, in_fov_mask(cameras, config.original_height, config.original_width)
            )),
        }
        if not sequence_audits[hidden_end]["exact_half_open_support"]:
            raise RuntimeError("temporal crop support mismatch")
        selected_frames = (0, 5, 10, 15, 20)
        for column, frame in enumerate(selected_frames):
            camera, support = cameras[frame], supports[frame]
            camera_render = original_to_render(camera, config)
            image = render_worm(camera_render, 2.8, 192, 256, edge_softness=0.55)["image"]
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, column].plot(camera_render[support, 0], camera_render[support, 1], color="tab:orange", linewidth=0.7)
            hidden_count = int(sequence["hidden_count"][frame])
            axes[row, column].set_title(f"{hidden_end}: frame {frame}, hidden {hidden_count}%", fontsize=8)
            axes[row, column].axis("off")
    fig.savefig(figure_dir / "crop_sequence_montage.png", dpi=150)
    plt.close(fig)

    runtime = time.perf_counter() - start
    metrics = {
        "status": "ACCEPT",
        "evidence_tier": "Tier C controlled synthetic truth",
        "dev_samples": args.dev_samples,
        "held_out_samples": args.held_out_samples,
        "num_points": config.num_points,
        "crop_cases": crop_cases,
        "all_finite": True,
        "exact_half_open_fov_agreement": True,
        "exact_hidden_counts": True,
        "max_crop_roundtrip_error_px": max_roundtrip,
        "max_raster_roundtrip_error_px": max_raster_roundtrip,
        "renderer_gradient_finite": gradient_finite,
        "renderer_gradient_l1": gradient_norm,
        "development_body_length_range_px": [min(lengths["development"]), max(lengths["development"])],
        "held_out_body_length_range_px": [min(lengths["held_out"]), max(lengths["held_out"])],
        "held_out_lower_band_count": sum(value <= 299 for value in lengths["held_out"]),
        "held_out_upper_band_count": sum(value >= 601 for value in lengths["held_out"]),
        "development_bend_amplitude_range_rad": [min(bend_amplitudes["development"]), max(bend_amplitudes["development"])],
        "held_out_bend_amplitude_range_rad": [min(bend_amplitudes["held_out"]), max(bend_amplitudes["held_out"])],
        "temporal_crop_audits": sequence_audits,
        "runtime_seconds": runtime,
        "most_curved_seed": curved_seed,
        "most_curved_profile": curved_profile,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "stdout.log").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
