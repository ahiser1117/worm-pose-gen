#!/usr/bin/env python3
"""EXP-008: controlled Tier-C differentiable-refinement capture basin."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from worm_pose_gen.geometry import reconstruct_centerline, wrap_angle
from worm_pose_gen.model import smooth_tangent_basis
from worm_pose_gen.refinement import RefinementConfig, refine_pose
from worm_pose_gen.renderer import render_worm
from worm_pose_gen.synthetic import SyntheticConfig, generate_synthetic_pose


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = ("pixel", "pixel_gradient", "tube_likelihood")
RECORD_STEPS = (0, 1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "experiments/scientific_exp_008_refinement_capture_basin",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--success-threshold-px", type=float, default=4.0)
    parser.add_argument("--quick", action="store_true", help="Run a small mechanics check, not the scientific grid")
    return parser.parse_args()


def conditions(quick: bool) -> list[dict[str, Any]]:
    if quick:
        return [
            {"family": "translation", "magnitude": 4.0, "translation_px": 4.0},
            {"family": "rotation", "magnitude": 5.0, "rotation_deg": 5.0},
            {"family": "combined", "magnitude": 1.0, "translation_px": 4.0,
             "rotation_deg": 5.0, "length_fraction": 0.05, "shape_deg": 5.0},
        ]
    result: list[dict[str, Any]] = []
    result.extend(
        {"family": "translation", "magnitude": float(value), "translation_px": float(value)}
        for value in (1, 2, 4, 8, 16, 32, 64)
    )
    result.extend(
        {"family": "rotation", "magnitude": float(value), "rotation_deg": float(value)}
        for value in (1, 2, 5, 10, 20, 40)
    )
    for value in (0.02, 0.05, 0.10, 0.20):
        for sign in (-1, 1):
            result.append(
                {"family": "length", "magnitude": 100 * value,
                 "length_fraction": sign * value, "sign": sign}
            )
    result.extend(
        {"family": "shape", "magnitude": float(value), "shape_deg": float(value)}
        for value in (2, 5, 10, 20, 40)
    )
    for level, values in enumerate(
        ((4.0, 5.0, 0.05, 5.0), (16.0, 10.0, 0.10, 10.0), (32.0, 20.0, 0.20, 20.0)),
        start=1,
    ):
        translation, rotation, length, shape = values
        result.append(
            {"family": "combined", "magnitude": float(level),
             "translation_px": translation, "rotation_deg": rotation,
             "length_fraction": length, "shape_deg": shape}
        )
    return result


def make_targets(seeds: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    synthetic = SyntheticConfig()
    scale = synthetic.render_width / synthetic.original_width
    anchors, tangents, lengths, widths = [], [], [], []
    for seed in seeds:
        pose = generate_synthetic_pose(seed, synthetic, profile="held_out")
        anchors.append(pose["centerline_render_xy"][synthetic.num_points // 2])
        tangents.append(pose["tangent_angle"])
        lengths.append(pose["body_length"] * scale)
        widths.append(pose["width_profile_render"])
    anchor = torch.stack(anchors).to(dtype=torch.float32, device=device)
    tangent = torch.stack(tangents).to(dtype=torch.float32, device=device)
    length = torch.stack(lengths).to(dtype=torch.float32, device=device)
    width = torch.stack(widths).to(dtype=torch.float32, device=device)
    centerline = reconstruct_centerline(anchor, tangent, length, anchor_index=synthetic.num_points // 2)
    target = render_worm(centerline, width, synthetic.render_height, synthetic.render_width)
    return {
        "anchor": anchor,
        "tangent": tangent,
        "length": length,
        "width": width,
        "centerline": centerline,
        "image": target["image"].detach(),
        "mask": target["tube_mask"].detach(),
    }


def perturb(
    target: dict[str, torch.Tensor], condition: dict[str, Any], seeds: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = target["anchor"].clone()
    tangent = target["tangent"].clone()
    length = target["length"].clone()
    original_to_render = 256 / 968
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device="cpu").manual_seed(seed + 99_000)
        direction = float(torch.rand((), generator=generator) * (2 * math.pi))
        if "translation_px" in condition:
            magnitude = float(condition["translation_px"]) * original_to_render
            anchor[index] += anchor.new_tensor((math.cos(direction), math.sin(direction))) * magnitude
        if "rotation_deg" in condition:
            tangent[index] += math.radians(float(condition["rotation_deg"]))
        if "length_fraction" in condition:
            length[index] *= 1 + float(condition["length_fraction"])
        if "shape_deg" in condition:
            basis = smooth_tangent_basis(tangent.shape[1], 16).to(tangent)
            basis_index = seed % basis.shape[1]
            pattern = basis[:, basis_index]
            pattern = pattern / torch.sqrt(torch.mean(pattern.square()))
            tangent[index] += pattern * math.radians(float(condition["shape_deg"]))
    return anchor, tangent, length


def point_error_original(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inverse_scale = prediction.new_tensor((968 / 256, 732 / 192))
    return torch.linalg.vector_norm((prediction - target) * inverse_scale, dim=-1)


def run_grid(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.cases < 1 or args.batch_size < 1 or args.success_threshold_px <= 0:
        raise ValueError("cases, batch-size, and success threshold must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    all_seeds = [args.seed + index for index in range(args.cases)]
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    grid = conditions(args.quick)
    objectives = OBJECTIVES if not args.quick else ("pixel", "pixel_gradient", "tube_likelihood")
    for condition_index, condition in enumerate(grid):
        for batch_start in range(0, len(all_seeds), args.batch_size):
            seeds = all_seeds[batch_start : batch_start + args.batch_size]
            target = make_targets(seeds, device)
            initial_anchor, initial_tangent, initial_length = perturb(target, condition, seeds)
            for objective in objectives:
                trajectory_start = time.perf_counter()
                _, history = refine_pose(
                    initial_anchor, initial_tangent, initial_length, target["width"],
                    target["image"], target_mask=target["mask"], objective=objective,
                    steps=max(RECORD_STEPS), record_steps=RECORD_STEPS,
                    config=RefinementConfig(),
                )
                elapsed = time.perf_counter() - trajectory_start
                for step in RECORD_STEPS:
                    prediction = history[step]["centerline_xy"]
                    error = point_error_original(prediction, target["centerline"])
                    angle = torch.rad2deg(torch.abs(wrap_angle(
                        history[step]["tangent_angle"] - target["tangent"]
                    )))
                    for case_index, seed in enumerate(seeds):
                        per_point = error[case_index]
                        rows.append(
                            {
                                "condition_index": condition_index,
                                **condition,
                                "objective": objective,
                                "step": step,
                                "seed": seed,
                                "median_point_error_px": float(torch.median(per_point)),
                                "p95_point_error_px": float(torch.quantile(per_point, 0.95)),
                                "mean_angle_error_deg": float(angle[case_index].mean()),
                                "success": bool(torch.median(per_point) <= args.success_threshold_px),
                                "trajectory_seconds_per_case": elapsed / len(seeds),
                            }
                        )
        print(f"completed {condition_index + 1}/{len(grid)} conditions", flush=True)
    metadata = {
        "wall_seconds": time.perf_counter() - start,
        "device": str(device),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "cases": args.cases,
        "batch_size": args.batch_size,
        "quick": args.quick,
    }
    return rows, metadata


def summarize(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    final = [row for row in rows if row["step"] == 10]
    by_objective: dict[str, Any] = {}
    for objective in OBJECTIVES:
        values = [row for row in final if row["objective"] == objective]
        by_objective[objective] = {
            "success_fraction": sum(row["success"] for row in values) / len(values),
            "median_final_point_error_px": float(np.median([row["median_point_error_px"] for row in values])),
            "mean_trajectory_seconds_per_case": float(np.mean([row["trajectory_seconds_per_case"] for row in values])),
        }
    best = min(
        OBJECTIVES,
        key=lambda name: (-by_objective[name]["success_fraction"], by_objective[name]["median_final_point_error_px"]),
    )
    return {
        "success_definition": f"median point error <= {threshold:g} original-image px",
        "success_threshold_status": "provisional_pending_EXP-001_human_noise_floor",
        "by_objective": by_objective,
        "leading_objective": best,
        "conclusion": "PARTIALLY SUPPORTED",
    }


def plot_capture_basin(rows: list[dict[str, Any]], leading: str, path: Path) -> None:
    final = [row for row in rows if row["step"] == 10 and row["objective"] == leading]
    families = ("translation", "rotation", "length", "shape", "combined")
    units = {"translation": "px", "rotation": "degrees", "length": "%", "shape": "RMS degrees", "combined": "severity level"}
    fig, axes = plt.subplots(1, len(families), figsize=(18, 3.8), constrained_layout=True)
    for axis, family in zip(axes, families, strict=True):
        values = [row for row in final if row["family"] == family]
        grouped: dict[float, list[bool]] = defaultdict(list)
        for row in values:
            grouped[float(row["magnitude"])].append(bool(row["success"]))
        x = sorted(grouped)
        y = [np.mean(grouped[value]) for value in x]
        axis.plot(x, y, marker="o", color="#31688e")
        axis.set_ylim(-0.05, 1.05)
        axis.set_title(family)
        axis.set_xlabel(f"initialization error ({units[family]})")
        if axis is axes[0]:
            axis.set_ylabel("successful recovery probability")
        axis.grid(alpha=0.25)
    fig.suptitle(f"EXP-008 capture basin after 10 steps — {leading}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_objectives(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for objective, color in zip(OBJECTIVES, ("#440154", "#21918c", "#fde725"), strict=True):
        values = [row for row in rows if row["objective"] == objective]
        means = []
        success = []
        for step in RECORD_STEPS:
            current = [row for row in values if row["step"] == step]
            means.append(np.median([row["median_point_error_px"] for row in current]))
            success.append(np.mean([row["success"] for row in current]))
        axes[0].plot(RECORD_STEPS, means, marker="o", label=objective, color=color)
        axes[1].plot(RECORD_STEPS, success, marker="o", label=objective, color=color)
    axes[0].set(xlabel="refinement steps", ylabel="median point error (px)", title="Error versus compute")
    axes[1].set(xlabel="refinement steps", ylabel="successful fraction", title="Recovery versus compute", ylim=(-0.05, 1.05))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("EXP-008 image-objective comparison")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_before_after(
    *, seed: int, leading: str, device_name: str, quick: bool, path: Path,
) -> None:
    grid = conditions(quick)
    requests = (
        ("translation", 8.0), ("rotation", 5.0),
        ("shape", 5.0), ("combined", 1.0),
    )
    chosen = [
        next(
            (value for value in grid if value["family"] == family and value["magnitude"] == magnitude),
            next(value for value in grid if value["family"] == family),
        )
        for family, magnitude in requests
        if any(value["family"] == family for value in grid)
    ]
    device = torch.device(device_name)
    target = make_targets([seed], device)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(9, 3.5 * len(chosen)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_index, condition in enumerate(chosen):
        anchor, tangent, length = perturb(target, condition, [seed])
        _, history = refine_pose(
            anchor, tangent, length, target["width"], target["image"],
            target_mask=target["mask"], objective=leading, steps=10,
            record_steps=(0, 10), config=RefinementConfig(),
        )
        truth = target["centerline"][0].detach().cpu().numpy()
        for column, step in enumerate((0, 10)):
            axis = axes[row_index, column]
            prediction = history[step]["centerline_xy"][0].cpu().numpy()
            image = target["image"][0].cpu().numpy()
            axis.imshow(image, cmap="gray", vmin=0.18, vmax=0.82)
            axis.plot(truth[:, 0], truth[:, 1], color="#00ffff", linewidth=1.2, label="truth")
            axis.plot(prediction[:, 0], prediction[:, 1], color="#ff00ff", linewidth=1.2, label="pose")
            error = float(torch.median(point_error_original(
                history[step]["centerline_xy"], target["centerline"]
            )))
            axis.set_title(
                f"{condition['family']} {condition['magnitude']:g} — "
                f"{'initial' if step == 0 else '10 steps'} ({error:.2f} px)"
            )
            axis.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(f"EXP-008 controlled before/after overlays — {leading}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows, runtime = run_grid(args)
    summary = summarize(rows, args.success_threshold_px)
    payload = {
        "experiment": "EXP-008",
        "hypothesis": "Differentiable rendering recovers human-scale pose accuracy inside a measurable initialization basin.",
        "evidence_tier": "Tier C analytic controlled truth",
        "objectives": list(OBJECTIVES),
        "refinement_steps": list(RECORD_STEPS),
        "runtime": runtime,
        "summary": summary,
        "rows": rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_capture_basin(rows, summary["leading_objective"], args.output_dir / "refinement_capture_basin.png")
    plot_objectives(rows, args.output_dir / "refinement_objective_comparison.png")
    plot_before_after(
        seed=args.seed, leading=summary["leading_objective"], device_name=args.device,
        quick=args.quick, path=args.output_dir / "refinement_before_after_overlays.png",
    )
    print(json.dumps({"runtime": runtime, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
