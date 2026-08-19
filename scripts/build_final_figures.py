#!/usr/bin/env python3
"""Build the canonical, explicitly negative-result project figures.

This script reads only repository evidence, controlled Tier-C samples, and the
rejected EXP-0007 checkpoint.  It never opens the source recordings or the
audited holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

from worm_pose_gen.geometry import tangent_angles, wrap_angle
from worm_pose_gen.model import WormProposalModule
from worm_pose_gen.renderer import render_worm
from worm_pose_gen.synthetic import (
    SyntheticConfig,
    generate_synthetic_pose,
    moving_crop_sequence,
    original_to_render,
)
from worm_pose_gen.training_data import SyntheticTierCDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIER_C_SEED = 20260818 + 5_000_000 + 2 * 100_000
ORIGINAL_SCALE = torch.tensor((968 / 256, 732 / 192), dtype=torch.float32)
COLORS = {
    "accepted": "#2a9d8f",
    "rejected": "#d1495b",
    "revised": "#e9c46a",
    "evidence": "#457b9d",
    "target": "#00a896",
    "prediction": "#e63946",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align_case(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support_probability: torch.Tensor,
    support_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction_original = prediction * ORIGINAL_SCALE
    target_original = target * ORIGINAL_SCALE
    forward = torch.linalg.vector_norm(prediction_original - target_original, dim=-1)
    reverse = torch.linalg.vector_norm(prediction_original - target_original.flip(0), dim=-1)
    reverse_selected = bool(reverse.mean() < forward.mean())
    chosen_target = target.flip(0) if reverse_selected else target
    chosen_support = support_target.flip(0) if reverse_selected else support_target
    chosen_original = chosen_target * ORIGINAL_SCALE
    point = torch.linalg.vector_norm(prediction_original - chosen_original, dim=-1)
    angle = (
        wrap_angle(tangent_angles(prediction_original) - tangent_angles(chosen_original)).abs()
        * 180
        / torch.pi
    )
    return {
        "target": chosen_target,
        "support": chosen_support.bool(),
        "support_probability": support_probability,
        "point": point,
        "angle": angle,
        "prediction_angle": tangent_angles(prediction_original),
        "target_angle": tangent_angles(chosen_original),
    }


@torch.inference_mode()
def evaluate_tier_c(module: WormProposalModule) -> list[dict[str, torch.Tensor | int]]:
    dataset = SyntheticTierCDataset(128, seed=TIER_C_SEED, profile="held_out")
    results: list[dict[str, torch.Tensor | int]] = []
    for start in range(0, len(dataset), 32):
        samples = [dataset[index] for index in range(start, min(start + 32, len(dataset)))]
        images = torch.stack([sample["image"] for sample in samples])
        output = module(images)
        for offset, sample in enumerate(samples):
            aligned = align_case(
                output["centerline_xy"][offset].cpu(),
                sample["centerline_xy"].cpu(),
                output["image_support_probability"][offset].cpu(),
                sample["image_support_target"].cpu(),
            )
            results.append(
                {
                    "index": start + offset,
                    "sample_seed": int(sample["sample_seed"]),
                    "image": sample["image"][0].cpu(),
                    "prediction": output["centerline_xy"][offset].cpu(),
                    **aligned,
                }
            )
    return results


def plot_flow(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 7.2))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 8)
    axis.axis("off")
    boxes = [
        (0.4, 6.2, 2.7, 1.0, "Audit + frozen split\n4/12 readable", "evidence"),
        (3.7, 6.2, 2.7, 1.0, "EXP-0001/2\nproxy + controlled truth", "accepted"),
        (7.0, 6.2, 2.7, 1.0, "EXP-0003/5\ncrop designs", "rejected"),
        (10.3, 6.2, 2.9, 1.0, "EXP-0006\nbalanced crop artifact", "accepted"),
        (2.0, 3.6, 3.2, 1.15, "EXP-0004 coordinates\nzigzag topology\nREJECT", "rejected"),
        (5.8, 3.6, 3.2, 1.15, "EXP-0004 intrinsic\n116.92 px / 23.35°\nREVISE", "revised"),
        (9.6, 3.6, 3.2, 1.15, "EXP-0007 4×4 rescue\n87.54 px / 27.68°\nREJECT", "rejected"),
        (4.8, 1.0, 4.4, 1.25, "SCIENTIFIC STOP\nNo reliable proposal; holdout preserved\nTemporal/refinement/uncertainty blocked", "rejected"),
    ]
    for x, y, width, height, label, status in boxes:
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=COLORS[status], edgecolor="white", linewidth=1.5, alpha=0.95,
        )
        axis.add_patch(box)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", color="white", fontsize=10, weight="bold")
    arrows = [
        ((3.1, 6.7), (3.7, 6.7)), ((6.4, 6.7), (7.0, 6.7)), ((9.7, 6.7), (10.3, 6.7)),
        ((5.05, 6.2), (3.6, 4.75)), ((5.05, 6.2), (7.4, 4.75)),
        ((5.2, 4.15), (5.8, 4.15)), ((9.0, 4.15), (9.6, 4.15)),
        ((11.2, 3.6), (8.6, 2.25)),
    ]
    for start, stop in arrows:
        axis.annotate("", xy=stop, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#263238"})
    axis.text(7, 7.7, "Worm-pose research decision path", ha="center", va="center", fontsize=18, weight="bold")
    axis.text(7, 0.45, "Green = accepted evidence component · yellow = revised · red = rejected/blocked", ha="center", fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_representative(results: list[dict], output: Path) -> None:
    visible = [case for case in results if bool(case["support"].all())]
    med = float(torch.tensor([case["point"].median() for case in visible]).median())
    case = min(visible, key=lambda value: abs(float(value["point"].median()) - med))
    figure, axis = plt.subplots(figsize=(8, 5.8))
    axis.imshow(case["image"], cmap="gray", vmin=0, vmax=1)
    axis.plot(case["target"][:, 0], case["target"][:, 1], color=COLORS["target"], lw=2.2, label="controlled target")
    axis.plot(case["prediction"][:, 0], case["prediction"][:, 1], color=COLORS["prediction"], lw=2.2, label="EXP-0007 prediction")
    axis.scatter(case["target"][[0, -1], 0], case["target"][[0, -1], 1], color=COLORS["target"], s=25)
    axis.set_title(f"Representative fully-visible Tier C failure\nmedian point error {float(case['point'].median()):.1f} original-image px")
    axis.legend(loc="lower right")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_failure_montage(results: list[dict], output: Path) -> None:
    worst = sorted(results, key=lambda value: float(value["point"].mean()), reverse=True)[:12]
    figure, axes = plt.subplots(3, 4, figsize=(13, 9.4))
    for axis, case in zip(axes.flat, worst, strict=True):
        axis.imshow(case["image"], cmap="gray", vmin=0, vmax=1)
        axis.plot(case["target"][:, 0], case["target"][:, 1], color=COLORS["target"], lw=1.3)
        axis.plot(case["prediction"][:, 0], case["prediction"][:, 1], color=COLORS["prediction"], lw=1.3)
        fraction = 1.0 - float(case["support"].float().mean())
        axis.set_title(f"mean {float(case['point'].mean()):.0f}px · hidden {fraction:.0%}", fontsize=9)
        axis.axis("off")
    figure.suptitle("EXP-0007 worst controlled cases: systematic location/scale/shape shortcut", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_error_by_body(results: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    s = np.linspace(0, 1, 100)
    for label, subset, color in (
        ("fully visible", [case for case in results if bool(case["support"].all())], COLORS["evidence"]),
        ("artificial crop", [case for case in results if not bool(case["support"].all())], COLORS["rejected"]),
    ):
        points = torch.stack([case["point"] for case in subset]).numpy()
        angles = torch.stack([case["angle"] for case in subset]).numpy()
        for axis, values, ylabel in ((axes[0], points, "point error (original-image px)"), (axes[1], angles, "absolute tangent error (degrees)")):
            median = np.median(values, axis=0)
            low, high = np.quantile(values, (0.25, 0.75), axis=0)
            axis.plot(s, median, color=color, lw=2, label=label)
            axis.fill_between(s, low, high, color=color, alpha=0.18)
            axis.set_xlabel("normalized body position")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0].axhline(4, ls="--", color="black", lw=1, label="fully-visible median gate")
    axes[1].axhline(8, ls="--", color="black", lw=1, label="fully-visible mean gate")
    for axis in axes:
        axis.legend(fontsize=8)
    figure.suptitle("Errors remain large across the body, including both endpoints")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_crop_robustness(results: list[dict], output: Path) -> None:
    fractions = [0.05, 0.10, 0.20, 0.30, 0.40]
    visible_error, hidden_error, visible_angle, hidden_angle = [], [], [], []
    cropped = [case for case in results if not bool(case["support"].all())]
    for fraction_index, _ in enumerate(fractions):
        group = [case for case in cropped if int(case["index"]) % 5 == fraction_index]
        visible_error.append(float(torch.cat([case["point"][case["support"]] for case in group]).mean()))
        hidden_error.append(float(torch.cat([case["point"][~case["support"]] for case in group]).mean()))
        visible_angle.append(float(torch.cat([case["angle"][case["support"]] for case in group]).mean()))
        hidden_angle.append(float(torch.cat([case["angle"][~case["support"]] for case in group]).mean()))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, observed, inferred, ylabel in (
        (axes[0], visible_error, hidden_error, "mean point error (original-image px)"),
        (axes[1], visible_angle, hidden_angle, "mean tangent error (degrees)"),
    ):
        axis.plot(np.array(fractions) * 100, observed, "o-", lw=2, color=COLORS["evidence"], label="image-supported body")
        axis.plot(np.array(fractions) * 100, inferred, "o-", lw=2, color=COLORS["rejected"], label="hidden body")
        axis.set_xlabel("controlled hidden fraction (%)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Single-frame EXP-0007 crop robustness (Tier C controlled truth)")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_support_calibration(results: list[dict], output: Path) -> None:
    probability = torch.cat([case["support_probability"] for case in results])
    target = torch.cat([case["support"] for case in results]).float()
    bins = torch.clamp((probability * 10).long(), max=9)
    confidence, accuracy, counts = [], [], []
    for index in range(10):
        mask = bins == index
        if bool(mask.any()):
            confidence.append(float(probability[mask].mean()))
            accuracy.append(float(target[mask].mean()))
            counts.append(int(mask.sum()))
    brier = float((probability - target).square().mean())
    ece = sum(count / len(probability) * abs(conf - acc) for conf, acc, count in zip(confidence, accuracy, counts, strict=True))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].plot([0, 1], [0, 1], "--", color="black", lw=1)
    axes[0].plot(confidence, accuracy, "o-", color=COLORS["evidence"], lw=2)
    axes[0].set(xlabel="predicted image-support probability", ylabel="empirical support frequency", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_title(f"Support calibration only\nECE={ece:.3f}, Brier={brier:.3f}")
    axes[0].grid(alpha=0.25)
    axes[1].bar(range(len(counts)), counts, color=COLORS["evidence"])
    axes[1].set(xlabel="occupied probability bin", ylabel="body-point count", title="Calibration evidence distribution")
    figure.suptitle("This is not pose or angle uncertainty calibration")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def plot_angle_heatmap(module: WormProposalModule, output: Path) -> None:
    config = SyntheticConfig()
    pose = generate_synthetic_pose(20260818, config, profile="held_out")
    sequence = moving_crop_sequence(pose["centerline_xy"], hidden_end="tail", config=config)
    images, targets = [], []
    for index, (camera, support) in enumerate(zip(sequence["centerline_camera_xy"], sequence["support_mask"], strict=True)):
        generator = torch.Generator().manual_seed(90_000 + index)
        rendered = render_worm(
            original_to_render(camera, config),
            pose["width_profile_render"].float(),
            config.render_height,
            config.render_width,
            foreground=0.2,
            background=0.8,
            noise=torch.randn((config.render_height, config.render_width), generator=generator) * 0.01,
            image_support_target=support,
        )["image"].float()
        images.append(rendered)
        targets.append(original_to_render(camera, config).float())
    batch = torch.stack(images)[:, None]
    predictions = module(batch)
    target_angle, prediction_angle, aligned_support = [], [], []
    for index, target in enumerate(targets):
        aligned = align_case(
            predictions["centerline_xy"][index].cpu(), target,
            predictions["image_support_probability"][index].cpu(), sequence["support_mask"][index].cpu(),
        )
        target_angle.append(aligned["target_angle"])
        prediction_angle.append(aligned["prediction_angle"])
        aligned_support.append(aligned["support"])
    target_values = torch.stack(target_angle)
    predicted_values = torch.stack(prediction_angle)
    error = wrap_angle(predicted_values - target_values).abs() * 180 / torch.pi
    support = torch.stack(aligned_support)
    extent = (0, 1, len(images) - 1, 0)
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    images_to_plot = [target_values, predicted_values, error]
    titles = ["controlled target angle", "EXP-0007 predicted angle", "absolute circular error"]
    cmaps = ["twilight", "twilight", "magma"]
    ranges = [(-torch.pi, torch.pi), (-torch.pi, torch.pi), (0, 90)]
    for axis, values, title, cmap, limits in zip(axes, images_to_plot, titles, cmaps, ranges, strict=True):
        plotted = axis.imshow(values.numpy(), aspect="auto", extent=extent, cmap=cmap, vmin=float(limits[0]), vmax=float(limits[1]))
        axis.set_title(title)
        axis.set_xlabel("normalized body position")
        figure.colorbar(plotted, ax=axis, shrink=0.8, label="radians" if "angle" in title else "degrees")
    axes[0].contour(np.linspace(0, 1, 100), np.arange(len(images)), (~support).numpy(), levels=[0.5], colors="white", linewidths=1)
    axes[0].set_ylabel("moving-camera frame (5% → 40% tail hidden)")
    figure.suptitle("Controlled moving-FOV body-angle heatmap; no temporal model claim")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_pareto(output: Path) -> None:
    variants = [
        ("EXP-0004\ncoordinates", 208.6642, 2417.93, 46.49),
        ("EXP-0004\nintrinsic 2×2", 116.9192, 2320.38, 23.35),
        ("EXP-0007\nintrinsic 4×4", 87.5404, 2461.33, 27.68),
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.6))
    for label, error, fps, angle in variants:
        axis.scatter(error, fps, s=130, color=COLORS["rejected"], edgecolor="white", linewidth=1.5)
        axis.annotate(
            f"{label}\n{angle:.1f}°", (error, fps), xytext=(0, -12),
            textcoords="offset points", ha="center", va="top", fontsize=9,
        )
    axis.axvspan(0, 4, color=COLORS["accepted"], alpha=0.12, label="fully-visible median gate ≤4 px")
    axis.axhline(20, color="black", ls="--", lw=1, label="acquisition-rate floor (20 fps)")
    axis.set(xlabel="fully-visible Tier C median point error (original-image px; lower is better)", ylabel="batch-32 end-to-end samples/s (higher is better)")
    axis.set_title("Accuracy–throughput frontier: every proposal is fast, none is reliable", pad=12)
    axis.set_ylim(-100, 2800)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/final_figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = WormProposalModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    module.eval()
    if module.variant != "intrinsic" or tuple(module.hparams.encoder_pool_output) != (4, 4):
        raise RuntimeError("final figures require the frozen rejected EXP-0007 checkpoint")
    results = evaluate_tier_c(module)
    outputs = {
        "experiment_flow_overview.png": lambda path: plot_flow(path),
        "representative_overlay.png": lambda path: plot_representative(results, path),
        "failure_montage.png": lambda path: plot_failure_montage(results, path),
        "error_by_body_position.png": lambda path: plot_error_by_body(results, path),
        "crop_robustness.png": lambda path: plot_crop_robustness(results, path),
        "support_calibration.png": lambda path: plot_support_calibration(results, path),
        "body_angle_heatmap.png": lambda path: plot_angle_heatmap(module, path),
        "accuracy_throughput_pareto.png": lambda path: plot_pareto(path),
    }
    for name, builder in outputs.items():
        builder(args.output_dir / name)
    manifest = {
        "schema_version": 1,
        "scientific_status": "negative_result_no_accepted_model",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_recordings_opened": False,
        "audited_holdout_opened": False,
        "figures": {name: sha256_file(args.output_dir / name) for name in outputs},
    }
    (args.output_dir / "final_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
