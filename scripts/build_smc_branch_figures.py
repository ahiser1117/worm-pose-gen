#!/usr/bin/env python3
"""Render the complete development-only segmentation-anchored SMC flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "smc_figures"
INPUTS = {
    "seg0": "experiments/exp_smc_001_segmentation_baseline/metrics.json",
    "anchor0": "experiments/exp_smc_002_mask_anchors/metrics.json",
    "seg1": "experiments/exp_smc_001b_hysteresis_terminal_recovery/metrics.json",
    "anchor1": "experiments/exp_smc_002b_width_relative_fov/metrics.json",
    "expert": "experiments/exp_smc_002c_expert_visual_adjudication/adjudication.json",
    "density": "experiments/exp_smc_002d_contiguous_anchor_density/metrics.json",
    "latent": "experiments/exp_smc_003_latent_representation/results/metrics.json",
    "width": "experiments/exp_smc_004_width_model/metrics.json",
    "likelihood": "experiments/exp_smc_005_observation_energy/metrics.json",
    "dynamics": "experiments/exp_smc_006_dynamics_predictability/metrics.json",
    "prior": "experiments/exp_smc_006_dynamics_predictability/decision_addendum.json",
    "smc": "experiments/exp_smc_007_controlled_smc/metrics.json",
    "row22": "experiments/exp_smc_008a_row22_anchor_bracket/metrics.json",
}
COLORS = {
    "supported": "#16856b",
    "limited": "#c48719",
    "failed": "#c43d4b",
    "neutral": "#59636e",
    "text": "#17202a",
}


def _load(relative: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != 1:
        raise ValueError(f"unexpected metrics schema: {path}")
    return document


def _box(axis: Any, x: float, y: float, width: float, height: float,
         title: str, body: str, status: str) -> None:
    color = COLORS[status]
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.6, edgecolor=color, facecolor=color + "14",
    ))
    axis.text(x + 0.014, y + height - 0.035, title, ha="left", va="top",
              fontsize=9.2, weight="bold", color=COLORS["text"])
    axis.text(x + 0.014, y + height - 0.095, body, ha="left", va="top",
              fontsize=7.5, linespacing=1.35, color=COLORS["text"])


def _arrow(axis: Any, start: tuple[float, float], end: tuple[float, float],
           *, dashed: bool = False) -> None:
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.5, color=COLORS["neutral"],
        linestyle="--" if dashed else "-",
    ))


def _validate(values: dict[str, dict[str, Any]]) -> None:
    expected = {
        "expert": "EXP-SMC-002C", "density": "EXP-SMC-002D",
        "latent": "EXP-SMC-003", "width": "EXP-SMC-004",
        "likelihood": "EXP-SMC-005", "dynamics": "EXP-SMC-006",
        "prior": "EXP-SMC-006", "smc": "EXP-SMC-007",
        "row22": "EXP-SMC-008A",
    }
    for key, experiment in expected.items():
        if values[key].get("experiment") != experiment:
            raise ValueError(f"wrong experiment in {INPUTS[key]}")
    protected = [
        values["density"]["protected_2025_holdout_opened"],
        values["latent"]["evidence_boundary"]["protected_2025_holdout_opened"],
        values["width"]["evidence_boundary"]["protected_2025_holdout_opened"],
        values["likelihood"]["evidence_boundary"]["protected_2025_holdout_opened"],
        values["dynamics"]["evidence_boundary"]["protected_2025_holdout_opened"],
        values["smc"]["evidence_boundary"]["protected_2025_holdout_opened"],
        values["row22"]["protected_2025_holdout_opened"],
    ]
    if any(protected):
        raise ValueError("protected holdout must remain closed")


def build(output_dir: Path) -> Path:
    values = {name: _load(path) for name, path in INPUTS.items()}
    _validate(values)
    s0 = values["seg0"]["gate"]["observed"]
    a0 = values["anchor0"]["summary"]
    s1 = values["seg1"]["gate"]["observed"]
    a1 = values["anchor1"]["summary"]
    density = values["density"]
    latent = values["latent"]["decision"]
    width = values["width"]["decision"]["observed"]
    likelihood = values["likelihood"]
    dynamics = values["dynamics"]
    smc = values["smc"]
    row22 = values["row22"]

    figure = plt.figure(figsize=(15.2, 10.2), facecolor="white")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.35, 0.92),
                               hspace=0.22, wspace=0.22)
    figure.suptitle("Segmentation-anchored SMC branch: evidence flow and stopping decision",
                    x=0.055, y=0.985, ha="left", fontsize=16, weight="bold")

    upstream = figure.add_subplot(grid[0, :])
    upstream.set(xlim=(0, 1), ylim=(0, 1)); upstream.axis("off")
    upstream.text(0.01, 0.96, "Frozen proxy gates", fontsize=11, weight="bold", va="top")
    upstream_boxes = [
        ("EXP-SMC-001", f"soft mask\nterminal {s0['median_terminal_containment']:.2f} < 0.90", "failed"),
        ("EXP-SMC-002", "strict anchors\n"
         f"precision {a0['conditional_accepted_precision_proxy']['numerator']}/"
         f"{a0['conditional_accepted_precision_proxy']['denominator']}; "
         f"trunc. reject {a0['truncated_rejected_frames']}/{a0['truncated_frames']}", "failed"),
        ("EXP-SMC-001B", f"hysteresis repair\nterminal still {s1['median_terminal_containment']:.2f}", "failed"),
        ("EXP-SMC-002B", "FOV guard\n"
         f"precision {a1['conditional_accepted_precision_proxy']['numerator']}/"
         f"{a1['conditional_accepted_precision_proxy']['denominator']}; "
         f"trunc. reject {a1['truncated_rejected_frames']}/{a1['truncated_frames']}", "failed"),
        ("EXP-SMC-002C", "expert review\ncontinuation for development only", "limited"),
    ]
    width_box, gap, y, height = 0.174, 0.028, 0.25, 0.52
    for index, (title, body, status) in enumerate(upstream_boxes):
        x = 0.01 + index * (width_box + gap)
        _box(upstream, x, y, width_box, height, title, body, status)
        if index:
            _arrow(upstream, (x - gap + 0.003, y + height / 2),
                   (x - 0.004, y + height / 2), dashed=index == 4)
    upstream.text(0.01, 0.09,
                  "Red = frozen numeric gate failed. Dashed handoff = expert continuation superseded the stop only for bounded development; it did not convert those gates to passes.",
                  fontsize=8.2, color=COLORS["neutral"])

    downstream = figure.add_subplot(grid[1, :])
    downstream.set(xlim=(0, 1), ylim=(0, 1)); downstream.axis("off")
    downstream.text(0.01, 0.98, "Authorized component studies and final natural-bout feasibility test",
                    fontsize=11, weight="bold", va="top")
    boxes = [
        ("EXP-SMC-002D", f"anchor density\n{density['total_strict_accepted']}/{density['total_frames']} = "
         f"{100*density['overall_strict_accepted_density']:.1f}%\npairs 7 / 45 / 0", "failed"),
        ("EXP-SMC-003", f"pose oracle\nK={latent['selected_shape_coefficients']} cubic\n"
         f"{latent['selection_metrics']['per_frame_median_point_distance_px']['median']:.2f} px median", "supported"),
        ("EXP-SMC-004", f"width proxy\nIoU {width['selected_median_mask_iou']:.3f}\n"
         f"10 px shift: −{width['translation_stress_median_iou_drop']:.3f}", "supported"),
        ("EXP-SMC-005", f"likelihood proxy\nsoft Dice selected\n"
         f"near-zero minima {100*likelihood['candidate_summary']['soft_dice']['overall_near_zero_minimum_fraction']:.0f}%", "supported"),
        ("EXP-SMC-006", "natural dynamics\npairs 7 / 45 / 0\nvelocity 25.6% worse", "failed"),
        ("EXP-SMC-007", "synthetic control\n128 particles, T=0.03\n"
         f"SMC {smc['evaluation']['aggregate_by_scenario']['nominal']['methods']['forward_bootstrap_smc_map_genealogy']['trajectory_mean_px']['median']:.2f} px", "limited"),
        ("EXP-SMC-008A", f"row-22 bracket\n{row22['strict_accepted_count']}/{row22['frames_scanned']} anchors\nno natural bout", "failed"),
    ]
    box_width, box_height = 0.126, 0.58
    gap = (0.98 - 7 * box_width) / 6
    for index, (title, body, status) in enumerate(boxes):
        x = 0.01 + index * (box_width + gap)
        _box(downstream, x, 0.26, box_width, box_height, title, body, status)
        if index:
            _arrow(downstream, (x - gap + 0.003, 0.55), (x - 0.004, 0.55))
    downstream.text(0.01, 0.07,
                    "Evidence boundary: 002D/006/008A use bounded 2023 natural frames; 003 is a single-annotator trace oracle; 004/005 use classical masks; 007 uses synthetic truth only.",
                    fontsize=8.2, color=COLORS["neutral"])

    density_axis = figure.add_subplot(grid[2, 0])
    sessions = list(density["sessions"].values())
    labels = [session["recording"][5:] for session in sessions]
    rates = [100 * session["strict_accepted_density"] for session in sessions]
    bars = density_axis.bar(labels, rates, color="#688bb1", width=0.62)
    density_axis.bar_label(bars, labels=[f"{rate:.1f}%" for rate in rates], padding=3, fontsize=8)
    density_axis.set_ylim(0, max(rates) * 1.25)
    density_axis.set_ylabel("strict anchors (% of 101 frames)")
    density_axis.set_title("Natural anchor density is session-dependent", loc="left",
                           fontsize=11, weight="bold")
    density_axis.grid(axis="y", alpha=0.2)
    density_axis.spines[["top", "right"]].set_visible(False)

    comparison = figure.add_subplot(grid[2, 1])
    aggregate = smc["evaluation"]["aggregate_by_scenario"]
    scenario_names = list(aggregate)
    interpolation = [aggregate[name]["methods"]["two_anchor_latent_interpolation"]["trajectory_mean_px"]["median"] for name in scenario_names]
    forward = [aggregate[name]["methods"]["forward_bootstrap_smc_map_genealogy"]["trajectory_mean_px"]["median"] for name in scenario_names]
    reranked = [aggregate[name]["methods"]["terminal_right_anchor_reweighted_genealogy"]["trajectory_mean_px"]["median"] for name in scenario_names]
    x = np.arange(len(scenario_names)); bar_width = 0.25
    comparison.bar(x - bar_width, interpolation, bar_width, label="two-anchor interpolation", color="#c48719")
    comparison.bar(x, forward, bar_width, label="forward SMC", color="#3f7fb0")
    comparison.bar(x + bar_width, reranked, bar_width, label="terminal reranked", color="#7b5aa6")
    comparison.set_xticks(x, ["nom.", "0.5×", "2×", "drop", "width", "FOV"])
    comparison.set_ylabel("median trajectory error (px)")
    comparison.set_title("Synthetic recovery passes absolute gates, not comparison",
                         loc="left", fontsize=11, weight="bold")
    comparison.legend(fontsize=7.5, frameon=False, ncols=3, loc="upper left")
    comparison.grid(axis="y", alpha=0.2)
    comparison.spines[["top", "right"]].set_visible(False)

    figure.legend(
        handles=[Patch(facecolor=COLORS[name] + "28", edgecolor=COLORS[name], label=label)
                 for name, label in (("supported", "supported within proxy/oracle scope"),
                                     ("limited", "development/synthetic only"),
                                     ("failed", "not supported / stop"))],
        loc="lower center", ncols=3, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.text(0.5, 0.055,
                "FINAL STOP: no temporal anchor bracket and no supported natural dynamics → natural-bout SMC is not authorized. Protected 2025 holdout closed.",
                ha="center", fontsize=10.3, weight="bold", color="#8f2330")
    figure.subplots_adjust(left=0.055, right=0.985, top=0.945, bottom=0.105)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "smc_branch_gate_summary.png"
    figure.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
