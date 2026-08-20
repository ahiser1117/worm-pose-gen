#!/usr/bin/env python3
"""Aggregate the frozen 3-architecture x 3-seed EXP-003 Tier-A comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.evaluate_tier_a_primary import _read_frames, _verified_inputs, sha256_file
except ModuleNotFoundError:
    from evaluate_tier_a_primary import _read_frames, _verified_inputs, sha256_file


VARIANTS = (
    "global_intrinsic_budget_matched",
    "dense_centerline_field",
    "anchored_intrinsic_grid",
)
SEEDS = (20260819, 20260820, 20260821)
COLORS = {
    "global_intrinsic_budget_matched": "#cc79a7",
    "dense_centerline_field": "#0072b2",
    "anchored_intrinsic_grid": "#009e73",
}
LABELS = {
    "global_intrinsic_budget_matched": "global intrinsic",
    "dense_centerline_field": "dense field",
    "anchored_intrinsic_grid": "anchored grid",
}


def _complete_values(document: dict[str, Any], metric: str) -> list[float]:
    return [
        float(case["complete_metrics"][metric])
        for case in document["per_case"]
        if case["complete_metrics"] is not None
    ]


def _visible_values(document: dict[str, Any], metric: str) -> list[float]:
    return [
        float(case["visible_metrics"][metric])
        for case in document["per_case"]
        if case["visible_metrics"] is not None
    ]


def _paired_bootstrap_difference(
    reference: list[float], candidate: list[float], *, seed: int, resamples: int = 2000
) -> dict[str, float]:
    first = np.asarray(reference)
    second = np.asarray(candidate)
    if first.shape != second.shape:
        raise RuntimeError("paired bootstrap inputs changed shape")
    generator = np.random.default_rng(seed)
    draw = generator.integers(0, len(first), size=(resamples, len(first)))
    difference = np.median(second[draw], axis=1) - np.median(first[draw], axis=1)
    return {
        "candidate_minus_global_median_px": float(np.median(second) - np.median(first)),
        "bootstrap_p2_5_px": float(np.percentile(difference, 2.5)),
        "bootstrap_p97_5_px": float(np.percentile(difference, 97.5)),
    }


def _plot_performance(documents: dict[tuple[str, int], dict], path: Path) -> None:
    metrics = (
        ("complete", "median point error (px)"),
        ("tangent", "mean tangent error (degrees)"),
        ("visible", "truncated visible distance (px)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for axis, (kind, ylabel) in zip(axes, metrics, strict=True):
        for index, variant in enumerate(VARIANTS):
            values = []
            for seed in SEEDS:
                document = documents[(variant, seed)]
                if kind == "complete":
                    value = np.median(_complete_values(document, "median_point_distance_px"))
                elif kind == "tangent":
                    value = np.median(_complete_values(document, "mean_tangent_error_deg"))
                else:
                    value = np.median(_visible_values(document, "median_visible_trace_distance_px"))
                values.append(value)
            jitter = np.linspace(-0.08, 0.08, len(values))
            axis.scatter(index + jitter, values, color=COLORS[variant], s=42, zorder=3)
            axis.plot([index - 0.18, index + 0.18], [np.median(values)] * 2,
                      color=COLORS[variant], lw=2.5)
        axis.set_xticks(range(len(VARIANTS)), [LABELS[value] for value in VARIANTS], rotation=16)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("17 complete traces")
    axes[1].set_title("17 complete traces")
    axes[2].set_title("12 truncated traces")
    fig.suptitle("EXP-003 Tier-A primary comparison: three fixed seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_body_position(documents: dict[tuple[str, int], dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.4))
    position = np.linspace(0, 1, 100)
    for variant in VARIANTS:
        arrays = []
        for seed in SEEDS:
            arrays.extend(
                case["complete_metrics"]["point_distance_px"]
                for case in documents[(variant, seed)]["per_case"]
                if case["complete_metrics"] is not None
            )
        values = np.asarray(arrays)
        axis.plot(position, values.mean(0), color=COLORS[variant], label=LABELS[variant])
        low, high = np.percentile(values, (25, 75), axis=0)
        axis.fill_between(position, low, high, color=COLORS[variant], alpha=0.13)
    axis.set_xlabel("normalized body position (orientation-symmetric alignment)")
    axis.set_ylabel("point distance (px), mean with interquartile band")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend()
    axis.set_title("EXP-003 complete-trace error by body position")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_overlays(
    documents: dict[tuple[str, int], dict],
    manifest: Path,
    annotations: Path,
    path: Path,
) -> None:
    _, rows = _verified_inputs(manifest, annotations)
    frames = _read_frames(rows)
    row_map = {annotation.sample_id: (source, annotation) for source, annotation in rows}
    primary = {variant: documents[(variant, SEEDS[0])] for variant in VARIANTS}
    global_cases = primary[VARIANTS[0]]["per_case"]
    def score(case: dict[str, Any]) -> float:
        if case["complete_metrics"]:
            return float(case["complete_metrics"]["median_point_distance_px"])
        if case["visible_metrics"]:
            return float(case["visible_metrics"]["median_visible_trace_distance_px"])
        return -1.0
    selected = sorted(global_cases, key=score, reverse=True)[:12]
    predictions = {
        variant: {
            case["sample_id"]: np.asarray(case["prediction_centerline_xy"])
            for case in document["per_case"]
        }
        for variant, document in primary.items()
    }
    fig, axes = plt.subplots(3, 4, figsize=(14, 9.5))
    for axis, case in zip(axes.flat, selected, strict=True):
        sample_id = case["sample_id"]
        source, annotation = row_map[sample_id]
        frame = frames[sample_id]
        lo, hi = np.percentile(frame, (1, 99))
        axis.imshow(frame, cmap="gray", vmin=lo, vmax=max(lo + 1, hi))
        if len(annotation.points_xy):
            axis.plot(annotation.points_xy[:, 0], annotation.points_xy[:, 1],
                      color="#00ffff", lw=1.8, label="manual visible")
        for variant in VARIANTS:
            curve = predictions[variant][sample_id]
            axis.plot(curve[:, 0], curve[:, 1], color=COLORS[variant], lw=1.15,
                      label=LABELS[variant])
        axis.set_title(f"{sample_id}\n{annotation.trace_state}; {source['selection_stratum']}", fontsize=8)
        axis.set_xlim(0, frame.shape[1]); axis.set_ylim(frame.shape[0], 0); axis.axis("off")
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("EXP-003 primary-seed overlays on globally difficult Tier-A cases")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", nargs=9, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite aggregate EXP-003 evidence")
    documents: dict[tuple[str, int], dict] = {}
    run_documents: dict[tuple[str, int], dict] = {}
    input_identity = []
    for path in args.evaluations:
        document = json.loads(path.read_text())
        key = (str(document["variant"]), int(document["model_seed"]))
        if key in documents or key[0] not in VARIANTS or key[1] not in SEEDS:
            raise RuntimeError(f"unexpected/duplicate EXP-003 evaluation identity: {key}")
        if document.get("evaluation") != "Tier-A-primary30":
            raise RuntimeError("aggregate input is not a Tier-A primary evaluation")
        run_path = Path(document["run_metrics"])
        if sha256_file(run_path) != document["run_metrics_sha256"]:
            raise RuntimeError("an EXP-003 run-metrics file changed after evaluation")
        run = json.loads(run_path.read_text())
        if (
            run.get("status") != "TRAINED_PENDING_EVALUATION"
            or (run.get("variant"), int(run.get("model_seed", -1))) != key
            or run.get("checkpoint_sha256") != document.get("checkpoint_sha256")
        ):
            raise RuntimeError("evaluation identity does not match its training run")
        documents[key] = document
        run_documents[key] = run
        input_identity.append({"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)})
    if set(documents) != {(variant, seed) for variant in VARIANTS for seed in SEEDS}:
        raise RuntimeError("aggregate requires the exact 3-architecture x 3-seed matrix")
    config_hashes = {value["config_sha256"] for value in documents.values()}
    annotation_hashes = {value["annotations_sha256"] for value in documents.values()}
    if len(config_hashes) != 1 or len(annotation_hashes) != 1:
        raise RuntimeError("EXP-003 evaluation inputs do not share frozen config/annotations")
    dataset_hash_sets = {
        json.dumps(value["materialized_dataset_sha256"], sort_keys=True)
        for value in run_documents.values()
    }
    source_hash_sets = {
        json.dumps(value["source_sha256"], sort_keys=True)
        for value in run_documents.values()
    }
    if len(dataset_hash_sets) != 1:
        raise RuntimeError("paired EXP-003 runs did not use byte-identical materialized datasets")
    if len(source_hash_sets) != 1:
        raise RuntimeError("paired EXP-003 runs did not use identical training source")
    order_hashes_by_seed = {
        seed: {
            run_documents[(variant, seed)]["training_order_sha256"]
            for variant in VARIANTS
        }
        for seed in SEEDS
    }
    if any(len(values) != 1 for values in order_hashes_by_seed.values()):
        raise RuntimeError("paired architectures did not share training order within each seed")

    runs: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    paired: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            document = documents[(variant, seed)]
            point = float(np.median(_complete_values(document, "median_point_distance_px")))
            tangent = float(np.median(_complete_values(document, "mean_tangent_error_deg")))
            visible = float(np.median(_visible_values(document, "median_visible_trace_distance_px")))
            runs[variant].append({
                "model_seed": seed,
                "complete_trace_median_point_px": point,
                "complete_trace_median_mean_tangent_deg": tangent,
                "truncated_trace_median_visible_distance_px": visible,
                "algorithmic_failure_frames": document["summary"]["algorithmic_failure_frames"],
                "checkpoint_sha256": document["checkpoint_sha256"],
            })
    for seed in SEEDS:
        reference = documents[(VARIANTS[0], seed)]
        reference_values = _complete_values(reference, "median_point_distance_px")
        reference_median = float(np.median(reference_values))
        for variant in VARIANTS[1:]:
            candidate = documents[(variant, seed)]
            candidate_values = _complete_values(candidate, "median_point_distance_px")
            candidate_median = float(np.median(candidate_values))
            improvement = 1.0 - candidate_median / reference_median
            paired.append({
                "model_seed": seed,
                "candidate_variant": variant,
                "global_median_point_px": reference_median,
                "candidate_median_point_px": candidate_median,
                "improvement_fraction": improvement,
                "passes_frozen_30_percent_gate": bool(
                    improvement >= 0.30
                    and candidate["summary"]["algorithmic_failure_frames"]
                    <= reference["summary"]["algorithmic_failure_frames"]
                ),
                "paired_frame_bootstrap": _paired_bootstrap_difference(
                    reference_values, candidate_values,
                    seed=seed + (1 if variant == VARIANTS[1] else 2),
                ),
            })
    aggregate = {
        "schema_version": 1,
        "experiment": "EXP-003",
        "evaluation": "Tier-A-primary30-three-architecture-three-seed",
        "inputs": input_identity,
        "config_sha256": next(iter(config_hashes)),
        "annotations_sha256": next(iter(annotation_hashes)),
        "paired_training_provenance": {
            "materialized_dataset_sha256": next(
                iter(run_documents.values())
            )["materialized_dataset_sha256"],
            "training_source_sha256": next(iter(run_documents.values()))["source_sha256"],
            "training_order_sha256_by_seed": {
                str(seed): next(iter(values))
                for seed, values in order_hashes_by_seed.items()
            },
            "all_runs_used_physical_gpu_0": all(
                value["gpu"]["name"] == "NVIDIA RTX 6000 Ada Generation"
                for value in run_documents.values()
            ),
        },
        "runs": runs,
        "paired_spatial_vs_global": paired,
        "gate_summary": {
            variant: {
                "seeds_passing": sum(
                    value["passes_frozen_30_percent_gate"]
                    for value in paired if value["candidate_variant"] == variant
                ),
                "required_seeds": 3,
            }
            for variant in VARIANTS[1:]
        },
        "evidence_boundary": {
            "protected_holdout_opened": False,
            "primary_Tier_A_used_for_gradients": False,
            "repeat_annotations_used": False,
            "quality_scores_calibrated": False,
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    _plot_performance(documents, args.output_dir / "architecture_comparison.png")
    _plot_body_position(documents, args.output_dir / "error_by_body_position.png")
    _plot_overlays(
        documents, args.manifest, args.annotations,
        args.output_dir / "primary_seed_overlays.png",
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "runs": runs,
        "gate_summary": aggregate["gate_summary"],
        "protected_holdout_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
