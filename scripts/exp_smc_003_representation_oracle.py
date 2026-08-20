#!/usr/bin/env python3
"""Run the frozen EXP-SMC-003 representation oracle on 17 complete traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_tier_a_primary import _verified_inputs
from exp_005_representation_oracle import (
    NUM_POINTS,
    _canonical_complete_annotations,
    _fit_fixed_basis,
    _pca_fit,
    _summarize,
    cosine_basis,
    cubic_bspline_basis,
    intrinsic_target,
    reconstruct_from_shape,
    reconstruction_metrics,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(row: dict[str, Any], section: str, statistic: str) -> float:
    return float(row["summary"][section][statistic])


def _eligible(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    return (
        _metric(row, "per_frame_median_point_distance_px", "median")
        <= rule["maximum_median_frame_point_error_px"]
        and _metric(row, "per_frame_median_point_distance_px", "p95")
        <= rule["maximum_p95_frame_point_error_px"]
        and _metric(row, "per_frame_mean_tangent_error_deg", "median")
        <= rule["maximum_median_frame_mean_tangent_error_deg"]
        and _metric(row, "per_frame_mean_tangent_error_deg", "p95")
        <= rule["maximum_p95_frame_mean_tangent_error_deg"]
    )


def _plot_capacity(representations: dict[str, list[dict[str, Any]]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    styles = {
        "cubic_tangent_spline": ("#0072b2", "cubic spline"),
        "cosine_tangent": ("#d55e00", "cosine"),
        "pca_tangent_leave_one_recording_out": ("#009e73", "PCA LORO"),
    }
    for family, (color, label) in styles.items():
        rows = representations[family]
        x = [row["shape_coefficients"] for row in rows]
        axes[0].plot(x, [
            _metric(row, "per_frame_median_point_distance_px", "median") for row in rows
        ], marker="o", color=color, label=label)
        axes[1].plot(x, [
            _metric(row, "per_frame_mean_tangent_error_deg", "median") for row in rows
        ], marker="o", color=color, label=label)
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="median gate")
    axes[1].axhline(4.0, color="#555555", linestyle="--", linewidth=1, label="median gate")
    axes[0].set_ylabel("median per-frame point error (px)")
    axes[1].set_ylabel("median per-frame mean tangent error (degrees)")
    for axis in axes:
        axis.set_xlabel("shape coefficients K")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8)
    figure.suptitle("EXP-SMC-003 oracle capacity on 17 complete development traces")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_selected_body_error(selected: dict[str, Any], path: Path) -> None:
    values = selected["summary"]["mean_error_by_body_position_px"]
    figure, axis = plt.subplots(figsize=(7.2, 4.1))
    axis.plot(np.linspace(0, 1, NUM_POINTS), values, color="#0072b2", linewidth=2)
    axis.set(xlabel="normalized body position", ylabel="mean reconstruction error (px)")
    axis.set_title(
        f"Selected {selected['family']} K={selected['shape_coefficients']}: error by body position"
    )
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/exp_smc_003_representation_oracle.json",
    )
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=PROJECT_ROOT / "experiments/exp_smc_003_latent_representation",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("experiment") != "EXP-SMC-003" or config.get("frozen_before_run") is not True:
        raise ValueError("EXP-SMC-003 requires its frozen config")
    evidence = config["evidence"]
    if evidence.get("protected_2025_holdout_opened") is not False:
        raise RuntimeError("protected holdout must remain closed")
    manifest = PROJECT_ROOT / evidence["manifest"]
    annotations = Path(evidence["annotations"])
    for path, expected in (
        (manifest, evidence["manifest_sha256"]),
        (annotations, evidence["annotations_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen input digest mismatch: {path}")
    results_dir = args.experiment_dir / "results"
    figures_dir = args.experiment_dir / "figures"
    if results_dir.exists() or figures_dir.exists():
        raise FileExistsError("refusing to overwrite existing EXP-SMC-003 outputs")

    payload, rows = _verified_inputs(manifest, annotations)
    identities, targets = _canonical_complete_annotations(rows)
    if len(targets) != evidence["required_complete_traces"]:
        raise RuntimeError("complete-trace count changed")
    intrinsic = [intrinsic_target(target) for target in targets]
    shapes = np.asarray([value[0] for value in intrinsic])
    rotations = np.asarray([value[1] for value in intrinsic])
    lengths = np.asarray([value[2] for value in intrinsic])
    coefficient_counts = config["representations"]["shape_coefficients"]
    representations: dict[str, list[dict[str, Any]]] = {
        "cubic_tangent_spline": [],
        "cosine_tangent": [],
        "pca_tangent_leave_one_recording_out": [],
    }
    fixed_basis = {
        "cubic_tangent_spline": cubic_bspline_basis,
        "cosine_tangent": cosine_basis,
    }
    for family, basis_function in fixed_basis.items():
        for count in coefficient_counts:
            cases = _fit_fixed_basis(
                targets, shapes, rotations, lengths,
                basis_function(NUM_POINTS - 1, count),
            )
            representations[family].append({
                "family": family,
                "shape_coefficients": count,
                "external_pose_values": 4,
                "summary": _summarize(cases),
                "per_case": [
                    dict(identity, metrics=metrics)
                    for identity, metrics in zip(identities, cases, strict=True)
                ],
            })

    recordings = sorted({identity["recording"] for identity in identities})
    fold_sizes = {
        recording: sum(identity["recording"] != recording for identity in identities)
        for recording in recordings
    }
    minimum_fold_rank = min(size - 1 for size in fold_sizes.values())
    unsupported_pca: list[dict[str, Any]] = []
    for count in config["representations"]["pca_requested_coefficients"]:
        if count > minimum_fold_rank:
            unsupported_pca.append({
                "shape_coefficients": count,
                "reason": "requested K exceeds at least one leave-one-recording-out training-fold rank",
                "minimum_training_fold_rank": minimum_fold_rank,
            })
            continue
        fitted = np.empty_like(shapes)
        folds: list[dict[str, Any]] = []
        for recording in recordings:
            test = np.asarray([identity["recording"] == recording for identity in identities])
            train = ~test
            fitted[test] = _pca_fit(shapes[train], shapes[test], count)
            folds.append({
                "held_out_recording": recording,
                "train_frames": int(train.sum()),
                "test_frames": int(test.sum()),
                "effective_components": count,
            })
        cases = [
            reconstruction_metrics(
                reconstruct_from_shape(shape, rotation, length, target), target
            )
            for shape, rotation, length, target in zip(
                fitted, rotations, lengths, targets, strict=True
            )
        ]
        representations["pca_tangent_leave_one_recording_out"].append({
            "family": "pca_tangent_leave_one_recording_out",
            "shape_coefficients": count,
            "external_pose_values": 4,
            "folds": folds,
            "summary": _summarize(cases),
        })

    rule = config["decision_rule"]
    candidates = [
        row for family in rule["eligible_families"]
        for row in representations[family]
    ]
    for row in candidates:
        row["passes_frozen_rule"] = _eligible(row, rule)
    eligible = [row for row in candidates if row["passes_frozen_rule"]]
    eligible.sort(key=lambda row: (
        row["shape_coefficients"],
        _metric(row, "per_frame_mean_tangent_error_deg", "median"),
        _metric(row, "per_frame_median_point_distance_px", "median"),
        row["family"],
    ))
    selected = eligible[0] if eligible else None
    decision = {
        "passed": selected is not None,
        "status": "SUPPORTED_ORACLE_ONLY" if selected else "NOT_SUPPORTED",
        "selected_family": selected["family"] if selected else None,
        "selected_shape_coefficients": selected["shape_coefficients"] if selected else None,
        "external_pose_values": 4 if selected else None,
        "selection_metrics": selected["summary"] if selected else None,
        "interpretation": rule["pass_interpretation"] if selected else rule["failure_interpretation"],
        "dynamics_or_smc_authorized": False,
        "authorization_reason": "EXP-SMC-001/002 upstream gates remain NOT_SUPPORTED",
    }
    output = {
        "schema_version": 1,
        "experiment": "EXP-SMC-003",
        "hypothesis": "a compact intrinsic tangent representation reconstructs easy-frame traces below declared oracle tolerances",
        "config": str(args.config.resolve(strict=True)),
        "config_sha256": sha256_file(args.config),
        "inputs": {
            "manifest": str(manifest.resolve(strict=True)),
            "manifest_sha256": sha256_file(manifest),
            "manifest_records_sha256": payload["manifest_records_sha256"],
            "annotations": str(annotations.resolve(strict=True)),
            "annotations_sha256": sha256_file(annotations),
        },
        "case_identity": identities,
        "representations": representations,
        "pca_unsupported": unsupported_pca,
        "minimum_pca_training_fold_rank": minimum_fold_rank,
        "decision_rule": rule,
        "decision": decision,
        "evidence_boundary": {
            "single_annotator_complete_development_traces": len(targets),
            "truncated_traces_excluded": evidence["excluded_truncated_traces"],
            "not_identifiable_traces_excluded": evidence["excluded_not_identifiable_traces"],
            "single_annotator_repeatability_available": False,
            "protected_2025_holdout_opened": False,
            "manual_mask_truth_used": False,
            "image_model_evaluated": False,
        },
        "source_reuse": {
            "reused_math_from": "scripts/exp_005_representation_oracle.py",
            "reused_source_sha256": sha256_file(PROJECT_ROOT / "scripts/exp_005_representation_oracle.py"),
            "old_experiment_outputs_modified": False,
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "seed": config["seed"],
    }
    results_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    (results_dir / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    _plot_capacity(representations, figures_dir / "representation_capacity.png")
    if selected is not None:
        _plot_selected_body_error(selected, figures_dir / "selected_error_by_body_position.png")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
