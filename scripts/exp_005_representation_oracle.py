#!/usr/bin/env python3
"""Fit intrinsic representations directly to complete Tier-A primary traces.

No image model is trained.  This measures optimistic representation capacity on
the 17 fully visible primary annotations and a leave-one-recording-out PCA
generalization diagnostic.  It cannot establish proximity to human precision
until the delayed repeat tranche is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.evaluate_tier_a_primary import _verified_inputs
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from evaluate_tier_a_primary import _verified_inputs
from worm_pose_gen.annotation import ValidatedAnnotation, resample_polyline


SEED = 20260819
NUM_POINTS = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.remainder(value + np.pi, 2 * np.pi) - np.pi


def cubic_bspline_basis(samples: int, coefficients: int) -> np.ndarray:
    """Uniform clamped cubic B-spline design matrix."""

    degree = 3
    if samples < 2 or coefficients < degree + 1:
        raise ValueError("cubic splines need >=2 samples and >=4 coefficients")
    internal = np.linspace(0.0, 1.0, coefficients - degree + 1)[1:-1]
    knots = np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))
    x = np.linspace(0.0, 1.0, samples)
    basis = np.zeros((samples, len(knots) - 1), dtype=np.float64)
    for index in range(len(knots) - 1):
        basis[:, index] = (x >= knots[index]) & (x < knots[index + 1])
    basis[-1, :] = 0.0
    basis[-1, coefficients - 1] = 1.0
    for order in range(1, degree + 1):
        updated = np.zeros((samples, len(knots) - order - 1), dtype=np.float64)
        for index in range(updated.shape[1]):
            left_width = knots[index + order] - knots[index]
            right_width = knots[index + order + 1] - knots[index + 1]
            if left_width > 0:
                updated[:, index] += (
                    (x - knots[index]) / left_width * basis[:, index]
                )
            if right_width > 0:
                updated[:, index] += (
                    (knots[index + order + 1] - x)
                    / right_width
                    * basis[:, index + 1]
                )
        basis = updated
    result = basis[:, :coefficients]
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("B-spline basis lost partition of unity")
    return result


def cosine_basis(samples: int, coefficients: int) -> np.ndarray:
    """Orthonormal nonconstant DCT-II shape basis."""

    if samples < 2 or coefficients < 1 or coefficients >= samples:
        raise ValueError("cosine coefficient count must lie in [1, samples)")
    position = np.arange(samples, dtype=np.float64)[:, None] + 0.5
    frequency = np.arange(1, coefficients + 1, dtype=np.float64)[None, :]
    return np.sqrt(2.0 / samples) * np.cos(np.pi * position * frequency / samples)


def intrinsic_target(points_xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return rotation-free segment angles, global rotation, and arc length."""

    difference = np.diff(points_xy, axis=0)
    length = float(np.linalg.norm(difference, axis=1).sum())
    angle = np.unwrap(np.arctan2(difference[:, 1], difference[:, 0]))
    rotation = float(angle.mean())
    return angle - rotation, rotation, length


def reconstruct_from_shape(
    shape_angle: np.ndarray,
    rotation: float,
    length: float,
    target_xy: np.ndarray,
) -> np.ndarray:
    """Integrate uniform arc steps and optimally translate to the target mean."""

    angle = shape_angle + rotation
    step = length / len(angle)
    difference = step * np.column_stack((np.cos(angle), np.sin(angle)))
    prediction = np.vstack((np.zeros(2), np.cumsum(difference, axis=0)))
    prediction += target_xy.mean(axis=0) - prediction.mean(axis=0)
    return prediction


def reconstruction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    point = np.linalg.norm(prediction - target, axis=1)
    pred_angle = np.arctan2(np.diff(prediction, axis=0)[:, 1], np.diff(prediction, axis=0)[:, 0])
    target_angle = np.arctan2(np.diff(target, axis=0)[:, 1], np.diff(target, axis=0)[:, 0])
    tangent = np.abs(wrap_angle(pred_angle - target_angle))
    pred_turn = wrap_angle(np.diff(np.unwrap(pred_angle)))
    target_turn = wrap_angle(np.diff(np.unwrap(target_angle)))
    curvature = np.abs(wrap_angle(pred_turn - target_turn))
    return {
        "point_distance_px": point.tolist(),
        "tangent_error_deg": np.rad2deg(tangent).tolist(),
        "curvature_turn_error_deg": np.rad2deg(curvature).tolist(),
        "median_point_distance_px": float(np.median(point)),
        "p95_point_distance_px": float(np.percentile(point, 95)),
        "mean_tangent_error_deg": float(np.rad2deg(tangent).mean()),
        "mean_curvature_turn_error_deg": float(np.rad2deg(curvature).mean()),
    }


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(values: list[float]) -> dict[str, float]:
        sample = np.asarray(values, dtype=np.float64)
        return {
            "median": float(np.median(sample)),
            "mean": float(np.mean(sample)),
            "p95": float(np.percentile(sample, 95)),
        }

    return {
        "frames": len(cases),
        "per_frame_median_point_distance_px": summary(
            [case["median_point_distance_px"] for case in cases]
        ),
        "all_body_positions_point_distance_px": summary(
            [value for case in cases for value in case["point_distance_px"]]
        ),
        "per_frame_mean_tangent_error_deg": summary(
            [case["mean_tangent_error_deg"] for case in cases]
        ),
        "per_frame_mean_curvature_turn_error_deg": summary(
            [case["mean_curvature_turn_error_deg"] for case in cases]
        ),
        "mean_error_by_body_position_px": np.mean(
            np.asarray([case["point_distance_px"] for case in cases]), axis=0
        ).tolist(),
    }


def _fit_fixed_basis(
    targets: list[np.ndarray],
    shapes: np.ndarray,
    rotations: np.ndarray,
    lengths: np.ndarray,
    basis: np.ndarray,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for target, shape, rotation, length in zip(
        targets, shapes, rotations, lengths, strict=True
    ):
        coefficient = np.linalg.lstsq(basis, shape, rcond=None)[0]
        prediction = reconstruct_from_shape(basis @ coefficient, rotation, length, target)
        cases.append(reconstruction_metrics(prediction, target))
    return cases


def _pca_fit(train_shapes: np.ndarray, test_shapes: np.ndarray, components: int) -> np.ndarray:
    mean = train_shapes.mean(axis=0)
    _, _, vh = np.linalg.svd(train_shapes - mean, full_matrices=False)
    effective = min(components, vh.shape[0], max(1, len(train_shapes) - 1))
    basis = vh[:effective]
    centered = test_shapes - mean
    return mean + (centered @ basis.T) @ basis


def _canonical_complete_annotations(
    rows: list[tuple[dict[str, Any], ValidatedAnnotation]]
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    complete = [(source, annotation) for source, annotation in rows if annotation.is_complete]
    values: list[dict[str, Any]] = []
    targets: list[np.ndarray] = []
    known_shapes: list[np.ndarray] = []
    pending: list[tuple[int, np.ndarray]] = []
    for source, annotation in complete:
        points = resample_polyline(annotation.points_xy, NUM_POINTS)
        if annotation.head_tail_state == "start_is_tail":
            points = points[::-1].copy()
        shape, _, _ = intrinsic_target(points)
        index = len(targets)
        targets.append(points)
        values.append({
            "sample_id": annotation.sample_id,
            "recording": source["recording"],
            "selection_stratum": source["selection_stratum"],
            "difficulty": list(annotation.difficulty),
            "head_tail_state": annotation.head_tail_state,
            "ambiguous_orientation_reversed": False,
        })
        if annotation.head_tail_state == "ambiguous":
            pending.append((index, points))
        else:
            known_shapes.append(shape)
    reference = np.mean(known_shapes, axis=0)
    for index, points in pending:
        forward, _, _ = intrinsic_target(points)
        reverse, _, _ = intrinsic_target(points[::-1])
        if np.linalg.norm(reverse - reference) < np.linalg.norm(forward - reference):
            targets[index] = points[::-1].copy()
            values[index]["ambiguous_orientation_reversed"] = True
    return values, targets


def _plot_capacity(representations: dict[str, Any], path: Path) -> None:
    families = {
        "cubic_tangent_spline": ("#0072b2", "cubic tangent spline"),
        "cosine_tangent": ("#d55e00", "cosine tangent"),
        "pca_tangent_in_sample": ("#009e73", "PCA tangent (in-sample)")
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for family, (color, label) in families.items():
        rows = representations[family]
        x = [row["shape_coefficients"] for row in rows]
        y = [row["summary"]["per_frame_median_point_distance_px"]["median"] for row in rows]
        angle = [row["summary"]["per_frame_mean_tangent_error_deg"]["median"] for row in rows]
        axes[0].plot(x, y, marker="o", color=color, label=label)
        axes[1].plot(x, angle, marker="o", color=color, label=label)
    axes[0].set_ylabel("median per-frame point error (px)")
    axes[1].set_ylabel("median per-frame mean tangent error (degrees)")
    for axis in axes:
        axis.set_xlabel("shape coefficients")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Coordinate reconstruction floor")
    axes[1].set_title("Tangent reconstruction floor")
    axes[0].legend(fontsize=8)
    fig.suptitle("EXP-005 oracle fits on 17 complete primary traces")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_body_position(representations: dict[str, Any], path: Path) -> None:
    selections = [
        ("cubic_tangent_spline", 16, "spline 16", "#0072b2"),
        ("cosine_tangent", 16, "cosine 16", "#d55e00"),
        ("pca_tangent_in_sample", 8, "PCA 8 in-sample", "#009e73"),
    ]
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    for family, count, label, color in selections:
        row = next(
            item for item in representations[family]
            if item["shape_coefficients"] == count
        )
        axis.plot(
            np.linspace(0, 1, NUM_POINTS),
            row["summary"]["mean_error_by_body_position_px"],
            label=label,
            color=color,
        )
    axis.set_xlabel("normalized body position (orientation aligned where known)")
    axis.set_ylabel("mean reconstruction error (px)")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend()
    axis.set_title("Representation-floor error by body position")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite an existing oracle experiment")
    payload, rows = _verified_inputs(args.manifest, args.annotations)
    case_identity, targets = _canonical_complete_annotations(rows)
    if len(targets) != 17:
        raise RuntimeError(f"frozen primary tranche expected 17 complete traces, got {len(targets)}")
    intrinsic = [intrinsic_target(target) for target in targets]
    shapes = np.asarray([value[0] for value in intrinsic])
    rotations = np.asarray([value[1] for value in intrinsic])
    lengths = np.asarray([value[2] for value in intrinsic])

    representations: dict[str, Any] = {
        "direct_100_point_coordinates": {
            "coordinate_values": 200,
            "summary": _summarize([reconstruction_metrics(target, target) for target in targets]),
            "interpretation": "identity control; no compression",
        },
        "full_99_tangent_angles": {
            "shape_coefficients": 99,
            "external_pose_values": 4,
            "summary": _summarize(_fit_fixed_basis(
                targets, shapes, rotations, lengths, np.eye(NUM_POINTS - 1)
            )),
            "interpretation": "numerical integration control with uniform arc steps",
        },
        "cubic_tangent_spline": [],
        "cosine_tangent": [],
        "pca_tangent_in_sample": [],
        "pca_tangent_leave_one_recording_out": [],
    }
    for count in (8, 12, 16, 24, 32):
        cases = _fit_fixed_basis(
            targets, shapes, rotations, lengths,
            cubic_bspline_basis(NUM_POINTS - 1, count),
        )
        representations["cubic_tangent_spline"].append({
            "shape_coefficients": count,
            "external_pose_values": 4,
            "summary": _summarize(cases),
            "per_case": [dict(identity, metrics=metrics) for identity, metrics in zip(case_identity, cases, strict=True)],
        })
    for count in (4, 8, 16, 24, 32, 72):
        cases = _fit_fixed_basis(
            targets, shapes, rotations, lengths,
            cosine_basis(NUM_POINTS - 1, count),
        )
        representations["cosine_tangent"].append({
            "shape_coefficients": count,
            "external_pose_values": 4,
            "summary": _summarize(cases),
            "per_case": [dict(identity, metrics=metrics) for identity, metrics in zip(case_identity, cases, strict=True)],
        })
    for count in (4, 8, 12, 16):
        fitted = _pca_fit(shapes, shapes, count)
        cases = [
            reconstruction_metrics(
                reconstruct_from_shape(shape, rotation, length, target), target
            )
            for shape, rotation, length, target in zip(
                fitted, rotations, lengths, targets, strict=True
            )
        ]
        representations["pca_tangent_in_sample"].append({
            "shape_coefficients": count,
            "external_pose_values": 4,
            "basis_training": "same_17_complete_primary_traces_optimistic_oracle",
            "summary": _summarize(cases),
        })
    recordings = sorted({value["recording"] for value in case_identity})
    for count in (4, 8):
        fitted = np.empty_like(shapes)
        fold_details = []
        for recording in recordings:
            test = np.asarray([value["recording"] == recording for value in case_identity])
            train = ~test
            fitted[test] = _pca_fit(shapes[train], shapes[test], count)
            fold_details.append({
                "held_out_recording": recording,
                "train_frames": int(train.sum()),
                "test_frames": int(test.sum()),
                "effective_components": min(count, int(train.sum()) - 1),
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
            "shape_coefficients": count,
            "external_pose_values": 4,
            "folds": fold_details,
            "summary": _summarize(cases),
        })

    metrics = {
        "schema_version": 1,
        "experiment": "EXP-005-representation-oracle-primary17",
        "hypothesis": (
            "a compact intrinsic representation can reconstruct complete Tier-A "
            "traces without sacrificing local geometry"
        ),
        "evidence_scope": (
            "17 complete primary traces from one annotator; oracle fitting only; "
            "no network and no human repeatability threshold"
        ),
        "inputs": {
            "manifest": str(args.manifest.resolve(strict=True)),
            "manifest_sha256": sha256_file(args.manifest),
            "manifest_records_sha256": payload["manifest_records_sha256"],
            "annotations": str(args.annotations.resolve(strict=True)),
            "annotations_sha256": sha256_file(args.annotations),
        },
        "case_identity": case_identity,
        "representations": representations,
        "evidence_boundary": {
            "protected_holdout_opened": False,
            "truncated_traces_excluded": 12,
            "not_identifiable_traces_excluded": 1,
            "human_precision_comparison_deferred_until_repeat_pass": True,
            "pca_16_is_in_sample_and_rank_limited_by_17_frames": True,
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip(),
        "seed": SEED,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _plot_capacity(representations, args.output_dir / "representation_capacity.png")
    _plot_body_position(representations, args.output_dir / "representation_error_by_body_position.png")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "complete_traces": len(targets),
        "summaries": {
            name: [
                {
                    "shape_coefficients": row["shape_coefficients"],
                    "median_point_px": row["summary"]["per_frame_median_point_distance_px"]["median"],
                    "median_mean_tangent_deg": row["summary"]["per_frame_mean_tangent_error_deg"]["median"],
                }
                for row in value
            ]
            for name, value in representations.items()
            if isinstance(value, list)
        },
        "protected_holdout_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
