#!/usr/bin/env python3
"""EXP-SMC-006 leave-one-recording-out strict-anchor dynamics diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import h5py
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.latent import (
    decode_centerline,
    encode_centerline,
    orient_to_reference,
    unwrap_latent_rotation,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/smc_exp_006_dynamics_predictability.json"
DEFAULT_OUTPUT = REPO / "experiments/exp_smc_006_dynamics_predictability"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, float | int | None]:
    sample = np.asarray(values, dtype=np.float64)
    if not len(sample):
        return {"n": 0, "median": None, "mean": None, "p95": None, "maximum": None}
    return {
        "n": int(len(sample)),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "p95": float(np.percentile(sample, 95)),
        "maximum": float(np.max(sample)),
    }


def contiguous_runs(accepted: np.ndarray) -> list[np.ndarray]:
    positions = np.flatnonzero(accepted)
    if not len(positions):
        return []
    splits = np.flatnonzero(np.diff(positions) != 1) + 1
    return [run for run in np.split(positions, splits) if len(run)]


def _canonical_first(curve: np.ndarray) -> tuple[np.ndarray, bool]:
    first = tuple(curve[0].tolist())
    last = tuple(curve[-1].tolist())
    return (curve[::-1].copy(), True) if last < first else (curve.copy(), False)


def load_runs(cache: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with h5py.File(cache, "r") as handle:
        if not bool(handle.attrs["complete"]) or handle.attrs["experiment"] != "EXP-SMC-002D":
            raise RuntimeError("anchor cache is incomplete or has the wrong experiment identity")
        if bool(handle.attrs["protected_2025_holdout_opened"]):
            raise RuntimeError("anchor cache reports protected holdout access")
        for recording in sorted(handle.keys()):
            if recording.startswith("2025-"):
                raise RuntimeError("protected recording in anchor cache")
            group = handle[recording]
            accepted = group["accepted"][:]
            frame_index = group["frame_index"][:]
            centerlines = group["centerline_xy"][:]
            recording_runs: list[dict[str, Any]] = []
            for positions in contiguous_runs(accepted):
                curves: list[np.ndarray] = []
                reversals: list[bool] = []
                for offset, position in enumerate(positions):
                    curve = centerlines[position].astype(np.float64)
                    if offset == 0:
                        curve, reversed_order = _canonical_first(curve)
                    else:
                        curve, reversed_order = orient_to_reference(curve, curves[-1])
                    curves.append(curve)
                    reversals.append(reversed_order)
                states: list[np.ndarray] = []
                for curve in curves:
                    state = encode_centerline(curve)
                    if states:
                        state = unwrap_latent_rotation(state, states[-1])
                    states.append(state)
                recording_runs.append({
                    "frames": frame_index[positions].astype(int),
                    "curves": np.stack(curves),
                    "states": np.stack(states),
                    "reversed": reversals,
                })
            result[recording] = recording_runs
    return result


def _shape_training(runs: list[dict[str, Any]], order: int) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for run in runs:
        states = run["states"]
        for index in range(order, len(states)):
            features.append(np.concatenate([states[index - lag - 1, :16] for lag in range(order)]))
            targets.append(states[index, :16])
    if not features:
        return np.empty((0, 16 * order)), np.empty((0, 16))
    return np.stack(features), np.stack(targets)


def fit_diagonal_ar(runs: list[dict[str, Any]], order: int, ridge: float) -> np.ndarray | None:
    features, targets = _shape_training(runs, order)
    if len(features) < order + 2:
        return None
    parameters = np.empty((16, order + 1), dtype=np.float64)
    for dimension in range(16):
        columns = [np.ones(len(features))]
        for lag in range(order):
            columns.append(features[:, lag * 16 + dimension])
        design = np.column_stack(columns)
        penalty = ridge * np.eye(order + 1)
        penalty[0, 0] = 0
        parameters[dimension] = np.linalg.solve(
            design.T @ design + penalty, design.T @ targets[:, dimension]
        )
    return parameters


def predict_state(
    model: str,
    history: np.ndarray,
    horizon: int,
    ar1: np.ndarray | None,
    ar2: np.ndarray | None,
) -> np.ndarray | None:
    current = history[-1].copy()
    previous = history[-2].copy() if len(history) >= 2 else None
    if model == "persistence":
        return current
    if previous is None:
        return None
    velocity = current - previous
    if model == "constant_latent_velocity":
        return current + horizon * velocity
    prediction = current.copy()
    prediction[-4] += horizon * velocity[-4]
    prediction[-2:] += horizon * velocity[-2:]
    prediction[-3] = current[-3]
    if model == "hybrid_global_velocity_shape_hold":
        prediction[:16] = current[:16]
        return prediction
    if model == "shape_ar1":
        if ar1 is None:
            return None
        shape = current[:16].copy()
        for _ in range(horizon):
            shape = ar1[:, 0] + ar1[:, 1] * shape
        prediction[:16] = shape
        return prediction
    if model == "shape_ar2":
        if ar2 is None:
            return None
        first, second = previous[:16].copy(), current[:16].copy()
        for _ in range(horizon):
            first, second = second, ar2[:, 0] + ar2[:, 1] * second + ar2[:, 2] * first
        prediction[:16] = second
        return prediction
    raise ValueError(model)


def curve_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    forward_point = float(np.mean(np.linalg.norm(prediction - target, axis=1)))
    reversed_target = target[::-1]
    reverse_point = float(np.mean(np.linalg.norm(prediction - reversed_target, axis=1)))
    if reverse_point < forward_point:
        target = reversed_target
        point = reverse_point
    else:
        point = forward_point
    pred_angle = np.arctan2(np.diff(prediction, axis=0)[:, 1], np.diff(prediction, axis=0)[:, 0])
    target_angle = np.arctan2(np.diff(target, axis=0)[:, 1], np.diff(target, axis=0)[:, 0])
    difference = np.remainder(pred_angle - target_angle + np.pi, 2 * np.pi) - np.pi
    return point, float(np.rad2deg(np.abs(difference).mean()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    output = args.output.resolve()
    config = json.loads(config_path.read_text())
    anchor_metrics_path = (REPO / config["inputs"]["anchor_density_metrics"]).resolve(strict=True)
    anchor_metrics = json.loads(anchor_metrics_path.read_text())
    cache_path = Path(config["inputs"]["anchor_cache"]).resolve(strict=True)
    representation_path = (REPO / config["inputs"]["representation_metrics"]).resolve(strict=True)
    representation = json.loads(representation_path.read_text())
    if anchor_metrics["experiment"] != "EXP-SMC-002D":
        raise RuntimeError("wrong anchor-density prerequisite")
    if representation["decision"]["selected_shape_coefficients"] != 16:
        raise RuntimeError("wrong latent representation prerequisite")
    allowed_existing = {
        "config.json",
        "notes.md",
        "metrics.json",
        "metrics_unpaired_invalid.json",
        "decision_addendum.json",
        "figures",
    }
    if output.exists() and any(path.name not in allowed_existing for path in output.iterdir()):
        raise FileExistsError("refusing existing EXP-SMC-006 generated output")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    runs = load_runs(cache_path)
    models = config["evaluation"]["models"]
    horizons = list(map(int, config["evaluation"]["horizons_frames"]))
    ridge = float(config["evaluation"]["ridge"])
    cases: list[dict[str, Any]] = []
    fold_diagnostics: dict[str, Any] = {}
    for held_out in sorted(runs):
        training_runs = [run for recording, values in runs.items() if recording != held_out for run in values]
        ar1 = fit_diagonal_ar(training_runs, 1, ridge)
        ar2 = fit_diagonal_ar(training_runs, 2, ridge)
        fold_diagnostics[held_out] = {
            "training_adjacent_pairs": int(sum(max(0, len(run["states"]) - 1) for run in training_runs)),
            "training_triples": int(sum(max(0, len(run["states"]) - 2) for run in training_runs)),
            "ar1_fitted": ar1 is not None,
            "ar2_fitted": ar2 is not None,
        }
        for run_index, run in enumerate(runs[held_out]):
            states, curves, frames = run["states"], run["curves"], run["frames"]
            for start in range(len(states)):
                for horizon in horizons:
                    target_index = start + horizon
                    if target_index >= len(states):
                        continue
                    history = states[: start + 1]
                    for model in models:
                        prediction_state = predict_state(model, history, horizon, ar1, ar2)
                        if prediction_state is None or not np.isfinite(prediction_state).all():
                            continue
                        prediction_curve = decode_centerline(prediction_state)
                        point, tangent = curve_metrics(prediction_curve, curves[target_index])
                        cases.append({
                            "recording": held_out,
                            "run_index": run_index,
                            "start_frame": int(frames[start]),
                            "target_frame": int(frames[target_index]),
                            "horizon_frames": horizon,
                            "model": model,
                            "mean_point_error_px": point,
                            "mean_tangent_error_deg": tangent,
                        })

    aggregate: dict[str, Any] = {}
    for model in models:
        aggregate[model] = {}
        for horizon in horizons:
            selected = [case for case in cases if case["model"] == model and case["horizon_frames"] == horizon]
            aggregate[model][str(horizon)] = {
                "mean_point_error_px": summary([case["mean_point_error_px"] for case in selected]),
                "mean_tangent_error_deg": summary([case["mean_tangent_error_deg"] for case in selected]),
                "by_recording": {
                    recording: {
                        "mean_point_error_px": summary([
                            case["mean_point_error_px"] for case in selected if case["recording"] == recording
                        ]),
                        "mean_tangent_error_deg": summary([
                            case["mean_tangent_error_deg"] for case in selected if case["recording"] == recording
                        ]),
                    }
                    for recording in sorted(runs)
                },
            }
    adjacent = {
        recording: int(sum(max(0, len(run["states"]) - 1) for run in values))
        for recording, values in runs.items()
    }
    horizon5 = {
        recording: int(aggregate["persistence"]["5"]["by_recording"][recording]["mean_point_error_px"]["n"])
        for recording in sorted(runs)
    }
    def prediction_key(case: dict[str, Any]) -> tuple[Any, ...]:
        return (
            case["recording"],
            case["run_index"],
            case["start_frame"],
            case["target_frame"],
            case["horizon_frames"],
        )

    persistence_by_key = {
        prediction_key(case): case
        for case in cases
        if case["model"] == "persistence" and case["horizon_frames"] == 1
    }
    valid_candidates: list[tuple[float, str]] = []
    for model in models[1:]:
        candidate_cases = [
            case for case in cases if case["model"] == model and case["horizon_frames"] == 1
        ]
        paired = [case for case in candidate_cases if prediction_key(case) in persistence_by_key]
        if paired:
            valid_candidates.append((float(np.median([case["mean_point_error_px"] for case in paired])), model))
    selected_model = min(valid_candidates)[1] if valid_candidates else None
    selected_cases = [
        case
        for case in cases
        if case["model"] == selected_model
        and case["horizon_frames"] == 1
        and prediction_key(case) in persistence_by_key
    ]
    selected_h1 = float(np.median([case["mean_point_error_px"] for case in selected_cases])) if selected_cases else None
    paired_persistence = [persistence_by_key[prediction_key(case)] for case in selected_cases]
    persistence_h1 = float(np.median([case["mean_point_error_px"] for case in paired_persistence])) if paired_persistence else None
    improvement = (
        float((persistence_h1 - selected_h1) / persistence_h1)
        if selected_h1 is not None and persistence_h1
        else None
    )
    per_recording_improvement: dict[str, float | None] = {}
    for recording in sorted(runs):
        recording_candidates = [case for case in selected_cases if case["recording"] == recording]
        recording_baselines = [persistence_by_key[prediction_key(case)] for case in recording_candidates]
        baseline = float(np.median([case["mean_point_error_px"] for case in recording_baselines])) if recording_baselines else None
        candidate = float(np.median([case["mean_point_error_px"] for case in recording_candidates])) if recording_candidates else None
        per_recording_improvement[recording] = (
            float((baseline - candidate) / baseline) if baseline and candidate is not None else None
        )
    gate = config["gate"]
    checks = {
        "recordings_with_adjacent_pairs": sum(value > 0 for value in adjacent.values()) >= int(gate["recordings_with_adjacent_pairs_min"]),
        "adjacent_pairs_per_recording": min(adjacent.values()) >= int(gate["adjacent_pairs_per_recording_min"]),
        "horizon5_predictions_per_recording": min(horizon5.values()) >= int(gate["horizon5_predictions_per_recording_min"]),
        "selected_model_horizon1_improvement": improvement is not None and improvement >= float(gate["selected_model_median_horizon1_improvement_over_persistence_min"]),
        "nonnegative_horizon1_improvement_each_recording": all(value is not None and value >= 0 for value in per_recording_improvement.values()),
    }
    passed = all(checks.values())
    decision = {
        "passed": passed,
        "decision": "SUPPORTED" if passed else "NOT_SUPPORTED_SESSION_GENERAL",
        "best_diagnostic_model": selected_model,
        "horizon1_improvement_over_persistence": improvement,
        "horizon1_improvement_by_recording": per_recording_improvement,
        "paired_horizon1_comparison": {
            "n": len(selected_cases),
            "persistence_median_mean_point_error_px": persistence_h1,
            "selected_model_median_mean_point_error_px": selected_h1,
        },
        "checks": checks,
        "consequence": config["fallback_if_not_supported"] if not passed else {"model": selected_model},
    }
    metrics = {
        "schema_version": 1,
        "experiment": "EXP-SMC-006",
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "anchor_density_metrics": str(anchor_metrics_path),
            "anchor_density_metrics_sha256": sha256(anchor_metrics_path),
            "anchor_cache": str(cache_path),
            "anchor_cache_sha256": sha256(cache_path),
            "representation_metrics": str(representation_path),
            "representation_metrics_sha256": sha256(representation_path),
            "protected_2025_holdout_opened": False,
        },
        "evidence_boundary": config["evidence_boundary"],
        "adjacent_pairs_by_recording": adjacent,
        "horizon5_predictions_by_recording": horizon5,
        "fold_diagnostics": fold_diagnostics,
        "aggregate": aggregate,
        "decision": decision,
        "per_prediction": cases,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted_counts: dict[int, int] = {}
    for color, model in zip(colors, models, strict=True):
        x, point, tangent = [], [], []
        for horizon in horizons:
            by_model = {
                candidate: {
                    prediction_key(case): case
                    for case in cases
                    if case["model"] == candidate and case["horizon_frames"] == horizon
                }
                for candidate in models
            }
            common_keys = set.intersection(*(set(values) for values in by_model.values()))
            selected = [by_model[model][key] for key in sorted(common_keys)]
            if selected:
                x.append(horizon)
                point.append(float(np.median([case["mean_point_error_px"] for case in selected])))
                tangent.append(float(np.median([case["mean_tangent_error_deg"] for case in selected])))
                plotted_counts[horizon] = len(selected)
        axes[0].plot(x, point, marker="o", label=model, color=color)
        axes[1].plot(x, tangent, marker="o", label=model, color=color)
    axes[0].set_xlabel("forecast horizon (frames)")
    axes[0].set_ylabel("median mean point error (px)")
    axes[1].set_xlabel("forecast horizon (frames)")
    axes[1].set_ylabel("median mean tangent error (degrees)")
    axes[0].legend(fontsize=7)
    count_text = ", ".join(f"h{horizon}: n={plotted_counts.get(horizon, 0)}" for horizon in horizons)
    fig.suptitle(f"EXP-SMC-006 paired strict-anchor forecasts ({count_text})")
    fig.tight_layout()
    fig.savefig(output / "figures/dynamics_horizon.png", dpi=170)
    plt.close(fig)

    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
