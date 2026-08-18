#!/usr/bin/env python3
"""Deterministic, fail-closed EXP-0007 decision and expansion engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


SCHEMA_VERSION = 1
PRIMARY_SEED = 20260818
REPEAT_SEEDS = (20260819, 20260820)
FOLDS = (0, 1, 2)
PRIMARY_FOLD = 2
EXP4_PRIMARY_MEDIAN_PX = 116.92
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate(name: str, value: Any, threshold: Any, passed: bool, kind: str) -> dict[str, Any]:
    return {"name": name, "kind": kind, "value": value, "threshold": threshold, "pass": bool(passed)}


def _maximum(name: str, values: dict, field: str, maximum: float) -> dict:
    value = values.get(field)
    return _gate(name, value, maximum, isinstance(value, (int, float)) and value <= maximum, "maximum")


def _minimum(name: str, value: Any, minimum: float) -> dict:
    return _gate(name, value, minimum, isinstance(value, (int, float)) and value >= minimum, "minimum")


def _exact(name: str, value: Any, expected: Any) -> dict:
    return _gate(name, value, expected, value == expected, "exact")


def _endpoint_gates(prefix: str, values: dict, maximum: float) -> list[dict]:
    endpoints = values.get("mean_endpoint_error_px_each")
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        return [_gate(f"{prefix}.mean_endpoint_error_px_each", endpoints, [maximum, maximum], False, "maximum_each")]
    return [
        _gate(
            f"{prefix}.mean_endpoint_error_px_each[{index}]",
            value,
            maximum,
            isinstance(value, (int, float)) and value <= maximum,
            "maximum",
        )
        for index, value in enumerate(endpoints)
    ]


def _qualitative_gates(metrics: dict) -> list[dict]:
    review = metrics.get("qualitative_review")
    if not isinstance(review, dict):
        return [_exact("qualitative_review", review, "completed_without_systematic_failure")]
    return [
        _exact("qualitative_review.completed", review.get("completed"), True),
        _exact(
            "qualitative_review.systematic_topology_or_shortcut_failure",
            review.get("systematic_topology_or_shortcut_failure"),
            False,
        ),
        _exact("qualitative_review.random_overlays_reviewed", review.get("random_overlays_reviewed"), True),
        _exact("qualitative_review.worst_overlays_reviewed", review.get("worst_overlays_reviewed"), True),
        _exact(
            "qualitative_review.figure_evidence",
            isinstance(review.get("figure_evidence"), list)
            and len(review["figure_evidence"]) >= 5
            and all(
                isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str)
                for item in review["figure_evidence"]
            ),
            True,
        ),
    ]


def _stratum_gates(prefix: str, values: Any, decision: dict, candidate: bool) -> list[dict]:
    if not isinstance(values, dict):
        return [_exact(prefix, values, "required metric object")]
    if candidate:
        names = {
            "median_point_px": "candidate_proxy_median_point_px",
            "p95_point_px": "candidate_proxy_p95_point_px",
            "mean_angle_degrees": "candidate_proxy_mean_angle_degrees",
            "p95_frame_angle_degrees": "candidate_proxy_p95_frame_angle_degrees",
            "median_body_length_error_fraction": "candidate_proxy_median_body_length_error_fraction",
            "support_brier": "candidate_proxy_support_brier",
            "support_ece_10_bin": "candidate_proxy_support_ece",
        }
        endpoint_key = "candidate_proxy_mean_endpoint_px_each"
    else:
        names = {
            "median_point_px": "reliability_tier_c_median_point_px",
            "p95_point_px": "reliability_tier_c_p95_point_px",
            "mean_angle_degrees": "reliability_tier_c_mean_angle_degrees",
            "p95_frame_angle_degrees": "reliability_tier_c_p95_frame_angle_degrees",
            "median_body_length_error_fraction": "reliability_tier_c_median_body_length_error_fraction",
            "support_brier": "reliability_tier_c_support_brier",
            "support_ece_10_bin": "reliability_tier_c_support_ece",
        }
        endpoint_key = "reliability_tier_c_mean_endpoint_px_each"
    gates = [
        _maximum(f"{prefix}.{field}", values, field, float(decision[config_key]))
        for field, config_key in names.items()
    ]
    gates.extend(_endpoint_gates(prefix, values, float(decision[endpoint_key])))
    gates.extend(
        [
            _exact(f"{prefix}.point_error_units", values.get("point_error_units"), "original_image_pixels_968x732"),
            _exact(
                f"{prefix}.reported_vs_recomputed_in_fov_exact_agreement",
                values.get("reported_vs_recomputed_in_fov_exact_agreement"),
                True,
            ),
            _exact(f"{prefix}.failed_inference_count", values.get("failed_inference_count"), 0),
        ]
    )
    if not candidate:
        gates.extend(
            [
                _exact(f"{prefix}.samples", values.get("samples"), 43),
                _exact(f"{prefix}.hidden_mean_point_px", values.get("hidden_mean_point_px"), None),
                _exact(f"{prefix}.hidden_mean_angle_degrees", values.get("hidden_mean_angle_degrees"), None),
            ]
        )
    return gates


def _benchmark_gates(benchmark: dict, config: dict) -> tuple[list[dict], float | None, float | None]:
    protocol = benchmark.get("protocol") if isinstance(benchmark.get("protocol"), dict) else {}
    forward1 = benchmark.get("forward_batch1") if isinstance(benchmark.get("forward_batch1"), dict) else {}
    forwardn = benchmark.get("forward_batched") if isinstance(benchmark.get("forward_batched"), dict) else {}
    prep1 = benchmark.get("preprocessing_batch1") if isinstance(benchmark.get("preprocessing_batch1"), dict) else {}
    end1 = benchmark.get("end_to_end_batch1") if isinstance(benchmark.get("end_to_end_batch1"), dict) else {}
    prepn = benchmark.get("preprocessing_batched") if isinstance(benchmark.get("preprocessing_batched"), dict) else {}
    endn = benchmark.get("end_to_end_batched") if isinstance(benchmark.get("end_to_end_batched"), dict) else {}
    gpu = benchmark.get("gpu") if isinstance(benchmark.get("gpu"), dict) else {}
    checkpoint = benchmark.get("checkpoint") if isinstance(benchmark.get("checkpoint"), dict) else {}
    environment = benchmark.get("environment") if isinstance(benchmark.get("environment"), dict) else {}
    throughput = endn.get("samples_per_second_from_total")
    reference = float(config["benchmark"]["reference_batched_samples_per_second"])
    ratio = throughput / reference if isinstance(throughput, (int, float)) else None
    offline_fps = endn.get("samples_per_second_from_total")
    gates = [
        _minimum("benchmark.throughput_ratio", ratio, float(config["decision"]["minimum_throughput_ratio_vs_exp_0004_intrinsic"])),
        _minimum("benchmark.end_to_end_offline_fps", offline_fps, float(config["benchmark"]["minimum_end_to_end_offline_fps"])),
        _exact("benchmark.forward_batched.batch_size", forwardn.get("batch_size"), int(config["benchmark"]["batch_size"])),
        _exact("benchmark.protocol.measured_iterations", protocol.get("measured_iterations"), int(config["benchmark"]["iterations"])),
        _exact("benchmark.protocol.precision", protocol.get("precision"), "float32"),
        _exact("benchmark.protocol.cuda_synchronization", protocol.get("cuda_synchronization"), "before and after every measured forward/end-to-end iteration"),
        _exact("benchmark.gpu.logical_device", gpu.get("logical_device"), 0),
        _exact("benchmark.gpu.physical_index", (gpu.get("physical_device") or {}).get("physical_index"), 0),
        _exact("benchmark.gpu.mapping.visible_logical_index", (gpu.get("mapping") or {}).get("visible_logical_index"), 0),
        _exact("benchmark.gpu.mapping.physical_index", (gpu.get("mapping") or {}).get("physical_index"), 0),
        _exact("benchmark.variant", benchmark.get("variant"), "intrinsic"),
        _exact("benchmark.encoder_pool_output", benchmark.get("encoder_pool_output"), [4, 4]),
        _exact("benchmark.data_seed", benchmark.get("data_seed"), int(config["data_seed"])),
        _exact(
            "benchmark.parameters_within_ceiling",
            isinstance(benchmark.get("parameters"), int)
            and 0 < benchmark["parameters"] <= int(config["model"]["maximum_parameters"]),
            True,
        ),
        _exact("benchmark.protocol.warmup_iterations", protocol.get("warmup_iterations"), 10),
        _exact(
            "benchmark.protocol.input",
            protocol.get("input"),
            "grayscale uint8 732x968 for preprocessing/end-to-end; float32 [B,1,192,256] in [0,1] for forward-only",
        ),
    ]
    for name, section, expected_batch in (
        ("forward_batch1", forward1, 1),
        ("forward_batched", forwardn, int(config["benchmark"]["batch_size"])),
        ("preprocessing_batch1", prep1, 1),
        ("end_to_end_batch1", end1, 1),
        ("preprocessing_batched", prepn, int(config["benchmark"]["batch_size"])),
        ("end_to_end_batched", endn, int(config["benchmark"]["batch_size"])),
    ):
        gates.append(_exact(f"benchmark.{name}.batch_size", section.get("batch_size"), expected_batch))
        gates.append(_exact(f"benchmark.{name}.p50_milliseconds_present", isinstance(section.get("p50_milliseconds"), (int, float)), True))
        gates.append(_exact(f"benchmark.{name}.p95_milliseconds_present", isinstance(section.get("p95_milliseconds"), (int, float)), True))
    gates.extend(
        [
            _exact("benchmark.forward_batch1.peak_memory_bytes_present", isinstance(forward1.get("peak_memory_bytes"), (int, float)), True),
            _exact("benchmark.forward_batched.peak_memory_bytes_present", isinstance(forwardn.get("peak_memory_bytes"), (int, float)), True),
            _exact("benchmark.parameters_present", isinstance(benchmark.get("parameters"), int) and benchmark["parameters"] > 0, True),
            _exact("benchmark.checkpoint.sha256_present", isinstance(checkpoint.get("sha256"), str), True),
            _exact("benchmark.checkpoint.path_present", isinstance(checkpoint.get("path"), str), True),
            _exact("benchmark.environment.python_present", isinstance(environment.get("python"), str), True),
            _exact("benchmark.environment.torch_present", isinstance(environment.get("torch"), str), True),
            _exact("benchmark.environment.lightning_present", isinstance(environment.get("lightning"), str), True),
            _exact("benchmark.gpu.name_present", isinstance(gpu.get("name"), str), True),
            _exact("benchmark.gpu.cuda_runtime_present", isinstance(gpu.get("cuda_runtime"), str), True),
            _exact("benchmark.gpu.physical_uuid_present", isinstance((gpu.get("physical_device") or {}).get("uuid"), str), True),
            _exact("benchmark.gpu.physical_pci_present", isinstance((gpu.get("physical_device") or {}).get("pci_bus_id"), str), True),
            _exact("benchmark.gpu.driver_present", isinstance((gpu.get("physical_device") or {}).get("driver_version"), str), True),
        ]
    )
    return gates, ratio, offline_fps


def evaluate_run(metrics: dict, benchmark: dict, config: dict) -> dict[str, Any]:
    model_seed = metrics.get("model_seed")
    fold = metrics.get("fold")
    gates = [
        _exact("evaluation.experiment", metrics.get("experiment"), "EXP-0007"),
        _exact("evaluation.variant", metrics.get("variant"), "intrinsic"),
        _exact("evaluation.encoder_pool_output", metrics.get("encoder_pool_output"), [4, 4]),
        _exact("evaluation.data_seed", metrics.get("data_seed"), int(config["data_seed"])),
        _exact("evaluation.advancement_scope", metrics.get("advancement_scope"), "geometry_only_rescue_no_temporal_authorization"),
        _exact("benchmark.model_seed", benchmark.get("model_seed"), model_seed),
        _exact("benchmark.fold", benchmark.get("fold"), fold),
        _exact("benchmark.checkpoint_binding", (benchmark.get("checkpoint") or {}).get("sha256"), metrics.get("checkpoint_sha256")),
        _exact(
            "qualitative_review.checkpoint_binding",
            (metrics.get("qualitative_review") or {}).get("checkpoint_sha256"),
            metrics.get("checkpoint_sha256"),
        ),
    ]
    gates.extend(_stratum_gates("candidate_proxy", metrics.get("tier_B_candidate_proxy"), config["decision"], True))
    gates.extend(_stratum_gates("tier_c_fully_visible", metrics.get("tier_C"), config["decision"], False))
    gates.extend(_qualitative_gates(metrics))
    benchmark_gates, throughput_ratio, offline_fps = _benchmark_gates(benchmark, config)
    gates.extend(benchmark_gates)
    if model_seed in (PRIMARY_SEED, *REPEAT_SEEDS) and fold == PRIMARY_FOLD:
        tier_c = metrics.get("tier_C") if isinstance(metrics.get("tier_C"), dict) else {}
        median = tier_c.get("median_point_px")
        improvement = (
            (EXP4_PRIMARY_MEDIAN_PX - median) / EXP4_PRIMARY_MEDIAN_PX
            if isinstance(median, (int, float))
            else None
        )
        gates.append(
            _minimum(
                "primary_fold.improvement_fraction_vs_exp0004_intrinsic",
                improvement,
                float(config["decision"]["minimum_primary_fold_median_improvement_fraction_vs_exp_0004_intrinsic"]),
            )
        )
    numeric_gates = [gate for gate in gates if gate["kind"] in ("minimum", "maximum")]
    missing_numeric_evidence = any(not isinstance(gate["value"], (int, float)) for gate in numeric_gates)
    exact_failure = (
        any(not gate["pass"] and gate["kind"] == "exact" for gate in gates)
        or missing_numeric_evidence
    )
    return {
        "model_seed": model_seed,
        "fold": fold,
        "checkpoint_sha256": metrics.get("checkpoint_sha256"),
        "throughput_ratio": throughput_ratio,
        "end_to_end_offline_fps": offline_fps,
        "gates": gates,
        "all_gates_pass": all(gate["pass"] for gate in gates),
        "exact_or_qualitative_failure": exact_failure,
        "numeric_gates": numeric_gates,
    }


def _near_numeric_gate(run: dict, fraction: float) -> bool:
    for gate in run["numeric_gates"]:
        value, threshold = gate["value"], gate["threshold"]
        if isinstance(value, (int, float)) and isinstance(threshold, (int, float)) and threshold > 0:
            if abs(value - threshold) / abs(threshold) <= fraction:
                return True
    return False


def _far_failed_numeric_gate(run: dict, fraction: float) -> bool:
    for gate in run["numeric_gates"]:
        value, threshold = gate["value"], gate["threshold"]
        if gate["pass"]:
            continue
        if not (
            isinstance(value, (int, float))
            and isinstance(threshold, (int, float))
            and threshold > 0
            and abs(value - threshold) / abs(threshold) <= fraction
        ):
            return True
    return False


def decide(metrics_documents: list[dict], benchmark_documents: list[dict], config: dict) -> dict[str, Any]:
    if config.get("experiment") != "EXP-0007":
        raise ValueError("decision engine requires EXP-0007 config")
    if (
        int(config.get("model_seed", -1)) != PRIMARY_SEED
        or tuple(int(value) for value in config["decision"].get("repeat_model_seeds", []))
        != REPEAT_SEEDS
        or tuple(int(value) for value in config["training"].get("folds", [])) != FOLDS
    ):
        raise ValueError("decision engine constants differ from the frozen EXP-0007 config")
    metrics_index: dict[tuple[int, int], dict] = {}
    benchmark_index: dict[tuple[int, int], dict] = {}
    for document in metrics_documents:
        key = (document.get("model_seed"), document.get("fold"))
        if key in metrics_index:
            raise ValueError(f"duplicate evaluation identity {key}")
        metrics_index[key] = document
    for document in benchmark_documents:
        key = (document.get("model_seed"), document.get("fold"))
        if key in benchmark_index:
            raise ValueError(f"duplicate benchmark identity {key}")
        benchmark_index[key] = document
    if set(metrics_index) != set(benchmark_index):
        raise ValueError("evaluation and benchmark identities must match exactly")
    keys = set(metrics_index)
    single_fold_sets = [{(seed, PRIMARY_FOLD)} for seed in (PRIMARY_SEED, *REPEAT_SEEDS)]
    primary_all = {(PRIMARY_SEED, fold) for fold in FOLDS}
    final_all = {(seed, fold) for seed in (PRIMARY_SEED, *REPEAT_SEEDS) for fold in FOLDS}
    recognized = keys in (*single_fold_sets, primary_all, final_all)
    runs = [
        evaluate_run(metrics_index[key], benchmark_index[key], config)
        for key in sorted(keys)
    ]
    near = any(
        _near_numeric_gate(run, float(config["decision"]["near_gate_relative_band_fraction"]))
        for run in runs
    )
    exact_failure = any(run["exact_or_qualitative_failure"] for run in runs)
    far_numeric_failure = any(
        _far_failed_numeric_gate(
            run, float(config["decision"]["near_gate_relative_band_fraction"])
        )
        for run in runs
    )
    all_pass = recognized and all(run["all_gates_pass"] for run in runs)
    authorize_additional_folds = False
    authorize_repeat_seeds = False
    final_accept = False
    if not recognized:
        decision = "INCOMPLETE_FAIL_CLOSED"
    elif len(keys) == 1:
        only_seed = next(iter(keys))[0]
        label = "PRIMARY_FOLD" if only_seed == PRIMARY_SEED else "REPEAT_FOLD"
        decision = f"{label}_PASS" if all_pass else f"{label}_FAIL"
        authorize_additional_folds = all_pass
    elif keys == primary_all:
        decision = "PRIMARY_SEED_PASS" if all_pass else "PRIMARY_SEED_FAIL"
        authorize_repeat_seeds = near and not exact_failure and not far_numeric_failure
        final_accept = all_pass and not near
    else:
        decision = "FINAL_PASS" if all_pass else "FINAL_FAIL"
        final_accept = all_pass
    return {
        "runs": runs,
        "input_identity_set_recognized": recognized,
        "near_positive_numeric_gate": near,
        "exact_or_qualitative_failure": exact_failure,
        "decision": decision,
        "authorize_additional_folds": authorize_additional_folds,
        "additional_folds_model_seed": (
            next(iter(keys))[0] if authorize_additional_folds else None
        ),
        "authorize_repeat_seeds": authorize_repeat_seeds,
        "authorized_repeat_model_seeds": list(REPEAT_SEEDS) if authorize_repeat_seeds else [],
        "final_accept_geometry_rescue": final_accept,
        "authorize_temporal_modeling": False,
    }


def build_artifact(
    metric_paths: list[Path],
    benchmark_paths: list[Path],
    config_path: Path,
    review_paths: list[Path] | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    metrics = [json.loads(path.read_text()) for path in metric_paths]
    benchmarks = [json.loads(path.read_text()) for path in benchmark_paths]
    reviews = [json.loads(path.read_text()) for path in (review_paths or [])]
    if reviews:
        review_index = {(item.get("model_seed"), item.get("fold")): item for item in reviews}
        if len(review_index) != len(reviews):
            raise ValueError("duplicate qualitative-review identity")
        metric_keys = {(item.get("model_seed"), item.get("fold")) for item in metrics}
        if set(review_index) != metric_keys:
            raise ValueError("qualitative-review identities must match evaluations exactly")
        for item in metrics:
            item["qualitative_review"] = review_index[
                (item.get("model_seed"), item.get("fold"))
            ]
        for review in reviews:
            for figure in review.get("figure_evidence", []):
                if sha256_file(Path(figure["path"])) != figure.get("sha256"):
                    raise RuntimeError("qualitative-review figure hash mismatch")
    result = decide(metrics, benchmarks, config)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    source_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts/evaluate.py",
        PROJECT_ROOT / "scripts/benchmark_model.py",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "EXP-0007 deterministic decision",
        "experiment": "EXP-0007",
        "config_path": str(config_path.resolve(strict=True)),
        "config_sha256": sha256_file(config_path),
        "code_git_commit": git_commit,
        "evaluated_model_seeds": sorted({int(item["model_seed"]) for item in result["runs"]}),
        "evaluated_folds_by_seed": {
            str(seed): sorted(
                int(item["fold"]) for item in result["runs"] if int(item["model_seed"]) == seed
            )
            for seed in sorted({int(item["model_seed"]) for item in result["runs"]})
        },
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in source_paths},
        "inputs": {
            "evaluations": [{"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)} for path in metric_paths],
            "benchmarks": [{"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)} for path in benchmark_paths],
            "qualitative_reviews": [
                {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}
                for path in (review_paths or [])
            ],
        },
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--benchmarks", type=Path, nargs="+", required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite decision artifact: {args.output}")
    artifact = build_artifact(
        args.metrics, args.benchmarks, args.config, review_paths=args.reviews
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
