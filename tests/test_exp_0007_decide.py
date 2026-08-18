import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.exp_0007_decide import build_artifact, decide


CONFIG = yaml.safe_load(Path("configs/spatial_rescue.yaml").read_text())


def passing_evaluation(seed: int, fold: int, digest: str | None = None) -> dict:
    digest = digest or f"{'a' * 60}{seed % 10}{fold}"
    common = {
        "median_point_px": 3.0,
        "p95_point_px": 8.0,
        "mean_angle_degrees": 6.0,
        "p95_frame_angle_degrees": 12.0,
        "mean_endpoint_error_px_each": [3.0, 3.5],
        "median_body_length_error_fraction": 0.03,
        "support_brier": 0.03,
        "support_ece_10_bin": 0.02,
        "point_error_units": "original_image_pixels_968x732",
        "reported_vs_recomputed_in_fov_exact_agreement": True,
        "failed_inference_count": 0,
    }
    tier_c = {**common, "samples": 43, "hidden_mean_point_px": None, "hidden_mean_angle_degrees": None}
    return {
        "experiment": "EXP-0007",
        "variant": "intrinsic",
        "encoder_pool_output": [4, 4],
        "model_seed": seed,
        "data_seed": 20260818,
        "fold": fold,
        "checkpoint_sha256": digest,
        "advancement_scope": "geometry_only_rescue_no_temporal_authorization",
        "tier_B_candidate_proxy": copy.deepcopy(common),
        "tier_C": tier_c,
        "qualitative_review": {
            "checkpoint_sha256": digest,
            "completed": True,
            "systematic_topology_or_shortcut_failure": False,
            "random_overlays_reviewed": True,
            "worst_overlays_reviewed": True,
            "figure_evidence": [
                {"path": f"/tmp/figure-{index}.png", "sha256": "b" * 64}
                for index in range(5)
            ],
        },
    }


def passing_benchmark(evaluation: dict) -> dict:
    summary = {
        "batch_size": 1,
        "p50_milliseconds": 1.0,
        "p95_milliseconds": 1.2,
        "samples_per_second_from_total": 1000.0,
    }
    batched = {**summary, "batch_size": 32, "samples_per_second_from_total": 2300.0}
    return {
        "model_seed": evaluation["model_seed"],
        "fold": evaluation["fold"],
        "variant": "intrinsic",
        "encoder_pool_output": [4, 4],
        "data_seed": 20260818,
        "parameters": 722137,
        "checkpoint": {"path": "/tmp/model.ckpt", "sha256": evaluation["checkpoint_sha256"]},
        "gpu": {
            "logical_device": 0,
            "mapping": {"physical_index": 0, "visible_logical_index": 0},
            "name": "test GPU",
            "cuda_runtime": "13.0",
            "physical_device": {
                "physical_index": 0,
                "uuid": "GPU-test",
                "pci_bus_id": "0000:01:00.0",
                "driver_version": "test",
            },
        },
        "environment": {"python": "3.13", "torch": "2", "lightning": "2"},
        "protocol": {
            "warmup_iterations": 10,
            "measured_iterations": 100,
            "precision": "float32",
            "input": "grayscale uint8 732x968 for preprocessing/end-to-end; float32 [B,1,192,256] in [0,1] for forward-only",
            "cuda_synchronization": "before and after every measured forward/end-to-end iteration",
        },
        "forward_batch1": {**summary, "peak_memory_bytes": 100},
        "forward_batched": {**batched, "peak_memory_bytes": 200},
        "preprocessing_batch1": summary,
        "end_to_end_batch1": summary,
        "preprocessing_batched": batched,
        "end_to_end_batched": batched,
    }


def documents(keys):
    evaluations = [passing_evaluation(seed, fold) for seed, fold in keys]
    return evaluations, [passing_benchmark(item) for item in evaluations]


class Exp0007DecisionTests(unittest.TestCase):
    def test_primary_fold_pass_authorizes_only_primary_seed_folds(self) -> None:
        evaluations, benchmarks = documents([(20260818, 2)])
        result = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(result["decision"], "PRIMARY_FOLD_PASS")
        self.assertTrue(result["authorize_additional_folds"])
        self.assertEqual(result["additional_folds_model_seed"], 20260818)
        self.assertFalse(result["authorize_repeat_seeds"])
        self.assertFalse(result["authorize_temporal_modeling"])

    def test_primary_failure_and_missing_diagnostic_fail_closed(self) -> None:
        evaluations, benchmarks = documents([(20260818, 2)])
        del evaluations[0]["tier_C"]["median_body_length_error_fraction"]
        result = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(result["decision"], "PRIMARY_FOLD_FAIL")
        self.assertFalse(result["authorize_additional_folds"])
        failed = [gate["name"] for gate in result["runs"][0]["gates"] if not gate["pass"]]
        self.assertIn("tier_c_fully_visible.median_body_length_error_fraction", failed)

    def test_checkpoint_binding_failure_is_exact_and_blocks_repeat(self) -> None:
        evaluations, benchmarks = documents([(20260818, fold) for fold in range(3)])
        benchmarks[1]["checkpoint"]["sha256"] = "wrong"
        result = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(result["decision"], "PRIMARY_SEED_FAIL")
        self.assertTrue(result["exact_or_qualitative_failure"])
        self.assertFalse(result["authorize_repeat_seeds"])

    def test_all_primary_near_gate_authorizes_declared_repeats(self) -> None:
        evaluations, benchmarks = documents([(20260818, fold) for fold in range(3)])
        evaluations[0]["tier_C"]["median_point_px"] = 3.9
        result = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(result["decision"], "PRIMARY_SEED_PASS")
        self.assertTrue(result["near_positive_numeric_gate"])
        self.assertTrue(result["authorize_repeat_seeds"])
        self.assertEqual(result["authorized_repeat_model_seeds"], [20260819, 20260820])

    def test_near_numeric_failure_can_repeat_but_far_failure_cannot(self) -> None:
        evaluations, benchmarks = documents([(20260818, fold) for fold in range(3)])
        evaluations[0]["tier_C"]["median_point_px"] = 4.2
        near = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(near["decision"], "PRIMARY_SEED_FAIL")
        self.assertTrue(near["authorize_repeat_seeds"])
        evaluations[0]["tier_C"]["median_point_px"] = 6.0
        far = decide(evaluations, benchmarks, CONFIG)
        self.assertFalse(far["authorize_repeat_seeds"])

    def test_final_repeats_require_every_seed_and_fold_gate(self) -> None:
        keys = [(seed, fold) for seed in (20260818, 20260819, 20260820) for fold in range(3)]
        evaluations, benchmarks = documents(keys)
        passed = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(passed["decision"], "FINAL_PASS")
        self.assertTrue(passed["final_accept_geometry_rescue"])
        evaluations[-1]["tier_B_candidate_proxy"]["support_brier"] = 0.5
        failed = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(failed["decision"], "FINAL_FAIL")
        self.assertFalse(failed["final_accept_geometry_rescue"])

    def test_repeat_fold_pass_authorizes_only_that_repeat_seed(self) -> None:
        evaluations, benchmarks = documents([(20260819, 2)])
        result = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(result["decision"], "REPEAT_FOLD_PASS")
        self.assertTrue(result["authorize_additional_folds"])
        self.assertEqual(result["additional_folds_model_seed"], 20260819)

    def test_primary_all_folds_without_near_gate_is_final_geometry_accept(self) -> None:
        evaluations, benchmarks = documents([(20260818, fold) for fold in range(3)])
        result = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(result["decision"], "PRIMARY_SEED_PASS")
        self.assertFalse(result["authorize_repeat_seeds"])
        self.assertTrue(result["final_accept_geometry_rescue"])

    def test_incomplete_identity_set_and_duplicate_are_rejected(self) -> None:
        evaluations, benchmarks = documents([(20260818, 1), (20260818, 2)])
        result = decide(evaluations, benchmarks, CONFIG)
        self.assertEqual(result["decision"], "INCOMPLETE_FAIL_CLOSED")
        self.assertFalse(result["authorize_additional_folds"])
        with self.assertRaises(ValueError):
            decide([evaluations[0], copy.deepcopy(evaluations[0])], [benchmarks[0]], CONFIG)

    def test_artifact_has_stable_authorization_and_hash_fields(self) -> None:
        evaluation = passing_evaluation(20260818, 2)
        benchmark = passing_benchmark(evaluation)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metric_path, benchmark_path = root / "metrics.json", root / "benchmark.json"
            metric_path.write_text(json.dumps(evaluation))
            benchmark_path.write_text(json.dumps(benchmark))
            artifact = build_artifact(
                [metric_path], [benchmark_path], Path("configs/spatial_rescue.yaml")
            )
        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(artifact["experiment"], "EXP-0007")
        self.assertEqual(artifact["evaluated_model_seeds"], [20260818])
        self.assertEqual(artifact["evaluated_folds_by_seed"], {"20260818": [2]})
        self.assertIn("config_sha256", artifact)
        self.assertIn("code_git_commit", artifact)
        self.assertTrue(artifact["authorize_additional_folds"])


if __name__ == "__main__":
    unittest.main()
