import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.evaluate_exp_004_tier_c import controlled_gate_decision
from scripts.train_exp_004 import _validate, _validate_repeat_authorization
from worm_pose_gen.training_data import (
    MaterializedPoseDataset,
    SyntheticTierCDataset,
    materialized_dataset_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/scientific_exp_004_analytic_5k_control.yaml"


class Exp004ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG_PATH.read_text())

    def test_preregistration_identity_is_exact_and_fail_closed(self) -> None:
        _validate(self.config)
        mutations = (
            ("training", "synthetic_training_samples", 4999),
            ("training", "maximum_steps", 10799),
            ("evidence_boundary", "audited_holdout_allowed", True),
            ("controlled_gate", "median_full_latent_point_distance_px_at_most", 17),
            ("controlled_gate", "Tier_A_tuning_allowed", True),
            ("resources", "physical_device_index", 1),
        )
        for section, name, value in mutations:
            with self.subTest(section=section, name=name):
                changed = copy.deepcopy(self.config)
                changed[section][name] = value
                with self.assertRaisesRegex(RuntimeError, "invalid EXP-004"):
                    _validate(changed)

    def test_frozen_tier_c_hash_and_fully_visible_count(self) -> None:
        training = self.config["training"]
        dataset = MaterializedPoseDataset(
            SyntheticTierCDataset(
                int(training["synthetic_validation_samples"]),
                seed=int(training["data_seed"])
                + 5_000_000
                + int(training["fold"]) * 100_000,
                profile="held_out",
            )
        )
        self.assertEqual(
            materialized_dataset_sha256(dataset),
            self.config["provenance"][
                "expected_tier_c_validation_materialized_sha256"
            ],
        )
        self.assertEqual(
            sum(bool(dataset[index]["image_support_target"].all()) for index in range(len(dataset))),
            43,
        )

    def test_controlled_gate_is_conjunctive_and_finite(self) -> None:
        gate = self.config["controlled_gate"]
        at_threshold = {
            "median_full_latent_point_distance_px": 16.0,
            "median_mean_tangent_error_degrees": 15.0,
            "median_body_length_error_fraction": 0.15,
        }
        self.assertEqual(
            controlled_gate_decision(at_threshold, gate),
            (True, "AUTHORIZE_REPEAT_SEEDS_ONLY"),
        )
        for name in at_threshold:
            with self.subTest(name=name):
                failed = dict(at_threshold)
                failed[name] += 1e-6
                self.assertFalse(controlled_gate_decision(failed, gate)[0])
        nonfinite = dict(at_threshold)
        nonfinite["median_full_latent_point_distance_px"] = math.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            controlled_gate_decision(nonfinite, gate)

    def test_repeat_seeds_require_hash_bound_primary_authorization(self) -> None:
        config_hash = sha256_file(CONFIG_PATH)
        primary = int(self.config["training"]["primary_model_seed"])
        repeat = int(self.config["training"]["repeat_model_seeds"][0])
        self.assertIsNone(
            _validate_repeat_authorization(
                None, config_sha256=config_hash, seed=primary, primary_seed=primary
            )
        )
        with self.assertRaisesRegex(RuntimeError, "repeat seed is closed"):
            _validate_repeat_authorization(
                None, config_sha256=config_hash, seed=repeat, primary_seed=primary
            )
        payload = {
            "experiment": "EXP-004",
            "phase": "analytic_5k_scale_control",
            "model_seed": primary,
            "config_sha256": config_hash,
            "controlled_gate": {
                "passed": True,
                "decision": "AUTHORIZE_REPEAT_SEEDS_ONLY",
            },
            "evidence_boundary": {
                "Tier_A_evaluated": False,
                "protected_holdout_opened": False,
                "repeat_annotations_used": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(payload))
            authorization = _validate_repeat_authorization(
                path, config_sha256=config_hash, seed=repeat, primary_seed=primary
            )
            self.assertEqual(authorization["sha256"], sha256_file(path))
            payload["controlled_gate"]["passed"] = False
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "invalid EXP-004"):
                _validate_repeat_authorization(
                    path, config_sha256=config_hash, seed=repeat, primary_seed=primary
                )


if __name__ == "__main__":
    unittest.main()
