import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
import json

import lightning as L
from lightning.pytorch.callbacks import Timer
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from scripts.evaluate import (
    aggregate_results,
    aligned_geometry_metrics,
    evaluate,
    expected_calibration_error,
    summarize_cases,
)
from scripts.exp_0007_baseline import tensor_sha256, validate_exp_0007_config
from scripts.train import (
    FullyVisibleCountContract,
    ImmutableStepCheckpoint,
    StatefulFixedBatchSampler,
    TrainingBatchSamplerState,
    checkpoint_training_elapsed_seconds,
    fixed_training_order,
    resolve_protocol,
    resolve_resume_checkpoint,
)
from worm_pose_gen.model import WormProposalModule


class _TinyDataset(Dataset):
    def __len__(self): return 2
    def __getitem__(self, index):
        x = torch.linspace(20, 220, 100)
        y = 96 + 20 * torch.sin(torch.linspace(0, 4, 100))
        return {"image": torch.zeros(1, 192, 256), "centerline_xy": torch.stack((x, y), -1),
                "image_support_target": torch.ones(100, dtype=torch.bool),
                "sample_seed": index, "record": "synthetic", "frame_index": -1}


class ModelTests(unittest.TestCase):
    def test_step_checkpoint_and_visible_count_callbacks_are_exact(self) -> None:
        class _Trainer:
            global_step = 300
            sanity_checking = False
            callback_metrics = {"val_tier_c_fully_visible_count": torch.tensor(43.0)}

            @staticmethod
            def save_checkpoint(path: Path) -> None:
                path.write_bytes(b"checkpoint")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step300.ckpt"
            callback = ImmutableStepCheckpoint(path, 300)
            callback.on_train_batch_end(_Trainer(), None, None, None, 0)
            self.assertEqual(path.read_bytes(), b"checkpoint")
            with self.assertRaises(FileExistsError):
                callback.on_train_batch_end(_Trainer(), None, None, None, 0)
        FullyVisibleCountContract(43).on_validation_epoch_end(_Trainer(), None)
        with self.assertRaises(RuntimeError):
            FullyVisibleCountContract(42).on_validation_epoch_end(_Trainer(), None)

    def test_immutable_step_checkpoint_is_exact_in_real_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step1.ckpt"
            trainer = L.Trainer(
                max_steps=1,
                limit_train_batches=1,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[
                    ImmutableStepCheckpoint(path, 1),
                    Timer(duration=timedelta(minutes=1)),
                ],
            )
            trainer.fit(
                WormProposalModule("intrinsic"),
                DataLoader(_TinyDataset(), batch_size=2),
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertGreaterEqual(checkpoint_training_elapsed_seconds(checkpoint), 0.0)

    def test_evaluate_preserves_failed_inference_identity_and_count(self) -> None:
        class _PartlyFinite(torch.nn.Module):
            def forward(self, image):
                x = torch.linspace(20, 220, 100)
                y = 96 + 20 * torch.sin(torch.linspace(0, 4, 100))
                centerline = torch.stack((x, y), -1).expand(len(image), -1, -1).clone()
                centerline[1] = torch.nan
                support = torch.ones(len(image), 100)
                return {
                    "centerline_xy": centerline,
                    "image_support_probability": support,
                    "in_fov_mask": support.bool(),
                }

        cases, summary = evaluate(
            _PartlyFinite(), _TinyDataset(), batch_size=2, device=torch.device("cpu")
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(summary["requested_samples"], 2)
        self.assertEqual(summary["evaluated_samples"], 1)
        self.assertEqual(summary["failed_inference_count"], 1)
        self.assertEqual(summary["failed_inference_cases"][0]["case_id"], "seed:1")

    def test_fully_visible_checkpoint_metric_ignores_cropped_case(self) -> None:
        from unittest.mock import patch

        module = WormProposalModule("intrinsic")
        horizontal = torch.stack(
            (torch.linspace(20, 220, 100), torch.full((100,), 96.0)), -1
        )
        vertical = torch.stack(
            (torch.full((100,), 128.0), torch.linspace(10, 182, 100)), -1
        )
        target = torch.stack((horizontal, horizontal))
        support = torch.ones(2, 100, dtype=torch.bool)
        support[1, -20:] = False
        prediction = torch.stack((horizontal, vertical))
        output = {
            "centerline_xy": prediction,
            "image_support_logits": torch.zeros(2, 100),
        }
        logged: dict[str, torch.Tensor] = {}

        def capture(name, value, **kwargs):
            logged[name] = value.detach()

        batch = {
            "image": torch.zeros(2, 1, 192, 256),
            "centerline_xy": target,
            "image_support_target": support,
        }
        with patch.object(module, "forward", return_value=output), patch.object(
            module, "log", side_effect=capture
        ):
            module.validation_step(batch, 0, dataloader_idx=1)
        self.assertAlmostEqual(
            float(logged["val_tier_c_fully_visible_angle_mae_degrees"]), 0.0, places=5
        )

    def test_exp7_pool_shape_parameter_budget_and_protocol(self) -> None:
        config = yaml.safe_load(Path("configs/spatial_rescue.yaml").read_text())
        validate_exp_0007_config(config)
        variant, pool = resolve_protocol(config, None)
        model = WormProposalModule(
            variant,
            encoder_pool_output=pool,
            model_seed=config["model_seed"],
            data_seed=config["data_seed"],
        )
        self.assertEqual(model.encoder.pool_output, (4, 4))
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 1_000_000)
        self.assertEqual(model(torch.zeros(1, 1, 192, 256))["centerline_xy"].shape, (1, 100, 2))
        self.assertEqual(resolve_protocol(yaml.safe_load(Path("configs/representation_ablation.yaml").read_text()), "intrinsic")[1], (2, 2))

    def test_tensor_hash_is_canonical_little_endian_float32(self) -> None:
        import hashlib
        import numpy as np

        tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        expected = hashlib.sha256(np.asarray([[1.0, 2.0]], dtype="<f4").tobytes(order="C")).hexdigest()
        self.assertEqual(tensor_sha256(tensor), expected)

    def test_resume_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "last.ckpt").touch()
            with self.assertRaises(FileExistsError):
                resolve_resume_checkpoint(output, resume_last=False, resume_from=None)
            self.assertEqual(
                resolve_resume_checkpoint(output, resume_last=True, resume_from=None),
                output / "last.ckpt",
            )

    def test_fixed_training_order_is_repeatable_and_seed_specific(self) -> None:
        first, first_hash = fixed_training_order(571, 20260818)
        repeated, repeated_hash = fixed_training_order(571, 20260818)
        other, other_hash = fixed_training_order(571, 20260819)
        self.assertEqual(first, repeated)
        self.assertEqual(first_hash, repeated_hash)
        self.assertNotEqual(first, other)
        self.assertNotEqual(first_hash, other_hash)
        self.assertEqual(sorted(first), list(range(571)))
        self.assertEqual(first[12 * 16 :], repeated[12 * 16 :])

    def test_stateful_batch_sampler_restores_exact_suffix(self) -> None:
        sampler = StatefulFixedBatchSampler(571, 16, 20260818)
        iterator = iter(sampler)
        consumed = [next(iterator) for _ in range(12)]
        state = sampler.state_dict()
        restored = StatefulFixedBatchSampler(571, 16, 20260818)
        restored.load_state_dict(state)
        remaining = list(restored)
        expected = StatefulFixedBatchSampler(571, 16, 20260818)
        self.assertEqual(consumed + remaining, list(expected))
        self.assertEqual(state["cursor"], 12)
        with self.assertRaises(RuntimeError):
            StatefulFixedBatchSampler(571, 16, 20260819).load_state_dict(state)

    def test_lightning_mid_epoch_resume_preserves_sample_sequence(self) -> None:
        class _IndexDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, index):
                return torch.tensor(index)

        class _Recorder(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.seen: list[int] = []

            def training_step(self, batch, batch_idx):
                self.seen.extend(int(value) for value in batch)
                return (self.weight - batch.float().mean() / 10).square()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "step2.ckpt"
            first_sampler = StatefulFixedBatchSampler(10, 2, 20260818)
            first_module = _Recorder()
            first_trainer = L.Trainer(
                max_steps=2,
                max_epochs=1,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[
                    TrainingBatchSamplerState(first_sampler),
                    ImmutableStepCheckpoint(checkpoint_path, 2),
                ],
            )
            first_trainer.fit(
                first_module,
                DataLoader(_IndexDataset(), batch_sampler=first_sampler),
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(
                checkpoint["callbacks"]["EXP0007TrainingBatchSampler"]["cursor"],
                2,
            )

            resumed_sampler = StatefulFixedBatchSampler(10, 2, 20260818)
            resumed_module = _Recorder()
            resumed_trainer = L.Trainer(
                max_steps=5,
                max_epochs=1,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[TrainingBatchSamplerState(resumed_sampler)],
            )
            resumed_trainer.fit(
                resumed_module,
                DataLoader(_IndexDataset(), batch_sampler=resumed_sampler),
                ckpt_path=checkpoint_path,
            )
            expected = [
                value
                for batch in StatefulFixedBatchSampler(10, 2, 20260818)
                for value in batch
            ]
            self.assertEqual(first_module.seen + resumed_module.seen, expected)
            self.assertEqual(resumed_trainer.global_step, 5)

            uninterrupted_sampler = StatefulFixedBatchSampler(10, 2, 20260818)
            uninterrupted_module = _Recorder()
            uninterrupted_trainer = L.Trainer(
                max_steps=5,
                max_epochs=1,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[TrainingBatchSamplerState(uninterrupted_sampler)],
            )
            uninterrupted_trainer.fit(
                uninterrupted_module,
                DataLoader(_IndexDataset(), batch_sampler=uninterrupted_sampler),
            )
            self.assertEqual(uninterrupted_module.seen, expected)
            self.assertTrue(
                torch.equal(resumed_module.weight, uninterrupted_module.weight)
            )

    def test_calibration_includes_probability_one(self) -> None:
        probability = torch.tensor([0.0, 1.0])
        target = torch.tensor([0, 0], dtype=torch.bool)
        self.assertAlmostEqual(expected_calibration_error(probability, target), 0.5)

    def test_render_displacement_is_reported_in_original_pixels(self) -> None:
        target = torch.stack((torch.linspace(20, 220, 100), torch.full((100,), 96.0)), -1)[None]
        prediction = target + torch.tensor((1.0, 0.0))
        support = torch.ones(1, 100, dtype=torch.bool)
        metrics = aligned_geometry_metrics(
            prediction, target, support.float(), support, torch.ones_like(support)
        )
        self.assertAlmostEqual(float(metrics["point"].mean()), 968 / 256, places=6)
        self.assertAlmostEqual(float(metrics["endpoint"].mean()), 968 / 256, places=6)
        self.assertAlmostEqual(float(metrics["body_length_error_fraction"]), 0.0)

    def test_fully_visible_summary_does_not_invent_hidden_evidence(self) -> None:
        target = torch.stack((torch.linspace(20, 220, 100), torch.full((100,), 96.0)), -1)[None]
        support = torch.ones(1, 100, dtype=torch.bool)
        metrics = aligned_geometry_metrics(
            target, target, support.float(), support, torch.ones_like(support)
        )
        case = {name: value[0] for name, value in metrics.items() if name != "chosen_target"}
        summary = summarize_cases([case])
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["median_point_px"], 0.0)
        self.assertEqual(summary["visible_mean_point_px"], 0.0)
        self.assertIsNone(summary["hidden_mean_point_px"])
        self.assertEqual(summary["mean_head_endpoint_error_px"], 0.0)
        self.assertEqual(summary["mean_tail_endpoint_error_px"], 0.0)
        self.assertEqual(summary["median_body_length_error_fraction"], 0.0)

    def test_aggregate_decision_reliability_branches(self) -> None:
        config = yaml.safe_load(Path("configs/representation_ablation.yaml").read_text())

        def run(intrinsic_angle: float, coordinate_median: float) -> dict:
            with tempfile.TemporaryDirectory() as directory:
                paths = []
                for variant in ("coordinate", "intrinsic"):
                    for fold in range(3):
                        cases = [
                            {
                                "case_id": f"seed:{index}",
                                "mean_angle_degrees": 6.0 if variant == "coordinate" else intrinsic_angle,
                            }
                            for index in range(12)
                        ]
                        median = coordinate_median if variant == "coordinate" else 3.0
                        document = {
                            "variant": variant,
                            "fold": fold,
                            "tier_C": {
                                "cases": cases,
                                "point_error_units": "original_image_pixels_968x732",
                                "median_point_px": median,
                                "p95_point_px": 8.0,
                                "mean_angle_degrees": 6.0 if variant == "coordinate" else intrinsic_angle,
                                "p95_frame_angle_degrees": 12.0,
                                "reported_vs_recomputed_in_fov_exact_agreement": True,
                                "predicted_geometric_fov_accuracy": 0.9,
                                "mean_endpoint_error_px": 3.0,
                                "mean_body_length_error_px": 5.0,
                                "failed_inference_count": 0,
                                "support_brier": 0.1,
                                "support_ece_10_bin": 0.05,
                                "visible_mean_point_px": 3.0,
                                "hidden_mean_point_px": 4.0,
                                "visible_mean_angle_degrees": 5.0,
                                "hidden_mean_angle_degrees": 7.0,
                            },
                            "tier_B_candidate_proxy": {
                                "median_point_px": 2.0,
                                "point_error_units": "original_image_pixels_968x732",
                            },
                        }
                        path = Path(directory) / f"{variant}-{fold}.json"
                        path.write_text(json.dumps(document))
                        paths.append(path)
                return aggregate_results(
                    paths, config, coordinate_throughput=100.0, intrinsic_throughput=95.0
                )

        accepted = run(intrinsic_angle=4.0, coordinate_median=3.0)
        self.assertEqual(accepted["decision"], "ACCEPT_INTRINSIC")
        self.assertIn("coordinate", accepted["reliability_gates_by_variant"])
        retained = run(intrinsic_angle=5.9, coordinate_median=3.0)
        self.assertEqual(retained["decision"], "RETAIN_COORDINATE")
        revise = run(intrinsic_angle=5.9, coordinate_median=5.0)
        self.assertEqual(revise["decision"], "REVISE_NO_RELIABLE_PROPOSAL")

    def test_both_variants_can_extrapolate_on_all_four_sides(self) -> None:
        image = torch.zeros(1, 1, 192, 256)
        coordinate = WormProposalModule("coordinate").eval()
        final = coordinate.head[-1]
        with torch.no_grad():
            final.weight.zero_(); final.bias.zero_()
            points = final.bias[:200].reshape(100, 2)
            points[:25, 0] = -1.0; points[25:50, 0] = 1.0
            points[50:75, 1] = -1.0; points[75:, 1] = 1.0
        xy = coordinate(image)["centerline_xy"][0]
        self.assertTrue(bool((xy[:, 0] < 0).any() and (xy[:, 0] >= 256).any()))
        self.assertTrue(bool((xy[:, 1] < 0).any() and (xy[:, 1] >= 192).any()))

        intrinsic = WormProposalModule("intrinsic").eval()
        final = intrinsic.head[-1]
        with torch.no_grad():
            final.weight.zero_(); final.bias.zero_(); final.bias[2] = 5.0
            final.bias[3] = 1.0
        horizontal = intrinsic(image)["centerline_xy"][0]
        self.assertTrue(bool((horizontal[:, 0] < 0).any() and (horizontal[:, 0] >= 256).any()))
        with torch.no_grad():
            final.bias[3] = 0.0; final.bias[4] = 1.0
        vertical = intrinsic(image)["centerline_xy"][0]
        self.assertTrue(bool((vertical[:, 1] < 0).any() and (vertical[:, 1] >= 192).any()))

    def test_variants_shape_parameter_ceiling_and_gradients(self) -> None:
        for variant in ("coordinate", "intrinsic"):
            model = WormProposalModule(variant)
            self.assertLessEqual(sum(p.numel() for p in model.parameters()), 1_000_000)
            output = model(torch.rand(2, 1, 192, 256))
            self.assertEqual(output["centerline_xy"].shape, (2, 100, 2))
            self.assertEqual(output["image_support_probability"].shape, (2, 100))
            self.assertEqual(output["in_fov_mask"].dtype, torch.bool)
            output["centerline_xy"].sum().backward()
            self.assertTrue(any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.head.parameters()))
            if variant == "intrinsic":
                self.assertEqual(output["tangent_coefficients"].shape[-1], 16)
                self.assertTrue(bool((output["body_length"] > 0).all()))

    def test_tiny_trainer_checkpoint_load_and_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = WormProposalModule("intrinsic")
            loader = DataLoader(_TinyDataset(), batch_size=2)
            trainer = L.Trainer(max_steps=1, limit_train_batches=1, accelerator="cpu", logger=False,
                                enable_checkpointing=False, enable_progress_bar=False)
            trainer.fit(model, loader)
            path = Path(directory) / "model.ckpt"
            trainer.save_checkpoint(path)
            loaded = WormProposalModule.load_from_checkpoint(path).eval()
            with torch.inference_mode(): result = loaded(next(iter(loader))["image"])
            self.assertEqual(result["centerline_xy"].shape, (2, 100, 2))


if __name__ == "__main__": unittest.main()
