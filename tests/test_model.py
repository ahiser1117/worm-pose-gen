import tempfile
import unittest
from pathlib import Path
import json

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from scripts.evaluate import (
    aggregate_results,
    aligned_geometry_metrics,
    expected_calibration_error,
    summarize_cases,
)
from worm_pose_gen.model import WormProposalModule


class _TinyDataset(Dataset):
    def __len__(self): return 2
    def __getitem__(self, index):
        x = torch.linspace(20, 220, 100)
        y = 96 + 20 * torch.sin(torch.linspace(0, 4, 100))
        return {"image": torch.zeros(1, 192, 256), "centerline_xy": torch.stack((x, y), -1),
                "image_support_target": torch.ones(100, dtype=torch.bool)}


class ModelTests(unittest.TestCase):
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
