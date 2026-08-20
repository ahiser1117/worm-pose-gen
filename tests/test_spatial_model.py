import unittest

import torch

from worm_pose_gen.spatial_model import (
    SpatialPoseModule,
    spatial_soft_argmax,
    symmetric_dense_heatmap_loss,
)
from worm_pose_gen.topology_rescue_model import SoftAnchoredIntrinsicModule


class SpatialModelTests(unittest.TestCase):
    def test_soft_anchor_rescue_is_ordered_and_differentiable(self) -> None:
        model = SoftAnchoredIntrinsicModule(model_seed=7)
        images = torch.rand((2, 1, 192, 256))
        output = model(images)
        self.assertEqual(output["centerline_xy"].shape, (2, 100, 2))
        self.assertEqual(output["anchor_probability"].shape, (2, 192))
        torch.testing.assert_close(
            output["anchor_probability"].sum(-1), torch.ones(2), rtol=1e-5, atol=1e-6
        )
        self.assertLess(sum(value.numel() for value in model.parameters()), 2_000_000)
        output["centerline_xy"].square().mean().backward()
        self.assertTrue(any(
            value.grad is not None and bool(torch.isfinite(value.grad).all())
            for value in model.parameters()
        ))

    def test_soft_argmax_coordinate_contract(self):
        logits = torch.full((1, 2, 4, 8), -20.0)
        logits[0, 0, 1, 2] = 20.0
        logits[0, 1, 3, 7] = 20.0
        coordinates, probability = spatial_soft_argmax(
            logits, image_height=192, image_width=256, temperature=0.25
        )
        torch.testing.assert_close(
            coordinates[0], torch.tensor([[80.0, 72.0], [240.0, 168.0]]), atol=1e-5, rtol=0
        )
        self.assertEqual(probability.shape, (1, 2, 32))

    def test_dense_heatmap_loss_is_reversal_symmetric(self):
        logits = torch.randn(2, 100, 12, 16)
        target = torch.rand(2, 100, 2) * torch.tensor((256.0, 192.0))
        support = torch.ones(2, 100, dtype=torch.bool)
        forward = symmetric_dense_heatmap_loss(
            logits, target, support, image_height=192, image_width=256
        )
        reverse = symmetric_dense_heatmap_loss(
            logits, target.flip(1), support.flip(1), image_height=192, image_width=256
        )
        torch.testing.assert_close(forward, reverse)

    def test_spatial_variants_shapes_parameters_and_gradients(self):
        batch = {
            "image": torch.rand(2, 1, 192, 256),
            "centerline_xy": torch.rand(2, 100, 2) * torch.tensor((256.0, 192.0)),
            "image_support_target": torch.ones(2, 100, dtype=torch.bool),
        }
        for variant in ("dense_centerline_field", "anchored_intrinsic_grid"):
            model = SpatialPoseModule(variant)
            self.assertLessEqual(sum(value.numel() for value in model.parameters()), 2_000_000)
            output = model(batch["image"])
            self.assertEqual(output["centerline_xy"].shape, (2, 100, 2))
            self.assertEqual(output["image_support_probability"].shape, (2, 100))
            self.assertEqual(output["selection_score"].shape, (2,))
            loss = model._shared_step(batch, "test")
            loss.backward()
            self.assertTrue(any(
                value.grad is not None and bool(torch.isfinite(value.grad).all())
                for value in model.parameters()
            ))

    def test_anchor_grid_uses_soft_training_and_hard_evaluation(self):
        image = torch.rand(1, 1, 192, 256)
        model = SpatialPoseModule("anchored_intrinsic_grid")
        model.train()
        training = model(image)
        torch.testing.assert_close(training["centerline_xy"], training["soft_centerline_xy"])
        model.eval()
        evaluation = model(image)
        index = int(evaluation["selected_cell_index"][0])
        torch.testing.assert_close(
            evaluation["centerline_xy"][0], evaluation["candidate_centerline_xy"][0, index]
        )


if __name__ == "__main__":
    unittest.main()
