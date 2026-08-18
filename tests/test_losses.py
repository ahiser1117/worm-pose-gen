import unittest

import torch

from worm_pose_gen.losses import proposal_loss, symmetric_point_loss, symmetric_tangent_loss


class LossTests(unittest.TestCase):
    def test_joint_reverse_selection_reverses_support(self) -> None:
        target = torch.stack((torch.linspace(20, 220, 100), torch.linspace(30, 160, 100)), -1)[None]
        support = torch.arange(100)[None] >= 30
        prediction = target.flip(-2).clone()
        logits = torch.where(support.flip(-1), torch.tensor(8.0), torch.tensor(-8.0))
        output = {"centerline_xy": prediction, "image_support_logits": logits}
        forward_truth = proposal_loss(output, target, support)
        reversed_truth = proposal_loss(output, target.flip(-2), support.flip(-1))
        self.assertAlmostEqual(float(forward_truth["loss"]), float(reversed_truth["loss"]), places=7)
        self.assertEqual(float(forward_truth["reversed_fraction"]), 1.0)

    def test_tangent_loss_uses_original_pixel_aspect(self) -> None:
        x = torch.linspace(20, 220, 100)
        prediction = torch.stack((x, torch.full_like(x, 50)), -1)[None]
        target = torch.stack((x, 50 + 0.5 * (x - 20)), -1)[None]
        output = {"centerline_xy": prediction, "image_support_logits": torch.zeros(1, 100)}
        values = proposal_loss(output, target, torch.ones(1, 100, dtype=torch.bool))
        expected_angle = torch.atan(torch.tensor(0.5 * (732 / 192) / (968 / 256)))
        expected = 1 - torch.cos(expected_angle)
        self.assertAlmostEqual(float(values["angle_loss"]), float(expected), places=5)

    def test_forward_reverse_orientation_symmetry(self) -> None:
        target = torch.stack((torch.linspace(10, 200, 100), torch.linspace(20, 150, 100)), -1)[None]
        prediction = target + 0.25
        self.assertAlmostEqual(
            float(symmetric_point_loss(prediction, target)),
            float(symmetric_point_loss(prediction, target.flip(-2))),
            places=7,
        )
        self.assertAlmostEqual(
            float(symmetric_tangent_loss(prediction, target)),
            float(symmetric_tangent_loss(prediction, target.flip(-2))),
            places=7,
        )


if __name__ == "__main__": unittest.main()
