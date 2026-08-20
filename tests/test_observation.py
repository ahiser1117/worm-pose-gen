import unittest

import torch

from worm_pose_gen.observation import (
    balanced_soft_bce_energy,
    hybrid_mask_energy,
    signed_distance_energy,
    signed_distance_from_mask,
    soft_dice_energy,
)
from worm_pose_gen.renderer import render_worm


class ObservationEnergyTests(unittest.TestCase):
    def test_energies_prefer_matching_mask_and_have_gradients(self) -> None:
        target = torch.zeros(24, 32)
        target[8:16, 7:25] = 1
        matching = (0.01 + 0.98 * target).requires_grad_()
        shifted = torch.roll(matching.detach(), 5, dims=1)
        sdf = signed_distance_from_mask(target.bool())
        energies = [
            balanced_soft_bce_energy(matching, target),
            soft_dice_energy(matching, target),
            signed_distance_energy(matching, sdf, edge_softness=1.0),
            hybrid_mask_energy(matching, target, sdf, edge_softness=1.0),
        ]
        shifted_energies = [
            balanced_soft_bce_energy(shifted, target),
            soft_dice_energy(shifted, target),
            signed_distance_energy(shifted, sdf, edge_softness=1.0),
            hybrid_mask_energy(shifted, target, sdf, edge_softness=1.0),
        ]
        for exact, wrong in zip(energies, shifted_energies, strict=True):
            self.assertLess(float(exact.detach()), float(wrong.detach()))
        sum(energies).backward()
        self.assertTrue(bool(torch.isfinite(matching.grad).all()))
        self.assertGreater(float(matching.grad.abs().sum()), 0)

    def test_signed_distance_sign_and_validation(self) -> None:
        target = torch.zeros(9, 9, dtype=torch.bool)
        target[2:7, 2:7] = True
        sdf = signed_distance_from_mask(target)
        self.assertGreater(float(sdf[4, 4]), 0)
        self.assertLess(float(sdf[0, 0]), 0)
        self.assertEqual(float(sdf[2, 2]), 0)
        with self.assertRaisesRegex(ValueError, "foreground and background"):
            signed_distance_from_mask(torch.zeros(4, 4, dtype=torch.bool))

    def test_balanced_bce_matches_separate_class_normalization(self) -> None:
        prediction = torch.tensor([[0.8, 0.7], [0.3, 0.1]])
        target = torch.tensor([[1.0, 0.5], [0.0, 0.0]], dtype=torch.float64)
        expected_foreground = -(target * prediction.double().log()).sum() / target.sum()
        expected_background = -(
            (1 - target) * torch.log1p(-prediction.double())
        ).sum() / (1 - target).sum()
        observed = balanced_soft_bce_energy(prediction, target)
        self.assertTrue(
            torch.allclose(
                observed.double(), 0.5 * (expected_foreground + expected_background)
            )
        )

    def test_energy_gradient_flows_through_pose_parameter(self) -> None:
        base = torch.tensor(
            [[5.0, 12.0], [10.0, 12.0], [15.0, 12.0], [20.0, 12.0]]
        )
        target = render_worm(base, 5.0, 24, 28)["tube_mask"].detach()
        translation = torch.tensor([0.25, -0.1], requires_grad=True)
        prediction = render_worm(base + translation, 5.0, 24, 28)["tube_mask"]
        soft_dice_energy(prediction, target).backward()
        self.assertTrue(bool(torch.isfinite(translation.grad).all()))
        self.assertGreater(float(translation.grad.norm()), 0)

    def test_partial_fov_and_self_overlap_pose_gradients_are_finite(self) -> None:
        base = torch.tensor(
            [
                [-4.0, 11.0],
                [4.0, 6.0],
                [12.0, 12.0],
                [4.0, 18.0],
                [13.0, 11.0],
                [22.0, 12.0],
            ]
        )
        target = render_worm(base, 4.0, 24, 24)["tube_mask"].detach()
        shift = torch.tensor([0.2, -0.15], requires_grad=True)
        prediction = render_worm(base + shift, 4.0, 24, 24)["tube_mask"]
        soft_dice_energy(prediction, target).backward()
        self.assertTrue(bool(torch.isfinite(shift.grad).all()))
        self.assertGreater(float(shift.grad.norm()), 0)


if __name__ == "__main__":
    unittest.main()
