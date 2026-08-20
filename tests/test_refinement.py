from __future__ import annotations

import unittest

import torch

from worm_pose_gen.refinement import RefinementConfig, RefinablePose, refine_pose
from worm_pose_gen.renderer import render_worm


class RefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RefinementConfig(image_height=48, image_width=64, anchor_index=10, coefficients=8)
        self.tangent = torch.linspace(-0.25, 0.25, 20).unsqueeze(0)
        self.length = torch.tensor([35.0])
        self.anchor = torch.tensor([[32.0, 24.0]])
        self.width = torch.tensor([[2.0] * 20])

    def test_initial_pose_reconstructs_declared_intrinsic_state(self) -> None:
        module = RefinablePose(self.anchor, self.tangent, self.length, config=self.config)
        pose = module.pose()
        torch.testing.assert_close(pose["anchor_xy"], self.anchor)
        torch.testing.assert_close(pose["tangent_angle"], self.tangent)
        torch.testing.assert_close(pose["body_length"], self.length)
        torch.testing.assert_close(pose["centerline_xy"][:, 10], self.anchor)

    def test_pixel_refinement_reduces_small_translation_error(self) -> None:
        truth = RefinablePose(self.anchor, self.tangent, self.length, config=self.config).pose()["centerline_xy"].detach()
        target = render_worm(truth, self.width, 48, 64)
        _, history = refine_pose(
            self.anchor + torch.tensor([[3.0, 0.0]]), self.tangent, self.length,
            self.width, target["image"], target_mask=target["tube_mask"],
            objective="pixel_gradient", steps=5, record_steps=(0, 5), config=self.config,
        )
        initial = torch.linalg.vector_norm(history[0]["centerline_xy"] - truth, dim=-1).mean()
        final = torch.linalg.vector_norm(history[5]["centerline_xy"] - truth, dim=-1).mean()
        self.assertLess(float(final), float(initial))

    def test_record_step_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "record_steps"):
            refine_pose(
                self.anchor, self.tangent, self.length, self.width,
                torch.zeros(1, 48, 64), steps=2, record_steps=(3,), config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
