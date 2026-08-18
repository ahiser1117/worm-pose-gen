import unittest

import torch

from worm_pose_gen.renderer import render_worm


class RendererTests(unittest.TestCase):
    def test_shapes_batch_and_fov_support(self) -> None:
        x = torch.linspace(-8, 70, 100)
        first = torch.stack((x, torch.full_like(x, 20)), -1)
        second = first + torch.tensor([0.0, 10.0])
        result = render_worm(torch.stack((first, second)), 7.0, 48, 64)
        self.assertEqual(result["image"].shape, (2, 48, 64))
        self.assertEqual(result["tube_mask"].shape, (2, 48, 64))
        self.assertEqual(result["image_support_target"].shape, (2, 100))
        self.assertTrue(torch.equal(result["image_support_target"], result["in_fov_mask"]))
        self.assertFalse(bool(result["image_support_target"][:, 0].any()))
        self.assertTrue(bool(result["observable_pixel_mask"].all()))
        self.assertTrue(bool(((result["image"] >= 0) & (result["image"] <= 1)).all()))

    def test_finite_nonzero_pose_width_and_intensity_gradients(self) -> None:
        centerline = torch.stack(
            (torch.linspace(5, 58, 100), 18 + 4 * torch.sin(torch.linspace(0, 5, 100))), -1
        ).requires_grad_()
        width = torch.tensor(8.0, requires_grad=True)
        foreground = torch.tensor(0.2, requires_grad=True)
        result = render_worm(centerline, width, 40, 64, foreground=foreground)
        weights = torch.linspace(0.1, 1.0, 64)[None, :]
        loss = (result["image"] * weights).sum() + 0.1 * result["tube_mask"].square().sum()
        loss.backward()
        for gradient in (centerline.grad, width.grad, foreground.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_nuisance_variation(self) -> None:
        line = torch.stack((torch.linspace(2, 29, 50), torch.full((50,), 12.0)), -1)
        gradient = torch.tensor([[0.1, -0.05]])
        result = render_worm(line, torch.linspace(3, 8, 50), 24, 32, illumination_gradient=gradient)
        self.assertGreater(float(result["image"].std()), 0)

    def test_image_support_is_distinct_from_geometric_membership(self) -> None:
        line = torch.stack((torch.linspace(2, 29, 50), torch.full((50,), 12.0)), -1)
        evidence = torch.ones(50, dtype=torch.bool)
        evidence[20:25] = False  # synthetic occlusion despite geometric membership
        result = render_worm(line, 5.0, 24, 32, image_support_target=evidence)
        self.assertTrue(bool(result["in_fov_mask"].all()))
        self.assertTrue(torch.equal(result["image_support_target"], evidence))


if __name__ == "__main__":
    unittest.main()
