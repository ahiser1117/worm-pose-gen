import math
import unittest

import torch

from worm_pose_gen.geometry import (
    canonicalize_orientation,
    curvature,
    curvature_from_angles,
    in_fov_mask,
    reconstruct_centerline,
    reconstruct_from_coefficients,
    resample_centerline,
    tangent_angles,
    wrap_angle,
)


class GeometryTests(unittest.TestCase):
    def test_wrap_is_half_open(self) -> None:
        values = torch.tensor([-3 * math.pi, -math.pi, math.pi, 3 * math.pi, 0.25])
        expected = torch.tensor([-math.pi, -math.pi, -math.pi, -math.pi, 0.25])
        torch.testing.assert_close(wrap_angle(values), expected)
        self.assertTrue(bool(torch.all(wrap_angle(values) >= -math.pi)))
        self.assertTrue(bool(torch.all(wrap_angle(values) < math.pi)))

    def test_pixel_center_half_open_bounds(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0], [9.999, 7.999], [10.0, 4.0], [4.0, 8.0], [-0.001, 0.0]]
        )
        self.assertEqual(in_fov_mask(points, 8, 10).tolist(), [True, True, False, False, False])

    def test_straight_reconstruction_and_translation_rotation(self) -> None:
        theta = torch.zeros(6)
        line = reconstruct_centerline(torch.tensor([3.0, 4.0]), theta, 10.0)
        expected = torch.stack((torch.linspace(3, 13, 6), torch.full((6,), 4.0)), -1)
        torch.testing.assert_close(line, expected)
        rotated = reconstruct_centerline(torch.tensor([-2.0, 1.0]), theta + math.pi / 2, 10.0)
        torch.testing.assert_close(rotated[:, 0], torch.full((6,), -2.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(rotated[:, 1], torch.linspace(1, 11, 6), atol=1e-6, rtol=0)

    def test_constant_curvature_profile(self) -> None:
        length, kappa, n = 20.0, 0.05, 101
        s = torch.linspace(0, length, n)
        theta = kappa * s
        curve = reconstruct_centerline(torch.zeros(2), theta, length)
        exact = torch.stack((torch.sin(kappa * s) / kappa, (1 - torch.cos(kappa * s)) / kappa), -1)
        torch.testing.assert_close(curve, exact, atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(
            curvature_from_angles(theta, length), torch.full((n,), kappa), atol=2e-6, rtol=0
        )

    def test_batched_curvature_uses_each_pixel_length(self) -> None:
        angles = torch.stack((torch.linspace(0, 1, 11), torch.linspace(0, 1, 11)))
        result = curvature_from_angles(angles, torch.tensor([10.0, 20.0]))
        torch.testing.assert_close(result[0], torch.full((11,), 0.1), atol=2e-7, rtol=0)
        torch.testing.assert_close(result[1], torch.full((11,), 0.05), atol=2e-7, rtol=0)

    def test_sinusoidal_basis_profile(self) -> None:
        n = 81
        s = torch.linspace(0, 1, n)
        basis = torch.stack((torch.sin(2 * math.pi * s), torch.sin(4 * math.pi * s)), -1)
        coefficients = torch.tensor([0.4, -0.1])
        curve, theta = reconstruct_from_coefficients(
            torch.tensor([2.0, 3.0]), 0.2, 30.0, coefficients, basis
        )
        expected_theta = wrap_angle(0.2 + basis @ coefficients)
        torch.testing.assert_close(theta, expected_theta)
        self.assertEqual(curve.shape, (n, 2))
        torch.testing.assert_close(curve[0], torch.tensor([2.0, 3.0]))

    def test_anchor_index(self) -> None:
        result = reconstruct_centerline(torch.tensor([5.0, 7.0]), torch.zeros(5), 8.0, anchor_index=2)
        torch.testing.assert_close(result[2], torch.tensor([5.0, 7.0]))
        torch.testing.assert_close(result[:, 0], torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0]))

    def test_resampling_is_uniform_in_arc_length(self) -> None:
        uneven = torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [10.0, 0.0]])
        result = resample_centerline(uneven, 6)
        torch.testing.assert_close(result[:, 0], torch.linspace(0, 10, 6))
        torch.testing.assert_close(result[:, 1], torch.zeros(6))
        batch = resample_centerline(torch.stack((uneven, uneven + 2)), 3)
        self.assertEqual(batch.shape, (2, 3, 2))

    def test_reconstruction_and_resampling_gradients(self) -> None:
        anchor = torch.tensor([0.0, 0.0], requires_grad=True)
        theta = torch.linspace(-0.4, 0.7, 20, requires_grad=True)
        length = torch.tensor(12.0, requires_grad=True)
        curve = reconstruct_centerline(anchor, theta, length)
        sampled = resample_centerline(curve, 11)
        sampled.square().sum().backward()
        for value in (anchor.grad, theta.grad, length.grad):
            self.assertIsNotNone(value)
            self.assertTrue(bool(torch.all(torch.isfinite(value))))
        self.assertGreater(float(theta.grad.abs().sum()), 0)

    def test_tangent_and_curvature_sign_in_y_down_coordinates(self) -> None:
        radius = 50.0
        phi = torch.linspace(0, 0.8, 401, dtype=torch.float64)
        # Increasing phi bends from right toward down: clockwise on screen.
        curve = torch.stack((radius * torch.sin(phi), radius * (1 - torch.cos(phi))), -1)
        estimated_theta = tangent_angles(curve)
        self.assertGreater(float(estimated_theta[-1]), float(estimated_theta[0]))
        estimated_curvature = curvature(curve)
        torch.testing.assert_close(
            estimated_curvature[3:-3], torch.full_like(estimated_curvature[3:-3], 1 / radius),
            atol=3e-5, rtol=2e-3
        )

    def test_canonicalization_semantics(self) -> None:
        centerline = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        theta = torch.zeros(3)
        kappa = torch.tensor([1.0, 2.0, 3.0])
        result = canonicalize_orientation(
            centerline, 0.2, tangent_angle=theta, curvature_values=kappa,
            image_support_probability=torch.tensor([0.1, 0.5, 0.9])
        )
        torch.testing.assert_close(result["centerline_xy"], centerline.flip(0))
        torch.testing.assert_close(result["tangent_angle"], torch.full((3,), -math.pi))
        torch.testing.assert_close(result["curvature"], torch.tensor([-3.0, -2.0, -1.0]))
        torch.testing.assert_close(result["image_support_probability"], torch.tensor([0.9, 0.5, 0.1]))
        self.assertAlmostEqual(float(result["head_tail_probability"]), 0.8)
        unchanged = canonicalize_orientation(centerline, 0.5)
        torch.testing.assert_close(unchanged["centerline_xy"], centerline)
        self.assertFalse(bool(unchanged["reversed"]))


if __name__ == "__main__":
    unittest.main()
