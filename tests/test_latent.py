import unittest

import numpy as np
import torch

from worm_pose_gen.latent import (
    decode_centerline,
    decode_centerline_torch,
    encode_centerline,
    orient_to_reference,
    unwrap_latent_rotation,
)


class LatentPoseTests(unittest.TestCase):
    def test_numpy_torch_decode_and_gradient(self) -> None:
        s = np.linspace(0, 1, 100)
        angle = 0.8 * np.sin(2 * np.pi * s[:-1]) + 0.4
        difference = 3.0 * np.column_stack((np.cos(angle), np.sin(angle)))
        curve = np.vstack(([20.0, 30.0], [20.0, 30.0] + np.cumsum(difference, axis=0)))
        latent = encode_centerline(curve)
        numpy_curve = decode_centerline(latent)
        tensor = torch.tensor(latent, dtype=torch.float64, requires_grad=True)
        torch_curve = decode_centerline_torch(tensor)
        self.assertTrue(np.allclose(numpy_curve, torch_curve.detach().numpy(), atol=1e-9))
        self.assertLess(float(np.median(np.linalg.norm(numpy_curve - curve, axis=1))), 1.0)
        torch_curve.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(tensor.grad).all()))

    def test_orientation_and_rotation_unwrap(self) -> None:
        curve = np.column_stack((np.linspace(0, 99, 100), np.zeros(100)))
        oriented, reversed_order = orient_to_reference(curve[::-1], curve)
        self.assertTrue(reversed_order)
        self.assertTrue(np.array_equal(oriented, curve))
        previous = encode_centerline(curve)
        current = previous.copy()
        current[-4] += 2 * np.pi
        self.assertAlmostEqual(unwrap_latent_rotation(current, previous)[-4], previous[-4])


if __name__ == "__main__":
    unittest.main()
