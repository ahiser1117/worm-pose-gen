import unittest

import numpy as np

from scripts.exp_005_representation_oracle import (
    cosine_basis,
    cubic_bspline_basis,
    intrinsic_target,
    reconstruct_from_shape,
    reconstruction_metrics,
)


class RepresentationOracleTests(unittest.TestCase):
    def test_cubic_basis_is_partition_of_unity(self):
        basis = cubic_bspline_basis(99, 16)
        self.assertEqual(basis.shape, (99, 16))
        np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(basis[0], np.eye(16)[0], atol=1e-12)
        np.testing.assert_allclose(basis[-1], np.eye(16)[-1], atol=1e-12)

    def test_cosine_basis_is_orthonormal_and_zero_mean(self):
        basis = cosine_basis(99, 16)
        np.testing.assert_allclose(basis.T @ basis, np.eye(16), atol=1e-12)
        np.testing.assert_allclose(basis.mean(axis=0), 0.0, atol=1e-12)

    def test_full_tangent_reconstructs_uniform_arc_curve(self):
        angle = np.linspace(-0.8, 0.9, 99)
        difference = 3.0 * np.column_stack((np.cos(angle), np.sin(angle)))
        target = np.vstack((np.array([[30.0, 40.0]]), np.array([[30.0, 40.0]]) + np.cumsum(difference, axis=0)))
        shape, rotation, length = intrinsic_target(target)
        prediction = reconstruct_from_shape(shape, rotation, length, target)
        result = reconstruction_metrics(prediction, target)
        self.assertLess(result["p95_point_distance_px"], 1e-10)
        self.assertLess(result["mean_tangent_error_deg"], 1e-10)


if __name__ == "__main__":
    unittest.main()
