import unittest

import numpy as np

from worm_pose_gen.width import (
    fit_profile_parameters,
    fit_width_profile_model,
    reconstruct_width_profile,
)


class WidthProfileTests(unittest.TestCase):
    def test_mean_scale_and_pca_projection(self) -> None:
        s = np.linspace(0, 1, 20)
        mean = 5 + 10 * np.sin(np.pi * s)
        variation = np.cos(2 * np.pi * s)
        profiles = np.stack((mean - variation, mean, mean + variation))
        model = fit_width_profile_model(profiles, components=1)
        self.assertEqual(model.components.shape, (1, 20))
        prediction, scale, coefficient = fit_profile_parameters(
            profiles[2], model, fit_scale=False
        )
        np.testing.assert_allclose(prediction, profiles[2], atol=1e-10)
        self.assertEqual(scale, 1.0)
        self.assertEqual(coefficient.shape, (1,))

        scaled, fitted_scale, _ = fit_profile_parameters(
            1.1 * mean,
            fit_width_profile_model(np.stack((mean, mean)), components=0),
            fit_scale=True,
        )
        self.assertAlmostEqual(fitted_scale, 1.1)
        np.testing.assert_allclose(scaled, 1.1 * mean)

    def test_validation_and_bounding(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_width_profile_model(np.zeros((2, 10)))
        model = fit_width_profile_model(np.full((2, 10), 5.0), maximum=6.0)
        np.testing.assert_allclose(reconstruct_width_profile(model, scale=2.0), 6.0)
        with self.assertRaisesRegex(ValueError, "scale_bounds"):
            fit_profile_parameters(np.full(10, 5.0), model, fit_scale=True, scale_bounds=(2, 1))


if __name__ == "__main__":
    unittest.main()
