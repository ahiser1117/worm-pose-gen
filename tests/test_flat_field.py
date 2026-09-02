import unittest

import numpy as np

from worm_pose_gen.flat_field import estimate_illumination, fit_flat_field


class FlatFieldTests(unittest.TestCase):
    @staticmethod
    def vignette(shape: tuple[int, int] = (81, 101)) -> np.ndarray:
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        y = (yy - (shape[0] - 1) / 2) / (shape[0] / 2)
        x = (xx - (shape[1] - 1) / 2) / (shape[1] / 2)
        return 190.0 - 65.0 * np.minimum(x * x + y * y, 1.2)

    def test_upper_quantile_suppresses_a_moving_dark_object(self) -> None:
        background = self.vignette()

        def frames():
            for index in range(9):
                frame = background.copy()
                x0 = 4 + 10 * index
                frame[35:46, x0 : x0 + 13] -= 70.0
                yield frame

        estimate = estimate_illumination(
            frames(), temporal_quantile=0.8, spatial_radius=3, smoothing_passes=1
        )
        # Compare to the equally smoothed empty field, not to the unsmoothed
        # analytical vignette at its edge.
        expected = estimate_illumination(
            [background], temporal_quantile=0.8, spatial_radius=3, smoothing_passes=1
        )
        np.testing.assert_allclose(estimate, expected, atol=1e-10)

    def test_correction_flattens_the_recording_background(self) -> None:
        background = self.vignette((101, 121))
        model = fit_flat_field(
            [background] * 3,
            spatial_radius=5,
            smoothing_passes=2,
            reference_fraction=0.3,
        )
        corrected = model.apply(background)
        interior = corrected[8:-8, 8:-8]
        original_interior = background[8:-8, 8:-8]
        self.assertLess(float(np.std(interior)), 0.08 * float(np.std(original_interior)))
        self.assertLess(float(np.ptp(interior)), 0.15 * float(np.ptp(original_interior)))

    def test_gain_is_bounded_and_output_is_unclipped_float(self) -> None:
        illumination = np.full((15, 17), 100.0)
        illumination[:, 0] = 2.0
        model = fit_flat_field(
            [illumination], spatial_radius=0, smoothing_passes=0,
            dark_level=1.0, reference_level=101.0, max_gain=2.0,
        )
        self.assertEqual(float(model.gain[:, 0].max()), 2.0)
        corrected = model.apply(np.full_like(illumination, 200.0))
        self.assertEqual(corrected.dtype, np.float64)
        self.assertGreater(float(corrected[:, 0].max()), 255.0)
        clipped = model.apply(np.full_like(illumination, 200.0), clip=(0, 255))
        self.assertEqual(float(clipped.max()), 255.0)

    def test_validation_rejects_bad_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            estimate_illumination(iter(()))
        with self.assertRaisesRegex(ValueError, "same shape"):
            estimate_illumination([np.zeros((2, 3)), np.zeros((3, 2))])
        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_illumination([np.asarray([[np.nan]])])
        with self.assertRaisesRegex(ValueError, "gain bounds"):
            fit_flat_field([np.ones((2, 2))], min_gain=3.0, max_gain=2.0)


if __name__ == "__main__":
    unittest.main()
