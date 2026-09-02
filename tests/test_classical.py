import unittest

import numpy as np

from worm_pose_gen.classical import (
    ClassicalConfig,
    _connected_extension,
    box_blur,
    extract_centerline,
    resample_centerline,
    segment_dark_ridge,
)


class ClassicalTests(unittest.TestCase):
    def test_box_blur_preserves_constant(self) -> None:
        image = np.full((19, 23), 17, dtype=np.uint8)
        np.testing.assert_allclose(box_blur(image, 4), 17.0)

    def test_resampling_is_uniform_and_preserves_endpoints(self) -> None:
        points = np.asarray([[0, 0], [3, 0], [3, 4]], dtype=float)
        output = resample_centerline(points, 8)
        self.assertEqual(output.shape, (8, 2))
        np.testing.assert_allclose(output[[0, -1]], points[[0, -1]])
        steps = np.linalg.norm(np.diff(output, axis=0), axis=1)
        self.assertLess(float(steps.max() - steps.min()), 0.25)

    def test_extracts_easy_dark_curved_tube(self) -> None:
        height, width = 360, 520
        yy, xx = np.mgrid[:height, :width]
        image = np.full((height, width), 185.0)
        x_curve = np.linspace(80, 440, 500)
        y_curve = 180 + 55 * np.sin((x_curve - 80) / 360 * 2 * np.pi)
        distance_sq = np.full_like(image, np.inf)
        for x, y in zip(x_curve[::3], y_curve[::3]):
            distance_sq = np.minimum(distance_sq, (xx - x) ** 2 + (yy - y) ** 2)
        image[distance_sq <= 9**2] = 105
        config = ClassicalConfig(min_area=1_000, max_area=20_000, min_length=250,
                                 max_length=600, foreground_z=1.8,
                                 connected_foreground_z=None)
        result = extract_centerline(image.astype(np.uint8), config)
        self.assertTrue(result.accepted, result.rejection_reasons)
        self.assertEqual(result.centerline_xy.shape, (100, 2))
        self.assertGreaterEqual(result.qc["tube_support_fraction"], 0.95)
        self.assertEqual(result.qc["backbone_endpoint_count"], 2)
        self.assertGreaterEqual(result.qc["boundary_distance_px"], config.boundary_margin)

    def test_local_normalization_is_offset_invariant(self) -> None:
        height, width = 360, 520
        yy, xx = np.mgrid[:height, :width]
        tube = (np.abs(yy - (180 + 45 * np.sin((xx - 80) / 360 * np.pi))) <= 9) & (xx >= 80) & (xx <= 440)
        dark = np.full((height, width), 160, dtype=np.uint8)
        bright = np.full((height, width), 215, dtype=np.uint8)
        dark[tube], bright[tube] = 90, 145
        config = ClassicalConfig(min_area=1_000, max_area=20_000, min_length=250,
                                 max_length=600, foreground_z=1.8,
                                 connected_foreground_z=None)
        first = extract_centerline(dark, config)
        second = extract_centerline(bright, config)
        self.assertTrue(first.accepted, first.rejection_reasons)
        self.assertTrue(second.accepted, second.rejection_reasons)
        direct = np.linalg.norm(first.centerline_xy - second.centerline_xy, axis=1).mean()
        flipped = np.linalg.norm(first.centerline_xy - second.centerline_xy[::-1], axis=1).mean()
        self.assertLess(min(direct, flipped), 0.25)

    def test_rejects_boundary_tube(self) -> None:
        image = np.full((300, 420), 180, dtype=np.uint8)
        image[0:17, 40:370] = 90
        config = ClassicalConfig(min_area=500, max_area=20_000, min_length=200,
                                 max_length=500, foreground_z=1.5,
                                 connected_foreground_z=None)
        result = extract_centerline(image, config)
        self.assertFalse(result.accepted)
        self.assertIn("boundary_contact", result.rejection_reasons)
        # Local normalization can make a truncated ridge terminate a few pixels
        # inboard; the conservative width-scale clearance still rejects it.
        self.assertLess(result.qc["boundary_distance_px"], config.boundary_margin)

    def test_connected_extension_recovers_only_seeded_faint_section(self) -> None:
        seed = np.zeros((12, 18), dtype=bool)
        seed[5:7, 2:6] = True
        eligible = seed.copy()
        eligible[5:7, 6:14] = True
        eligible[1:3, 14:17] = True
        recovered = _connected_extension(seed, eligible)
        self.assertTrue(np.all(recovered[5:7, 2:14]))
        self.assertFalse(np.any(recovered[1:3, 14:17]))

    def test_dark_ridge_hysteresis_reports_recovery(self) -> None:
        image = np.full((90, 140), 190, dtype=np.uint8)
        image[42:49, 20:70] = 90
        image[42:49, 70:115] = 135
        high_only = segment_dark_ridge(
            image,
            ClassicalConfig(
                local_radius=17, smooth_radius=1, foreground_z=2.2, close_radius=1
            ),
        )
        with_recovery = segment_dark_ridge(
            image,
            ClassicalConfig(
                local_radius=17,
                smooth_radius=1,
                foreground_z=2.2,
                connected_foreground_z=0.8,
                close_radius=1,
            ),
        )
        self.assertGreater(with_recovery.recovered_area, 0)
        self.assertGreater(with_recovery.component.sum(), high_only.component.sum())

    def test_dark_ridge_rejects_invalid_connected_cutoff(self) -> None:
        image = np.full((20, 30), 100, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "below foreground_z"):
            segment_dark_ridge(
                image,
                ClassicalConfig(foreground_z=2.0, connected_foreground_z=2.0),
            )


if __name__ == "__main__":
    unittest.main()
