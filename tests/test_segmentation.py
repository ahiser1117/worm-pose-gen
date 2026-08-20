import unittest

import numpy as np

from worm_pose_gen.segmentation import (
    ClassicalSoftForegroundConfig,
    classical_soft_foreground,
    connected_low_threshold_extension,
    fill_small_enclosed_holes,
)


class ClassicalSoftForegroundTests(unittest.TestCase):
    def test_probability_is_bounded_and_monotone_with_darkness(self) -> None:
        image = np.full((80, 120), 180, dtype=np.uint8)
        image[20:60, 20:45] = 145
        image[20:60, 75:100] = 95
        result = classical_soft_foreground(
            image,
            ClassicalSoftForegroundConfig(
                local_radius=9, smooth_radius=1, score_midpoint=0.0,
                logistic_temperature=1.0, close_radius=0,
            ),
        )
        self.assertGreaterEqual(float(result.probability_map.min()), 0.0)
        self.assertLessEqual(float(result.probability_map.max()), 1.0)
        self.assertGreater(
            float(result.probability_map[30:50, 80:95].mean()),
            float(result.probability_map[30:50, 25:40].mean()),
        )
        self.assertFalse(bool(result.qc["is_pretrained"]))

    def test_cleanup_retains_only_largest_component(self) -> None:
        image = np.full((100, 140), 190, dtype=np.uint8)
        image[40:55, 20:105] = 80
        image[10:16, 120:126] = 70
        result = classical_soft_foreground(
            image,
            ClassicalSoftForegroundConfig(
                local_radius=11, smooth_radius=1, score_midpoint=1.0,
                logistic_temperature=0.4, close_radius=1,
            ),
        )
        self.assertGreaterEqual(int(result.qc["closed_component_count"]), 2)
        self.assertTrue(result.cleaned_mask[47, 50])
        self.assertFalse(result.cleaned_mask[12, 123])
        self.assertLessEqual(result.cleaned_mask.sum(), result.raw_mask.sum() + 100)

    def test_small_holes_fill_but_large_enclosed_background_remains(self) -> None:
        mask = np.zeros((60, 80), dtype=bool)
        mask[8:52, 10:70] = True
        mask[20:22, 20:23] = False  # six-pixel texture hole
        mask[27:37, 35:47] = False  # large coil-like enclosed region
        cleaned, count, area = fill_small_enclosed_holes(mask, max_hole_area=12)
        self.assertTrue(bool(cleaned[20:22, 20:23].all()))
        self.assertFalse(bool(cleaned[27:37, 35:47].any()))
        self.assertEqual(count, 1)
        self.assertEqual(area, 6)

    def test_hole_fill_qc_is_exposed(self) -> None:
        # Direct cleanup behavior is tested above; this asserts the public
        # segmentation result always carries explicit accounting.
        image = np.full((50, 70), 180, dtype=np.uint8)
        image[20:30, 10:60] = 80
        result = classical_soft_foreground(
            image,
            ClassicalSoftForegroundConfig(local_radius=9, smooth_radius=1, max_hole_area=8),
        )
        self.assertIn("filled_hole_count", result.qc)
        self.assertIn("filled_hole_area", result.qc)

    def test_connected_low_threshold_recovers_terminal_not_noise(self) -> None:
        high = np.zeros((30, 70), dtype=bool)
        high[12:18, 20:40] = True
        low = high.copy()
        low[12:18, 40:58] = True  # connected dim terminal
        low[3:8, 3:9] = True      # disconnected low-confidence noise
        extended, recovered, disconnected = connected_low_threshold_extension(high, low)
        self.assertTrue(bool(extended[12:18, 40:58].all()))
        self.assertFalse(bool(extended[3:8, 3:9].any()))
        self.assertEqual(recovered, 6 * 18)
        self.assertEqual(disconnected, 5 * 6)

    def test_hysteresis_is_opt_in_and_reports_qc(self) -> None:
        image = np.full((70, 120), 180, dtype=np.uint8)
        image[31:40, 25:75] = 80
        default = classical_soft_foreground(
            image,
            ClassicalSoftForegroundConfig(
                local_radius=11, smooth_radius=1, score_midpoint=1.0,
                logistic_temperature=0.6, close_radius=0, max_hole_area=0,
            ),
        )
        explicit_disabled = classical_soft_foreground(
            image,
            ClassicalSoftForegroundConfig(
                local_radius=11, smooth_radius=1, score_midpoint=1.0,
                logistic_temperature=0.6, close_radius=0, max_hole_area=0,
                low_probability_threshold=None,
            ),
        )
        np.testing.assert_array_equal(default.cleaned_mask, explicit_disabled.cleaned_mask)
        self.assertFalse(bool(default.qc["hysteresis_enabled"]))
        self.assertEqual(default.qc["hysteresis_recovered_area"], 0)

    def test_image_hysteresis_recovers_dim_terminal_but_not_dim_noise(self) -> None:
        image = np.full((80, 140), 180, dtype=np.uint8)
        image[36:45, 25:85] = 80    # high-confidence body
        image[36:45, 85:120] = 155  # connected dim terminal
        image[10:17, 8:20] = 155    # disconnected dim distractor
        common = dict(
            local_radius=11, smooth_radius=1, score_midpoint=2.6,
            logistic_temperature=1.0, probability_threshold=0.8,
            close_radius=0, max_hole_area=0,
        )
        baseline = classical_soft_foreground(
            image, ClassicalSoftForegroundConfig(**common)
        )
        extended = classical_soft_foreground(
            image,
            ClassicalSoftForegroundConfig(
                **common, low_probability_threshold=0.4
            ),
        )
        baseline_terminal = int(baseline.cleaned_mask[:, 85:120].sum())
        extended_terminal = int(extended.cleaned_mask[:, 85:120].sum())
        self.assertGreater(extended_terminal, baseline_terminal + 250)
        self.assertFalse(bool(extended.cleaned_mask[10:17, 8:20].any()))
        self.assertTrue(bool(extended.qc["hysteresis_enabled"]))
        self.assertGreater(int(extended.qc["hysteresis_recovered_area"]), 250)
        self.assertGreater(int(extended.qc["hysteresis_disconnected_low_area"]), 0)

    def test_invalid_low_threshold_is_rejected(self) -> None:
        image = np.zeros((10, 10), dtype=np.uint8)
        for value in (0.0, 0.5, 0.8):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "low_probability_threshold"):
                    classical_soft_foreground(
                        image,
                        ClassicalSoftForegroundConfig(
                            probability_threshold=0.5,
                            low_probability_threshold=value,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
