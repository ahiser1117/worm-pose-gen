import unittest

import numpy as np

from worm_pose_gen.anchors import AnchorConfig, extract_mask_anchor
from worm_pose_gen.segmentation import fill_small_enclosed_holes


def easy_tube() -> np.ndarray:
    yy, xx = np.mgrid[:80, :140]
    closest_x = np.clip(xx, 20, 120)
    return (xx - closest_x) ** 2 + (yy - 40) ** 2 <= 6**2


class AnchorTests(unittest.TestCase):
    def config(self) -> AnchorConfig:
        return AnchorConfig(
            min_area=500,
            max_area=5_000,
            min_length=70,
            max_length=130,
            boundary_margin=5,
            min_width=8,
            max_width=20,
            max_width_jump=5,
            min_render_iou=0.75,
        )

    def test_easy_tube_has_width_and_high_render_iou(self) -> None:
        probability = np.where(easy_tube(), 0.95, 0.05)
        result = extract_mask_anchor(easy_tube(), probability, self.config())
        self.assertTrue(result.accepted, result.rejection_reasons)
        self.assertEqual(result.centerline_xy.shape, (100, 2))
        self.assertEqual(result.estimated_width.shape, (100,))
        self.assertGreater(float(np.median(result.estimated_width)), 10.0)
        self.assertLess(float(np.median(result.estimated_width)), 16.0)
        self.assertGreater(float(result.qc["mask_render_iou"]), 0.75)
        self.assertAlmostEqual(result.head_tail_probability, 0.5)
        self.assertEqual(result.qc["endpoint_count"], 2)
        self.assertEqual(result.qc["branch_pixels"], 0)
        # Unlike the legacy proxy extractor, mask-native topology assessment
        # must not peel a fixed eight pixels from each anatomical endpoint.
        self.assertLessEqual(float(result.centerline_xy[:, 0].min()), 22.0)
        self.assertGreaterEqual(float(result.centerline_xy[:, 0].max()), 118.0)
        self.assertGreater(float(result.qc["mean_probability_in_mask"]), 0.9)

    def test_branch_is_rejected_with_raw_topology_qc(self) -> None:
        mask = easy_tube()
        mask[10:41, 65:76] = True
        result = extract_mask_anchor(mask, config=self.config())
        self.assertFalse(result.accepted)
        self.assertGreater(int(result.qc["branch_pixels"]), 0)
        self.assertIn("branch_pixels", result.rejection_reasons)
        self.assertIn("mask_render_iou", result.qc)
        self.assertGreater(int(result.qc["raw_branch_pixels"]), 0)

    def test_crossing_is_rejected(self) -> None:
        yy, xx = np.mgrid[:100, :100]
        crossing = (np.abs(yy - xx) <= 4) | (np.abs(yy - (99 - xx)) <= 4)
        crossing &= (xx >= 12) & (xx <= 87) & (yy >= 12) & (yy <= 87)
        result = extract_mask_anchor(
            crossing,
            config=AnchorConfig(min_area=100, max_area=5_000, min_length=20, max_length=500),
        )
        self.assertFalse(result.accepted)
        self.assertTrue(
            int(result.qc["branch_pixels"]) > 0 or int(result.qc["endpoint_count"]) != 2
        )

    def test_short_side_spur_is_assessment_only_and_keeps_body_endpoints(self) -> None:
        mask = easy_tube()
        mask[28:35, 68:73] = True
        config = AnchorConfig(
            min_area=500, max_area=5_000, min_length=70, max_length=130,
            boundary_margin=5, min_width=8, max_width=20,
            max_width_jump=8, min_render_iou=0.7,
            max_topology_spur_length=10,
        )
        result = extract_mask_anchor(mask, config=config)
        self.assertTrue(result.accepted, result.rejection_reasons)
        self.assertGreater(int(result.qc["raw_branch_pixels"]), 0)
        self.assertEqual(result.qc["branch_pixels"], 0)
        self.assertEqual(result.qc["topology_pruned_spur_count"], 1)
        # Curve recovery uses the original skeleton's longest path; pruning is
        # never endpoint peeling on the accepted anatomical path.
        self.assertLessEqual(float(result.centerline_xy[:, 0].min()), 22.0)
        self.assertGreaterEqual(float(result.centerline_xy[:, 0].max()), 118.0)

    def test_boundary_contact_is_rejected(self) -> None:
        mask = easy_tube()
        mask[34:47, :21] = True
        result = extract_mask_anchor(mask, config=self.config())
        self.assertFalse(result.accepted)
        self.assertTrue(bool(result.qc["mask_touches_boundary"]))
        self.assertIn("boundary_contact", result.rejection_reasons)

    def test_width_relative_boundary_clearance_is_opt_in_and_rejects(self) -> None:
        yy, xx = np.mgrid[:80, :140]
        closest_x = np.clip(xx, 20, 120)
        near_boundary = (xx - closest_x) ** 2 + (yy - 10) ** 2 <= 6**2

        # The backward-compatible default leaves the established absolute
        # boundary-margin behavior unchanged.
        default_result = extract_mask_anchor(near_boundary, config=self.config())
        self.assertTrue(default_result.accepted, default_result.rejection_reasons)
        self.assertEqual(
            float(default_result.qc["min_boundary_clearance_widths"]), 0.0
        )

        relative_config = AnchorConfig(
            **{
                **self.config().__dict__,
                "min_boundary_clearance_widths": 1.0,
            }
        )
        safe_result = extract_mask_anchor(easy_tube(), config=relative_config)
        self.assertTrue(safe_result.accepted, safe_result.rejection_reasons)
        self.assertGreaterEqual(
            float(safe_result.qc["centerline_boundary_distance_px"]),
            float(safe_result.qc["required_boundary_clearance_px"]),
        )

        result = extract_mask_anchor(near_boundary, config=relative_config)
        self.assertFalse(result.accepted)
        self.assertNotIn("boundary_contact", result.rejection_reasons)
        self.assertNotIn("boundary_margin", result.rejection_reasons)
        self.assertIn("width_relative_boundary_clearance", result.rejection_reasons)
        self.assertLess(
            float(result.qc["centerline_boundary_distance_px"]),
            float(result.qc["required_boundary_clearance_px"]),
        )
        self.assertLess(float(result.qc["boundary_clearance_width_ratio"]), 1.0)

    def test_width_relative_boundary_clearance_validates_multiplier(self) -> None:
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                config = AnchorConfig(
                    **{
                        **self.config().__dict__,
                        "min_boundary_clearance_widths": value,
                    }
                )
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    extract_mask_anchor(easy_tube(), config=config)

    def test_porous_easy_tube_becomes_acceptable_after_small_hole_fill(self) -> None:
        porous = easy_tube()
        porous[38:41, 45:48] = False
        porous[39:42, 82:84] = False
        cleaned, count, area = fill_small_enclosed_holes(porous, max_hole_area=12)
        self.assertEqual(count, 2)
        self.assertEqual(area, 15)
        result = extract_mask_anchor(cleaned, config=self.config())
        self.assertTrue(result.accepted, result.rejection_reasons)
        self.assertGreater(float(result.qc["mask_render_iou"]), 0.75)


if __name__ == "__main__":
    unittest.main()
