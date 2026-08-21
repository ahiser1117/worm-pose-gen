import unittest

import numpy as np

from worm_pose_gen.anchors import (
    AnchorConfig,
    extend_centerline_to_mask_boundary,
    extract_mask_anchor,
)
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

    def test_centerline_extension_reaches_both_straight_mask_boundaries(self) -> None:
        mask = np.zeros((31, 51), dtype=bool)
        mask[10:21, 5:46] = True
        centerline = np.column_stack((np.linspace(14, 36, 12), np.full(12, 15.0)))

        extended = extend_centerline_to_mask_boundary(centerline, mask)

        np.testing.assert_array_equal(extended[extended[:, 0] == 14][0], centerline[0])
        np.testing.assert_array_equal(extended[extended[:, 0] == 36][0], centerline[-1])
        self.assertAlmostEqual(float(extended[0, 0]), 4.5, places=5)
        self.assertAlmostEqual(float(extended[-1, 0]), 45.5, places=5)
        np.testing.assert_allclose(extended[:, 1], 15.0, atol=1e-10)

    def test_centerline_extension_continues_curvature_at_both_ends(self) -> None:
        height = width = 101
        yy, xx = np.mgrid[:height, :width]
        dx, dy = xx - 50.0, yy - 50.0
        radius = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx)
        mask = (np.abs(radius - 25.0) <= 4.0) & (np.abs(angle) <= 1.0)
        source_angle = np.linspace(-0.45, 0.45, 19)
        centerline = np.column_stack(
            (50.0 + 25.0 * np.cos(source_angle), 50.0 + 25.0 * np.sin(source_angle))
        )

        extended = extend_centerline_to_mask_boundary(
            centerline, mask, context_points=9, step=0.2
        )

        terminal_radius = np.hypot(extended[[0, -1], 0] - 50, extended[[0, -1], 1] - 50)
        terminal_angle = np.arctan2(
            extended[[0, -1], 1] - 50, extended[[0, -1], 0] - 50
        )
        np.testing.assert_allclose(terminal_radius, 25.0, atol=0.08)
        self.assertLess(float(terminal_angle[0]), -0.97)
        self.assertGreater(float(terminal_angle[1]), 0.97)

        # A terminal tangent ray would drift radially outward, demonstrating
        # that the observed result used local curvature rather than a line.
        tail_added_length = np.linalg.norm(extended[-1] - centerline[-1])
        tangent = centerline[-1] - centerline[-2]
        tangent /= np.linalg.norm(tangent)
        tangent_endpoint = centerline[-1] + tail_added_length * tangent
        tangent_radius = np.linalg.norm(tangent_endpoint - np.asarray([50.0, 50.0]))
        self.assertGreater(tangent_radius, 25.5)

    def test_centerline_extension_stops_at_first_exit_and_is_reversal_invariant(self) -> None:
        mask = np.zeros((25, 55), dtype=bool)
        mask[7:18, 5:17] = True
        mask[7:18, 25:50] = True  # A separate foreground island after a gap.
        centerline = np.column_stack((np.linspace(8, 14, 7), np.full(7, 12.0)))

        extended = extend_centerline_to_mask_boundary(centerline, mask)
        reversed_extended = extend_centerline_to_mask_boundary(centerline[::-1], mask)

        self.assertAlmostEqual(float(extended[0, 0]), 4.5, places=5)
        self.assertAlmostEqual(float(extended[-1, 0]), 16.5, places=5)
        self.assertLess(float(extended[:, 0].max()), 25.0)
        np.testing.assert_allclose(extended, reversed_extended[::-1], atol=1e-10)

    def test_centerline_extension_validates_support_and_guard(self) -> None:
        mask = np.ones((20, 20), dtype=bool)
        line = np.column_stack((np.linspace(5, 10, 6), np.full(6, 10.0)))
        outside = line.copy()
        outside[0] = [-1.0, 10.0]
        with self.assertRaisesRegex(ValueError, "endpoints must lie inside"):
            extend_centerline_to_mask_boundary(outside, mask)
        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            extend_centerline_to_mask_boundary(line, mask, max_extension=1.0)


if __name__ == "__main__":
    unittest.main()
