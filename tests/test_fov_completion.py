import unittest

import numpy as np

from worm_pose_gen.fov_completion import (
    build_boundary_stable_skeleton,
    classify_boundary_truncation,
    complete_centerline_to_length,
)


def horizontal_tube(left: int, right: int, *, height: int = 50, width: int = 80) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    nearest_x = np.clip(xx, left, right)
    return (xx - nearest_x) ** 2 + (yy - 25) ** 2 <= 4**2


class BoundaryClassificationTests(unittest.TestCase):
    def test_fully_visible(self) -> None:
        result = classify_boundary_truncation(horizontal_tube(15, 65), worm_radius_px=4)
        self.assertEqual(result.state, "fully_visible")
        self.assertEqual(result.contacts, ())

    def test_raw_mask_detects_contact_that_closing_can_move_inboard(self) -> None:
        mask = horizontal_tube(0, 55)
        result = classify_boundary_truncation(mask, close_radius=2, worm_radius_px=4)
        self.assertEqual(result.state, "one_end_truncated")
        self.assertEqual(len(result.contacts), 1)
        self.assertEqual(result.contacts[0].sides, ("left",))
        self.assertEqual(result.diagnostics["classification_basis"], "raw_pre_morphology_mask")

    def test_two_sided_and_corner_contacts(self) -> None:
        both = classify_boundary_truncation(horizontal_tube(0, 79), worm_radius_px=4)
        self.assertEqual(both.state, "two_sided_truncated")
        self.assertEqual(set(both.contact_ends), {"start", "end"})

        yy, xx = np.mgrid[:60, :60]
        # Diagonal tube clipped through the top-left corner.
        corner = np.abs(yy - xx) <= 3
        corner &= xx <= 45
        classified = classify_boundary_truncation(corner, worm_radius_px=4)
        self.assertEqual(classified.state, "one_end_truncated")
        self.assertEqual(set(classified.contacts[0].sides), {"top", "left"})

    def test_side_contact_without_endpoint_is_uncertain(self) -> None:
        mask = horizontal_tube(15, 65)
        mask[:26, 36:45] = True  # T-shaped mid-body bridge to the top boundary.
        result = classify_boundary_truncation(mask, worm_radius_px=4)
        self.assertEqual(result.state, "boundary_uncertain")


class StableSkeletonTests(unittest.TestCase):
    def test_virtual_tube_places_path_beyond_fov(self) -> None:
        mask = horizontal_tube(0, 55)
        truncation = classify_boundary_truncation(mask, worm_radius_px=4)
        result = build_boundary_stable_skeleton(
            mask, truncation, extension_px=12, tube_radius_px=4
        )
        self.assertIsNotNone(result.centerline_xy)
        assert result.centerline_xy is not None
        self.assertLess(float(result.centerline_xy[:, 0].min()), 0.0)
        self.assertGreater(len(result.centerline_xy), len(result.visible_centerline_xy))
        self.assertEqual(result.diagnostics["virtual_contact_count"], 1)

    def test_virtual_tube_follows_oblique_endpoint_tangent(self) -> None:
        yy, xx = np.mgrid[:70, :70]
        # Tube around y = 0.5*x + 10, clipped at the left boundary.
        distance = np.abs(yy - (0.5 * xx + 10.0)) / np.sqrt(1.25)
        mask = (distance <= 3.0) & (xx <= 55)
        truncation = classify_boundary_truncation(mask, worm_radius_px=3)
        self.assertEqual(truncation.state, "one_end_truncated")
        result = build_boundary_stable_skeleton(
            mask, truncation, extension_px=12, tube_radius_px=3
        )
        assert result.centerline_xy is not None
        outside = result.centerline_xy[result.centerline_xy[:, 0] < 0]
        self.assertGreater(len(outside), 2)
        # Pure edge-normal extension would have constant y; the collar should
        # instead continue the observed diagonal.
        self.assertGreater(float(np.ptp(outside[:, 1])), 2.0)


class CompletionTests(unittest.TestCase):
    def test_one_contact_gets_all_missing_length_and_support_metadata(self) -> None:
        mask = horizontal_tube(0, 55)
        truncation = classify_boundary_truncation(mask, worm_radius_px=4)
        visible = np.column_stack((np.linspace(0, 50, 26), np.full(26, 25.0)))
        result = complete_centerline_to_length(visible, mask.shape, 70.0, truncation)
        self.assertTrue(result.complete)
        self.assertFalse(result.ambiguous)
        self.assertAlmostEqual(result.diagnostics["completed_length_px"], 70.0, places=6)
        self.assertEqual(int(result.observed_support.sum()), len(visible))
        self.assertTrue(np.all(result.observed_support[result.in_fov]))
        self.assertLess(float(result.centerline_xy[:, 0].min()), 0.0)

    def test_curve_orientation_is_independent_of_raw_skeleton_orientation(self) -> None:
        mask = horizontal_tube(0, 55)
        truncation = classify_boundary_truncation(mask, worm_radius_px=4)
        visible = np.column_stack((np.linspace(0, 50, 26), np.full(26, 25.0)))[::-1]
        result = complete_centerline_to_length(visible, mask.shape, 70.0, truncation)
        self.assertTrue(result.complete)
        self.assertGreater(float(result.diagnostics["end_extension_px"]), 0.0)

    def test_two_contacts_require_explicit_split(self) -> None:
        mask = horizontal_tube(0, 79)
        truncation = classify_boundary_truncation(mask, worm_radius_px=4)
        visible = np.column_stack((np.linspace(0, 79, 80), np.full(80, 25.0)))
        ambiguous = complete_centerline_to_length(visible, mask.shape, 99.0, truncation)
        self.assertTrue(ambiguous.ambiguous)
        self.assertFalse(ambiguous.complete)
        np.testing.assert_array_equal(ambiguous.centerline_xy, visible)

        completed = complete_centerline_to_length(
            visible,
            mask.shape,
            99.0,
            truncation,
            missing_length_by_end={"start": 7.0, "end": 13.0},
        )
        self.assertTrue(completed.complete)
        self.assertAlmostEqual(completed.diagnostics["completed_length_px"], 99.0, places=6)
        self.assertEqual(completed.diagnostics["start_extension_px"], 7.0)
        self.assertEqual(completed.diagnostics["end_extension_px"], 13.0)

    def test_fully_visible_curve_is_not_unjustifiably_extended(self) -> None:
        mask = horizontal_tube(15, 65)
        truncation = classify_boundary_truncation(mask, worm_radius_px=4)
        visible = np.column_stack((np.linspace(15, 65, 26), np.full(26, 25.0)))
        result = complete_centerline_to_length(visible, mask.shape, 70.0, truncation)
        self.assertTrue(result.ambiguous)
        self.assertFalse(result.complete)
        self.assertEqual(result.diagnostics["reason"], "no_unique_censored_endpoint")


if __name__ == "__main__":
    unittest.main()
