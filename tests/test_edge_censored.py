import unittest

import numpy as np

from worm_pose_gen.edge_censored import repair_edge_censored_centerline
from worm_pose_gen.fov_completion import (
    BoundaryContact,
    BoundaryTruncationResult,
)


def truncation(mask, contacts, state="one_end_truncated", band=10):
    ends = tuple(contact.endpoint for contact in contacts if contact.endpoint is not None)
    return BoundaryTruncationResult(
        state, tuple(contacts), ends, band, mask, {"fixture": True}
    )


class EdgeCensoredRepairTest(unittest.TestCase):
    def test_repairs_start_to_raw_mask_crossing_without_off_fov_points(self):
        mask = np.zeros((80, 100), dtype=bool)
        mask[47:54, :76] = True
        x = np.linspace(0, 75, 31)
        y = np.full_like(x, 50.0)
        y[:5] += np.asarray((8.0, 6.0, 4.0, 2.0, 1.0))
        path = np.column_stack((x, y))
        # Deliberately use the opposite raw-skeleton label. Association must use
        # contact geometry relative to the supplied A3 ordering.
        contact = BoundaryContact((3.0, 50.0), ("left",), "end", 40, 1.0)

        result = repair_edge_censored_centerline(
            path, mask.shape, truncation(mask, [contact]), min_core_length_px=15
        )

        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual(result.censored_ends, ("start",))
        self.assertIsNotNone(result.centerline_xy)
        assert result.centerline_xy is not None
        self.assertEqual(len(result.centerline_xy), len(path))
        self.assertAlmostEqual(result.centerline_xy[0, 0], 0.0)
        self.assertAlmostEqual(result.centerline_xy[0, 1], 50.0, delta=0.6)
        self.assertTrue(np.all(result.centerline_xy >= 0))
        self.assertLessEqual(float(result.centerline_xy[:, 0].max()), 99.0)
        assert result.reliable_core_mask is not None
        self.assertFalse(result.reliable_core_mask[0])
        self.assertTrue(result.reliable_core_mask[-1])
        assert result.censored_endpoint_mask is not None
        np.testing.assert_array_equal(
            np.flatnonzero(result.censored_endpoint_mask), np.asarray((0,))
        )

    def test_two_sided_repair_terminates_on_both_camera_boundaries(self):
        mask = np.zeros((60, 120), dtype=bool)
        mask[27:34, :] = True
        x = np.linspace(0, 119, 61)
        path = np.column_stack((x, 30.0 + 0.01 * (x - 60.0)))
        contacts = [
            BoundaryContact((2.0, 30.0), ("left",), "start", 30, 0.0),
            BoundaryContact((117.0, 30.0), ("right",), "end", 30, 0.0),
        ]

        result = repair_edge_censored_centerline(
            path,
            mask.shape,
            truncation(mask, contacts, "two_sided_truncated"),
            min_core_length_px=30,
        )

        self.assertTrue(result.success, result.failure_reason)
        assert result.centerline_xy is not None
        self.assertAlmostEqual(result.centerline_xy[0, 0], 0.0)
        self.assertAlmostEqual(result.centerline_xy[-1, 0], 119.0)
        self.assertEqual(set(result.boundary_crossings_xy), {"start", "end"})
        assert result.censored_endpoint_mask is not None
        self.assertTrue(result.censored_endpoint_mask[0])
        self.assertTrue(result.censored_endpoint_mask[-1])

    def test_short_reliable_core_fails_closed(self):
        mask = np.ones((30, 30), dtype=bool)
        x = np.linspace(0, 29, 20)
        path = np.column_stack((x, np.full_like(x, 15.0)))
        contacts = [
            BoundaryContact((1.0, 15.0), ("left",), "start", 10, 0.0),
            BoundaryContact((28.0, 15.0), ("right",), "end", 10, 0.0),
        ]
        result = repair_edge_censored_centerline(
            path,
            mask.shape,
            truncation(mask, contacts, "two_sided_truncated", band=13),
            min_core_points=8,
        )
        self.assertFalse(result.success)
        self.assertIn(
            result.failure_reason, {"insufficient_core_points", "insufficient_core_length"}
        )
        self.assertIsNone(result.centerline_xy)

    def test_no_contact_fails_closed(self):
        mask = np.zeros((50, 50), dtype=bool)
        path = np.column_stack((np.linspace(15, 35, 12), np.full(12, 25.0)))
        result = repair_edge_censored_centerline(
            path, mask.shape, truncation(mask, [], "fully_visible")
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "no_endpoint_boundary_contact")


if __name__ == "__main__":
    unittest.main()
