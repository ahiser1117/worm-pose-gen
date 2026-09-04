"""Synthetic validation of visible-only recovery at camera boundaries."""

from __future__ import annotations

import unittest

import numpy as np

from tests.synthetic_edge_fixtures import (
    edge_region,
    make_synthetic_edge_cases,
    nearest_curve_distance,
)
from worm_pose_gen.classical import _skeleton_longest_path, _thin
from worm_pose_gen.edge_censored import repair_edge_censored_centerline
from worm_pose_gen.fov_completion import classify_boundary_truncation


class SyntheticEdgeCensoredRecoveryTests(unittest.TestCase):
    def test_cropped_curved_tubes_recover_only_the_visible_midline(self) -> None:
        """Exercise four sides and an oblique corner against known truth."""

        raw_edge_errors: list[float] = []
        repaired_edge_errors: list[float] = []
        for case in make_synthetic_edge_cases():
            with self.subTest(case=case.name):
                truncation = classify_boundary_truncation(
                    case.mask, close_radius=2, worm_radius_px=7.0
                )
                self.assertEqual(truncation.state, "one_end_truncated")
                self.assertEqual(
                    set(truncation.contacts[0].sides), set(case.expected_sides)
                )
                raw_path, _, _ = _skeleton_longest_path(_thin(case.mask))
                self.assertIsNotNone(raw_path)
                assert raw_path is not None

                repaired = repair_edge_censored_centerline(
                    raw_path,
                    case.mask.shape,
                    truncation,
                    raw_mask=case.mask,
                    edge_band_px=truncation.edge_band_px,
                    smoothness=2.0,
                    context_points=10,
                    min_core_points=8,
                    min_core_length_px=20.0,
                )
                self.assertTrue(repaired.success, repaired.failure_reason)
                self.assertEqual(repaired.centerline_xy.shape, raw_path.shape)
                self.assertEqual(len(repaired.censored_ends), 1)
                self.assertEqual(int(repaired.censored_endpoint_mask.sum()), 1)

                # A censored endpoint is a point on the camera boundary, not a
                # license to reconstruct any anatomy beyond the observation.
                height, width = case.mask.shape
                recovered = repaired.centerline_xy
                self.assertTrue(np.isfinite(recovered).all())
                self.assertTrue(np.all(recovered[:, 0] >= 0.0))
                self.assertTrue(np.all(recovered[:, 0] <= width - 1.0))
                self.assertTrue(np.all(recovered[:, 1] >= 0.0))
                self.assertTrue(np.all(recovered[:, 1] <= height - 1.0))
                self.assertTrue(repaired.observed_support.all())

                # The core deliberately excludes the cap-biased edge band.
                raw_in_edge_band = edge_region(
                    raw_path, case.mask.shape, truncation.edge_band_px
                )
                self.assertFalse(
                    np.any(repaired.reliable_core_mask & raw_in_edge_band)
                )
                self.assertGreaterEqual(int(repaired.reliable_core_mask.sum()), 8)

                # Directed truth-to-curve error measures whether the visible
                # centerline is covered all the way to the camera boundary;
                # a skeleton that terminates at the truncated cap scores badly.
                truth_edge = case.visible_truth_xy[
                    edge_region(
                        case.visible_truth_xy,
                        case.mask.shape,
                        truncation.edge_band_px,
                    )
                ]
                raw_error = float(
                    np.percentile(nearest_curve_distance(truth_edge, raw_path), 90)
                )
                repaired_error = float(
                    np.percentile(nearest_curve_distance(truth_edge, recovered), 90)
                )
                raw_edge_errors.append(raw_error)
                repaired_edge_errors.append(repaired_error)
                self.assertLess(repaired_error, 3.0)

        # Require a material aggregate improvement without making every
        # raster orientation win independently (thinning is grid-anisotropic).
        self.assertLess(
            float(np.mean(repaired_edge_errors)),
            0.8 * float(np.mean(raw_edge_errors)),
            (raw_edge_errors, repaired_edge_errors),
        )

    def test_fully_visible_curve_is_left_unchanged(self) -> None:
        case = make_synthetic_edge_cases()[0]
        mask = case.mask.copy()
        # Removing the edge portion turns this into an ordinary anatomical
        # endpoint sufficiently far from every camera side.
        mask[:, :20] = False
        truncation = classify_boundary_truncation(mask, worm_radius_px=7.0)
        self.assertEqual(truncation.state, "fully_visible")
        raw_path, _, _ = _skeleton_longest_path(_thin(mask))
        assert raw_path is not None
        result = repair_edge_censored_centerline(
            raw_path, mask.shape, truncation, raw_mask=mask
        )
        # This repair primitive applies only to classified camera censoring;
        # ordinary anatomical endpoints fail closed instead of being altered.
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "no_endpoint_boundary_contact")
        self.assertIsNone(result.centerline_xy)
        self.assertEqual(result.censored_ends, ())
        self.assertIsNone(result.censored_endpoint_mask)


if __name__ == "__main__":
    unittest.main()
