import unittest

import numpy as np

from scripts.evaluate_tier_a_primary import (
    complete_curve_metrics,
    visible_trace_metrics,
)


class TierAPrimaryEvaluationTests(unittest.TestCase):
    def test_complete_metrics_are_orientation_symmetric(self):
        target = np.column_stack((np.linspace(10, 50, 20), np.linspace(5, 25, 20)))
        result = complete_curve_metrics(target[::-1], target)
        self.assertTrue(result["target_reversed"])
        self.assertLess(result["p95_point_distance_px"], 1e-10)
        self.assertLess(result["mean_tangent_error_deg"], 1e-10)

    def test_visible_metric_does_not_penalize_unannotated_extent(self):
        prediction = np.column_stack((np.linspace(0, 100, 101), np.zeros(101)))
        visible = np.column_stack((np.linspace(25, 75, 21), np.zeros(21)))
        result = visible_trace_metrics(prediction, visible)
        self.assertLess(result["p95_visible_trace_distance_px"], 1e-10)
        self.assertLess(result["mean_visible_trace_axis_error_deg"], 1e-10)

    def test_visible_metric_detects_offset(self):
        prediction = np.column_stack((np.linspace(0, 100, 101), np.zeros(101)))
        visible = np.column_stack((np.linspace(25, 75, 21), np.full(21, 7.0)))
        result = visible_trace_metrics(prediction, visible)
        self.assertAlmostEqual(result["median_visible_trace_distance_px"], 7.0, places=6)


if __name__ == "__main__":
    unittest.main()
