import unittest

import torch

from scripts.aggregate_exp_003 import _paired_bootstrap_difference
from scripts.evaluate_exp_003_tier_c import _case_metrics, _heatmap_argmax, _oracle_candidate
from scripts.evaluate_exp_003_tier_a import summarize


class Exp003EvaluationTests(unittest.TestCase):
    def test_controlled_decoder_diagnostics(self) -> None:
        logits = torch.zeros((1, 2, 3, 4))
        logits[0, 0, 1, 2] = 4
        logits[0, 1, 2, 3] = 5
        decoded = _heatmap_argmax(logits, image_height=6, image_width=8)
        torch.testing.assert_close(
            decoded, torch.tensor([[[5.0, 3.0], [7.0, 5.0]]]), rtol=0, atol=0
        )
        target = torch.zeros((1, 100, 2))
        candidates = torch.stack((target + 10, target + 2, target + 7), dim=1)
        torch.testing.assert_close(_oracle_candidate(candidates, target), target + 2)

    def test_tier_c_metrics_are_orientation_symmetric(self) -> None:
        target = torch.stack((
            torch.linspace(0, 99, 100),
            torch.linspace(10, 30, 100),
        ), dim=-1).unsqueeze(0)
        support = torch.ones((1, 100), dtype=torch.bool)
        forward = _case_metrics(target, target, support)[0]
        reverse = _case_metrics(target.flip(-2), target, support)[0]
        self.assertEqual(forward["median_full_latent_point_distance_px"], 0.0)
        self.assertEqual(reverse["median_full_latent_point_distance_px"], 0.0)
        self.assertLess(reverse["mean_full_latent_tangent_error_deg"], 1e-4)

    def test_paired_bootstrap_identity(self):
        result = _paired_bootstrap_difference([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], seed=7, resamples=50)
        self.assertEqual(result["candidate_minus_global_median_px"], 0.0)
        self.assertEqual(result["bootstrap_p2_5_px"], 0.0)
        self.assertEqual(result["bootstrap_p97_5_px"], 0.0)

    def test_summary_keeps_complete_and_truncated_metrics_separate(self):
        complete = {
            "median_point_distance_px": 2.0,
            "point_distance_px": [1.0, 3.0],
            "mean_tangent_error_deg": 4.0,
            "mean_endpoint_error_px": 5.0,
            "body_length_error_fraction": 0.1,
        }
        visible = {
            "median_visible_trace_distance_px": 6.0,
            "visible_trace_distance_px": [5.0, 7.0],
            "mean_visible_trace_axis_error_deg": 8.0,
        }
        result = summarize([
            {"algorithmic_success": True, "complete_metrics": complete, "visible_metrics": None},
            {"algorithmic_success": True, "complete_metrics": None, "visible_metrics": visible},
            {"algorithmic_success": False, "complete_metrics": None, "visible_metrics": None},
        ])
        self.assertEqual(result["algorithmic_failure_frames"], 1)
        self.assertEqual(result["complete_trace_scored_frames"], 1)
        self.assertEqual(result["truncated_visible_trace_scored_frames"], 1)


if __name__ == "__main__":
    unittest.main()
