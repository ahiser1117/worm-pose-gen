from __future__ import annotations

import numpy as np
import unittest

from scripts.exp_smc_001_002_audit import (
    complete_curve_metrics,
    evaluate_exp_smc_001_gate,
    evaluate_exp_smc_002_gate,
    summarize_anchor_rows,
    summarize_segmentation_rows,
)


SEGMENTATION_GATE = {
    "median_cleaned_mask_trace_containment_min": 0.95,
    "p10_cleaned_mask_trace_containment_min": 0.90,
    "median_terminal_containment_min": 0.90,
    "median_soft_probability_on_trace_min": 0.80,
    "non_identifiable_evidence_used_max": 0,
}

ANCHOR_GATE = {
    "accepted_complete_median_frame_point_error_px_max": 8.0,
    "accepted_complete_p95_frame_point_error_px_max": 20.0,
    "accepted_complete_median_frame_tangent_error_deg_max": 15.0,
    "accepted_complete_median_frame_endpoint_error_px_max": 15.0,
    "accepted_complete_median_frame_length_error_fraction_max": 0.08,
    "accepted_complete_fraction_individually_at_most_8px_min": 0.90,
    "accepted_complete_frames_at_or_above_20px_max": 0,
    "truncated_rejection_fraction_min": 0.90,
}


def _segmentation_row(containment: float, *, trace_state: str = "complete") -> dict:
    return {
        "trace_state": trace_state,
        "used_as_evidence": trace_state != "not_identifiable",
        "cleaned_mask_trace_containment": containment if trace_state != "not_identifiable" else None,
        "terminal_containment": 0.95 if trace_state != "not_identifiable" else None,
        "soft_probability_on_trace": 0.90 if trace_state != "not_identifiable" else None,
        "median_nearest_cleaned_mask_distance_px": 0.0 if trace_state != "not_identifiable" else None,
        "terminal_omission_fraction": 0.05 if trace_state != "not_identifiable" else None,
        "cleaned_mask_area_px": 1000,
        "component_count": 1,
        "hole_count": 0,
        "boundary_contact_pixels": 0,
        "adjacent_area_relative_change": [0.01, 0.02],
        "adjacent_centroid_displacement_px": [1.0, 2.0],
        "segmentation_qc": {},
    }


def _complete_metrics(error: float) -> dict:
    return {
        "median_point_distance_px": error,
        "mean_tangent_error_deg": 10.0,
        "mean_endpoint_error_px": 10.0,
        "body_length_error_fraction": 0.05,
    }


def _anchor_row(
    *, accepted: bool, trace_state: str, error: float | None = None, stratum: str = "ordinary"
) -> dict:
    return {
        "accepted": accepted,
        "trace_state": trace_state,
        "selection_stratum": stratum,
        "complete_metrics": _complete_metrics(error) if accepted and trace_state == "complete" else None,
        "visible_metrics": (
            {"median_visible_trace_distance_px": 3.0}
            if accepted and trace_state == "truncated"
            else None
        ),
        "mask_render_iou_trace_width_proxy": 0.8 if accepted else None,
        "rejection_reasons": [] if accepted else ["endpoint_count"],
    }


class SMCAuditGateTests(unittest.TestCase):
    def test_exp_smc_001_gate_pass_is_capped_at_partially_supported(self) -> None:
        rows = [_segmentation_row(value) for value in [0.96] * 9 + [0.95]]
        rows.append(_segmentation_row(0.0, trace_state="not_identifiable"))
        summary = summarize_segmentation_rows(rows)
        decision = evaluate_exp_smc_001_gate(summary, SEGMENTATION_GATE)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["decision"], "PARTIALLY_SUPPORTED")
        self.assertEqual(decision["evidence_ceiling"], "PARTIALLY_SUPPORTED")
        self.assertEqual(summary["non_identifiable_evidence_used"], 0)

    def test_exp_smc_001_gate_fails_p10_and_non_identifiable_leakage(self) -> None:
        rows = [_segmentation_row(value) for value in [0.99] * 8 + [0.40, 0.40]]
        invalid = _segmentation_row(0.0, trace_state="not_identifiable")
        invalid["used_as_evidence"] = True
        rows.append(invalid)
        decision = evaluate_exp_smc_001_gate(
            summarize_segmentation_rows(rows), SEGMENTATION_GATE
        )
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["checks"]["p10_cleaned_mask_trace_containment"])
        self.assertFalse(decision["checks"]["non_identifiable_evidence_used"])

    def test_exp_smc_001_structural_revision_can_leave_soft_score_diagnostic(self) -> None:
        rows = [_segmentation_row(0.98) for _ in range(10)]
        for row in rows:
            row["soft_probability_on_trace"] = 0.1
            row["segmentation_qc"] = {
                "hysteresis_enabled": True,
                "largest_high_confidence_component_area": 1000,
                "hysteresis_recovered_area": 100,
            }
        gate = {
            **SEGMENTATION_GATE,
            "median_soft_probability_on_trace_min": None,
            "median_hysteresis_recovered_area_fraction_max": 0.15,
            "p95_hysteresis_recovered_area_fraction_max": 0.30,
            "p95_adjacent_area_relative_change_max": 0.05,
        }
        summary = summarize_segmentation_rows(rows)
        decision = evaluate_exp_smc_001_gate(summary, gate)
        self.assertTrue(decision["passed"])
        self.assertEqual(
            summary["hysteresis_recovered_area_fraction"]["median"], 0.1
        )
        self.assertTrue(decision["checks"]["median_soft_probability_on_trace"])

        rows[0]["segmentation_qc"]["hysteresis_recovered_area"] = 1000
        failed = evaluate_exp_smc_001_gate(
            summarize_segmentation_rows(rows), gate
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(
            failed["checks"]["p95_hysteresis_recovered_area_fraction"]
        )

    def test_exp_smc_002_gate_passes_with_secondary_coverage(self) -> None:
        rows = [
            _anchor_row(accepted=True, trace_state="complete", error=6.0)
            for _ in range(9)
        ]
        rows.append(_anchor_row(accepted=True, trace_state="complete", error=8.0))
        rows.extend(_anchor_row(accepted=False, trace_state="complete") for _ in range(10))
        rows.extend(_anchor_row(accepted=False, trace_state="truncated") for _ in range(10))
        summary = summarize_anchor_rows(rows)
        decision = evaluate_exp_smc_002_gate(summary, ANCHOR_GATE)
        self.assertAlmostEqual(summary["coverage"], 1 / 3)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["decision"], "SUPPORTED")

    def test_exp_smc_002_gate_fails_outlier_and_truncated_acceptance(self) -> None:
        rows = [
            _anchor_row(accepted=True, trace_state="complete", error=5.0)
            for _ in range(9)
        ]
        rows.append(_anchor_row(accepted=True, trace_state="complete", error=20.0))
        rows.extend(_anchor_row(accepted=False, trace_state="truncated") for _ in range(8))
        rows.extend(_anchor_row(accepted=True, trace_state="truncated") for _ in range(2))
        decision = evaluate_exp_smc_002_gate(summarize_anchor_rows(rows), ANCHOR_GATE)
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["checks"]["frames_at_or_above_20px"])
        self.assertFalse(decision["checks"]["truncated_rejection_fraction"])

    def test_exp_smc_002_gate_cannot_pass_without_accepted_complete_anchor(self) -> None:
        rows = [_anchor_row(accepted=False, trace_state="complete")]
        rows.extend(_anchor_row(accepted=False, trace_state="truncated") for _ in range(10))
        decision = evaluate_exp_smc_002_gate(summarize_anchor_rows(rows), ANCHOR_GATE)
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["checks"]["has_accepted_complete"])

    def test_complete_curve_metric_is_orientation_symmetric(self) -> None:
        target = np.column_stack((np.arange(10, dtype=float), np.zeros(10)))
        metrics = complete_curve_metrics(target[::-1], target, num_points=10)
        self.assertEqual(metrics["metric_orientation"], "symmetric")
        self.assertEqual(metrics["median_point_distance_px"], 0.0)
        self.assertEqual(metrics["mean_tangent_error_deg"], 0.0)
