import math
import unittest

import torch

from worm_pose_gen.metrics import (
    FOVCropTransform,
    anatomical_support_mask,
    binary_brier_score,
    binary_calibration_bins,
    circular_angle_error,
    circular_angle_mae,
    expected_calibration_error,
    masked_point_mae,
    point_errors,
    support_regions,
)


class MetricTests(unittest.TestCase):
    def test_circular_error_at_wrap_boundary(self) -> None:
        epsilon = 0.02
        prediction = torch.tensor([math.pi - epsilon, -math.pi + epsilon])
        target = torch.tensor([-math.pi + epsilon, math.pi - epsilon])
        torch.testing.assert_close(circular_angle_error(prediction, target), torch.full((2,), 2 * epsilon))
        self.assertAlmostEqual(float(circular_angle_mae(prediction, target, degrees=True)), math.degrees(0.04), places=4)

    def test_masked_angle_and_point_metrics(self) -> None:
        prediction_angle = torch.tensor([0.0, 1.0, 2.0])
        target_angle = torch.tensor([0.1, -2.0, 2.2])
        mask = torch.tensor([True, False, True])
        self.assertAlmostEqual(float(circular_angle_mae(prediction_angle, target_angle, mask)), 0.15, places=6)
        target_xy = torch.zeros(3, 2)
        prediction_xy = torch.tensor([[3.0, 4.0], [0.0, 2.0], [6.0, 8.0]])
        torch.testing.assert_close(point_errors(prediction_xy, target_xy), torch.tensor([5.0, 2.0, 10.0]))
        self.assertAlmostEqual(float(masked_point_mae(prediction_xy, target_xy, mask)), 7.5)
        self.assertAlmostEqual(float(masked_point_mae(prediction_xy, target_xy, mask, normalization=5.0)), 1.5)
        with self.assertRaises(ValueError):
            masked_point_mae(prediction_xy, target_xy, torch.zeros(3, dtype=torch.bool))

    def test_support_calibration_primitives(self) -> None:
        probability = torch.tensor([0.05, 0.25, 0.75, 0.95])
        target = torch.tensor([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(float(binary_brier_score(probability, target)), 0.0325, places=6)
        bins = binary_calibration_bins(probability, target, num_bins=2)
        self.assertEqual(bins["count"].tolist(), [2, 2])
        torch.testing.assert_close(bins["confidence"], torch.tensor([0.15, 0.85]))
        torch.testing.assert_close(bins["accuracy"], torch.tensor([0.0, 1.0]))
        self.assertAlmostEqual(float(expected_calibration_error(bins)), 0.15, places=6)

    def test_binary_metrics_validate_evidence_and_masks(self) -> None:
        invalid_probabilities = (
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.tensor([-0.1]),
            torch.tensor([1.1]),
        )
        for probability in invalid_probabilities:
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    binary_brier_score(probability, torch.tensor([1.0]))
                with self.assertRaises(ValueError):
                    binary_calibration_bins(probability, torch.tensor([1.0]))
        for target in (
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.tensor([0.5]),
            torch.tensor([2.0]),
        ):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    binary_brier_score(torch.tensor([0.5]), target)
                with self.assertRaises(ValueError):
                    binary_calibration_bins(torch.tensor([0.5]), target)
        with self.assertRaisesRegex(ValueError, "not broadcastable"):
            binary_brier_score(torch.zeros(2), torch.zeros(3))
        with self.assertRaisesRegex(ValueError, "not broadcastable"):
            binary_calibration_bins(torch.zeros(2), torch.zeros(3))
        with self.assertRaisesRegex(ValueError, "mask is not broadcastable"):
            binary_brier_score(
                torch.tensor([0.2, 0.8]), torch.tensor([0.0, 1.0]),
                torch.ones(3, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "mask is not broadcastable"):
            binary_calibration_bins(
                torch.tensor([0.2, 0.8]), torch.tensor([0.0, 1.0]),
                mask=torch.ones(3, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "selects no values"):
            binary_brier_score(
                torch.tensor([0.2, 0.8]), torch.tensor([0.0, 1.0]),
                torch.zeros(2, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "selects no values"):
            binary_calibration_bins(
                torch.tensor([0.2, 0.8]), torch.tensor([0.0, 1.0]),
                mask=torch.zeros(2, dtype=torch.bool),
            )

    def test_binary_metrics_broadcast_shapes_deliberately(self) -> None:
        probability = torch.tensor([[0.0], [1.0]])
        target = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(float(binary_brier_score(probability, target)), 0.5)
        bins = binary_calibration_bins(probability, target, num_bins=2)
        self.assertEqual(bins["count"].tolist(), [2, 2])

    def test_anatomical_support_hidden_fractions(self) -> None:
        tail = anatomical_support_mask(100, 0.2, hidden_end="tail")
        self.assertEqual(int((~tail).sum()), 20)
        self.assertTrue(bool(tail[79]))
        self.assertFalse(bool(tail[80]))
        head = anatomical_support_mask(10, 0.3, hidden_end="head")
        self.assertEqual(head.tolist(), [False, False, False, True, True, True, True, True, True, True])
        regions = support_regions(tail, boundary_points=1)
        expected_boundary = torch.zeros(100, dtype=torch.bool)
        expected_boundary[79:81] = True
        torch.testing.assert_close(regions["boundary"], expected_boundary)
        self.assertFalse(bool((regions["visible"] & regions["hidden"]).any()))

    def test_support_regions_exact_five_per_side_and_multiple_transitions(self) -> None:
        support = anatomical_support_mask(100, 0.2, hidden_end="tail")
        regions = support_regions(support, boundary_points=5)
        expected = torch.zeros(100, dtype=torch.bool)
        expected[75:85] = True
        torch.testing.assert_close(regions["boundary"], expected)
        self.assertEqual(int(regions["boundary"].sum()), 10)

        alternating = torch.tensor(
            [True, True, False, False, True, True, False, False]
        )
        multiple = support_regions(alternating, boundary_points=1)
        expected_multiple = torch.tensor(
            [False, True, True, True, True, True, True, False]
        )
        torch.testing.assert_close(multiple["boundary"], expected_multiple)
        self.assertFalse(bool((multiple["visible"] & multiple["boundary"]).any()))
        self.assertFalse(bool((multiple["hidden"] & multiple["boundary"]).any()))

        no_band = support_regions(alternating, boundary_points=0)
        self.assertFalse(bool(no_band["boundary"].any()))

    def test_crop_transform_roundtrip_and_half_open_support(self) -> None:
        transform = FOVCropTransform(x0=10, y0=20, width=5, height=4)
        original = torch.tensor([[10.0, 20.0], [14.999, 23.999], [15.0, 22.0], [12.0, 24.0]])
        crop = transform.to_crop(original)
        torch.testing.assert_close(transform.to_original(crop), original)
        self.assertEqual(transform.support_mask(original).tolist(), [True, True, False, False])
        self.assertEqual(transform.as_dict(), {"x0": 10, "y0": 20, "width": 5, "height": 4})

    def test_metric_gradients(self) -> None:
        prediction = torch.tensor([math.pi - 0.1, 0.5], requires_grad=True)
        target = torch.tensor([-math.pi + 0.1, 0.0])
        loss = circular_angle_mae(prediction, target)
        loss.backward()
        self.assertTrue(bool(torch.all(torch.isfinite(prediction.grad))))
        self.assertGreater(float(prediction.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
