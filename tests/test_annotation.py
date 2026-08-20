from __future__ import annotations

import unittest

import numpy as np

from worm_pose_gen.annotation import (
    annotation_pair_metrics,
    annotation_semantic_pair_metrics,
    bootstrap_interval,
    resample_polyline,
    validate_annotation,
)
from worm_pose_gen.annotation_selection import select_recording_frames


def record(sample_id: str, annotator: str, points: list[list[float]], **updates: object) -> dict:
    value = {
        "schema_version": "1.0.0",
        "sample_id": sample_id,
        "annotation_id": f"{sample_id}-{annotator}",
        "annotator_id": annotator,
        "tool_name": "test",
        "tool_version": "1",
        "started_at_utc": "2026-08-19T00:00:00Z",
        "completed_at_utc": "2026-08-19T00:01:00Z",
        "configured_source_path": "nir_videos/test.h5",
        "resolved_source_path": "/data/test.h5",
        "source_size_bytes": 123,
        "source_mtime_ns": 456,
        "source_dataset_path": "/img_nir",
        "frame_index": 7,
        "timestamp_raw": 100.0,
        "timestamp_mapping": "one_timestamp_per_img_nir_frame",
        "split_role": "development_tier_a",
        "selection_stratum": "test",
        "annotation_view": "temporal_window",
        "temporal_window_indices": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "annotation_overlays": [],
        "parent_annotation_id": None,
        "head_tail_state": "ambiguous",
        "trace_state": "complete",
        "worm_width_px": 10.0,
        "difficulty": ["ordinary"],
        "vertices": [{"xy": point, "support_state": "supported"} for point in points],
    }
    value.update(updates)
    return value


class AnnotationTests(unittest.TestCase):
    def test_validation_and_endpoint_preserving_resampling(self) -> None:
        annotation = validate_annotation(
            record("sample", "a", [[1, 2], [5, 3], [9, 7]]),
            image_height=10,
            image_width=10,
        )
        result = resample_polyline(annotation.points_xy, 100)
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(result[-1], [9, 7])

    def test_reverse_orientation_is_scored_symmetrically(self) -> None:
        points = [[1, 2], [4, 4], [8, 7]]
        first = validate_annotation(record("sample", "a", points), image_height=10, image_width=10)
        second = validate_annotation(record("sample", "b", points[::-1]), image_height=10, image_width=10)
        metrics = annotation_pair_metrics(first, second)
        self.assertTrue(metrics["second_trace_reversed"])
        self.assertAlmostEqual(metrics["median_point_distance_px"], 0.0)
        self.assertAlmostEqual(metrics["mean_tangent_angle_error_deg"], 0.0)
        self.assertAlmostEqual(metrics["support_state_agreement_fraction"], 1.0)
        semantic = annotation_semantic_pair_metrics(first, second)
        self.assertTrue(semantic["head_tail_state_agreement"])
        self.assertTrue(semantic["truncation_end_agreement"])

    def test_known_anatomical_orientation_controls_alignment(self) -> None:
        points = [[1, 2], [4, 4], [8, 7]]
        first = validate_annotation(
            record("sample", "a", points, head_tail_state="start_is_head"),
            image_height=10, image_width=10,
        )
        second = validate_annotation(
            record("sample", "b", points[::-1], head_tail_state="start_is_tail"),
            image_height=10, image_width=10,
        )
        metrics = annotation_pair_metrics(first, second)
        self.assertFalse(metrics["first_trace_reversed"])
        self.assertTrue(metrics["second_trace_reversed"])
        self.assertAlmostEqual(metrics["median_point_distance_px"], 0.0)

    def test_truncated_trace_requires_coordinate_free_terminal_marker(self) -> None:
        value = record("sample", "a", [[1, 2], [4, 4]])
        value.update(
            trace_state="truncated",
            vertices=[
                {"xy": [None, None], "support_state": "outside_fov"},
                {"xy": [1, 2], "support_state": "supported"},
                {"xy": [4, 4], "support_state": "supported"},
            ],
        )
        annotation = validate_annotation(value, image_height=10, image_width=10)
        self.assertFalse(annotation.is_complete)
        value["vertices"][0]["xy"] = [-1, 2]
        with self.assertRaisesRegex(ValueError, "must not invent"):
            validate_annotation(value, image_height=10, image_width=10)

    def test_not_identifiable_annotation_needs_no_fake_points(self) -> None:
        value = record("sample", "a", [])
        value.update(trace_state="not_identifiable", vertices=[])
        annotation = validate_annotation(value, image_height=10, image_width=10)
        self.assertEqual(annotation.points_xy.shape, (0, 2))
        self.assertFalse(annotation.is_complete)

        value["vertices"] = [
            {"xy": [1, 2], "support_state": "not_identifiable"},
            {"xy": [2, 3], "support_state": "not_identifiable"},
        ]
        with self.assertRaisesRegex(ValueError, "must not invent"):
            validate_annotation(value, image_height=10, image_width=10)

    def test_delayed_same_annotator_repeat_is_explicitly_supported(self) -> None:
        points = [[1, 2], [4, 4], [8, 7]]
        first = validate_annotation(record("sample", "a", points), image_height=10, image_width=10)
        second = validate_annotation(record("sample", "a", points), image_height=10, image_width=10)
        with self.assertRaisesRegex(ValueError, "different annotators"):
            annotation_pair_metrics(first, second)
        metrics = annotation_pair_metrics(first, second, allow_same_annotator=True)
        self.assertAlmostEqual(metrics["median_point_distance_px"], 0.0)

    def test_bootstrap_is_deterministic(self) -> None:
        values = np.asarray([1.0, 2.0, 4.0, 8.0])
        self.assertEqual(bootstrap_interval(values, resamples=100), bootstrap_interval(values, resamples=100))

    def test_selection_has_exact_count_and_two_complete_double_windows(self) -> None:
        proxy = [
            {"frame_index": index, "accepted": index % 3 == 0, "rejection_reasons": ("screen",)}
            for index in np.linspace(5, 994, 48, dtype=int)
        ]
        selected = select_recording_frames(
            recording="r1", frame_count=1000, target_count=85, proxy_rows=proxy, seed=7
        )
        self.assertEqual(len(selected), 85)
        self.assertEqual(len({value.frame_index for value in selected}), 85)
        windows = {}
        for value in selected:
            if value.temporal_window_id:
                windows.setdefault(value.temporal_window_id, []).append(value.frame_index)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(len(indices) == 11 for indices in windows.values()))
        self.assertEqual(sum(value.double_annotate for value in selected), 22)


if __name__ == "__main__":
    unittest.main()
