from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from worm_pose_gen.annotation_tool import (
    AnnotationProtocol,
    AnnotationSession,
    build_single_annotator_worklist,
)


def manifest_record(recording: str, index: int, stratum: str) -> dict:
    return {
        "sample_id": f"{recording}-f{index:06d}",
        "recording": recording,
        "frame_index": index,
        "selection_stratum": stratum,
        "difficulty_hints": [],
        "double_annotate": False,
        "temporal_window_id": None,
        "temporal_window_indices": list(range(max(0, index - 2), index + 3)),
        "split_role": "development_tier_a",
        "configured_source_path": f"nir_videos/{recording}.h5",
        "resolved_source_path": f"/data/{recording}.h5",
        "source_size_bytes": 123,
        "source_mtime_ns": 456,
        "source_dataset_path": "/img_nir",
        "image_height": 20,
        "image_width": 30,
        "timestamp_raw": float(index),
        "timestamp_mapping": "one_timestamp_per_img_nir_frame",
        "annotation_overlays": [],
        "primary_annotation_view": "raw_temporal_context_without_pose_overlay",
    }


def small_manifest() -> dict:
    strata = (
        "proxy_difficult", "proxy_easy", "uniform_coverage",
        "double_annotation_temporal_window",
    )
    records = [
        manifest_record(recording, index + 2, strata[index % len(strata)])
        for recording in ("r1", "r2", "r3")
        for index in range(8)
    ]
    return {
        "schema_version": 1,
        "protected_holdout_opened": False,
        "records_sha256": "test-records",
        "records": records,
    }


class AnnotationToolTests(unittest.TestCase):
    def test_balanced_worklist_is_deterministic(self) -> None:
        records = small_manifest()["records"]
        first = build_single_annotator_worklist(records, primary_count=6, repeat_count=3, seed=4)
        second = build_single_annotator_worklist(records, primary_count=6, repeat_count=3, seed=4)
        self.assertEqual(first, second)
        primary = [row for row in first if row["annotation_pass"] == "primary"]
        repeat = [row for row in first if row["annotation_pass"] == "repeat"]
        self.assertEqual(len(primary), 6)
        self.assertEqual(len(repeat), 3)
        self.assertEqual({row["sample_id"].split("-")[0] for row in primary}, {"r1", "r2", "r3"})
        self.assertTrue({row["sample_id"] for row in repeat} <= {row["sample_id"] for row in primary})
        one_repeat = build_single_annotator_worklist(
            records, primary_count=6, repeat_count=1, seed=4
        )
        self.assertEqual(
            sum(row["annotation_pass"] == "repeat" for row in one_repeat), 1
        )

    def test_session_locks_primary_and_delays_blind_repeat(self) -> None:
        now = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            output_path = root / "annotations.json"
            manifest_path.write_text(json.dumps(small_manifest()))
            protocol = AnnotationProtocol(
                primary_count=3, repeat_count=3, repeat_delay_hours=24, selection_seed=7
            )
            session = AnnotationSession(manifest_path, output_path, "solo", protocol, now=now)
            primary = next(task for task in session.worklist if task["annotation_pass"] == "primary")
            request = {
                **primary,
                "started_at_utc": (now - timedelta(minutes=2)).isoformat(),
                "trace_state": "complete",
                "head_tail_state": "ambiguous",
                "outside_fov_at_start": False,
                "outside_fov_at_end": False,
                "worm_width_px": "4.5",
                "difficulty": [],
                "vertices": [
                    {"xy": [1, 2], "support_state": "supported"},
                    {"xy": [8, 9], "support_state": "supported"},
                ],
            }
            saved = session.save_annotation(request, now=now)
            self.assertEqual(saved["annotation_pass"], "primary")
            self.assertEqual(saved["annotator_id"], "solo")
            self.assertEqual(saved["worm_width_px"], 4.5)
            with self.assertRaisesRegex(ValueError, "already locked"):
                session.save_annotation(request, now=now)

            repeat = next(
                task for task in session.worklist
                if task["annotation_pass"] == "repeat" and task["sample_id"] == primary["sample_id"]
            )
            repeat_request = {**request, **repeat}
            with self.assertRaisesRegex(ValueError, "not available"):
                session.save_annotation(repeat_request, now=now + timedelta(hours=23))
            repeat_request["started_at_utc"] = (now + timedelta(hours=25)).isoformat()
            repeated = session.save_annotation(repeat_request, now=now + timedelta(hours=25, minutes=1))
            self.assertEqual(repeated["repeat_of_annotation_id"], saved["annotation_id"])
            self.assertEqual(repeated["annotator_id"], saved["annotator_id"])

            resumed = AnnotationSession(manifest_path, output_path, "solo", protocol)
            self.assertEqual(len(resumed.payload["annotations"]), 2)

    def test_not_identifiable_saves_without_vertices(self) -> None:
        now = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            output_path = root / "annotations.json"
            manifest_path.write_text(json.dumps(small_manifest()))
            protocol = AnnotationProtocol(primary_count=3, repeat_count=0)
            session = AnnotationSession(manifest_path, output_path, "solo", protocol, now=now)
            task = session.worklist[0]
            saved = session.save_annotation({
                **task,
                "started_at_utc": (now - timedelta(minutes=1)).isoformat(),
                "trace_state": "not_identifiable",
                "vertices": [{"xy": [1, 2], "support_state": "supported"}],
            }, now=now)
            self.assertEqual(saved["vertices"], [])


if __name__ == "__main__":
    unittest.main()
