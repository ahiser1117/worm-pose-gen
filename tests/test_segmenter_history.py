from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from worm_pose_gen.run_records import checkpoint_fingerprint, split_manifest, timestamp_slug
from worm_pose_gen.segmentation_dataset import SegmentationStore


def _load_plot_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plot_segmenter_history.py"
    spec = importlib.util.spec_from_file_location("plot_segmenter_history", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluation(evaluated_at: str, sample_ids: list[str], iou: float) -> dict:
    per_sample = [
        {
            "sample_id": sid, "recording": "rec", "frame_index": i, "revision": 1, "saved_at": evaluated_at,
            "label_source": "network+manual" if i % 2 else "bootstrap_classical",
            "network": {k: iou for k in ("iou", "dice", "precision", "recall")},
            "classical": {k: iou - 0.1 for k in ("iou", "dice", "precision", "recall")},
        }
        for i, sid in enumerate(sample_ids)
    ]
    stat = {"n": len(per_sample), "median": iou, "mean": iou, "min": iou}
    stat_c = {"n": len(per_sample), "median": iou - 0.1, "mean": iou - 0.1, "min": iou - 0.1}
    summary = {
        "samples": len(per_sample),
        "network": {k: stat for k in ("iou", "dice", "precision", "recall")},
        "classical": {k: stat_c for k in ("iou", "dice", "precision", "recall")},
        "hand_refined_samples": len(per_sample) // 2,
        "hand_refined_network_iou": stat, "hand_refined_classical_iou": stat_c,
        "network_beats_classical": len(per_sample), "hand_refined_network_beats_classical": len(per_sample) // 2,
    }
    return {
        "evaluated_at": evaluated_at, "note": "", "git": {"commit": "abc", "dirty": False},
        "checkpoint": {"sha256": "0" * 64, "modified_at": evaluated_at},
        "dataset_root": "/x", "store_counts": {"train": 8, "val": 1, "test": 1},
        "split_membership": {"val": [], "test": []},
        "splits": {s: {"summary": summary, "per_sample": per_sample} for s in ("val", "test")},
    }


class SegmenterHistoryTests(unittest.TestCase):
    def test_timestamp_slug_and_fingerprint(self) -> None:
        self.assertEqual(timestamp_slug("2026-09-03T18:43:04+00:00"), "2026-09-03T18-43-04Z")
        self.assertEqual(timestamp_slug("2026-09-03T14:43:04-04:00"), "2026-09-03T18-43-04Z")
        self.assertIsNone(checkpoint_fingerprint(None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            path.write_bytes(b"weights")
            info = checkpoint_fingerprint(path)
            self.assertTrue(info["exists"])
            self.assertEqual(info["size_bytes"], 7)
            self.assertEqual(len(info["sha256"]), 64)
            self.assertFalse(checkpoint_fingerprint(Path(directory) / "missing.ckpt")["exists"])

    def test_split_manifest_lists_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SegmentationStore(directory)
            image = np.zeros((8, 8), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            for index in range(12):
                store.save("rec", index, image, mask, source_path="/x", label_source="bootstrap")
            manifest = split_manifest(store, ("val", "test"))
            self.assertEqual(set(manifest), {"val", "test"})
            self.assertEqual(len(manifest["val"]) + len(manifest["test"]), 2)
            self.assertEqual(set(manifest["val"][0]), {"sample_id", "label_source", "revision", "saved_at"})

    def test_plots_render_from_records_and_from_nothing(self) -> None:
        plots = _load_plot_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            store = SegmentationStore(root / "dataset")
            # Empty everything still produces the four figures.
            outputs = plots.make_plots(checkpoints, store.root, checkpoints / "plots")
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(p.exists() and p.stat().st_size > 0 for p in outputs))
            # Now with a training log, run record, two evaluations, and a store.
            image = np.zeros((8, 8), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            for index in range(10):
                store.save("rec", index, image, mask, source_path="/x", label_source="network+manual" if index % 2 else "bootstrap")
            log_dir = checkpoints / "logs" / "version_0"
            log_dir.mkdir(parents=True)
            (log_dir / "metrics.csv").write_text(
                "epoch,step,train_loss_step,train_loss_epoch,val_loss,val_iou\n"
                "0,4,0.9,,,\n0,9,,0.8,0.7,0.5\n1,14,0.6,,,\n1,19,,0.5,0.4,0.8\n"
            )
            (checkpoints / "runs").mkdir()
            (checkpoints / "runs" / "2026-09-03T18-00-00Z.json").write_text(json.dumps({
                "started_at": "2026-09-03T18:00:00+00:00", "log_dir": str(log_dir),
                "counts": {"train": 8, "val": 1, "test": 1}, "init_checkpoint": None,
            }))
            for stamp, iou in (("2026-09-03T18:10:00+00:00", 0.8), ("2026-09-03T19:10:00+00:00", 0.9)):
                folder = checkpoints / "evaluations" / timestamp_slug(stamp)
                folder.mkdir(parents=True)
                (folder / "evaluation.json").write_text(json.dumps(_evaluation(stamp, [f"rec_f{i:06d}" for i in range(6)], iou)))
            runs = plots.load_runs(checkpoints)
            self.assertEqual(len(runs), 1)
            self.assertIn("train 8", runs[0]["label"])
            self.assertEqual(runs[0]["metrics"]["val_iou"], [(0.0, 0.5), (1.0, 0.8)])
            evaluations = plots.load_evaluations(checkpoints)
            self.assertEqual([e["evaluated_at"] for e in evaluations][-1], "2026-09-03T19:10:00+00:00")
            outputs = plots.make_plots(checkpoints, store.root, checkpoints / "plots")
            self.assertTrue(all(p.exists() and p.stat().st_size > 0 for p in outputs))


if __name__ == "__main__":
    unittest.main()
