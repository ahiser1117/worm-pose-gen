import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from worm_pose_gen.corpus import CorpusStore
from worm_pose_gen.segmentation_dataset import SegmentationStore


class CorpusTests(unittest.TestCase):
    def test_identity_revisions_pledges_snapshot_and_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((64, 64), 120, np.uint8)
            mask = np.zeros_like(image)
            mask[10:20, 20:40] = 1
            mask[10, 20:40] = 255
            legacy = SegmentationStore(root / "corpus")
            old = legacy.save("same", 1, image, mask, source_path=str(root / "a" / "same.h5"), label_source="manual", split="val")
            store = CorpusStore(legacy.root)
            # Deleting and relabeling a legacy sample must preserve its ID and held-out pledge.
            store.delete(old.sample_id)
            restored = store.save_frame(old.source_path, "/img_nir", 1, image, mask)
            self.assertEqual((restored.sample_id, restored.split, restored.revision), (old.sample_id, "val", 2))
            other = store.save_frame(root / "b" / "same.h5", "/img_nir", 1, image, mask, split="train")
            dataset = store.save_frame(old.source_path, "/other", 1, image, mask)
            self.assertEqual(len({old.sample_id, other.sample_id, dataset.sample_id}), 3)
            manifest = store.snapshot(root / "snapshot")
            copied = SegmentationStore(root / "snapshot")
            before = copied.load(other.sample_id)[1].copy()
            store.update_label(other.sample_id, np.ones_like(mask), other.revision)
            store.delete(old.sample_id)
            self.assertTrue(np.array_equal(copied.load(other.sample_id)[1], before))
            self.assertEqual(copied.get(old.sample_id).split, "val")
            self.assertEqual(int(before[10, 20]), 255)
            for sample in manifest["samples"]:
                self.assertEqual(hashlib.sha256(copied.sample_path(sample["sample_id"]).read_bytes()).hexdigest(), sample["sha256"])
