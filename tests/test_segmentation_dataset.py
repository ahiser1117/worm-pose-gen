from __future__ import annotations

from collections import Counter
import tempfile
import unittest

import numpy as np
import torch

from worm_pose_gen.segmentation_dataset import (
    SPLITS,
    SegmentationDataModule,
    SegmentationDataset,
    SegmentationStore,
    assign_split,
    make_sample_id,
)


def _sample(seed: int, height: int = 48, width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:20, 8:56] = 1
    mask[9, 8:56] = 255
    return image, mask


class SegmentationDatasetTests(unittest.TestCase):
    def test_split_assignment_tracks_80_10_10(self) -> None:
        counts = {name: 0 for name in SPLITS}
        for _ in range(50):
            counts[assign_split(counts)] += 1
        self.assertEqual(counts, {"train": 40, "val": 5, "test": 5})
        self.assertEqual(assign_split({}), "train")

    def test_store_round_trip_keeps_split_and_increments_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SegmentationStore(directory)
            image, mask = _sample(0)
            first = store.save("rec", 7, image, mask, source_path="/x.h5", label_source="bootstrap")
            self.assertEqual(first.revision, 1)
            self.assertTrue(store.has("rec", 7))
            edited = mask.copy()
            edited[30:35, :] = 1
            second = store.save("rec", 7, image, edited, source_path="/x.h5", label_source="manual")
            self.assertEqual(second.split, first.split)
            self.assertEqual(second.revision, 2)
            loaded_image, loaded_mask, record = store.load(make_sample_id("rec", 7))
            self.assertTrue(np.array_equal(loaded_image, image))
            self.assertTrue(np.array_equal(loaded_mask, edited))
            self.assertEqual(record.label_source, "manual")
            self.assertAlmostEqual(record.ignore_fraction, 48 / (48 * 64), places=6)
            self.assertEqual(sum(store.counts().values()), 1)
            self.assertTrue(store.delete(record.sample_id))
            self.assertEqual(sum(store.counts().values()), 0)
            self.assertFalse(store.sample_path(record.sample_id).exists())

    def test_store_rejects_bad_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SegmentationStore(directory)
            image, mask = _sample(1)
            with self.assertRaisesRegex(ValueError, "0, 1, or 255"):
                store.save("rec", 0, image, mask.astype(np.int32) + 3, source_path="/x", label_source="t")
            with self.assertRaisesRegex(ValueError, "shapes must match"):
                store.save("rec", 0, image, mask[:10], source_path="/x", label_source="t")

    def test_datasets_and_datamodule_produce_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SegmentationStore(directory)
            for index in range(12):
                image, mask = _sample(index)
                store.save("rec", index, image, mask, source_path="/x.h5", label_source="bootstrap")
            counts = store.counts()
            self.assertEqual(counts, {"train": 10, "val": 1, "test": 1})
            train = SegmentationDataset(store, "train", augment=True, crop_size=32, seed=1)
            item = train[0]
            self.assertEqual(tuple(item["image"].shape), (1, 32, 32))
            self.assertEqual(tuple(item["mask"].shape), (32, 32))
            self.assertTrue(torch.all((item["valid"] == 0) | (item["valid"] == 1)))
            module = SegmentationDataModule(directory, batch_size=4, crop_size=32, num_workers=0)
            module.setup()
            batch = next(iter(module.train_dataloader()))
            self.assertEqual(tuple(batch["image"].shape), (4, 1, 32, 32))
            val_batch = next(iter(module.val_dataloader()))
            self.assertEqual(tuple(val_batch["image"].shape[-2:]), (48, 64))
            self.assertIsInstance(batch["sample_id"], list)
            splits = Counter(record.split for record in store.records())
            self.assertEqual(splits["train"], 10)


if __name__ == "__main__":
    unittest.main()
