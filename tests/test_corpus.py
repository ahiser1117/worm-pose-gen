from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from worm_pose_gen.corpus import CorpusStore
from worm_pose_gen.segmentation_dataset import SegmentationStore, SegmentationDataModule
from worm_pose_gen.training import fine_tune_job, run_training, list_checkpoints, validate_params


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

    def test_mixed_frame_shapes_pad_as_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CorpusStore(directory)
            for n, shape in enumerate([(36, 48), (64, 96)]):
                image = np.full(shape, 120, np.uint8)
                store.save_frame(f"/r{n}.h5", "/img_nir", 0, image, np.ones(shape, np.uint8), split="train")
            data = SegmentationDataModule(directory, batch_size=2, crop_size=128, num_workers=0, pad_batches=True)
            data.setup()
            batch = next(iter(data.train_dataloader()))
            self.assertEqual(tuple(batch["image"].shape), (2, 1, 64, 96))
            self.assertEqual(int(batch["valid"].sum()), 36 * 48 + 64 * 96)

    def test_parameter_validation(self):
        for invalid in ({"epochs": 1.5}, {"learning_rate": float("nan")}, {"device": "bad"}, {"pretrained": True}):
            with self.assertRaises(ValueError):
                validate_params(invalid)

    def test_cpu_fine_tune_from_frozen_synthetic_checkpoint(self):
        import lightning as L
        from worm_pose_gen.segmenter import SegmentationModule
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.ckpt"
            threads = torch.get_num_threads()
            torch.set_num_threads(2)
            try:
                model = SegmentationModule(pretrained=False)
                torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                            "pytorch-lightning_version": L.__version__}, initial)
                del model
                config = SimpleNamespace(corpus_root=root / "corpus", checkpoints_root=root / "checkpoints", checkpoint=initial)
                store = CorpusStore(config.corpus_root)
                image = np.full((64, 64), 120, np.uint8)
                mask = np.zeros_like(image)
                mask[20:40, 28:36] = 1
                mask[19, :] = 255
                train = store.save_frame(root / "r.h5", "/img_nir", 0, image, mask, split="train")
                store.save_frame(root / "r.h5", "/img_nir", 1, image, mask, split="val")
                spec, argv = fine_tune_job(SimpleNamespace(config=config), {"params": {"epochs": 1, "batch_size": 1, "crop_size": 64, "device": "cpu"}})
                run_dir = Path(spec.params["run_dir"])
                store.update_label(train.sample_id, np.zeros_like(mask))
                initial_bytes = hashlib.sha256(initial.read_bytes()).hexdigest()
                result = run_training(run_dir)
                self.assertEqual(result["state"], "completed")
                self.assertTrue((run_dir / "best.ckpt").is_file())
                self.assertTrue((run_dir / "last.ckpt").is_file())
                self.assertEqual(hashlib.sha256(initial.read_bytes()).hexdigest(), initial_bytes)
                self.assertEqual(result["dataset"]["samples"][0]["revision"], 1)
                self.assertFalse(result["promotion"])
                self.assertEqual(len(list_checkpoints(config)), 3)
            finally:
                torch.set_num_threads(threads)
