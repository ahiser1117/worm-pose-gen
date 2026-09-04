from __future__ import annotations

import tempfile
import unittest

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from worm_pose_gen.segmenter import (
    IGNORE_LABEL,
    ResNet18UNet,
    SegmentationModule,
    load_segmenter,
    masked_binary_metrics,
    normalize_frame,
)


class SegmenterTests(unittest.TestCase):
    def _check_comparison(self, checkpoint: str, directory: str) -> None:
        from worm_pose_gen.segmentation_dataset import SegmentationStore
        from worm_pose_gen.segmenter_eval import compare_on_split, score_split, summarize_iou

        store = SegmentationStore(f"{directory}/store")
        image = np.zeros((40, 52), dtype=np.uint8)
        mask = np.zeros((40, 52), dtype=np.uint8)
        mask[10:20, 10:30] = 1
        for index in range(12):
            store.save("rec", index, image, mask, source_path="/x", label_source="network+manual")
        rows = score_split(load_segmenter(checkpoint, device="cpu"), store, "val")
        self.assertEqual(len(rows), store.counts()["val"])
        self.assertEqual(set(rows[0]), {"sample_id", "label_source", "revision", "label_pixels", "iou", "loss"})
        self.assertGreater(rows[0]["loss"], 0.0)
        self.assertEqual(summarize_iou([])["n"], 0)
        # No incumbent: the candidate is promoted by default.
        first = compare_on_split(checkpoint, f"{directory}/missing.ckpt", store, "val", device="cpu")
        self.assertTrue(first["promote"])
        self.assertIsNone(first["incumbent"])
        # The same weights as incumbent: not strictly better, so not promoted.
        same = compare_on_split(checkpoint, checkpoint, store, "val", device="cpu")
        self.assertFalse(same["promote"])
        self.assertEqual(same["candidate_score"]["loss"], same["incumbent_score"]["loss"])
        self.assertEqual(len(same["per_sample"]), store.counts()["val"])
        self.assertIn(">=", same["reason"])

    def test_normalize_frame_shape_and_scale(self) -> None:
        frame = np.full((10, 12), 255, dtype=np.uint8)
        tensor = normalize_frame(frame)
        self.assertEqual(tuple(tensor.shape), (1, 10, 12))
        self.assertAlmostEqual(float(tensor.max()), (1.0 - 0.449) / 0.226, places=5)

    def test_network_returns_full_resolution_logits_for_odd_sizes(self) -> None:
        network = ResNet18UNet(pretrained=False).eval()
        with torch.no_grad():
            logits = network(torch.randn(1, 1, 45, 70))
        self.assertEqual(tuple(logits.shape), (1, 1, 45, 70))

    def test_masked_metrics_ignore_invalid_pixels(self) -> None:
        probability = torch.zeros(1, 4, 4)
        probability[0, :2] = 1.0
        target = torch.zeros(1, 4, 4)
        target[0, :2] = 1.0
        valid = torch.ones(1, 4, 4)
        valid[0, 0] = 0.0  # a wrong row would be hidden by ignore
        probability[0, 0] = 0.0
        metrics = masked_binary_metrics(probability, target, valid)
        self.assertAlmostEqual(float(metrics["iou"][0]), 1.0, places=4)

    def test_empty_label_scoring(self) -> None:
        target = torch.zeros(2, 4, 4)
        valid = torch.ones(2, 4, 4)
        probability = torch.zeros(2, 4, 4)
        probability[1, 0, 0] = 1.0  # one hallucinated pixel on an empty frame
        metrics = masked_binary_metrics(probability, target, valid)
        for name in ("iou", "dice", "precision", "recall"):
            self.assertAlmostEqual(float(metrics[name][0]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["iou"][1]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["precision"][1]), 0.0, places=6)

    def test_loss_ignores_masked_pixels(self) -> None:
        module = SegmentationModule(pretrained=False)
        logits = torch.full((1, 1, 6, 6), 8.0)
        target = torch.zeros(1, 6, 6)
        target[0, :3] = 1.0
        valid = torch.zeros(1, 6, 6)
        valid[0, :3] = 1.0  # only the correctly predicted half is scored
        total, parts = module.loss(logits, target, valid)
        self.assertLess(float(parts["bce"]), 1e-2)
        self.assertLess(float(total), 0.05)

    def test_fit_one_step_and_reload_checkpoint(self) -> None:
        module = SegmentationModule(pretrained=False, learning_rate=1e-3)
        image = torch.randn(2, 1, 64, 64)
        mask = (torch.rand(2, 64, 64) > 0.8).float()
        batch = {"image": image, "mask": mask, "valid": torch.ones(2, 64, 64)}
        loader = DataLoader([batch], batch_size=None)
        with tempfile.TemporaryDirectory() as directory:
            trainer = L.Trainer(
                max_epochs=1, accelerator="cpu", devices=1, logger=False,
                enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            )
            trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
            path = f"{directory}/model.ckpt"
            trainer.save_checkpoint(path)
            reloaded = load_segmenter(path, device="cpu")
            self._check_comparison(path, directory)
            probability = reloaded.predict_probability(np.zeros((40, 52), dtype=np.uint8))
            self.assertEqual(probability.shape, (40, 52))
            frames = np.random.default_rng(0).integers(0, 255, (5, 40, 52), dtype=np.uint8)
            batched = reloaded.predict_probability_batch(frames, batch_size=2)
            self.assertEqual(batched.shape, (5, 40, 52))
            single = reloaded.predict_probability(frames[3])
            self.assertTrue(np.allclose(batched[3], single, atol=1e-4))
            self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
            self.assertFalse(reloaded.training)
            self.assertEqual(IGNORE_LABEL, 255)


if __name__ == "__main__":
    unittest.main()
