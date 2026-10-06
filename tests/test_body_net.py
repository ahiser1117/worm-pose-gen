import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from worm_pose_gen.body_net import (
    BodyFieldDataset,
    BodyFieldModule,
    collate_body_fields,
    heatmap_focal_loss,
)
from worm_pose_gen.body_targets import point_heatmap, render_body_targets, self_contact
from worm_pose_gen.label_app import RecordingSource
from worm_pose_gen.segmentation_dataset import SegmentationStore
from worm_pose_gen.segmenter import INPUT_STD
from worm_pose_gen.temporal_context import difference_channels, read_context


def tube_mask(shape, centerline, diameter):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    points = np.stack((xx.ravel(), yy.ravel()), 1).astype(float)
    start, end = centerline[:-1], centerline[1:]
    segment = end - start
    t = np.clip(((points[:, None] - start) * segment).sum(-1) / (segment**2).sum(-1), 0, 1)
    distance = np.linalg.norm(points[:, None] - (start + t[..., None] * segment), axis=-1)
    return (distance.min(1) <= diameter / 2).reshape(shape)


class BodyTargetTests(unittest.TestCase):
    def test_straight_body_ap_runs_head_to_tail(self):
        centerline = np.stack((np.linspace(10, 90, 50), np.full(50, 20.0)), 1)
        mask = tube_mask((40, 100), centerline, 8.0)
        targets = render_body_targets(mask, centerline, np.full(50, 8.0))
        self.assertAlmostEqual(float(targets.ap[20, 10]), 0.0, places=2)
        self.assertAlmostEqual(float(targets.ap[20, 90]), 1.0, places=2)
        self.assertAlmostEqual(float(targets.ap[20, 50]), 0.5, places=2)
        self.assertTrue(np.isnan(targets.ap[0, 0]))
        self.assertFalse(targets.overlap.any())
        np.testing.assert_allclose(targets.head_xy, [10, 20])

    def test_crossing_is_overlap_with_undefined_ap(self):
        # Right along y=50, down, back left, then up through the first run at (60, 50).
        corners = np.array([[10, 50], [100, 50], [100, 90], [60, 90], [60, 10]], float)
        pieces = [np.linspace(a, b, 40, endpoint=False) for a, b in zip(corners[:-1], corners[1:])]
        centerline = np.concatenate(pieces + [corners[-1:]])
        mask = tube_mask((100, 120), centerline, 6.0)
        targets = render_body_targets(mask, centerline, np.full(len(centerline), 6.0))
        self.assertTrue(targets.overlap[50, 60])
        self.assertFalse(targets.overlap[50, 30])
        self.assertTrue(np.isnan(targets.ap[targets.overlap]).all())
        self.assertTrue(np.isfinite(targets.ap[mask & ~targets.overlap]).all())
        self.assertTrue(self_contact(centerline, np.full(len(centerline), 6.0)))
        straight = np.stack((np.linspace(10, 90, 50), np.full(50, 20.0)), 1)
        self.assertFalse(self_contact(straight, np.full(50, 6.0)))

    def test_heatmap_peaks_at_point_and_is_empty_for_nan(self):
        heat = point_heatmap((20, 30), np.array([7.0, 4.0]), 2.0)
        self.assertEqual(np.unravel_index(heat.argmax(), heat.shape), (4, 7))
        self.assertAlmostEqual(float(heat.max()), 1.0)
        self.assertFalse(point_heatmap((20, 30), np.array([np.nan, 1.0]), 2.0).any())


def write_recording(path, frames):
    with h5py.File(path, "w") as handle:
        handle["img_nir"] = frames


class TemporalContextTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_edges_repeat_nearest_frame_and_are_invalid(self):
        frames = np.stack([np.full((12, 16), 10 * k, np.uint8) for k in range(6)])
        write_recording(self.root / "rec.h5", frames)
        cache = self.root / "fields"
        cache.mkdir()
        # A unit flat field keeps intensities, so the arithmetic is checkable.
        np.savez(cache / "rec.npz", illumination=np.ones((12, 16)), dark_level=0.0, reference_level=1.0, gain=np.ones((12, 16)))
        source = RecordingSource(self.root / "rec.h5", cache)
        context, valid = read_context(source, 1, max_lag=3)
        source.close()
        np.testing.assert_array_equal(valid, [False, False, True, True, True, True, True])
        np.testing.assert_array_equal(context[:, 0, 0], [0, 0, 0, 10, 20, 30, 40])
        channels = difference_channels(context, valid, [1, 3])
        self.assertAlmostEqual(float(channels[0, 0, 0]), 20 / (255 * INPUT_STD), places=5)
        self.assertFalse(channels[1].any())  # lag 3 reaches before the recording


class BodyFieldDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = SegmentationStore(self.root)
        centerline = np.stack((np.linspace(10, 90, 50), np.full(50, 32.0)), 1)
        self.mask = tube_mask((64, 96), centerline, 8.0)
        image = np.where(self.mask, 60, 200).astype(np.uint8)
        for k, split in enumerate(("train", "val", "test")):
            record = self.store.save("rec", k, image, self.mask.astype(np.uint8), source_path="rec.h5", label_source="manual", split=split)
            targets = render_body_targets(self.mask, centerline, np.full(50, 8.0))
            context = np.stack([np.roll(image, shift, axis=1) for shift in range(-2, 3)])
            fields = self.root / "body_fields"
            fields.mkdir(exist_ok=True)
            meta = {"mask_revision": record.revision, "max_lag": 2, "has_body": True, "fit_iou": 0.97}
            np.savez(
                fields / f"{record.sample_id}.npz", meta=json.dumps(meta), context=context,
                context_valid=np.ones(5, bool), ap=targets.ap.astype(np.float16), overlap=targets.overlap,
                head_xy=targets.head_xy, tail_xy=targets.tail_xy, diameter_px=np.float64(8.0),
            )

    def test_item_shapes_and_loss_backpropagate(self):
        dataset = BodyFieldDataset(self.store, "train", (1, 2))
        item = dataset[0]
        self.assertEqual(tuple(item["image"].shape), (3, 64, 96))
        self.assertGreater(float(item["image"][1].abs().max()), 0.0)
        batch = collate_body_fields([item, BodyFieldDataset(self.store, "val", (1, 2))[0]])
        module = BodyFieldModule(lags=(1, 2), pretrained=False)
        logits = module(batch["image"])
        self.assertEqual(tuple(logits.shape), (2, 5, 64, 96))
        total, parts = module.loss(logits, batch["targets"])
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(float(parts["ap_l1"]), 0.0)
        # The heatmap prior keeps the untrained focal loss on the scale of the others.
        self.assertLess(float(parts["head_focal"]), 20.0)
        metrics = module.endpoint_metrics(logits.detach(), batch["targets"])
        self.assertEqual(metrics["head_error_px"].numel(), 2)

    def test_augmented_item_keeps_targets_aligned(self):
        dataset = BodyFieldDataset(self.store, "train", (), augment=True, crop_size=48)
        for _ in range(4):
            item = dataset[0]
            mask = item["targets"][0]
            ap_valid = item["targets"][3]
            self.assertTrue(bool((ap_valid <= mask).all()))
            self.assertEqual(tuple(item["image"].shape), (1, 48, 48))

    def test_poor_fit_trains_mask_only(self):
        path = self.root / "body_fields" / "rec_f000000.npz"
        with np.load(path) as archive:
            arrays = {k: archive[k] for k in archive.files}
        meta = json.loads(str(arrays.pop("meta")))
        meta["fit_iou"] = 0.5
        np.savez(path, meta=json.dumps(meta), **arrays)
        targets = BodyFieldDataset(self.store, "train", ())[0]["targets"]
        self.assertFalse(bool(targets[3].any()))
        self.assertTrue(bool(torch.isnan(targets[4]).all()))

    def test_review_overrides_fit_quality(self):
        from worm_pose_gen import body_fields

        body_fields.set_review(self.root, "rec_f000000", "rejected")
        self.assertFalse(bool(BodyFieldDataset(self.store, "train", ())[0]["targets"][3].any()))
        path = body_fields.field_path(self.root, "rec_f000000")
        arrays, meta = body_fields.load(path)
        meta.update(fit_iou=0.5, review="accepted")
        body_fields.save(path, meta, arrays)
        self.assertTrue(bool(BodyFieldDataset(self.store, "train", ())[0]["targets"][3].any()))

    def test_focal_loss_prefers_peak_at_target(self):
        target = torch.as_tensor(point_heatmap((16, 16), np.array([8.0, 8.0]), 1.5))[None]
        good = torch.full((1, 16, 16), -6.0)
        good[0, 8, 8] = 6.0
        bad = torch.full((1, 16, 16), -6.0)
        bad[0, 2, 2] = 6.0
        self.assertLess(float(heatmap_focal_loss(good, target)), float(heatmap_focal_loss(bad, target)))


if __name__ == "__main__":
    unittest.main()
