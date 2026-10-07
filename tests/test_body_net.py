import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from worm_pose_gen.body_net import heatmap_focal_loss
from worm_pose_gen.body_targets import point_heatmap, render_body_targets, self_contact
from worm_pose_gen.recordings import RecordingSource
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


class HeatmapLossTests(unittest.TestCase):
    def test_focal_loss_prefers_peak_at_target(self):
        target = torch.as_tensor(point_heatmap((16, 16), np.array([8.0, 8.0]), 1.5))[None]
        good = torch.full((1, 16, 16), -6.0)
        good[0, 8, 8] = 6.0
        bad = torch.full((1, 16, 16), -6.0)
        bad[0, 2, 2] = 6.0
        self.assertLess(float(heatmap_focal_loss(good, target)), float(heatmap_focal_loss(bad, target)))


if __name__ == "__main__":
    unittest.main()
