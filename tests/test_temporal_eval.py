import tempfile
import unittest
from pathlib import Path

import numpy as np

from worm_pose_gen import temporal_eval
from worm_pose_gen.temporal_eval import Clip, ClipPrediction, clip_metrics, load_prediction, save_prediction

SHAPE = (60, 200)
FRAMES = 12


def moving_bar(frames: int = FRAMES) -> ClipPrediction:
    """A horizontal body moving right one pixel per frame, head at its left end, A-P increasing to the right."""

    mask = np.zeros((frames, *SHAPE), bool)
    ap = np.zeros((frames, *SHAPE), np.uint8)
    head = np.zeros((frames, 3))
    tail = np.zeros((frames, 3))
    for t in range(frames):
        x0, x1 = 20 + t, 160 + t
        mask[t, 25:35, x0:x1] = True
        values = np.linspace(0, 1, x1 - x0)
        ap[t, 25:35, x0:x1] = np.rint(254 * values) + 1
        head[t] = (x0, 30, 0.9)
        tail[t] = (x1 - 1, 30, 0.9)
    return ClipPrediction(mask=mask, ap=ap, overlap=np.zeros_like(mask), head=head, tail=tail)


class TemporalMetricsTests(unittest.TestCase):
    def test_smooth_motion_has_no_events(self):
        m = clip_metrics(moving_bar())
        self.assertEqual(m["mask_flicker"], 0.0)
        self.assertEqual(m["ap_flip_frames"], 0)
        self.assertEqual(m["head_jumps"], 0)
        self.assertEqual(m["head_swaps"], 0)
        self.assertLess(m["ap_jitter"], 0.01)

    def test_one_flipped_frame_counts_both_transitions_and_a_head_swap(self):
        p = moving_bar()
        t = 6
        on = p.mask[t]
        p.ap[t][on] = (256 - p.ap[t][on].astype(int)).astype(np.uint8)
        p.head[t], p.tail[t] = p.tail[t].copy(), p.head[t].copy()
        m = clip_metrics(p)
        self.assertEqual(m["ap_flip_frames"], 2)
        self.assertEqual(m["head_jumps"], 2)
        self.assertGreaterEqual(m["head_swaps"], 1)

    def test_flicker_counts_a_one_frame_blob(self):
        p = moving_bar()
        p.mask[5, 0:10, 0:10] = True
        m = clip_metrics(p)
        self.assertAlmostEqual(m["mask_flicker"] * sum(p.mask[t].sum() for t in range(1, FRAMES - 1)), 100.0)

    def test_dropout(self):
        p = moving_bar()
        p.head[4, 2] = 0.0
        self.assertEqual(clip_metrics(p)["head_dropouts"], 1)

    def test_prediction_round_trip(self):
        p = moving_bar()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.npz"
            save_prediction(path, p, Clip("c", "sequence", "/x/r.h5", 0, FRAMES))
            q = load_prediction(path)
        np.testing.assert_array_equal(q.mask, p.mask)
        np.testing.assert_array_equal(q.ap, p.ap)
        np.testing.assert_array_equal(q.head, p.head)

    def test_exclusion(self):
        clip = temporal_eval.clips(names=["spiral_0131"])[0]
        self.assertTrue(temporal_eval.excluded(clip.recording, clip.start + 10))
        self.assertFalse(temporal_eval.excluded(clip.recording, clip.start + clip.frames + 1000))
        self.assertTrue(temporal_eval.excluded("/a/b/2024-06-18-12.h5", 5))


if __name__ == "__main__":
    unittest.main()
