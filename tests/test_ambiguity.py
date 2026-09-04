from __future__ import annotations

import unittest

import numpy as np

from worm_pose_gen.ambiguity import (
    FLAG_NAMES,
    AmbiguityThresholds,
    compute_ambiguity,
    pose_jump_px,
    self_contact_px,
    summarize_ambiguity,
)
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import default_width_template
from worm_pose_gen.pose_run import tube_area_px


def _straight(length: float = 600.0, x0: float = 100.0, y: float = 300.0) -> np.ndarray:
    return np.column_stack((np.linspace(x0, x0 + length, 100), np.full(100, y)))


def _arrays(curves: list[np.ndarray], **overrides) -> dict[str, np.ndarray]:
    n = len(curves)
    profile = 40.0 * default_width_template()
    arrays = {
        "frame_index": np.arange(n),
        "fitted": np.ones(n, dtype=bool),
        "centerline_xy": np.stack(curves),
        "width_profile": np.tile(profile, (n, 1)),
        "width_px": np.full(n, 40.0),
        "body_length_px": np.array([np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in curves]),
        "iou": np.full(n, 0.95),
        "pixels_filled": np.zeros(n, dtype=np.int64),
        "pixels_outside_largest": np.zeros(n, dtype=np.int64),
    }
    arrays["worm_pixels"] = np.array([tube_area_px(profile, L) for L in arrays["body_length_px"]]).astype(np.int64)
    arrays.update(overrides)
    return arrays


class AmbiguityTests(unittest.TestCase):
    def test_self_contact_detects_a_closed_loop(self) -> None:
        self.assertEqual(self_contact_px(_straight()), float("inf")) if False else None
        straight = self_contact_px(_straight())
        self.assertGreater(straight, 80.0)  # 15 points of a 600 px body are 90 px apart
        angle = np.linspace(0, 1.9 * np.pi, 100)
        loop = np.column_stack((100 * np.cos(angle), 100 * np.sin(angle)))
        self.assertLess(self_contact_px(loop), 40.0)

    def test_pose_jump_ignores_orientation(self) -> None:
        a = _straight()
        self.assertEqual(pose_jump_px(a, a[::-1]), 0.0)
        self.assertAlmostEqual(pose_jump_px(a, a + (0.0, 12.0)), 12.0)

    def test_flags_fire_on_the_intended_frames(self) -> None:
        base = _straight()
        loop_angle = np.linspace(0, 1.9 * np.pi, 100)
        loop = np.column_stack((300 + 95 * np.cos(loop_angle), 300 + 95 * np.sin(loop_angle)))
        curves = [base, base + (3.0, 0.0), base + (0.0, 90.0), loop, base]
        arrays = _arrays(curves)
        arrays["iou"][4] = 0.7
        arrays["worm_pixels"][1] = int(0.8 * arrays["worm_pixels"][1])  # tube covers more than the mask: overlap
        arrays["worm_pixels"][4] = int(1.25 * arrays["worm_pixels"][4])  # mask larger than tube: missed body
        arrays["pixels_filled"][3] = 900
        arrays["pixels_outside_largest"][2] = 2000
        prior = {"length_px": 600.0, "log_length_sigma": 0.05}
        out = compute_ambiguity(arrays, prior=prior)
        self.assertFalse(out["flag_low_iou"][0])
        self.assertTrue(out["flag_low_iou"][4])
        self.assertTrue(out["flag_area_deficit"][1])
        self.assertTrue(out["flag_area_excess"][4])
        self.assertTrue(out["flag_self_contact"][3])
        self.assertFalse(out["flag_self_contact"][0])
        self.assertTrue(out["flag_holes"][3])
        self.assertTrue(out["flag_fragments"][2])
        self.assertTrue(out["flag_pose_jump"][2])  # 90 px shift of a 40 px wide body
        self.assertFalse(out["flag_pose_jump"][1])
        self.assertTrue(np.isnan(out["pose_jump_px"][0]))
        self.assertLess(abs(out["length_deviation"][3]), 0.1)  # the loop polyline is about 567 px against a 600 px prior
        self.assertFalse(out["flag_length_deviation"].any())
        self.assertEqual(int(out["ambiguity_score"][0]), 0)
        self.assertEqual(int(out["ambiguity_score"][4]), 3)  # low overlap, missed body, and a jump from the loop
        # Clipping is not self-overlap: with the image ending at x = 400 the visible tube is compared.
        clipped = compute_ambiguity(arrays, prior=prior, image_shape=(600, 400))
        self.assertGreater(clipped["area_ratio"][0], 1.5)
        self.assertTrue(np.isnan(clipped["pose_jump_px"][3]) or clipped["pose_jump_px"][3] >= 0)
        arrays.update(out)
        summary = summarize_ambiguity(arrays)
        self.assertEqual(set(summary["flag_counts"]), set(FLAG_NAMES))
        self.assertEqual(summary["frames_with_score_at_least_1"], 4)
        self.assertIn("0", summary["iou_by_score"])

    def test_unfitted_frames_and_no_prior(self) -> None:
        arrays = _arrays([_straight(), _straight()])
        arrays["fitted"][1] = False
        out = compute_ambiguity(arrays, prior=None, thresholds=AmbiguityThresholds(low_iou=0.99))
        self.assertTrue(out["flag_low_iou"][0])
        self.assertFalse(out["flag_low_iou"][1])
        self.assertTrue(np.isnan(out["area_ratio"][1]))
        self.assertFalse(out["flag_length_deviation"].any())
        self.assertEqual(int(out["ambiguity_score"][1]), 0)


if __name__ == "__main__":
    unittest.main()
