from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import default_width_template
from worm_pose_gen.pose_run import (
    EXTRA_RGB,
    MISSED_RGB,
    boundary,
    clean_mask,
    draw_overlay,
    draw_residual,
    overlay_caption,
    render_tube,
    residual_rows,
    run_label,
    tube_area_px,
    write_overlay_video,
)


def _pose(height: int = 120, width: int = 160):
    latent = np.concatenate((np.zeros(16), [0.0, 100.0], [width / 2, height / 2]))
    return decode_centerline(latent), 10.0 * default_width_template()


class PoseRunTests(unittest.TestCase):
    def test_render_tube_window_matches_full_render(self) -> None:
        curve, profile = _pose()
        full = render_tube(curve, profile, 120, 160, device="cpu")
        x0, x1 = int(curve[:, 0].min()), int(curve[:, 0].max()) + 1
        y0, y1 = int(curve[:, 1].min()), int(curve[:, 1].max()) + 1
        windowed = render_tube(curve, profile, 120, 160, window=(x0, x1, y0, y1), margin=16, device="cpu")
        np.testing.assert_array_equal(windowed, full)
        self.assertGreater(full.sum(), 0.8 * tube_area_px(profile, 100.0))
        self.assertLess(full.sum(), 1.2 * tube_area_px(profile, 100.0))

    def test_residual_tints_missed_and_extra_pixels(self) -> None:
        curve, profile = _pose()
        tube = render_tube(curve, profile, 120, 160, device="cpu")
        mask = np.roll(tube, 6, axis=0)  # shifted mask: some pixels missed, some extra
        frame = np.full((120, 160), 200, dtype=np.uint8)
        image = np.asarray(draw_residual(frame, mask, tube, None, ""))
        missed = mask & ~tube
        extra = tube & ~mask
        expected_missed = (0.5 * 200 + np.asarray(MISSED_RGB)).astype(np.uint8)
        expected_extra = (0.5 * 200 + np.asarray(EXTRA_RGB)).astype(np.uint8)
        np.testing.assert_array_equal(image[missed][0], expected_missed)
        np.testing.assert_array_equal(image[extra][0], expected_extra)
        untouched = ~mask & ~tube
        untouched[:20] = False  # caption area
        self.assertTrue((image[untouched] == 200).all())
        cropped = draw_residual(frame, mask, tube, curve, "x", window=(10, 90, 20, 100))
        self.assertEqual(cropped.size, (80, 80))

    def test_overlay_marks_tube_boundary_and_scales(self) -> None:
        curve, profile = _pose()
        tube = render_tube(curve, profile, 120, 160, device="cpu")
        frame = np.zeros((120, 160), dtype=np.uint8)
        image = draw_overlay(frame, curve, tube, "", 1.0)
        edge = boundary(tube)
        self.assertTrue((image[edge] != 0).any(axis=1).all())
        self.assertEqual(draw_overlay(frame, None, None, "", 0.5).shape, (60, 80, 3))

    def test_residual_rows_and_clean_mask(self) -> None:
        arrays = {
            "fitted": np.array([True, True, False, True]),
            "iou": np.array([0.9, 0.5, np.nan, 0.7]),
            "frame_index": np.array([10, 11, 12, 13]),
        }
        self.assertEqual(residual_rows(arrays, 2, [10, 12, 99]), [0, 1, 3])
        self.assertEqual(residual_rows(arrays, 0, []), [])
        probability = np.zeros((40, 40), dtype=np.float32)
        probability[5:15, 5:30] = 0.9
        probability[8:10, 12:14] = 0.1  # narrow hole
        probability[30:33, 30:33] = 0.9  # small second component
        mask, stats = clean_mask(probability, 0.5, 2, "cpu")
        self.assertTrue(mask[9, 13])
        self.assertFalse(mask[31, 31])
        self.assertEqual(stats["components"], 2)
        self.assertEqual(stats["pixels_filled"], 4)
        self.assertEqual(stats["worm_pixels"], 10 * 25)
        self.assertEqual(run_label({}), "symmetric template")
        self.assertEqual(run_label({"width_model": {"coefficients": 6}}), "6 width coefficients")

    def test_overlay_video_renders_from_the_final_arrays(self) -> None:
        import tempfile

        import imageio.v2 as imageio

        curve, profile = _pose()
        n = 6
        frames = np.full((n, 120, 160), 180, dtype=np.uint8)
        arrays = {
            "frame_index": np.arange(10, 10 + n),
            "fitted": np.array([True, True, False, True, True, True]),
            "centerline_xy": np.tile(curve, (n, 1, 1)),
            "width_profile": np.tile(profile, (n, 1)),
            "crop": np.tile(np.array([20, 140, 30, 90]), (n, 1)),
            "iou": np.full(n, 0.95),
            "body_length_px": np.full(n, 100.0),
            "width_px": np.full(n, 10.0),
            "points_in_fov": np.full(n, 100),
            "taper_asymmetry": np.full(n, -0.2),
            "orientation_gap": np.full(n, np.nan),
            "source": np.array([0, 1, 0, 2, 0, 0], dtype=np.int8),
            "ambiguity_score": np.array([0, 2, 0, 1, 0, 0]),
        }
        caption = overlay_caption("rec", arrays, 1)
        self.assertIn("forward", caption)
        self.assertIn("score 2", caption)
        self.assertNotIn("gap", caption)
        self.assertEqual(overlay_caption("rec", arrays, 2), "rec frame 12  no fit")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.mp4"
            # A dataset indexed by absolute frame number: pad the front so frame 10 is the first stored frame.
            dataset = np.concatenate((np.zeros((10, 120, 160), dtype=np.uint8), frames))
            write_overlay_video(path, dataset, None, arrays, caption_prefix="rec", scale=0.5, slab=4, device="cpu")
            reader = imageio.get_reader(str(path))
            count = sum(1 for _ in reader)
            reader.close()
        self.assertEqual(count, n)


if __name__ == "__main__":
    unittest.main()
