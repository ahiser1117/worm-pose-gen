"""Host target assembly preserves camera censoring and crop placement."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from worm_pose_gen.batch_fit import _window_targets
from worm_pose_gen.mask_fit import CropWindow, signed_edge_distance


class WindowTargetsRuntimeTests(unittest.TestCase):
    def test_group_matches_per_frame_targets_with_camera_overhang(self) -> None:
        masks = [np.zeros((8, 10), dtype=bool), np.zeros((16, 20), dtype=bool), np.ones((12, 16), dtype=bool)]
        masks[0][:6, :7] = True
        masks[1][3:11, 4:13] = True
        windows = [CropWindow(-3, 13, -2, 10, 8, 10), CropWindow(0, 16, 1, 13, 16, 20),
                   CropWindow(0, 16, 0, 12, 12, 16)]
        expected_target = torch.zeros((3, 12, 16))
        expected_valid = torch.zeros_like(expected_target)
        expected_distance = torch.full_like(expected_target, -float("inf"))
        for f, (mask, window) in enumerate(zip(masks, windows)):
            x0, x1 = max(0, window.x0), min(window.image_width, window.x1)
            y0, y1 = max(0, window.y0), min(window.image_height, window.y1)
            local = torch.from_numpy(mask[y0:y1, x0:x1])
            rows, columns = slice(y0 - window.y0, y1 - window.y0), slice(x0 - window.x0, x1 - window.x0)
            expected_target[f, rows, columns] = local.float()
            expected_valid[f, rows, columns] = 1
            expected_distance[f, rows, columns] = float("inf") if local.all() else signed_edge_distance(local)
        actual = _window_targets(masks, windows, 12, 16, torch.device("cpu"))
        for result, expected in zip(actual, (expected_target, expected_distance, expected_valid)):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
