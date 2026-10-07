"""Host target assembly preserves camera censoring and crop placement."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from worm_pose_gen.batch_fit import _window_targets
from worm_pose_gen.mask_fit import CropWindow, signed_edge_distance


def expected_targets(masks: list[np.ndarray], windows: list[CropWindow], height: int, width: int) -> tuple[torch.Tensor, ...]:
    """The targets frame by frame, as the group assembly must reproduce them."""

    target = torch.zeros((len(masks), height, width))
    valid = torch.zeros_like(target)
    distance = torch.full_like(target, -float("inf"))
    for f, (mask, window) in enumerate(zip(masks, windows)):
        x0, x1 = max(0, window.x0), min(window.image_width, window.x1)
        y0, y1 = max(0, window.y0), min(window.image_height, window.y1)
        local = torch.from_numpy(mask[y0:y1, x0:x1])
        rows, columns = slice(y0 - window.y0, y1 - window.y0), slice(x0 - window.x0, x1 - window.x0)
        target[f, rows, columns] = local.float()
        valid[f, rows, columns] = 1
        distance[f, rows, columns] = float("inf") if local.all() else signed_edge_distance(local)
    return target, distance, valid


class WindowTargetsRuntimeTests(unittest.TestCase):
    def assert_targets(self, masks: list[np.ndarray], windows: list[CropWindow], height: int, width: int) -> None:
        actual = _window_targets(masks, windows, height, width, torch.device("cpu"))
        for result, expected in zip(actual, expected_targets(masks, windows, height, width)):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.device.type, "cpu")

    def test_group_matches_per_frame_targets_with_camera_overhang(self) -> None:
        masks = [np.zeros((8, 10), dtype=bool), np.zeros((16, 20), dtype=bool), np.ones((12, 16), dtype=bool)]
        masks[0][:6, :7] = True
        masks[1][3:11, 4:13] = True
        windows = [CropWindow(-3, 13, -2, 10, 8, 10), CropWindow(0, 16, 1, 13, 16, 20),
                   CropWindow(0, 16, 0, 12, 12, 16)]
        self.assert_targets(masks, windows, 12, 16)

    def test_frames_sharing_a_mask_get_their_own_window(self) -> None:
        # Propagation passes one mask object once per start; the same mask in
        # another window, or an equal copy, must still get its own targets.
        mask = np.zeros((16, 20), dtype=bool)
        mask[3:11, 4:13] = True
        mask[6, 8] = False
        other = np.zeros((16, 20), dtype=bool)
        other[5:12, 2:9] = True
        windows = [CropWindow(0, 16, 1, 13, 16, 20), CropWindow(2, 18, 0, 12, 16, 20)]
        masks = [mask, other, mask, mask, mask.copy(), other]
        frame_windows = [windows[0], windows[0], windows[0], windows[1], windows[0], windows[0]]
        self.assert_targets(masks, frame_windows, 12, 16)


if __name__ == "__main__":
    unittest.main()
