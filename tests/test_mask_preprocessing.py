"""Equivalence checks for exact distance and morphology preprocessing."""
from __future__ import annotations

from collections import deque
import unittest

import numpy as np
import torch

from worm_pose_gen.mask_fit import fill_narrow_holes, signed_edge_distance


def _reference_hole_fill(mask: np.ndarray, radius: int) -> np.ndarray:
    # Breadth-first 8-connected background components of the padded box, and
    # a brute-force search for a (2r+1) square in each enclosed one.
    yy, xx = np.nonzero(mask)
    y0, y1 = max(0, int(yy.min()) - 1), min(mask.shape[0], int(yy.max()) + 2)
    x0, x1 = max(0, int(xx.min()) - 1), min(mask.shape[1], int(xx.max()) + 2)
    local = mask[y0:y1, x0:x1]
    h, w = local.shape
    seen = local.copy()
    addition = np.zeros_like(local)
    size = 2 * radius + 1
    for start in np.ndindex(h, w):
        if seen[start]:
            continue
        seen[start] = True
        component, queue = [start], deque([start])
        while queue:
            y, x = queue.popleft()
            for dy, dx in np.ndindex(3, 3):
                ny, nx = y + dy - 1, x + dx - 1
                if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx]:
                    seen[ny, nx] = True
                    component.append((ny, nx))
                    queue.append((ny, nx))
        region = np.zeros_like(local)
        region[tuple(np.transpose(component))] = True
        if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
            continue
        if not any(region[y : y + size, x : x + size].all() for y in range(h - size + 1) for x in range(w - size + 1)):
            addition |= region
    output = mask.copy()
    output[y0:y1, x0:x1] |= addition
    return output


class MaskPreprocessingTests(unittest.TestCase):
    def test_signed_distance_matches_brute_force_including_camera_edges(self) -> None:
        rng = np.random.default_rng(921)
        masks = [rng.random((13, 17)) < density for density in (0.1, 0.5, 0.9)]
        masks += [np.eye(12, dtype=bool), np.arange(15)[None, :] < 7]
        edge = np.zeros((19, 23), dtype=bool)
        edge[:12, :17] = True
        masks.extend([edge, edge[::-1, ::-1].copy(), edge.T])
        for binary in masks:
            with self.subTest(shape=binary.shape, area=binary.sum()):
                expected = np.empty(binary.shape, dtype=np.float32)
                for y, x in np.ndindex(binary.shape):
                    opposite = np.argwhere(binary != binary[y, x])
                    distance = np.sqrt(((opposite - [y, x]) ** 2).sum(axis=1).min()) - 0.5
                    expected[y, x] = distance if binary[y, x] else -distance
                actual = signed_edge_distance(torch.from_numpy(binary), chunk_pixels=1)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device.type, "cpu")
                np.testing.assert_array_equal(actual.numpy(), expected)

    def test_signed_distance_rejects_degenerate_masks(self) -> None:
        for mask in (torch.zeros((5, 7)), torch.ones((5, 7)), torch.zeros(7)):
            with self.assertRaises(ValueError):
                signed_edge_distance(mask)

    def test_hole_fill_matches_shortest_path_reference(self) -> None:
        rng = np.random.default_rng(170)
        random = rng.random((17, 21)) < 0.7
        cropped = np.zeros((25, 31), dtype=bool)
        cropped[4:21, 5:26] = random
        masks = [random, cropped, np.ones((11, 13), dtype=bool)]
        for binary in masks:
            for radius in (1, 2, 3, 15):
                with self.subTest(shape=binary.shape, radius=radius):
                    expected = _reference_hole_fill(binary, radius)
                    actual, added = fill_narrow_holes(binary, radius, device="cpu")
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(added, int(expected.sum() - binary.sum()))

    def test_hole_fill_preserves_empty_masks_and_zero_radius(self) -> None:
        for binary in (np.zeros((8, 9), dtype=bool), np.eye(9, dtype=bool)):
            actual, added = fill_narrow_holes(binary, 0, device="cpu")
            np.testing.assert_array_equal(actual, binary)
            self.assertEqual(added, 0)
            self.assertFalse(np.shares_memory(actual, binary))


if __name__ == "__main__":
    unittest.main()
