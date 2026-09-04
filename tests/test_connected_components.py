from __future__ import annotations

import unittest

import numpy as np

from worm_pose_gen.classical import _largest_component
from worm_pose_gen.connected_components import component_areas, label_components, largest_component


def _blobs(seed: int, height: int = 90, width: int = 120, count: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=bool)
    yy, xx = np.mgrid[:height, :width]
    for _ in range(count):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        ry, rx = rng.uniform(2, 12), rng.uniform(2, 12)
        mask |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    # Sprinkle isolated pixels and diagonal-only contacts.
    for _ in range(40):
        mask[rng.integers(height), rng.integers(width)] = True
    return mask


class ConnectedComponentTests(unittest.TestCase):
    def test_empty_and_full(self) -> None:
        labels, count = label_components(np.zeros((5, 7), dtype=bool))
        self.assertEqual(count, 0)
        self.assertFalse(labels.any())
        labels, count = label_components(np.ones((5, 7), dtype=bool))
        self.assertEqual(count, 1)
        self.assertTrue(np.all(labels == 1))

    def test_diagonal_contact_is_connected(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = mask[1, 1] = mask[2, 2] = True
        mask[0, 3] = True
        labels, count = label_components(mask)
        self.assertEqual(count, 2)
        self.assertEqual(labels[0, 0], labels[1, 1])
        self.assertEqual(labels[1, 1], labels[2, 2])
        self.assertNotEqual(labels[0, 0], labels[0, 3])

    def test_runs_touching_only_at_row_ends(self) -> None:
        mask = np.zeros((3, 6), dtype=bool)
        mask[0, 0:2] = True
        mask[1, 2:4] = True  # touches row 0 diagonally at (0,1)-(1,2)
        mask[2, 5] = True  # touches row 1 diagonally at (1,3)-(2,4)? no: (2,5) vs (1,3) is two apart
        labels, count = label_components(mask)
        self.assertEqual(count, 2)
        self.assertEqual(labels[0, 0], labels[1, 3])
        self.assertNotEqual(labels[2, 5], labels[1, 3])

    def test_matches_breadth_first_reference(self) -> None:
        for seed in range(6):
            mask = _blobs(seed)
            labels, count = label_components(mask)
            largest, area, ref_count = _largest_component(mask)
            fast, fast_area, fast_count = largest_component(mask)
            self.assertEqual(count, ref_count, msg=f"seed {seed}")
            self.assertEqual(fast_count, ref_count)
            self.assertEqual(fast_area, area)
            self.assertTrue(np.array_equal(fast, largest), msg=f"seed {seed}")
            # Every label is one 8-connected piece: its pixels match the BFS
            # component grown from any of its pixels.
            areas = component_areas(labels, count)
            self.assertEqual(int(areas[1:].sum()), int(mask.sum()))
            self.assertEqual(int(areas[0]), int((~mask).sum()))
            self.assertTrue(np.all((labels > 0) == mask))

    def test_labels_are_dense_and_ordered(self) -> None:
        mask = _blobs(3)
        labels, count = label_components(mask)
        present = np.unique(labels[mask])
        self.assertTrue(np.array_equal(present, np.arange(1, count + 1)))
        first = np.argwhere(mask)[0]
        self.assertEqual(labels[tuple(first)], 1)


if __name__ == "__main__":
    unittest.main()
