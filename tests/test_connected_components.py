from __future__ import annotations

from collections import deque
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

    def test_matches_classical_reference(self) -> None:
        for seed in range(6):
            mask = _blobs(seed)
            labels, count = label_components(mask)
            largest, area, ref_count = _largest_component(mask)
            fast, fast_area, fast_count = largest_component(mask)
            self.assertEqual(count, ref_count, msg=f"seed {seed}")
            self.assertEqual(fast_count, ref_count)
            self.assertEqual(fast_area, area)
            self.assertTrue(np.array_equal(fast, largest), msg=f"seed {seed}")
            # Components partition the foreground without losing pixels.
            areas = component_areas(labels, count)
            self.assertEqual(int(areas[1:].sum()), int(mask.sum()))
            self.assertEqual(int(areas[0]), int((~mask).sum()))
            self.assertTrue(np.all((labels > 0) == mask))

    def test_full_labels_match_independent_raster_flood_fill(self) -> None:
        rng = np.random.default_rng(825)
        for density in (0.0, 0.05, 0.3, 0.6, 1.0):
            mask = rng.random((23, 29)) < density
            expected = np.zeros(mask.shape, dtype=np.int32)
            count = 0
            for y, x in np.ndindex(mask.shape):
                if not mask[y, x] or expected[y, x]:
                    continue
                count += 1
                expected[y, x] = count
                queue = deque([(y, x)])
                while queue:
                    cy, cx = queue.popleft()
                    for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                        for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                            if mask[ny, nx] and not expected[ny, nx]:
                                expected[ny, nx] = count
                                queue.append((ny, nx))
            labels, actual_count = label_components(mask)
            np.testing.assert_array_equal(labels, expected)
            self.assertEqual(actual_count, count)
            self.assertEqual(labels.dtype, np.int32)
            self.assertTrue(labels.flags.c_contiguous)

    def test_equal_area_tie_and_zero_sized_inputs(self) -> None:
        mask = np.zeros((8, 12), dtype=bool)
        mask[1, 9] = mask[2, 10] = True
        mask[5, 1:3] = True
        largest, area, count = largest_component(mask)
        expected = np.zeros_like(mask)
        expected[1, 9] = expected[2, 10] = True
        np.testing.assert_array_equal(largest, expected)
        self.assertEqual((area, count), (2, 2))
        for shape in ((0, 5), (5, 0)):
            labels, count = label_components(np.zeros(shape, dtype=bool))
            self.assertEqual(labels.shape, shape)
            self.assertEqual(count, 0)

    def test_labels_are_dense_and_ordered(self) -> None:
        mask = _blobs(3)
        labels, count = label_components(mask)
        present = np.unique(labels[mask])
        self.assertTrue(np.array_equal(present, np.arange(1, count + 1)))
        first = np.argwhere(mask)[0]
        self.assertEqual(labels[tuple(first)], 1)


if __name__ == "__main__":
    unittest.main()
