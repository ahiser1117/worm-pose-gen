"""Boundary and connectivity invariants for accelerated mask cleanup."""
from __future__ import annotations

import unittest

import numpy as np

from worm_pose_gen.classical import _connected_extension, _dilate, _erode, _largest_component


class ClassicalMorphologyTests(unittest.TestCase):
    def test_square_filters_match_direct_neighborhoods(self) -> None:
        rng = np.random.default_rng(25)
        for shape in ((1, 11), (13, 1), (13, 19)):
            mask = rng.random(shape) < 0.6
            for radius in (0, 1, 2, 8, 17):
                with self.subTest(shape=shape, radius=radius):
                    size = 2 * radius + 1
                    windows = np.lib.stride_tricks.sliding_window_view(
                        np.pad(mask, radius, constant_values=False), (size, size)
                    )
                    np.testing.assert_array_equal(_dilate(mask, radius), windows.any(axis=(-1, -2)))
                    np.testing.assert_array_equal(_erode(mask, radius), windows.all(axis=(-1, -2)))

    def test_equal_area_components_keep_first_raster_component(self) -> None:
        mask = np.zeros((8, 12), dtype=bool)
        mask[1, 8] = mask[2, 9] = True  # Diagonal connectivity, encountered first.
        mask[5, 1:3] = True
        component, area, count = _largest_component(mask)
        expected = np.zeros_like(mask)
        expected[1, 8] = expected[2, 9] = True
        np.testing.assert_array_equal(component, expected)
        self.assertEqual((area, count), (2, 2))
        empty, area, count = _largest_component(np.zeros_like(mask))
        self.assertFalse(empty.any())
        self.assertEqual((area, count), (0, 0))

    def test_connected_extension_retains_ineligible_seed_and_grows_diagonally(self) -> None:
        seed = np.zeros((8, 8), dtype=bool)
        seed[0, 0] = True
        eligible = np.eye(8, dtype=bool)
        eligible[0, 0] = False
        eligible[0, 7] = True  # Disconnected debris.
        np.testing.assert_array_equal(_connected_extension(seed, eligible), np.eye(8, dtype=bool))
        np.testing.assert_array_equal(_connected_extension(np.zeros_like(seed), eligible), np.zeros_like(seed))


if __name__ == "__main__":
    unittest.main()
