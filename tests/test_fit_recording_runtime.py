"""The recording runner reuses exact cleaned masks across temporal passes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from worm_pose_gen.pipeline import SegmentParams


SPEC = importlib.util.spec_from_file_location("fit_recording_runtime", Path(__file__).resolve().parents[1] / "scripts" / "fit_recording.py")
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class _Frames:
    def __init__(self, masks: list[np.ndarray], indices: np.ndarray) -> None:
        self.raw = {int(index): np.where(mask, 0, 255).astype(np.uint8) for index, mask in zip(indices, masks, strict=True)}
        self.calls: list[list[int]] = []

    def corrected(self, indices):
        self.calls.append(list(indices))
        return np.stack([self.raw[index] for index in indices]), 0.0, 0.0


class _CountingSegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def predict_probability_batch(self, frames, batch_size=16):
        self.calls += 1
        return 1.0 - frames.astype(np.float32) / 255.0


class RecordingMaskCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.masks = [np.zeros((6, 9), dtype=bool) for _ in range(4)]
        for mask, count in zip(self.masks, [6, 0, 8, 1], strict=True):
            mask.ravel()[:count] = True
        self.indices = np.array([10, 13, 17, 20])
        self.frames = _Frames(self.masks, self.indices)
        self.model = _CountingSegmenter()
        self.params = SegmentParams(fill_holes=False, largest_only=False, hole_radius=0, min_worm_pixels=4, batch_size=2)

    def rows(self, rows, cache):
        return runner._segment_cached_rows(rows, self.frames, self.model, self.indices, self.params, torch.device("cpu"), cache)

    def test_independent_masks_and_overlapping_requests_do_not_resegment(self) -> None:
        cache = runner._PackedMaskCache()
        masks, stats, _ = runner.segment_frames(self.frames, self.model, self.indices[:2], self.params, torch.device("cpu"))
        for row, mask, stat in zip([0, 1], masks, stats, strict=True):
            cache.put(row, mask, stat["worm_pixels"])
        first = self.rows([0, 1, 2], cache)
        second = self.rows([2, 3, 1, 0], cache)
        third = self.rows([3, 1, 2], cache)
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(self.frames.calls, [[10, 13], [17], [20]])
        self.assertEqual(list(first), [0, 2])
        self.assertEqual(list(second), [2, 0])
        self.assertEqual(list(third), [2])
        for output in (first, second, third):
            for row, mask in output.items():
                np.testing.assert_array_equal(mask, self.masks[row])
        third[2][:] = False
        np.testing.assert_array_equal(self.rows([2], cache)[2], self.masks[2])
        self.assertEqual(self.model.calls, 3)

    def test_evicted_masks_fall_back_to_equivalent_segmentation(self) -> None:
        packed_bytes = np.packbits(self.masks[0].ravel()).nbytes
        cache = runner._PackedMaskCache(max_bytes=packed_bytes)
        first = self.rows([0], cache)
        self.rows([2], cache)
        self.assertIsNone(cache.get(0))
        again = self.rows([0], cache)
        np.testing.assert_array_equal(first[0], again[0])
        self.assertEqual(self.frames.calls, [[10], [17], [10]])
        self.assertLessEqual(cache.nbytes, cache.max_bytes)
        disabled = runner._PackedMaskCache(max_bytes=packed_bytes - 1)
        self.rows([0], disabled)
        self.assertEqual(disabled.nbytes, 0)
        self.assertIsNone(disabled.get(0))

    def test_lru_keeps_recently_read_masks(self) -> None:
        packed_bytes = np.packbits(self.masks[0].ravel()).nbytes
        cache = runner._PackedMaskCache(max_bytes=2 * packed_bytes)
        for row in (0, 1):
            cache.put(row, self.masks[row], int(self.masks[row].sum()))
        cache.get(0)
        cache.put(2, self.masks[2], 8)
        self.assertIsNone(cache.get(1))
        self.assertIsNotNone(cache.get(0))
        self.assertIsNotNone(cache.get(2))

    def test_compile_energy_is_opt_in(self) -> None:
        for flags, expected in [([], {}), (["--compile-energy"], {"compile_energy": True})]:
            with patch("sys.argv", ["fit_recording.py", "--recording", "development.h5", *flags]):
                params = runner.fit_params(runner.parse_args())
            self.assertEqual(params.overrides, expected)


if __name__ == "__main__":
    unittest.main()
