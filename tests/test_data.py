from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np

from worm_pose_gen.data import HDF5FrameSource


class HDF5FrameSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "source.h5"
        with h5py.File(self.path, "w") as handle:
            handle.create_dataset("frames", data=np.arange(5 * 3 * 4, dtype=np.uint16).reshape(5, 3, 4))
            handle.create_group("not_a_dataset")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source(self, **kwargs: object) -> HDF5FrameSource:
        return HDF5FrameSource(
            self.path, "/frames", expected_ndim=3, expected_frame_shape=(3, 4),
            allowed_dtypes=(np.uint16,), max_frames_per_read=4, **kwargs,
        )

    def test_lazy_read_only_access_and_close(self) -> None:
        source = self.source()
        self.assertFalse(source.is_open)
        np.testing.assert_array_equal(source.read_frame(2), np.arange(24, 36).reshape(3, 4))
        self.assertTrue(source.is_open)
        source.close()
        self.assertFalse(source.is_open)

    def test_schema_is_explicit_and_validated(self) -> None:
        with self.assertRaises(ValueError):
            HDF5FrameSource(self.path, "frames")
        with self.assertRaises(ValueError):
            HDF5FrameSource(self.path, "/frames", expected_frame_shape=(4, 3)).info
        with self.assertRaises(TypeError):
            HDF5FrameSource(self.path, "/frames", allowed_dtypes=(np.float32,)).info
        with self.assertRaises(KeyError):
            HDF5FrameSource(self.path, "/unknown").info
        with self.assertRaises(TypeError):
            HDF5FrameSource(self.path, "/not_a_dataset").info

    def test_reads_are_bounded_and_indices_unambiguous(self) -> None:
        source = self.source()
        with self.assertRaises(ValueError):
            source.read_slice(0, 5)
        with self.assertRaises(IndexError):
            source.read_frame(-1)
        np.testing.assert_array_equal(source.read_indices([3, 1, 3])[:, 0, 0], [36, 12, 36])

    def test_window_padding_mapping_is_explicit(self) -> None:
        source = self.source()
        with self.assertRaises(IndexError):
            source.read_window(0, before=1, after=1)
        edge = source.read_window(0, before=1, after=1, padding="edge")
        np.testing.assert_array_equal(edge.source_indices, [0, 0, 1])
        np.testing.assert_array_equal(edge.valid_mask, [False, True, True])
        constant = source.read_window(
            4, before=0, after=2, padding="constant", constant_value=99
        )
        np.testing.assert_array_equal(constant.source_indices, [4, -1, -1])
        np.testing.assert_array_equal(constant.valid_mask, [True, False, False])
        self.assertTrue(np.all(constant.frames[1:] == 99))

    def test_pickling_drops_live_handle(self) -> None:
        source = self.source()
        source.read_frame(0)
        restored = pickle.loads(pickle.dumps(source))
        self.assertFalse(restored.is_open)
        np.testing.assert_array_equal(restored.read_frame(1), source.read_frame(1))
        restored.close()
        source.close()


if __name__ == "__main__":
    unittest.main()
