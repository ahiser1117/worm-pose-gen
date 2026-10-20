"""``recordings.RecordingSource``: reading frames and the flat field fitted once per recording, cached on disk."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import h5py
import numpy as np

from worm_pose_gen.recordings import RecordingSource


def _write_recording(path: str, frames: int = 6, height: int = 96, width: int = 128) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:height, :width]
    stack = np.empty((frames, height, width), dtype=np.uint8)
    for index in range(frames):
        background = np.full((height, width), 190.0)
        body = np.abs(yy - (48 + 10 * np.sin(xx / 20 + index))) < 6
        body &= (xx > 12) & (xx < 116)
        image = background - 80 * body + rng.normal(0, 3, (height, width))
        stack[index] = np.clip(image, 0, 255).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=stack)


class RecordingSourceTests(unittest.TestCase):
    def test_flat_field_concurrent_writers_use_independent_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "new.h5"
            _write_recording(str(recording), frames=4)
            cache = Path(directory) / "fields"
            sources = [RecordingSource(recording, cache) for _ in range(2)]
            barrier = threading.Barrier(2)
            save = np.savez_compressed

            def overlapping_save(handle, **values):
                save(handle, **values)
                barrier.wait(timeout=10)  # Both writes finish before either rename.

            try:
                with mock.patch("worm_pose_gen.recordings.np.savez_compressed", side_effect=overlapping_save):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        results = list(pool.map(lambda source: source.flat_field(), sources))
                with np.load(cache / "new.npz") as saved:
                    np.testing.assert_allclose(saved["gain"], results[0].gain)
                self.assertEqual(list(cache.glob("*.partial")), [])
            finally:
                for source in sources:
                    source.close()

    def test_flat_field_shared_source_calculates_once(self) -> None:
        from worm_pose_gen.flat_field import estimate_flat_field
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "new.h5"
            _write_recording(str(recording), frames=4)
            source = RecordingSource(recording, Path(directory) / "fields")
            started, release = threading.Event(), threading.Event()

            def calculate(*args, **kwargs):
                started.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("test did not release calculation")
                return estimate_flat_field(*args, **kwargs)

            try:
                with mock.patch("worm_pose_gen.recordings.estimate_flat_field", side_effect=calculate) as fit:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        first = pool.submit(source.flat_field)
                        second = pool.submit(source.flat_field)
                        try:
                            self.assertTrue(started.wait(timeout=10))
                        finally:
                            release.set()
                        self.assertIs(first.result(timeout=10), second.result(timeout=10))
                    self.assertEqual(fit.call_count, 1)
            finally:
                source.close()

    def test_flat_field_failed_save_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "new.h5"
            _write_recording(str(recording), frames=4)
            cache = Path(directory) / "fields"
            source = RecordingSource(recording, cache)
            try:
                with mock.patch("worm_pose_gen.recordings.np.savez_compressed", side_effect=OSError("disk full")):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        source.flat_field()
                self.assertFalse((cache / "new.npz").exists())
                self.assertEqual(list(cache.glob("*.partial")), [])
                source.flat_field()
                self.assertTrue((cache / "new.npz").exists())
            finally:
                source.close()

    def test_recording_source_reads_and_corrects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording = f"{directory}/rec-b.h5"
            _write_recording(recording, frames=4)
            source = RecordingSource(recording, f"{directory}/ff")
            raw, corrected = source.corrected(1)
            self.assertEqual(raw.shape, (96, 128))
            self.assertEqual(corrected.dtype, np.uint8)
            with self.assertRaises(IndexError):
                source.read(10)
            source.close()


if __name__ == "__main__":
    unittest.main()
