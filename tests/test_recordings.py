from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

import h5py
import numpy as np
from PIL import Image

from worm_pose_gen.recordings import (
    RecordingInfo,
    find_recordings,
    list_recordings,
    probe_frames,
    probe_recording,
    thumbnail_png,
)


HEIGHT, WIDTH, FRAMES = 96, 128, 6


def _write_recording(path: Path, frames: int = FRAMES, height: int = HEIGHT, width: int = WIDTH) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:height, :width]
    stack = np.empty((frames, height, width), dtype=np.uint8)
    for index in range(frames):
        background = np.full((height, width), 190.0)
        body = np.abs(yy - (48 + 10 * np.sin(xx / 20 + index))) < 6
        body &= (xx > 12) & (xx < 116)
        image = background - 80 * body + rng.normal(0, 3, (height, width))
        stack[index] = np.clip(image, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=stack)


def _write_garbage(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not an HDF5 file" * 64)


class RecordingsFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.data = self.root / "data"
        self.rec_a = self.data / "2024-01-31" / "2024-01-31-02.h5"
        self.rec_b = self.data / "2024-05-28" / "2024-05-28-02.h5"
        self.bad = self.data / "2023-03-30" / "2023-03-30-01.h5"
        _write_recording(self.rec_a)
        _write_recording(self.rec_b, frames=4, height=64, width=80)
        _write_garbage(self.bad)
        (self.data / "2024-01-31" / "2024-01-31-02.nd2").write_bytes(b"ignored")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def listing(self, **kwargs) -> list[RecordingInfo]:
        kwargs.setdefault("poses_root", None)
        kwargs.setdefault("workspaces_root", None)
        kwargs.setdefault("prior_cache", None)
        kwargs.setdefault("cache", None)
        return list_recordings([self.data], **kwargs)


class ListRecordingsTests(RecordingsFixture):
    def test_finds_h5_files_recursively_and_ignores_other_suffixes(self) -> None:
        found = find_recordings([self.data, self.root / "missing"])
        self.assertEqual([p.name for p in found], ["2023-03-30-01.h5", "2024-01-31-02.h5", "2024-05-28-02.h5"])

    def test_readable_recordings_report_shape_and_garbage_reports_error(self) -> None:
        infos = self.listing()
        by_name = {info.name: info for info in infos}
        self.assertEqual(sorted(by_name), ["2023-03-30-01", "2024-01-31-02", "2024-05-28-02"])
        a = by_name["2024-01-31-02"]
        self.assertTrue(a.readable)
        self.assertIsNone(a.error)
        self.assertEqual((a.frames, a.height, a.width), (FRAMES, HEIGHT, WIDTH))
        self.assertEqual(a.path, str(self.rec_a))
        self.assertEqual(a.size_bytes, self.rec_a.stat().st_size)
        self.assertTrue(a.modified_at.endswith("+00:00"))
        self.assertFalse(a.prior_cached)
        self.assertEqual((a.runs, a.workspaces), ([], []))
        b = by_name["2024-05-28-02"]
        self.assertEqual((b.frames, b.height, b.width), (4, 64, 80))
        bad = by_name["2023-03-30-01"]
        self.assertFalse(bad.readable)
        self.assertIsNotNone(bad.error)
        self.assertEqual((bad.frames, bad.height, bad.width), (None, None, None))
        self.assertEqual(bad.size_bytes, self.bad.stat().st_size)
        json.dumps([info.to_dict() for info in infos])  # the payload is JSON-serialisable

    def test_probe_rejects_files_without_the_dataset(self) -> None:
        other = self.data / "other.h5"
        with h5py.File(other, "w") as handle:
            handle.create_dataset("/something", data=np.zeros((2, 2), dtype=np.uint8))
        facts = probe_recording(other)
        self.assertFalse(facts["readable"])
        self.assertIn("img_nir", facts["error"])

    def test_cache_is_reused_until_the_file_changes(self) -> None:
        cache = self.root / "index" / "recordings.json"
        self.listing(cache=cache)
        self.assertTrue(cache.exists())
        data = json.loads(cache.read_text())
        self.assertEqual(data["version"], 1)
        self.assertEqual(set(data["entries"]), {str(self.rec_a), str(self.rec_b), str(self.bad)})
        # Plant a sentinel in the cached entry: an unchanged file must come back from the cache verbatim.
        data["entries"][str(self.rec_a)]["frames"] = 999
        cache.write_text(json.dumps(data))
        infos = {info.name: info for info in self.listing(cache=cache)}
        self.assertEqual(infos["2024-01-31-02"].frames, 999)
        self.assertEqual(infos["2024-05-28-02"].frames, 4)
        # A rewritten file (new size and mtime) is probed again and the cache refreshed.
        _write_recording(self.rec_a, frames=7)
        os.utime(self.rec_a, ns=(self.rec_a.stat().st_atime_ns, self.rec_a.stat().st_mtime_ns + 5_000_000_000))
        infos = {info.name: info for info in self.listing(cache=cache)}
        self.assertEqual(infos["2024-01-31-02"].frames, 7)
        data = json.loads(cache.read_text())
        self.assertEqual(data["entries"][str(self.rec_a)]["frames"], 7)
        # Garbage entries are cached too, so the failing open is not repeated every scan.
        self.assertFalse(data["entries"][str(self.bad)]["readable"])

    def test_concurrent_listings_share_one_cache_file(self) -> None:
        # The app serves listings from a thread pool; each writes the cache when it probed something.
        cache = self.root / "index" / "recordings.json"
        errors: list[BaseException] = []
        counts: list[int] = []

        def listing() -> None:
            try:
                counts.append(len(self.listing(cache=cache)))
            except BaseException as error:  # noqa: BLE001 - collected for the assertion
                errors.append(error)

        for _ in range(3):
            if cache.exists():
                cache.unlink()
            threads = [threading.Thread(target=listing) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(counts, [3] * 18)
        self.assertEqual(json.loads(cache.read_text())["version"], 1)
        self.assertEqual([p.name for p in cache.parent.iterdir()], ["recordings.json"])

    def test_cache_survives_a_corrupt_index_file(self) -> None:
        cache = self.root / "recordings.json"
        cache.write_text("{not json")
        infos = self.listing(cache=cache)
        self.assertEqual(len(infos), 3)
        self.assertEqual(json.loads(cache.read_text())["version"], 1)

    def test_runs_workspaces_and_priors_are_matched_to_recordings(self) -> None:
        poses = self.root / "poses"
        for name, recording in (
            ("2026-09-04T00-00-00Z_2024-01-31-02_f000000-000399", str(self.rec_a)),
            ("2026-09-04T00-00-01Z_2024-01-31-02_moved", "/elsewhere/2024-01-31-02.h5"),
            ("2026-09-04T00-00-02Z_other", "/elsewhere/2023-08-22-01.h5"),
        ):
            (poses / name).mkdir(parents=True)
            (poses / name / "summary.json").write_text(json.dumps({"recording": recording}))
        (poses / "incomplete").mkdir()
        (poses / "stray.txt").write_text("not a run")
        workspaces = self.root / "workspaces"
        (workspaces / "clip-a").mkdir(parents=True)
        (workspaces / "clip-a" / "workspace.json").write_text(json.dumps({"name": "clip-a", "recording": str(self.rec_b)}))
        priors = self.root / "priors"
        priors.mkdir()
        (priors / "2024-05-28-02_k6.json").write_text("{}")
        infos = {
            info.name: info
            for info in self.listing(poses_root=poses, workspaces_root=workspaces, prior_cache=priors)
        }
        self.assertEqual(
            infos["2024-01-31-02"].runs,
            ["2026-09-04T00-00-00Z_2024-01-31-02_f000000-000399", "2026-09-04T00-00-01Z_2024-01-31-02_moved"],
        )
        self.assertEqual(infos["2024-01-31-02"].workspaces, [])
        self.assertFalse(infos["2024-01-31-02"].prior_cached)
        self.assertEqual(infos["2024-05-28-02"].runs, [])
        self.assertEqual(infos["2024-05-28-02"].workspaces, ["clip-a"])
        self.assertTrue(infos["2024-05-28-02"].prior_cached)

    def test_probe_samples_first_middle_and_last_frames(self) -> None:
        self.assertEqual(probe_frames(0), ())
        self.assertEqual(probe_frames(1), (0,))
        self.assertEqual(probe_frames(2), (0, 1))
        self.assertEqual(probe_frames(20481), (0, 10240, 20480))

    def test_same_stem_in_two_directories_lists_both_paths(self) -> None:
        copy = self.data / "2024-05-28" / "2024-01-31-02.h5"
        copy.write_bytes(self.rec_a.read_bytes())
        infos = [info for info in self.listing() if info.name == "2024-01-31-02"]
        self.assertEqual([info.path for info in infos], [str(self.rec_a), str(copy)])
        self.assertTrue(all(info.readable for info in infos))

    def test_missing_roots_yield_an_empty_catalog(self) -> None:
        self.assertEqual(list_recordings([self.root / "nowhere"], poses_root=None, workspaces_root=None, prior_cache=None), [])


class ThumbnailTests(RecordingsFixture):
    def test_raw_thumbnail_is_a_scaled_png(self) -> None:
        png = thumbnail_png(self.rec_a, 2, scale=0.25, dataset_root=self.root / "no-dataset")
        self.assertTrue(png.startswith(b"\x89PNG"))
        image = Image.open(io.BytesIO(png))
        self.assertEqual(image.size, (WIDTH // 4, HEIGHT // 4))
        self.assertEqual(image.mode, "L")
        full = np.asarray(Image.open(io.BytesIO(thumbnail_png(self.rec_a, 2, scale=1.0, dataset_root=self.root / "no-dataset"))))
        with h5py.File(self.rec_a, "r") as handle:
            self.assertTrue(np.array_equal(full, handle["/img_nir"][2]))
        # The dark worm body stays darker than the background after downscaling.
        small = np.asarray(image, dtype=np.float64)
        self.assertLess(small[:, 16].min(), small[0, 16] - 30)

    def test_flat_fielded_thumbnail_uses_the_cached_field(self) -> None:
        from worm_pose_gen.label_app import RecordingSource

        dataset_root = self.root / "dataset"
        source = RecordingSource(self.rec_a, dataset_root / "flat_fields")
        source.flat_field()
        source.close()
        self.assertTrue((dataset_root / "flat_fields" / "2024-01-31-02.npz").exists())
        png = thumbnail_png(self.rec_a, 1, scale=0.5, dataset_root=dataset_root)
        image = Image.open(io.BytesIO(png))
        self.assertEqual(image.size, (WIDTH // 2, HEIGHT // 2))
        raw = np.asarray(Image.open(io.BytesIO(thumbnail_png(self.rec_a, 1, scale=0.5, dataset_root=self.root / "none"))))
        self.assertFalse(np.array_equal(np.asarray(image), raw))

    def test_thumbnail_errors_are_explicit(self) -> None:
        with self.assertRaises(IndexError):
            thumbnail_png(self.rec_a, FRAMES, dataset_root=self.root / "none")
        with self.assertRaises(OSError):
            thumbnail_png(self.bad, 0, dataset_root=self.root / "none")
        with self.assertRaises(ValueError):
            thumbnail_png(self.rec_a, 0, scale=0.0, dataset_root=self.root / "none")


if __name__ == "__main__":
    unittest.main()
