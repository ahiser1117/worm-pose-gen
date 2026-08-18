import tempfile
import unittest
from pathlib import Path

import numpy as np

from worm_pose_gen.real_crop import (
    CropRequest,
    atomic_publish,
    attempt_real_crop,
    canonical_manifest_sha256,
    half_open_support,
    support_bitmask,
    target_support,
)


class RealCropTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
        self.line = np.stack((np.arange(2, 22, dtype=np.float64), np.full(20, 10.0)), axis=1)
        self.request = CropRequest("head", 0.2, 8, 16)

    def test_exact_half_open_support(self) -> None:
        points = np.asarray([[2, 3], [11.999, 8.999], [12, 5], [5, 9], [1.999, 3]])
        self.assertEqual(
            half_open_support(points, (2, 3), 6, 10).tolist(),
            [True, True, False, False, False],
        )
        self.assertEqual(target_support(20, "head", 0.2).tolist(), [False] * 4 + [True] * 16)
        self.assertEqual(support_bitmask(np.asarray([True, False, True])), "101")

    def test_manifest_hash_is_canonical_and_strict(self) -> None:
        first = {"proxy_sha256": "abc", "entries": [{"support": "101", "x": 2}]}
        reordered = {"entries": [{"x": 2, "support": "101"}], "proxy_sha256": "abc"}
        self.assertEqual(canonical_manifest_sha256(first), canonical_manifest_sha256(reordered))
        changed = {"proxy_sha256": "abc", "entries": [{"support": "100", "x": 2}]}
        self.assertNotEqual(canonical_manifest_sha256(first), canonical_manifest_sha256(changed))

    def test_round_trip_and_direct_pixel_provenance(self) -> None:
        attempt = attempt_real_crop(self.image, self.line, self.request)
        self.assertIsNone(attempt.rejection_reason)
        crop = attempt.crop
        self.assertIsNotNone(crop)
        x0, y0 = crop.source_origin_xy
        np.testing.assert_array_equal(crop.image, self.image[y0 : y0 + 8, x0 : x0 + 16])
        restored = crop.crop_to_source(crop.source_to_crop(self.line))
        self.assertLessEqual(float(np.max(np.abs(restored - self.line))), 1e-5)
        np.testing.assert_array_equal(
            crop.support, half_open_support(self.line, crop.source_origin_xy, 8, 16)
        )

    def test_crop_is_deterministic(self) -> None:
        first = attempt_real_crop(self.image, self.line, self.request).crop
        second = attempt_real_crop(self.image, self.line, self.request).crop
        self.assertEqual(first.source_origin_xy, second.source_origin_xy)
        np.testing.assert_array_equal(first.image, second.image)

    def test_invalid_and_unusable_crop_rejection(self) -> None:
        with self.assertRaises(ValueError):
            attempt_real_crop(self.image, self.line, CropRequest("middle", 0.2, 8, 16))
        too_tall = self.line.copy()
        too_tall[:, 1] = np.arange(20)
        rejected = attempt_real_crop(self.image, too_tall, CropRequest("head", 0.05, 4, 16))
        self.assertIsNone(rejected.crop)
        self.assertEqual(rejected.rejection_reason, "visible_support_does_not_fit")

    def test_source_collision_and_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5"
            source.write_bytes(b"source")
            with self.assertRaises(ValueError):
                atomic_publish(source, source, lambda path: path.write_bytes(b"bad"))
            output = root / "output.h5"
            output.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                atomic_publish(source, output, lambda path: path.write_bytes(b"bad"))
            self.assertEqual(output.read_bytes(), b"existing")
            published = root / "published.h5"
            atomic_publish(source, published, lambda path: path.write_bytes(b"complete"))
            self.assertEqual(published.read_bytes(), b"complete")
            self.assertFalse((root / "published.h5.partial").exists())
            with self.assertRaises(FileExistsError):
                atomic_publish(source, published, lambda path: path.write_bytes(b"bad"))


if __name__ == "__main__":
    unittest.main()
