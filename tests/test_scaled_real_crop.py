import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional

from worm_pose_gen.real_crop import (
    ScaledCropRequest,
    atomic_publish,
    attempt_scaled_real_crop,
    half_open_support,
)


class ScaledRealCropTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.indices((90, 120))
        self.image = (3 * x + 7 * y).astype(np.uint16)
        self.line = np.stack((np.arange(10, 90, dtype=np.float64), np.full(80, 45.0)), 1)
        self.request = ScaledCropRequest("head", 0.2, 24, 32, 8, 20)

    def test_direct_window_pixels_and_frozen_interpolation(self) -> None:
        attempt = attempt_scaled_real_crop(self.image, self.line, self.request)
        crop = attempt.crop
        self.assertIsNotNone(crop)
        x0, y0 = crop.source_origin_xy
        h, w = crop.source_window_shape
        np.testing.assert_array_equal(crop.source_window, self.image[y0 : y0 + h, x0 : x0 + w])
        expected = torch_functional.interpolate(
            torch.from_numpy(crop.source_window.astype(np.float32))[None, None],
            size=(24, 32), mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        np.testing.assert_array_equal(crop.image, expected)
        changed = self.image.copy()
        outside = np.ones_like(changed, dtype=bool)
        outside[y0 : y0 + h, x0 : x0 + w] = False
        changed[outside] += 1000
        repeated = attempt_scaled_real_crop(changed, self.line, self.request).crop
        self.assertEqual(repeated.source_origin_xy, crop.source_origin_xy)
        np.testing.assert_array_equal(repeated.image, crop.image)

    def test_support_before_and_after_transform_and_roundtrips(self) -> None:
        crop = attempt_scaled_real_crop(self.image, self.line, self.request).crop
        self.assertIsNotNone(crop)
        h, w = crop.source_window_shape
        before = half_open_support(self.line, crop.source_origin_xy, h, w)
        after = half_open_support(crop.centerline_resized_xy, (0, 0), 24, 32)
        np.testing.assert_array_equal(before, crop.support)
        np.testing.assert_array_equal(after, crop.support)
        source_restored = crop.window_to_source(crop.source_to_window(self.line))
        resized_restored = crop.resized_to_source(crop.source_to_resized(self.line))
        self.assertLessEqual(float(np.max(np.abs(source_restored - self.line))), 1e-5)
        self.assertLessEqual(float(np.max(np.abs(resized_restored - self.line))), 1e-4)
        self.assertAlmostEqual(crop.scale, 32 / (4 * crop.source_window_k))

    def test_smallest_scale_and_deterministic_origin(self) -> None:
        first = attempt_scaled_real_crop(self.image, self.line, self.request).crop
        second = attempt_scaled_real_crop(self.image, self.line, self.request).crop
        self.assertEqual(first.source_window_k, second.source_window_k)
        self.assertEqual(first.source_origin_xy, second.source_origin_xy)
        for k in range(self.request.k_min, first.source_window_k):
            restricted = ScaledCropRequest("head", 0.2, 24, 32, k, k)
            self.assertIsNone(attempt_scaled_real_crop(self.image, self.line, restricted).crop)

    def test_invalid_and_unusable_requests(self) -> None:
        invalid = [
            ScaledCropRequest("middle", 0.2, 24, 32, 8, 20),
            ScaledCropRequest("head", 0.0, 24, 32, 8, 20),
            ScaledCropRequest("head", 1.0, 24, 32, 8, 20),
            ScaledCropRequest("head", 0.2, 25, 32, 8, 20),
            ScaledCropRequest("head", 0.2, 0, 32, 8, 20),
            ScaledCropRequest("head", 0.2, 24, 32, 0, 20),
            ScaledCropRequest("head", 0.2, 24, 32, 20, 8),
        ]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(ValueError):
                attempt_scaled_real_crop(self.image, self.line, request)
        with self.assertRaises(ValueError):
            attempt_scaled_real_crop(self.image[..., None], self.line, self.request)
        nonfinite = self.line.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaises(ValueError):
            attempt_scaled_real_crop(self.image, nonfinite, self.request)
        long_line = np.stack((np.linspace(0, 119, 80), np.linspace(0, 89, 80)), 1)
        rejected = attempt_scaled_real_crop(
            self.image, long_line, ScaledCropRequest("head", 0.05, 24, 32, 8, 10)
        )
        self.assertIsNone(rejected.crop)
        self.assertEqual(rejected.rejection_reason, "visible_support_does_not_fit_maximum_window")

    def test_collision_refuse_overwrite_and_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "proxy.h5"
            source.write_bytes(b"proxy")
            with self.assertRaises(ValueError):
                atomic_publish(source, source, lambda path: path.write_bytes(b"bad"))
            output = root / "scaled.h5"
            atomic_publish(source, output, lambda path: path.write_bytes(b"complete"))
            self.assertEqual(output.read_bytes(), b"complete")
            with self.assertRaises(FileExistsError):
                atomic_publish(source, output, lambda path: path.write_bytes(b"bad"))


if __name__ == "__main__":
    unittest.main()
