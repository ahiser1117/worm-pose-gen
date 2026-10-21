import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

from worm_pose_gen.stage_alignment import (
    StageCalibration,
    align_frame,
    bandpass,
    clean_stage,
    fit_calibration,
    register_shift,
    shift_image,
    substrate_shift_px,
)


def speckle(shape: tuple[int, int] = (128, 160), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 + 40.0 * ndimage.gaussian_filter(rng.normal(size=shape), 1.5)


class StageTrackTests(unittest.TestCase):
    def test_gaps_are_interpolated_and_long_ones_flagged(self) -> None:
        raw = np.stack([np.arange(12.0) * 10, np.arange(12.0) * -5], axis=1)
        raw[[0, 3, 4, 7, 8, 9]] = np.nan  # leading read, a gap of 2, a gap of 3
        track = clean_stage(raw, max_gap=2)
        np.testing.assert_allclose(track.filled[1:], np.stack([np.arange(1, 12.0) * 10, np.arange(1, 12.0) * -5], axis=1))
        np.testing.assert_array_equal(track.gap, [1, 0, 0, 2, 2, 0, 0, 3, 3, 3, 0, 0])
        np.testing.assert_array_equal(track.reliable, [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])

    def test_fractional_lookup_is_nan_where_unreliable(self) -> None:
        raw = np.stack([np.arange(10.0), np.zeros(10)], axis=1)
        raw[5:8] = np.nan
        track = clean_stage(raw, max_gap=2)
        at = track.at([1.25, 3.5, 4.5, 8.5, 9.5, -0.5])
        np.testing.assert_allclose(at[[0, 1, 3], 0], [1.25, 3.5, 8.5])
        self.assertTrue(np.isnan(at[[2, 4, 5]]).all())  # touches the gap of 3, or leaves the recording


class RegistrationTests(unittest.TestCase):
    def test_synthetic_shift_is_recovered_with_a_masked_object(self) -> None:
        reference = speckle()
        shift = np.array([5.3, -2.7])
        moving = ndimage.shift(reference, shift[::-1], order=3, mode="wrap")
        invalid = np.zeros(reference.shape, dtype=bool)
        invalid[50:80, 60:100] = True
        moving[invalid] = 0.0  # a dark worm that does not move with the background
        found, score = register_shift(bandpass(moving), bandpass(reference), invalid)
        np.testing.assert_allclose(found, shift, atol=0.15)
        self.assertGreater(score, 10.0)

    def test_shift_image_moves_content_and_marks_pixels_from_outside(self) -> None:
        image = speckle((40, 50))
        shifted, valid = shift_image(image, (3.0, -2.0))
        np.testing.assert_allclose(shifted[5:30, 10:40], image[7:32, 7:37])
        self.assertFalse(valid[:, :3].any())
        self.assertFalse(valid[-2:, :].any())
        self.assertTrue(valid[:-2, 3:].all())

    def test_align_frame_uses_the_difference_of_substrate_shifts(self) -> None:
        image = speckle((40, 50))
        direct, _ = shift_image(image, (1.5, 0.5))
        aligned, valid = align_frame(image, (2.0, -1.0), (0.5, -1.5))
        np.testing.assert_allclose(aligned[valid], direct[valid])


class CalibrationTests(unittest.TestCase):
    def test_fit_recovers_matrix_and_time_offset(self) -> None:
        rng = np.random.default_rng(1)
        raw = np.cumsum(rng.normal(scale=40.0, size=(400, 2)), axis=0)
        track = clean_stage(raw)
        matrix = np.array([[0.078, 0.002], [-0.001, 0.079]])
        offset = -0.5
        t = rng.integers(0, 380, size=150).astype(np.float64)
        lag = rng.choice([1.0, 4.0, 16.0], size=150)
        truth = (track.at(t + lag + offset) - track.at(t + offset)) @ matrix.T
        measured = truth + rng.normal(scale=0.1, size=truth.shape)
        measured[:10] = 0.0  # registrations locked onto the camera's fixed pattern
        shifts = np.column_stack([t, lag, measured, np.full(150, 30.0)])
        calibration = fit_calibration(track, shifts)
        np.testing.assert_allclose(calibration.matrix, matrix, atol=2e-3)
        self.assertAlmostEqual(calibration.offset_frames, offset, places=6)
        self.assertGreaterEqual(calibration.outliers, 8)
        self.assertLess(calibration.residual_px, 0.3)

    def test_substrate_shift_cancels_stage_motion_and_round_trips(self) -> None:
        raw = np.array([[0.0, 0.0], [10.0, 0.0], [np.nan, np.nan], [30.0, 20.0], [40.0, 40.0]])
        calibration = StageCalibration(matrix=((0.1, 0.0), (0.0, -0.2)), offset_frames=0.0, residual_px=0.5, pairs=10, outliers=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rec.h5"
            with h5py.File(path, "w") as handle:
                handle["/pos_stage"] = raw
            shift = substrate_shift_px(path, [0, 2, 4], calibration)
            calibration.save(Path(directory) / "c.json")
            self.assertEqual(StageCalibration.load(Path(directory) / "c.json"), calibration)
        # The background moves by +A d in the image; adding the shift undoes it.
        np.testing.assert_allclose(shift, [[0.0, 0.0], [-2.0, 2.0], [-4.0, 8.0]])


if __name__ == "__main__":
    unittest.main()
