"""Regression coverage for batched path costs and inference buffer reuse."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch import nn

from worm_pose_gen.pipeline import Frames
from worm_pose_gen.latent import cubic_bspline_basis
from worm_pose_gen.propagation import PropagationConfig, _path_edge_costs, pose_distance_px
from worm_pose_gen.segmenter import INPUT_MEAN, INPUT_STD, SegmentationModule


class PipelineRuntimeTests(unittest.TestCase):
    def test_cached_basis_remains_independently_writable(self) -> None:
        expected = cubic_bspline_basis(99, 16)
        changed = cubic_bspline_basis(99, 16)
        changed[:] = 0.0
        np.testing.assert_array_equal(cubic_bspline_basis(99, 16), expected)
        np.testing.assert_allclose(expected.sum(axis=1), 1.0, atol=1e-12)
        self.assertEqual(cubic_bspline_basis(31, 6).shape, (31, 6))

    def test_batched_path_costs_match_scalar_with_partial_visibility(self) -> None:
        rng = np.random.default_rng(193)
        previous = [(rng.normal(15, 30, (31, 2)), 20 + k, 6.0 + k, 100.0 + k) for k in range(5)]
        current = [(rng.normal(15, 30, (31, 2)), 20 - k, 7.0 + k, 110.0 - k) for k in range(7)]
        current.append((np.full((31, 2), -100.0), 0, 7.0, 110.0))
        current.append((np.full((31, 2), np.nan), 0, 7.0, 110.0))
        config = PropagationConfig()
        for shape in (None, (32, 48)):
            expected = np.empty((len(previous), len(current)))
            for i, a in enumerate(previous):
                for j, b in enumerate(current):
                    distance = pose_distance_px(a[0], b[0], shape)
                    if not np.isfinite(distance):
                        distance = 0.0
                    length_change = math.log(max(b[3], 1.0) / max(a[3], 1.0)) / config.path_length_sigma
                    expected[i, j] = (
                        config.path_distance_weight * (distance / max(0.5 * (a[2] + b[2]), 1.0)) ** 2
                        + config.path_inview_weight * abs(a[1] - b[1]) / 31
                        + config.path_length_weight * length_change**2
                    )
            np.testing.assert_allclose(_path_edge_costs(previous, current, config, 31, shape), expected, rtol=1e-14, atol=1e-14)

    def test_uncorrected_frames_do_not_copy_the_slab(self) -> None:
        raw = np.arange(3 * 7 * 9, dtype=np.uint8).reshape(3, 7, 9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("frames", data=raw)
            frames = Frames(path, flat_field=False, dataset="frames")
            frames.read = lambda indices: raw
            try:
                corrected, _, _ = frames.corrected([0, 1, 2])
                self.assertIs(corrected, raw)
            finally:
                frames.close()

    def test_inference_preallocation_preserves_values_and_training_mode(self) -> None:
        class Predictor(nn.Module):
            predict_probability_batch = SegmentationModule.predict_probability_batch
            device = torch.device("cpu")

            def forward(self, image):
                self.assert_eval = not self.training
                return image * 2.0

        model = Predictor()
        frames = np.arange(5 * 7 * 9, dtype=np.uint8).reshape(5, 7, 9)
        for batch_size in (1, 2, 8):
            expected = torch.cat([
                torch.sigmoid(((torch.as_tensor(chunk).float() / 255.0 - INPUT_MEAN) / INPUT_STD) * 2)
                for chunk in (frames[start : start + batch_size] for start in range(0, len(frames), batch_size))
            ]).numpy()
            actual = model.predict_probability_batch(frames, batch_size=batch_size)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(model.training)
            self.assertTrue(model.assert_eval)
        self.assertEqual(model.predict_probability_batch(frames[:0]).shape, (0, 7, 9))


if __name__ == "__main__":
    unittest.main()
