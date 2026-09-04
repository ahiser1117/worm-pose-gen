from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from worm_pose_gen.batch_fit import BatchFitConfig, fit_masks
from worm_pose_gen.latent import cubic_bspline_basis, decode_centerline
from worm_pose_gen.mask_fit import (
    Initialization,
    MaskFitConfig,
    _MaskFitState,
    default_width_template,
    extend_start_to_length,
    fit_mask,
    init_from_skeleton,
    orientation_pair,
    render_tube_segments,
    standard_initializations,
)
from worm_pose_gen.pose_run import touches_border
from worm_pose_gen.recording_prior import (
    RecordingPrior,
    bootstrap_config,
    bootstrap_prior_from_masks,
    estimate_recording_prior,
)


TAIL_LAST = np.array([0.25, 0.2, 0.1, -0.05, -0.35, -0.9])
SMALL = BatchFitConfig(
    stage_downsample=(2, 1),
    stage_steps=(60, 60),
    stage_lr_scale=(1.0, 0.3),
    stage_point_stride=(2, 1),
    crop_padding=16,
    crop_multiple=8,
    compile_renderer=False,
    length_bounds_px=(60.0, 300.0),
    width_bounds_px=(4.0, 30.0),
    default_length_px=150.0,
    default_width_px=12.0,
    moment_arc_curvatures=(0.0, 0.01, -0.01),
)


def _worm(seed: int, *, height: int = 160, width: int = 220, length: float = 150.0, centroid=None, body_width: float = 12.0):
    rng = np.random.default_rng(seed)
    shape = np.convolve(rng.normal(0.0, 0.4, 16), [0.25, 0.5, 0.25], "same")
    centroid = (width / 2, height / 2) if centroid is None else centroid
    latent = np.concatenate((shape, [rng.uniform(-3, 3), length], centroid))
    curve = decode_centerline(latent)
    template = default_width_template()
    correction = cubic_bspline_basis(100, len(TAIL_LAST)) @ TAIL_LAST
    profile = body_width * template * np.exp(correction - correction.mean())
    rendered = render_tube_segments(
        torch.as_tensor(curve, dtype=torch.float32)[None], torch.as_tensor(profile, dtype=torch.float32)[None], height, width
    )[0]
    return curve, (rendered >= 0.5).numpy()


class RecordingPriorTests(unittest.TestCase):
    def test_bounds_can_be_removed_and_priors_enter_the_energy(self) -> None:
        curve, mask = _worm(0)
        start = init_from_skeleton(mask, config=SMALL)
        assert start is not None
        free = replace(SMALL, length_bounds_px=None, width_bounds_px=None)
        state = _MaskFitState([start], free, torch.device("cpu"))
        with torch.no_grad():
            self.assertEqual(float(state.size_regularization()[0]), 0.0)
        with_prior = replace(free, length_prior_px=300.0, length_prior_log_sigma=0.05, width_prior_px=12.0, width_prior_log_sigma=0.03)
        state = _MaskFitState([start], with_prior, torch.device("cpu"))
        with torch.no_grad():
            penalty = float(state.size_regularization()[0])
        expected = with_prior.prior_weight * (
            ((np.log(start.latent[17]) - np.log(300.0)) / 0.05) ** 2 + ((np.log(start.width_px) - np.log(12.0)) / 0.03) ** 2
        )
        self.assertAlmostEqual(penalty, expected, places=3)
        centered = replace(free, width_shape_prior_mean=tuple(TAIL_LAST))
        state = _MaskFitState([Initialization("s", start.latent, start.width_px, TAIL_LAST)], centered, torch.device("cpu"))
        self.assertAlmostEqual(float(state.width_prior()[0]), 0.0)
        with self.assertRaises(ValueError):
            _MaskFitState([start], replace(free, width_shape_prior_mean=(0.0, 0.0)), torch.device("cpu"))

    def test_orientation_pair_shares_the_width_shape(self) -> None:
        curve, mask = _worm(1)
        start = init_from_skeleton(mask, config=SMALL)
        assert start is not None
        pair = orientation_pair(replace(start, width_shape=TAIL_LAST), config=SMALL)
        self.assertEqual(len(pair), 2)
        np.testing.assert_array_equal(pair[1].width_shape, TAIL_LAST)
        np.testing.assert_allclose(decode_centerline(pair[1].latent), decode_centerline(pair[0].latent)[::-1], atol=1e-9)

    def test_asymmetric_prior_picks_the_true_orientation(self) -> None:
        curve, mask = _worm(2)
        config = replace(SMALL, length_bounds_px=None, width_bounds_px=None, width_shape_prior_mean=tuple(TAIL_LAST))
        skeleton = init_from_skeleton(mask, config=config)
        assert skeleton is not None
        starts = orientation_pair(replace(skeleton, width_shape=TAIL_LAST), config=config)
        result = fit_mask(mask, starts, config=config, device="cpu")
        # The winner's tail (last point) sits at the true tail, not the head.
        self.assertLess(np.linalg.norm(result.centerline_xy[-1] - curve[-1]), 12.0)
        self.assertGreater(np.linalg.norm(result.centerline_xy[-1] - curve[0]), 40.0)
        energies = [r["final_soft_dice_energy"] for r in result.records]
        self.assertLess(min(energies), max(energies) - 1e-3)

    def test_estimate_prior_and_round_trip(self) -> None:
        masks = []
        results = []
        for seed in range(10):
            _, mask = _worm(seed)
            starts = standard_initializations(mask, config=SMALL)
            results.append(fit_mask(mask, starts, config=SMALL, device="cpu"))
            masks.append(mask)
        prior = estimate_recording_prior(results, masks, config=SMALL, min_frames=5)
        self.assertGreaterEqual(prior.frames_used, 5)
        self.assertAlmostEqual(prior.length_px, 150.0, delta=8.0)
        # The width scale is the geometric mean over the body, so compare it with the fits, not the midbody width.
        self.assertAlmostEqual(prior.width_px, float(np.median([r.width_px for r in results])), delta=0.5)
        self.assertLess(prior.width_shape[-1], prior.width_shape[0])  # tail last is thinner
        applied = prior.apply(SMALL)
        self.assertIsNone(applied.length_bounds_px)
        self.assertEqual(applied.length_prior_px, prior.length_px)
        self.assertEqual(applied.width_shape_prior_mean, prior.width_shape)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.json"
            prior.save(path)
            loaded = RecordingPrior.load(path)
        self.assertEqual(loaded, prior)
        self.assertIsInstance(json.dumps(prior.to_dict()), str)
        with self.assertRaises(ValueError):
            estimate_recording_prior(results[:2], masks[:2], config=SMALL, min_frames=5)
        with self.assertRaises(ValueError):
            prior.apply(replace(SMALL, width_coefficients=8))

    def test_bootstrap_opens_the_bounds_and_length_prior_completes_a_clipped_body(self) -> None:
        # A worm longer than the default bound would allow; bootstrapping must recover its length.
        long_config = replace(SMALL, length_bounds_px=(60.0, 120.0))
        self.assertEqual(bootstrap_config(long_config).length_bounds_px, (100.0, 3000.0))
        masks = [_worm(seed, height=200, width=260, length=180.0)[1] for seed in range(8)]
        prior, results, used = bootstrap_prior_from_masks(masks, config=long_config, device="cpu", min_frames=4)
        self.assertEqual(len(results), len(used))
        self.assertAlmostEqual(prior.length_px, 180.0, delta=10.0)
        # A body leaving the image: the visible part pins the pose, the prior supplies the rest.
        curve, clipped = _worm(3, height=160, width=220, length=180.0, centroid=(200.0, 80.0))
        self.assertLess(clipped[:, -1].sum(), clipped.sum())  # touches the right edge
        starts = standard_initializations(clipped, config=long_config)
        bounded = fit_mask(clipped, starts, config=long_config, device="cpu")
        with_prior = fit_mask(clipped, starts, config=prior.apply(long_config), device="cpu")
        self.assertLess(bounded.body_length_px, 150.0)  # held near the (soft) 120 px bound
        self.assertAlmostEqual(with_prior.body_length_px, 180.0, delta=15.0)
        self.assertLess(with_prior.points_in_fov, 100)

    def test_extend_start_grows_only_ends_cut_by_the_border(self) -> None:
        curve, whole = _worm(4, length=150.0)
        start = init_from_skeleton(whole, config=SMALL)
        assert start is not None
        self.assertFalse(touches_border(whole))
        self.assertIs(extend_start_to_length(start, whole, 200.0, config=SMALL), start)
        # The same body placed so its end runs off the right edge.
        _, clipped = _worm(4, length=150.0, centroid=(190.0, 80.0))
        self.assertTrue(touches_border(clipped))
        clipped_start = init_from_skeleton(clipped, config=SMALL)
        assert clipped_start is not None
        extended = extend_start_to_length(clipped_start, clipped, 200.0, config=SMALL)
        self.assertAlmostEqual(extended.latent[17], 200.0, delta=2.0)
        before = decode_centerline(clipped_start.latent)
        after = decode_centerline(extended.latent)
        # The visible part is unchanged: the end far from the border stays put, the near end moved off camera.
        far_before, far_after = (before[0], after[0]) if before[0, 0] < before[-1, 0] else (before[-1], after[-1])
        self.assertLess(np.linalg.norm(far_after - far_before), 3.0)
        self.assertGreater(max(after[0, 0], after[-1, 0]), 220.0)

    def test_escape_penalty_ignores_points_past_the_camera_edge(self) -> None:
        _, clipped = _worm(4, length=150.0, centroid=(190.0, 80.0))
        start = init_from_skeleton(clipped, config=SMALL)
        assert start is not None
        long = extend_start_to_length(start, clipped, 230.0, config=SMALL)
        from worm_pose_gen.mask_fit import crop_window

        crop = crop_window(clipped, SMALL.crop_padding, 8)
        state = _MaskFitState([long], replace(SMALL, length_bounds_px=None), torch.device("cpu"))
        with torch.no_grad():
            centerline = state.centerline()
            penalty = float(state.regularization(centerline, crop)[0])
        outside = ((centerline[0, :, 0] >= clipped.shape[1]) | (centerline[0, :, 1] >= clipped.shape[0]) | (centerline[0] < 0).any(-1)).sum()
        self.assertGreater(int(outside), 10)
        self.assertLess(penalty, 1e-3)


if __name__ == "__main__":
    unittest.main()
