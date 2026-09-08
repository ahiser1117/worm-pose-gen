from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from worm_pose_gen.batch_fit import BatchFitConfig, fit_masks
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import (
    Initialization,
    _MaskFitState,
    default_width_template,
    hard_iou,
    init_from_moments,
    render_tube_segments,
)
from worm_pose_gen.batch_fit import PRESETS
from worm_pose_gen.propagation import (
    Candidate,
    PropagationConfig,
    ambiguous_stretches,
    continuity_summary,
    jump_seeds,
    pose_distance_px,
    predict_latent,
    prior_penalty,
    propagate,
    select_candidates,
    select_path,
    slow_schedule,
    track_length,
    warm_schedule,
)


SMALL = BatchFitConfig(
    stage_downsample=(2, 1),
    stage_steps=(60, 60),
    stage_lr_scale=(1.0, 0.3),
    stage_point_stride=(2, 1),
    crop_padding=16,
    crop_multiple=8,
    compile_renderer=False,
    length_bounds_px=None,
    width_bounds_px=None,
    length_prior_px=150.0,
    width_prior_px=12.0,
    default_length_px=150.0,
    default_width_px=12.0,
    width_shape_prior_mean=(0.0,) * 6,
)


def _sequence(n: int = 6, height: int = 160, width: int = 220):
    """A worm whose curvature grows frame by frame into a tight hook."""

    template = default_width_template()
    curves, masks, latents = [], [], []
    for k in range(n):
        shape = np.zeros(16)
        shape[10:] = 0.45 * k  # the second half bends progressively
        latent = np.concatenate((shape, [0.2, 150.0], [width / 2, height / 2]))
        curve = decode_centerline(latent)
        rendered = render_tube_segments(
            torch.as_tensor(curve, dtype=torch.float32)[None], torch.as_tensor(12.0 * template, dtype=torch.float32)[None], height, width
        )[0]
        curves.append(curve)
        masks.append((rendered >= 0.5).numpy())
        latents.append(latent)
    return curves, masks, latents


class PropagationTests(unittest.TestCase):
    def test_stretches_pad_and_merge(self) -> None:
        score = np.array([0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0])
        fitted = np.ones_like(score, dtype=bool)
        stretches = ambiguous_stretches(score, fitted, PropagationConfig(min_score=2, pad=1, max_gap=3))
        self.assertEqual(stretches, [(2, 9), (14, 16)])
        self.assertEqual(ambiguous_stretches(score, fitted, PropagationConfig(min_score=5)), [])
        fitted[8] = False
        self.assertEqual(ambiguous_stretches(score, fitted, PropagationConfig(min_score=2, pad=0, max_gap=0)), [(3, 3), (15, 15)])

    def test_prior_penalty_matches_the_fitter(self) -> None:
        latent = np.concatenate((np.zeros(16), [0.1, 170.0], [100.0, 80.0]))
        shape = np.array([0.1, -0.2, 0.0, 0.3, 0.1, -0.1])
        start = Initialization("s", latent, 13.0, shape)
        config = replace(SMALL, width_shape_prior=0.02)
        state = _MaskFitState([start], config, torch.device("cpu"))
        with torch.no_grad():
            expected = float((state.size_regularization() + state.width_prior())[0])
        length = float(np.linalg.norm(np.diff(decode_centerline(latent), axis=0), axis=1).sum())
        self.assertAlmostEqual(prior_penalty(config, length, 13.0, shape), expected, places=4)

    def test_warm_schedule_is_short_and_keeps_priors(self) -> None:
        warm = warm_schedule(SMALL)
        self.assertLess(sum(warm.stage_steps), sum(SMALL.stage_steps))
        self.assertEqual(warm.stage_downsample, SMALL.stage_downsample)  # same raster: energies stay comparable
        self.assertEqual(warm.length_prior_px, SMALL.length_prior_px)
        self.assertEqual(warm.length_prior_log_sigma, SMALL.length_prior_log_sigma)
        tight = warm_schedule(SMALL, length_sigma=0.02)
        self.assertEqual(tight.length_prior_log_sigma, 0.02)
        self.assertEqual(warm_schedule(replace(SMALL, length_prior_px=None), length_sigma=0.02).length_prior_log_sigma, SMALL.length_prior_log_sigma)

    def test_redirect_sends_a_folded_start_through_the_border(self) -> None:
        from worm_pose_gen.mask_fit import redirect_start_through_exit

        # A body that runs off the right edge of a 160 x 220 image: the mask reaches the border.
        angle = np.zeros(16)
        latent = np.concatenate((angle, [0.0, 200.0], [150.0, 80.0]))
        curve = decode_centerline(latent)
        template = default_width_template()
        rendered = render_tube_segments(
            torch.as_tensor(curve, dtype=torch.float32)[None], torch.as_tensor(12.0 * template, dtype=torch.float32)[None], 160, 220
        )[0]
        mask = (rendered >= 0.5).numpy()
        self.assertTrue(mask[:, -2:].any())
        # A start of the same length folded back inside the image: a hook whose far end turns around near the edge.
        folded_shape = np.zeros(16)
        folded_shape[9:] = 1.2
        folded = Initialization("s", np.concatenate((folded_shape, [0.0, 200.0], [110.0, 80.0])), 12.0, np.zeros(6))
        inside = decode_centerline(folded.latent)
        self.assertTrue((inside[:, 0] < 220).all())
        redirected = redirect_start_through_exit(folded, mask, config=SMALL)
        self.assertIsNotNone(redirected)
        out = decode_centerline(redirected.latent)
        # The folded end is unfolded toward the exit: the start reaches farther right, its last point
        # is closer to the border contact than the folded end was, and the total length is kept.
        exit_xy = np.array([219.0, 80.0])
        self.assertGreater(out[:, 0].max(), inside[:, 0].max() + 15.0)
        self.assertLess(np.linalg.norm(out[-1] - exit_xy), min(np.linalg.norm(inside[0] - exit_xy), np.linalg.norm(inside[-1] - exit_xy)))
        self.assertAlmostEqual(redirected.latent[17], 200.0, delta=3.0)
        self.assertTrue(redirected.name.endswith("_exit"))
        # A start that already leaves the image, or a mask clear of the border, is left alone.
        self.assertIsNone(redirect_start_through_exit(Initialization("s", latent, 12.0), mask, config=SMALL))
        whole = np.zeros_like(mask)
        whole[60:100, 40:180] = True
        self.assertIsNone(redirect_start_through_exit(folded, whole, config=SMALL))

    def test_prediction_extrapolates_with_damping_and_holds_length(self) -> None:
        previous = np.concatenate((np.zeros(16), [0.1, 150.0], [100.0, 80.0]))
        current = np.concatenate((np.full(16, 0.2), [0.3, 152.0], [104.0, 80.0]))
        predicted = predict_latent(current, previous, 0.5)
        np.testing.assert_allclose(predicted[:16], 0.3)          # shape moves on by half the step
        self.assertAlmostEqual(predicted[16], 0.4)              # rotation too
        self.assertAlmostEqual(predicted[17], 152.0)            # length held
        np.testing.assert_allclose(predicted[18:], [106.0, 80.0])  # centroid velocity
        np.testing.assert_array_equal(predict_latent(current, previous, 0.0), current)
        np.testing.assert_array_equal(predict_latent(current, None, 0.5), current)
        # A rotation on the other branch of 2 pi is not a full turn of velocity.
        wrapped = previous.copy(); wrapped[16] = 0.1 + 2 * np.pi
        self.assertAlmostEqual(predict_latent(current, wrapped, 0.5)[16], 0.4)

    def test_pose_distance_uses_points_inside_the_camera(self) -> None:
        curve = np.stack((np.linspace(-20, 80, 100), np.full(100, 10.0)), axis=1)
        shifted = curve + (0.0, 3.0)
        self.assertAlmostEqual(pose_distance_px(curve, shifted, None), 3.0)
        self.assertAlmostEqual(pose_distance_px(curve, shifted, (50, 60)), 3.0)
        self.assertTrue(np.isnan(pose_distance_px(curve, shifted, (5, 5))))

    def test_track_length_and_jump_seeds(self) -> None:
        n = 8
        length = np.array([100.0, 101.0, 130.0, 99.0, 100.0, 70.0, 101.0, 100.0])
        arrays = {
            "fitted": np.ones(n, dtype=bool), "body_length_px": length, "width_px": np.full(n, 10.0),
            "mask_on_border": np.array([False, False, True, False, False, True, False, False]),
            "iou": np.full(n, 0.95), "pose_jump_px": np.array([np.nan, 2.0, 3.0, 2.0, 25.0, 2.0, 2.0, 2.0]),
            "centerline_xy": np.zeros((n, 100, 2)), "points_in_fov": np.array([100, 100, 80, 100, 100, 90, 100, 100]),
        }
        track = track_length(arrays, window=2)
        # Clipped frames (2 and 5) do not vote; every frame's track sits near 100.
        self.assertTrue(np.all(np.abs(track - 100.0) <= 1.0))
        arrays["track_length_px"] = track
        seeds = jump_seeds(arrays)
        # Frame 2 and 5: length far off the track (and in-view changes at the border); frame 4: pose jump.
        self.assertTrue(seeds[2] and seeds[5] and seeds[4])
        self.assertTrue(seeds[1] and seeds[3])  # the frames on either side of the in-view change at frame 2
        self.assertFalse(seeds[7])
        stretches = ambiguous_stretches(np.zeros(n, dtype=int), arrays["fitted"], PropagationConfig(pad=0, max_gap=0), seeds)
        self.assertEqual(stretches, [(1, 6)])
        self.assertEqual(ambiguous_stretches(np.zeros(n, dtype=int), arrays["fitted"], PropagationConfig(pad=0)), [])
        # No whole body at all: the fallback length.
        empty = dict(arrays); empty["mask_on_border"] = np.ones(n, dtype=bool)
        self.assertTrue(np.all(track_length(empty, fallback_px=123.0) == 123.0))
        self.assertTrue(np.all(np.isnan(track_length(empty))))

    def test_continuity_summary_counts_jumps(self) -> None:
        n = 6
        arrays = {
            "fitted": np.ones(n, dtype=bool), "body_length_px": np.array([100.0, 101.0, 110.0, 101.0, 100.0, 100.0]),
            "width_px": np.full(n, 10.0), "pose_jump_px": np.array([np.nan, 2.0, 15.0, 2.0, 2.0, 2.0]),
            "centerline_xy": np.zeros((n, 100, 2)), "points_in_fov": np.array([100, 100, 90, 100, 100, 100]),
            "prediction_distance_px": np.array([np.nan, np.nan, 4.0, 6.0, np.nan, np.nan]),
        }
        out = continuity_summary(arrays)
        self.assertEqual(out["consecutive_pairs"], 5)
        self.assertEqual(out["length_jumps_over_fraction"], 2)
        self.assertEqual(out["pose_jumps_over_width"], 1)
        self.assertEqual(out["in_view_changes"], 2)
        self.assertEqual(out["frames_with_prediction"], 2)
        self.assertAlmostEqual(out["prediction_distance_px_p50_p90_max"][0], 5.0)

    def test_propagation_recovers_frames_a_cold_start_misses(self) -> None:
        curves, masks, latents = _sequence()
        n = len(masks)
        # Independent fits: good on the first and last frame, deliberately bad (straight start, few steps) in between.
        good_first = fit_masks([masks[0]], [[Initialization("s", latents[0], 12.0)]], config=SMALL, device="cpu")[0]
        good_last = fit_masks([masks[-1]], [[Initialization("s", latents[-1], 12.0)]], config=SMALL, device="cpu")[0]
        poor_config = replace(SMALL, stage_steps=(3, 3))
        poor = [
            fit_masks([masks[k]], [[init_from_moments(masks[k], config=poor_config)]], config=poor_config, device="cpu")[0]
            for k in range(1, n - 1)
        ]
        results = [good_first, *poor, good_last]
        arrays = {
            "frame_index": np.arange(n),
            "fitted": np.ones(n, dtype=bool),
            "latent": np.stack([r.latent for r in results]),
            "width_px": np.array([r.width_px for r in results]),
            "width_shape": np.stack([r.width_shape for r in results]),
            "total_energy": np.array([r.records[r.best_index]["final_energy"] for r in results]),
            "body_length_px": np.array([r.body_length_px for r in results]),
            "iou": np.array([hard_iou(r.rendered_hard_mask, m[r.crop.y0 : r.crop.y1, r.crop.x0 : r.crop.x1]) for r, m in zip(results, masks, strict=True)]),
        }
        self.assertLess(arrays["iou"][1:-1].max(), 0.9)
        score = np.array([0, 2, 2, 2, 2, 0])
        stretches = ambiguous_stretches(score, arrays["fitted"], PropagationConfig(pad=0))
        self.assertEqual(stretches, [(1, 4)])
        arrays["energy"] = np.array([r.records[r.best_index]["final_soft_dice_energy"] for r in results])
        arrays["centerline_xy"] = np.stack([r.centerline_xy for r in results])
        arrays["points_in_fov"] = np.array([r.points_in_fov for r in results])
        propagation = PropagationConfig(pad=0, beam=2)
        candidates, info = propagate(arrays, stretches, {k: masks[k] for k in range(n)}, config=SMALL, device="cpu", propagation=propagation)
        self.assertEqual(info["chains"], 2)
        self.assertEqual(info["lockstep_steps"], 4)
        self.assertEqual(info["beam"], 2)
        self.assertEqual(info["independent_refits"], 4)
        self.assertEqual(sorted(candidates), [1, 2, 3, 4])
        self.assertEqual({c.source for c in candidates[2]}, {"independent", "forward", "backward"})
        # Step 6a bookkeeping: the first chain frame has no velocity (the
        # anchor's own neighbour is inside the stretch), later frames carry a
        # prediction and a distance to it; the winning start is named.
        forward = [c for c in candidates[3] if c.source == "forward"]
        self.assertLessEqual(len(forward), 2)
        self.assertEqual(sorted(c.beam for c in forward), list(range(len(forward))))
        self.assertIsNotNone(forward[0].prediction_xy)
        self.assertEqual(forward[0].prediction_xy.shape, (100, 2))
        self.assertTrue(np.isfinite(forward[0].distance_to_prediction_px))
        self.assertIn(forward[0].start_name, ("warm_forward", "predicted_forward"))
        first = [c for c in candidates[1] if c.source == "forward"][0]
        self.assertIsNone(first.prediction_xy)
        self.assertGreaterEqual(info["predicted_starts_offered"], 1)
        self.assertLessEqual(info["predicted_starts_won"], info["predicted_starts_offered"])
        # Beam states of one direction are distinct poses.
        for row in (2, 3, 4):
            states = [c for c in candidates[row] if c.source == "forward"]
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    self.assertGreater(pose_distance_px(states[i].result.centerline_xy, states[j].result.centerline_xy, None), 0.5 * 12.0 * 0.5)
        chosen = select_candidates(candidates, arrays, SMALL)
        self.assertEqual(sorted(chosen), [1, 2, 3, 4])
        for row, candidate in chosen.items():
            r = candidate.result
            iou = hard_iou(r.rendered_hard_mask, masks[row][r.crop.y0 : r.crop.y1, r.crop.x0 : r.crop.x1])
            self.assertGreater(iou, 0.9, f"row {row} via {candidate.source}: {iou:.3f}")
        # The path (step 6c) also covers every stretch frame and keeps the good fits.
        path = select_path(candidates, arrays, stretches, SMALL, propagation, image_shape=masks[0].shape)
        self.assertEqual(sorted(path), [1, 2, 3, 4])
        anchor = arrays["centerline_xy"][0]
        for row, choice in path.items():
            r = choice.candidate.result
            iou = hard_iou(r.rendered_hard_mask, masks[row][r.crop.y0 : r.crop.y1, r.crop.x0 : r.crop.x1])
            self.assertGreater(iou, 0.9, f"row {row} via path: {iou:.3f}")
            # Whatever orientation the candidate was fit in, the path presents it the anchors' way round.
            oriented = r.centerline_xy[::-1] if choice.mirrored else r.centerline_xy
            self.assertLess(pose_distance_px(oriented, anchor, None), pose_distance_px(oriented[::-1], anchor, None))
            self.assertTrue(np.isfinite(choice.cost))

    def test_slow_schedule_takes_steps_from_the_preset_and_keeps_rasters(self) -> None:
        slow = slow_schedule(SMALL, replace(SMALL, stage_steps=(100, 200), within_stage_decay=0.03), length_sigma=0.02)
        self.assertEqual(slow.stage_steps, (100, 200))
        self.assertEqual(slow.stage_downsample, SMALL.stage_downsample)
        self.assertAlmostEqual(slow.within_stage_decay, 0.03)
        self.assertAlmostEqual(slow.length_prior_log_sigma, 0.02)
        with self.assertRaisesRegex(ValueError, "rasters"):
            slow_schedule(SMALL, PRESETS["reference"])

    def test_path_prefers_the_continuous_candidate_and_follows_the_anchors_orientation(self) -> None:
        curves, masks, latents = _sequence(5)
        # Fits of frames 1..3 (the stretch) and their anchors 0 and 4.
        fits = [fit_masks([masks[k]], [[Initialization("s", latents[k], 12.0)]], config=SMALL, device="cpu")[0] for k in range(5)]
        n = 5
        arrays = {
            "fitted": np.ones(n, dtype=bool),
            "centerline_xy": np.stack([f.centerline_xy for f in fits]),
            "points_in_fov": np.array([f.points_in_fov for f in fits]),
            "width_px": np.array([f.width_px for f in fits]),
            "body_length_px": np.array([f.body_length_px for f in fits]),
            "total_energy": np.array([f.records[0]["final_energy"] for f in fits]),
        }
        # Per stretch frame two candidates: the true fit at a slightly higher
        # energy and a "jumping" candidate, the same body 40 px away, at a lower one.
        candidates = {}
        for row in (1, 2, 3):
            jumping = replace(fits[row], centerline_xy=fits[row].centerline_xy + (40.0, 0.0))
            candidates[row] = [Candidate("forward", fits[row], 0.10 + 0.002 * row), Candidate("backward", jumping, 0.09)]
        propagation = PropagationConfig(path_temperature=0.01, path_distance_weight=1.0, path_inview_weight=10.0)
        chosen = select_path(candidates, arrays, [(1, 3)], SMALL, propagation, image_shape=masks[0].shape)
        self.assertEqual(sorted(chosen), [1, 2, 3])
        for row, choice in chosen.items():
            self.assertIs(choice.candidate.result, fits[row], f"row {row} took the jumping candidate")
            self.assertTrue(choice.override)  # the gap (0.012 to 0.016) is well above OVERRIDE_ENERGY
            self.assertAlmostEqual(choice.energy_gap, 0.01 + 0.002 * row)
            self.assertFalse(choice.mirrored)
        # Lowest energy per frame would have taken the jumping candidate.
        self.assertEqual({c.source for c in select_candidates(candidates, arrays).values()}, {"backward"})
        # With the anchors reversed, the path chooses the mirrored candidates.
        reversed_arrays = dict(arrays)
        reversed_arrays["centerline_xy"] = arrays["centerline_xy"][:, ::-1].copy()
        chosen = select_path(candidates, reversed_arrays, [(1, 3)], SMALL, propagation, image_shape=masks[0].shape)
        for row, choice in chosen.items():
            self.assertIs(choice.candidate.result, fits[row])
            self.assertTrue(choice.mirrored, f"row {row} kept the anchors' opposite orientation")

    def test_selection_keeps_the_independent_fit_when_it_is_better(self) -> None:
        curves, masks, latents = _sequence(3)
        result = fit_masks([masks[1]], [[Initialization("s", latents[1], 12.0)]], config=SMALL, device="cpu")[0]
        arrays = {"fitted": np.array([True, True, True]), "total_energy": np.array([0.05, 0.01, 0.05])}
        candidates = {1: [Candidate("forward", result, 0.02), Candidate("backward", result, 0.03)]}
        self.assertEqual(select_candidates(candidates, arrays), {})
        arrays["total_energy"][1] = 0.5
        self.assertEqual(select_candidates(candidates, arrays)[1].source, "forward")


if __name__ == "__main__":
    unittest.main()
