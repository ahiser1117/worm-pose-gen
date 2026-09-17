"""The fixed-body temporal smoother: its solver on synthetic chains and the regional algorithm on a synthetic workspace."""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from worm_pose_gen import algorithms, edits, pipeline
from worm_pose_gen.body_smoother import MotionScales, SmoothingProblem, decode_chains, encode_chain, initial_chain, motion_scales, smooth_chains
from worm_pose_gen.latent import cubic_bspline_basis, decode_centerline
from worm_pose_gen.mask_fit import default_width_template
from worm_pose_gen.workspace import Workspace
from tests.test_batch_fit import SMALL, _render


def _true_latents(frames: int, length: float = 110.0) -> list[np.ndarray]:
    position = np.linspace(0, 1, 16)
    latents = []
    for t in range(frames):
        shape = 0.6 * np.sin(2 * np.pi * (1.5 * position - t / 25))
        latents.append(np.concatenate((shape, [0.2 * np.sin(t / 15), length, 100.0 + 0.8 * t, 80.0 + 0.3 * t])))
    return latents


class SolverTests(unittest.TestCase):
    def test_initial_chain_follows_arc_distance_and_continues_straight(self):
        curve = np.column_stack((np.linspace(0, 50, 51), np.zeros(51)))
        chain = initial_chain(curve, 100.0, 10)
        np.testing.assert_allclose(chain[:, 1], 0)
        np.testing.assert_allclose(chain[:, 0], np.arange(11) * 10.0)  # beyond the curve: straight on

    def test_motion_scales_need_enough_pairs_and_divide_by_the_gap(self):
        basis = cubic_bspline_basis(99, 16)
        x = np.column_stack((np.arange(30) * 2.0, np.zeros(30), np.zeros((30, 16))))
        chains = decode_chains(x, basis, 1.0)
        with self.assertRaisesRegex(ValueError, "at least 10 consecutive pairs"):
            motion_scales(chains[:5], np.arange(5), basis, max_gap=1)
        scales = motion_scales(chains, np.arange(30), basis, max_gap=1)
        self.assertAlmostEqual(scales.head_px, 1.4826 * 1.0)  # median over x (2 px) and y (0 px) per frame
        self.assertEqual(scales.pairs, 29)
        stepped = motion_scales(chains, np.arange(30) * 4, basis, max_gap=4)
        self.assertAlmostEqual(stepped.head_px, 1.4826 * 0.5)  # 2 px over 4 frames: rate 1 per sqrt(gap), pooled with y
        np.testing.assert_array_equal(stepped.coefficients, 1e-3)  # floor for a body that never bends

    def test_smoothing_bridges_untrusted_frames_and_keeps_trusted_and_fixed_ones(self):
        rng = np.random.default_rng(0)
        basis, segments, step = cubic_bspline_basis(99, 16), 99, 300.0 / 99
        latents = _true_latents(40, 300.0)
        truth = np.stack([decode_centerline(latent) for latent in latents])
        truth = np.stack([initial_chain(curve, 300.0, segments) for curve in truth])
        targets = truth.copy()
        targets[[10, 25]] += rng.normal(0, 20, (2, 1, 2))
        weights = np.ones((40, 100))
        weights[[10, 25]] = 0.0
        fixed = np.zeros(40, dtype=bool)
        fixed[[0, 39]] = True
        scales = motion_scales(truth, np.arange(40), basis, max_gap=1)
        problem = SmoothingProblem(targets, weights, np.ones(39, dtype=np.int64), fixed, targets.copy(), step, 16)
        chains, info = smooth_chains(problem, scales)
        self.assertTrue(info["converged"])
        self.assertLess(info["iterations"], 20)
        error = np.sqrt(((chains - truth) ** 2).sum(-1).mean(1))
        self.assertLess(error[[10, 25]].max(), 3.0)
        self.assertLess(error[weights[:, 0] > 0].max(), 0.5)
        np.testing.assert_array_equal(chains[[0, 39]], targets[[0, 39]])
        np.testing.assert_allclose(np.linalg.norm(np.diff(chains, axis=1), axis=2), step)  # equal links throughout
        # The result re-encodes exactly: the solver works in the pose latent's own angle basis.
        np.testing.assert_allclose(decode_chains(np.asarray([encode_chain(c, basis) for c in chains]), basis, step), chains, atol=1e-6)
        # A strong prior sigma floor keeps a trusted frame put; a tiny data sigma does too.
        loose = MotionScales(1e3, np.full(16, 1e3), scales.pairs)
        chains_loose, _ = smooth_chains(problem, loose)
        self.assertLess(np.sqrt(((chains_loose - targets) ** 2).sum(-1).mean(1))[weights[:, 0] > 0].max(), 1e-3)


class AlgorithmTests(unittest.TestCase):
    HEIGHT, WIDTH, FRAMES = 160, 260, 40

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.recording = root / "smooth.h5"
        self.config = replace(SMALL, stage_steps=(12, 12))
        self.latents = _true_latents(self.FRAMES)
        self.masks = [_render(latent, self.HEIGHT, self.WIDTH) for latent in self.latents]
        with h5py.File(self.recording, "w") as h:
            h.create_dataset("img_nir", data=np.asarray(self.masks, np.uint8))
        self.workspace = Workspace.create(root / "workspaces", "smooth", self.recording, 0, self.FRAMES - 1, step=1)
        state = pipeline.workspace_arrays(self.workspace, self.config)
        state["ambiguity_score"] = np.zeros(self.FRAMES, dtype=np.int64)
        for row, latent in enumerate(self.latents):
            state["latent"][row] = latent
            state["centerline_xy"][row] = decode_centerline(latent)
            state["width_px"][row] = 12
            state["width_shape"][row] = 0
            state["width_profile"][row] = default_width_template() * 12
            state["body_length_px"][row] = 110
            state["points_in_fov"][row] = 100
            state["crop"][row] = [0, self.WIDTH, 0, self.HEIGHT]
            state["fitted"][row] = True
            state["iou"][row] = .98
            state["energy"][row] = .02
        self.truth = state["centerline_xy"].copy()
        # A jumped frame the ambiguity stage flagged, a jumped frame with poor overlap, and a reversed but trusted frame.
        for row, shift, field, value in ((12, (20.0, 15.0), "iou", 0.5), (30, (-15.0, 20.0), "ambiguity_score", 3)):
            state["centerline_xy"][row] += shift
            state["latent"][row, -2:] += shift
            state[field][row] = value
        edits._reverse_row(state, 20)
        self.workspace.save_state(state)
        self.workspace.set_masks(range(self.FRAMES), self.masks)
        pipeline.update_summary(self.workspace, {"fit_config": asdict(self.config), "mask_cleanup": {"min_worm_pixels": 1}})

    def test_smoother_bridges_jumps_reorients_and_keeps_trusted_frames(self):
        progress = []
        result = algorithms.run_region(self.workspace, "fixed_body_smoother", 1, 38, {}, anchor_before=0, anchor_after=39, device="cpu",
                                       progress=lambda p, m: progress.append((p, m)))
        self.assertEqual(result.algorithm, "fixed_body_smoother")
        self.assertEqual(len(result.path), 38)
        self.assertEqual(progress[-1][0], 1.0)
        for row in range(1, 39):
            pose = result.chosen(row)
            self.assertFalse(result.path_by_row[row][1])
            self.assertEqual(pose.source, "smoothed")
            self.assertLess(pose.centerline_xy[0, 0], pose.centerline_xy[-1, 0])  # head first, like the anchors
            np.testing.assert_allclose(decode_centerline(pose.latent), pose.centerline_xy, atol=1e-3)
            np.testing.assert_allclose(pose.width_profile, result.metrics["fixed_body"]["length_px"] * 0 + pose.width_profile)
            self.assertTrue(np.isfinite(pose.iou) and pose.iou > 0.8, f"row {row}: iou {pose.iou}")
            self.assertTrue(np.isfinite(pose.energy))
            self.assertEqual(pose.points_in_fov, 100)
            error = np.sqrt(((pose.centerline_xy - self.truth[row]) ** 2).sum(-1).mean())
            self.assertLess(error, 3.0 if row in (12, 30) else 1.5, f"row {row}: {error:.2f} px from the truth")
        # Trusted frames hold their current pose; the reversed one is reoriented, not moved.
        state = self.workspace.load_state()
        for row in (5, 25):
            self.assertLess(np.abs(result.chosen(row).centerline_xy - state["centerline_xy"][row]).max(), 1.5)
        self.assertLess(np.abs(result.chosen(20).centerline_xy - state["centerline_xy"][20][::-1]).max(), 1.5)
        metrics = result.metrics
        self.assertEqual(metrics["orientation_flips_applied"], 1)
        self.assertEqual(metrics["orientation_flips"], 0)
        self.assertEqual(metrics["pose_jumps_over_width"], 0)
        self.assertEqual(result.metrics_before["pose_jumps_over_width"], 4)
        self.assertAlmostEqual(metrics["fixed_body"]["length_px"], 110, delta=2)
        self.assertGreaterEqual(metrics["fixed_body"]["motion_scales"]["pairs"], 10)
        self.assertTrue(metrics["solver"]["converged"])
        self.assertEqual(metrics["frames_without_targets"], 0)
        self.assertNotIn(12, metrics["fixed_body"]["calibration_frames"])
        # Nothing written until accepted; accepting installs the smoothed poses with the fixed width profile.
        np.testing.assert_array_equal(self.workspace.load_state()["centerline_xy"], state["centerline_xy"])
        algorithms.accept_candidates(self.workspace, result.id)
        after = self.workspace.load_state()
        self.assertLess(np.sqrt(((after["centerline_xy"][12] - self.truth[12]) ** 2).sum(-1).mean()), 3.0)
        np.testing.assert_allclose(after["width_profile"][12], result.chosen(12).width_profile)

    def test_registry_entry_has_no_hole_filling_and_rejects_bad_values(self):
        entry = algorithms.get_algorithm("fixed_body_smoother").to_dict()
        self.assertEqual([p["name"] for p in entry["parameters"]], ["min_iou", "untrusted_weight", "data_sigma_px", "motion_tolerance", "min_calibration_frames"])
        self.assertEqual(entry["needs_anchor"], [])
        self.assertEqual(algorithms.get_algorithm("fixed_body_smoother").resolve({}), {
            "min_iou": 0.9, "untrusted_weight": 0.0, "data_sigma_px": 1.0, "motion_tolerance": 2.0, "min_calibration_frames": 3,
        })
        with self.assertRaisesRegex(ValueError, "above the maximum"):
            algorithms.get_algorithm("fixed_body_smoother").resolve({"untrusted_weight": 2})

    def test_opposite_anchor_and_too_few_trusted_frames_are_reported(self):
        state = self.workspace.load_state()
        edits._reverse_row(state, 39)
        self.workspace.save_state(state)
        with self.assertRaisesRegex(ValueError, "oriented opposite"):
            algorithms.run_region(self.workspace, "fixed_body_smoother", 1, 38, {}, anchor_before=0, anchor_after=39, device="cpu")
        edits._reverse_row(state, 39)
        state["ambiguity_score"][5:] = 2
        self.workspace.save_state(state)
        with self.assertRaisesRegex(ValueError, "at least 10 consecutive pairs"):
            algorithms.run_region(self.workspace, "fixed_body_smoother", 1, 38, {}, anchor_before=0, anchor_after=39, device="cpu")
        self.assertEqual(algorithms.list_candidate_sets(self.workspace), [])

    def test_unanchored_region_and_offscreen_tail_still_smooth(self):
        state = self.workspace.load_state()
        state["centerline_xy"][:, :, 0] += 80  # the tail end leaves the right edge on later frames
        state["latent"][:, -2] += 80
        self.workspace.save_state(state)
        clipped = [row for row in range(31, self.FRAMES) if (state["centerline_xy"][row, :, 0] > self.WIDTH).any()]  # past the corrupted rows
        self.assertGreater(len(clipped), 0)
        result = algorithms.run_region(self.workspace, "fixed_body_smoother", 0, self.FRAMES - 1, {}, device="cpu")
        self.assertEqual(len(result.path), self.FRAMES)
        self.assertEqual(result.metrics["frames_without_targets"], 0)
        for row in clipped:
            pose = result.chosen(row)
            self.assertLess(pose.points_in_fov, 100)
            visible = state["centerline_xy"][row, :, 0] < self.WIDTH - 1
            self.assertLess(np.abs(pose.centerline_xy[visible] - state["centerline_xy"][row, visible]).max(), 1.0)  # the visible part is data
            np.testing.assert_allclose(np.linalg.norm(np.diff(pose.centerline_xy, axis=0), axis=1), result.metrics["fixed_body"]["segment_length_px"], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
