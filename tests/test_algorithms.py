"""The algorithm registry on a synthetic workspace (CPU, no network).

The six-frame recording of ``tests/test_pipeline.py`` is segmented and fit
once (``checkpoint=None``, the SMALL schedule); the tests then run region
algorithms on rows of it with the outer frames as anchors, check the
candidate sets, their paths and metrics, and installing a result as the
Refit fix's Keep does (a preview, then ``fixes.keep``: one ``set_pose``
edit).  Tests that change the workspace put the pristine arrays back
afterwards.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from worm_pose_gen import algorithms, edits, fixes, pipeline
from worm_pose_gen.algorithms import (
    REGISTRY,
    CandidatePose,
    CandidateSet,
    Parameter,
    build_context,
    list_algorithms,
    propose_anchors,
    region_metrics,
    resolve_params,
    run_algorithm,
)
from worm_pose_gen.body_smoother import MotionScales
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.pipeline import read_summary, run_stage
from worm_pose_gen.propagation import pose_distance_px
from worm_pose_gen.workspace import Workspace

from tests.slow import slow
from tests.test_pipeline import FIT_PARAMS, FRAMES, HEIGHT, SEGMENT_PARAMS, WIDTH, _body_curve, _StubPredictor, _write_recording


PRISTINE = ("state.npz", "hypotheses.npz", "provenance.npz", "summary.json", "edits.jsonl")


def keep(workspace: Workspace, candidate_set: CandidateSet) -> edits.EditResult:
    """Install a region result as the Refit fix's Keep does: a preview of its path, then ``fixes.keep``."""

    preview = fixes.preview_from(workspace, fixes.next_preview_id(workspace), "refit", candidate_set, workspace.load_state())
    fixes.save_preview(workspace, preview)
    return fixes.keep(workspace, preview.id)


class RegistryTests(unittest.TestCase):
    def test_registry_lists_the_contract_algorithms_with_parameter_dicts(self) -> None:
        listed = list_algorithms()
        self.assertEqual([a["id"] for a in listed], ["independent_multistart", "chain_forward", "chain_backward", "beam_path", "slow_refit", "tracked_head", "fixed_body_smoother", "mirror"])
        for entry in listed:
            self.assertEqual(set(entry), {"id", "label", "scope", "description", "parameters", "needs_anchor"})
            self.assertEqual(entry["scope"], "region")
            self.assertTrue(entry["label"] and entry["description"])
            for parameter in entry["parameters"]:
                self.assertEqual(set(parameter), {"name", "type", "default", "help", "choices", "minimum", "maximum"})
                self.assertIn(parameter["type"], algorithms.PARAMETER_TYPES)
        beam = {p["name"]: p for p in next(a for a in listed if a["id"] == "beam_path")["parameters"]}
        for name in ("beam", "prediction_damping", "temporal_prior_weight", "temporal_prior_sigma_widths", "path_temperature", "path_distance_weight",
                     "path_inview_weight", "path_length_weight", "refit_independent", "anchor_diversity", "preset"):
            self.assertIn(name, beam)
        self.assertEqual(beam["preset"]["choices"], ["fast", "balanced", "reference"])
        self.assertEqual(set(REGISTRY), {a["id"] for a in listed})
        self.assertEqual(algorithms.get_algorithm("tracked_head").resolve({}), {
            "preset": "fast", "tracking_weight": 0.2, "previous_pose_weight": 0.005,
            "previous_head_weight": 0.5, "head_sigma_px": 6.0, "max_head_step_px": 8.0,
            "keep_head_in_frame": True, "fill_holes": "off",
        })
        # The chains declare the anchor they start from; a request without it is refused before anything runs.
        needs = {a["id"]: a["needs_anchor"] for a in listed}
        self.assertEqual((needs["chain_forward"], needs["chain_backward"], needs["mirror"], needs["beam_path"]), (["before"], ["after"], [], []))
        with self.assertRaisesRegex(ValueError, "needs an anchor before"):
            algorithms.get_algorithm("chain_forward").check_anchors(None, 5)
        with self.assertRaisesRegex(ValueError, "needs an anchor after"):
            algorithms.get_algorithm("chain_backward").check_anchors(0, None)
        algorithms.get_algorithm("chain_forward").check_anchors(0, None)

    def test_parameters_coerce_and_check_values(self) -> None:
        parameters = [Parameter("beam", "int", 3, "", minimum=1, maximum=8), Parameter("preset", "choice", "fast", "", choices=["fast", "balanced"]), Parameter("on", "bool", True, "")]
        resolved = resolve_params(parameters, {"beam": "2", "on": "false", "not_a_parameter": 1})
        self.assertEqual(resolved, {"beam": 2, "preset": "fast", "on": False})
        with self.assertRaisesRegex(ValueError, "not one of"):
            resolve_params(parameters, {"preset": "slow"})
        with self.assertRaisesRegex(ValueError, "above the maximum"):
            resolve_params(parameters, {"beam": 9})
        with self.assertRaisesRegex(ValueError, "unknown type"):
            Parameter("x", "list", None, "")
        with self.assertRaisesRegex(ValueError, "unknown algorithm"):
            algorithms.get_algorithm("no_such")

class RegionMetricTests(unittest.TestCase):
    def test_metrics_count_jumps_and_flips_at_the_boundary_too(self) -> None:
        n, points = 6, 100
        arrays = {
            "frame_index": np.arange(n), "fitted": np.ones(n, dtype=bool), "iou": np.array([0.95, 0.85, 0.92, 0.95, 0.96, 0.97]),
            "width_px": np.full(n, 10.0), "body_length_px": np.array([100.0, 100.0, 100.0, 108.0, 108.0, 108.0]),
            "points_in_fov": np.full(n, points), "centerline_xy": np.zeros((n, points, 2)),
        }
        base = np.stack((np.linspace(0, 99, points), np.full(points, 50.0)), axis=1)
        for row in range(n):
            arrays["centerline_xy"][row] = base
        arrays["centerline_xy"][2] = base + (30.0, 0.0)  # a jump into row 2 and out of it
        arrays["centerline_xy"][4] = base[::-1]  # reversed: flips at (3,4) and (4,5)
        metrics = region_metrics(arrays, [1, 2, 3], image_shape=(100, 200))
        self.assertEqual(metrics["frames"], 3)
        self.assertAlmostEqual(metrics["median_iou"], 0.92)
        self.assertEqual(metrics["frames_below_0_9"], 1)
        self.assertEqual(metrics["pairs"], 4)  # (0,1) (1,2) (2,3) (3,4)
        self.assertEqual(metrics["pose_jumps_over_width"], 2)
        self.assertEqual(metrics["length_jumps_over_3pct"], 1)  # (2,3): 100 -> 108
        # A workspace of every other frame pairs its adjacent rows just the same; a wider gap breaks the pair.
        stepped = {**arrays, "frame_index": np.arange(n) * 2}
        self.assertEqual(algorithms.frame_step(stepped["frame_index"]), 2)
        with_step = region_metrics(stepped, [1, 2, 3], image_shape=(100, 200))
        self.assertEqual((with_step["pairs"], with_step["pose_jumps_over_width"], with_step["length_jumps_over_3pct"]), (4, 2, 1))
        gapped = {**arrays, "frame_index": np.array([0, 1, 2, 3, 10, 11])}
        self.assertEqual(region_metrics(gapped, [1, 2, 3], image_shape=(100, 200))["pairs"], 3)  # (3,4) is no pair
        self.assertEqual(metrics["orientation_flips"], 1)  # (3,4); the pose jump is orientation-blind
        self.assertIsNone(metrics["seconds"])
        for name in algorithms.METRIC_NAMES:
            self.assertIn(name, metrics)
        # Nothing fitted: no overlap numbers, no pairs.
        empty = region_metrics({**arrays, "fitted": np.zeros(n, dtype=bool)}, [1, 2])
        self.assertIsNone(empty["median_iou"])
        self.assertEqual((empty["pairs"], empty["frames_fitted"]), (0, 0))


class RegionRunTests(unittest.TestCase):
    """Region algorithms on the fitted synthetic workspace: rows 1..4 between the anchors 0 and 5."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.recording = root / "rec.h5"
        cls.masks = _write_recording(cls.recording)
        cls.root = root / "workspaces"
        cls.workspace = Workspace.create(cls.root, "synthetic", cls.recording, 0, FRAMES - 1)
        run_stage(cls.workspace, "segment", SEGMENT_PARAMS, device="cpu")
        run_stage(cls.workspace, "fit", FIT_PARAMS, device="cpu", job="jfit")
        # Without a prior the fit orients each frame by its taper, which the synthetic body's
        # symmetric profile leaves to noise: make every frame follow row 0 so the anchors agree.
        state = cls.workspace.load_state()
        for row in range(1, FRAMES):
            curves = state["centerline_xy"]
            same = np.linalg.norm(curves[row, 0] - curves[0, 0]) + np.linalg.norm(curves[row, -1] - curves[0, -1])
            swapped = np.linalg.norm(curves[row, 0] - curves[0, -1]) + np.linalg.norm(curves[row, -1] - curves[0, 0])
            if swapped < same:
                edits._reverse_row(state, row)
        cls.workspace.save_state(state)
        run_stage(cls.workspace, "ambiguity", {}, device="cpu")
        cls.pristine = root / "pristine"
        cls.pristine.mkdir()
        for name in PRISTINE:
            source = cls.workspace.path / name
            if source.exists():
                shutil.copyfile(source, cls.pristine / name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def _restore(self) -> None:
        for name in PRISTINE:
            target = self.workspace.path / name
            source = self.pristine / name
            if source.exists():
                shutil.copyfile(source, target)
            elif target.exists():
                target.unlink()
        shutil.rmtree(self.workspace.path / "edits", ignore_errors=True)
        shutil.rmtree(self.workspace.path / "fixes", ignore_errors=True)

    def _oriented_like_anchor(self, curve: np.ndarray, anchor: np.ndarray) -> bool:
        return pose_distance_px(curve, anchor, None) < pose_distance_px(curve[::-1], anchor, None)

    # ----- anchors

    def test_propose_anchors_takes_the_nearest_good_rows_outside(self) -> None:
        state = self.workspace.load_state()
        good = [r for r in range(FRAMES) if state["iou"][r] >= 0.9 and state["ambiguity_score"][r] == 0]
        before, after = propose_anchors(state, 1, 4)
        self.assertEqual(before, 0 if 0 in good else None)
        self.assertEqual(after, 5 if 5 in good else None)
        # An anchor candidate with a poor overlap or a flag is skipped.
        state["iou"][0] = 0.5
        state["ambiguity_score"][5] = 2
        self.assertEqual(propose_anchors(state, 1, 4), (None, None))
        self.assertEqual(propose_anchors(state, 0, FRAMES - 1), (None, None))

    # ----- context

    def test_build_context_validates_anchors_and_segments_missing_masks(self) -> None:
        ctx = build_context(self.workspace, 1, 4, 0, 5, device="cpu")
        self.assertEqual((ctx.first, ctx.last, ctx.anchor_before, ctx.anchor_after), (1, 4, 0, 5))
        self.assertEqual(sorted(ctx.masks), [0, 1, 2, 3, 4, 5])
        self.assertEqual(ctx.rows, [1, 2, 3, 4])
        self.assertEqual(ctx.image_shape, (HEIGHT, WIDTH))
        self.assertTrue(np.array_equal(ctx.masks[2], self.masks[2]))
        self.assertIsNone(ctx.prior)
        self.assertAlmostEqual(ctx.anchor_length(), float(np.exp(np.mean(np.log(ctx.state["body_length_px"][[0, 5]])))))
        for bad in ((1, 4, 2, 5), (1, 4, 0, 4), (1, 9, None, None), (1, 4, -1, None)):
            with self.assertRaises(ValueError):
                build_context(self.workspace, *bad, device="cpu")
        state = self.workspace.load_state()
        state["fitted"][5] = False
        self.workspace.save_state(state)
        try:
            with self.assertRaisesRegex(ValueError, "no fitted pose"):
                build_context(self.workspace, 1, 4, 0, 5, device="cpu")
        finally:
            self._restore()
        # A workspace without stored masks gets the region segmented from the recording.
        copy = Workspace.create(self.root, "no_masks", self.recording, 0, FRAMES - 1)
        try:
            for name in ("state.npz", "summary.json"):
                shutil.copyfile(self.workspace.path / name, copy.path / name)
            self.assertFalse(copy.has_masks())
            self.assertIsNone(read_summary(copy)["checkpoint"])  # the dark-pixel stand-in segments again
            ctx = build_context(copy, 2, 3, 1, 4, device="cpu")
            self.assertEqual(sorted(ctx.masks), [1, 2, 3, 4])
            self.assertTrue(np.array_equal(ctx.masks[3], self.masks[3]))
            # The segmented masks are kept in the workspace, so the next run reads them instead of segmenting again.
            self.assertEqual(copy.mask_rows().tolist(), [1, 2, 3, 4])
            self.assertTrue(np.array_equal(copy.effective_mask(3), self.masks[3]))
            again = build_context(copy, 2, 3, 1, 4, device="cpu", segment_missing=False)
            self.assertEqual(sorted(again.masks), [1, 2, 3, 4])
            # Rows never segmented stay missing without segmenting.
            self.assertEqual(sorted(build_context(copy, 0, 0, None, 1, device="cpu", segment_missing=False).masks), [1])
        finally:
            shutil.rmtree(copy.path)

    # ----- mirror

    def test_mirror_follows_the_anchors_orientation_without_fitting(self) -> None:
        # Reverse rows 1..4 by hand: the region's orientation now disagrees with both anchors.
        edits.flip_orientation(self.workspace, [1, 2, 3, 4])
        try:
            state = self.workspace.load_state()
            before = region_metrics(state, [1, 2, 3, 4], (HEIGHT, WIDTH))
            self.assertEqual(before["orientation_flips"], 2)
            progress: list[tuple[float, str]] = []
            candidate_set = run_algorithm(self.workspace, "mirror", 1, 4, {}, anchor_before=0, anchor_after=5, device="cpu", progress=lambda p, m: progress.append((p, m)))
            self.assertEqual(candidate_set.algorithm, "mirror")
            self.assertEqual(candidate_set.rows, [1, 2, 3, 4])
            self.assertEqual(candidate_set.frames, [1, 4])
            self.assertEqual(candidate_set.workspace, "synthetic")
            self.assertEqual(sorted(candidate_set.mask_revisions), ["0", "1", "2", "3", "4", "5"])
            self.assertEqual(progress[0][0], 0.0)
            for row in (1, 2, 3, 4):
                poses = candidate_set.candidates[row]
                self.assertEqual([p.source for p in poses], ["current", "mirrored"])
                np.testing.assert_array_equal(poses[0].centerline_xy, state["centerline_xy"][row])
                np.testing.assert_array_equal(poses[1].centerline_xy, state["centerline_xy"][row][::-1])
                np.testing.assert_allclose(decode_centerline(poses[1].latent), poses[1].centerline_xy, atol=1e-3)
                np.testing.assert_array_equal(poses[1].width_profile, poses[0].width_profile[::-1])
                self.assertEqual(poses[0].iou, poses[1].iou)
                self.assertTrue(np.isfinite(poses[0].energy))
            path = candidate_set.path_by_row
            self.assertEqual(sorted(path), [1, 2, 3, 4])
            anchor = state["centerline_xy"][0]
            for row in (1, 2, 3, 4):
                chosen = candidate_set.chosen(row)
                self.assertIsNotNone(chosen)
                self.assertTrue(self._oriented_like_anchor(chosen.centerline_xy, anchor), f"row {row}")
                self.assertEqual(path[row], (1, False))  # the mirrored candidate, as is
            after = candidate_set.metrics
            self.assertEqual(after["orientation_flips"], 0)
            self.assertAlmostEqual(after["median_iou"], before["median_iou"])
            self.assertEqual(candidate_set.metrics_before["orientation_flips"], 2)
            self.assertGreater(after["seconds"], 0.0)
            # Nothing was written: a run only returns its result.
            np.testing.assert_array_equal(self.workspace.load_state()["centerline_xy"], state["centerline_xy"])
            self.assertEqual(len(self.workspace.edits()), 1)  # the flip above
            # Keeping it writes the path into the state as one edit attributed to the algorithm.
            result = keep(self.workspace, candidate_set)
            self.assertEqual(result.kind, "set_pose")
            self.assertEqual(result.rows, [1, 2, 3, 4])
            state = self.workspace.load_state()
            for row in (1, 2, 3, 4):
                self.assertTrue(self._oriented_like_anchor(state["centerline_xy"][row], anchor))
            provenance = self.workspace.load_provenance()
            self.assertEqual(set(provenance["algorithm"][1:5].tolist()), {"mirror"})
            self.assertEqual(len(set(provenance["job"][1:5].tolist())), 1)
            self.assertTrue(str(provenance["job"][1]).startswith(fixes.FIX_JOB_PREFIX))
            self.assertEqual(region_metrics(state, [1, 2, 3, 4], (HEIGHT, WIDTH))["orientation_flips"], 0)
            # Undo puts the reversed poses back.
            edits.undo(self.workspace, result.edit_id)
            state = self.workspace.load_state()
            self.assertEqual(region_metrics(state, [1, 2, 3, 4], (HEIGHT, WIDTH))["orientation_flips"], 2)
        finally:
            self._restore()

    # ----- independent multi-start

    @slow
    def test_independent_multistart_keeps_every_start_and_round_trips(self) -> None:
        try:
            candidate_set = run_algorithm(self.workspace, "independent_multistart", 1, 4, {"preset": "fast"}, anchor_before=0, anchor_after=5, device="cpu")
            self.assertEqual(candidate_set.params["preset"], "fast")
            state = self.workspace.load_state()
            config = pipeline.workspace_setup(self.workspace).config
            for row in (1, 2, 3, 4):
                poses = candidate_set.candidates[row]
                # Skeleton plus the moment arcs, no prior: one orientation each.
                self.assertGreaterEqual(len(poses), 2)
                self.assertLessEqual(len(poses), 4)
                self.assertTrue(all(p.source == "independent" for p in poses))
                self.assertTrue(any(p.start == "skeleton_longest_path" for p in poses))
                for pose in poses:
                    self.assertEqual(pose.centerline_xy.shape, (config.n_points, 2))
                    self.assertEqual(pose.latent.shape, (config.coefficients + 4,))
                    self.assertEqual(pose.width_shape.shape, (config.width_coefficients,))
                    self.assertTrue(np.isfinite(pose.latent).all())
                    self.assertTrue(np.isfinite(pose.width_profile).all())
                    self.assertGreater(pose.iou, 0.5)
                    self.assertLessEqual(pose.soft_dice, pose.energy + 1e-9)
                    x0, x1, y0, y1 = pose.crop.tolist()
                    self.assertTrue(0 <= x0 < x1 <= WIDTH and 0 <= y0 < y1 <= HEIGHT)
                    self.assertEqual(pose.points_in_fov, config.n_points)
                # Ordered by energy within the source.
                energies = [p.energy for p in poses]
                self.assertEqual(energies, sorted(energies))
            self.assertEqual(sorted(candidate_set.path_by_row), [1, 2, 3, 4])
            anchor = state["centerline_xy"][0]
            for row in (1, 2, 3, 4):
                chosen = candidate_set.chosen(row)
                self.assertGreater(chosen.iou, 0.8, f"row {row}")
                self.assertTrue(self._oriented_like_anchor(chosen.centerline_xy, anchor), f"row {row}")
            self.assertGreater(candidate_set.metrics["median_iou"], 0.8)
            self.assertEqual(candidate_set.metrics["frames"], 4)
            self.assertEqual(candidate_set.metrics["orientation_flips"], 0)
            # Keeping it writes every path row and leaves the rows' stored candidates alone.
            result = keep(self.workspace, candidate_set)
            self.assertEqual(result.rows, [1, 2, 3, 4])
            after = self.workspace.load_state()
            for row in (1, 2, 3, 4):
                chosen = candidate_set.chosen(row)
                np.testing.assert_allclose(after["centerline_xy"][row], chosen.centerline_xy)
                np.testing.assert_allclose(after["latent"][row], chosen.latent)
                self.assertAlmostEqual(float(after["iou"][row]), chosen.iou)
                self.assertEqual(after["crop"][row].tolist(), chosen.crop.tolist())
                self.assertEqual(str(after["best_start"][row]), chosen.start)
            np.testing.assert_array_equal(after["centerline_xy"][0], state["centerline_xy"][0])
            provenance = self.workspace.load_provenance()
            self.assertEqual(provenance["algorithm"][1:5].tolist(), ["independent_multistart"] * 4)
            self.assertEqual(str(provenance["algorithm"][0]), "independent_fit")
            # A second Keep of the same result is refused: its preview is gone.
            with self.assertRaises(FileNotFoundError):
                fixes.keep(self.workspace, self.workspace.edits()[-1]["payload"]["fix"]["preview"])
        finally:
            self._restore()

    # ----- chains and the beam path

    @slow
    def test_beam_path_with_non_adjacent_anchors_and_the_chains(self) -> None:
        try:
            params = {"beam": 2, "preset": "fast", "anchor_diversity": True, "refit_independent": True}
            candidate_set = run_algorithm(self.workspace, "beam_path", 2, 3, params, anchor_before=0, anchor_after=5, device="cpu")
            self.assertEqual(candidate_set.params["beam"], 2)
            self.assertEqual(candidate_set.params["temporal_prior_weight"], 0.01)  # the defaults are filled in
            for row in (2, 3):
                sources = [p.source for p in candidate_set.candidates[row]]
                self.assertEqual(set(sources), {"independent", "forward", "backward"}, f"row {row}: {sources}")
                # Stored in the propagate stage's order: independent, then the forward and the backward beam.
                self.assertEqual(sources, sorted(sources, key=lambda s: pipeline.SOURCE_CODES[s]))
                self.assertLessEqual(sources.count("forward"), 2)
                self.assertTrue(all(p.start for p in candidate_set.candidates[row]))
            self.assertEqual(sorted(candidate_set.path_by_row), [2, 3])
            state = self.workspace.load_state()
            for row in (2, 3):
                chosen = candidate_set.chosen(row)
                self.assertGreater(chosen.iou, 0.8, f"row {row}")
                self.assertTrue(self._oriented_like_anchor(chosen.centerline_xy, state["centerline_xy"][0]))
            self.assertGreater(candidate_set.metrics["median_iou"], 0.8)
            self.assertEqual(candidate_set.metrics_before["frames"], 2)
            # One direction only: the backward chain from the anchor after the region.
            backward = run_algorithm(self.workspace, "chain_backward", 1, 4, {"beam": 1, "prediction_damping": 0.6}, anchor_after=5, device="cpu")
            for row in (1, 2, 3, 4):
                self.assertEqual([p.source for p in backward.candidates[row]], ["backward"])
                self.assertIn(backward.candidates[row][0].start, ("warm_backward", "predicted_backward"))
            self.assertEqual(sorted(backward.path_by_row), [1, 2, 3, 4])
            self.assertGreater(backward.metrics["median_iou"], 0.8)
            self.assertIsNone(backward.anchor_before)
            with self.assertRaisesRegex(ValueError, "anchor before"):
                run_algorithm(self.workspace, "chain_forward", 1, 4, {}, anchor_after=5, device="cpu")
            # Nothing was written to the state.
            np.testing.assert_array_equal(self.workspace.load_state()["centerline_xy"], state["centerline_xy"])
            self.assertEqual(set(self.workspace.load_provenance()["algorithm"].tolist()), {"independent_fit"})
        finally:
            self._restore()

    def test_slow_refit_refits_the_current_poses_with_the_anchor_length(self) -> None:
        candidate_set = run_algorithm(self.workspace, "slow_refit", 2, 3, {"preset": "fast", "length_sigma": 0.02}, anchor_before=1, anchor_after=4, device="cpu")
        for row in (2, 3):
            self.assertEqual([(p.source, p.start) for p in candidate_set.candidates[row]], [("independent", "slow_refit")])
            self.assertGreater(candidate_set.candidates[row][0].iou, 0.8)
        self.assertEqual(sorted(candidate_set.path_by_row), [2, 3])
        self.assertGreater(candidate_set.metrics["median_iou"], 0.8)

    def test_candidate_pose_mirrors_and_converts_to_a_pose_dict(self) -> None:
        state = self.workspace.load_state()
        config = pipeline.workspace_setup(self.workspace).config
        pose = CandidatePose.from_state(state, 2, config)
        self.assertEqual(pose.source, "current")
        self.assertAlmostEqual(pose.soft_dice, float(state["energy"][2]))
        self.assertGreaterEqual(pose.energy, pose.soft_dice)
        mirrored = pose.mirrored(config.coefficients, "mirrored")
        np.testing.assert_array_equal(mirrored.centerline_xy, pose.centerline_xy[::-1])
        np.testing.assert_array_equal(mirrored.width_shape, pose.width_shape[::-1])
        np.testing.assert_allclose(decode_centerline(mirrored.latent, config.coefficients), mirrored.centerline_xy, atol=1e-3)
        as_dict = mirrored.to_pose()
        self.assertEqual(set(as_dict), set(edits.POSE_FIELDS))
        self.assertEqual(as_dict["source"], 0)
        with self.assertRaisesRegex(ValueError, "no stored pose"):
            CandidatePose.from_state({**state, "fitted": np.zeros(FRAMES, dtype=bool)}, 2, config)


class RegionEvidenceTests(unittest.TestCase):
    """Region algorithms on a workspace fit with the body-field network (the stub of tests/test_pipeline.py that knows the bodies)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        recording = root / "rec.h5"
        _write_recording(recording)
        cls.predictors: list[_StubPredictor] = []
        cls.patch = mock.patch.object(pipeline, "field_predictor", side_effect=lambda checkpoint, frames, device: cls.predictors.append(_StubPredictor(frames)) or cls.predictors[-1])
        cls.patch.start()
        cls.workspace = Workspace.create(root / "workspaces", "fields", recording, 0, FRAMES - 1)
        run_stage(cls.workspace, "segment", SEGMENT_PARAMS, device="cpu")
        run_stage(cls.workspace, "fit", {**FIT_PARAMS, "body_net": "stub.ckpt"}, device="cpu")
        run_stage(cls.workspace, "ambiguity", {}, device="cpu")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.patch.stop()
        cls.directory.cleanup()

    def _heads_first(self, candidate_set: CandidateSet) -> None:
        for row in candidate_set.rows:
            chosen = candidate_set.chosen(row)
            self.assertLess(float(np.linalg.norm(chosen.centerline_xy[0] - _body_curve(row)[0])), 6.0)

    def test_context_holds_the_evidence_and_the_trace_starts(self) -> None:
        ctx = build_context(self.workspace, 1, 4, 0, 5, device="cpu")
        self.assertEqual(sorted(ctx.evidence), list(range(FRAMES)))
        self.assertEqual(sorted(ctx.network_starts), [1, 2, 3, 4])
        self.assertIs(ctx.fields_of([1, 2])[1], ctx.evidence[2])
        self.assertEqual([s.name for s in ctx.with_trace(2, [])], ["network_trace"])
        self.assertEqual(sorted(build_context(self.workspace, 1, 4, 0, 5, device="cpu", trace_starts=False).network_starts), [])

    def test_every_fit_is_scored_against_the_evidence_and_offered_the_trace(self) -> None:
        calls: list = []

        def recording(masks, starts, **kwargs):
            calls.append((kwargs.get("fields"), [[s.name for s in frame] for frame in starts]))
            return algorithms_fit_masks(masks, starts, **kwargs)

        algorithms_fit_masks = algorithms.fit_masks
        with mock.patch.object(algorithms, "fit_masks", side_effect=recording):
            candidate_set = run_algorithm(self.workspace, "independent_multistart", 1, 4, {"preset": "fast"}, anchor_before=0, anchor_after=5, device="cpu")
        self.assertTrue(calls)
        for fields, _ in calls:
            self.assertTrue(fields is not None and all(f is not None for f in fields))
        self.assertIn("network_trace", {name for _, names in calls for frame in names for name in frame})
        for row in candidate_set.rows:
            starts = {p.start for p in candidate_set.candidates[row]}
            self.assertIn("network_trace", starts)
            self.assertTrue(any(s.endswith("_reversed") for s in starts))  # both orientations without a prior
            for pose in candidate_set.candidates[row]:
                self.assertGreater(pose.field_energy, 0.0)
                # Whichever end a start converged on, the orientation with the true head first pays less evidence.
                head_first = np.linalg.norm(pose.centerline_xy[0] - _body_curve(row)[0]) < np.linalg.norm(pose.centerline_xy[-1] - _body_curve(row)[0])
                self.assertEqual(head_first, pose.field_energy < pose.mirror_field_energy)
        self._heads_first(candidate_set)
        self.assertEqual(candidate_set.metrics["orientation_flips"], 0)

    def test_a_mirror_pays_its_own_evidence(self) -> None:
        candidate_set = run_algorithm(self.workspace, "mirror", 1, 4, {}, anchor_before=0, anchor_after=5, device="cpu")
        for row in candidate_set.rows:
            current, mirrored = candidate_set.candidates[row]
            self.assertEqual(mirrored.field_energy, current.mirror_field_energy)
            self.assertGreater(mirrored.energy - current.energy, 0.0)
            self.assertAlmostEqual(mirrored.energy - current.energy, mirrored.field_energy - current.field_energy)
            # Mirroring the mirror gives the pose and its energy back.
            back = mirrored.mirrored(len(mirrored.latent) - 4)
            self.assertAlmostEqual(back.energy, current.energy)
            self.assertEqual((back.field_energy, back.mirror_field_energy), (current.field_energy, current.mirror_field_energy))
        self._heads_first(candidate_set)

    def test_chains_use_the_evidence_and_placed_poses_are_scored(self) -> None:
        propagated: list = []
        propagate = algorithms.propagate

        def recording_propagate(*args, **kwargs):
            propagated.append(kwargs)
            return propagate(*args, **kwargs)

        with mock.patch.object(algorithms, "propagate", side_effect=recording_propagate):
            beam = run_algorithm(self.workspace, "beam_path", 2, 3, {"beam": 1}, anchor_before=0, anchor_after=5, device="cpu")
        # The local view puts the anchors next to the region: local rows 0 (anchor 0), 1-2 (rows 2-3), 3 (anchor 5).
        self.assertEqual(sorted(propagated[0]["evidence"]), [0, 1, 2, 3])
        self.assertEqual(sorted(propagated[0]["network_starts"]), [1, 2])
        self.assertTrue(all(p.field_energy > 0 for row in beam.rows for p in beam.candidates[row]))
        self._heads_first(beam)
        # The smoother places poses rather than fitting them; they are scored like the fits.
        state = self.workspace.load_state()
        with mock.patch.object(algorithms, "motion_scales", return_value=MotionScales(2.0, np.full(16, 0.05), 20)), \
                mock.patch.object(algorithms, "calibrate_body", return_value=([0, 5], float(state["body_length_px"][0]), state["width_profile"][0])):
            smoothed = run_algorithm(self.workspace, "fixed_body_smoother", 1, 4, {}, anchor_before=0, anchor_after=5, device="cpu")
        for row in smoothed.rows:
            pose = smoothed.candidates[row][0]
            self.assertGreater(pose.field_energy, 0.0)
            self.assertGreater(pose.mirror_field_energy, pose.field_energy)
            self.assertAlmostEqual(pose.energy - pose.field_energy, pose.soft_dice + algorithms.prior_penalty(
                pipeline.workspace_setup(self.workspace).config, pose.body_length_px, pose.width_px, pose.width_shape), places=6)

if __name__ == "__main__":
    unittest.main()
