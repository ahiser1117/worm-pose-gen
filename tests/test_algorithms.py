"""The algorithm registry on a synthetic workspace (CPU, no network).

The six-frame recording of ``tests/test_pipeline.py`` is segmented and fit
once (``checkpoint=None``, the SMALL schedule); the tests then run region
algorithms on rows of it with the outer frames as anchors, check the
candidate sets, their paths and metrics, the npz round trip, the outcome
log, acceptance through ``edits.accept_path``, the job argv and the command
line entry point.  Tests that change the workspace put the pristine arrays
back afterwards.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from worm_pose_gen import algorithms, edits, pipeline
from worm_pose_gen.algorithms import (
    REGISTRY,
    CandidatePose,
    CandidateSet,
    Parameter,
    accept_candidates,
    build_context,
    delete_candidate_set,
    list_algorithms,
    list_candidate_sets,
    load_candidate_set,
    outcomes,
    propose_region,
    region_metrics,
    resolve_params,
    run_region,
)
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.pipeline import read_summary, region_command, run_stage, update_summary
from worm_pose_gen.propagation import pose_distance_px
from worm_pose_gen.workspace import Workspace

from tests.test_pipeline import FIT_PARAMS, FRAMES, HEIGHT, SEGMENT_PARAMS, WIDTH, _write_recording


PRISTINE = ("state.npz", "hypotheses.npz", "provenance.npz", "summary.json", "edits.jsonl")


class RegistryTests(unittest.TestCase):
    def test_registry_lists_the_contract_algorithms_with_parameter_dicts(self) -> None:
        listed = list_algorithms()
        self.assertEqual([a["id"] for a in listed], ["independent_multistart", "chain_forward", "chain_backward", "beam_path", "slow_refit", "mirror"])
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

    def test_region_command_round_trips_the_spec(self) -> None:
        spec = {"algorithm": "beam_path", "first": 3, "last": 9, "params": {"beam": 2}, "anchor_before": 2, "anchor_after": None, "id": "j0000000a"}
        command = region_command(Path("/tmp/ws"), spec)
        self.assertEqual(command[:3], [".venv/bin/python", "-m", "worm_pose_gen.pipeline"])
        self.assertEqual(command[command.index("--workspace") + 1], "/tmp/ws")
        decoded = json.loads(command[command.index("--region-run") + 1])
        self.assertEqual(decoded, {"algorithm": "beam_path", "first": 3, "last": 9, "params": {"beam": 2}, "anchor_before": 2, "id": "j0000000a"})
        self.assertNotIn("anchor_after", decoded)  # None anchors are simply absent
        # A JobSpec-like object whose params is the dictionary works too; a spec without an algorithm does not.
        from worm_pose_gen.jobs import JobSpec

        self.assertEqual(region_command("/tmp/ws", JobSpec(kind="region", params=spec)), command)
        with self.assertRaisesRegex(ValueError, "algorithm"):
            region_command("/tmp/ws", {"first": 1, "last": 2})
        with self.assertRaisesRegex(ValueError, "first and last"):
            region_command("/tmp/ws", {"algorithm": "mirror", "first": 1})


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

    def _oriented_like_anchor(self, curve: np.ndarray, anchor: np.ndarray) -> bool:
        return pose_distance_px(curve, anchor, None) < pose_distance_px(curve[::-1], anchor, None)

    # ----- proposing

    def test_propose_region_uses_the_stretch_and_finds_good_anchors(self) -> None:
        state = self.workspace.load_state()
        self.assertTrue(state["fitted"].all())
        n = FRAMES
        # Without a stretch: ten frames either side, clipped to the workspace.
        proposal = propose_region(self.workspace, 3)
        self.assertEqual((proposal["first"], proposal["last"]), (0, n - 1))
        self.assertIsNone(proposal["anchor_before"])
        self.assertIsNone(proposal["anchor_after"])
        self.assertIn("no propagation stretch", proposal["reason"])
        # With a stretch in the summary: padded by ``pad`` and anchored on the nearest good rows outside.
        update_summary(self.workspace, {"propagation": {"stretches": [[2, 3]]}})
        try:
            proposal = propose_region(self.workspace, 3, pad=1)
            self.assertEqual((proposal["first"], proposal["last"]), (1, 4))
            self.assertEqual(proposal["stretch"], [2, 3])
            good = [r for r in range(n) if state["iou"][r] >= 0.9 and state["ambiguity_score"][r] == 0]
            self.assertEqual(proposal["anchor_before"], 0 if 0 in good else None)
            self.assertEqual(proposal["anchor_after"], 5 if 5 in good else None)
            self.assertEqual(propose_region(self.workspace, 2, pad=0)["first"], 2)
            # An anchor candidate with a poor overlap or a flag is skipped.
            state["iou"][0] = 0.5
            state["ambiguity_score"][5] = 2
            self.workspace.save_state(state)
            proposal = propose_region(self.workspace, 3, pad=1)
            self.assertIsNone(proposal["anchor_before"])
            self.assertIsNone(proposal["anchor_after"])
            # Row runs of non-zero source stand in for the stretches when the summary has none.
            update_summary(self.workspace, {"propagation": {}})
            state["source"][1:3] = 1
            self.workspace.save_state(state)
            self.assertEqual(propose_region(self.workspace, 2, pad=0)["stretch"], [1, 2])
            with self.assertRaisesRegex(ValueError, "outside"):
                propose_region(self.workspace, n)
        finally:
            self._restore()

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
        # A workspace without stored masks (an imported run) gets the region segmented from the recording.
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
            candidate_set = run_region(self.workspace, "mirror", 1, 4, {}, anchor_before=0, anchor_after=5, device="cpu", progress=lambda p, m: progress.append((p, m)))
            self.assertEqual(candidate_set.algorithm, "mirror")
            self.assertEqual(candidate_set.id, "c000001")
            self.assertEqual(candidate_set.rows, [1, 2, 3, 4])
            self.assertEqual(candidate_set.frames, [1, 4])
            self.assertEqual(progress[0][0], 0.0)
            self.assertEqual(progress[-1][0], 1.0)
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
            # Stored under candidates/ with a json sidecar, listed, and logged.
            self.assertTrue((self.workspace.path / "candidates" / "c000001.npz").exists())
            self.assertTrue((self.workspace.path / "candidates" / "c000001.json").exists())
            listed = list_candidate_sets(self.workspace)
            self.assertEqual([e["id"] for e in listed], ["c000001"])
            self.assertEqual(listed[0]["algorithm"], "mirror")
            self.assertEqual(listed[0]["frames"], [1, 4])
            self.assertFalse(listed[0]["accepted"])
            logged = outcomes(self.root)
            self.assertEqual(logged[0]["candidate_set"], "c000001")
            self.assertEqual(logged[0]["algorithm"], "mirror")
            self.assertEqual(logged[0]["anchors"], {"before": 0, "after": 5})
            self.assertEqual(logged[0]["metrics_before"]["orientation_flips"], 2)
            self.assertEqual(logged[0]["metrics_after"]["orientation_flips"], 0)
            self.assertFalse(logged[0]["accepted"])
            # Accepting writes the path into the state as one edit attributed to the algorithm.
            hyps_before = self.workspace.load_hypotheses()
            result = accept_candidates(self.workspace, "c000001")
            self.assertEqual(result.kind, "accept_path")
            self.assertEqual(result.rows, [1, 2, 3, 4])
            state = self.workspace.load_state()
            for row in (1, 2, 3, 4):
                self.assertTrue(self._oriented_like_anchor(state["centerline_xy"][row], anchor))
            provenance = self.workspace.load_provenance()
            self.assertEqual(set(provenance["algorithm"][1:5].tolist()), {"mirror"})
            self.assertEqual(set(provenance["job"][1:5].tolist()), {"candidates:c000001"})
            self.assertEqual(region_metrics(state, [1, 2, 3, 4], (HEIGHT, WIDTH))["orientation_flips"], 0)
            self.assertTrue(outcomes(self.root)[0]["accepted"])
            self.assertEqual(outcomes(self.root)[0]["accepted_edit"], result.edit_id)
            reloaded = load_candidate_set(self.workspace, "c000001")
            self.assertTrue(reloaded.accepted)
            self.assertEqual(reloaded.accepted_edit, result.edit_id)
            self.assertTrue(list_candidate_sets(self.workspace)[0]["accepted"])
            self.assertEqual(algorithms.candidate_sets_covering(self.workspace, 2), [])
            self.assertEqual([s.id for s in algorithms.candidate_sets_covering(self.workspace, 2, include_accepted=True)], ["c000001"])
            # The hypotheses table now holds the set's candidates and points at the accepted one.
            hyps = self.workspace.load_hypotheses()
            self.assertEqual(hyps["hypotheses_count"][1:5].tolist(), [2, 2, 2, 2])
            self.assertEqual(hyps["path_index"][1:5].tolist(), [1, 1, 1, 1])
            self.assertEqual(str(hyps["hypotheses_source"][2, 1]), "mirrored")
            self.assertEqual(reloaded.accepted_rows, [1, 2, 3, 4])
            # Accepting again (a double click) makes no second edit.
            with self.assertRaisesRegex(ValueError, "already accepted"):
                accept_candidates(self.workspace, "c000001")
            self.assertEqual(len(self.workspace.edits()), 2)  # the flip above and the accept
            # Undo puts the reversed poses back, and the rows' previous hypotheses, and un-marks the set.
            undone = edits.undo(self.workspace, result.edit_id)
            state = self.workspace.load_state()
            self.assertEqual(region_metrics(state, [1, 2, 3, 4], (HEIGHT, WIDTH))["orientation_flips"], 2)
            hyps = self.workspace.load_hypotheses()
            # The fit stage stored no hypotheses, so the rows are empty again (the table shows nothing rather than the set's candidates).
            self.assertEqual(hyps_before, {})
            self.assertEqual(hyps["hypotheses_count"][1:5].tolist(), [0, 0, 0, 0])
            self.assertEqual(hyps["path_index"][1:5].tolist(), [-1, -1, -1, -1])
            self.assertEqual(hyps["hypotheses_source"][1:5, :2].tolist(), [["", ""]] * 4)
            self.assertTrue(np.isnan(hyps["hypotheses_centerline_xy"][1:5]).all())
            reloaded = load_candidate_set(self.workspace, "c000001")
            self.assertEqual((reloaded.accepted, reloaded.accepted_rows, reloaded.accepted_edit), (False, [], None))
            self.assertFalse(list_candidate_sets(self.workspace)[0]["accepted"])
            self.assertEqual([s.id for s in algorithms.candidate_sets_covering(self.workspace, 2)], ["c000001"])
            logged = outcomes(self.root)[0]
            self.assertEqual((logged["accepted"], logged["accepted_rows"], logged["accepted_edit"]), (False, [], None))
            lines = [json.loads(l) for l in algorithms.outcomes_path(self.root).read_text().splitlines()]
            self.assertEqual([l["kind"] for l in lines], ["run", "accepted", "unaccepted"])
            self.assertEqual((lines[2]["undoes"], lines[2]["edit"], lines[2]["rows"]), (result.edit_id, undone.edit_id, [1, 2, 3, 4]))
            # The set can be accepted anew.
            again = accept_candidates(self.workspace, "c000001")
            self.assertEqual(again.rows, [1, 2, 3, 4])
            self.assertTrue(load_candidate_set(self.workspace, "c000001").accepted)
            self.assertTrue(outcomes(self.root)[0]["accepted"])
            edits.undo(self.workspace, again.edit_id)
            with self.assertRaisesRegex(ValueError, "path"):
                accept_candidates(self.workspace, "c000001", use_path=False)
            with self.assertRaisesRegex(ValueError, "none of the requested rows"):
                accept_candidates(self.workspace, "c000001", rows=[0])
            self.assertTrue(delete_candidate_set(self.workspace, "c000001"))
            self.assertFalse(delete_candidate_set(self.workspace, "c000001"))
            self.assertEqual(list_candidate_sets(self.workspace), [])
            with self.assertRaises(FileNotFoundError):
                load_candidate_set(self.workspace, "c000001")
        finally:
            self._restore()
            shutil.rmtree(self.workspace.path / "candidates", ignore_errors=True)
            algorithms.outcomes_path(self.root).unlink(missing_ok=True)

    # ----- independent multi-start

    def test_independent_multistart_keeps_every_start_and_round_trips(self) -> None:
        try:
            candidate_set = run_region(self.workspace, "independent_multistart", 1, 4, {"preset": "fast"}, anchor_before=0, anchor_after=5, device="cpu", job="j0000000a")
            self.assertEqual(candidate_set.id, "j0000000a")
            self.assertEqual(candidate_set.job, "j0000000a")
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
            # npz + json round trip.
            reloaded = CandidateSet.from_npz(self.workspace.path / "candidates" / "j0000000a.npz")
            self.assertEqual(reloaded.id, "j0000000a")
            self.assertEqual((reloaded.first, reloaded.last, reloaded.anchor_before, reloaded.anchor_after), (1, 4, 0, 5))
            self.assertEqual(reloaded.path, candidate_set.path)
            self.assertEqual(reloaded.metrics, json.loads(json.dumps(algorithms._json_safe(candidate_set.metrics))))
            self.assertEqual(reloaded.rows, [1, 2, 3, 4])
            self.assertEqual(reloaded.frames, [1, 4])
            self.assertEqual(reloaded.workspace, "synthetic")
            for row in (1, 2, 3, 4):
                self.assertEqual(len(reloaded.candidates[row]), len(candidate_set.candidates[row]))
                for a, b in zip(reloaded.candidates[row], candidate_set.candidates[row], strict=True):
                    np.testing.assert_array_equal(a.centerline_xy, b.centerline_xy)
                    np.testing.assert_array_equal(a.latent, b.latent)
                    np.testing.assert_array_equal(a.width_profile, b.width_profile)
                    np.testing.assert_array_equal(a.width_shape, b.width_shape)
                    np.testing.assert_array_equal(a.crop, b.crop)
                    self.assertEqual((a.width_px, a.body_length_px, a.points_in_fov, a.energy, a.soft_dice, a.iou, a.source, a.start),
                                     (b.width_px, b.body_length_px, b.points_in_fov, b.energy, b.soft_dice, b.iou, b.source, b.start))
            self.assertEqual(reloaded.summary()["candidates"], sum(len(v) for v in candidate_set.candidates.values()))
            # Accepting a subset of rows writes only those and grows the hypotheses to hold every candidate.
            result = accept_candidates(self.workspace, "j0000000a", rows=[2, 3])
            self.assertEqual(result.rows, [2, 3])
            after = self.workspace.load_state()
            for row in (2, 3):
                chosen = candidate_set.chosen(row)
                np.testing.assert_allclose(after["centerline_xy"][row], chosen.centerline_xy)
                np.testing.assert_allclose(after["latent"][row], chosen.latent)
                self.assertAlmostEqual(float(after["iou"][row]), chosen.iou)
                self.assertEqual(after["crop"][row].tolist(), chosen.crop.tolist())
                self.assertEqual(str(after["best_start"][row]), chosen.start)
            np.testing.assert_array_equal(after["centerline_xy"][1], state["centerline_xy"][1])
            provenance = self.workspace.load_provenance()
            self.assertEqual(provenance["algorithm"][2:4].tolist(), ["independent_multistart"] * 2)
            self.assertEqual(provenance["job"][2:4].tolist(), ["candidates:j0000000a"] * 2)
            self.assertEqual(str(provenance["algorithm"][1]), "independent_fit")
            hyps = self.workspace.load_hypotheses()
            self.assertGreaterEqual(hyps["hypotheses_energy"].shape[1], max(len(candidate_set.candidates[r]) for r in (2, 3)))
            self.assertEqual(hyps["hypotheses_count"][2:4].tolist(), [len(candidate_set.candidates[2]), len(candidate_set.candidates[3])])
            self.assertEqual(hyps["path_index"][2:4].tolist(), [candidate_set.path_by_row[2][0], candidate_set.path_by_row[3][0]])
            self.assertEqual(hyps["hypotheses_count"][1], 0)
            # Two of four path rows: the set is not accepted as a whole, the rows taken are recorded, the set still
            # overlays the rows it has not been accepted on, and the same rows cannot be accepted twice.
            logged = outcomes(self.root)[0]
            self.assertEqual((logged["candidate_set"], logged["accepted"], logged["accepted_rows"]), ("j0000000a", False, [2, 3]))
            partial = load_candidate_set(self.workspace, "j0000000a")
            self.assertEqual((partial.accepted, partial.accepted_rows, partial.accepted_edit), (False, [2, 3], result.edit_id))
            self.assertEqual([s.id for s in algorithms.candidate_sets_covering(self.workspace, 2)], [])
            self.assertEqual([s.id for s in algorithms.candidate_sets_covering(self.workspace, 1)], ["j0000000a"])
            with self.assertRaisesRegex(ValueError, "already accepted"):
                accept_candidates(self.workspace, "j0000000a", rows=[2])
            # Accepting the rest completes the set (only the rows still open are written).
            rest = accept_candidates(self.workspace, "j0000000a")
            self.assertEqual(rest.rows, [1, 4])
            complete = load_candidate_set(self.workspace, "j0000000a")
            self.assertEqual((complete.accepted, complete.accepted_rows), (True, [1, 2, 3, 4]))
            self.assertTrue(outcomes(self.root)[0]["accepted"])
            self.assertEqual(outcomes(self.root)[0]["accepted_rows"], [1, 2, 3, 4])
            # Undoing the second accept leaves the first's rows accepted.
            edits.undo(self.workspace, rest.edit_id)
            reopened = load_candidate_set(self.workspace, "j0000000a")
            self.assertEqual((reopened.accepted, reopened.accepted_rows), (False, [2, 3]))
            self.assertEqual(outcomes(self.root)[0]["accepted_rows"], [2, 3])
            np.testing.assert_array_equal(self.workspace.load_state()["centerline_xy"][1], state["centerline_xy"][1])
            hyps = self.workspace.load_hypotheses()
            self.assertEqual(hyps["hypotheses_count"][1], 0)
            self.assertEqual(hyps["hypotheses_count"][2], len(candidate_set.candidates[2]))
        finally:
            self._restore()
            shutil.rmtree(self.workspace.path / "candidates", ignore_errors=True)
            algorithms.outcomes_path(self.root).unlink(missing_ok=True)

    # ----- chains and the beam path

    def test_beam_path_with_non_adjacent_anchors_and_the_chains(self) -> None:
        try:
            params = {"beam": 2, "preset": "fast", "anchor_diversity": True, "refit_independent": True}
            candidate_set = run_region(self.workspace, "beam_path", 2, 3, params, anchor_before=0, anchor_after=5, device="cpu")
            self.assertEqual(candidate_set.id, "c000001")
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
            backward = run_region(self.workspace, "chain_backward", 1, 4, {"beam": 1, "prediction_damping": 0.6}, anchor_after=5, device="cpu")
            self.assertEqual(backward.id, "c000002")
            for row in (1, 2, 3, 4):
                self.assertEqual([p.source for p in backward.candidates[row]], ["backward"])
                self.assertIn(backward.candidates[row][0].start, ("warm_backward", "predicted_backward"))
            self.assertEqual(sorted(backward.path_by_row), [1, 2, 3, 4])
            self.assertGreater(backward.metrics["median_iou"], 0.8)
            self.assertIsNone(backward.anchor_before)
            with self.assertRaisesRegex(ValueError, "anchor before"):
                run_region(self.workspace, "chain_forward", 1, 4, {}, anchor_after=5, device="cpu")
            # The outcome log holds both runs, newest first, and nothing was accepted or written to the state.
            logged = outcomes(self.root)
            self.assertEqual([o["candidate_set"] for o in logged], ["c000002", "c000001"])
            self.assertEqual([o["algorithm"] for o in logged], ["chain_backward", "beam_path"])
            self.assertFalse(any(o["accepted"] for o in logged))
            np.testing.assert_array_equal(self.workspace.load_state()["centerline_xy"], state["centerline_xy"])
            self.assertEqual(set(self.workspace.load_provenance()["algorithm"].tolist()), {"independent_fit"})
            self.assertEqual([e["id"] for e in list_candidate_sets(self.workspace)], ["c000002", "c000001"])
        finally:
            self._restore()
            shutil.rmtree(self.workspace.path / "candidates", ignore_errors=True)
            algorithms.outcomes_path(self.root).unlink(missing_ok=True)

    def test_slow_refit_refits_the_current_poses_with_the_anchor_length(self) -> None:
        try:
            candidate_set = run_region(self.workspace, "slow_refit", 2, 3, {"preset": "fast", "length_sigma": 0.02}, anchor_before=1, anchor_after=4, device="cpu")
            for row in (2, 3):
                self.assertEqual([(p.source, p.start) for p in candidate_set.candidates[row]], [("independent", "slow_refit")])
                self.assertGreater(candidate_set.candidates[row][0].iou, 0.8)
            self.assertEqual(sorted(candidate_set.path_by_row), [2, 3])
            self.assertGreater(candidate_set.metrics["median_iou"], 0.8)
        finally:
            shutil.rmtree(self.workspace.path / "candidates", ignore_errors=True)
            algorithms.outcomes_path(self.root).unlink(missing_ok=True)

    # ----- the command line

    def test_cli_runs_a_region_spec_and_reports_progress(self) -> None:
        progress_file = self.workspace.path / "progress.json"
        spec = {"algorithm": "mirror", "first": 1, "last": 4, "params": {}, "anchor_before": 0, "anchor_after": 5, "id": "j00000042"}
        command = region_command(self.workspace.path, spec)
        previous = os.environ.get("WORM_POSE_PROGRESS_FILE")
        os.environ["WORM_POSE_PROGRESS_FILE"] = str(progress_file)
        try:
            code = pipeline.main(command[command.index("--workspace") :])
            self.assertEqual(code, 0)
            report = json.loads(progress_file.read_text())
            self.assertEqual(report["progress"], 1.0)
            result = report["result"]
            self.assertEqual(result["candidate_set"], "j00000042")
            self.assertEqual(result["algorithm"], "mirror")
            self.assertEqual(result["rows"], [1, 4])
            self.assertEqual(result["frames"], [1, 4])
            self.assertEqual(result["anchors"], {"before": 0, "after": 5})
            self.assertEqual(result["path_rows"], 4)
            self.assertIn("median_iou", result["metrics"])
            self.assertIn("median_iou", result["metrics_before"])
            self.assertTrue((self.workspace.path / "candidates" / "j00000042.npz").exists())
            self.assertEqual(load_candidate_set(self.workspace, "j00000042").job, "j00000042")
            self.assertEqual(outcomes(self.root)[0]["job"], "j00000042")
            # Without an id in the spec the job id from the environment names the set.
            os.environ["WORM_POSE_JOB_ID"] = "j00000043"
            try:
                pipeline.main(["--workspace", str(self.workspace.path), "--region-run", json.dumps({"algorithm": "mirror", "first": 2, "last": 3, "anchor_before": 1, "anchor_after": 4})])
            finally:
                del os.environ["WORM_POSE_JOB_ID"]
            self.assertTrue((self.workspace.path / "candidates" / "j00000043.json").exists())
            with self.assertRaises(SystemExit):
                pipeline.main(["--workspace", str(self.workspace.path)])
        finally:
            if previous is None:
                del os.environ["WORM_POSE_PROGRESS_FILE"]
            else:
                os.environ["WORM_POSE_PROGRESS_FILE"] = previous
            progress_file.unlink(missing_ok=True)
            shutil.rmtree(self.workspace.path / "candidates", ignore_errors=True)
            algorithms.outcomes_path(self.root).unlink(missing_ok=True)

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


if __name__ == "__main__":
    unittest.main()
