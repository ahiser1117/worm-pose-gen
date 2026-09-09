"""Manual edits on a workspace built from a synthetic run.

The run comes from ``tests/test_pose_viewer._write_run`` (six straight
poses, a three-frame propagation stretch at the end with two hypotheses on
the last frame); the fixture here adds the richer ``hypotheses_*`` arrays
for one candidate and leaves the other without them, so a pick exercises
both the stored fields and their reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

import fcntl

from worm_pose_gen import edits
from worm_pose_gen.ambiguity import compute_ambiguity
from worm_pose_gen.pipeline import WORKSPACE_LOCK_FILE, WorkspaceBusy
from worm_pose_gen.edits import (
    EDIT_KINDS,
    EditResult,
    accept_path,
    edit_of_row,
    edited_rows,
    flip_frame,
    flip_orientation,
    flip_segment,
    list_edits,
    pick_hypothesis,
    pose_from_hypothesis,
    segment_info,
    segment_of,
    set_pose,
    undo,
)
from worm_pose_gen.latent import decode_centerline, encode_centerline
from worm_pose_gen.mask_fit import default_width_template, taper_asymmetry
from worm_pose_gen.workspace import Workspace

from tests.test_pose_viewer import FRAMES, HEIGHT, WIDTH, _write_recording, _write_run


N_POINTS = 100
RICH_CROP = [10, 120, 20, 80]


def _write_fixture_run(path: Path, recording: Path, *, rich: bool) -> None:
    """The viewer's synthetic run with an asymmetric width profile, hypotheses on rows 1 and 5, richer arrays on row 5 index 0."""

    _write_run(path, recording)
    with np.load(path / "poses.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count, H = arrays["hypotheses_energy"].shape
    template = default_width_template(N_POINTS)
    # A tapered profile so a flip changes the taper sign.
    arrays["width_profile"] = np.tile(10.0 * template * np.linspace(1.2, 0.8, N_POINTS), (count, 1))
    arrays["taper_asymmetry"] = np.array([taper_asymmetry(p) for p in arrays["width_profile"]])
    curve = arrays["centerline_xy"][0]
    # Row 1 (outside the stretch): two candidates without richer fields.
    bent = curve.copy()
    bent[:, 1] += 6.0 * np.sin(np.linspace(0, np.pi, N_POINTS))
    arrays["hypotheses_centerline_xy"][1, 0] = curve + (2.0, 0.0)
    arrays["hypotheses_centerline_xy"][1, 1] = bent
    arrays["hypotheses_energy"][1, :2] = [0.05, 0.06]
    arrays["hypotheses_iou"][1, :2] = [0.96, 0.93]
    arrays["hypotheses_source"][1, :2] = ["independent", "backward"]
    arrays["hypotheses_start"][1, :2] = ["independent_refit", "predicted_backward"]
    arrays["hypotheses_beam"][1, :2] = [0, 1]
    arrays["hypotheses_count"][1] = 2
    # The refit candidate of the last row overlaps well, so picking it clears the low-overlap flag.
    arrays["hypotheses_iou"][-1, 0] = 0.95
    if rich:
        arrays["hypotheses_latent"] = np.full((count, H, 20), np.nan)
        arrays["hypotheses_width_px"] = np.full((count, H), np.nan)
        arrays["hypotheses_width_shape"] = np.full((count, H, 6), np.nan)
        arrays["hypotheses_width_profile"] = np.full((count, H, N_POINTS), np.nan)
        arrays["hypotheses_body_length_px"] = np.full((count, H), np.nan)
        arrays["hypotheses_points_in_fov"] = np.zeros((count, H), dtype=np.int64)
        arrays["hypotheses_crop"] = np.zeros((count, H, 4), dtype=np.int64)
        arrays["hypotheses_soft_dice"] = np.full((count, H), np.nan)
        rich_curve = arrays["hypotheses_centerline_xy"][-1, 0]
        arrays["hypotheses_latent"][-1, 0] = encode_centerline(rich_curve)
        arrays["hypotheses_width_px"][-1, 0] = 9.0
        arrays["hypotheses_width_shape"][-1, 0] = [0.1, 0.05, 0.0, 0.0, -0.05, -0.1]
        arrays["hypotheses_width_profile"][-1, 0] = 9.0 * template * np.linspace(0.9, 1.1, N_POINTS)
        arrays["hypotheses_body_length_px"][-1, 0] = 101.0
        arrays["hypotheses_points_in_fov"][-1, 0] = 98
        arrays["hypotheses_crop"][-1, 0] = RICH_CROP
        arrays["hypotheses_soft_dice"][-1, 0] = 0.07
    # Flags and scores consistent with the poses, as a pipeline run leaves them (the viewer fixture sets them by hand).
    arrays.update(compute_ambiguity(arrays, prior=None, image_shape=(HEIGHT, WIDTH)))
    arrays["score_independent"] = arrays["ambiguity_score"].copy()
    np.savez_compressed(path / "poses.npz", **arrays)


class EditFixture(unittest.TestCase):
    """A fresh workspace per test, imported from the synthetic run."""

    rich = True

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.recording = root / "rec.h5"
        _write_recording(self.recording)
        self.run = root / "runs" / "2026-09-06T10-00-00Z_demo"
        _write_fixture_run(self.run, self.recording, rich=self.rich)
        self.workspace = Workspace.import_run(root / "workspaces", self.run)
        self.original = self.workspace.load_arrays()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def state(self) -> dict[str, np.ndarray]:
        return self.workspace.load_state()

    def hypotheses(self) -> dict[str, np.ndarray]:
        return self.workspace.load_hypotheses()

    def provenance_of(self, row: int) -> tuple[str, str, float]:
        provenance = self.workspace.load_provenance()
        return str(provenance["algorithm"][row]), str(provenance["job"][row]), float(provenance["time"][row])


class SegmentTests(EditFixture):
    def test_segment_of_uses_the_summary_stretches_and_fitted_runs_between_them(self) -> None:
        # The fixture's stretch is the last three rows.
        self.assertEqual(segment_of(self.workspace, 4), (3, 5))
        self.assertEqual(segment_of(self.workspace, 3), (3, 5))
        self.assertEqual(segment_of(self.workspace, 1), (0, 2))
        info = segment_info(self.workspace, 5)
        self.assertEqual(info, {"frames": [3, 5], "rows": [3, 5], "in_stretch": True})
        self.assertEqual(segment_info(self.workspace, 0)["in_stretch"], False)
        with self.assertRaises(ValueError):
            segment_of(self.workspace, FRAMES)

    def test_segment_of_falls_back_to_source_runs_and_unfitted_gaps(self) -> None:
        summary_path = self.workspace.path / "imported_summary.json"
        summary = json.loads(summary_path.read_text())
        del summary["propagation"]
        summary_path.write_text(json.dumps(summary))
        # Only the last row has a non-zero source.
        self.assertEqual(segment_of(self.workspace, 5), (5, 5))
        self.assertEqual(segment_of(self.workspace, 2), (0, 4))
        state = self.state()
        state["fitted"][2] = False
        self.workspace.save_state(state)
        self.assertEqual(segment_of(self.workspace, 1), (0, 1))
        self.assertEqual(segment_of(self.workspace, 3), (3, 4))
        self.assertEqual(segment_of(self.workspace, 2), (2, 2))


class PoseFromHypothesisTests(EditFixture):
    def test_stored_fields_are_used_when_present(self) -> None:
        arrays = self.workspace.load_arrays()
        pose = pose_from_hypothesis(arrays, arrays, 5, 0, False)
        np.testing.assert_array_equal(pose["centerline_xy"], arrays["hypotheses_centerline_xy"][5, 0])
        np.testing.assert_array_equal(pose["latent"], arrays["hypotheses_latent"][5, 0])
        self.assertEqual(pose["width_px"], 9.0)
        np.testing.assert_array_equal(pose["width_shape"], arrays["hypotheses_width_shape"][5, 0])
        np.testing.assert_array_equal(pose["width_profile"], arrays["hypotheses_width_profile"][5, 0])
        self.assertEqual((pose["body_length_px"], pose["points_in_fov"]), (101.0, 98))
        self.assertEqual(pose["crop"].tolist(), RICH_CROP)
        self.assertEqual((pose["iou"], pose["energy"], pose["total_energy"]), (0.95, 0.07, 0.08))
        self.assertEqual((pose["source"], pose["best_start"]), (0, "independent_refit"))

    def test_missing_fields_are_reconstructed_from_the_centerline_and_the_row(self) -> None:
        arrays = self.workspace.load_arrays()
        pose = pose_from_hypothesis(arrays, arrays, 5, 1, False, image_shape=(HEIGHT, WIDTH))
        curve = arrays["hypotheses_centerline_xy"][5, 1]
        np.testing.assert_allclose(decode_centerline(pose["latent"]), curve, atol=1e-6)
        self.assertEqual(pose["width_px"], float(arrays["width_px"][5]))
        np.testing.assert_array_equal(pose["width_shape"], arrays["width_shape"][5])
        np.testing.assert_array_equal(pose["width_profile"], arrays["width_profile"][5])
        self.assertEqual(pose["crop"].tolist(), arrays["crop"][5].tolist())
        self.assertEqual(pose["points_in_fov"], N_POINTS)
        self.assertAlmostEqual(pose["body_length_px"], 100.0, places=6)
        # No stored overlap energy: the total energy stands in.
        self.assertEqual((pose["energy"], pose["total_energy"]), (0.09, 0.09))
        self.assertEqual((pose["source"], pose["best_start"]), (1, "predicted_forward"))

    def test_mirrored_reverses_curve_profile_and_shape_and_re_encodes(self) -> None:
        arrays = self.workspace.load_arrays()
        plain = pose_from_hypothesis(arrays, arrays, 5, 0, False)
        mirrored = pose_from_hypothesis(arrays, arrays, 5, 0, True)
        np.testing.assert_array_equal(mirrored["centerline_xy"], plain["centerline_xy"][::-1])
        np.testing.assert_array_equal(mirrored["width_profile"], plain["width_profile"][::-1])
        np.testing.assert_array_equal(mirrored["width_shape"], plain["width_shape"][::-1])
        np.testing.assert_allclose(decode_centerline(mirrored["latent"]), plain["centerline_xy"][::-1], atol=1e-6)
        self.assertEqual(mirrored["width_px"], plain["width_px"])
        self.assertEqual(mirrored["crop"].tolist(), plain["crop"].tolist())

    def test_bad_index_raises(self) -> None:
        arrays = self.workspace.load_arrays()
        with self.assertRaises(ValueError):
            pose_from_hypothesis(arrays, arrays, 5, 2, False)
        with self.assertRaises(ValueError):
            pose_from_hypothesis(arrays, arrays, 0, 0, False)


class PickTests(EditFixture):
    def test_pick_writes_the_hypothesis_provenance_snapshot_and_log(self) -> None:
        before = time.time()
        result = pick_hypothesis(self.workspace, 5, 0, note="the refit is right")
        self.assertIsInstance(result, EditResult)
        self.assertEqual((result.edit_id, result.kind, result.rows, result.undone), ("e000001", "pick_hypothesis", [5], None))
        state, hyps = self.state(), self.hypotheses()
        np.testing.assert_array_equal(state["centerline_xy"][5], self.original["hypotheses_centerline_xy"][5, 0])
        np.testing.assert_array_equal(state["latent"][5], self.original["hypotheses_latent"][5, 0])
        self.assertEqual(float(state["width_px"][5]), 9.0)
        self.assertEqual(state["crop"][5].tolist(), RICH_CROP)
        self.assertEqual((float(state["iou"][5]), float(state["energy"][5]), float(state["total_energy"][5])), (0.95, 0.07, 0.08))
        self.assertEqual(int(state["source"][5]), 0)
        self.assertEqual(str(state["best_start"][5]), "independent_refit")
        self.assertFalse(state["reversed"][5])
        self.assertTrue(np.isnan(state["orientation_gap"][5]))
        self.assertAlmostEqual(float(state["taper_asymmetry"][5]), taper_asymmetry(self.original["hypotheses_width_profile"][5, 0]))
        self.assertTrue(state["fitted"][5])
        self.assertEqual((int(hyps["path_index"][5]), bool(hyps["path_mirrored"][5])), (0, False))
        # Other rows untouched.
        np.testing.assert_array_equal(state["centerline_xy"][:5], self.original["centerline_xy"][:5])
        # Provenance: the manual pick, attributed to the edit, stamped now.
        algorithm, job, stamp = self.provenance_of(5)
        self.assertEqual((algorithm, job), ("manual:pick", "edit:e000001"))
        self.assertGreaterEqual(stamp, before - 1)
        self.assertEqual(self.provenance_of(4)[0], "independent_fit")
        # Ambiguity refreshed for the row: the better overlap clears the low-overlap flag and the score;
        # the independent score (what seeds the stretches) is not an edit's to change.
        self.assertTrue(self.original["flag_low_iou"][5])
        self.assertEqual(int(self.original["ambiguity_score"][5]), 1)
        self.assertFalse(state["flag_low_iou"][5])
        self.assertEqual(int(state["ambiguity_score"][5]), 0)
        self.assertEqual(int(state["score_independent"][5]), 1)
        self.assertAlmostEqual(float(state["pose_jump_px"][5]), 3.0, places=6)
        np.testing.assert_array_equal(state["ambiguity_score"][:5], self.original["ambiguity_score"][:5])
        # Snapshot file with the changed slices, referenced from the log.
        snapshot = self.workspace.path / "edits" / "e000001.npz"
        self.assertTrue(snapshot.exists())
        with np.load(snapshot, allow_pickle=False) as archive:
            saved = {name: archive[name] for name in archive.files}
        self.assertEqual(saved["rows"].tolist(), [4, 5])
        self.assertEqual(saved["edited_rows"].tolist(), [5])
        for key in ("state:centerline_xy", "state:iou", "state:width_px", "hypotheses:path_index", "provenance:algorithm", "provenance:time"):
            self.assertIn(key, saved)
        np.testing.assert_array_equal(saved["state:centerline_xy"][1], self.original["centerline_xy"][5])
        self.assertNotIn("state:frame_index", saved)
        log = self.workspace.edits()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["kind"], "pick_hypothesis")
        self.assertEqual(log[0]["payload"]["snapshot"], "edits/e000001.npz")
        self.assertEqual(log[0]["payload"]["note"], "the refit is right")
        self.assertEqual(log[0]["payload"]["choices"], [{"row": 5, "index": 0, "mirrored": False}])
        change = result.summary["changes"][0]
        self.assertEqual((change["row"], change["frame"]), (5, 5))
        self.assertEqual((change["before"]["iou"], change["after"]["iou"]), (0.85, 0.95))
        self.assertEqual((change["before"]["source"], change["after"]["source"]), (1, 0))
        self.assertEqual((change["before"]["algorithm"], change["after"]["algorithm"]), ("chain_forward", "manual:pick"))
        self.assertEqual((change["before"]["path_index"], change["after"]["path_index"]), (1, 0))
        # The workspace summary counts the edit and the JSON form has no NaN.
        self.assertEqual(self.workspace.summary()["edits"], 1)
        json.dumps(edits.json_safe(result), allow_nan=False)

    def test_pick_mirrored_reverses_the_hypothesis(self) -> None:
        result = pick_hypothesis(self.workspace, 5, 1, mirrored=True)
        state, hyps = self.state(), self.hypotheses()
        curve = self.original["hypotheses_centerline_xy"][5, 1]
        np.testing.assert_array_equal(state["centerline_xy"][5], curve[::-1])
        np.testing.assert_allclose(decode_centerline(state["latent"][5]), curve[::-1], atol=1e-6)
        np.testing.assert_array_equal(state["width_profile"][5], self.original["width_profile"][5][::-1])
        self.assertEqual((int(hyps["path_index"][5]), bool(hyps["path_mirrored"][5])), (1, True))
        self.assertFalse(state["reversed"][5])
        self.assertEqual(int(state["source"][5]), 1)
        self.assertEqual(result.summary["changes"][0]["after"]["path_mirrored"], True)

    def test_pick_outside_the_hypotheses_raises_and_writes_nothing(self) -> None:
        with self.assertRaises(ValueError):
            pick_hypothesis(self.workspace, 0, 0)
        with self.assertRaises(ValueError):
            pick_hypothesis(self.workspace, 5, 2)
        self.assertEqual(self.workspace.edits(), [])
        self.assertFalse((self.workspace.path / "edits").exists())


class PickWithoutRicherArraysTests(EditFixture):
    rich = False

    def test_pick_on_an_older_workspace_reconstructs_the_pose(self) -> None:
        self.assertNotIn("hypotheses_latent", self.original)
        pick_hypothesis(self.workspace, 1, 1)
        state = self.state()
        curve = self.original["hypotheses_centerline_xy"][1, 1]
        np.testing.assert_array_equal(state["centerline_xy"][1], curve)
        # The 16-coefficient latent reproduces the bent candidate to a fraction of a pixel.
        np.testing.assert_allclose(decode_centerline(state["latent"][1]), curve, atol=0.2)
        self.assertEqual(float(state["width_px"][1]), float(self.original["width_px"][1]))
        self.assertEqual(state["crop"][1].tolist(), self.original["crop"][1].tolist())
        self.assertEqual(int(state["points_in_fov"][1]), N_POINTS)
        self.assertEqual(int(state["source"][1]), 2)
        self.assertEqual((float(state["iou"][1]), float(state["total_energy"][1])), (0.93, 0.06))
        self.assertEqual(self.provenance_of(1)[:2], ("manual:pick", "edit:e000001"))
        # The bent candidate is a jump from its straight neighbours.
        self.assertGreater(float(state["pose_jump_px"][2]), 1.0)
        undo(self.workspace)
        np.testing.assert_array_equal(self.state()["centerline_xy"], self.original["centerline_xy"])

    def test_fallback_profile_follows_the_candidate_direction_not_the_rows(self) -> None:
        # Row 1's pose is stored the way propagation stores a mirrored path: the reverse of hypothesis 1.
        state = self.state()
        hyps = self.hypotheses()
        state["centerline_xy"][1] = self.original["hypotheses_centerline_xy"][1, 1][::-1]
        hyps["path_index"][1], hyps["path_mirrored"][1] = 1, True
        self.workspace.save_state(state)
        self.workspace.save_hypotheses(hyps)
        arrays = self.workspace.load_arrays()
        row_profile = arrays["width_profile"][1]
        # Picking hypothesis 1 as is (the row's curve reversed) turns the row's profile and shape round with it.
        plain = pose_from_hypothesis(arrays, arrays, 1, 1, False)
        np.testing.assert_array_equal(plain["centerline_xy"], self.original["hypotheses_centerline_xy"][1, 1])
        np.testing.assert_array_equal(plain["width_profile"], row_profile[::-1])
        np.testing.assert_array_equal(plain["width_shape"], arrays["width_shape"][1][::-1])
        # Mirrored, the pick reproduces the row: the profile stays the row's.
        mirrored = pose_from_hypothesis(arrays, arrays, 1, 1, True)
        np.testing.assert_array_equal(mirrored["centerline_xy"], state["centerline_xy"][1])
        np.testing.assert_array_equal(mirrored["width_profile"], row_profile)
        # Hypothesis 0 runs the same way as hypothesis 1, so the row is reversed relative to it too.
        other = pose_from_hypothesis(arrays, arrays, 1, 0, False)
        np.testing.assert_array_equal(other["width_profile"], row_profile[::-1])
        # A row stored the candidates' way (row 5 before any edit) keeps its profile as is.
        np.testing.assert_array_equal(pose_from_hypothesis(arrays, arrays, 5, 1, False)["width_profile"], arrays["width_profile"][5])
        self.assertEqual(np.sign(taper_asymmetry(plain["width_profile"])), -np.sign(taper_asymmetry(row_profile)))


class FlipTests(EditFixture):
    def test_flip_frame_reverses_the_row_and_toggles_reversed(self) -> None:
        result = flip_frame(self.workspace, 2)
        self.assertEqual((result.kind, result.rows), ("flip_orientation", [2]))
        state, hyps = self.state(), self.hypotheses()
        np.testing.assert_array_equal(state["centerline_xy"][2], self.original["centerline_xy"][2][::-1])
        np.testing.assert_array_equal(state["width_profile"][2], self.original["width_profile"][2][::-1])
        np.testing.assert_array_equal(state["width_shape"][2], self.original["width_shape"][2][::-1])
        np.testing.assert_allclose(decode_centerline(state["latent"][2]), self.original["centerline_xy"][2][::-1], atol=1e-6)
        self.assertTrue(state["reversed"][2])
        self.assertAlmostEqual(float(state["taper_asymmetry"][2]), -float(self.original["taper_asymmetry"][2]))
        self.assertTrue(np.isnan(state["orientation_gap"][2]))
        self.assertEqual(self.provenance_of(2)[:2], ("manual:flip", "edit:e000001"))
        # No path chose this row: its path flags stay.
        self.assertEqual(int(hyps["path_index"][2]), -1)
        self.assertFalse(hyps["path_mirrored"][2])
        # A second flip puts it back.
        flip_frame(self.workspace, 2)
        state = self.state()
        np.testing.assert_array_equal(state["centerline_xy"][2], self.original["centerline_xy"][2])
        self.assertFalse(state["reversed"][2])
        self.assertEqual(self.provenance_of(2)[1], "edit:e000002")

    def test_flip_of_a_path_row_toggles_path_mirrored(self) -> None:
        self.assertEqual(int(self.original["path_index"][5]), 1)
        flip_frame(self.workspace, 5)
        self.assertTrue(self.hypotheses()["path_mirrored"][5])
        flip_frame(self.workspace, 5)
        self.assertFalse(self.hypotheses()["path_mirrored"][5])

    def test_flip_segment_reverses_the_stretch_or_the_run(self) -> None:
        result = flip_segment(self.workspace, 4)
        self.assertEqual(result.rows, [3, 4, 5])
        state = self.state()
        self.assertEqual(state["reversed"].tolist(), [False, False, False, True, True, True])
        for row in (3, 4, 5):
            np.testing.assert_array_equal(state["centerline_xy"][row], self.original["centerline_xy"][row][::-1])
            self.assertEqual(self.provenance_of(row)[:2], ("manual:flip", "edit:e000001"))
        self.assertEqual(self.provenance_of(2)[0], "independent_fit")
        result = flip_segment(self.workspace, 0)
        self.assertEqual(result.rows, [0, 1, 2])
        self.assertTrue(self.state()["reversed"].all())
        self.assertEqual(len(self.workspace.edits()), 2)

    def test_flip_skips_unfitted_rows_and_needs_one_fitted(self) -> None:
        state = self.state()
        state["fitted"][1] = False
        self.workspace.save_state(state)
        result = flip_orientation(self.workspace, [0, 1])
        self.assertEqual(result.rows, [0])
        with self.assertRaises(ValueError):
            flip_orientation(self.workspace, [1])
        with self.assertRaises(ValueError):
            flip_orientation(self.workspace, [])


class AcceptPathAndSetPoseTests(EditFixture):
    def test_accept_path_is_one_edit_with_the_given_attribution(self) -> None:
        result = accept_path(self.workspace, [(5, 0, False), (1, 1, True)], algorithm="beam_path", job="j000042", note="region run")
        self.assertEqual((result.edit_id, result.kind, result.rows), ("e000001", "accept_path", [1, 5]))
        state, hyps = self.state(), self.hypotheses()
        np.testing.assert_array_equal(state["centerline_xy"][5], self.original["hypotheses_centerline_xy"][5, 0])
        np.testing.assert_array_equal(state["centerline_xy"][1], self.original["hypotheses_centerline_xy"][1, 1][::-1])
        self.assertEqual(hyps["path_index"][[1, 5]].tolist(), [1, 0])
        self.assertEqual(hyps["path_mirrored"][[1, 5]].tolist(), [True, False])
        for row in (1, 5):
            self.assertEqual(self.provenance_of(row)[:2], ("beam_path", "j000042"))
        self.assertEqual(len(self.workspace.edits()), 1)
        self.assertEqual(sorted(p.name for p in (self.workspace.path / "edits").iterdir()), ["e000001.npz"])
        self.assertEqual(len(result.summary["changes"]), 2)
        with self.assertRaises(ValueError):
            accept_path(self.workspace, [(5, 0, False), (5, 1, False)], algorithm="x", job="y")
        with self.assertRaises(ValueError):
            accept_path(self.workspace, [], algorithm="x", job="y")

    def test_set_pose_writes_an_explicit_pose(self) -> None:
        curve = self.original["centerline_xy"][3] + (0.0, 5.0)
        result = set_pose(self.workspace, 3, {"centerline_xy": curve, "iou": 0.91, "best_start": "region:slow"}, algorithm="slow_refit", job="j7")
        self.assertEqual((result.kind, result.rows), ("set_pose", [3]))
        state, hyps = self.state(), self.hypotheses()
        np.testing.assert_array_equal(state["centerline_xy"][3], curve)
        np.testing.assert_allclose(decode_centerline(state["latent"][3]), curve, atol=1e-6)
        self.assertEqual(float(state["width_px"][3]), float(self.original["width_px"][3]))
        self.assertEqual(float(state["iou"][3]), 0.91)
        self.assertTrue(np.isnan(state["energy"][3]))
        self.assertEqual(str(state["best_start"][3]), "region:slow")
        self.assertEqual(int(state["source"][3]), 0)
        self.assertEqual(int(hyps["path_index"][3]), -1)
        self.assertEqual(self.provenance_of(3)[:2], ("slow_refit", "j7"))
        with self.assertRaises(ValueError):
            set_pose(self.workspace, 3, {"iou": 0.5}, algorithm="a", job="b")
        with self.assertRaises(ValueError):
            set_pose(self.workspace, 3, {"centerline_xy": curve[:10]}, algorithm="a", job="b")


class UndoTests(EditFixture):
    def test_undo_restores_the_newest_edit_then_the_one_before(self) -> None:
        pick_hypothesis(self.workspace, 5, 0)
        after_pick = self.workspace.load_arrays()
        pick_provenance = self.workspace.load_provenance()
        flip_frame(self.workspace, 5)
        result = undo(self.workspace)
        self.assertEqual((result.edit_id, result.kind, result.rows, result.undone), ("e000003", "undo", [5], "e000002"))
        arrays = self.workspace.load_arrays()
        for key in ("centerline_xy", "latent", "width_profile", "width_shape", "reversed", "taper_asymmetry", "ambiguity_score", "path_mirrored"):
            np.testing.assert_array_equal(arrays[key], after_pick[key], err_msg=key)
        provenance = self.workspace.load_provenance()
        self.assertEqual((str(provenance["algorithm"][5]), str(provenance["job"][5])), ("manual:pick", "edit:e000001"))
        self.assertEqual(float(provenance["time"][5]), float(pick_provenance["time"][5]))
        listed = list_edits(self.workspace)
        self.assertEqual([e["id"] for e in listed], ["e000003", "e000002", "e000001"])
        self.assertEqual([e["kind"] for e in listed], ["undo", "flip_orientation", "pick_hypothesis"])
        self.assertEqual([e["undone"] for e in listed], [False, True, False])
        self.assertEqual([e["undoable"] for e in listed], [False, False, True])
        self.assertEqual(listed[0]["undoes"], "e000002")
        self.assertEqual((listed[2]["rows"], listed[2]["frames"]), (1, [5, 5]))
        self.assertIn("changes", listed[2]["summary"])
        # The pick is next: the state is back to the import.
        result = undo(self.workspace)
        self.assertEqual(result.undone, "e000001")
        arrays = self.workspace.load_arrays()
        for key in ("centerline_xy", "latent", "iou", "source", "path_index", "path_mirrored", "ambiguity_score", "flag_low_iou", "pose_jump_px"):
            np.testing.assert_array_equal(arrays[key], self.original[key], err_msg=key)
        provenance = self.workspace.load_provenance()
        self.assertEqual(str(provenance["algorithm"][5]), "chain_forward")
        self.assertTrue(str(provenance["job"][5]).startswith("import:"))
        self.assertFalse(edited_rows(self.workspace, FRAMES).any())
        self.assertIsNone(edit_of_row(self.workspace, 5))
        # Nothing left, an undone edit, and an undo itself all refuse.
        with self.assertRaises(ValueError):
            undo(self.workspace)
        with self.assertRaises(ValueError):
            undo(self.workspace, "e000001")
        with self.assertRaises(ValueError):
            undo(self.workspace, "e000003")
        with self.assertRaises(ValueError):
            undo(self.workspace, "e999999")
        self.assertEqual(len(self.workspace.edits()), 4)

    def test_undo_after_two_edits_of_one_row_restores_the_first_edits_provenance(self) -> None:
        # The second pick keeps the algorithm (manual:pick) and changes only job and time, so its
        # snapshot must still carry every provenance array for the undo to restore the triple.
        pick_hypothesis(self.workspace, 5, 0)
        first = self.provenance_of(5)
        pick_hypothesis(self.workspace, 5, 1)
        with np.load(self.workspace.path / "edits" / "e000002.npz") as archive:
            self.assertTrue({"provenance:algorithm", "provenance:job", "provenance:time"} <= set(archive.files))
        result = undo(self.workspace)
        self.assertEqual(result.undone, "e000002")
        self.assertEqual(self.provenance_of(5), first)
        np.testing.assert_array_equal(self.state()["centerline_xy"][5], self.original["hypotheses_centerline_xy"][5, 0])
        # The same for two flips of one frame (the second undoes the first's geometry but not its log).
        flip_frame(self.workspace, 2)
        flipped = self.provenance_of(2)
        flip_frame(self.workspace, 2)
        undo(self.workspace)
        self.assertEqual(self.provenance_of(2), flipped)
        np.testing.assert_array_equal(self.state()["centerline_xy"][2], self.original["centerline_xy"][2][::-1])
        # An older snapshot with only the changed provenance arrays restores too: drop one and undo.
        flip_frame(self.workspace, 0)
        flip_frame(self.workspace, 0)
        snapshot = self.workspace.path / "edits" / "e000008.npz"
        with np.load(snapshot) as archive:
            saved = {k: archive[k] for k in archive.files if k != "provenance:algorithm"}
        np.savez_compressed(snapshot, **saved)
        undo(self.workspace, "e000008")
        self.assertEqual(self.provenance_of(0)[:2], ("manual:flip", "edit:e000007"))

    def test_edits_do_not_wait_for_a_stage_holding_the_lock(self) -> None:
        edits.LOCK_TIMEOUT, saved = 0.2, edits.LOCK_TIMEOUT
        try:
            with open(self.workspace.path / WORKSPACE_LOCK_FILE, "w") as holder:
                fcntl.flock(holder, fcntl.LOCK_EX)
                started = time.monotonic()
                with self.assertRaises(WorkspaceBusy):
                    pick_hypothesis(self.workspace, 5, 0)
                with self.assertRaises(WorkspaceBusy):
                    flip_frame(self.workspace, 0)
                with self.assertRaises(WorkspaceBusy):
                    undo(self.workspace)
                self.assertLess(time.monotonic() - started, 5.0)
                fcntl.flock(holder, fcntl.LOCK_UN)
            self.assertEqual(self.workspace.edits(), [])
            pick_hypothesis(self.workspace, 5, 0)
        finally:
            edits.LOCK_TIMEOUT = saved

    def test_undo_by_id_and_row_helpers(self) -> None:
        flip_frame(self.workspace, 0)
        pick_hypothesis(self.workspace, 5, 0)
        self.assertEqual(edited_rows(self.workspace, FRAMES).tolist(), [True, False, False, False, False, True])
        self.assertEqual(edit_of_row(self.workspace, 0)["id"], "e000001")
        self.assertEqual(edit_of_row(self.workspace, 5)["id"], "e000002")
        result = undo(self.workspace, "e000001")
        self.assertEqual(result.undone, "e000001")
        state = self.state()
        np.testing.assert_array_equal(state["centerline_xy"][0], self.original["centerline_xy"][0])
        # The later pick survives an undo of the earlier, disjoint edit.
        np.testing.assert_array_equal(state["centerline_xy"][5], self.original["hypotheses_centerline_xy"][5, 0])
        self.assertEqual(edited_rows(self.workspace, FRAMES).tolist(), [False, False, False, False, False, True])
        self.assertIsNone(edit_of_row(self.workspace, 0))
        self.assertEqual([e["undoable"] for e in list_edits(self.workspace)], [False, True, False])


class ContractTests(unittest.TestCase):
    def test_edit_kinds(self) -> None:
        self.assertEqual(EDIT_KINDS, ("pick_hypothesis", "flip_orientation", "accept_path", "set_pose", "undo"))


if __name__ == "__main__":
    unittest.main()
