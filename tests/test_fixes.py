"""Issues and fixes (``worm_pose_gen.fixes``) on synthetic arrays and a synthetic workspace (CPU, no network).

The workspace is the bending body of ``tests/test_pipeline.py`` over ten
frames, segmented by the dark-pixel stand-in and given its true midlines as
poses (no fit stage: the fit is what takes minutes on a CPU), with the
small fit schedule of the pipeline tests as its fit configuration.  The
refit tests use ``mirror`` (no fitting); the stitch test fits two short
gaps.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from worm_pose_gen import edits, fixes, pipeline
from worm_pose_gen.algorithms import Keyframe
from worm_pose_gen.ambiguity import FLAG_NAMES
from worm_pose_gen.latent import encode_centerline
from worm_pose_gen.mask_fit import default_width_template
from worm_pose_gen.pipeline import FitParams, build_fit_config, run_stage, update_summary, workspace_arrays
from worm_pose_gen.propagation import pose_distance_px
from worm_pose_gen.workspace import Workspace

from tests.test_pipeline import FIT_PARAMS, HEIGHT, SEGMENT_PARAMS, WIDTH, _body_curve, _write_recording


N = 10
PRISTINE = ("state.npz", "hypotheses.npz", "provenance.npz", "summary.json", "edits.jsonl", "human_review.json")


def posed_workspace(root: Path, name: str = "synthetic", n: int = N) -> Workspace:
    """A segmented synthetic workspace whose poses are the bodies' true midlines, with the small fit schedule as its configuration."""

    recording = root / f"{name}.h5"
    _write_recording(recording, n)
    workspace = Workspace.create(root / "workspaces", name, recording, 0, n - 1)
    run_stage(workspace, "segment", SEGMENT_PARAMS, device="cpu")
    config = build_fit_config(FitParams.from_dict(FIT_PARAMS))
    update_summary(workspace, {"fit_config": asdict(config)})
    arrays = workspace_arrays(workspace, config)
    profile = 12.0 * default_width_template(config.n_points)
    for row in range(n):
        curve = _body_curve(row)
        edits._write_pose(arrays, row, {
            "centerline_xy": curve, "latent": encode_centerline(curve, config.coefficients), "width_px": 12.0,
            "width_shape": np.zeros(config.width_coefficients), "width_profile": profile,
            "body_length_px": float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()), "points_in_fov": config.n_points,
            "crop": np.array([0, WIDTH, 0, HEIGHT]), "iou": 0.97, "energy": 0.03, "total_energy": 0.03,
        })
    workspace.save_state(arrays)
    workspace.set_provenance(list(range(n)), "independent_fit", "jfit")
    run_stage(workspace, "ambiguity", {}, device="cpu")
    return workspace


def straight_state(n: int, points: int = 20) -> dict[str, np.ndarray]:
    """A state of ``n`` fitted frames of the same straight body, head at x = 0, no flags."""

    curve = np.stack((np.linspace(0, 100, points), np.full(points, 50.0)), axis=1)
    state: dict[str, np.ndarray] = {
        "frame_index": np.arange(n), "fitted": np.ones(n, dtype=bool), "centerline_xy": np.repeat(curve[None], n, axis=0),
        "iou": np.full(n, 0.95), "mask_stale": np.zeros(n, dtype=bool),
    }
    for flag in FLAG_NAMES:
        state[f"flag_{flag}"] = np.zeros(n, dtype=bool)
    return state


class ReasonTests(unittest.TestCase):
    def test_every_flag_has_a_reason_and_every_reason_plain_words(self) -> None:
        self.assertEqual(set(fixes.FLAG_REASONS), set(FLAG_NAMES))
        self.assertTrue(set(fixes.FLAG_REASONS.values()) <= set(fixes.REASONS))
        self.assertEqual(fixes.REASONS["head_tail"], "head/tail uncertain")
        self.assertEqual(fixes.REASONS["coiled"], "coiled")
        self.assertEqual(fixes.REASONS["poor_fit"], "mask fits poorly")
        self.assertEqual(fixes.REASONS["leaves_view"], "leaves the view")

    def test_flags_unfitted_and_edited_rows_give_their_reasons(self) -> None:
        state = straight_state(6)
        state["flag_self_contact"][1] = True
        state["flag_low_iou"][2] = True
        state["flag_pose_jump"][3] = True
        state["flag_edge_inside"][3] = True
        state["fitted"][4] = False
        state["fitted"][5] = False
        state["mask_stale"][5] = True
        reasons = fixes.row_reasons(state)
        self.assertEqual({code: np.flatnonzero(rows).tolist() for code, rows in reasons.items() if rows.any()}, {
            "coiled": [1], "poor_fit": [2], "jump": [3], "leaves_view": [3], "no_pose": [4], "mask_edited": [5],
        })
        self.assertEqual(fixes.codes_of(reasons, 0, 3), ["coiled", "poor_fit", "jump", "leaves_view"])

    def test_head_tail_marks_the_shorter_side_of_a_swap(self) -> None:
        state = straight_state(30)
        state["centerline_xy"][10:14] = state["centerline_xy"][10:14, ::-1]
        self.assertEqual(np.flatnonzero(fixes.orientation_flips(state)).tolist(), [10, 14])
        self.assertEqual(np.flatnonzero(fixes.head_tail_rows(state)).tolist(), [10, 11, 12, 13])
        # A gap in the track (an unfitted frame) breaks the segments; a swap across it is not seen.
        state = straight_state(30)
        state["fitted"][15] = False
        state["centerline_xy"][16:] = state["centerline_xy"][16:, ::-1]
        self.assertFalse(fixes.head_tail_rows(state).any())
        # A tie marks both sides; frames a step apart pair, wider gaps do not.
        state = straight_state(6)
        state["centerline_xy"][3:] = state["centerline_xy"][3:, ::-1]
        self.assertEqual(np.flatnonzero(fixes.head_tail_rows(state)).tolist(), [0, 1, 2, 3, 4, 5])
        state["frame_index"] = np.array([0, 2, 4, 6, 8, 10])
        self.assertEqual(np.flatnonzero(fixes.orientation_flips(state)).tolist(), [3])
        state["frame_index"] = np.array([0, 2, 4, 9, 11, 13])
        self.assertFalse(fixes.orientation_flips(state).any())


class PlacedTests(unittest.TestCase):
    def test_kept_fixes_count_as_placed_whatever_their_algorithm(self) -> None:
        state = straight_state(4)
        algorithm = np.array(["chain_forward", "chain_forward", "independent_fit", "beam_path"])
        job = np.array(["j1", "fix:p000001", "candidates:c000001", "fix:p000002"], dtype="<U64")
        self.assertEqual(pipeline.placed_job(job).tolist(), [False, True, True, True])
        self.assertEqual(pipeline.placed_rows(state, algorithm, job).tolist(), [False, True, True, True])


class IssueTests(unittest.TestCase):
    def test_runs_closer_than_eight_frames_merge_and_short_ones_stay(self) -> None:
        rows = np.zeros(60, dtype=bool)
        rows[[2, 3, 9, 30, 50, 51]] = True
        # 3 -> 9 leaves 5 clean frames (merged); 9 -> 30 leaves 20; 30 -> 50 leaves 19.
        self.assertEqual(fixes.issue_spans(rows), [(2, 9), (30, 30), (50, 51)])
        rows[38] = True  # 30 -> 38 leaves 7: merged; 38 -> 50 leaves 11: not
        self.assertEqual(fixes.issue_spans(rows), [(2, 9), (30, 38), (50, 51)])
        self.assertEqual(fixes.issue_spans(np.zeros(5, dtype=bool)), [])

    def test_issue_report_states_reasons_and_refit_choice(self) -> None:
        n = 60
        state = straight_state(n)
        state["frame_index"] = np.arange(100, 100 + n)
        state["flag_self_contact"][5:8] = True
        state["flag_low_iou"][20:22] = True
        state["centerline_xy"][40:44] = state["centerline_xy"][40:44, ::-1]
        placed = np.zeros(n, dtype=bool)
        reviewed = np.zeros(n, dtype=bool)
        report = fixes.issue_report(state, placed, reviewed)
        issues = report["issues"]
        self.assertEqual([i["id"] for i in issues], ["105-107", "120-121", "140-143"])
        self.assertEqual([i["rows"] for i in issues], [[5, 7], [20, 21], [40, 43]])
        self.assertEqual([i["reasons"] for i in issues], [["coiled"], ["mask fits poorly"], ["head/tail uncertain"]])
        self.assertEqual([i["refit"]["algorithm"] for i in issues], ["beam_path", "slow_refit", "mirror"])
        self.assertEqual({i["state"] for i in issues}, {"unreviewed"})
        self.assertEqual(report["summary"], {
            "analysed": True, "issues": 3, "unreviewed": 3, "reviewed": 0, "fixed": 0, "done": 0, "frames": n,
            "frames_with_reasons": 9, "frames_placed": 0,
        })
        # Looks OK on the first; a person placed the third's rows; one row of the second is fixed, one not.
        reviewed[5:8] = True
        placed[40:44] = True
        placed[20] = True
        states = [i["state"] for i in fixes.issue_report(state, placed, reviewed)["issues"]]
        self.assertEqual(states, ["reviewed", "unreviewed", "fixed"])
        reviewed[21] = True
        report = fixes.issue_report(state, placed, reviewed)
        self.assertEqual([i["state"] for i in report["issues"]], ["reviewed", "fixed", "fixed"])
        self.assertEqual((report["summary"]["done"], report["summary"]["frames_placed"]), (3, 5))
        # A placed row without a reason is an issue of its own (fixed by hand), and a refit with no reason refits from the neighbours.
        placed[55] = True
        last = fixes.issue_report(state, placed, reviewed)["issues"][-1]
        self.assertEqual((last["rows"], last["reasons"], last["state"], last["refit"]["algorithm"]), ([55, 55], [], "fixed", "beam_path"))
        # Not analysed: no issues at all.
        state["fitted"][:] = False
        report = fixes.issue_report(state, np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))
        self.assertEqual((report["issues"], report["summary"]["analysed"]), ([], False))

    def test_refit_choice_follows_the_order_of_reasons(self) -> None:
        self.assertEqual(fixes.refit_algorithm(["head_tail"]), "mirror")
        self.assertEqual(fixes.refit_algorithm(["head_tail", "poor_fit"]), "slow_refit")
        self.assertEqual(fixes.refit_algorithm(["head_tail", "poor_fit", "coiled"]), "beam_path")
        self.assertEqual(fixes.refit_algorithm(["mask_edited"]), "beam_path")
        self.assertEqual(fixes.refit_algorithm([]), "beam_path")
        self.assertEqual({algorithm for _, algorithm in fixes.REFIT_ORDER} | {fixes.DEFAULT_REFIT}, {"beam_path", "slow_refit", "mirror"})
        self.assertEqual({code for code, _ in fixes.REFIT_ORDER}, set(fixes.REASONS))


class PlanTests(unittest.TestCase):
    def test_anchors_are_the_nearest_trusted_rows_and_the_refit_reaches_them(self) -> None:
        n = 40
        state = straight_state(n)
        state["flag_self_contact"][10:13] = True
        state["iou"][9] = 0.85  # a poor overlap: no anchor
        state["flag_holes"][13] = True  # a reason: no anchor
        placed = np.zeros(n, dtype=bool)
        plan = fixes.plan_refit(state, placed, 10, 12)
        self.assertEqual((plan.anchor_before, plan.anchor_after, plan.first, plan.last), (8, 14, 9, 13))
        self.assertEqual((plan.algorithm, plan.codes), ("beam_path", ["coiled"]))
        # A row a person placed anchors even with a reason; the developer's choice overrides the algorithm.
        placed[13] = True
        plan = fixes.plan_refit(state, placed, 10, 12, algorithm="mirror")
        self.assertEqual((plan.anchor_after, plan.last, plan.algorithm), (13, 12, "mirror"))
        # No trusted row within reach: the requested end stays, without an anchor.
        state["iou"][:10] = 0.5
        plan = fixes.plan_refit(state, placed, 10, 12)
        self.assertEqual((plan.anchor_before, plan.first), (None, 10))
        with self.assertRaisesRegex(ValueError, "unknown algorithm"):
            fixes.plan_refit(state, placed, 10, 12, algorithm="teleport")
        with self.assertRaisesRegex(ValueError, "outside"):
            fixes.plan_refit(state, placed, 10, n)

    def test_keyframes_are_the_ends_and_evenly_spaced_frames(self) -> None:
        self.assertEqual(fixes.propose_keyframes(1204, 1260), [1204, 1213, 1223, 1232, 1241, 1251, 1260])
        self.assertEqual(fixes.propose_keyframes(0, 20, 10), [0, 10, 20])
        self.assertEqual(fixes.propose_keyframes(0, 21, 10), [0, 7, 14, 21])
        self.assertEqual(fixes.propose_keyframes(3, 3), [3])
        self.assertEqual(fixes.propose_keyframes(0, 4, 1), [0, 1, 2, 3, 4])
        for first, last, spacing in ((0, 100, 7), (5, 6, 10), (10, 97, 3)):
            frames = fixes.propose_keyframes(first, last, spacing)
            self.assertEqual((frames[0], frames[-1]), (first, last))
            self.assertLessEqual(max(np.diff(frames), default=0), spacing)
        with self.assertRaises(ValueError):
            fixes.propose_keyframes(5, 4)
        with self.assertRaises(ValueError):
            fixes.propose_keyframes(0, 4, 0)


class WorkspaceFixTests(unittest.TestCase):
    """Flip, refit, keep, undo and stitch on the posed synthetic workspace; each test puts the pristine files back."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.workspace = posed_workspace(root)
        cls.pristine = root / "pristine"
        cls.pristine.mkdir()
        for name in PRISTINE:
            if (cls.workspace.path / name).exists():
                shutil.copyfile(cls.workspace.path / name, cls.pristine / name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def setUp(self) -> None:
        # A fresh object: the workspace counts edit ids in memory, and every test starts from an empty log.
        self.workspace = Workspace.open(self.workspace.path)

    def tearDown(self) -> None:
        for name in PRISTINE:
            target, source = self.workspace.path / name, self.pristine / name
            if source.exists():
                shutil.copyfile(source, target)
            elif target.exists():
                target.unlink()
        for directory in ("edits", fixes.FIXES_DIR):
            shutil.rmtree(self.workspace.path / directory, ignore_errors=True)

    def reverse(self, rows: list[int]) -> None:
        """Turn rows around in the state directly, as a pipeline mistake would (no edit, no provenance)."""

        state = self.workspace.load_state()
        for row in rows:
            edits._reverse_row(state, row)
        self.workspace.save_state(state)

    def oriented_like(self, curve: np.ndarray, row: int) -> bool:
        return pose_distance_px(curve, _body_curve(row), None) < pose_distance_px(curve[::-1], _body_curve(row), None)

    def report(self) -> dict:
        state = self.workspace.load_state()
        provenance = self.workspace.load_provenance()
        placed = pipeline.placed_rows(state, provenance["algorithm"], provenance["job"])
        return fixes.issue_report(state, placed, np.zeros(self.workspace.n, dtype=bool))

    def test_the_true_poses_have_no_issue(self) -> None:
        state = self.workspace.load_state()
        self.assertTrue(state["fitted"].all())
        self.assertEqual(self.report()["issues"], [])

    def test_refit_previews_keeps_through_the_edit_log_and_undoes(self) -> None:
        # Rows 4-5 turned around: segments 0-3, 4-5, 6-9, and the short one is the suspect.
        self.reverse([4, 5])
        issue = self.report()["issues"][0]
        self.assertEqual((issue["rows"], issue["codes"], issue["refit"]["algorithm"]), ([4, 5], ["head_tail"], "mirror"))
        state = self.workspace.load_state()
        provenance = self.workspace.load_provenance()
        plan = fixes.plan_refit(state, pipeline.placed_rows(state, provenance["algorithm"], provenance["job"]), 4, 5)
        self.assertEqual((plan.algorithm, plan.first, plan.last, plan.anchor_before, plan.anchor_after), ("mirror", 4, 5, 3, 6))
        preview = fixes.run_refit(self.workspace, plan, preview_id=fixes.next_preview_id(self.workspace), job="j1", device="cpu")
        self.assertEqual((preview.id, preview.kind, preview.algorithm, preview.rows, preview.job), ("p000001", "refit", "mirror", [4, 5], "j1"))
        self.assertEqual(preview.watch_rows, [3, 4, 5, 6])
        self.assertEqual(preview.metrics_before["orientation_flips"], 2)
        self.assertEqual(preview.metrics["orientation_flips"], 0)
        # Saved, listed and loaded back whole; the state is untouched until Keep.
        listed = fixes.list_previews(self.workspace)
        self.assertEqual([m["id"] for m in listed], ["p000001"])
        loaded = fixes.load_preview(self.workspace, "p000001")
        self.assertEqual((loaded.rows, loaded.codes, loaded.anchor_before, loaded.anchor_after), ([4, 5], ["head_tail"], 3, 6))
        for row, pose in zip(loaded.rows, loaded.poses, strict=True):
            self.assertTrue(self.oriented_like(pose.centerline_xy, row))
            before, iou = loaded.before(row)
            np.testing.assert_array_equal(before, state["centerline_xy"][row])
            self.assertAlmostEqual(iou, 0.97)
        np.testing.assert_array_equal(self.workspace.load_state()["centerline_xy"], state["centerline_xy"])
        self.assertIsNone(fixes.preview_problem(self.workspace, loaded))
        # Keep: one set_pose edit, provenance the algorithm under the fix's job, the issue gone, the preview deleted.
        result = fixes.keep(self.workspace, "p000001")
        self.assertEqual((result.kind, result.rows), ("set_pose", [4, 5]))
        after = self.workspace.load_state()
        for row in (4, 5):
            self.assertTrue(self.oriented_like(after["centerline_xy"][row], row))
        provenance = self.workspace.load_provenance()
        self.assertEqual(provenance["algorithm"][4:6].tolist(), ["mirror"] * 2)
        self.assertEqual(provenance["job"][4:6].tolist(), ["fix:p000001"] * 2)
        self.assertEqual(fixes.list_previews(self.workspace), [])
        entry = self.workspace.edits()[-1]
        self.assertEqual(entry["payload"]["fix"], {
            "kind": "refit", "preview": "p000001", "algorithm": "mirror", "codes": ["head_tail"], "anchors": [3, 6], "keyframes": [],
        })
        report = self.report()
        self.assertEqual([(i["rows"], i["state"], i["reasons"]) for i in report["issues"]], [([4, 5], "fixed", [])])
        # The fixes list says what happened in plain words; Undo puts the reversed poses back.
        listed = fixes.fixes_list(self.workspace)
        self.assertEqual([(f["id"], f["kind"], f["title"], f["frames"], f["undoable"]) for f in listed], [
            (result.edit_id, "refit", "Refit: match head/tail to the neighbouring frames", [4, 5], True),
        ])
        fixes.undo_fix(self.workspace, result.edit_id)
        np.testing.assert_array_equal(self.workspace.load_state()["centerline_xy"], state["centerline_xy"])
        self.assertEqual(fixes.fixes_list(self.workspace), [])
        self.assertEqual(self.report()["issues"][0]["state"], "unreviewed")
        with self.assertRaisesRegex(ValueError, "no fix"):
            fixes.undo_fix(self.workspace, result.edit_id)

    def test_keep_refuses_a_preview_whose_frames_changed_and_discard_deletes_it(self) -> None:
        self.reverse([4, 5])
        plan = fixes.RefitPlan("mirror", 4, 5, 3, 6, ["head_tail"])
        preview = fixes.run_refit(self.workspace, plan, preview_id=fixes.next_preview_id(self.workspace), device="cpu")
        edits.flip_orientation(self.workspace, [6])  # the anchor changed
        problem = fixes.preview_problem(self.workspace, fixes.load_preview(self.workspace, preview.id))
        self.assertIn("poses on these frames changed", problem)
        edits_before = len(self.workspace.edits())
        with self.assertRaisesRegex(ValueError, "changed since the fix ran"):
            fixes.keep(self.workspace, preview.id)
        self.assertEqual(len(self.workspace.edits()), edits_before)
        # A mask edit is caught too.
        edits.undo(self.workspace)
        self.assertIsNone(fixes.preview_problem(self.workspace, fixes.load_preview(self.workspace, preview.id)))
        labels = self.workspace.effective_mask(4).astype(np.uint8)
        edits.set_mask(self.workspace, 4, labels)
        self.assertIn("mask changed", fixes.preview_problem(self.workspace, fixes.load_preview(self.workspace, preview.id)))
        self.assertTrue(fixes.delete_preview(self.workspace, preview.id))
        self.assertFalse(fixes.delete_preview(self.workspace, preview.id))
        with self.assertRaises(FileNotFoundError):
            fixes.load_preview(self.workspace, preview.id)
        with self.assertRaises(ValueError):
            fixes.load_preview(self.workspace, "../state")
        self.assertEqual(fixes.next_preview_id(self.workspace), "p000002")

    def test_fixes_list_blocks_an_undo_under_a_later_fix(self) -> None:
        first = edits.flip_orientation(self.workspace, [3, 4], note="Flip head/tail frames 3-4")
        second = edits.flip_orientation(self.workspace, [5])
        third = edits.flip_orientation(self.workspace, [8])
        listed = {f["id"]: f for f in fixes.fixes_list(self.workspace)}
        self.assertEqual(list(listed), [third.edit_id, second.edit_id, first.edit_id])
        self.assertEqual((listed[first.edit_id]["kind"], listed[first.edit_id]["title"]), ("flip", "Flipped head/tail"))
        self.assertEqual(listed[first.edit_id]["note"], "Flip head/tail frames 3-4")
        self.assertEqual(listed[first.edit_id]["frames"], [3, 4])
        # Frame 5 neighbours frame 4: the first flip waits for the second; the third is apart.
        self.assertEqual((listed[first.edit_id]["undoable"], listed[first.edit_id]["blocked_by"]), (False, second.edit_id))
        self.assertTrue(listed[second.edit_id]["undoable"] and listed[third.edit_id]["undoable"])
        with self.assertRaisesRegex(ValueError, f"undo {second.edit_id} first"):
            fixes.undo_fix(self.workspace, first.edit_id)
        fixes.undo_fix(self.workspace, second.edit_id)
        fixes.undo_fix(self.workspace, first.edit_id)
        self.assertEqual([f["id"] for f in fixes.fixes_list(self.workspace)], [third.edit_id])
        np.testing.assert_array_equal(self.workspace.load_state()["centerline_xy"][3:6], np.stack([_body_curve(r) for r in (3, 4, 5)]))

    def test_stitch_pins_the_keyframes_and_refits_the_gaps(self) -> None:
        self.reverse([2, 3, 4, 5, 6, 7])
        # Labeled keyframes: the true bodies, one of them traced with fewer points.
        coarse = _body_curve(4)[::3]
        profile = 12.0 * default_width_template(100)
        keyframes = [
            Keyframe(1, _body_curve(1), profile), Keyframe(4, coarse, profile[::3]), Keyframe(8, _body_curve(8), profile),
        ]
        progress: list[float] = []
        preview = fixes.run_stitch(
            self.workspace, keyframes, {"beam": 2, "anchor_diversity": False}, preview_id=fixes.next_preview_id(self.workspace), device="cpu",
            progress=lambda p, m: progress.append(p),
        )
        self.assertEqual((preview.kind, preview.algorithm, preview.first, preview.last, preview.keyframes), ("stitch", "stitch", 1, 8, [1, 4, 8]))
        self.assertEqual(preview.rows, list(range(1, 9)))
        self.assertEqual((progress[0], progress[-1]), (0.0, 1.0))
        poses = dict(zip(preview.rows, preview.poses, strict=True))
        for row in (1, 4, 8):
            self.assertEqual((poses[row].source, poses[row].start), ("keyframe", "label"))
        np.testing.assert_allclose(poses[1].centerline_xy, _body_curve(1))
        self.assertLess(pose_distance_px(poses[4].centerline_xy, _body_curve(4), None), 1.0)  # resampled from 34 points
        self.assertGreater(poses[1].iou, 0.9)
        for row in (2, 3, 5, 6, 7):
            self.assertTrue(self.oriented_like(poses[row].centerline_xy, row), f"row {row}")
            self.assertGreater(poses[row].iou, 0.85, f"row {row}")
        self.assertEqual(preview.metrics["orientation_flips"], 0)
        self.assertGreater(preview.metrics_before["orientation_flips"], 0)
        # Keep is one undoable edit over the whole stretch, attributed to the stitch.
        result = fixes.keep(self.workspace, preview.id)
        self.assertEqual(result.rows, list(range(1, 9)))
        self.assertEqual(set(self.workspace.load_provenance()["algorithm"][1:9].tolist()), {"stitch"})
        self.assertEqual(self.workspace.edits()[-1]["payload"]["fix"]["keyframes"], [1, 4, 8])
        self.assertEqual(fixes.fixes_list(self.workspace)[0]["title"], "Relabeled from 3 keyframes")
        self.assertEqual(fixes.fixes_list(self.workspace)[0]["kind"], "relabel")
        fixes.undo_fix(self.workspace, result.edit_id)
        self.assertEqual(self.workspace.load_provenance()["algorithm"][1:9].tolist(), ["independent_fit"] * 8)

    def test_keyframes_are_checked(self) -> None:
        profile = 12.0 * default_width_template(100)
        with self.assertRaisesRegex(ValueError, "same frame"):
            fixes.save_keyframes(self.workspace, "p000009", [Keyframe(1, _body_curve(1), profile), Keyframe(1, _body_curve(1), profile)])
        with self.assertRaisesRegex(ValueError, "one diameter per"):
            fixes.save_keyframes(self.workspace, "p000009", [Keyframe(1, _body_curve(1), profile[:5])])
        with self.assertRaisesRegex(ValueError, "positive"):
            fixes.save_keyframes(self.workspace, "p000009", [Keyframe(1, _body_curve(1), 0 * profile)])
        with self.assertRaisesRegex(ValueError, "at least one"):
            fixes.save_keyframes(self.workspace, "p000009", [])
        path = fixes.save_keyframes(self.workspace, "p000009", [Keyframe(2, _body_curve(2)[::2], profile[::2])])
        loaded = fixes.load_keyframes(self.workspace, "p000009")
        self.assertEqual([k.row for k in loaded], [2])
        np.testing.assert_array_equal(loaded[0].centerline_xy, _body_curve(2)[::2])
        self.assertTrue(path.exists())
        # One keyframe is a stitch with no gap: the label's pose alone.
        preview = fixes.run_stitch(self.workspace, loaded, preview_id="p000009", device="cpu")
        self.assertEqual(preview.rows, [2])
        self.assertTrue(fixes.delete_preview(self.workspace, "p000009"))
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
