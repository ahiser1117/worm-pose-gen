"""A tracked regional refit preserves head identity and stays candidate-only."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

import h5py
import numpy as np

from worm_pose_gen import algorithms, edits, pipeline
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import default_width_template
from worm_pose_gen.workspace import Workspace
from tests.test_batch_fit import SMALL, _render


class TrackedRegionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.recording = root / "tracked.h5"
        self.config = replace(SMALL, stage_steps=(12, 12))
        self.latents, self.masks = [], []
        features = np.ones((9, 3, 3), np.float32)
        for frame in range(9):
            latent = np.concatenate((np.zeros(16), [0., 110., 100., 55. + frame]))
            self.latents.append(latent)
            self.masks.append(_render(latent, 120, 200))
            features[frame, :2, 0] = decode_centerline(latent)[0] + 1
        with h5py.File(self.recording, "w") as h:
            h.create_dataset("img_nir", data=np.asarray(self.masks, np.uint8))
            h.create_dataset("pos_feature", data=features)
        self.workspace = Workspace.create(root / "workspaces", "tracked", self.recording, 0, 8, step=2)
        state = pipeline.workspace_arrays(self.workspace, self.config)
        for row, frame in enumerate(self.workspace.frame_index):
            curve = decode_centerline(self.latents[frame])
            state["latent"][row] = self.latents[frame]
            state["centerline_xy"][row] = curve
            state["width_px"][row] = 12
            state["width_shape"][row] = 0
            state["width_profile"][row] = default_width_template() * 12
            state["body_length_px"][row] = 110
            state["points_in_fov"][row] = 100
            state["crop"][row] = [0, 200, 0, 120]
            state["fitted"][row] = True
            state["iou"][row] = .99
            state["energy"][row] = .01
        self.workspace.save_state(state)
        self.workspace.set_masks(range(5), [self.masks[f] for f in self.workspace.frame_index])
        pipeline.update_summary(self.workspace, {"fit_config": asdict(self.config), "mask_cleanup": {"min_worm_pixels": 1}})

    def test_wrong_orientation_distractor_and_missing_tracking_do_not_jump(self):
        state = self.workspace.load_state()
        for row in (1, 2, 3):
            edits._reverse_row(state, row)
        self.workspace.save_state(state)
        with h5py.File(self.recording, "r+") as h:
            h["pos_feature"][4, 0, 0] = 150  # confident but on another body part
            h["pos_feature"][6, 2, 0] = .05  # acquisition lost this landmark
        result = algorithms.run_region(self.workspace, "tracked_head", 1, 4, {"max_head_step_px": 1.5, "fill_holes": "workspace"}, anchor_before=0, device="cpu")
        self.assertEqual(len(result.path), 4)
        previous = state["centerline_xy"][0, 0]
        for row in range(1, 5):
            pose = result.chosen(row)
            self.assertFalse(result.path_by_row[row][1])
            self.assertLess(np.linalg.norm(pose.centerline_xy[0] - previous), 3.0001)
            self.assertLess(pose.centerline_xy[0, 0], pose.centerline_xy[-1, 0])
            previous = pose.centerline_xy[0]
        self.assertEqual(result.metrics["tracking_fallback_frames"], 1)
        self.assertEqual(result.metrics["tracked_frames"], 3)
        stored = algorithms.load_candidate_set(self.workspace, result.id)
        self.assertEqual(stored.params["max_head_step_px"], 1.5)
        self.assertEqual(stored.metrics["head_tracking"]["landmark"], "nose")
        np.testing.assert_array_equal(self.workspace.load_state()["centerline_xy"], state["centerline_xy"])
        algorithms.accept_candidates(self.workspace, result.id)
        self.assertLess(self.workspace.load_state()["centerline_xy"][2, 0, 0], 80)

    def test_unanchored_initialization_uses_nose_and_missing_schema_is_clear(self):
        result = algorithms.run_region(self.workspace, "tracked_head", 0, 0, {"fill_holes": "workspace"}, device="cpu")
        np.testing.assert_allclose(result.chosen(0).centerline_xy[0], decode_centerline(self.latents[0])[0], atol=3)
        with h5py.File(self.recording, "r+") as h:
            del h["pos_feature"]
        with self.assertRaisesRegex(ValueError, "Head tracking unavailable.*Choose another"):
            algorithms.run_region(self.workspace, "tracked_head", 0, 0, {"fill_holes": "workspace"}, device="cpu")

    def test_skipped_mask_uses_elapsed_source_frames(self):
        self.workspace.set_masks([2], [np.zeros((120, 200), bool)])
        result = algorithms.run_region(self.workspace, "tracked_head", 1, 3, {"max_head_step_px": .5, "fill_holes": "workspace"}, anchor_before=0, device="cpu")
        self.assertEqual([row for row, _, _ in result.path], [1, 3])
        self.assertLessEqual(np.linalg.norm(result.chosen(3).centerline_xy[0] - result.chosen(1).centerline_xy[0]), 2.0001)
        self.assertLessEqual(result.metrics["max_head_step_px_per_frame"], .5001)

    def test_selected_anchors_are_fixed_and_boundary_limits_are_checked(self):
        state = self.workspace.load_state()
        edits._reverse_row(state, 0)
        self.workspace.save_state(state)
        with self.assertRaisesRegex(ValueError, "Correct its orientation"):
            algorithms.run_region(self.workspace, "tracked_head", 1, 2, {"fill_holes": "workspace"}, anchor_before=0, device="cpu")
        edits._reverse_row(state, 0)
        state["centerline_xy"][4, :, 1] += 30
        state["latent"][4, -1] += 30
        self.workspace.save_state(state)
        with self.assertRaisesRegex(ValueError, "cannot reconnect to the after anchor"):
            algorithms.run_region(self.workspace, "tracked_head", 1, 3, {"max_head_step_px": 1.5, "fill_holes": "workspace"}, anchor_before=0, anchor_after=4, device="cpu")
        self.assertEqual(algorithms.list_candidate_sets(self.workspace), [])

    def test_lost_tracking_region_can_continue_from_trusted_anchor(self):
        with h5py.File(self.recording, "r+") as h:
            h["pos_feature"][2:, 2, 0] = .1
        result = algorithms.run_region(self.workspace, "tracked_head", 1, 2, {"fill_holes": "workspace"}, anchor_before=0, device="cpu")
        self.assertEqual(result.metrics["tracking_fallback_frames"], 2)
        self.assertEqual(len(result.path), 2)

    def test_nonfinite_parameters_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "must be finite"):
                algorithms.get_algorithm("tracked_head").resolve({"max_head_step_px": value})

    def test_job_process_loads_method_and_reports_progress(self):
        progress = self.workspace.path / "progress.json"
        spec = {"algorithm": "tracked_head", "first": 1, "last": 2, "anchor_before": 0,
                "params": {"max_head_step_px": 2., "fill_holes": "on"}, "id": "tracked_job"}
        command = pipeline.region_command(self.workspace.path, spec) + ["--device", "cpu"]
        completed = subprocess.run(command, env={**os.environ, "WORM_POSE_PROGRESS_FILE": str(progress)},
                                   capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(progress.read_text())
        self.assertEqual(report["progress"], 1.)
        self.assertEqual(report["result"]["candidate_set"], "tracked_job")
        candidate_set = algorithms.load_candidate_set(self.workspace, "tracked_job")
        self.assertEqual(candidate_set.params["fill_holes"], "on")
        self.assertEqual(candidate_set.metrics["tracked_frames"], 2)


if __name__ == "__main__":
    unittest.main()
