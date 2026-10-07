"""Mask edits preserve labels, invalidate fits, restore exact before slices, and make results fit to the old masks stale."""
import unittest
from unittest.mock import patch
import numpy as np
from tests.test_algorithms import keep
from tests.test_edits import EditFixture
from tests.test_frame_view import HEIGHT, WIDTH
from worm_pose_gen import edits, algorithms, fixes


class MaskEditTests(EditFixture):
    def test_labels_empty_invalidation_and_undo(self):
        row = 5
        before = self.workspace.load_arrays()
        provenance = self.workspace.load_provenance()
        labels = np.zeros((HEIGHT, WIDTH), np.uint8)
        labels[10:20, 10:20] = 255
        revision = self.workspace.mask_revision(row)
        result = edits.set_mask(self.workspace, row, labels, revision=revision)
        np.testing.assert_array_equal(self.workspace.get_override_mask(row), labels)
        self.assertFalse(self.workspace.effective_mask(row).any())
        self.assertFalse(self.state()['fitted'][row])
        self.assertTrue(self.state()['mask_stale'][row])
        self.assertEqual(self.state()['worm_pixels'][row], 0)
        self.assertEqual(self.hypotheses()['hypotheses_count'][row], 0)
        with self.assertRaisesRegex(ValueError, 'changed'):
            edits.set_mask(self.workspace, row, labels, revision=revision)
        edits.undo(self.workspace, result.edit_id)
        self.assertIsNone(self.workspace.get_override_mask(row))
        self.assertFalse(self.state()['mask_stale'][row])
        for key, value in before.items():
            np.testing.assert_array_equal(self.workspace.load_arrays()[key], value, err_msg=key)
        for key, value in provenance.items():
            np.testing.assert_array_equal(self.workspace.load_provenance()[key], value)

    def test_clear_undo_shape_and_label_validation(self):
        labels = np.zeros((HEIGHT, WIDTH), np.uint8)
        labels[3:8, 3:8] = 1
        labels[20, 20] = 255
        self.workspace.set_masks([1], [np.ones_like(labels, dtype=bool)])
        edits.set_mask(self.workspace, 1, labels)
        result = edits.set_mask(self.workspace, 1, None)
        self.assertEqual(self.state()['worm_pixels'][1], HEIGHT * WIDTH)
        edits.undo(self.workspace, result.edit_id)
        np.testing.assert_array_equal(self.workspace.get_override_mask(1), labels)
        self.assertTrue(self.state()['mask_stale'][1])
        for bad in (np.zeros((2, 2)), np.full(labels.shape, 2)):
            with self.assertRaises(ValueError):
                edits.set_mask(self.workspace, 1, bad)
        np.testing.assert_array_equal(self.workspace.get_override_mask(1), labels)

    def test_result_mask_versions_include_anchors(self):
        revisions = {str(r): self.workspace.mask_revision(r) for r in range(1, 5)}
        algorithms.validate_mask_revisions(self.workspace, revisions, [2, 3], [1, 4], 'test')
        edits.set_mask(self.workspace, 1, np.zeros((HEIGHT, WIDTH), np.uint8))
        with self.assertRaisesRegex(ValueError, 'stale'):
            algorithms.validate_mask_revisions(self.workspace, revisions, [2, 3], [1, 4], 'test')
        with self.assertRaisesRegex(ValueError, 'unversioned'):
            algorithms.validate_mask_revisions(self.workspace, {}, [2, 3], [1, 4], 'test')

    def test_write_failure_rolls_back_and_keeps_label_artifact(self):
        labels = np.ones((HEIGHT, WIDTH), np.uint8)
        original = self.workspace.load_arrays()
        save = self.workspace.save_state
        calls = 0
        def fail_once(arrays):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected write failure")
            save(arrays)
        with patch.object(self.workspace, "save_state", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected"):
                edits.set_mask(self.workspace, 5, labels)
        self.assertIsNone(self.workspace.get_override_mask(5))
        for key, value in original.items():
            np.testing.assert_array_equal(self.workspace.load_arrays()[key], value)
        self.assertEqual(self.workspace.edits(), [])
        with np.load(self.workspace.path / "edits" / "e000001.npz") as saved:
            np.testing.assert_array_equal(saved["mask:after_labels"], labels)

    def _single_row_result(self, anchor_before=None):
        from worm_pose_gen.pipeline import workspace_setup
        pose = algorithms.CandidatePose.from_state(self.workspace.load_state(), 5, workspace_setup(self.workspace).config)
        candidate = algorithms.CandidateSet(algorithm="slow_refit", params={}, first=5, last=5,
            anchor_before=anchor_before, anchor_after=None, rows=[5], candidates={5: [pose]}, path=[(5, 0, False)], metrics={})
        candidate.mask_revisions = {str(r): self.workspace.mask_revision(r) for r in (anchor_before, 5) if r is not None}
        preview = fixes.preview_from(self.workspace, fixes.next_preview_id(self.workspace), "refit", candidate, self.workspace.load_state())
        fixes.save_preview(self.workspace, preview)
        return preview

    def test_keep_rejects_changed_mask_without_mutating_state(self):
        preview = self._single_row_result()
        edits.set_mask(self.workspace, 5, np.ones((HEIGHT, WIDTH), np.uint8))
        before = self.workspace.load_arrays()
        with self.assertRaisesRegex(ValueError, "mask changed"):
            fixes.keep(self.workspace, preview.id)
        for key, value in before.items():
            np.testing.assert_array_equal(self.workspace.load_arrays()[key], value)

    def test_keep_checks_anchor_revision_after_edit_lock_acquisition(self):
        preview = self._single_row_result(anchor_before=4)
        original_set_poses = edits.set_poses
        def race(*args, **kwargs):
            self.workspace.set_override_mask(4, np.ones((HEIGHT, WIDTH), np.uint8))
            return original_set_poses(*args, **kwargs)
        before = self.workspace.load_arrays()
        with patch.object(edits, "set_poses", side_effect=race):
            with self.assertRaisesRegex(ValueError, "mask changed"):
                fixes.keep(self.workspace, preview.id)
        for key, value in before.items():
            np.testing.assert_array_equal(self.workspace.load_arrays()[key], value)

    def test_changed_cached_base_mask_is_rejected(self):
        from worm_pose_gen.workspace import Workspace
        initial = np.zeros((HEIGHT, WIDTH), np.uint8)
        self.workspace.set_masks([5], [initial])
        revisions = {"5": self.workspace.mask_revision(5)}
        Workspace.open(self.workspace.path).set_masks([5], [np.ones_like(initial)])
        with self.assertRaisesRegex(ValueError, "stale"):
            algorithms.validate_mask_revisions(self.workspace, revisions, [5], [], "test")

    def test_propagation_never_restores_invalid_or_explicitly_accepted_baseline(self):
        from worm_pose_gen import pipeline
        state = self.workspace.load_state()
        pipeline.independent_copies(state)
        self.workspace.save_state(state)
        edits.set_mask(self.workspace, 5, np.zeros((HEIGHT, WIDTH), np.uint8))
        changed = self.workspace.load_state()
        self.assertTrue(np.isnan(changed["latent_independent"][5]).all())
        self.assertTrue(np.isnan(changed["body_length_px"][5]))
        self.assertEqual(pipeline.restore_independent_rows(changed,np.full(6,"chain_forward")), [0,1,2,3,4])
        # A kept chain refit has the same algorithm id as a pipeline chain,
        # but its fix job provenance makes it a fixed manual choice.
        state = self.workspace.load_state()
        state["centerline_xy"][0] += 10
        curve = state["centerline_xy"][0].copy()
        provenance = np.full(6,"chain_forward")
        jobs = np.full(6,"stage", dtype="<U64")
        jobs[0] = "fix:p000001"
        self.assertNotIn(0,pipeline.restore_independent_rows(state,provenance,jobs))
        np.testing.assert_array_equal(state["centerline_xy"][0],curve)
        self.assertTrue(pipeline.placed_rows(state,provenance,jobs)[0])

    def test_actual_tiny_region_refit_kept_preserves_outside_poses(self):
        from dataclasses import asdict
        import torch
        from worm_pose_gen import pipeline
        from worm_pose_gen.batch_fit import BatchFitConfig
        torch.set_num_threads(1)
        config = BatchFitConfig(stage_downsample=(2,), stage_steps=(2,), stage_lr_scale=(1.0,),
            stage_point_stride=(2,), compile_renderer=False, crop_padding=8, crop_multiple=8,
            length_bounds_px=None, width_bounds_px=None, length_prior_px=50., width_prior_px=10.,
            default_length_px=50., default_width_px=10.)
        pipeline.update_summary(self.workspace,{"fit_config":asdict(config),"mask_cleanup":{"min_worm_pixels":1}})
        labels = np.zeros((HEIGHT,WIDTH),np.uint8)
        labels[40:50,35:85] = 1
        edits.set_mask(self.workspace,5,labels)
        before = self.workspace.load_state()
        before_provenance = self.workspace.load_provenance()
        candidate = algorithms.run_algorithm(self.workspace,"slow_refit",5,5,{"preset":"fast"},device="cpu")
        self.assertEqual(len(candidate.path),1)
        self.assertFalse(self.workspace.load_state()["fitted"][5])
        result = keep(self.workspace,candidate)
        after = self.workspace.load_state()
        self.assertTrue(after["fitted"][5])
        self.assertFalse(after["mask_stale"][5])
        self.assertTrue(np.isfinite(after["iou"][5]))
        for key in edits.POSE_FIELDS:
            np.testing.assert_array_equal(after[key][:5],before[key][:5],err_msg=key)
        for key,value in before_provenance.items():
            np.testing.assert_array_equal(self.workspace.load_provenance()[key][:5],value[:5])
        edits.undo(self.workspace,result.edit_id)
        self.assertTrue(self.workspace.load_state()["mask_stale"][5])
        np.testing.assert_array_equal(self.workspace.get_override_mask(5),labels)


if __name__ == '__main__':
    unittest.main()
