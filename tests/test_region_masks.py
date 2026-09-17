"""Per-refit mask cleanup must be temporary and respect manual labels."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import h5py
import numpy as np

from worm_pose_gen import algorithms
from worm_pose_gen.pipeline import SegmentParams, run_stage
from worm_pose_gen.workspace import Workspace
from tests.test_pipeline import FRAMES, HEIGHT, WIDTH, SEGMENT_PARAMS, _write_recording


class RegionMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        recording = root / "rec.h5"
        _write_recording(recording)
        self.workspace = Workspace.create(root / "workspaces", "holes", recording, 0, FRAMES - 1)
        run_stage(self.workspace, "segment", SEGMENT_PARAMS, device="cpu")
        self.raw = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.raw[5:35, 5:35] = True
        self.raw[18:20, 18:20] = False
        self.filled = self.raw.copy()
        self.filled[18:20, 18:20] = True
        self.workspace.set_masks([0, 1], [self.filled, self.raw])

    def test_off_recovers_holes_from_segmenter_without_changing_saved_masks(self):
        original = self.workspace.mask_revision(0)
        with patch.object(algorithms, "segment_rows", return_value={0: self.raw}) as segment:
            ctx = algorithms.build_context(self.workspace, 0, 0, device="cpu", fill_holes="off")
        self.assertFalse(segment.call_args.args[3].fill_holes)
        np.testing.assert_array_equal(ctx.masks[0], self.raw)
        np.testing.assert_array_equal(self.workspace.get_mask(0), self.filled)
        self.assertEqual(ctx.mask_revisions, {"0": original})
        self.assertEqual(self.workspace.mask_revision(0), original)
        default = algorithms.build_context(self.workspace, 0, 0, device="cpu")
        np.testing.assert_array_equal(default.masks[0], self.filled)

    def test_off_resegments_actual_video_and_recovers_filled_hole(self):
        with h5py.File(self.workspace.recording, "r+") as handle:
            handle["img_nir"][0] = np.where(self.raw, 0, 200).astype(np.uint8)
        params = SegmentParams(checkpoint=None, flat_field=False, min_worm_pixels=1)
        with patch.object(algorithms, "_segment_params", return_value=params):
            result = algorithms.build_context(self.workspace, 0, 0, device="cpu", fill_holes="off")
        np.testing.assert_array_equal(result.masks[0], self.raw)
        np.testing.assert_array_equal(self.workspace.get_mask(0), self.filled)

    def test_legacy_masks_without_checkpoint_metadata_use_default_segmenter(self):
        with patch.object(algorithms, "read_summary", return_value={}):
            params = algorithms._segment_params(SimpleNamespace(info=SimpleNamespace(settings={})))
        self.assertEqual(params.checkpoint, SegmentParams().checkpoint)

    def test_on_fills_and_off_preserves_manual_edits_and_ignore(self):
        labels = self.raw.astype(np.uint8)
        labels[18, 18] = 255
        self.workspace.set_override_mask(1, labels)
        revision = self.workspace.mask_revision(1)
        with patch.object(algorithms, "segment_rows", return_value={}) as segment:
            off = algorithms.build_context(self.workspace, 1, 1, device="cpu", fill_holes="off")
        self.assertEqual(segment.call_args.args[1], [])
        np.testing.assert_array_equal(off.masks[1], labels == 1)
        on = algorithms.build_context(self.workspace, 1, 1, device="cpu", fill_holes="on")
        self.assertFalse(on.masks[1][18, 18])
        self.assertTrue(on.masks[1][19, 19])
        np.testing.assert_array_equal(self.workspace.get_override_mask(1), labels)
        self.assertEqual(self.workspace.mask_revision(1), revision)
        algorithms.validate_mask_revisions(self.workspace, on.mask_revisions, [1], [])
        labels[6, 6] = 0
        self.workspace.set_override_mask(1, labels)
        with self.assertRaisesRegex(ValueError, "stale"):
            algorithms.validate_mask_revisions(self.workspace, on.mask_revisions, [1], [])

    def test_run_passes_and_persists_option_without_changing_masks(self):
        algorithm = algorithms.get_algorithm("independent_multistart")
        captured = {}
        def fit(ctx, params, progress):
            captured.update(mask=ctx.masks[1], params=params)
            return algorithms.CandidateSet(algorithm.id, params, 1, 1, None, None, [1], {}, [], {})
        with patch.object(algorithm, "run", side_effect=fit):
            result = algorithms.run_region(self.workspace, algorithm.id, 1, 1, {"fill_holes": "on"}, device="cpu")
        self.assertTrue(captured["mask"][19, 19])
        self.assertEqual(captured["params"]["fill_holes"], "on")
        self.assertEqual(algorithms.load_candidate_set(self.workspace, result.id).params["fill_holes"], "on")
        self.assertEqual(algorithms.outcomes(self.workspace.path.parent)[0]["params"]["fill_holes"], "on")
        np.testing.assert_array_equal(self.workspace.get_mask(1), self.raw)

    def test_registry_exposes_option_only_for_fitting(self):
        for entry in algorithms.list_algorithms():
            params = {p["name"]: p for p in entry["parameters"]}
            if entry["id"] == "mirror":
                self.assertNotIn("fill_holes", params)
            else:
                self.assertEqual(params["fill_holes"]["choices"], ["workspace", "on", "off"])
                self.assertEqual(algorithms.get_algorithm(entry["id"]).resolve({})["fill_holes"], "off" if entry["id"] == "tracked_head" else "workspace")
        with self.assertRaises(ValueError):
            algorithms.build_context(self.workspace, 0, 0, device="cpu", fill_holes="invalid")
