"""Motion previews must not validate an entire region's masks on every frame."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import torch

from worm_pose_gen import algorithms
from worm_pose_gen.app.workspace_view import WorkspaceView
from worm_pose_gen.label_app import RecordingSource
from worm_pose_gen.pose_viewer import Segmenters
from worm_pose_gen.workspace import Workspace
from tests.test_pose_viewer import FRAMES, _write_recording, _write_run


class WorkspacePreviewTests(unittest.TestCase):
    def test_motion_defers_mask_reads_and_candidates_but_pause_validates_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, run = root / "recording.h5", root / "run"
            _write_recording(recording)
            _write_run(run, recording)
            workspace = Workspace.import_run(root / "workspaces", run, "demo")
            with h5py.File(recording) as handle:
                masks = handle["/img_nir"][:] < 140
            workspace.set_masks(list(range(FRAMES)), masks)
            sets = [algorithms.run_region(workspace, "mirror", 1, 4, {},
                                         anchor_before=0, anchor_after=5, device="cpu")
                    for _ in range(2)]
            source = RecordingSource(recording, root / "flat_fields")
            view = WorkspaceView(workspace, source, None)
            device = torch.device("cpu")
            segmenters = Segmenters(device, fallback=None)

            def frame(detail):
                return view.frame(2, segmenters, None, device, raw=False, detail=detail)

            try:
                # Refresh opens the view's own Workspace instance. Guard reads
                # on that instance, while leaving its frame cache cold.
                view.run
                for warm_cache in (False, True):
                    with self.subTest(warm_cache=warm_cache):
                        if warm_cache:
                            full = frame("full")
                            self.assertFalse(full["details_deferred"])
                            self.assertTrue(full["has_stored_mask"])
                            self.assertIn("tube", full["layers"])
                            self.assertEqual(len(full["candidate_sets"]), 2)
                            self.assertFalse(any(s["stale"] for s in full["candidate_sets"]))
                        with ExitStack() as stack:
                            for target, names in (
                                (view.workspace, ("get_mask", "get_override_mask", "mask_revision", "clear_mask_cache")),
                                (algorithms, ("list_candidate_sets", "validate_candidate_masks")),
                            ):
                                for name in names:
                                    stack.enter_context(patch.object(target, name, side_effect=AssertionError(f"{name} during motion")))
                            stack.enter_context(patch("worm_pose_gen.pose_viewer.render_tube", side_effect=AssertionError("tube during motion")))
                            light = frame("light")
                        self.assertEqual(list(light["layers"]), ["image"])
                        self.assertEqual(light["errors"], [])
                        self.assertIn("centerline_xy", light["pose"])
                        self.assertEqual(light["provenance"], view.row_provenance(2))
                        self.assertTrue(light["details_deferred"])
                        for key in ("has_stored_mask", "has_override", "mask_revision", "candidate_sets"):
                            self.assertNotIn(key, light)
                        self.assertNotIn("candidate_sets", light["pose"])

                # Change an anchor through another writer after warming the view's
                # chunk cache: paused validation and acceptance must see that edit.
                writer = Workspace.open(workspace.path)
                changed = masks[0].copy()
                changed[0, 0] = ~changed[0, 0]
                writer.set_masks([0], [changed])
                full = frame("full")
                self.assertTrue(all(s["stale"] for s in full["candidate_sets"]))
                self.assertEqual(full["pose"]["candidate_sets"], full["candidate_sets"])
                self.assertEqual(full["mask_revision"], workspace.mask_revision(2))
                for candidate in sets:
                    with self.assertRaisesRegex(ValueError, "stale"):
                        algorithms.accept_candidates(workspace, candidate.id)
            finally:
                source.close()


if __name__ == "__main__":
    unittest.main()
