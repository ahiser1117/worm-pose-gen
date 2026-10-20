"""Motion previews must not read masks or render tubes on every frame; a rested frame reads the masks as they are now."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import torch

from worm_pose_gen.app.frame_view import Segmenters
from worm_pose_gen.app.workspace_view import WorkspaceView
from worm_pose_gen.recordings import RecordingSource
from worm_pose_gen.workspace import Workspace
from tests.test_frame_view import FRAMES, _write_recording, _write_workspace


class WorkspacePreviewTests(unittest.TestCase):
    def test_motion_defers_mask_reads_and_a_rested_frame_sees_other_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recording.h5"
            _write_recording(recording)
            workspace = _write_workspace(root / "workspaces", "demo", recording)
            with h5py.File(recording) as handle:
                masks = handle["/img_nir"][:] < 140
            workspace.set_masks(list(range(FRAMES)), masks)
            source = RecordingSource(recording, root / "flat_fields")
            view = WorkspaceView(workspace, source, None)
            device = torch.device("cpu")
            segmenters = Segmenters(device)

            def frame(detail):
                return view.frame(2, segmenters, device, raw=False, detail=detail)

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
                        with ExitStack() as stack:
                            for name in ("get_mask", "get_override_mask", "mask_revision", "clear_mask_cache"):
                                stack.enter_context(patch.object(view.workspace, name, side_effect=AssertionError(f"{name} during motion")))
                            stack.enter_context(patch("worm_pose_gen.app.frame_view.render_tube", side_effect=AssertionError("tube during motion")))
                            light = frame("light")
                        self.assertEqual(list(light["layers"]), ["image"])
                        self.assertEqual(light["errors"], [])
                        self.assertIn("centerline_xy", light["pose"])
                        self.assertEqual(light["provenance"], view.row_provenance(2))
                        self.assertTrue(light["details_deferred"])
                        for key in ("has_stored_mask", "has_override", "mask_revision"):
                            self.assertNotIn(key, light)

                # Another writer edits the masks after the view warmed its chunk cache: the rested frame reports the new revision.
                writer = Workspace.open(workspace.path)
                before = frame("full")["mask_revision"]
                changed = masks[2].copy()
                changed[0, 0] = ~changed[0, 0]
                writer.set_masks([2], [changed])
                view.invalidate()
                full = frame("full")
                self.assertNotEqual(full["mask_revision"], before)
                self.assertEqual(full["mask_revision"], writer.mask_revision(2))
            finally:
                source.close()


if __name__ == "__main__":
    unittest.main()
