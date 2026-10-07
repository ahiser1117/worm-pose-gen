"""Propagate and track score with the body-field network the fit used (``fit_params.body_net`` in the workspace summary)."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_frame_view import _write_recording, _write_workspace
from worm_pose_gen import pipeline
from worm_pose_gen.pipeline import update_summary, workspace_predictions
from worm_pose_gen.workspace import Workspace


class FitNetworkTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.recording = self.root / "movie.h5"
        _write_recording(self.recording)
        # Fit stages record fit_params.body_net (and a fit_config).
        for name, body_net in (("stage_with", "/nets/weights.ckpt"), ("stage_without", None)):
            workspace = _write_workspace(self.root / "workspaces", name, self.recording)
            update_summary(workspace, {"fit_params": {"body_net": body_net}, "fit_config": {"field_ap_weight": 0.001 if body_net else 0.0}})

    def test_propagate_and_track_read_the_network_from_the_fit_summary(self):
        """``workspace_predictions`` (propagate, track) follows the fit's ``fit_params.body_net``, not the stage's own params."""

        workspaces = self.root / "workspaces"
        params = pipeline.SegmentParams(dataset_root=str(self.root / "cache"))
        with mock.patch.object(pipeline, "field_predictor") as predictor:
            with workspace_predictions(Workspace.open(workspaces / "stage_with"), params, "cpu") as predictions_of:
                self.assertIsNotNone(predictions_of)
            self.assertEqual(predictor.call_args.args[0], "/nets/weights.ckpt")
            predictor.reset_mock()
            with workspace_predictions(Workspace.open(workspaces / "stage_without"), params, "cpu") as predictions_of:
                self.assertIsNone(predictions_of)
            predictor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
