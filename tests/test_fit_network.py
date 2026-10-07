"""Which network a workspace was fit with, and the body-field checkpoint the Run panel offers."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from tests.test_pose_viewer import _write_recording, _write_run
from worm_pose_gen import pipeline
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.pipeline import update_summary, workspace_predictions
from worm_pose_gen.workspace import Workspace


class FitNetworkTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.recording = self.root / "movie.h5"
        _write_recording(self.recording)
        workspaces = self.root / "workspaces"
        # Fit stages of this app record fit_params.body_net (and a fit_config).
        for name, body_net in (("stage_with", "/nets/best.ckpt"), ("stage_without", None)):
            run = self.root / "runs" / name
            _write_run(run, self.recording)
            workspace = Workspace.import_run(workspaces, run, name)
            update_summary(workspace, {"fit_params": {"body_net": body_net}, "fit_config": {"field_ap_weight": 0.001 if body_net else 0.0}})
        # Imported runs: the run's summary, with or without body-field weights in its fit_config.
        for name, weights in (("imported_with", {"field_ap_weight": 0.001, "field_end_weight": 0.01}), ("imported_without", {})):
            run = self.root / "runs" / name
            _write_run(run, self.recording)
            summary = json.loads((run / "summary.json").read_text())
            summary["fit_config"].update(weights)
            (run / "summary.json").write_text(json.dumps(summary))
            Workspace.import_run(workspaces, run, name)
        Workspace.create(workspaces, "fresh", self.recording, 0, 5, 1)
        self.checkpoint = self.root / "nets" / "best.ckpt"

    def app(self, body_net):
        config = AppConfig(workspaces_root=self.root / "workspaces", recording_roots=(self.root,), poses_root=self.root / "poses",
                           corpus_root=self.root / "corpus", dataset_root=self.root / "cache", checkpoint=None, body_net=body_net,
                           prior_cache=None, notes=self.root / "notes.json", device="cpu", gpus=())
        app = create_app(config)
        client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(client.close)
        self.addCleanup(app.state.app_state.close)
        return client

    def test_state_carries_the_app_checkpoint(self):
        client = self.app(self.checkpoint)
        self.assertEqual(client.get("/api/state").json()["body_net"], {"path": str(self.checkpoint.resolve()), "exists": False})
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b"net")
        self.assertEqual(client.get("/api/state").json()["body_net"]["exists"], True)
        self.assertEqual(self.app(None).get("/api/state").json()["body_net"], {"path": None, "exists": False})

    def test_workspace_rows_and_payload_carry_the_fit_network(self):
        client = self.app(self.checkpoint)
        expected = {"stage_with": ("with", "/nets/best.ckpt"), "stage_without": ("without", None),
                    "imported_with": ("with", None), "imported_without": ("without", None), "fresh": ("not_fitted", None)}
        for rows in (client.get("/api/workspaces").json(), client.get("/api/state").json()["workspaces"]):
            self.assertEqual({r["name"]: (r["fit_network"], r["fit_network_checkpoint"]) for r in rows}, expected)
        entry = client.get("/api/workspaces/stage_with").json()["entry"]
        self.assertEqual((entry["fit_network"], entry["fit_network_checkpoint"]), ("with", "/nets/best.ckpt"))
        self.assertEqual(client.get("/api/workspaces/imported_without").json()["entry"]["fit_network"], "without")

    def test_propagate_and_track_read_the_network_from_the_fit_summary(self):
        """``workspace_predictions`` (propagate, track) follows the fit's ``fit_params.body_net``, not the stage's own params."""

        workspaces = self.root / "workspaces"
        params = pipeline.SegmentParams(dataset_root=str(self.root / "cache"))
        with mock.patch.object(pipeline, "field_predictor") as predictor:
            with workspace_predictions(Workspace.open(workspaces / "stage_with"), params, "cpu") as predictions_of:
                self.assertIsNotNone(predictions_of)
            self.assertEqual(predictor.call_args.args[0], "/nets/best.ckpt")
            predictor.reset_mock()
            with workspace_predictions(Workspace.open(workspaces / "stage_without"), params, "cpu") as predictions_of:
                self.assertIsNone(predictions_of)
            predictor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
