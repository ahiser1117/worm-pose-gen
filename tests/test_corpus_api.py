"""HTTP contracts for independent corpus edits, snapshots and checkpoint selection."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient
import h5py
import numpy as np

from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.corpus import CorpusStore
from worm_pose_gen.label_app import data_url, mask_to_png_values, decode_mask_data_url
from worm_pose_gen.workspace import Workspace


class CorpusApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = AppConfig(workspaces_root=self.root / "workspaces", recording_roots=(self.root,),
                                poses_root=self.root / "poses", corpus_root=self.root / "my-corpus",
                                checkpoints_root=self.root / "my-checkpoints", dataset_root=self.root / "flat-field-cache",
                                checkpoint=None, prior_cache=None, notes=self.root / "notes.json", device="cpu", gpus=())
        self.app = create_app(self.config)
        # No lifespan: submission is tested while the queue remains stopped.
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.state = self.app.state.app_state
        self.image = np.full((64, 64), 100, np.uint8)
        self.label = np.zeros_like(self.image)
        self.label[20:40, 25:35] = 1
        self.label[19, 25:35] = 255

    def tearDown(self):
        self.state.close()
        self.client.close()
        self.directory.cleanup()

    def test_null_checkpoint_and_corpus_configuration(self):
        response = self.client.get("/api/checkpoints")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["checkpoints"], [])
        corpus = self.client.get("/api/corpus").json()
        self.assertEqual(corpus["root"], str(self.config.corpus_root))
        self.assertEqual(corpus["counts"], {"train": 0, "val": 0, "test": 0})
        self.assertEqual(corpus["training"]["defaults"]["epochs"], 30)
        response = self.client.post("/api/jobs", json={"kind": "fine_tune"})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("checkpoint", response.json()["error"])

    def test_save_browse_update_delete_and_independent_workspace(self):
        recording = self.root / "video.h5"
        with h5py.File(recording, "w") as handle:
            handle.create_dataset("/other", data=np.stack([self.image] * 3))
        workspace = Workspace.create(self.config.workspaces_root, "demo", recording, 0, 2, 1,
                                     settings={"dataset": "/other"})
        workspace.set_override_mask(0, self.label)
        view = self.state.view("demo")
        with mock.patch.object(view.source, "corrected", return_value=(self.image + 3, self.image)):
            response = self.client.post("/api/corpus/labels", json={"workspace": "demo", "frame": 0, "split": "val"})
        self.assertEqual(response.status_code, 200, response.text)
        sample = response.json()["sample"]
        self.assertEqual(sample["dataset_path"], "/other")
        self.assertEqual(sample["split"], "val")
        self.assertFalse(self.config.dataset_root.joinpath("index.json").exists())
        url = "/api/corpus/labels/" + sample["sample_id"]
        read = self.client.get(url)
        self.assertEqual(read.status_code, 200, read.text)
        np.testing.assert_array_equal(decode_mask_data_url(read.json()["label"], (64, 64)), self.label)
        edited = self.label.copy()
        edited[0, 0] = 1
        response = self.client.put(url, json={"mask": data_url(mask_to_png_values(edited)), "revision": 1})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["sample"]["revision"], 2)
        np.testing.assert_array_equal(workspace.get_override_mask(0), self.label)
        stale = self.client.put(url, json={"mask": data_url(mask_to_png_values(edited)), "revision": 1})
        self.assertEqual(stale.status_code, 400, stale.text)
        self.assertEqual(self.client.delete(url).status_code, 200)
        self.assertEqual(self.client.get(url).status_code, 404)
        np.testing.assert_array_equal(workspace.get_override_mask(0), self.label)
        with mock.patch.object(view.source, "corrected", return_value=(self.image + 3, self.image)):
            saved = self.client.post("/api/corpus/labels", json={"workspace": "demo", "frame": 0}).json()["sample"]
        self.assertEqual((saved["split"], saved["revision"]), ("val", 3))

    def test_checkpoint_selection_and_submission_freeze_requested_initialization(self):
        store = CorpusStore(self.config.corpus_root)
        for frame, split in enumerate(("train", "val")):
            store.save_frame(self.root / "video.h5", "/img_nir", frame, self.image, self.label, split=split)
        base = self.root / "base.ckpt"
        base.write_bytes(b"configured initial weights")
        self.config.checkpoint = base
        requested = self.root / "chosen.ckpt"
        requested.write_bytes(b"requested initial weights")
        response = self.client.post("/api/jobs", json={"kind": "fine_tune", "checkpoint": str(requested), "params": {"device": "cpu"}})
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        run_dir = Path(job["spec"]["params"]["run_dir"])
        self.assertEqual((run_dir / "init.ckpt").read_bytes(), requested.read_bytes())
        before = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(before["init_checkpoint"]["sha256"], hashlib.sha256(requested.read_bytes()).hexdigest())
        self.assertEqual(job["state"], "queued")
        self.assertEqual(len(self.client.get("/api/checkpoints").json()["checkpoints"]), 1)
        (run_dir / "best.ckpt").write_bytes(b"trained output")
        # A partially written run remains unavailable until completion.
        self.assertEqual(len(self.client.get("/api/checkpoints").json()["checkpoints"]), 1)
        before["state"] = "completed"
        (run_dir / "run.json").write_text(json.dumps(before))
        entries = self.client.get("/api/checkpoints").json()["checkpoints"]
        self.assertEqual(len(entries), 2)
        workspace = Workspace.create(self.config.workspaces_root, "demo", self.root / "missing.h5", 0, 0, 1)
        historical = {"checkpoint": {"path": str(base), "sha256": "old-fingerprint"}, "recording": str(self.root / "missing.h5")}
        (workspace.path / "summary.json").write_text(json.dumps(historical))
        response = self.client.post("/api/workspaces/demo/checkpoint", json={"checkpoint": entries[1]["id"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(Workspace.open(workspace.path).info.settings["checkpoint"], entries[1]["path"])
        self.assertEqual(json.loads((workspace.path / "summary.json").read_text()), historical)
