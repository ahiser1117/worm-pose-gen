"""HTTP contracts of /api/training and the train/evaluate job kinds: the picker table, details, plans, jobs, runs."""

from pathlib import Path
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

from worm_pose_gen import library, model_eval
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.library.roots import write_json

sys.path.insert(0, str(Path(__file__).parent))
from test_model_training import make_library  # noqa: E402


class TrainingApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_library(self.root)
        config = AppConfig(
            workspaces_root=self.root / "workspaces", recording_roots=(), poses_root=self.root / "poses",
            dataset_root=self.root / "cache", checkpoint=None, prior_cache=None, notes=self.root / "notes.json",
            device="cpu", gpus=(0,), lab_library=self.libraries.lab, library=self.libraries.personal,
        )
        app = create_app(config)
        # No lifespan: jobs stay queued, so submissions can be inspected.
        self.state = app.state.app_state
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.state.close)
        self.addCleanup(self.client.close)

    def get(self, url, status=200, **params):
        response = self.client.get(url, params=params)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def post(self, url, payload, status=200):
        response = self.client.post(url, json=payload)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def test_picker_rows(self):
        model_eval.evaluate(self.libraries, "lab:body", "lab:rig-v1", device="cpu")
        table = self.get("/api/training/models", setup="lab:rig")
        self.assertEqual((table["benchmark"], table["others"]), ("lab:rig-v1", False))
        rows = {row["ref"]: row for row in table["models"]}
        body, seg = rows["lab:body"], rows["lab:seg"]
        self.assertEqual((body["default_for"], body["roles"], body["scope"]), (["body"], ["mask", "body"], "lab"))
        self.assertEqual(body["inputs"]["lags_s"], [0.05, 0.1])
        self.assertEqual(body["missing"], [])
        self.assertEqual(body["evaluation"]["labels"], 1)
        self.assertIsNone(seg["evaluation"])
        self.assertIn("no head/tail: orientation from body taper only", seg["missing"])
        self.assertEqual(seg["default_for"], ["mask"])
        self.get("/api/training/models", status=400, setup="lab:rig", benchmark="lab:nope")
        self.get("/api/training/models", status=404, setup="lab:missing")
        # A new microscope lists the lab models of the other setups with their numbers there.
        library.create_setup(self.libraries, "scope2", name="Scope 2")
        other = self.get("/api/training/models", setup="mine:scope2")
        self.assertTrue(other["others"])
        self.assertIsNone(other["benchmark"])
        found = {row["ref"]: row for row in other["models"]}
        self.assertEqual(found["lab:body"]["setup_name"], "Rig")
        self.assertEqual(found["lab:body"]["evaluation"]["benchmark"], "lab:rig-v1")

    def test_details_and_overlays(self):
        model_eval.evaluate(self.libraries, "lab:body", "lab:rig-v1", device="cpu")
        details = self.get("/api/training/models/lab:body")
        self.assertEqual(details["card"]["name"], "body")
        evaluation = details["evaluations"]["lab:rig-v1"]
        self.assertEqual(len(evaluation["rows"]), 1)
        url = evaluation["worst"][0]["url"]
        response = self.client.get(url)
        self.assertEqual((response.status_code, response.headers["content-type"]), (200, "image/png"))
        self.assertEqual(self.client.get(url.replace("worst1", "worst9")).status_code, 404)
        self.assertEqual(self.client.get("/api/training/models/lab:body/overlays/lab:rig-v1/..%2Fx.png").status_code, 404)
        self.assertEqual(details["curve"], [])
        write_json(library.training_dir(self.libraries, "lab:body") / "labels.json",
                   {"train": [r.identity for r in library.labels(self.libraries, ["lab:base"], "train")]})
        labels = self.get("/api/training/models/lab:body")["labels"]
        self.assertEqual(labels, [{"dataset": "lab:base", "recording": "rec-train", "train": 2, "val": 0}])
        self.get("/api/training/models/lab:none", status=404)

    def test_plan_and_schema(self):
        plan = self.post("/api/training/plan", {"setup": "lab:rig", "datasets": ["lab:base"], "start_from": "lab:body"})
        self.assertEqual((plan["kind"], plan["lags"], plan["train"], plan["val"], plan["name"]), ("body_net", [1, 2], 2, 1, "base-2"))
        self.post("/api/training/plan", {"setup": "lab:rig", "datasets": []}, status=400)
        schema = self.get("/api/training/schema")
        names = [p["name"] for p in schema["parameters"]]
        self.assertEqual(names[:3], ["max_epochs", "learning_rate", "batch_size"])
        self.assertEqual(schema["contexts"], {"none": [], "short": [1, 4, 16]})

    def test_train_and_evaluate_jobs(self):
        job = self.post("/api/jobs", {"kind": "train", "setup": "lab:rig", "datasets": ["lab:base"], "start_from": "lab:body",
                                      "params": {"max_epochs": 2}, "notes": "n", "run_on": "local"})
        self.assertEqual((job["spec"]["kind"], job["state"], job["spec"]["params"]["name"]), ("train", "queued", "base-2"))
        command = job["command"]
        self.assertEqual(command[1:3], ["-m", "worm_pose_gen.model_training"])
        self.assertIn("--max-epochs", command)
        self.assertEqual(command[command.index("--max-epochs") + 1], "2")
        self.assertEqual(command[command.index("--lab-library") + 1], str(self.libraries.lab))
        # The queued run claims its name.
        again = self.post("/api/training/plan", {"setup": "lab:rig", "datasets": ["lab:base"]})
        self.assertEqual(again["name"], "base-2-2")
        self.post("/api/jobs", {"kind": "train", "setup": "lab:rig", "datasets": ["lab:base"], "name": "base-2"}, status=400)
        runs = self.get("/api/training/runs", setup="lab:rig")["runs"]
        self.assertEqual([r["id"] for r in runs], [job["id"]])
        self.post(f"/api/training/runs/{job['id']}/dismiss", {}, status=409)
        self.state.runner.cancel(job["id"])
        self.assertEqual(self.get("/api/training/runs", setup="lab:rig")["runs"], [])  # a cancelled run is not shown
        evaluation = self.post("/api/jobs", {"kind": "evaluate", "model": "lab:seg", "benchmark": "lab:rig-v1"})
        self.assertEqual(evaluation["command"][1:3], ["-m", "worm_pose_gen.model_eval"])
        self.post("/api/jobs", {"kind": "evaluate", "model": "lab:seg", "benchmark": "lab:none"}, status=404)

    def test_fill_missing_evaluations_once(self):
        filled = self.post("/api/training/evaluations", {"setup": "lab:rig"})
        self.assertEqual(sorted(j["spec"]["params"]["model"] for j in filled["queued"]), ["lab:body", "lab:seg"])
        self.assertEqual(self.post("/api/training/evaluations", {"setup": "lab:rig"})["queued"], [])
        table = self.get("/api/training/models", setup="lab:rig")
        self.assertTrue(all(row["evaluating"]["state"] == "queued" for row in table["models"]))

    def test_failed_run_shows_until_dismissed(self):
        job = self.post("/api/jobs", {"kind": "train", "setup": "lab:rig", "datasets": ["lab:base"], "start_from": "lab:seg"})
        record = self.state.runner.get(job["id"])
        record.state, record.error = "failed", "CUDA out of memory"
        runs = self.get("/api/training/runs", setup="lab:rig")["runs"]
        self.assertEqual((runs[0]["state"], runs[0]["error"]), ("failed", "CUDA out of memory"))
        self.post(f"/api/training/runs/{job['id']}/dismiss", {})
        self.assertEqual(self.get("/api/training/runs", setup="lab:rig")["runs"], [])
        self.assertEqual(self.get("/api/training/runs", setup="lab:other")["runs"], [])


if __name__ == "__main__":
    unittest.main()
