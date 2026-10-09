"""The Workspace page's endpoints (``app/routers/analysis.py``): the Recordings screen, Analyse, status, kymograph, and the refit a mask edit keeps.

The analysis runs for real on the six-frame synthetic recording of
``tests/test_pipeline.py``: ``analysis.analysis_params`` is patched to the
small CPU schedule (no network, no flat field) and ``analysis.analysis_command``
to this interpreter on the source tree with ``--device cpu``, so the job
process is the same ``pipeline --stages`` an app on a GPU starts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from worm_pose_gen import edits, fixes, library, pipeline
from worm_pose_gen.app import AppConfig, analysis, create_app
from worm_pose_gen.jobs import JobRecord, JobSpec
from worm_pose_gen.library import Libraries
from worm_pose_gen.workspace import list_workspaces

from tests.slow import slow
from tests.test_fixes import posed_workspace
from tests.test_fixes_api import _cpu_fix_command
from tests.test_pipeline import FIT_PARAMS, FRAMES, SEGMENT_PARAMS, _write_recording

SRC = str(Path(__file__).resolve().parents[1] / "src")
CPU_PARAMS = {**SEGMENT_PARAMS, **FIT_PARAMS, "min_score": 9, "jump_seeds": False, "beam": 2}
# The real parameters, before the class patches them to the CPU ones.
ORIGINAL_ANALYSIS_PARAMS = analysis.analysis_params


def cpu_analysis_params(app, setup, models, mask_source="segmenter") -> dict:
    return {**CPU_PARAMS, "models_seen": {role: None if m is None else m["ref"] for role, m in models.items()}, "mask_source_seen": mask_source}


def cpu_analysis_command(workspace_path, stages, params) -> list[str]:
    argv = ["--workspace", str(workspace_path), "--stages", ",".join(stages), "--params", json.dumps(params), "--device", "cpu"]
    code = f"import sys; sys.path.insert(0, {SRC!r}); from worm_pose_gen.pipeline import main; sys.exit(main({argv!r}))"
    return [sys.executable, "-c", code]


def write_library(root: Path, recordings: Path) -> Libraries:
    """A lab library with setup ``nir`` over ``recordings``, a segmenter (its default mask model) and a body-field net."""

    lab, personal = root / "lab", root / "mine"
    library.write_setup(lab, "nir", name="NIR", fps=20.0, pixel_size_um=2.5, recording_roots=[str(recordings)], defaults={"mask": "lab:seg"})
    weights = root / "w.ckpt"
    weights.write_bytes(b"w")
    inputs = library.make_inputs([], fps=20.0, pixel_size_um=2.5)
    library.write_model(lab, "seg", {"name": "seg-284", "kind": "segmenter", "setup": "lab:nir", "outputs": ["mask"], "inputs": inputs}, weights)
    library.write_model(lab, "body", {"name": "body-lags3", "kind": "body_net", "setup": "lab:nir",
                                      "outputs": ["mask", "ap", "head", "tail", "overlap"], "inputs": inputs}, weights)
    return Libraries(lab=lab, personal=personal)


class WorkspacePageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls._directory.name)
        recordings = root / "recordings"
        recordings.mkdir()
        cls.recording = recordings / "2026-01-02-03.h5"
        _write_recording(cls.recording)
        cls.libraries = write_library(root, recordings)
        config = AppConfig(
            workspaces_root=root / "workspaces", dataset_root=root / "dataset", gpus=(), device="cpu", job_interval=0.1,
            lab_library=cls.libraries.lab, library=cls.libraries.personal,
        )
        cls.client = TestClient(create_app(config), raise_server_exceptions=False)
        cls.client.__enter__()
        patches = (mock.patch.object(analysis, "analysis_params", cpu_analysis_params), mock.patch.object(analysis, "analysis_command", cpu_analysis_command))
        for patch in patches:
            patch.start()
            cls.addClassCleanup(patch.stop)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls._directory.cleanup()

    def call(self, method: str, path: str, payload: dict | None = None, status: int = 200) -> dict:
        response = self.client.request(method, path, json=payload)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def wait_for_job(self, job_id: str, timeout: float = 600.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.call("GET", f"/api/jobs/{job_id}")
            if record["state"] in ("done", "failed", "cancelled"):
                return record
            time.sleep(0.2)
        self.fail(f"job {job_id} did not finish")

    @slow
    def test_analyse_a_recording_end_to_end(self) -> None:
        home = self.call("GET", "/api/home")
        self.assertEqual(home["setup"]["ref"], "lab:nir")
        row = next(r for r in home["recordings"] if r["id"] == "2026-01-02-03")
        self.assertEqual((row["frames"], row["workspace"], row["status"]), (FRAMES, None, None))

        # Models are checked before anything is made: a mask role needs a segmenter.
        self.call("POST", "/api/analyse", {"path": str(self.recording), "models": {"mask": "lab:body"}}, 400)
        self.call("POST", "/api/analyse", {"path": str(self.recording), "models": {"body": "lab:seg"}}, 400)
        self.call("POST", "/api/analyse", {"path": str(self.root / "elsewhere.h5")}, 404)
        self.assertEqual(list_workspaces(self.root / "workspaces"), [])

        started = self.call("POST", "/api/analyse", {"path": str(self.recording), "models": {"body": "lab:body"}})
        name = started["workspace"]
        self.assertEqual(name, "2026-01-02-03")
        job = started["job"]
        self.assertEqual((job["spec"]["kind"], job["spec"]["params"]["stages"]), ("analyse", list(pipeline.DEFAULT_STAGES)))
        self.assertNotIn("export", job["spec"]["params"]["stages"])
        self.assertEqual(job["spec"]["params"]["params"]["models_seen"], {"mask": "lab:seg", "body": "lab:body"})
        self.assertEqual(job["spec"]["params"]["params"]["mask_source_seen"], "segmenter")
        self.assertIn(started["status"]["state"], ("queued", "analysing"))
        # One analysis at a time per workspace.
        self.call("POST", "/api/analyse", {"path": str(self.recording)}, 409)

        record = self.wait_for_job(job["id"])
        self.assertEqual(record["state"], "done", record.get("error"))
        self.assertEqual(record["message"], "analysis: done")
        done = self.call("GET", f"/api/workspaces/{name}/status")["analysis"]["stages"]
        self.assertEqual([(s["stage"], s["state"]) for s in done], [(stage, "done") for stage in pipeline.DEFAULT_STAGES])

        info = self.call("GET", f"/api/workspaces/{name}")
        self.assertEqual(info["frames"], [0, FRAMES - 1])
        self.assertEqual(info["settings"]["setup"], "lab:nir")
        self.assertEqual(info["settings"]["models"], {"mask": "lab:seg", "body": "lab:body"})
        self.assertEqual(info["settings"]["checkpoint"], str(self.libraries.lab / "models" / "seg" / "weights.ckpt"))
        self.assertEqual(info["settings"]["mask_source"], "segmenter")
        self.assertTrue(info["summary"]["fitted"] > 0)
        self.assertFalse((self.root / "workspaces" / name / "exports").exists())

        status = self.call("GET", f"/api/workspaces/{name}/status")
        self.assertEqual((status["state"], status["issues"], status["analysed"]), ("analysed", None, True))
        self.assertEqual(status["models"]["mask"]["name"], "seg-284")
        self.assertEqual(status["models"]["body"]["name"], "body-lags3")
        self.assertEqual(status["mask_source"], "segmenter")
        self.assertEqual((status["setup"]["pixel_size_um"], status["setup"]["fps"]), (2.5, 20.0))
        self.assertEqual(status["analysis"]["state"], "done")

        # Opening the issues stores their counts for the Recordings screen.
        issues = self.call("GET", f"/api/workspaces/{name}/issues")
        status = self.call("GET", f"/api/workspaces/{name}/status")
        self.assertEqual(status["issues"], issues["summary"])
        self.assertEqual(status["state"], "issues" if issues["summary"]["unreviewed"] else "reviewed")
        for issue in issues["issues"]:
            issues = self.call("POST", f"/api/workspaces/{name}/issues/review", {"first": issue["frames"][0], "last": issue["frames"][1], "revision": issues["revision"]})
        self.assertEqual(self.call("GET", f"/api/workspaces/{name}/status")["state"], "reviewed")
        self.call("POST", f"/api/workspaces/{name}/export", {"pixel_size_um": 2.5, "fps": 20.0, "setup": "lab:nir"})
        self.assertEqual(self.call("GET", f"/api/workspaces/{name}/status")["state"], "exported")

        opened = self.call("POST", f"/api/workspaces/{name}/opened")["last_opened"]
        row = next(r for r in self.call("GET", "/api/home", None)["recordings"] if r["id"] == "2026-01-02-03")
        self.assertEqual((row["workspace"], row["status"]["state"], row["status"]["last_opened"]), (name, "exported", opened))

        kymograph = self.call("GET", f"/api/workspaces/{name}/kymograph")
        self.assertEqual((kymograph["rows"], kymograph["frames"], kymograph["step"]), (FRAMES, [0, FRAMES - 1], 1))
        self.assertTrue(kymograph["image"].startswith("data:image/png;base64,"))

        # Analysing again reuses the workspace.
        again = self.call("POST", "/api/analyse", {"path": str(self.recording), "stages": ["ambiguity"]})
        self.assertEqual(again["workspace"], name)
        self.assertEqual(self.wait_for_job(again["job"]["id"])["state"], "done")
        self.assertEqual(self.call("GET", f"/api/workspaces/{name}")["settings"]["models"], {"mask": "lab:seg", "body": None})
        self.call("POST", "/api/analyse", {"path": str(self.recording), "stages": ["export"]}, 400)

        # Masks from the body model: it is required, and the workspace records that no mask model was used.
        self.call("POST", "/api/analyse", {"path": str(self.recording), "mask_source": "body_net", "stages": ["ambiguity"]}, 400)
        self.call("POST", "/api/analyse", {"path": str(self.recording), "mask_source": "network", "models": {"body": "lab:body"}}, 400)
        net = self.call("POST", "/api/analyse", {"path": str(self.recording), "mask_source": "body_net", "models": {"body": "lab:body"}, "stages": ["ambiguity"]})
        self.assertEqual(net["job"]["spec"]["params"]["params"]["models_seen"], {"mask": None, "body": "lab:body"})
        self.assertEqual(net["job"]["spec"]["params"]["params"]["mask_source_seen"], "body_net")
        self.assertEqual(self.wait_for_job(net["job"]["id"])["state"], "done")
        settings = self.call("GET", f"/api/workspaces/{name}")["settings"]
        self.assertEqual((settings["mask_source"], settings["models"], settings["checkpoint"]), ("body_net", {"mask": None, "body": "lab:body"}, None))
        self.assertEqual(self.call("GET", f"/api/workspaces/{name}/status")["mask_source"], "body_net")

    def test_analysis_params_of_each_mask_source(self) -> None:
        setup = library.get_setup(self.libraries, "lab:nir")
        app = self.client.app.state.app_state
        segmenter = analysis.resolve_models(self.libraries, setup, {"body": "lab:body"})
        body = analysis.resolve_models(self.libraries, setup, {"mask": "lab:seg", "body": "lab:body"}, "body_net")
        self.assertEqual((segmenter["mask"]["ref"], body["mask"]), ("lab:seg", None))
        params = ORIGINAL_ANALYSIS_PARAMS(app, setup, body, "body_net")
        self.assertEqual((params["mask_source"], params["checkpoint"]), ("body_net", None))
        self.assertEqual(params["body_net"], str(self.libraries.lab / "models" / "body" / "weights.ckpt"))
        self.assertEqual(ORIGINAL_ANALYSIS_PARAMS(app, setup, segmenter)["mask_source"], "segmenter")

    def test_stage_progress(self) -> None:
        def record(state: str, progress: float, message: str) -> JobRecord:
            spec = JobSpec(kind=analysis.ANALYSE_JOB_KIND, params={"stages": ["segment", "prior", "fit", "track"]})
            return JobRecord(id="1", spec=spec, state=state, progress=progress, message=message)

        def stages(job: JobRecord) -> list[tuple[str, float]]:
            return [(s["state"], round(s["progress"], 3)) for s in analysis.stage_progress(job)]

        self.assertEqual(stages(record("running", 0.625, "fit: frame 3")), [("done", 1.0), ("done", 1.0), ("running", 0.5), ("waiting", 0.0)])
        self.assertEqual(stages(record("failed", 0.3, "prior: estimating")), [("done", 1.0), ("failed", 0.2), ("waiting", 0.0), ("waiting", 0.0)])
        # Before the job reports a stage (queued, waiting for SLURM) the overall progress is all there is.
        self.assertEqual(stages(record("queued", 0.0, "submitted to SLURM as job 7")), [("waiting", 0.0)] * 4)
        self.assertEqual(stages(record("done", 1.0, "analysis: done")), [("done", 1.0)] * 4)

    def test_kymograph_encoding(self) -> None:
        # A circle of radius r has curvature 1/r; times its length 2*pi*r that is 2*pi everywhere.
        angle = np.linspace(0, 2 * np.pi, 200)
        circle = np.stack((50 + 10 * np.cos(angle), 50 + 10 * np.sin(angle)), axis=1)
        values = analysis.curvature_kymograph(np.stack((circle, circle[::-1])), np.array([2 * np.pi * 10] * 2), np.array([True, False]))
        self.assertEqual(values.shape, (200, 2))
        np.testing.assert_allclose(values[5:-5, 0], 2 * np.pi, rtol=1e-3)
        self.assertTrue(np.isnan(values[:, 1]).all())
        encoded = analysis.encode_kymograph(np.array([[np.nan, 0.0, 100.0, -100.0]]))
        self.assertEqual(encoded.tolist(), [[0, 128, 255, 1]])


class MaskRefitKeepTests(unittest.TestCase):
    def test_a_refit_with_keep_installs_its_own_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = posed_workspace(root, "synthetic")
            state = workspace.load_state()
            for row in (4, 5):
                edits._reverse_row(state, row)
            workspace.save_state(state)
            config = AppConfig(
                workspaces_root=root / "workspaces", dataset_root=root / "dataset", gpus=(), device="cpu", job_interval=0.1,
                lab_library=root / "lab", library=root / "mine",
            )
            with TestClient(create_app(config), raise_server_exceptions=False) as client, mock.patch.object(fixes, "fix_command", _cpu_fix_command):
                answer = client.post("/api/workspaces/synthetic/fixes/refit", json={"first": 4, "last": 5, "keep": True}).json()
                self.assertTrue(answer["job"]["spec"]["params"]["keep"])
                deadline = time.monotonic() + 600
                while (record := client.get(f"/api/jobs/{answer['job']['id']}").json())["state"] not in ("done", "failed", "cancelled"):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.2)
                self.assertEqual(record["state"], "done", record.get("error"))
                self.assertTrue(record["result"]["kept"].startswith("e"))
                self.assertEqual(client.get("/api/workspaces/synthetic/fixes/previews").json(), [])
                listed = client.get("/api/workspaces/synthetic/fixes").json()["fixes"]
                self.assertEqual((listed[0]["id"], listed[0]["kind"]), (record["result"]["kept"], "refit"))


if __name__ == "__main__":
    unittest.main()
