"""The FastAPI pose app over a synthetic recording, an imported run, and a job runner on a fake GPU pool.

Jobs run real subprocesses: a python one-liner for the ``command`` kind, and
the pipeline's segment stage on the CPU for the ``stage`` kind (with
``pipeline.stage_command`` patched so the child never touches a GPU).
``RegionTests`` (Phase 3) fits the six-frame body of ``tests/test_pipeline.py``
once and runs region jobs on it the same way, with ``pipeline.region_command``
patched.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

import h5py
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from worm_pose_gen import algorithms, edits, pipeline
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.workspace import Workspace

from tests import test_pipeline
from tests.test_pose_viewer import FRAMES, HEIGHT, WIDTH, _write_recording, _write_run

SRC = str(Path(__file__).resolve().parents[1] / "src")
RUN_NAME = "2026-09-06T10-00-00Z_demo"
SEGMENT_PARAMS = {"checkpoint": None, "flat_field": False, "min_worm_pixels": 200, "slab": 4}


def _cpu_stage_command(workspace_path: Path | str, stage: str, params: dict | None) -> list[str]:
    """``pipeline.stage_command`` for tests: the same CLI, this interpreter, the CPU."""

    argv = ["--workspace", str(workspace_path), "--stage", stage, "--params", json.dumps(params or {}), "--device", "cpu"]
    code = f"import sys; sys.path.insert(0, {SRC!r}); from worm_pose_gen.pipeline import main; sys.exit(main({argv!r}))"
    return [sys.executable, "-c", code]


def _cpu_region_command(workspace_path: Path | str, spec: dict) -> list[str]:
    """``pipeline.region_command`` for tests: the same CLI, this interpreter, the CPU."""

    payload = {k: v for k, v in spec.items() if k in pipeline.REGION_SPEC_KEYS and v is not None}
    argv = ["--workspace", str(workspace_path), "--region-run", json.dumps(payload), "--device", "cpu"]
    code = f"import sys; sys.path.insert(0, {SRC!r}); from worm_pose_gen.pipeline import main; sys.exit(main({argv!r}))"
    return [sys.executable, "-c", code]


def _reporting_command(message: str, result: dict) -> list[str]:
    code = (
        f"import sys, os; sys.path.insert(0, {SRC!r}); from worm_pose_gen.jobs import report_progress\n"
        "report_progress(0.5, 'half way')\nprint('hello from the job')\n"
        f"report_progress(1.0, {message!r}, {{**{result!r}, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}})\n"
    )
    return [sys.executable, "-c", code]


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.root = root
        cls.recording = root / "recordings" / "rec-a.h5"
        cls.recording.parent.mkdir()
        _write_recording(cls.recording)
        _write_run(root / "runs" / RUN_NAME, cls.recording)
        config = AppConfig(
            workspaces_root=root / "workspaces", recording_roots=(root / "recordings",), poses_root=root / "runs",
            dataset_root=root / "dataset", checkpoint=None, prior_cache=None, notes=root / "notes.json", gpus=(0,), device="cpu",
            job_interval=0.1,
        )
        cls.app = create_app(config)
        cls.client = TestClient(cls.app, raise_server_exceptions=False)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls._directory.cleanup()

    # ----- helpers

    def get(self, path: str, status: int = 200) -> dict | list:
        response = self.client.get(path)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def post(self, path: str, payload: dict, status: int = 200) -> dict | list:
        response = self.client.post(path, json=payload)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def wait_for_job(self, job_id: str, timeout: float = 120.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(f"/api/jobs/{job_id}")
            if record["state"] in ("done", "failed", "cancelled"):
                return record
            time.sleep(0.1)
        self.fail(f"job {job_id} did not finish: {self.get(f'/api/jobs/{job_id}')}")

    def imported(self) -> dict:
        """The demo run imported as a workspace (once per class)."""

        if not self.get("/api/workspaces") or not any(w["name"] == RUN_NAME for w in self.get("/api/workspaces")):
            self.post("/api/workspaces/import", {"run": RUN_NAME})
        return self.get(f"/api/workspaces/{RUN_NAME}")

    # ----- tests

    def test_static_files_and_state(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn("Pose viewer", page.text)
        script = self.client.get("/app.js")
        self.assertIn("javascript", script.headers["content-type"])
        self.assertIn("use strict", script.text)
        self.assertIn("text/css", self.client.get("/style.css").headers["content-type"])
        self.assertEqual(self.client.get("/static/app.js").text, self.client.get("/app.js").text)
        self.assertEqual(self.client.get("/static/nope.js").status_code, 404)
        self.assertEqual(self.client.get("/static/nope.js").json(), {"error": "no static file 'nope.js'"})
        self.assertEqual(self.client.get("/nowhere").json(), {"error": "Not Found"})

        state = self.get("/api/state")
        self.assertEqual([r["name"] for r in state["runs"]], [RUN_NAME])
        self.assertEqual(state["recording_roots"], [str(self.root / "recordings")])
        self.assertEqual(state["gpus"], [0])
        self.assertIsInstance(state["jobs_running"], int)
        self.assertIn("flag_groups", state)
        self.assertIsInstance(state["workspaces"], list)
        rescanned = self.get("/api/state?rescan=1")
        self.assertEqual(rescanned["added"], 0)
        # The old run endpoints still serve run directories.
        run = self.get(f"/api/run?name={RUN_NAME}")
        self.assertEqual(run["series"]["frame_index"], list(range(FRAMES)))
        frame = self.get(f"/api/frame?run={RUN_NAME}&frame=2&detail=light")
        self.assertEqual(sorted(frame["layers"]), ["image", "tube"])
        self.assertEqual(self.get(f"/api/frame?run={RUN_NAME}&frame=2&detail=medium", 400)["error"], "detail must be 'full' or 'light'")
        self.assertIn("frame", self.get(f"/api/frame?run={RUN_NAME}&frame=x", 400)["error"])
        self.assertEqual(self.get("/api/run?name=missing", 404), {"error": "unknown workspace 'missing'"})
        self.assertTrue(self.get(f"/api/pose?run={RUN_NAME}&frame=1")["present"])

    def test_recordings_and_thumbnails(self) -> None:
        self.imported()
        listed = self.get("/api/recordings")
        self.assertEqual([r["name"] for r in listed], ["rec-a"])
        entry = listed[0]
        self.assertEqual((entry["frames"], entry["height"], entry["width"]), (FRAMES, HEIGHT, WIDTH))
        self.assertTrue(entry["readable"])
        self.assertEqual(entry["runs"], [RUN_NAME])
        self.assertIn(RUN_NAME, entry["workspaces"])
        self.assertTrue((self.root / "workspaces" / "recordings_index.json").exists())
        self.assertEqual(len(self.get("/api/recordings?rescan=1")), 1)
        response = self.client.get(f"/api/recordings/thumbnail?path={self.recording}&frame=1&scale=0.5")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertTrue(response.content.startswith(b"\x89PNG"))
        self.assertEqual(self.client.get(f"/api/recordings/thumbnail?path={self.recording}&frame=99").status_code, 400)
        self.assertEqual(self.client.get(f"/api/recordings/thumbnail?path={self.root}/recordings/none.h5&frame=0").status_code, 404)
        # Only files under the configured roots are served, and never larger than the frame.
        outside = self.root / "elsewhere.h5"
        shutil.copyfile(self.recording, outside)
        self.assertIn("recording roots", self.client.get(f"/api/recordings/thumbnail?path={outside}&frame=0").json()["error"])
        self.assertEqual(self.client.get(f"/api/recordings/thumbnail?path={outside}&frame=0").status_code, 404)
        self.assertEqual(self.client.get("/api/recordings/thumbnail?path=/etc/passwd&frame=0").status_code, 404)
        huge = self.client.get(f"/api/recordings/thumbnail?path={self.recording}&frame=0&scale=40")
        self.assertEqual(huge.status_code, 200)
        self.assertEqual(Image.open(io.BytesIO(huge.content)).size, (WIDTH, HEIGHT))
        self.assertEqual(self.client.get(f"/api/recordings/thumbnail?path={self.recording}&frame=0&scale=0").status_code, 400)
        garbage = self.root / "recordings" / "garbage.h5"
        garbage.write_bytes(b"not hdf5" * 32)
        try:
            response = self.client.get(f"/api/recordings/thumbnail?path={garbage}&frame=0")
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["error"], "garbage.h5 cannot be read as a recording")
        finally:
            garbage.unlink()

    def test_imported_workspace_serves_the_viewer_payloads(self) -> None:
        payload = self.imported()
        self.assertEqual(payload["name"], RUN_NAME)
        self.assertEqual(payload["kind"], "workspace")
        self.assertEqual(payload["frames"], [0, FRAMES - 1])
        self.assertEqual(payload["imported_runs"], [RUN_NAME])
        self.assertEqual(payload["summary"]["fitted"], FRAMES)
        self.assertEqual(payload["provenance_counts"], {"chain_forward": 1, "independent_fit": FRAMES - 1})
        self.assertEqual(payload["provenance"]["algorithm"][-1], "chain_forward")
        self.assertEqual(payload["provenance"]["job"][0], f"import:{RUN_NAME}")
        self.assertTrue(payload["has_hypotheses"])
        self.assertTrue(payload["has_independent_pose"])
        self.assertEqual(payload["image_shape"], [HEIGHT, WIDTH])
        self.assertEqual(payload["stretches"], [[FRAMES - 3, FRAMES - 1]])
        series = payload["series"]
        self.assertEqual(series["frame_index"], list(range(FRAMES)))
        self.assertEqual(series["classification"], ["clean"] * (FRAMES - 1) + ["ambiguous"])
        self.assertEqual(payload["entry"]["recording"], "rec-a")
        self.assertEqual(payload["entry"]["mask_cleanup"], "fill + largest")
        # The run it was imported from is a compatible entry of the same recording.
        self.assertIn(RUN_NAME, [c["name"] for c in payload["compatible_runs"]])
        listed = self.get("/api/workspaces")
        self.assertIn(RUN_NAME, [w["name"] for w in listed])
        self.assertIn("summary", listed[0])
        self.assertEqual(self.get("/api/workspaces/missing", 404), {"error": "unknown workspace 'missing'"})
        self.assertEqual(self.get("/api/workspaces/a%5Cb", 400)["error"], "invalid workspace name " + repr("a\\b"))

        frame = self.get(f"/api/workspaces/{RUN_NAME}/frame?frame={FRAMES - 1}&raw=1")
        self.assertEqual((frame["height"], frame["width"]), (HEIGHT, WIDTH))
        self.assertTrue(frame["layers"]["image"].startswith("data:image/jpeg"))
        self.assertTrue(frame["image_raw"].startswith("data:image/jpeg"))
        self.assertTrue(frame["layers"]["tube"].startswith("data:image/png"))
        self.assertIn("tube_independent", frame["layers"])
        self.assertNotIn("mask_final", frame["layers"])  # no checkpoint, no stored masks
        self.assertFalse(frame["has_stored_mask"])
        self.assertTrue(any("checkpoint" in e for e in frame["errors"]))
        self.assertEqual(frame["provenance"]["algorithm"], "chain_forward")
        self.assertEqual(frame["stats"]["source_name"], "forward")
        self.assertEqual([h["source"] for h in frame["pose"]["hypotheses"]], ["independent", "forward"])
        light = self.get(f"/api/workspaces/{RUN_NAME}/frame?frame=2&detail=light")
        self.assertEqual(sorted(light["layers"]), ["image", "tube"])
        self.assertEqual(light["errors"], [])
        self.assertEqual(self.get(f"/api/workspaces/{RUN_NAME}/frame?frame=99", 400)["error"], f"frame 99 is not in this run")
        pose = self.get(f"/api/workspaces/{RUN_NAME}/pose?frame=3")
        self.assertTrue(pose["present"])
        self.assertEqual(len(pose["pose"]["centerline_xy"]), 100)
        self.assertEqual(pose["provenance"]["algorithm"], "independent_fit")
        self.assertFalse(self.get(f"/api/workspaces/{RUN_NAME}/pose?frame=42")["present"])
        # The viewer's endpoints accept the workspace name too.
        self.assertEqual(self.get(f"/api/run?name={RUN_NAME}")["series"]["frame_index"], list(range(FRAMES)))
        self.assertEqual(self.client.get(f"/api/starts?run={RUN_NAME}&frame=2").status_code, 400)  # no checkpoint anywhere

        snapshot = self.post(f"/api/workspaces/{RUN_NAME}/snapshot", {"label": "before edits"})
        self.assertTrue(snapshot["name"].endswith("before-edits"))
        self.assertEqual(snapshot["snapshots"], [snapshot["name"]])
        self.assertTrue((Path(snapshot["path"]) / "state.npz").exists())
        self.assertEqual(self.get(f"/api/workspaces/{RUN_NAME}/edits"), {"edits": [], "count": 0})
        self.assertEqual(self.get(f"/api/workspaces/{RUN_NAME}")["summary"]["snapshots"], [snapshot["name"]])

        notes = self.post("/api/note", {"workspace": RUN_NAME, "frame_index": 5, "tags": ["coil"], "comment": "gap closed"})["notes"]
        self.assertEqual((notes[0]["recording"], notes[0]["workspace"], notes[0]["tags"]), ("rec-a", RUN_NAME, ["coil"]))
        self.assertEqual(len(self.get("/api/notes")["notes"]), 1)
        self.assertEqual(self.post("/api/note/delete", {"index": 0})["notes"], [])

    def test_create_workspace_and_run_a_segment_job(self) -> None:
        created = self.post("/api/workspaces", {"name": "fresh", "recording": str(self.recording), "first": 0, "last": FRAMES - 1})
        self.assertEqual((created["name"], created["frames"], created["frame_count"]), ("fresh", [0, FRAMES - 1], FRAMES))
        self.assertEqual(created["summary"]["fitted"], 0)
        self.assertFalse(created["summary"]["has_masks"])
        self.assertIn("exists", self.post("/api/workspaces", {"name": "fresh", "recording": str(self.recording), "first": 0, "last": 2}, 400)["error"])
        self.assertIn("does not exist", self.post("/api/workspaces", {"name": "x", "recording": str(self.root / "recordings" / "nope.h5"), "first": 0, "last": 2}, 400)["error"])
        self.assertIn("recording roots", self.post("/api/workspaces", {"name": "x", "recording": "/nope.h5", "first": 0, "last": 2}, 404)["error"])
        self.assertIn("name", self.post("/api/workspaces", {"recording": str(self.recording), "first": 0, "last": 2}, 400)["error"])
        self.assertIn("'first' must be an integer", self.post("/api/workspaces", {"name": "x", "recording": str(self.recording), "first": "abc", "last": 2}, 400)["error"])
        self.assertIn("'last' is required", self.post("/api/workspaces", {"name": "x", "recording": str(self.recording), "first": 0}, 400)["error"])
        # An empty workspace still opens in the viewer: every row unfitted.
        empty = self.get("/api/workspaces/fresh")
        self.assertEqual(empty["series"]["fitted"], [0] * FRAMES)
        self.assertEqual(empty["series"]["classification"], ["unfitted"] * FRAMES)
        self.assertEqual(empty["provenance_counts"], {})
        frame = self.get("/api/workspaces/fresh/frame?frame=2")
        self.assertEqual(sorted(frame["layers"]), ["image"])
        self.assertIsNone(frame["pose"])

        stages = self.get("/api/stages")
        self.assertEqual([s["name"] for s in stages], list(pipeline.STAGES))
        segment = next(s for s in stages if s["name"] == "segment")
        threshold = next(p for p in segment["params"] if p["name"] == "threshold")
        self.assertEqual((threshold["type"], threshold["default"]), ("float", 0.5))
        self.assertTrue(threshold["help"])

        self.assertIn("unknown stage", self.post("/api/jobs", {"kind": "stage", "workspace": "fresh", "stage": "nope"}, 400)["error"])
        self.assertEqual(self.post("/api/jobs", {"kind": "stage", "workspace": "missing", "stage": "segment"}, 404)["error"], "unknown workspace 'missing'")
        self.assertIn("unknown job kind", self.post("/api/jobs", {"kind": "dance"}, 400)["error"])
        self.assertEqual(self.get("/api/jobs/j00000099", 404)["error"], "unknown job 'j00000099'")
        self.assertIn("unknown job state", self.get("/api/jobs?state=sleeping", 400)["error"])

        with mock.patch.object(pipeline, "stage_command", _cpu_stage_command):
            submitted = self.post("/api/jobs", {"kind": "stage", "workspace": "fresh", "stage": "segment", "params": SEGMENT_PARAMS, "label": "seg"})
        self.assertEqual(submitted["state"], "queued")
        self.assertEqual(submitted["spec"], {"kind": "stage", "params": {"stage": "segment", "params": SEGMENT_PARAMS}, "workspace": "fresh", "frames": [0, FRAMES - 1], "gpus": 1, "label": "seg"})
        self.assertEqual(submitted["command"][0], sys.executable)
        self.assertIn("--stage', 'segment'", submitted["command"][-1])
        record = self.wait_for_job(submitted["id"])
        self.assertEqual(record["state"], "done", record)
        self.assertEqual(record["gpu"], 0)
        self.assertEqual(record["progress"], 1.0)
        self.assertEqual(record["result"]["frames"], FRAMES)
        self.assertEqual(record["result"]["frames_with_worm"], FRAMES)
        self.assertEqual(self.get(f"/api/jobs/{record['id']}/log?tail=5")["id"], record["id"])
        self.assertIn(record["id"], [j["id"] for j in self.get("/api/jobs?state=done")])
        self.assertEqual(self.get("/api/jobs?state=running"), [])

        # The workspace now has masks; the viewer serves the stored one without a segmenter.
        workspace = Workspace.open(self.root / "workspaces" / "fresh")
        self.assertEqual(workspace.mask_rows().tolist(), list(range(FRAMES)))
        opened = self.get("/api/workspaces/fresh")
        self.assertTrue(opened["summary"]["has_masks"])
        self.assertEqual(opened["mask_rows"], list(range(FRAMES)))
        self.assertGreater(opened["series"]["worm_pixels"][2], 200)
        frame = self.get("/api/workspaces/fresh/frame?frame=2")
        self.assertEqual(sorted(frame["layers"]), ["image", "mask_final"])
        self.assertEqual(frame["mask_final_source"], "stored")
        self.assertTrue(frame["has_stored_mask"])
        self.assertEqual(frame["errors"], [])
        self.assertEqual(frame["mask_stats_stored"]["worm_pixels"], int(workspace.get_mask(2).sum()))
        starts = self.get("/api/workspaces/fresh/starts?frame=2")
        self.assertTrue(starts["starts"])
        self.assertEqual(len(starts["starts"][0]["centerline_xy"]), 100)
        self.assertGreater(self.get("/api/state")["workspaces"][0]["summary"]["mask_rows"], 0)

    def test_file_explorer_registers_a_recording_with_its_own_dataset(self) -> None:
        listing = self.get(f"/api/files?path={self.root}")
        self.assertEqual(listing["path"], str(self.root.resolve()))
        self.assertIn("recordings", [e["name"] for e in listing["entries"] if e["kind"] == "dir"])
        self.assertEqual([s["path"] for s in listing["shortcuts"]][0], str(self.root / "recordings"))
        inside = self.get(f"/api/files?path={self.root / 'recordings'}")
        self.assertEqual([(e["name"], e["kind"], e["registered"]) for e in inside["entries"]], [("rec-a.h5", "h5", False)])
        self.assertEqual(self.get("/api/files")["path"], str((self.root / "recordings").resolve()))
        (self.root / "recordings" / "notes.txt").write_text("x")
        self.addCleanup(lambda: (self.root / "recordings" / "notes.txt").unlink())
        self.assertEqual([e["name"] for e in self.get(f"/api/files?path={self.root / 'recordings'}")["entries"]], ["rec-a.h5"])
        everything = self.get(f"/api/files?path={self.root / 'recordings'}&all=1")
        self.assertEqual([(e["name"], e["kind"]) for e in everything["entries"]], [("notes.txt", "file"), ("rec-a.h5", "h5")])
        self.assertTrue(everything["all_files"])
        self.assertEqual(self.get("/api/state")["server"], "app")
        self.assertEqual(self.get(f"/api/files?path={self.root}/nope", 404)["error"], f"{self.root}/nope does not exist")
        self.assertIn("not a directory", self.get(f"/api/files?path={self.recording}", 400)["error"])

        # A camera file outside the roots whose frames live under another dataset name.
        outside = self.root / "elsewhere" / "cam.h5"
        outside.parent.mkdir(exist_ok=True)
        with h5py.File(self.recording, "r") as source, h5py.File(outside, "w") as handle:
            handle.create_dataset("/camera/frames", data=source["/img_nir"][...])
            handle.create_dataset("/camera/times", data=np.arange(FRAMES, dtype=np.float64))
        datasets = self.get(f"/api/recordings/datasets?path={outside}")
        self.assertEqual(datasets["default"], "/camera/frames")
        self.assertEqual([(d["name"], d["video"]) for d in datasets["datasets"]], [("/camera/frames", True), ("/camera/times", False)])
        self.assertFalse(datasets["registered"])
        self.assertEqual(self.get(f"/api/recordings/datasets?path={self.root}/nope.h5", 404)["error"], f"no file at {self.root}/nope.h5")
        self.assertIn("recording roots", self.client.get(f"/api/recordings/thumbnail?path={outside}&frame=0").json()["error"])
        self.assertIn("cannot be read", self.post("/api/recordings/register", {"path": str(outside), "dataset": "/camera/times"}, 400)["error"])

        registered = self.post("/api/recordings/register", {"path": str(outside)})
        # Whatever happens below, the registration must not leak into the other tests of this class.
        self.addCleanup(lambda: self.client.post("/api/recordings/unregister", json={"path": str(outside)}))
        self.assertEqual((registered["name"], registered["dataset"], registered["registered"], registered["frames"]), ("cam", "/camera/frames", True, FRAMES))
        self.assertIn("cam", [r["name"] for r in self.get("/api/recordings")])
        self.assertTrue(self.get(f"/api/recordings/datasets?path={outside}")["registered"])
        self.assertTrue(self.get(f"/api/files?path={outside.parent}")["entries"][0]["registered"])
        thumb = self.client.get(f"/api/recordings/thumbnail?path={outside}&frame=1&scale=0.5")
        self.assertEqual(thumb.status_code, 200, thumb.text)
        self.assertEqual(Image.open(io.BytesIO(thumb.content)).size, (WIDTH // 2, HEIGHT // 2))

        # A workspace on it carries the dataset and the segment stage reads it.
        created = self.post("/api/workspaces", {"name": "cam_ws", "recording": str(outside), "first": 0, "last": 3})
        self.assertEqual(created["frame_count"], 4)
        self.assertEqual(Workspace.open(self.root / "workspaces" / "cam_ws").info.settings["dataset"], "/camera/frames")
        frame = self.get("/api/workspaces/cam_ws/frame?frame=1")
        self.assertIn("image", frame["layers"])  # read through /camera/frames
        self.assertEqual((frame["height"], frame["width"]), (HEIGHT, WIDTH))
        self.assertEqual(frame["errors"], ["no segmenter checkpoint available; mask layers skipped"])  # this fixture has no checkpoint
        with mock.patch.object(pipeline, "stage_command", _cpu_stage_command):
            job = self.post("/api/jobs", {"kind": "stage", "workspace": "cam_ws", "stage": "segment", "params": SEGMENT_PARAMS})
        record = self.wait_for_job(job["id"])
        self.assertEqual(record["state"], "done", record)
        self.assertEqual(Workspace.open(self.root / "workspaces" / "cam_ws").mask_rows().tolist(), [0, 1, 2, 3])

        self.assertEqual(self.post("/api/recordings/unregister", {"path": str(outside)}), {"removed": True})
        self.assertEqual(self.post("/api/recordings/unregister", {"path": str(outside)}), {"removed": False})
        self.assertNotIn("cam", [r["name"] for r in self.get("/api/recordings")])
        self.assertEqual(self.client.get(f"/api/recordings/thumbnail?path={outside}&frame=0").status_code, 404)

    def test_command_jobs_and_cancel(self) -> None:
        submitted = self.post("/api/jobs", {"kind": "command", "command": _reporting_command("all done", {"answer": 42}), "label": "one-liner"})
        self.assertEqual((submitted["spec"]["kind"], submitted["spec"]["label"]), ("command", "one-liner"))
        record = self.wait_for_job(submitted["id"], timeout=30.0)
        self.assertEqual(record["state"], "done", record)
        self.assertEqual(record["message"], "all done")
        self.assertEqual(record["result"], {"answer": 42, "gpu": "0"})
        self.assertIn("hello from the job", self.get(f"/api/jobs/{record['id']}/log")["log"])
        self.assertIn("a command job needs", self.post("/api/jobs", {"kind": "command", "command": "ls"}, 400)["error"])

        failing = self.post("/api/jobs", {"kind": "command", "command": [sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"]})
        record = self.wait_for_job(failing["id"], timeout=30.0)
        self.assertEqual(record["state"], "failed")
        self.assertIn("boom", record["error"])

        waiting = self.post("/api/jobs", {"kind": "command", "command": [sys.executable, "-c", "import time; time.sleep(30)"]})
        deadline = time.monotonic() + 30
        while self.get(f"/api/jobs/{waiting['id']}")["state"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.1)
        cancelled = self.post(f"/api/jobs/{waiting['id']}/cancel", {})
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(self.post("/api/jobs/j00000099/cancel", {}, 404)["error"], "unknown job 'j00000099'")
        newest_first = [j["id"] for j in self.get("/api/jobs")]
        self.assertEqual(newest_first, sorted(newest_first, reverse=True))

    # ----- Phase 2: edits

    EDITED = "edited"

    def edited(self) -> dict:
        """The demo run imported a second time, for the edit tests (the shared import above stays untouched)."""

        if not any(w["name"] == self.EDITED for w in self.get("/api/workspaces")):
            self.post("/api/workspaces/import", {"run": RUN_NAME, "name": self.EDITED})
        return self.get(f"/api/workspaces/{self.EDITED}")

    def edit(self, payload: dict, status: int = 200) -> dict:
        return self.post(f"/api/workspaces/{self.EDITED}/edits", payload, status)

    def test_edits_pick_flip_undo_and_provenance(self) -> None:
        payload = self.edited()
        last = FRAMES - 1
        # The provenance block: distinct algorithms, a per-row index into them, no manual edits yet.
        self.assertEqual(payload["provenance"]["algorithms"], ["chain_forward", "independent_fit"])
        self.assertEqual(payload["provenance"]["index"], [1] * (FRAMES - 1) + [0])
        self.assertEqual(payload["provenance"]["edited"], [0] * FRAMES)
        self.assertEqual(payload["provenance"]["algorithm"][-1], "chain_forward")
        self.assertEqual(payload["edits_count"], 0)
        self.assertEqual(self.get(f"/api/workspaces/{self.EDITED}/edits"), {"edits": [], "count": 0})
        self.assertIsNone(self.get(f"/api/workspaces/{self.EDITED}/frame?frame={last}&detail=light")["provenance"]["edit"])

        # Segments: the propagation stretch, or the run of fitted rows outside every stretch.
        self.assertEqual(self.get(f"/api/workspaces/{self.EDITED}/segment?frame={last}"), {"frames": [FRAMES - 3, last], "rows": [FRAMES - 3, last], "in_stretch": True})
        self.assertEqual(self.get(f"/api/workspaces/{self.EDITED}/segment?frame=0"), {"frames": [0, FRAMES - 4], "rows": [0, FRAMES - 4], "in_stretch": False})
        self.assertEqual(self.get(f"/api/workspaces/{self.EDITED}/segment?frame=99", 400)["error"], "frame 99 is not in this run")

        before = self.get(f"/api/workspaces/{self.EDITED}/frame?frame={last}&detail=light")
        self.assertEqual([h["chosen"] for h in before["pose"]["hypotheses"]], [False, True])
        self.assertEqual(before["stats"]["iou"], 0.85)

        # Pick the independent hypothesis of the last frame.
        picked = self.edit({"kind": "pick_hypothesis", "frame": last, "index": 0, "note": "coil looks better"})
        edit = picked["edit"]
        self.assertEqual((edit["kind"], edit["rows"], edit["undone"]), ("pick_hypothesis", [last], None))
        self.assertEqual(edit["edit_id"], "e000001")
        change = edit["summary"]["changes"][0]
        self.assertEqual((change["frame"], change["before"]["iou"], change["after"]["iou"]), (last, 0.85, 0.8))
        self.assertEqual((change["before"]["algorithm"], change["after"]["algorithm"]), ("chain_forward", "manual:pick"))
        self.assertEqual(change["after"]["job"], "edit:e000001")
        frame = picked["frame"]
        self.assertEqual(frame["stats"]["frame_index"], last)
        self.assertEqual(sorted(frame["layers"]), ["image", "tube"])  # detail=light
        self.assertEqual([h["chosen"] for h in frame["pose"]["hypotheses"]], [True, False])
        self.assertEqual(frame["pose"]["centerline_xy"], frame["pose"]["hypotheses"][0]["centerline_xy"])
        self.assertEqual((frame["stats"]["iou"], frame["stats"]["source_name"]), (0.8, "independent"))
        self.assertEqual(frame["provenance"]["algorithm"], "manual:pick")
        self.assertEqual(frame["provenance"]["job"], "edit:e000001")
        self.assertEqual((frame["provenance"]["edit"]["id"], frame["provenance"]["edit"]["note"]), ("e000001", "coil looks better"))
        self.assertEqual(picked["rows"], [last - 1, last])
        patch = picked["series_patch"]
        self.assertEqual(patch["iou"], {str(last - 1): payload["series"]["iou"][last - 1], str(last): 0.8})
        self.assertEqual(patch["source"][str(last)], 0)
        self.assertEqual(patch["provenance"], {str(last - 1): "independent_fit", str(last): "manual:pick"})
        self.assertEqual(patch["edited"], {str(last - 1): 0, str(last): 1})
        self.assertIn(patch["classification"][str(last)], ("clean", "watch", "ambiguous"))
        self.assertIn("ambiguity_score", patch)
        self.assertEqual(set(picked["flags_patch"]), set(payload["series"]["flags"]))
        self.assertEqual(sorted(picked["flags_patch"]["low_iou"]), [str(last - 1), str(last)])
        # The only chain_forward row was replaced, so the algorithm list loses it.
        self.assertEqual(picked["provenance"]["algorithms"], ["independent_fit", "manual:pick"])
        self.assertEqual(picked["provenance"]["index"], [0] * (FRAMES - 1) + [1])
        self.assertEqual(picked["provenance"]["edited"][last], 1)
        self.assertEqual(picked["edits_count"], 1)
        self.assertEqual([(e["id"], e["kind"], e["undoable"], e["undone"]) for e in picked["edits"]], [("e000001", "pick_hypothesis", True, False)])

        # Every later payload is built from the new arrays: run, light frame, full frame, pose.
        opened = self.get(f"/api/workspaces/{self.EDITED}")
        self.assertEqual(opened["series"]["iou"][last], 0.8)
        self.assertEqual(opened["series"]["source"][last], 0)
        self.assertEqual(opened["edits_count"], 1)
        self.assertEqual(opened["provenance"]["edited"], [0] * (FRAMES - 1) + [1])
        self.assertEqual(opened["provenance_counts"], {"independent_fit": FRAMES - 1, "manual:pick": 1})
        light = self.get(f"/api/workspaces/{self.EDITED}/frame?frame={last}&detail=light")
        self.assertEqual(light["pose"]["centerline_xy"], frame["pose"]["hypotheses"][0]["centerline_xy"])
        full = self.get(f"/api/workspaces/{self.EDITED}/frame?frame={last}")
        self.assertEqual(full["provenance"]["edit"]["id"], "e000001")
        self.assertEqual(self.get(f"/api/workspaces/{self.EDITED}/pose?frame={last}")["provenance"]["algorithm"], "manual:pick")
        self.assertEqual(self.get(f"/api/workspaces/{self.EDITED}/edits")["edits"][0]["frames"], [last, last])

        # Flip the frame: the centerline runs the other way, 'reversed' toggles, provenance is the flip.
        flipped = self.edit({"kind": "flip", "frame": last, "scope": "frame"})
        self.assertEqual((flipped["edit"]["kind"], flipped["edit"]["rows"]), ("flip_orientation", [last]))
        self.assertEqual(flipped["frame"]["pose"]["centerline_xy"], light["pose"]["centerline_xy"][::-1])
        self.assertEqual(flipped["frame"]["stats"]["reversed"], True)
        self.assertEqual(flipped["frame"]["provenance"]["algorithm"], "manual:flip")
        self.assertEqual(flipped["series_patch"]["reversed"][str(last)], True)
        self.assertEqual(flipped["frame"]["pose"]["path"], {**light["pose"]["path"], "mirrored": True})

        # Flip the segment around frame FRAMES-2: the whole stretch; the last frame is back the right way round.
        segment = self.edit({"kind": "flip", "frame": FRAMES - 2, "scope": "segment", "note": "whole coil"})
        self.assertEqual(segment["edit"]["rows"], [FRAMES - 3, FRAMES - 2, last])
        self.assertEqual(segment["frame"]["stats"]["frame_index"], FRAMES - 2)
        self.assertEqual(segment["rows"], list(range(FRAMES - 4, FRAMES)))
        self.assertEqual(segment["series_patch"]["reversed"], {str(FRAMES - 4): False, str(FRAMES - 3): True, str(FRAMES - 2): True, str(last): False})
        self.assertEqual(segment["series_patch"]["provenance"][str(FRAMES - 3)], "manual:flip")
        self.assertEqual(segment["provenance"]["edited"], [0] * (FRAMES - 3) + [1] * 3)
        self.assertEqual(segment["edits"][0]["note"], "whole coil")

        # An explicit list of frames.
        listed = self.edit({"kind": "flip", "frames": [0, 1]})
        self.assertEqual(listed["edit"]["rows"], [0, 1])
        self.assertEqual(listed["frame"]["stats"]["frame_index"], 0)
        self.assertEqual(listed["series_patch"]["reversed"], {"0": True, "1": True, "2": False})
        edits = self.get(f"/api/workspaces/{self.EDITED}/edits")
        self.assertEqual(edits["count"], 4)
        self.assertEqual([e["id"] for e in edits["edits"]], ["e000004", "e000003", "e000002", "e000001"])
        self.assertEqual([e["kind"] for e in edits["edits"]], ["flip_orientation", "flip_orientation", "flip_orientation", "pick_hypothesis"])
        self.assertEqual(edits["edits"][0]["frames"], [0, 1])
        self.assertEqual(edits["edits"][0]["rows"], 2)
        self.assertTrue(all(e["undoable"] and not e["undone"] for e in edits["edits"]))

        # Undo the newest: the frames flip is reversed and marked undone; the log gains an undo entry.
        undone = self.edit({"kind": "undo", "frame": 1})
        self.assertEqual((undone["edit"]["kind"], undone["edit"]["undone"], undone["edit"]["rows"]), ("undo", "e000004", [0, 1]))
        self.assertEqual(undone["frame"]["stats"]["frame_index"], 1)
        self.assertEqual(undone["frame"]["stats"]["reversed"], False)
        self.assertEqual(undone["series_patch"]["reversed"], {"0": False, "1": False, "2": False})
        self.assertEqual(undone["series_patch"]["provenance"]["0"], "independent_fit")
        self.assertEqual(undone["provenance"]["edited"][:2], [0, 0])
        self.assertEqual([(e["id"], e["kind"], e["undone"], e["undoable"]) for e in undone["edits"]][:2], [("e000005", "undo", False, False), ("e000004", "flip_orientation", True, False)])
        self.assertEqual(undone["edits"][0]["undoes"], "e000004")
        self.assertIsNone(self.get(f"/api/workspaces/{self.EDITED}/frame?frame=0&detail=light")["provenance"]["edit"])
        # Undoing it again is refused; so is an unknown id or an undo entry.
        self.assertEqual(self.edit({"kind": "undo", "edit": "e000004"}, 400)["error"], "e000004 is already undone")
        self.assertEqual(self.edit({"kind": "undo", "edit": "e000005"}, 400)["error"], "e000005 is an undo and cannot be undone")
        self.assertEqual(self.edit({"kind": "undo", "edit": "e000099"}, 400)["error"], "no edit e000099")
        # Without a frame the response describes the first row restored.
        undone = self.edit({"kind": "undo"})
        self.assertEqual((undone["edit"]["undone"], undone["frame"]["stats"]["frame_index"]), ("e000003", FRAMES - 3))
        self.assertEqual(undone["series_patch"]["reversed"], {str(FRAMES - 4): False, str(FRAMES - 3): False, str(FRAMES - 2): False, str(last): True})
        # Back to the pick (frame flip undone), then to the import (pick undone).
        undone = self.edit({"kind": "undo", "frame": last})
        self.assertEqual(undone["edit"]["undone"], "e000002")
        self.assertEqual(undone["frame"]["stats"]["reversed"], False)
        self.assertEqual(undone["frame"]["provenance"]["algorithm"], "manual:pick")
        undone = self.edit({"kind": "undo", "frame": last})
        self.assertEqual(undone["edit"]["undone"], "e000001")
        self.assertEqual((undone["frame"]["stats"]["iou"], undone["frame"]["stats"]["source_name"]), (0.85, "forward"))
        self.assertEqual(undone["frame"]["provenance"], {**before["provenance"], "edit": None})
        self.assertEqual([h["chosen"] for h in undone["frame"]["pose"]["hypotheses"]], [False, True])
        self.assertEqual(undone["provenance"]["algorithms"], ["chain_forward", "independent_fit"])
        self.assertEqual(undone["provenance"]["edited"], [0] * FRAMES)
        self.assertEqual(self.edit({"kind": "undo"}, 400)["error"], "nothing to undo")
        restored = self.get(f"/api/workspaces/{self.EDITED}")
        self.assertEqual(restored["series"]["iou"], payload["series"]["iou"])
        self.assertEqual(restored["series"]["reversed"], payload["series"]["reversed"])
        # Undo recomputes the ambiguity signals of the restored rows from the arrays rather than trusting the
        # fixture's hand-set flags, so only the rows no edit reached keep their classification here.
        self.assertEqual(restored["series"]["classification"][: FRAMES - 3], payload["series"]["classification"][: FRAMES - 3])
        self.assertEqual(restored["provenance"]["index"], payload["provenance"]["index"])
        self.assertEqual(restored["edits_count"], 8)
        listed = self.get(f"/api/workspaces/{self.EDITED}/edits")["edits"]
        self.assertEqual([e["kind"] for e in listed].count("undo"), 4)
        self.assertFalse(any(e["undoable"] for e in listed))
        self.assertEqual(restored["summary"]["edits"], 8)

    def test_edit_requests_are_validated(self) -> None:
        self.edited()
        last = FRAMES - 1
        count = self.get(f"/api/workspaces/{self.EDITED}/edits")["count"]
        self.assertIn("unknown edit kind", self.edit({"kind": "teleport", "frame": 0}, 400)["error"])
        self.assertIn("frame", self.edit({"kind": "pick_hypothesis", "index": 0}, 400)["error"])
        self.assertEqual(self.edit({"kind": "pick_hypothesis", "frame": 99, "index": 0}, 400)["error"], "frame 99 is not in this run")
        self.assertIn("does not exist", self.edit({"kind": "pick_hypothesis", "frame": last, "index": 7}, 400)["error"])
        self.assertIn("hypotheses", self.edit({"kind": "pick_hypothesis", "frame": 0, "index": 0}, 400)["error"])
        self.assertEqual(self.edit({"kind": "flip", "frame": last, "scope": "recording"}, 400)["error"], "scope must be 'frame' or 'segment'")
        self.assertEqual(self.edit({"kind": "flip", "frames": []}, 400)["error"], "'frames' is empty")
        self.assertEqual(self.post("/api/workspaces/missing/edits", {"kind": "undo"}, 404)["error"], "unknown workspace 'missing'")
        self.assertEqual(self.get("/api/workspaces/missing/segment?frame=0", 404)["error"], "unknown workspace 'missing'")
        # Nothing above touched the workspace.
        self.assertEqual(self.get(f"/api/workspaces/{self.EDITED}/edits")["count"], count)


class RegionTests(unittest.TestCase):
    """Phase 3 over a fitted synthetic workspace: the registry, region proposals, region jobs, candidate sets, acceptance, outcomes."""

    WS = "region"

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.root = root
        cls.recording = root / "recordings" / "rec-b.h5"
        cls.recording.parent.mkdir()
        test_pipeline._write_recording(cls.recording)
        config = AppConfig(
            workspaces_root=root / "workspaces", recording_roots=(root / "recordings",), poses_root=root / "runs",
            dataset_root=root / "dataset", checkpoint=None, prior_cache=None, notes=root / "notes.json", gpus=(0,), device="cpu",
            job_interval=0.1,
        )
        (root / "runs").mkdir()
        cls.app = create_app(config)
        cls.client = TestClient(cls.app, raise_server_exceptions=False)
        cls.client.__enter__()
        response = cls.client.post("/api/workspaces", json={"name": cls.WS, "recording": str(cls.recording), "first": 0, "last": test_pipeline.FRAMES - 1})
        assert response.status_code == 200, response.text
        cls.workspace = Workspace.open(root / "workspaces" / cls.WS)
        # Segment, fit and score once, directly (the stage jobs have their own tests); orient every frame like row 0
        # so the outer frames agree as anchors (the synthetic body's symmetric taper leaves orientation to noise).
        pipeline.run_stage(cls.workspace, "segment", test_pipeline.SEGMENT_PARAMS, device="cpu")
        pipeline.run_stage(cls.workspace, "fit", test_pipeline.FIT_PARAMS, device="cpu", job="jfit")
        state = cls.workspace.load_state()
        curves = state["centerline_xy"]
        for row in range(1, test_pipeline.FRAMES):
            same = np.linalg.norm(curves[row, 0] - curves[0, 0]) + np.linalg.norm(curves[row, -1] - curves[0, -1])
            swapped = np.linalg.norm(curves[row, 0] - curves[0, -1]) + np.linalg.norm(curves[row, -1] - curves[0, 0])
            if swapped < same:
                edits._reverse_row(state, row)
        cls.workspace.save_state(state)
        pipeline.run_stage(cls.workspace, "ambiguity", {}, device="cpu")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls._directory.cleanup()

    def get(self, path: str, status: int = 200) -> dict | list:
        response = self.client.get(path)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def post(self, path: str, payload: dict, status: int = 200) -> dict | list:
        response = self.client.post(path, json=payload)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def wait_for_job(self, job_id: str, timeout: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(f"/api/jobs/{job_id}")
            if record["state"] in ("done", "failed", "cancelled"):
                return record
            time.sleep(0.1)
        self.fail(f"job {job_id} did not finish: {self.get(f'/api/jobs/{job_id}')}")

    def region_job(self, payload: dict, status: int = 200) -> dict:
        with mock.patch.object(pipeline, "region_command", _cpu_region_command):
            return self.post("/api/jobs", {"kind": "region", "workspace": self.WS, **payload}, status)

    def test_registry_and_region_proposals(self) -> None:
        n = test_pipeline.FRAMES
        listed = self.get("/api/algorithms")
        self.assertEqual([a["id"] for a in listed], list(algorithms.REGISTRY))
        mirror = next(a for a in listed if a["id"] == "mirror")
        self.assertEqual(mirror["scope"], "region")
        self.assertEqual({p["name"] for p in mirror["parameters"]} & {"path_temperature", "path_distance_weight"}, {"path_temperature", "path_distance_weight"})
        beam = next(a for a in listed if a["id"] == "beam_path")
        preset = next(p for p in beam["parameters"] if p["name"] == "preset")
        self.assertEqual((preset["type"], preset["default"], preset["choices"]), ("choice", "fast", ["fast", "balanced", "reference"]))

        opened = self.get(f"/api/workspaces/{self.WS}")
        self.assertEqual(opened["summary"]["fitted"], n)
        self.assertEqual(opened["stretches"], [])
        # No propagation stretch: ten frames either side, clipped to the workspace; no anchors can lie outside it.
        proposal = self.get(f"/api/workspaces/{self.WS}/region?frame=3")
        self.assertEqual(proposal, {"frames": [0, n - 1], "rows": [0, n - 1], "anchor_before": None, "anchor_after": None, "reason": proposal["reason"], "stretch": None})
        self.assertIn("no propagation stretch", proposal["reason"])
        # A stretch in the summary: padded, with the nearest trusted rows outside as anchors.
        pipeline.update_summary(self.workspace, {"propagation": {"stretches": [[2, 3]]}})
        try:
            proposal = self.get(f"/api/workspaces/{self.WS}/region?frame=3&pad=1")
            self.assertEqual((proposal["frames"], proposal["rows"], proposal["stretch"]), ([1, 4], [1, 4], [2, 3]))
            state = self.workspace.load_state()
            good = [r for r in range(n) if state["iou"][r] >= 0.9 and state["ambiguity_score"][r] == 0]
            self.assertEqual(proposal["anchor_before"], {"row": 0, "frame": 0} if 0 in good else None)
            self.assertEqual(proposal["anchor_after"], {"row": 5, "frame": 5} if 5 in good else None)
            # Anchors for frames the user typed.
            typed = self.get(f"/api/workspaces/{self.WS}/region?first=2&last=3")
            self.assertEqual((typed["frames"], typed["stretch"]), ([2, 3], None))
            self.assertEqual(typed["anchor_before"], {"row": 1, "frame": 1} if 1 in good else proposal["anchor_before"])
            self.assertIn("frames given", typed["reason"])
        finally:
            pipeline.update_summary(self.workspace, {"propagation": {}})
        self.assertIn("not in workspace", self.get(f"/api/workspaces/{self.WS}/region?frame=99", 400)["error"])
        self.assertIn("after", self.get(f"/api/workspaces/{self.WS}/region?first=4&last=2", 400)["error"])
        self.assertIn("go together", self.get(f"/api/workspaces/{self.WS}/region?first=2", 400)["error"])
        self.assertIn("required", self.get(f"/api/workspaces/{self.WS}/region", 400)["error"])
        self.assertEqual(self.get("/api/workspaces/missing/region?frame=0", 404)["error"], "unknown workspace 'missing'")

    def test_region_job_candidate_sets_accept_discard_and_outcomes(self) -> None:
        n = test_pipeline.FRAMES
        before = self.get(f"/api/workspaces/{self.WS}")
        # Bad requests fail at submit time, not as failed jobs.
        self.assertIn("unknown algorithm", self.region_job({"algorithm": "teleport", "first": 1, "last": 4}, 400)["error"])
        self.assertIn("above the maximum", self.region_job({"algorithm": "beam_path", "first": 1, "last": 4, "params": {"beam": 99}}, 400)["error"])
        self.assertIn("outside the region", self.region_job({"algorithm": "mirror", "first": 1, "last": 4, "anchor_before": 2}, 400)["error"])
        self.assertIn("after", self.region_job({"algorithm": "mirror", "first": 4, "last": 1}, 400)["error"])
        self.assertIn("not in workspace", self.region_job({"algorithm": "mirror", "first": 1, "last": 99}, 400)["error"])
        self.assertIn("'first' is required", self.region_job({"algorithm": "mirror", "last": 4}, 400)["error"])
        self.assertIn("'params' must be an object", self.region_job({"algorithm": "mirror", "first": 1, "last": 4, "params": [1, 2]}, 400)["error"])
        self.assertIn("needs an anchor before", self.region_job({"algorithm": "chain_forward", "first": 1, "last": 4, "anchor_after": 5}, 400)["error"])
        self.assertIn("needs an anchor after", self.region_job({"algorithm": "chain_backward", "first": 1, "last": 4, "anchor_before": 0}, 400)["error"])
        self.assertEqual(self.region_job({"algorithm": "mirror", "first": 1, "last": 4, "workspace": "missing"}, 404)["error"], "unknown workspace 'missing'")
        self.assertEqual(self.get(f"/api/workspaces/{self.WS}/candidates"), [])
        self.assertEqual(self.get(f"/api/workspaces/{self.WS}/candidates/c000001", 404)["error"], f"workspace {self.WS} has no candidate set 'c000001'")
        self.assertEqual(self.get("/api/outcomes"), [])

        # Mirror over frames 1..4 between the anchors 0 and 5: the current poses already follow the anchors, so the path keeps them.
        submitted = self.region_job({"algorithm": "mirror", "first": 1, "last": 4, "anchor_before": 0, "anchor_after": 5, "params": {"path_temperature": 1.0}, "label": "mirror 1-4"})
        spec = submitted["spec"]
        self.assertEqual((spec["kind"], spec["workspace"], spec["frames"], spec["label"]), ("region", self.WS, [1, 4], "mirror 1-4"))
        self.assertEqual((spec["params"]["algorithm"], spec["params"]["first"], spec["params"]["last"]), ("mirror", 1, 4))
        self.assertEqual((spec["params"]["anchor_before"], spec["params"]["anchor_after"]), (0, 5))
        self.assertEqual(spec["params"]["params"]["path_temperature"], 1.0)
        self.assertEqual(spec["params"]["anchor_frames"], {"before": 0, "after": 5})
        self.assertNotIn("id", spec["params"])  # the process names the set after WORM_POSE_JOB_ID
        self.assertIn("--region-run", submitted["command"][-1])
        record = self.wait_for_job(submitted["id"])
        self.assertEqual(record["state"], "done", record)
        job_id = record["id"]
        result = record["result"]
        self.assertEqual(result["candidate_set"], job_id)
        self.assertEqual((result["rows"], result["frames"], result["anchors"]), ([1, 4], [1, 4], {"before": 0, "after": 5}))
        self.assertEqual((result["candidates"], result["path_rows"]), (8, 4))
        self.assertEqual(set(algorithms.METRIC_NAMES) - set(result["metrics"]), set())
        self.assertEqual(result["metrics"]["orientation_flips"], 0)

        # The list, the set and the frame payloads.
        listed = self.get(f"/api/workspaces/{self.WS}/candidates")
        self.assertEqual([e["id"] for e in listed], [job_id])
        entry = listed[0]
        self.assertEqual((entry["algorithm"], entry["frames"], entry["rows"], entry["accepted"], entry["job"]), ("mirror", [1, 4], [1, 4], False, job_id))
        self.assertEqual(entry["anchors"], {"before": {"row": 0, "frame": 0}, "after": {"row": 5, "frame": 5}})
        self.assertEqual(entry["metrics"]["median_iou"], result["metrics"]["median_iou"])
        self.assertIn("median_iou", entry["metrics_before"])
        full = self.get(f"/api/workspaces/{self.WS}/candidates/{job_id}")
        self.assertEqual((full["id"], full["algorithm"], full["frames"], full["params"]["path_temperature"]), (job_id, "mirror", [1, 4], 1.0))
        self.assertEqual([r["frame"] for r in full["per_row"]], [1, 2, 3, 4])
        row = full["per_row"][1]
        self.assertEqual([c["source"] for c in row["candidates"]], ["current", "mirrored"])
        self.assertEqual([c["chosen"] for c in row["candidates"]], [True, False])
        self.assertEqual((row["chosen"], row["mirrored"]), (0, False))
        self.assertEqual(row["chosen_centerline_xy"], row["candidates"][0]["centerline_xy"])
        self.assertEqual(row["candidates"][1]["centerline_xy"], row["candidates"][0]["centerline_xy"][::-1])
        self.assertEqual(len(row["candidates"][0]["centerline_xy"]), 100)
        self.assertEqual(row["candidates"][0]["iou"], before["series"]["iou"][2])
        self.assertEqual(full["path"], [{"row": r, "frame": r, "index": 0, "mirrored": False} for r in range(1, 5)])
        self.assertEqual(set(full["chosen_iou"]), {"1", "2", "3", "4"})
        self.assertEqual(full["current_metrics"]["median_iou"], full["metrics_before"]["median_iou"])
        self.assertEqual(full["current_metrics"]["frames"], 4)
        frame = self.get(f"/api/workspaces/{self.WS}/frame?frame=2&detail=light")
        self.assertEqual([(s["id"], s["algorithm"], s["index"], s["mirrored"], s["candidates"]) for s in frame["candidate_sets"]], [(job_id, "mirror", 0, False, 2)])
        self.assertEqual(frame["pose"]["candidate_sets"][0]["centerline_xy"], row["chosen_centerline_xy"])
        self.assertEqual(self.get(f"/api/workspaces/{self.WS}/frame?frame=5&detail=light")["candidate_sets"], [])
        self.assertEqual(self.get(f"/api/workspaces/{self.WS}/pose?frame=3")["pose"]["candidate_sets"][0]["id"], job_id)
        outcome = self.get(f"/api/outcomes?workspace={self.WS}")[0]
        self.assertEqual((outcome["candidate_set"], outcome["algorithm"], outcome["accepted"], outcome["first"], outcome["last"]), (job_id, "mirror", False, 1, 4))
        self.assertEqual(outcome["anchors"], {"before": 0, "after": 5})
        self.assertEqual(self.get("/api/outcomes?algorithm=beam_path"), [])
        self.assertEqual(self.get("/api/outcomes?workspace=other"), [])

        # A second set, from every start of the mask, to compare with.
        second = self.wait_for_job(self.region_job({"algorithm": "independent_multistart", "first": 2, "last": 3, "anchor_before": 1, "anchor_after": 4})["id"])
        self.assertEqual(second["state"], "done", second)
        self.assertEqual([e["id"] for e in self.get(f"/api/workspaces/{self.WS}/candidates")], [second["id"], job_id])
        both = self.get(f"/api/workspaces/{self.WS}/frame?frame=3&detail=light")["candidate_sets"]
        self.assertEqual([s["id"] for s in both], [second["id"], job_id])
        self.assertGreater(both[0]["candidates"], 1)
        self.assertEqual(len(self.get("/api/outcomes")), 2)

        # Accept two frames of the mirror set: one accept_path edit attributed to the algorithm and the set.
        accepted = self.post(f"/api/workspaces/{self.WS}/candidates/{job_id}/accept", {"rows": [2, 3], "note": "keep these"})
        edit = accepted["edit"]
        self.assertEqual((edit["kind"], edit["rows"], edit["edit_id"]), ("accept_path", [2, 3], "e000001"))
        change = edit["summary"]["changes"][0]
        self.assertEqual((change["frame"], change["after"]["algorithm"], change["after"]["job"]), (2, "mirror", f"candidates:{job_id}"))
        self.assertEqual(accepted["frame"]["stats"]["frame_index"], 2)
        self.assertEqual(accepted["frame"]["provenance"], {**accepted["frame"]["provenance"], "algorithm": "mirror", "job": f"candidates:{job_id}"})
        self.assertEqual(accepted["frame"]["provenance"]["edit"]["note"], "keep these")
        self.assertEqual(accepted["frame"]["pose"]["centerline_xy"], row["chosen_centerline_xy"])
        self.assertEqual(accepted["rows"], [1, 2, 3, 4])
        self.assertEqual(accepted["series_patch"]["provenance"], {"1": "independent_fit", "2": "mirror", "3": "mirror", "4": "independent_fit"})
        self.assertEqual(accepted["series_patch"]["edited"], {"1": 0, "2": 1, "3": 1, "4": 0})
        self.assertEqual(accepted["provenance"]["algorithms"], ["independent_fit", "mirror"])
        self.assertEqual([(e["id"], e["kind"]) for e in accepted["edits"]], [("e000001", "accept_path")])
        # Two of the path's four rows: the set records them and stays open for the rest.
        candidate_set = accepted["candidate_set"]
        self.assertEqual((candidate_set["id"], candidate_set["accepted"], candidate_set["accepted_edit"], candidate_set["accepted_rows"]), (job_id, False, "e000001", [2, 3]))
        self.assertEqual([(e["id"], e["accepted"], e["accepted_rows"]) for e in accepted["candidate_sets"]], [(second["id"], False, []), (job_id, False, [2, 3])])
        # The state carries the set's poses with the set's provenance; the set no longer overlays the frames it was accepted on.
        provenance = self.workspace.load_provenance()
        self.assertEqual([str(a) for a in provenance["algorithm"][1:5]], ["independent_fit", "mirror", "mirror", "independent_fit"])
        self.assertEqual(str(provenance["job"][2]), f"candidates:{job_id}")
        self.assertEqual(self.get(f"/api/workspaces/{self.WS}")["provenance_counts"], {"independent_fit": n - 2, "mirror": 2})
        self.assertEqual([s["id"] for s in self.get(f"/api/workspaces/{self.WS}/frame?frame=2&detail=light")["candidate_sets"]], [second["id"]])
        self.assertEqual([s["id"] for s in self.get(f"/api/workspaces/{self.WS}/frame?frame=1&detail=light")["candidate_sets"]], [job_id])
        self.assertFalse(self.get(f"/api/workspaces/{self.WS}/candidates/{job_id}")["accepted"])
        outcomes = self.get(f"/api/outcomes?workspace={self.WS}")
        self.assertEqual([(o["candidate_set"], o["accepted"], o["accepted_rows"]) for o in outcomes], [(second["id"], False, []), (job_id, False, [2, 3])])
        self.assertEqual(outcomes[1]["accepted_edit"], "e000001")
        # The same rows again (a double click), rows off the path, a missing set and use_path=false are refused.
        self.assertIn("already accepted", self.post(f"/api/workspaces/{self.WS}/candidates/{job_id}/accept", {"rows": [2, 3]}, 400)["error"])
        self.assertIn("none of the requested rows", self.post(f"/api/workspaces/{self.WS}/candidates/{job_id}/accept", {"rows": [5]}, 400)["error"])
        self.assertIn("use_path", self.post(f"/api/workspaces/{self.WS}/candidates/{job_id}/accept", {"use_path": False}, 400)["error"])
        self.assertIn("'rows' must be a list", self.post(f"/api/workspaces/{self.WS}/candidates/{job_id}/accept", {"rows": [[2]]}, 400)["error"])
        self.assertEqual(self.post(f"/api/workspaces/{self.WS}/candidates/c000009/accept", {}, 404)["error"], f"workspace {self.WS} has no candidate set 'c000009'")
        self.assertEqual(len(self.get(f"/api/workspaces/{self.WS}/edits")["edits"]), 1)
        # The rest of the path completes the set; the hypotheses table shows its candidates with the chosen one current.
        rest = self.post(f"/api/workspaces/{self.WS}/candidates/{job_id}/accept", {})
        self.assertEqual((rest["edit"]["rows"], rest["candidate_set"]["accepted"], rest["candidate_set"]["accepted_rows"]), ([1, 4], True, [1, 2, 3, 4]))
        self.assertEqual([s["id"] for s in self.get(f"/api/workspaces/{self.WS}/frame?frame=1&detail=light")["candidate_sets"]], [])
        table = self.get(f"/api/workspaces/{self.WS}/frame?frame=2&detail=light")["pose"]["hypotheses"]
        self.assertEqual([(h["source"], h["chosen"]) for h in table], [("current", True), ("mirrored", False)])
        self.assertTrue(self.get(f"/api/outcomes?workspace={self.WS}")[1]["accepted"])
        # Undo through the edit log restores the poses, the provenance and the rows' previous (empty) hypotheses, and reopens the set.
        undone = self.post(f"/api/workspaces/{self.WS}/edits", {"kind": "undo"})
        self.assertEqual(undone["edit"]["undone"], "e000002")
        self.assertEqual(self.get(f"/api/workspaces/{self.WS}")["provenance_counts"], {"independent_fit": n - 2, "mirror": 2})
        self.assertNotIn("hypotheses", self.get(f"/api/workspaces/{self.WS}/frame?frame=1&detail=light")["pose"])  # the rows had none before the accept
        reopened = self.get(f"/api/workspaces/{self.WS}/candidates/{job_id}")
        self.assertEqual((reopened["accepted"], reopened["accepted_rows"]), (False, [2, 3]))
        self.assertEqual([s["id"] for s in self.get(f"/api/workspaces/{self.WS}/frame?frame=1&detail=light")["candidate_sets"]], [job_id])
        self.assertEqual([(o["candidate_set"], o["accepted"], o["accepted_rows"]) for o in self.get(f"/api/outcomes?workspace={self.WS}")], [(second["id"], False, []), (job_id, False, [2, 3])])
        undone = self.post(f"/api/workspaces/{self.WS}/edits", {"kind": "undo"})
        self.assertEqual(undone["edit"]["undone"], "e000001")
        self.assertEqual(self.get(f"/api/workspaces/{self.WS}")["provenance_counts"], {"independent_fit": n})
        self.assertEqual(self.get(f"/api/workspaces/{self.WS}/candidates/{job_id}")["accepted_rows"], [])
        self.assertEqual([s["id"] for s in self.get(f"/api/workspaces/{self.WS}/frame?frame=2&detail=light")["candidate_sets"]], [second["id"], job_id])
        # A job running on the workspace makes edits and accepts a 409 rather than a wait.
        blocker = self.post("/api/jobs", {"kind": "command", "workspace": self.WS, "command": [sys.executable, "-c", "import time; time.sleep(60)"], "label": "blocker"})
        deadline = time.monotonic() + 30
        while self.get(f"/api/jobs/{blocker['id']}")["state"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.1)
        try:
            self.assertEqual(self.get(f"/api/jobs/{blocker['id']}")["state"], "running")
            busy = self.post(f"/api/workspaces/{self.WS}/edits", {"kind": "flip", "frame": 2}, 409)["error"]
            self.assertIn(blocker["id"], busy)
            self.assertIn("is writing workspace", self.post(f"/api/workspaces/{self.WS}/candidates/{job_id}/accept", {}, 409)["error"])
        finally:
            self.post(f"/api/jobs/{blocker['id']}/cancel", {})
        self.assertEqual(len(self.get(f"/api/workspaces/{self.WS}/edits")["edits"]), 4)

        # Discard the second set: its files go, the outcome line stays.
        discarded = self.client.delete(f"/api/workspaces/{self.WS}/candidates/{second['id']}")
        self.assertEqual(discarded.status_code, 200, discarded.text)
        self.assertEqual((discarded.json()["removed"], [e["id"] for e in discarded.json()["candidate_sets"]]), (True, [job_id]))
        self.assertFalse((self.workspace.path / "candidates" / f"{second['id']}.npz").exists())
        self.assertEqual(self.client.delete(f"/api/workspaces/{self.WS}/candidates/{second['id']}").status_code, 404)
        # The mirror set, its accepts undone, overlays its frames again.
        self.assertEqual([s["id"] for s in self.get(f"/api/workspaces/{self.WS}/frame?frame=3&detail=light")["candidate_sets"]], [job_id])
        self.assertEqual(len(self.get("/api/outcomes")), 2)


if __name__ == "__main__":
    unittest.main()
