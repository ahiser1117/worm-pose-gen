"""The FastAPI pose app over a synthetic recording, an imported run, and a job runner on a fake GPU pool.

Jobs run real subprocesses: a python one-liner for the ``command`` kind, and
the pipeline's segment stage on the CPU for the ``stage`` kind (with
``pipeline.stage_command`` patched so the child never touches a GPU).
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

from worm_pose_gen import pipeline
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.workspace import Workspace

from tests.test_pose_viewer import FRAMES, HEIGHT, WIDTH, _write_recording, _write_run

SRC = str(Path(__file__).resolve().parents[1] / "src")
RUN_NAME = "2026-09-06T10-00-00Z_demo"
SEGMENT_PARAMS = {"checkpoint": None, "flat_field": False, "min_worm_pixels": 200, "slab": 4}


def _cpu_stage_command(workspace_path: Path | str, stage: str, params: dict | None) -> list[str]:
    """``pipeline.stage_command`` for tests: the same CLI, this interpreter, the CPU."""

    argv = ["--workspace", str(workspace_path), "--stage", stage, "--params", json.dumps(params or {}), "--device", "cpu"]
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
        self.assertEqual(self.get(f"/api/workspaces/{RUN_NAME}/edits"), {"edits": []})
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


if __name__ == "__main__":
    unittest.main()
