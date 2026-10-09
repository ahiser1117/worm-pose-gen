"""The Issues panel and fixes API (``app/routers/fixes.py``) over the posed synthetic workspace of ``tests/test_fixes.py``.

Refit and stitch jobs run real subprocesses on the CPU, with
``fixes.fix_command`` patched to start this interpreter on the source tree.
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

from worm_pose_gen import edits, fixes
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.mask_fit import default_width_template

from tests.slow import slow
from tests.test_fixes import posed_workspace
from tests.test_pipeline import _body_curve

SRC = str(Path(__file__).resolve().parents[1] / "src")
WS = "synthetic"


def _cpu_fix_command(workspace_path: Path | str, spec: dict) -> list[str]:
    """``fixes.fix_command`` for tests: the same CLI, this interpreter, the CPU."""

    argv = ["--workspace", str(workspace_path), "--run", json.dumps(spec), "--device", "cpu"]
    code = f"import sys; sys.path.insert(0, {SRC!r}); from worm_pose_gen.fixes import main; sys.exit(main({argv!r}))"
    return [sys.executable, "-c", code]


class FixesApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.workspace = posed_workspace(root, WS)
        (root / "runs").mkdir()
        config = AppConfig(
            workspaces_root=root / "workspaces", dataset_root=root / "dataset", gpus=(0,), device="cpu", job_interval=0.1,
        )
        cls.client = TestClient(create_app(config), raise_server_exceptions=False)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls._directory.cleanup()

    def request(self, method: str, path: str, payload: dict | None = None, status: int = 200) -> dict | list:
        response = self.client.request(method, f"/api/workspaces/{WS}{path}", json=payload)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def wait_for_job(self, job_id: str, timeout: float = 600.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.client.get(f"/api/jobs/{job_id}").json()
            if record["state"] in ("done", "failed", "cancelled"):
                return record
            time.sleep(0.2)
        self.fail(f"job {job_id} did not finish")

    def submit(self, path: str, payload: dict, status: int = 200) -> dict:
        with mock.patch.object(fixes, "fix_command", _cpu_fix_command):
            return self.request("POST", path, payload, status)

    def test_issues_review_refit_keep_flip_and_undo(self) -> None:
        self.assertEqual(self.request("GET", "/issues")["issues"], [])
        state = self.workspace.load_state()
        for row in (4, 5):
            edits._reverse_row(state, row)
        self.workspace.save_state(state)

        payload = self.request("GET", "/issues")
        self.assertEqual(payload["min_issue_frames"], 8)
        self.assertEqual(payload["summary"]["unreviewed"], 1)
        issue = payload["issues"][0]
        self.assertEqual((issue["id"], issue["frames"], issue["reasons"], issue["state"]), ("4-5", [4, 5], ["head/tail uncertain"], "unreviewed"))
        self.assertEqual(issue["refit"], {"algorithm": "mirror", "label": "match head/tail to the neighbouring frames"})
        # Looks OK needs the current revision.
        self.request("POST", "/issues/review", {"first": 4, "last": 5, "revision": "stale"}, 409)
        reviewed = self.request("POST", "/issues/review", {"first": 4, "last": 5, "revision": payload["revision"]})
        self.assertEqual((reviewed["issues"][0]["state"], reviewed["summary"]["done"]), ("reviewed", 1))

        # Refit: a job whose preview the answer names.
        bad = self.submit("/fixes/refit", {"first": 4, "last": 5, "algorithm": "teleport"}, 400)
        self.assertIn("unknown algorithm", bad["error"])
        self.assertIn("after", self.submit("/fixes/refit", {"first": 5, "last": 4}, 400)["error"])
        self.assertIn("not in workspace", self.submit("/fixes/refit", {"first": 4, "last": 99}, 400)["error"])
        self.assertIn("above the maximum", self.submit("/fixes/refit", {"first": 4, "last": 5, "algorithm": "beam_path", "params": {"beam": 99}}, 400)["error"])
        submitted = self.submit("/fixes/refit", {"first": 4, "last": 5})
        self.assertEqual(submitted["plan"], {
            "algorithm": "mirror", "label": "match head/tail to the neighbouring frames", "frames": [4, 5],
            "anchors": {"before": {"row": 3, "frame": 3}, "after": {"row": 6, "frame": 6}}, "reasons": ["head/tail uncertain"], "codes": ["head_tail"],
        })
        self.assertEqual((submitted["job"]["spec"]["kind"], submitted["job"]["spec"]["workspace"]), ("fix", WS))
        record = self.wait_for_job(submitted["job"]["id"])
        self.assertEqual(record["state"], "done", record)
        preview_id = submitted["preview"]
        self.assertEqual(record["result"]["preview"], preview_id)
        listed = self.request("GET", "/fixes/previews")
        self.assertEqual([p["id"] for p in listed], [preview_id])
        self.assertEqual((listed[0]["kind"], listed[0]["algorithm"], listed[0]["job"], listed[0]["frames_placed"]), ("refit", "mirror", record["id"], 2))
        preview = self.request("GET", f"/fixes/previews/{preview_id}")
        self.assertIsNone(preview["stale"])
        self.assertEqual([f["frame"] for f in preview["per_frame"]], [4, 5])
        frame = preview["per_frame"][0]
        self.assertEqual(frame["before"]["centerline_xy"], frame["after"]["centerline_xy"][::-1])
        self.assertEqual((preview["metrics_before"]["orientation_flips"], preview["metrics_after"]["orientation_flips"]), (2, 0))
        self.request("GET", "/fixes/previews/p999999", status=404)

        # Keep answers like an edit, with the fixes list; the issue is fixed.
        kept = self.request("POST", f"/fixes/previews/{preview_id}/keep", {"frame": 5})
        self.assertEqual((kept["edit"]["kind"], kept["edit"]["rows"], kept["frame"]["stats"]["frame_index"]), ("set_pose", [4, 5], 5))
        self.assertEqual([(f["kind"], f["frames"]) for f in kept["fixes"]], [("refit", [4, 5])])
        self.assertEqual(kept["previews"], [])
        self.assertEqual(self.request("GET", "/issues")["issues"][0]["state"], "fixed")
        self.request("POST", f"/fixes/previews/{preview_id}/keep", {}, 404)

        # Undo from the fixes list.
        fixes_listed = self.request("GET", "/fixes")
        self.assertEqual(fixes_listed["count"], 1)
        undone = self.request("POST", f"/fixes/{kept['edit']['edit_id']}/undo", {})
        self.assertEqual((undone["edit"]["kind"], undone["fixes"]), ("undo", []))
        self.assertIn("no fix", self.request("POST", f"/fixes/{kept['edit']['edit_id']}/undo", {}, 400)["error"])
        self.assertEqual(self.request("GET", "/issues")["issues"][0]["state"], "unreviewed")

        # Flip the whole issue, then one frame back; the list says so and the first waits for the second.
        flipped = self.request("POST", "/fixes/flip", {"first": 4, "last": 5})
        self.assertEqual((flipped["edit"]["kind"], flipped["edit"]["rows"]), ("flip_orientation", [4, 5]))
        one = self.request("POST", "/fixes/flip", {"frame": 5})
        listed = self.request("GET", "/fixes")["fixes"]
        self.assertEqual([(f["title"], f["note"], f["undoable"]) for f in listed], [
            ("Flipped head/tail", "Flip head/tail frame 5", True), ("Flipped head/tail", "Flip head/tail frames 4-5", False),
        ])
        self.assertEqual(listed[1]["blocked_by"], one["edit"]["edit_id"])
        self.request("POST", f"/fixes/{one['edit']['edit_id']}/undo", {})
        self.request("POST", f"/fixes/{flipped['edit']['edit_id']}/undo", {})
        self.assertIn("'first' is required", self.request("POST", "/fixes/flip", {"last": 4}, 400)["error"])
        self.assertEqual(self.client.get("/api/workspaces/missing/issues").status_code, 404)

    @slow
    def test_keyframes_and_stitch(self) -> None:
        self.assertEqual(self.request("GET", "/fixes/keyframes?first=0&last=9"), {"frames": [0, 9], "spacing": 10})
        self.assertEqual(self.request("GET", "/fixes/keyframes?first=0&last=9&spacing=4")["frames"], [0, 3, 6, 9])
        self.assertIn("is after", self.request("GET", "/fixes/keyframes?first=5&last=2", status=400)["error"])

        profile = (12.0 * default_width_template(100)).tolist()
        keyframes = [{"frame": f, "centerline_xy": _body_curve(f).tolist(), "width_profile": profile} for f in (3, 6)]
        self.assertIn("non-empty list", self.submit("/fixes/stitch", {"keyframes": []}, 400)["error"])
        self.assertIn("needs frame", self.submit("/fixes/stitch", {"keyframes": [{"frame": 3}]}, 400)["error"])
        self.assertIn("same frame", self.submit("/fixes/stitch", {"keyframes": [keyframes[0], keyframes[0]]}, 400)["error"])
        self.assertIn("not in workspace", self.submit("/fixes/stitch", {"keyframes": [{**keyframes[0], "frame": 50}]}, 400)["error"])
        submitted = self.submit("/fixes/stitch", {"keyframes": keyframes, "params": {"beam": 1, "anchor_diversity": False}})
        self.assertEqual(submitted["plan"]["keyframes"], [3, 6])
        self.assertEqual(submitted["plan"]["frames"], [3, 6])
        record = self.wait_for_job(submitted["job"]["id"])
        self.assertEqual(record["state"], "done", record)
        preview = self.request("GET", f"/fixes/previews/{submitted['preview']}")
        self.assertEqual((preview["kind"], preview["keyframes"], [f["frame"] for f in preview["per_frame"]]), ("stitch", [3, 6], [3, 4, 5, 6]))
        self.assertEqual(preview["label"], "refit between the keyframes")
        for frame in preview["per_frame"]:
            self.assertGreater(frame["after"]["iou"], 0.85)
            after = np.asarray(frame["after"]["centerline_xy"])
            self.assertLess(np.linalg.norm(after - _body_curve(frame["frame"]), axis=1).mean(), 3.0)
        discarded = self.request("DELETE", f"/fixes/previews/{submitted['preview']}")
        self.assertEqual((discarded["removed"], discarded["previews"]), (True, []))
        self.request("DELETE", f"/fixes/previews/{submitted['preview']}", status=404)


if __name__ == "__main__":
    unittest.main()
