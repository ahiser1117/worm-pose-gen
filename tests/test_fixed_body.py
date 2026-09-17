"""Fixed geometry, censored observations, separate persistence and app delivery."""
from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import time
import unittest

import h5py
import numpy as np
from fastapi.testclient import TestClient

from worm_pose_gen import pipeline
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.batch_fit import BatchFitConfig
from worm_pose_gen.fixed_body import FILENAME, FixedBodyResult, fit_chain
from worm_pose_gen.latent import encode_centerline
from worm_pose_gen.workspace import Workspace


def make_workspace(root: Path) -> Workspace:
    recording = root / "recording.h5"
    with h5py.File(recording, "w") as handle:
        handle.create_dataset("/img_nir", data=np.full((7, 100, 160), 200, dtype=np.uint8))
    workspace = Workspace.create(root / "workspaces", "body", recording, 0, 6)
    arrays = pipeline.new_arrays(workspace.frame_index, BatchFitConfig())
    arrays["ambiguity_score"] = np.zeros(7)
    for row, length in enumerate([80, 78, 82, 80, 80, 140]):
        theta = .5 * np.sin(np.linspace(0, 2 * np.pi, 99)) if row == 3 else np.zeros(99)
        head = np.array([130., 50.]) if row >= 4 else np.array([20., 40.])
        curve = np.vstack((head, head + np.cumsum(length / 99 * np.column_stack((np.cos(theta), np.sin(theta))), axis=0)))
        if row == 5:
            curve[:, 0] = 130 + 70 * np.sin(np.linspace(0, np.pi, 100))
        arrays["centerline_xy"][row] = curve
        arrays["latent"][row] = encode_centerline(curve)
        arrays["fitted"][row] = True
        arrays["width_profile"][row] = 4 + 2 * np.sin(np.linspace(0, np.pi, 100))
        arrays["width_px"][row] = 6
        arrays["body_length_px"][row] = length
        arrays["iou"][row] = .98
        arrays["energy"][row] = arrays["total_energy"][row] = .01
        arrays["width_shape"][row] = 0
        arrays["points_in_fov"][row] = 100 if row < 4 else 30
    # This bent frame is fitted, but deliberately excluded from calibration.
    arrays["ambiguity_score"][3] = 3
    workspace.save_state(arrays)
    return workspace


class FixedBodyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = make_workspace(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **params):
        return pipeline.run_stage(self.workspace, "fixed_body", params, device="cpu")

    def test_fixed_geometry_head_and_straight_completion_without_pose_changes(self):
        before = self.workspace.state_path.read_bytes()
        metadata = self.build()
        self.assertEqual(metadata["anchor_frames"], [0, 1, 2])
        self.assertAlmostEqual(metadata["length_px"], 80.)
        result = FixedBodyResult(self.workspace.path)
        self.assertFalse(result.stale)
        for row in range(5):
            body = result.frame(row)
            curve = np.asarray(body["centerline_xy"])
            np.testing.assert_allclose(np.linalg.norm(np.diff(curve, axis=0), axis=1), 80 / 99, atol=1e-10)
            np.testing.assert_array_equal(curve[0], self.workspace.load_state()["centerline_xy"][row, 0])
            np.testing.assert_array_equal(body["width_profile"], result.frame(0)["width_profile"])
        self.assertLess(result.frame(3)["rms_px"], .1)
        clipped = result.frame(4)
        first_extra = np.flatnonzero(clipped["extrapolated"])[0]
        links = np.diff(clipped["centerline_xy"], axis=0)
        np.testing.assert_allclose(links[first_extra - 1:], np.tile(links[first_extra - 2], (len(links) - first_extra + 1, 1)), atol=1e-10)
        self.assertIsNone(clipped["length_error_px"])
        self.assertAlmostEqual(result.frame(1)["length_error_px"], -2)
        self.assertAlmostEqual(result.frame(2)["length_error_px"], 2)
        self.assertEqual(result.frame(5)["status"], "reentry_unresolved")
        self.assertNotIn("centerline_xy", result.frame(5))
        self.assertEqual(result.frame(6)["status"], "unfitted_or_stale")
        self.assertEqual(self.workspace.state_path.read_bytes(), before)
        self.assertFalse(self.workspace.provenance_path.exists())
        self.assertFalse(self.workspace.edits_path.exists())

    def test_edge_cases_do_not_invent_visible_heads_or_tiny_chains(self):
        points = np.column_stack((np.linspace(-5, 50, 100), np.full(100, 30)))
        self.assertEqual(fit_chain(points, 80, 99, (100, 160))["status"], "head_outside")
        points[:, 0] = np.linspace(158.9, 200, 100)
        self.assertEqual(fit_chain(points, 80, 4, (100, 160))["status"], "insufficient_visible_body")

    def test_calibration_rejects_bad_frames_and_preserves_previous_result_on_failure(self):
        self.build()
        before = (self.workspace.path / FILENAME).read_bytes()
        arrays = self.workspace.load_state()
        arrays["mask_stale"][0] = True
        self.workspace.save_state(arrays)
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.build()
        self.assertEqual((self.workspace.path / FILENAME).read_bytes(), before)
        self.assertTrue(FixedBodyResult(self.workspace.path).stale)
        with self.assertRaisesRegex(ValueError, "fully visible"):
            self.build(anchor_frames="0,1,2")
        self.build(anchor_frames="1,2,3")
        mask = np.zeros((100, 160), dtype=bool)
        mask[0, 40] = True
        self.workspace.set_masks([1], [mask])
        with self.assertRaisesRegex(ValueError, "border"):
            self.build(anchor_frames="1,2,3")

    def test_opt_in_and_invalid_parameters(self):
        self.assertNotIn("fixed_body", inspect.signature(pipeline.run_all).parameters["stages"].default)
        for params in ({"segments": 0}, {"segments": 10.5}, {"min_iou": float("nan")}, {"min_anchors": 0}, {"anchor_frames": "999"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.build(**params)

    def test_app_job_and_full_light_pose_payloads_refresh_after_edits(self):
        config = AppConfig(workspaces_root=self.root / "workspaces", recording_roots=(self.root,),
                           poses_root=self.root / "runs", dataset_root=self.root / "cache", checkpoint=None,
                           notes=self.root / "notes.json", prior_cache=None, gpus=(), device="cpu", job_interval=.05)
        with TestClient(create_app(config)) as client:
            stages = client.get("/api/stages").json()
            self.assertIn("fixed_body", [s["name"] for s in stages])
            response = client.post("/api/jobs", json={"kind": "stage", "stage": "fixed_body", "workspace": "body"})
            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["id"]
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["state"] in ("done", "failed", "cancelled"):
                    break
                time.sleep(.1)
            self.assertEqual(job["state"], "done", str(job))
            for endpoint in ("/api/frame?run=body&frame=4&detail=light", "/api/frame?run=body&frame=4", "/api/pose?run=body&frame=4"):
                response = client.get(endpoint)
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()["fixed_body"]
                self.assertFalse(body["stale"])
                self.assertEqual(len(body["centerline_xy"]), 100)
                self.assertTrue(any(body["extrapolated"]))
            arrays = self.workspace.load_state()
            arrays["centerline_xy"][0] = arrays["centerline_xy"][0, ::-1]
            self.workspace.save_state(arrays)
            body = client.get("/api/frame?run=body&frame=4&detail=light").json()["fixed_body"]
            self.assertTrue(body["stale"])
            self.assertNotIn("centerline_xy", body)


if __name__ == "__main__":
    unittest.main()
