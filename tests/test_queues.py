"""The Labeling page's backend: frame search, queues, opening and saving frames, body proposals, and the Relabel stitch.

Jobs that build body targets are replaced by a no-op command (the target
cache has its own tests); the Find-frames and stitch jobs run for real on
the CPU.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import h5py
import numpy as np
from fastapi.testclient import TestClient

from worm_pose_gen import body_fields, fixes, frame_search, library
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.app import labeling as labeling_service
from worm_pose_gen.batch_fit import PRESETS
from worm_pose_gen.label_app import data_url, mask_to_png_values
from worm_pose_gen.library.inference import LoadedModel
from worm_pose_gen.library.targets import write_targets
from worm_pose_gen.mask_fit import default_width_template

from tests.test_fixes_api import _cpu_fix_command

NO_OP = [sys.executable, "-c", "pass"]


def mask_url(mask: np.ndarray) -> str:
    return data_url(mask_to_png_values(mask))


def fast_fit_config():
    return replace(PRESETS["fast"], length_bounds_px=body_fields.FIT_LENGTH_BOUNDS_PX)


class FrameSearchTests(unittest.TestCase):
    def test_uncertainty_is_entropy_per_worm_pixel(self):
        sure = np.zeros((40, 40), np.float32)
        sure[10:20, 5:35] = 1.0
        unsure = np.where(sure == 1, 0.6, 0.02).astype(np.float32)
        self.assertLess(frame_search.uncertainty(sure), 0.01)
        self.assertGreater(frame_search.uncertainty(unsure), 0.5)
        # Without a predicted worm the entropy is divided by the minimum area.
        self.assertAlmostEqual(frame_search.uncertainty(np.full((10, 10), 0.5, np.float32)), 100 / frame_search.MIN_AREA, places=4)

    def test_share_windows_and_candidates(self):
        self.assertEqual(frame_search.share(7, [100, 300, 50]), [2, 3, 2])
        self.assertEqual(frame_search.share(5, [2, 100]), [2, 3])  # never more than a recording has
        self.assertEqual(frame_search.windows(100, 4), [(0, 25), (25, 50), (50, 75), (75, 100)])
        self.assertEqual(frame_search.candidates(0, 60, set(), 3), [10, 30, 50])
        self.assertEqual(frame_search.candidates(0, 60, {30}, 3), [10, 29, 50])

    def test_pick_favours_the_least_sure_candidate_of_each_window(self):
        def predict(frames):
            # Frame 37 is the only one the model hesitates on.
            return [np.full((8, 8), 0.5 if f == 37 else 0.01, np.float32) for f in frames]

        picked = frame_search.pick(100, 2, predict, per_window=6)
        self.assertEqual([p["frame"] for p in picked][0], 37)
        self.assertEqual(len(picked), 2)
        self.assertGreater(picked[0]["uncertainty"], picked[1]["uncertainty"])
        # Without a model: the middle candidate of each window, never an excluded frame.
        spread = frame_search.pick(100, 4, None, excluded={40})
        self.assertEqual([p["frame"] for p in spread], [15, 39, 65, 90])
        self.assertTrue(all(p["uncertainty"] is None for p in spread))


class FrameSearchJobTests(unittest.TestCase):
    def test_the_job_runs_a_library_model_over_the_recordings(self):
        import torch

        from tests.test_model_training import save_weights
        from worm_pose_gen.segmenter import SegmentationModule

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = library.Libraries(lab=None, personal=root / "mine")
            library.write_setup(libraries.personal, "rig", name="Rig", fps=20.0, video={"flat_field": False})
            torch.manual_seed(0)
            save_weights(SegmentationModule(pretrained=False), root / "seg.ckpt")
            library.create_model(libraries, "seg", {"name": "seg", "kind": "segmenter", "setup": "mine:rig", "outputs": ["mask"],
                                                    "inputs": library.make_inputs([], fps=20.0, pixel_size_um=1.0)}, root / "seg.ckpt")
            path = root / "2024-05-05-01.h5"
            write_recording(path, frames=30)
            spec = {"recordings": [{"path": str(path), "id": "2024-05-05-01", "frames": 30, "exclude": [15]}], "frames": 3,
                    "model": "mine:seg", "libraries": {"lab": None, "personal": str(libraries.personal)},
                    "video": {"flat_field": False, "dataset_path": "/img_nir"}, "fps": 20.0, "dataset_root": str(root / "cache")}
            progress = root / "progress.json"
            with mock.patch.dict("os.environ", {"WORM_POSE_PROGRESS_FILE": str(progress)}):
                self.assertEqual(frame_search.main(["--spec", json.dumps(spec), "--device", "cpu"]), 0)
            entries = json.loads(progress.read_text())["result"]["entries"]
            self.assertEqual([(e["recording"], e["path"]) for e in entries], [("2024-05-05-01", str(path))] * 3)
            self.assertEqual([e["frame"] // 10 for e in entries], [0, 1, 2])  # one per third of the recording
            self.assertTrue(all(isinstance(e["uncertainty"], float) and e["frame"] != 15 for e in entries))


class LabelingApiBase(unittest.TestCase):
    """A personal setup ``rig`` whose root holds the recordings, an app on it, and its client."""

    gpus: tuple[int, ...] = ()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.recordings = self.root / "recordings"
        self.recordings.mkdir()
        self.libraries = library.Libraries(lab=None, personal=self.root / "mine")
        library.write_setup(self.libraries.personal, "rig", name="Rig", fps=20.0, pixel_size_um=1.0,
                            video={"flat_field": False}, recording_roots=[str(self.root)])
        self.make_app()

    def make_app(self):
        config = AppConfig(
            workspaces_root=self.root / "workspaces", recording_roots=(self.recordings,), poses_root=self.root / "poses",
            dataset_root=self.root / "cache", checkpoint=None, prior_cache=None, notes=self.root / "notes.json", device="cpu",
            gpus=self.gpus, job_interval=0.1, lab_library=Path("/nonexistent-lab"), library=self.libraries.personal,
        )
        self.app = create_app(config)
        self.state = self.app.state.app_state
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        patcher = mock.patch.object(labeling_service, "targets_command", lambda libraries, identities: NO_OP)
        patcher.start()
        self.addCleanup(patcher.stop)

    def call(self, method, url, payload=None, status=200):
        response = self.client.request(method, url, json=payload)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def wait_for_job(self, job_id, timeout=600.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.call("GET", f"/api/jobs/{job_id}")
            if record["state"] in ("done", "failed", "cancelled"):
                return record
            time.sleep(0.2)
        self.fail(f"job {job_id} did not finish")


def write_recording(path: Path, frames: int = 40, shape=(48, 64)) -> np.ndarray:
    """Dark bars (value 60) on a bright background, one per frame at a different height; returns the masks."""

    masks = np.zeros((frames, *shape), np.uint8)
    for k in range(frames):
        top = 8 + k % 24
        masks[k, top : top + 6, 8:56] = 1
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=np.where(masks == 1, 60, 200).astype(np.uint8))
    return masks


class LabelingApiTests(LabelingApiBase):
    def setUp(self):
        super().setUp()
        self.path = self.recordings / "2024-05-05-01.h5"
        self.masks = write_recording(self.path)

    def entry(self, frame, path=True):
        return {"recording": "2024-05-05-01", "frame": frame, **({"path": str(self.path)} if path else {})}

    def test_open_save_creates_the_dataset_and_reopens_from_the_label(self):
        saving = self.call("GET", "/api/labeling/saving?setup=mine:rig")
        self.assertEqual((saving["dataset"], saving["create"], saving["extends"], saving["choices"]), (None, "mine:rig-labels", None, []))
        opened = self.call("POST", "/api/labeling/open", {"setup": "mine:rig", "entry": self.entry(5)})
        self.assertEqual((opened["mask_source"], opened["label"], opened["expected_revision"], opened["max_lag"]), ("empty", None, 0, 16))
        self.assertEqual((opened["width"], opened["height"], opened["models"]), (64, 48, {"mask": None, "body": None}))
        self.assertEqual(opened["context_valid"][:11], [False] * 11)
        self.assertIn("train", opened["split_note"] or "train")
        refused = self.client.post("/api/labeling/network", json={"setup": "mine:rig", "entry": self.entry(5)})
        self.assertIn("no default mask model", refused.json()["error"])

        refined = self.call("POST", "/api/labeling/refine", {"mask": mask_url(self.masks[5]), "width": 64, "height": 48, "method": "grow"})
        self.assertTrue(refined["mask"].startswith("data:image/png"))
        self.call("POST", "/api/labeling/refine", {"mask": mask_url(self.masks[5]), "width": 64, "height": 48, "method": "tube"}, 400)

        saved = self.call("POST", "/api/labeling/save", {
            "setup": "mine:rig", "entry": self.entry(5), "mask": mask_url(self.masks[5]), "head_xy": [10, 15], "expected_revision": 0,
        })
        self.assertEqual((saved["dataset"], saved["created"], saved["label"]["origin"], saved["label"]["status"]), ("mine:rig-labels", True, "spread", "complete"))
        self.assertEqual((saved["job"]["spec"]["kind"], saved["job"]["command"]), ("body_targets", NO_OP))
        self.assertIsNone(saved["queue"])
        dataset = library.Dataset(self.libraries, "mine:rig-labels")
        self.assertIsNone(dataset.extends)
        label = dataset.get("2024-05-05-01", 5).load()
        np.testing.assert_array_equal(label.mask, self.masks[5])
        np.testing.assert_array_equal(label.head_xy, [10, 15])

        # The label now opens from itself: no recording path needed.
        reopened = self.call("POST", "/api/labeling/open", {"setup": "mine:rig", "entry": self.entry(5, path=False)})
        self.assertEqual((reopened["mask_source"], reopened["expected_revision"], reopened["split"]), ("label", 1, "train"))
        self.assertEqual(reopened["body"], {"trace_xy": None, "head_xy": [10.0, 15.0], "mask_only": False})
        context = self.call("POST", "/api/labeling/context", {"setup": "mine:rig", "entry": self.entry(5, path=False)})
        self.assertEqual((len(context["frames"]), context["valid"][16]), (33, True))
        stale = self.client.post("/api/labeling/save", json={"setup": "mine:rig", "entry": self.entry(5), "mask": mask_url(self.masks[5]),
                                                              "expected_revision": 0})
        self.assertIn("changed since it was opened", stale.json()["error"])
        trace = [[10, 11], [30, 11], [50, 11]]
        again = self.call("POST", "/api/labeling/save", {"setup": "mine:rig", "entry": self.entry(5), "mask": mask_url(self.masks[5]),
                                                          "trace_xy": trace, "mask_only": True, "expected_revision": 1})
        self.assertEqual((again["created"], again["label"]["revision"], again["label"]["status"]), (False, 2, "mask_only"))
        self.call("POST", "/api/labeling/open", {"setup": "mine:rig", "entry": self.entry(6, path=False)}, 404)

    def test_a_new_queue_finds_frames_in_a_job(self):
        other = self.recordings / "2024-05-05-02.h5"
        write_recording(other, frames=20)
        self.call("POST", "/api/labeling/save", {"setup": "mine:rig", "entry": self.entry(4), "mask": mask_url(self.masks[4])})
        self.call("POST", "/api/queues", {"kind": "spread", "setup": "mine:rig", "recordings": [str(self.path)], "frames": 0}, 400)
        elsewhere = tempfile.TemporaryDirectory()
        self.addCleanup(elsewhere.cleanup)
        stray = Path(elsewhere.name) / "stray.h5"
        write_recording(stray, frames=5)
        self.assertIn("belongs to no setup", self.call("POST", "/api/queues", {"kind": "spread", "setup": "mine:rig", "recordings": [str(stray)], "frames": 2}, 400)["error"])
        created = self.call("POST", "/api/queues", {"kind": "spread", "setup": "mine:rig", "recordings": [str(self.path), str(other)], "frames": 5})
        self.assertEqual((created["state"], created["kind"], created["origin"], created["progress"]["total"]), ("finding", "spread", "spread", 0))
        spec = json.loads(self.state.runner.get(created["job"]).command[-1])
        self.assertEqual(spec["recordings"][0]["exclude"], [4])  # already labeled
        self.assertEqual(self.wait_for_job(created["job"])["state"], "done")
        queue = self.call("GET", f"/api/queues/{created['id']}")
        self.assertEqual(queue["state"], "ready")
        # No model: three frames spread over the longer recording, two over the shorter.
        self.assertEqual([(e["recording"], e["frame"]) for e in queue["entries"]],
                         [("2024-05-05-01", 8), ("2024-05-05-01", 21), ("2024-05-05-01", 35), ("2024-05-05-02", 6), ("2024-05-05-02", 16)])

        first = queue["entries"][0]
        saved = self.call("POST", "/api/labeling/save", {"setup": "mine:rig", "queue": queue["id"], "entry": first,
                                                          "mask": mask_url(self.masks[first["frame"]])})
        self.assertEqual((saved["label"]["origin"], saved["queue"]["progress"]), ("spread", {"total": 5, "saved": 1, "remaining": 4}))
        self.assertEqual(saved["queue"]["first_unsaved"], 1)
        listed = self.call("GET", "/api/queues?setup=mine:rig")["queues"]
        self.assertEqual([q["id"] for q in listed], [queue["id"]])
        self.assertNotIn("entries", listed[0])
        # The queue survives a restart of the app.
        self.make_app()
        self.assertEqual(self.call("GET", f"/api/queues/{queue['id']}")["progress"]["saved"], 1)
        self.call("DELETE", f"/api/queues/{queue['id']}")
        self.call("GET", f"/api/queues/{queue['id']}", status=404)


class ProposalTests(unittest.TestCase):
    """The body proposal and the trace fit, with a stub network that predicts the true fields of a looped body."""

    @mock.patch.object(body_fields, "fit_config", fast_fit_config)
    def test_proposal_and_trace_fit_of_a_label(self):
        from tests.test_body_proposal import StubModule, looped_body, prediction_from

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = library.Libraries(lab=None, personal=root / "mine")
            library.write_setup(libraries.personal, "rig", name="Rig")
            weights = root / "w.ckpt"
            weights.write_bytes(b"w")
            library.create_model(libraries, "body", {"name": "body", "kind": "body_net", "setup": "mine:rig",
                                                     "outputs": ["mask", "ap", "head", "tail", "overlap"],
                                                     "inputs": library.make_inputs([], fps=20.0, pixel_size_um=1.0)}, weights)
            library.set_default(libraries, "mine:rig", "body", "mine:body", reason="test")
            centerline, mask = looped_body()
            image = np.where(mask, 60, 200).astype(np.uint8)
            context = np.repeat(image[None], 5, 0)
            dataset = library.create_dataset(libraries, "rig-labels", setup="mine:rig")
            dataset.save(recording="rec", frame=3, image=image, image_raw=image, mask=mask.astype(np.uint8), context=context,
                         context_valid=np.ones(5, bool), origin="spread")
            app = create_app(AppConfig(workspaces_root=root / "workspaces", recording_roots=(root,), poses_root=root / "poses",
                                       dataset_root=root / "cache", checkpoint=None, prior_cache=None, notes=root / "notes.json",
                                       device="cpu", gpus=(), lab_library=root / "nolab", library=libraries.personal))
            service = app.state.app_state.labeling
            card = library.get_card(libraries, "mine:body")
            stub = LoadedModel(card, StubModule(prediction_from(centerline, mask)), lags=())
            with mock.patch.object(service, "model", return_value=stub):
                request = {"setup": "mine:rig", "entry": {"recording": "rec", "frame": 3}, "mask": mask_url(mask.astype(np.uint8))}
                proposal = service.proposal(request)
                self.assertEqual(proposal["status"], "ready")
                np.testing.assert_allclose(proposal["trace_xy"][0], [80, 150], atol=3)
                np.testing.assert_allclose(proposal["head_xy"], [80, 150], atol=12)
                self.assertGreater(proposal["fit_iou"], 0.8)
                self.assertTrue(proposal["ap"].startswith("data:image/png"))
                fitted = service.fit({**request, "trace_xy": proposal["trace_xy"][::-1]})
                np.testing.assert_allclose(fitted["centerline_xy"][0], proposal["centerline_xy"][-1], atol=15)
                empty = service.proposal({**request, "mask": mask_url(np.zeros(mask.shape, np.uint8))})
                self.assertEqual(empty["status"], "no_trace")
            app.state.app_state.close()


class RelabelStitchTests(LabelingApiBase):
    """Workspace Relabel → queue → labels → stitch, on the posed synthetic workspace of tests/test_fixes.py."""

    gpus = (0,)

    def setUp(self):
        from tests.test_fixes import posed_workspace

        super().setUp()
        # The recording <root>/synthetic.h5 (inside the setup's root) and the workspace <root>/workspaces/synthetic.
        posed_workspace(self.root, "synthetic")

    def test_relabel_round_trip(self):
        from tests.test_pipeline import _body_curve

        self.call("POST", "/api/queues", {"kind": "relabel", "workspace": "synthetic", "frames": [2, 99]}, 400)
        created = self.call("POST", "/api/queues", {"kind": "relabel", "workspace": "synthetic", "frames": [7, 2, 4]})
        self.assertEqual((created["kind"], created["origin"], created["workspace"], created["setup"]), ("relabel", "fix", "synthetic", "mine:rig"))
        queue = self.call("GET", f"/api/queues/{created['id']}")
        self.assertEqual([e["frame"] for e in queue["entries"]], [2, 4, 7])
        self.assertIn("not labeled yet", self.call("POST", f"/api/queues/{queue['id']}/stitch", {}, 400)["error"])

        workspace = self.state.workspace("synthetic")
        for k, entry in enumerate(queue["entries"]):
            opened = self.call("POST", "/api/labeling/open", {"setup": "mine:rig", "queue": queue["id"], "entry": entry})
            self.assertEqual(opened["mask_source"], "workspace")
            mask = workspace.get_mask(workspace.row_of(entry["frame"])).astype(np.uint8)
            if k == 0:
                mask[0, 0] = 1  # a hand edit, which the stitch writes back as an override
            saved = self.call("POST", "/api/labeling/save", {"setup": "mine:rig", "queue": queue["id"], "entry": entry, "mask": mask_url(mask),
                                                              "mask_only": k == 1, "expected_revision": 0})
            self.assertEqual(saved["label"]["origin"], "fix")
            # The targets the background job would build: the true body, head first.
            record = library.Dataset(self.libraries, saved["dataset"]).get(entry["recording"], entry["frame"])
            curve = _body_curve(entry["frame"])
            write_targets(self.libraries, record.sha256, None, {"has_body": True, "fit_iou": 0.97},
                          {"centerline_xy": curve, "width_profile": 12.0 * default_width_template(len(curve))})
        self.assertTrue(saved["queue"]["complete"])

        with mock.patch.object(fixes, "fix_command", _cpu_fix_command):
            answer = self.call("POST", f"/api/queues/{queue['id']}/stitch", {})
        self.assertEqual(set(answer), {"job", "preview", "plan"})
        self.assertEqual((answer["plan"]["keyframes"], answer["plan"]["frames"]), ([2, 7], [2, 7]))  # frame 4 is mask-only
        self.assertEqual(answer["job"]["spec"]["workspace"], "synthetic")
        workspace = self.state.workspace("synthetic")
        rows = [workspace.row_of(frame) for frame in (2, 4, 7)]
        self.assertEqual(workspace.override_rows(), rows)
        self.assertEqual(workspace.get_override_mask(rows[0])[0, 0], 1)
        record = self.wait_for_job(answer["job"]["id"])
        self.assertEqual(record["state"], "done", record)
        preview = self.call("GET", f"/api/workspaces/synthetic/fixes/previews/{answer['preview']}")
        self.assertEqual([f["frame"] for f in preview["per_frame"]], list(range(2, 8)))


if __name__ == "__main__":
    unittest.main()
