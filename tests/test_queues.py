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
from worm_pose_gen.app.images import data_url, decode_mask_data_url, mask_to_png_values, png_values_to_mask
from worm_pose_gen.segmenter import IGNORE_LABEL
from worm_pose_gen.library.inference import LoadedModel, Outputs
from worm_pose_gen.library.targets import write_targets
from worm_pose_gen.mask_fit import default_width_template

from tests.slow import slow
from tests.test_fixes_api import _cpu_fix_command

NO_OP = [sys.executable, "-c", "pass"]


def mask_url(mask: np.ndarray) -> str:
    return data_url(mask_to_png_values(mask))


def fast_fit_config():
    return replace(PRESETS["fast"], length_bounds_px=body_fields.FIT_LENGTH_BOUNDS_PX)


class MaskToolTests(unittest.TestCase):
    def test_png_label_conventions_round_trip(self) -> None:
        mask = np.array([[0, 1, IGNORE_LABEL]], dtype=np.uint8)
        png = mask_to_png_values(mask)
        self.assertEqual(png.tolist(), [[0, 255, 128]])
        self.assertTrue(np.array_equal(png_values_to_mask(png), mask))
        decoded = decode_mask_data_url(data_url(png), (1, 3))
        self.assertTrue(np.array_equal(decoded, mask))
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_mask_data_url(data_url(png), (2, 3))

    def test_refinements_preserve_ignore_and_change_worm(self) -> None:
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[10:30, 10:50] = 1
        mask[18:22, 28:32] = 0  # small hole
        mask[2:4, 2:4] = 1  # debris
        mask[35, 35] = IGNORE_LABEL
        filled, info = labeling_service.refine_mask(mask, "fill_holes", "cpu")
        self.assertTrue(filled[18:22, 28:32].all())
        self.assertEqual(filled[35, 35], IGNORE_LABEL)
        self.assertEqual(info["pixels_added"], 16)
        largest, info = labeling_service.refine_mask(mask, "largest", "cpu")
        self.assertFalse(largest[2:4, 2:4].any())
        self.assertEqual(info["components_removed"], 1)
        grown, _ = labeling_service.refine_mask(mask, "grow", "cpu")
        self.assertGreater(int((grown == 1).sum()), int((mask == 1).sum()))
        shrunk, _ = labeling_service.refine_mask(mask, "shrink", "cpu")
        self.assertLess(int((shrunk == 1).sum()), int((mask == 1).sum()))
        with self.assertRaisesRegex(ValueError, "unknown refinement"):
            labeling_service.refine_mask(mask, "nope", "cpu")


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
            return [Outputs(mask=np.full((8, 8), 0.5 if f == 37 else 0.01, np.float32)) for f in frames]

        picked = frame_search.pick(100, 2, predict, per_window=6)
        self.assertEqual([p["frame"] for p in picked][0], 37)
        self.assertEqual(len(picked), 2)
        self.assertGreater(picked[0]["uncertainty"], picked[1]["uncertainty"])
        # Without a model: the middle candidate of each window, never an excluded frame.
        spread = frame_search.pick(100, 4, None, excluded={40})
        self.assertEqual([p["frame"] for p in spread], [15, 39, 65, 90])
        self.assertTrue(all(p["uncertainty"] is None and p["types"] is None for p in spread))

    def test_frame_types(self):
        def mask(*boxes):
            image = np.zeros((60, 60), np.float32)
            for y0, y1, x0, x1 in boxes:
                image[y0:y1, x0:x1] = 1.0
            return image

        body = mask((20, 30, 10, 50))
        self.assertEqual(frame_search.frame_types(Outputs(mask=body)), ["clear"])
        self.assertEqual(frame_search.frame_types(Outputs(mask=np.zeros((60, 60), np.float32))), ["empty"])
        self.assertEqual(frame_search.frame_types(Outputs(mask=mask((20, 30, 0, 40)))), ["edge"])
        self.assertEqual(frame_search.frame_types(Outputs(mask=mask((20, 30, 10, 50), (40, 45, 10, 20)))), ["pieces"])
        # A loop encloses a hole; a body-field model's overlap output says so directly.
        loop = mask((10, 50, 10, 50))
        loop[18:42, 18:42] = 0
        self.assertEqual(frame_search.frame_types(Outputs(mask=loop)), ["contact"])
        overlap = np.zeros_like(body)
        overlap[20:30, 25:35] = 1
        self.assertEqual(frame_search.frame_types(Outputs(mask=body, overlap=overlap)), ["contact"])

    def test_pick_limited_to_types(self):
        clear, edge = np.zeros((60, 60), np.float32), np.zeros((60, 60), np.float32)
        clear[20:30, 10:50] = 0.9
        edge[20:30, 0:40] = 0.9

        def predict(frames):
            # Only frames 5, 12 and 13 show the body at the edge, all in the first of four windows.
            return [Outputs(mask=edge if f in (5, 12, 13) else clear) for f in frames]

        picked = frame_search.pick(100, 4, predict, per_window=25, types=["edge"])
        self.assertEqual([p["frame"] for p in picked], [5, 12, 13])  # the other windows give their frames to these
        self.assertTrue(all(p["types"] == ["edge"] for p in picked))
        self.assertEqual(len(frame_search.pick(100, 4, predict, per_window=25)), 4)
        with self.assertRaises(ValueError):
            frame_search.find_frames([{"path": "x", "id": "x", "frames": 10}], 2, lambda r: (None, lambda: None), types=["edge"])
        with self.assertRaises(ValueError):
            frame_search.find_frames([{"path": "x", "id": "x", "frames": 10}], 2, lambda r: (None, lambda: None), types=["blurry"])


class FrameSearchJobTests(unittest.TestCase):
    @slow
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
                    "video": {"flat_field": False, "dataset_path": "/img_nir"}, "fps": 20.0, "dataset_root": str(root / "cache"),
                    "prior_cache": str(root / "priors")}
            progress = root / "progress.json"
            with mock.patch.dict("os.environ", {"WORM_POSE_PROGRESS_FILE": str(progress)}):
                self.assertEqual(frame_search.main(["--spec", json.dumps(spec), "--device", "cpu"]), 0)
            entries = json.loads(progress.read_text())["result"]["entries"]
            self.assertEqual([(e["recording"], e["path"]) for e in entries], [("2024-05-05-01", str(path))] * 3)
            self.assertEqual([e["frame"] // 10 for e in entries], [0, 1, 2])  # one per third of the recording
            self.assertTrue(all(isinstance(e["uncertainty"], float) and e["frame"] != 15 for e in entries))
            self.assertTrue(all(set(e["types"]) <= set(frame_search.TYPES) and e["types"] for e in entries))
            # The job also leaves a body-length estimate per recording (none here: an untrained model finds no whole body).
            self.assertEqual(list(json.loads(progress.read_text())["result"]["lengths"]), ["2024-05-05-01"])


class BodyLengthTests(unittest.TestCase):
    """Where the body length a trace is extended to comes from (``Labeling.body_length``, ``frame_search.estimate_lengths``)."""

    def test_the_first_source_with_a_length_wins(self):
        from types import SimpleNamespace

        from worm_pose_gen import pipeline
        from worm_pose_gen.workspace import Workspace

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = library.Libraries(lab=None, personal=root / "mine")
            library.write_setup(libraries.personal, "rig", name="Rig")
            app = create_app(AppConfig(workspaces_root=root / "workspaces", dataset_root=root / "cache", device="cpu", gpus=(),
                                       lab_library=root / "nolab", library=libraries.personal))
            state = app.state.app_state
            service = state.labeling
            entry = {"recording": "rec", "frame": 3, "path": str(root / "rec.h5")}
            prior = lambda length: SimpleNamespace(length_px=length)  # noqa: E731
            sources = {"labels": None, "workspace": None, "analysis": None, "estimate": None, "setup": None}

            def length():
                with mock.patch.object(labeling_service, "recording_length", return_value=sources["labels"]), \
                        mock.patch.object(labeling_service, "setup_length", return_value=sources["setup"]), \
                        mock.patch.object(state, "workspace_of_recording", return_value=sources["workspace"]), \
                        mock.patch.object(Workspace, "open", return_value=None), \
                        mock.patch.object(pipeline, "workspace_prior", return_value=sources["analysis"]), \
                        mock.patch.object(pipeline, "cached_prior", return_value=sources["estimate"]):
                    return service.body_length("mine:rig", entry, None)

            self.assertEqual(length(), (None, None))
            sources["setup"] = 765.0
            self.assertEqual(length(), (765.0, "setup"))
            sources["estimate"] = prior(740.0)
            self.assertEqual(length(), (740.0, "estimate"))
            sources["workspace"], sources["analysis"] = "rec", prior(750.0)
            self.assertEqual(length(), (750.0, "analysis"))
            sources["labels"] = 776.0
            self.assertEqual(length(), (776.0, "labels"))
            state.close()

    def test_find_frames_estimates_the_recordings_without_one(self):
        from worm_pose_gen import pipeline

        recordings = [{"path": f"/data/{name}.h5", "id": name} for name in ("cached", "new", "empty")]
        cached = {"/data/cached.h5": mock.Mock(length_px=700.0)}

        def bootstrap(frames, params, config, device):
            if Path(frames.path).stem == "empty":
                raise ValueError("no bootstrap mask produced a start")
            return mock.Mock(length_px=760.0), "bootstrap", {}

        with mock.patch.object(pipeline, "cached_prior", side_effect=lambda path, cache: cached.get(path)), \
                mock.patch.object(pipeline, "resolve_prior", side_effect=bootstrap) as resolve, \
                mock.patch.object(pipeline, "Frames", side_effect=lambda path, **_: mock.Mock(path=path)):
            lengths = frame_search.estimate_lengths({"recordings": recordings, "video": {}, "dataset_root": "/cache"}, "seg.ckpt", "cpu")
        self.assertEqual(lengths, {"cached": 700.0, "new": 760.0, "empty": None})
        self.assertEqual(resolve.call_count, 2)  # the cached recording is not bootstrapped again

    def test_a_label_is_extended_to_the_length_saved_with_it(self):
        from worm_pose_gen.library import targets

        record = mock.Mock(setup="mine:rig", recording="rec", sha256="x")
        with mock.patch.object(targets, "recording_length", return_value=776.0):
            self.assertIsNone(targets.extension_length(None, record, {"trace_extend": False, "trace_length_px": 640.0}, None))
            self.assertEqual(targets.extension_length(None, record, {"trace_extend": True, "trace_length_px": 640.0}, None), 640.0)
            self.assertEqual(targets.extension_length(None, record, {"trace_extend": True}, None), 776.0)


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
            workspaces_root=self.root / "workspaces",
            dataset_root=self.root / "cache", device="cpu",
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

    def test_open_save_into_the_collection_and_reopen_from_the_label(self):
        library.create_dataset(self.libraries, "rig-set", setup="mine:rig")
        opened = self.call("POST", "/api/labeling/open", {"setup": "mine:rig", "entry": self.entry(5)})
        self.assertEqual((opened["mask_source"], opened["label"], opened["expected_revision"], opened["max_lag"]), ("empty", None, 0, 16))
        self.assertEqual((opened["width"], opened["height"], opened["models"]), (64, 48, {"mask": None, "body": None}))
        self.assertEqual(opened["context_valid"][:11], [False] * 11)
        refused = self.client.post("/api/labeling/network", json={"setup": "mine:rig", "entry": self.entry(5)})
        self.assertIn("no default mask model", refused.json()["error"])

        refined = self.call("POST", "/api/labeling/refine", {"mask": mask_url(self.masks[5]), "width": 64, "height": 48, "method": "grow"})
        self.assertTrue(refined["mask"].startswith("data:image/png"))
        self.call("POST", "/api/labeling/refine", {"mask": mask_url(self.masks[5]), "width": 64, "height": 48, "method": "tube"}, 400)

        saved = self.call("POST", "/api/labeling/save", {
            "setup": "mine:rig", "entry": self.entry(5), "mask": mask_url(self.masks[5]), "head_xy": [10, 15], "expected_revision": 0,
        })
        self.assertEqual((saved["label"]["scope"], saved["label"]["setup"], saved["label"]["origin"], saved["label"]["status"]),
                         ("mine", "mine:rig", "spread", "complete"))
        self.assertEqual((saved["job"]["spec"]["kind"], saved["job"]["command"]), ("body_targets", NO_OP))
        self.assertIsNone(saved["queue"])
        # The recording joins the setup's collection, and the dataset lists it as not included.
        self.assertEqual(library.Dataset(self.libraries, "mine:rig-set").summary()["not_included"], {"recordings": 1, "labels": 1})
        label = library.Collection(self.libraries, "mine:rig").get("2024-05-05-01", 5).load()
        np.testing.assert_array_equal(label.mask, self.masks[5])
        np.testing.assert_array_equal(label.head_xy, [10, 15])

        # The label now opens from itself: no recording path needed.
        reopened = self.call("POST", "/api/labeling/open", {"setup": "mine:rig", "entry": self.entry(5, path=False)})
        self.assertEqual((reopened["mask_source"], reopened["expected_revision"]), ("label", 1))
        self.assertEqual(reopened["body"], {"trace_xy": None, "head_xy": [10.0, 15.0], "mask_only": False, "trace_extend": False, "trace_length_px": None})
        context = self.call("POST", "/api/labeling/context", {"setup": "mine:rig", "entry": self.entry(5, path=False)})
        self.assertEqual((len(context["frames"]), context["valid"][16]), (33, True))
        stale = self.client.post("/api/labeling/save", json={"setup": "mine:rig", "entry": self.entry(5), "mask": mask_url(self.masks[5]),
                                                              "expected_revision": 0})
        self.assertIn("changed since it was opened", stale.json()["error"])
        trace = [[10, 11], [30, 11], [50, 11]]
        again = self.call("POST", "/api/labeling/save", {"setup": "mine:rig", "entry": self.entry(5), "mask": mask_url(self.masks[5]),
                                                          "trace_xy": trace, "trace_extend": True, "trace_length_px": 640.0,
                                                          "mask_only": True, "expected_revision": 1})
        self.assertEqual((again["label"]["revision"], again["label"]["status"]), (2, "mask_only"))
        # Extend off camera is saved with the trace and comes back when the frame reopens.
        reopened = self.call("POST", "/api/labeling/open", {"setup": "mine:rig", "entry": self.entry(5)})
        self.assertEqual((reopened["body"]["trace_xy"], reopened["body"]["trace_extend"], reopened["body"]["trace_length_px"]), (trace, True, 640.0))
        self.call("POST", "/api/labeling/open", {"setup": "mine:rig", "entry": self.entry(6, path=False)}, 404)

    def test_a_request_chooses_its_mask_and_body_models(self):
        weights = self.root / "stand-in.ckpt"
        weights.write_bytes(b"stand-in")
        inputs = library.make_inputs([], fps=20.0, pixel_size_um=1.0)
        cards = {"seg-a": ("segmenter", ["mask"]), "seg-b": ("segmenter", ["mask"]),
                 "fields": ("body_net", ["mask", "ap", "head", "tail", "overlap"])}
        for name, (kind, outputs) in cards.items():
            library.write_model(self.libraries.personal, name, {"name": name, "kind": kind, "setup": "mine:rig", "outputs": outputs,
                                                                "inputs": inputs}, weights)
        library.set_default(self.libraries, "mine:rig", "mask", "mine:seg-a", reason="test")
        used = []

        def model(setup, ref):
            used.append(ref)
            return mock.Mock(predict=lambda context, valid: Outputs(mask=np.full(context.shape[1:], 0.9, np.float32)))

        self.state.labeling.model = model
        request = {"setup": "mine:rig", "entry": self.entry(5)}
        opened = self.call("POST", "/api/labeling/open", request)
        self.assertEqual((opened["models"], opened["defaults"], used), ({"mask": "mine:seg-a", "body": None},) * 2 + (["mine:seg-a"],))
        chosen = {**request, "models": {"mask": "mine:seg-b", "body": "mine:fields"}}
        opened = self.call("POST", "/api/labeling/open", chosen)
        self.assertEqual((opened["models"], opened["defaults"]["mask"], opened["mask_source"]),
                         ({"mask": "mine:seg-b", "body": "mine:fields"}, "mine:seg-a", "network"))
        self.assertEqual(self.call("POST", "/api/labeling/network", chosen)["model"], "mine:seg-b")
        self.assertEqual(used[-2:], ["mine:seg-b", "mine:seg-b"])
        # A body model must be a body-field net; an unknown model is refused.
        refused = self.client.post("/api/labeling/proposal", json={**request, "models": {"body": "mine:seg-a"}, "mask": mask_url(self.masks[5])})
        self.assertIn("not a body-field model", refused.json()["error"])
        self.call("POST", "/api/labeling/network", {**request, "models": {"mask": "mine:gone"}}, 400)

    def test_a_labeling_manifest_becomes_a_queue(self):
        from worm_pose_gen.app.queues import manifest_queue

        manifest = self.root / "round" / "manifest.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({
            "name": "round 9", "recordings": {"a": {"path": str(self.path), "split": "val"}},
            "frames": [{"recording": "a", "frame_index": 7, "reasons": ["window:holes"]}, {"recording": "a", "frame_index": 3}],
        }))
        queue = manifest_queue(self.state, manifest)
        self.assertEqual((queue["kind"], queue["name"], queue["setup"], queue["origin"]), ("manifest", "round 9", "mine:rig", "fix"))
        listed = self.call("GET", f"/api/queues/{queue['id']}")
        self.assertEqual([(e["recording"], e["frame"]) for e in listed["entries"]], [("2024-05-05-01", 7), ("2024-05-05-01", 3)])
        outside = self.root.parent / f"{self.root.name}-outside.h5"
        write_recording(outside, frames=4)
        self.addCleanup(outside.unlink)
        manifest.write_text(json.dumps({"recordings": {"a": {"path": str(self.path)}, "b": {"path": str(outside)}}, "frames": []}))
        with self.assertRaisesRegex((ValueError, LookupError), "setup"):
            manifest_queue(self.state, manifest)

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
        # Image types need the setup's mask model to sort frames by.
        self.assertIn("unknown frame types", self.call("POST", "/api/queues", {"kind": "spread", "setup": "mine:rig", "recordings": [str(self.path)], "frames": 2, "types": ["blurry"]}, 400)["error"])
        self.assertIn("no mask model", self.call("POST", "/api/queues", {"kind": "spread", "setup": "mine:rig", "recordings": [str(self.path)], "frames": 2, "types": ["edge"]}, 400)["error"])
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

    @slow
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
            library.Collection(libraries, "mine:rig").save(recording="rec", frame=3, image=image, image_raw=image, mask=mask.astype(np.uint8),
                                                           context=context, context_valid=np.ones(5, bool), origin="spread")
            app = create_app(AppConfig(workspaces_root=root / "workspaces",
                                       dataset_root=root / "cache",
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

    @slow
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
            record = library.Collection(self.libraries, "mine:rig").get(entry["recording"], entry["frame"])
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
