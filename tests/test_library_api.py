"""HTTP contracts of /api/library: listing both libraries, a setup's labels, saving from a recording or a lab label, dataset splits, benchmarks, defaults."""

from pathlib import Path
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient
import h5py
import numpy as np

from worm_pose_gen import library
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.app.images import data_url, mask_to_png_values
from worm_pose_gen.library.targets import write_targets

sys.path.insert(0, str(Path(__file__).parent))
from test_library import SHAPE, label_inputs, make_libraries  # noqa: E402


class LibraryApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_libraries(self.root)
        self.config = AppConfig(
            workspaces_root=self.root / "workspaces",
            dataset_root=self.root / "flat-field-cache",
            device="cpu", gpus=(), lab_library=self.libraries.lab, library=self.libraries.personal,
        )
        app = create_app(self.config)
        self.state = app.state.app_state
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.state.close)
        self.addCleanup(self.client.close)
        weights = self.root / "w.ckpt"
        weights.write_bytes(b"w")
        library.write_model(self.libraries.lab, "seg", {"name": "seg", "kind": "segmenter", "setup": "lab:nir", "outputs": ["mask"],
                                                        "inputs": library.make_inputs([], fps=20.0, pixel_size_um=2.5)}, weights)

    def get(self, url, **params):
        response = self.client.get(url, params=params)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def post(self, url, payload, status=200):
        response = self.client.post(url, json=payload)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def mask_url(self):
        mask = np.zeros(SHAPE, np.uint8)
        mask[6:12, 3:29] = 1
        return data_url(mask_to_png_values(mask))

    def test_lists_both_libraries(self):
        roots = self.get("/api/library")
        self.assertEqual((roots["lab"], roots["personal"], roots["lab_available"]), (str(self.libraries.lab), str(self.libraries.personal), True))
        self.assertEqual([s["ref"] for s in self.get("/api/library/setups")["setups"]], ["lab:nir"])
        self.assertEqual(self.get("/api/library/setups/lab:nir")["fps"], 20.0)
        self.assertEqual(self.client.get("/api/library/setups/lab:nope").status_code, 404)
        self.assertEqual(self.client.get("/api/library/setups/nope").status_code, 400)
        datasets = self.get("/api/library/datasets", setup="lab:nir")["datasets"]
        self.assertEqual([(d["ref"], d["labels"], d["writable"]) for d in datasets], [("lab:base", 2, False)])
        models = self.get("/api/library/models", setup="lab:nir")["models"]
        self.assertEqual([(m["ref"], m["outputs"], m["evaluations"]) for m in models], [("lab:seg", ["mask"], {})])
        self.assertEqual(self.get("/api/library/models/lab:seg")["training_files"], [])

    def test_label_listing_filters_and_sorts_by_fit_iou(self):
        records = library.Collection(self.libraries, "lab:nir").labels()
        write_targets(self.libraries, records[1].sha256, None, {"has_body": False, "fit_iou": 0.7, "self_contact": True}, {})
        labels = "/api/library/setups/lab:nir/labels"
        rows = self.get(labels, sort="fit_iou")["labels"]
        self.assertEqual([(r["frame"], r["fit_iou"], r["targets"], r["split"]) for r in rows], [(20, 0.7, "built", None), (10, None, "missing", None)])
        self.assertEqual([r["frame"] for r in self.get(labels, status="complete")["labels"]], [20])
        self.assertEqual([r["frame"] for r in self.get(labels, contact="yes")["labels"]], [20])
        self.assertEqual(self.get(labels, recording="other")["total"], 0)
        self.assertEqual({r["split"] for r in self.get(labels, dataset="lab:base")["labels"]}, {"train"})
        self.assertEqual(self.get(labels, dataset="lab:base", split="none")["total"], 0)
        self.assertEqual(self.client.get(labels, params={"status": "bad"}).status_code, 400)
        self.assertEqual(self.client.get(labels, params={"split": "train"}).status_code, 400)  # a split needs a dataset
        self.assertEqual(self.get("/api/library/setups/lab:nir/collection"), {"setup": "lab:nir", "labels": 2, "recordings": [{"recording": "rec-a", "labels": 2}]})
        detail = self.get(f"{labels}/rec-a/20")
        self.assertEqual((detail["orientation"], detail["head_xy"], detail["max_lag"]), ("manual", [4.0, 11.0], 2))
        self.assertEqual(detail["targets"], {"meta": {"has_body": False, "fit_iou": 0.7, "self_contact": True}})
        self.assertEqual([(r["scope"], r["revision"]) for r in detail["revisions"]], [("lab", 1)])
        self.assertTrue(detail["image"].startswith("data:image/png;base64,"))
        self.assertEqual(self.get(f"{labels}/rec-a/20", revision=1, scope="lab")["label"]["sha256"], records[1].sha256)
        self.assertEqual(self.client.get(f"{labels}/rec-a/20", params={"revision": 1}).status_code, 400)
        context = self.get(f"{labels}/rec-a/20/context")
        self.assertEqual((len(context["frames"]), context["valid"]), (5, [True] * 5))
        self.assertEqual(self.client.get(f"{labels}/rec-a/99").status_code, 404)

    def test_save_edit_of_a_lab_label_into_the_collection(self):
        saved = self.post("/api/library/setups/lab:nir/labels", {
            "recording": "rec-a", "frame": 10, "mask": self.mask_url(), "origin": "fix", "expected_revision": 0,
            "orientation": "manual", "head_xy": [3, 9],
        })["label"]
        self.assertEqual((saved["scope"], saved["setup"], saved["revision"], saved["status"], saved["origin"]), ("mine", "lab:nir", 1, "complete", "fix"))
        stale = self.client.post("/api/library/setups/lab:nir/labels", json={
            "recording": "rec-a", "frame": 10, "mask": self.mask_url(), "origin": "fix", "expected_revision": 0})
        self.assertEqual(stale.status_code, 400, stale.text)
        collection = library.Collection(self.libraries, "lab:nir")
        lab_label = collection.get("rec-a", 10, revision=1, scope="lab").load()
        mine = collection.get("rec-a", 10).load()
        np.testing.assert_array_equal(mine.context, lab_label.context)  # taken from the lab label
        self.assertEqual(mine.mask[6, 3], 1)
        self.assertEqual([r["scope"] for r in self.get("/api/library/setups/lab:nir/labels")["labels"]], ["mine", "lab"])

    def test_datasets_start_empty_and_take_splits_per_recording(self):
        made = self.post("/api/library/datasets", {"id": "copper", "setup": "lab:nir", "name": "Copper"})
        self.assertEqual((made["ref"], made["writable"], made["labels"], made["not_included"]), ("mine:copper", True, 0, {"recordings": 1, "labels": 2}))
        self.assertEqual([(r["recording"], r["split"]) for r in made["recordings"]], [("rec-a", None)])
        changed = self.post("/api/library/datasets/mine:copper/splits", {"splits": {"rec-a": "test"}})
        self.assertEqual((changed["by_split"], changed["recordings_by_split"]["test"]), ({"train": 0, "val": 0, "test": 2}, 1))
        self.assertEqual(self.post("/api/library/datasets/mine:copper/splits", {"splits": {"rec-a": None}})["by_split"]["test"], 0)
        self.post("/api/library/datasets/mine:copper/splits", {"splits": {"rec-a": "everything"}}, status=400)
        self.post("/api/library/datasets/mine:copper/splits", {"splits": ["rec-a"]}, status=400)
        self.post("/api/library/datasets/lab:base/splits", {"splits": {"rec-a": "test"}}, status=403)
        copied = self.post("/api/library/datasets", {"id": "copy", "setup": "lab:nir", "splits": {"rec-a": "val"}})
        self.assertEqual(copied["by_split"]["val"], 2)
        self.assertEqual(self.post("/api/library/datasets", {"id": "copper", "setup": "lab:nir"}, status=400)["error"], "dataset mine:copper already exists")
        self.assertEqual(self.get("/api/library/datasets/mine:copper")["name"], "Copper")

    def test_save_from_a_recording_reads_frame_context_and_nose(self):
        recording = self.root / "recordings" / "2024-03-03-03.h5"
        recording.parent.mkdir()
        frames = np.random.default_rng(0).integers(40, 200, (40, *SHAPE), dtype=np.uint8)
        features = np.zeros((40, 3, 3), np.float32)
        features[:, 0, 0], features[:, 1, 0], features[:, 2, 0] = 5.0, 9.0, 0.99  # one-based x, y and confidence
        with h5py.File(recording, "w") as handle:
            handle.create_dataset("/img_nir", data=frames)
            handle.create_dataset("/pos_feature", data=features)
        saved = self.post("/api/library/setups/lab:nir/labels", {"path": str(recording), "frame": 2, "mask": self.mask_url(), "origin": "spread"})["label"]
        self.assertEqual((saved["recording"], saved["frame"], saved["split"]), ("2024-03-03-03", 2, None))
        label = library.Collection(self.libraries, "lab:nir").get("2024-03-03-03", 2).load()
        self.assertEqual(label.context.shape, (33, *SHAPE))
        self.assertEqual(label.context_valid.tolist(), [False] * 14 + [True] * 19)
        np.testing.assert_array_equal(label.image_raw, frames[2])
        np.testing.assert_array_equal(label.nose_xy[16], [4.0, 8.0])
        self.assertEqual(label.nose_valid.tolist(), label.context_valid.tolist())
        rows = self.get("/api/library/setups/lab:nir/recordings")["recordings"]
        self.assertEqual([(r["id"], r["frames"]) for r in rows], [("2024-03-03-03", 40)])
        self.assertEqual(self.get("/api/library/recording-setup", path=str(recording))["setup"], "lab:nir")
        elsewhere = self.root / "other.h5"
        elsewhere.write_bytes(recording.read_bytes())
        refused = self.client.post("/api/library/setups/lab:nir/labels", json={"path": str(elsewhere), "frame": 2, "mask": self.mask_url(), "origin": "spread"})
        self.assertIn("belongs to no setup", refused.json()["error"])

    def test_defaults_and_benchmarks(self):
        changed = self.post("/api/library/setups/lab:nir/defaults", {"role": "mask", "model": "lab:seg", "reason": "first model"})
        self.assertEqual(changed["defaults"], {"mask": "lab:seg"})
        self.assertEqual([e["reason"] for e in changed["defaults_log"]], ["first model"])
        self.assertEqual(self.post("/api/library/setups/lab:nir/defaults", {"role": "body", "model": "lab:seg", "reason": "x"}, status=400)["error"],
                         "lab:seg cannot be the body default: it has no ap, head, tail output")
        self.post("/api/library/datasets", {"id": "held", "setup": "lab:nir", "splits": {"rec-t": "test"}})
        library.Collection(self.libraries, "lab:nir").save(recording="rec-t", frame=4, origin="spread", **label_inputs(4))
        frozen = self.post("/api/library/benchmarks", {"dataset": "mine:held"})
        self.assertEqual((frozen["ref"], frozen["labels"], frozen["recordings"]), ("mine:held-b1", 1, ["rec-t"]))
        self.assertEqual(self.get("/api/library/benchmarks/mine:held-b1")["entries"][0]["frame"], 4)
        self.assertEqual([b["ref"] for b in self.get("/api/library/benchmarks", setup="lab:nir")["benchmarks"]], ["mine:held-b1"])


if __name__ == "__main__":
    unittest.main()
