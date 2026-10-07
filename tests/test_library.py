"""The library: references, setups and defaults, per-recording splits, label revisions and inheritance, benchmarks, model cards."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from worm_pose_gen import library
from worm_pose_gen.library import Libraries
from worm_pose_gen.library.datasets import Dataset, assign_split
from worm_pose_gen.library.roots import default_lab_root, default_personal_root, parse_ref, ref_filename, write_json


SHAPE = (24, 32)


def label_inputs(seed=0, *, max_lag=2):
    rng = np.random.default_rng(seed)
    context = rng.integers(0, 255, (2 * max_lag + 1, *SHAPE), dtype=np.uint8)
    mask = np.zeros(SHAPE, np.uint8)
    mask[8:14, 4:28] = 1
    return {
        "image": context[max_lag], "image_raw": context[max_lag] // 2 + 1, "mask": mask, "context": context,
        "context_valid": np.ones(2 * max_lag + 1, bool), "source_path": "/data/rec.h5",
    }


def make_libraries(root: Path) -> Libraries:
    """A lab library with setup ``nir`` and dataset ``base`` (two labels), and an empty personal one."""

    lab, personal = root / "lab", root / "mine"
    library.write_setup(lab, "nir", name="NIR", fps=20.0, pixel_size_um=2.5, recording_roots=[str(root / "recordings")])
    libraries = Libraries(lab=lab, personal=personal)
    base = library.create_dataset(libraries, "base", setup="lab:nir", scope="lab")
    base.save(recording="rec-a", frame=10, origin="spread", **label_inputs(1))
    base.save(recording="rec-a", frame=20, origin="spread", orientation="manual", head_xy=[4.0, 11.0], **label_inputs(2))
    return libraries


class RootTests(unittest.TestCase):
    def test_refs_and_host_defaults(self):
        self.assertEqual(parse_ref("lab:nir-labels"), ("lab", "nir-labels"))
        self.assertEqual(ref_filename("mine:copper_b1"), "mine.copper_b1")
        for bad in ("nir", "other:x", "lab:", "lab:a/b", "mine:a.b", "lab:../x"):
            with self.assertRaises(ValueError):
                parse_ref(bad)
        self.assertEqual(default_lab_root("flv-c3"), Path("/store1/shared/worm-pose-models"))
        self.assertIsNone(default_lab_root("laptop"))
        self.assertEqual(default_personal_root("flv-c2", "kim"), Path("/temp_data4/kim/worm-pose-library"))
        self.assertEqual(default_personal_root("laptop", "kim"), Path("~/worm-pose-library").expanduser())

    def test_json_files_take_numpy_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a" / "meta.json"
            write_json(path, {"cut": np.bool_(True), "iou": np.float32(0.5)})
            self.assertEqual(json.loads(path.read_text()), {"cut": True, "iou": 0.5})

    def test_missing_lab_library_lists_personal_items_only(self):
        with tempfile.TemporaryDirectory() as directory:
            libraries = Libraries(lab=Path(directory) / "absent", personal=Path(directory) / "mine")
            self.assertEqual(libraries.scopes(), ("mine",))
            library.create_setup(libraries, "rig", name="Rig")
            self.assertEqual([s.ref for s in library.list_setups(libraries)], ["mine:rig"])
            self.assertEqual(Libraries(lab=None, personal=libraries.personal).scopes(), ("mine",))


class SetupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_libraries(self.root)
        weights = self.root / "w.ckpt"
        weights.write_bytes(b"weights")
        inputs = library.make_inputs([], fps=20.0, pixel_size_um=2.5)
        library.write_model(self.libraries.lab, "seg", {"name": "seg", "kind": "segmenter", "setup": "lab:nir", "inputs": inputs, "outputs": ["mask"]}, weights)
        library.create_model(self.libraries, "body", {"name": "body", "kind": "body_net", "setup": "lab:nir",
                                                      "inputs": library.make_inputs([1, 4], fps=20.0, pixel_size_um=2.5),
                                                      "outputs": ["mask", "ap", "head", "tail", "overlap"]}, weights)

    def test_personal_override_of_a_lab_default_is_logged(self):
        setup = library.get_setup(self.libraries, "lab:nir")
        self.assertEqual((setup.fps, setup.pixel_size_um, setup.video["dataset_path"]), (20.0, 2.5, "/img_nir"))
        with self.assertRaises(ValueError):  # a mask-only model cannot drive the body
            library.set_default(self.libraries, "lab:nir", "body", "lab:seg", reason="try")
        with self.assertRaises(ValueError):
            library.set_default(self.libraries, "lab:nir", "body", "mine:body", reason=" ")
        updated = library.set_default(self.libraries, "lab:nir", "body", "mine:body", reason="better heads", who="kim")
        self.assertEqual(updated.defaults, {"body": "mine:body"})
        self.assertEqual(updated.overridden, ("body",))
        self.assertFalse((self.libraries.lab / "setups" / "nir.override.json").exists())
        updated = library.set_default(self.libraries, "lab:nir", "mask", "lab:seg", reason="lab mask")
        self.assertEqual(updated.defaults, {"body": "mine:body", "mask": "lab:seg"})
        log = library.defaults_log(self.libraries, "lab:nir")
        self.assertEqual([(e["role"], e["model"], e["previous"]) for e in log], [("body", "mine:body", None), ("mask", "lab:seg", None)])
        self.assertEqual(log[0]["who"], "kim")
        self.assertEqual(log[0]["reason"], "better heads")

    def test_recording_belongs_to_its_registration_else_its_root(self):
        recordings = self.root / "recordings" / "day"
        recordings.mkdir(parents=True)
        inside = recordings / "2024-01-01-01.h5"
        inside.write_bytes(b"")
        outside = self.root / "elsewhere" / "2024-02-02-02.h5"
        outside.parent.mkdir()
        outside.write_bytes(b"")
        self.assertEqual(library.setup_for_recording(self.libraries, inside), "lab:nir")
        self.assertIsNone(library.setup_for_recording(self.libraries, outside))
        library.create_setup(self.libraries, "rig2", name="Rig 2")
        entry = library.register_recording(self.libraries, outside, "mine:rig2")
        self.assertEqual(entry["id"], "2024-02-02-02")
        self.assertEqual(library.setup_for_recording(self.libraries, outside), "mine:rig2")
        self.assertEqual(library.recording_sources(self.libraries, "mine:rig2"), [outside.resolve()])
        with self.assertRaises(LookupError):
            library.register_recording(self.libraries, outside, "mine:nope")


class DatasetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_libraries(self.root)

    def test_splits_are_per_recording_and_balanced_by_label_count(self):
        counts = {"train": 0, "val": 0, "test": 0}
        for _ in range(10):
            counts[assign_split(counts)] += 1
        self.assertEqual(counts, {"train": 8, "val": 1, "test": 1})
        dataset = library.create_dataset(self.libraries, "solo", setup="lab:nir")
        splits = []
        for k, recording in enumerate(["r1", "r1", "r1", "r2", "r3", "r3", "r4", "r5"]):
            record = dataset.save(recording=recording, frame=k, origin="spread", **label_inputs(k))
            splits.append((recording, record.split))
        by_recording = {}
        for recording, split in splits:
            by_recording.setdefault(recording, set()).add(split)
        self.assertTrue(all(len(s) == 1 for s in by_recording.values()))  # a recording never straddles splits
        self.assertEqual(dataset.own_splits(), {r: s.pop() for r, s in by_recording.items()})
        self.assertEqual(dataset.own_splits()["r1"], "train")

    def test_revisions_are_immutable_and_concurrent_edits_refused(self):
        dataset = library.create_dataset(self.libraries, "edits", setup="lab:nir")
        first = dataset.save(recording="rec-b", frame=3, origin="spread", expected_revision=0, **label_inputs(3))
        with self.assertRaisesRegex(ValueError, "changed since it was opened"):
            dataset.save(recording="rec-b", frame=3, origin="fix", expected_revision=0, **label_inputs(3))
        inputs = label_inputs(3)
        inputs["mask"] = np.zeros(SHAPE, np.uint8)
        second = dataset.save(recording="rec-b", frame=3, origin="fix", expected_revision=1, mask_only=True, **{k: v for k, v in inputs.items() if k != "mask"}, mask=inputs["mask"])
        self.assertEqual((first.revision, second.revision), (1, 2))
        self.assertNotEqual(first.sha256, second.sha256)
        self.assertEqual([r.revision for r in dataset.revisions("rec-b", 3)], [1, 2])
        self.assertTrue(first.load().mask.any())  # the old revision still reads as it was
        self.assertEqual(second.status, "mask_only")
        self.assertEqual(dataset.get("rec-b", 3).revision, 2)
        self.assertEqual(dataset.get("rec-b", 3, revision=1).sha256, first.sha256)

    def test_label_validation(self):
        dataset = library.create_dataset(self.libraries, "checks", setup="lab:nir")
        inputs = label_inputs(4)
        bad = [
            dict(inputs, mask=np.full(SHAPE, 2, np.uint8)),
            dict(inputs, image=inputs["image"] + 1),  # not the centre context frame
            dict(inputs, context_valid=np.zeros(5, bool)),
        ]
        for case in bad:
            with self.assertRaises(ValueError):
                dataset.save(recording="rec-c", frame=1, origin="spread", **case)
        with self.assertRaises(ValueError):
            dataset.save(recording="rec-c", frame=1, origin="painted", **inputs)
        with self.assertRaises(ValueError):  # manual without a head or trace
            dataset.save(recording="rec-c", frame=1, origin="spread", orientation="manual", **inputs)
        traced = dataset.save(recording="rec-c", frame=1, origin="spread", trace_xy=[[4, 11], [16, 11], [27, 11]], **inputs)
        self.assertEqual((traced.orientation, traced.has_trace, traced.status), ("manual", True, "complete"))
        loaded = traced.load()
        self.assertIsNone(loaded.head_xy)
        np.testing.assert_array_equal(loaded.trace_xy[0], [4, 11])
        self.assertEqual(loaded.max_lag, 2)
        self.assertFalse(loaded.nose_valid.any())
        with self.assertRaises(PermissionError):
            Dataset(self.libraries, "lab:base").save(recording="rec-a", frame=10, origin="fix", **label_inputs(1))

    def test_personal_dataset_extends_and_overrides_the_lab_one(self):
        lab = Dataset(self.libraries, "lab:base")
        lab_split = lab.splits()["rec-a"]
        mine = library.create_dataset(self.libraries, "copper", setup="lab:nir", extends="lab:base")
        edited = mine.save(recording="rec-a", frame=10, origin="fix", expected_revision=0, **label_inputs(9))
        self.assertEqual(edited.split, lab_split)  # an inherited recording keeps the lab's split
        self.assertEqual(mine.own_splits(), {"rec-a": lab_split})
        new = mine.save(recording="rec-new", frame=5, origin="spread", **label_inputs(10))
        records = library.labels(self.libraries, ["mine:copper"])
        self.assertEqual([(r.dataset, r.key) for r in records],
                         [("mine:copper", "rec-a/000010"), ("lab:base", "rec-a/000020"), ("mine:copper", "rec-new/000005")])
        self.assertEqual(library.labels(self.libraries, ["lab:base", "mine:copper"]), records)
        self.assertEqual(mine.get("rec-a", 20).dataset, "lab:base")
        self.assertEqual(library.labels(self.libraries, ["lab:base"])[0].revision, 1)  # the lab label is untouched
        self.assertEqual(len(library.labels(self.libraries, ["mine:copper"], new.split, recording="rec-new")), 1)
        summary = mine.summary()
        self.assertEqual((summary["labels"], summary["own_labels"], summary["extends"]), (3, 2, "lab:base"))
        self.assertEqual({r["recording"]: r["labels"] for r in summary["recordings"]}, {"rec-a": 2, "rec-new": 1})
        self.assertIn("no test recording", summary["readiness"])
        statuses = {r.key: r.status for r in records}
        self.assertEqual(statuses, {"rec-a/000010": "auto", "rec-a/000020": "complete", "rec-new/000005": "auto"})
        with self.assertRaises(ValueError):  # a personal dataset for another setup cannot extend it
            library.create_setup(self.libraries, "other", name="Other")
            library.create_dataset(self.libraries, "x", setup="mine:other", extends="lab:base")

    def test_conflicting_splits_cannot_be_combined(self):
        a = library.create_dataset(self.libraries, "a", setup="lab:nir")
        b = library.create_dataset(self.libraries, "b", setup="lab:nir")
        a.save(recording="shared", frame=1, origin="spread", **label_inputs(1))
        (b.root / "splits.json").write_text(json.dumps({"shared": "test"}))
        b.save(recording="shared", frame=2, origin="spread", **label_inputs(2))
        with self.assertRaisesRegex(ValueError, "shared is train"):
            library.labels(self.libraries, ["mine:a", "mine:b"])

    def test_trained_on_names_exact_revisions(self):
        records = library.labels(self.libraries, ["lab:base"])
        summary = library.trained_on(records)
        self.assertEqual(summary[0]["dataset"], "lab:base")
        self.assertEqual(sum(summary[0]["counts"].values()), 2)
        self.assertEqual(library.fingerprint(records), library.fingerprint(records[::-1]))
        self.assertNotEqual(library.fingerprint(records), library.fingerprint(records[:1]))


class BenchmarkAndModelTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_libraries(self.root)

    def test_freeze_takes_spread_and_migrated_test_labels_once(self):
        mine = library.create_dataset(self.libraries, "copper", setup="lab:nir", extends="lab:base")
        (mine.root / "splits.json").write_text(json.dumps({"held": "test"}))
        spread = mine.save(recording="held", frame=1, origin="spread", **label_inputs(1))
        mine.save(recording="held", frame=2, origin="fix", **label_inputs(2))
        frozen = library.freeze_benchmark(self.libraries, "mine:copper")
        self.assertEqual(frozen.ref, "mine:copper-b1")
        self.assertEqual([e["frame"] for e in frozen.entries], [1])
        self.assertEqual(library.benchmark_labels(self.libraries, frozen.ref), [spread])
        # Frozen means frozen: a later label does not join it, and the next freeze is b2.
        mine.save(recording="held", frame=3, origin="spread", **label_inputs(3))
        self.assertEqual(len(library.get_benchmark(self.libraries, frozen.ref).entries), 1)
        self.assertEqual(library.freeze_benchmark(self.libraries, "mine:copper").ref, "mine:copper-b2")
        with self.assertRaises(FileExistsError):
            library.freeze_benchmark(self.libraries, "mine:copper", "copper-b1")
        self.assertEqual([b.ref for b in library.list_benchmarks(self.libraries, "lab:nir")], ["mine:copper-b1", "mine:copper-b2"])
        with self.assertRaises(ValueError):  # no test labels
            library.freeze_benchmark(self.libraries, "lab:base")

    def test_model_cards_weights_and_where_evaluations_go(self):
        weights = self.root / "best.ckpt"
        weights.write_bytes(b"lab weights")
        card = {"name": "nir-hand2", "kind": "segmenter", "setup": "lab:nir", "outputs": ["mask"],
                "inputs": library.make_inputs([], fps=20.0, pixel_size_um=2.5)}
        library.write_model(self.libraries.lab, "nir-hand2", card, weights, training_records={"labels.json": {"n": 2}})
        with self.assertRaises(FileExistsError):
            library.write_model(self.libraries.lab, "nir-hand2", card, weights)
        with self.assertRaises(ValueError):
            library.create_model(self.libraries, "bad", {**card, "outputs": ["mask", "colour"]}, weights)
        body = library.create_model(self.libraries, "body", {**card, "name": "body", "kind": "body_net", "parent": "lab:nir-hand2",
                                                             "inputs": library.make_inputs([1, 4, 16], fps=20.0, pixel_size_um=None),
                                                             "outputs": ["mask", "ap", "head", "tail", "overlap"]}, weights)
        self.assertEqual(body.inputs["lags_s"], [0.05, 0.2, 0.8])
        self.assertEqual(library.weights_path(self.libraries, "lab:nir-hand2").read_bytes(), b"lab weights")
        self.assertEqual(json.loads((library.training_dir(self.libraries, "lab:nir-hand2") / "labels.json").read_text()), {"n": 2})
        self.assertEqual([c.ref for c in library.list_models(self.libraries, "lab:nir")], ["lab:nir-hand2", "mine:body"])
        lab_eval = library.evaluation_path(self.libraries, "lab:nir-hand2", "mine:copper-b1")
        self.assertEqual(lab_eval, self.libraries.personal / "evaluations" / "lab.nir-hand2" / "mine.copper-b1.json")
        self.assertEqual(library.evaluation_path(self.libraries, "mine:body", "lab:v1"),
                         self.libraries.personal / "models" / "body" / "evaluations" / "lab.v1.json")
        # A lab evaluation published with the model, then a personal rescoring on the same benchmark wins.
        published = self.libraries.lab / "models" / "nir-hand2" / "evaluations" / "lab.v1.json"
        published.parent.mkdir()
        published.write_text(json.dumps({"benchmark": "lab:v1", "iou_mean": 0.9}))
        self.assertEqual(library.evaluations(self.libraries, "lab:nir-hand2")["lab:v1"]["iou_mean"], 0.9)
        library.save_evaluation(self.libraries, "lab:nir-hand2", "lab:v1", {"iou_mean": 0.95})
        library.save_evaluation(self.libraries, "lab:nir-hand2", "mine:copper-b1", {"iou_mean": 0.8})
        scores = library.evaluations(self.libraries, "lab:nir-hand2")
        self.assertEqual({k: v["iou_mean"] for k, v in scores.items()}, {"lab:v1": 0.95, "mine:copper-b1": 0.8})
        self.assertFalse((self.libraries.lab / "models" / "nir-hand2" / "evaluations" / "mine.copper-b1.json").exists())


if __name__ == "__main__":
    unittest.main()
