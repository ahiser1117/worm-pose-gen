"""scripts/migrate_to_library.py on two small synthetic stores: union, newest wins, recording ids, human fields, splits, cards."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from worm_pose_gen import body_fields, library
from worm_pose_gen.library import Libraries
from worm_pose_gen.segmentation_dataset import SegmentationStore

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_to_library.py"
spec = importlib.util.spec_from_file_location("migrate_to_library", SCRIPT)
migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migrate)

SHAPE = (24, 32)
LAG = 16


def frame(seed):
    return np.random.default_rng(seed).integers(0, 255, SHAPE, dtype=np.uint8)


def worm_mask():
    mask = np.zeros(SHAPE, np.uint8)
    mask[10:14, 4:28] = 1
    return mask


def add_sample(store, path, frame_index, split, *, review=None, orientation="nose", trace=False, body=True, mask=None):
    image = frame(frame_index)
    mask = worm_mask() if mask is None else mask
    record = store.save(Path(path).stem if "rec_" not in str(path) else "rec_abc", frame_index, image, mask,
                        source_path=path, label_source="manual", split=split)
    if not body:
        return record
    centerline = np.stack((np.linspace(4, 27, 10), np.full(10, 12.0)), 1)
    meta = {"sample_id": record.sample_id, "mask_revision": record.revision, "max_lag": LAG, "has_body": bool(mask.any()),
            "orientation": orientation, "nose_offset": 0, "fit_iou": 0.95, "overlap_px": 0, "fit_preset": "reference"}
    arrays = {"context": np.stack([image] * (2 * LAG + 1)), "context_valid": np.ones(2 * LAG + 1, bool),
              "centerline_xy": centerline, "width_profile": np.full(10, 4.0), "ap": np.zeros(SHAPE, np.float16),
              "overlap": np.zeros(SHAPE, bool), "head_xy": centerline[0], "tail_xy": centerline[-1],
              "diameter_px": np.float64(4.0), "nose_xy": np.array([4.0, 12.0])}
    if review:
        meta["review"] = review
    if trace:
        meta["fit_method"] = "traced"
        arrays["trace_xy"] = centerline[::3]
    body_fields.fields_dir(store.root).mkdir(exist_ok=True)
    body_fields.save(body_fields.field_path(store.root, record.sample_id), meta, arrays)
    return record


def write_run(directory, name, splits, lags=()):
    directory.mkdir(parents=True)
    (directory / "best.ckpt").write_bytes(name.encode())
    (directory / "metrics.csv").write_text("epoch,val_loss\n0,1.0\n")
    run = {"name": name, "args": {"epochs": 3, "dataset_root": "x", "num_workers": 2}, "lags": list(lags),
           "splits": {"train": [{"sample_id": s, "revision": 1} for s in splits["train"]],
                      "val": [{"sample_id": s, "revision": 1} for s in splits["val"]],
                      "test": [{"sample_id": s, "revision": 1} for s in splits["test"]]},
           "best_epoch": 0, "best_val_loss": 0.5, "test": {"test_iou": 0.9}, "finished_at": "2026-10-06T00:00:00+00:00"}
    (directory / "run.json").write_text(json.dumps(run))


class MigrationTests(unittest.TestCase):
    def test_union_splits_human_fields_and_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, corpus = SegmentationStore(root / "segmentation_v1"), SegmentationStore(root / "corpus")
            # One recording under two mount points and a corpus identity: one recording id.
            a1 = add_sample(old, "/store1/raw/2023-01-01-01.h5", 1, "train", review="accepted", orientation="manual")
            add_sample(old, "/mirror/raw/2023-01-01-01.h5", 2, "test", review="rejected")
            add_sample(old, "/store1/raw/2023-01-01-01.h5", 3, "train", trace=True, review="accepted")
            b = add_sample(old, "/store1/raw/2023-02-02-02.h5", 1, "val")
            c = add_sample(old, "/store1/raw/2023-03-03-03.h5", 5, "test", body=False)
            empty = add_sample(old, "/store1/raw/2023-03-03-03.h5", 6, "test", mask=np.zeros(SHAPE, np.uint8))
            # Saved later, so it wins over a1; without a record of its own it takes a1's context frames, not its review.
            newer = add_sample(corpus, "/mirror/raw/2023-01-01-01.h5", 1, "val", body=False)
            write_run(root / "seg", "r4", {"train": [a1.sample_id], "val": [b.sample_id], "test": [c.sample_id]})
            write_run(root / "body", "pass2", {"train": [a1.sample_id], "val": [b.sample_id], "test": [c.sample_id]}, lags=(1, 4, 16))
            out, seed = root / "lab", root / "personal"
            migrate.main(["--out", str(out), "--store", str(old.root), "--store", str(corpus.root), "--seed-cache", str(seed),
                          "--segmenter-run", str(root / "seg"), "--body-run", str(root / "body"), "--author", "alex"])

            libraries = Libraries(lab=out, personal=seed)
            records = {r.key: r for r in library.labels(libraries, ["lab:nir-labels"])}
            self.assertEqual(sorted(records), ["2023-01-01-01/000001", "2023-01-01-01/000002", "2023-01-01-01/000003",
                                               "2023-02-02-02/000001", "2023-03-03-03/000005", "2023-03-03-03/000006"])
            first = records["2023-01-01-01/000001"].load()
            self.assertEqual(first.meta["migrated_from"]["sample_id"], newer.sample_id)
            self.assertEqual((first.context_valid.sum(), first.record.status), (33, "auto"))
            self.assertEqual(records["2023-03-03-03/000005"].load().context_valid.sum(), 1)  # no record anywhere
            self.assertEqual(records["2023-01-01-01/000002"].status, "mask_only")
            self.assertEqual(records["2023-01-01-01/000003"].has_trace, True)
            self.assertEqual(records["2023-03-03-03/000006"].status, "complete")  # no worm in the frame
            self.assertEqual(records["2023-02-02-02/000001"].status, "auto")
            self.assertTrue(all(r.origin == "migrated" and r.author == "alex" for r in records.values()))
            # Any old training frame makes a recording train; held-out-only recordings keep val or test.
            self.assertEqual({r.recording: r.split for r in records.values()},
                             {"2023-01-01-01": "train", "2023-02-02-02": "val", "2023-03-03-03": "test"})
            self.assertEqual([e["frame"] for e in library.get_benchmark(libraries, "lab:nir-v1").entries], [5, 6])
            setup = library.get_setup(libraries, "lab:nir-flv")
            self.assertEqual(setup.defaults, {"mask": "lab:nir-hand284", "body": "lab:nir-body-lags3"})
            self.assertEqual(len(library.defaults_log(libraries, "lab:nir-flv")), 2)
            body = library.get_card(libraries, "lab:nir-body-lags3")
            self.assertEqual((body.kind, body.inputs["lags_frames"], len(body.outputs)), ("body_net", [1, 4, 16], 5))
            self.assertEqual(body.trained_on[0]["before_library"]["trained_on_now_in"], {"train": 1, "val": 1})
            self.assertEqual(library.weights_path(libraries, "lab:nir-hand284").read_bytes(), b"r4")
            used = json.loads((library.training_dir(libraries, "lab:nir-hand284") / "labels.json").read_text())
            self.assertEqual({row["old_sample_id"]: row["split_now"] for row in used},
                             {a1.sample_id: "train", b.sample_id: "val", c.sample_id: "test"})
            # Seeded targets: one per label with a current body record (frame 1's winner has none).
            self.assertIsNone(library.cached_meta(libraries, records["2023-01-01-01/000001"]))
            self.assertEqual(library.cached_meta(libraries, records["2023-01-01-01/000003"])["fit_iou"], 0.95)
            self.assertFalse(library.cached_meta(libraries, records["2023-03-03-03/000006"])["has_body"])
            with self.assertRaises(SystemExit):  # never into a library that has content
                migrate.main(["--out", str(out), "--store", str(old.root), "--segmenter-run", str(root / "seg"), "--body-run", str(root / "body")])


if __name__ == "__main__":
    unittest.main()
