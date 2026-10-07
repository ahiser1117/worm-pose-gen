"""Training and evaluation on library datasets: label items, plans, one CPU epoch end to end, scoring, the inference helper and the CLI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import lightning as L
import numpy as np
import torch

from worm_pose_gen import library, model_eval, model_training
from worm_pose_gen.body_net import BodyFieldModule
from worm_pose_gen.body_targets import render_body_targets, self_contact
from worm_pose_gen.library import Libraries
from worm_pose_gen.library.inference import Outputs, lags_at, load_model
from worm_pose_gen.library.roots import write_json
from worm_pose_gen.library.targets import write_targets
from worm_pose_gen.segmenter import SegmentationModule

SHAPE = (64, 96)
MAX_LAG = 2
REPO = Path(__file__).resolve().parents[1]
BODY_OUTPUTS = ["mask", "ap", "head", "tail", "overlap"]


def straight_worm(y=32.0, x0=10.0, x1=86.0):
    centerline = np.stack((np.linspace(x0, x1, 50), np.full(50, y)), 1)
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    points = np.stack((xx.ravel(), yy.ravel()), 1).astype(float)
    distance = np.min(np.linalg.norm(points[:, None] - centerline[None], axis=-1), axis=1).reshape(SHAPE)
    return centerline, distance <= 4.0


def label_inputs(shift=0):
    centerline, mask = straight_worm()
    image = np.where(mask, 60, 200).astype(np.uint8)
    context = np.stack([np.roll(image, s + shift, axis=1) for s in range(-MAX_LAG, MAX_LAG + 1)])
    context[MAX_LAG] = image
    return centerline, dict(image=image, image_raw=image, mask=mask.astype(np.uint8), context=context,
                            context_valid=np.ones(2 * MAX_LAG + 1, bool))


def store_targets(libraries, record, centerline, *, fit_iou=0.97):
    label = record.load()
    targets = render_body_targets(label.mask == 1, centerline, np.full(len(centerline), 8.0))
    meta = {"has_body": True, "fit_iou": fit_iou, "label": record.identity, "fit_method": "independent",
            "self_contact": self_contact(centerline, np.full(len(centerline), 8.0))}
    arrays = {"centerline_xy": centerline, "width_profile": np.full(len(centerline), 8.0), "ap": targets.ap.astype(np.float16),
              "overlap": targets.overlap, "head_xy": targets.head_xy, "tail_xy": targets.tail_xy, "diameter_px": np.float64(8.0)}
    write_targets(libraries, record.sha256, meta, arrays)


def save_weights(module, path):
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams), "pytorch-lightning_version": L.__version__,
                "epoch": 0, "global_step": 0}, path)


def make_library(root: Path) -> Libraries:
    """Lab setup ``rig`` (20 fps) with dataset ``base``: recordings in train (2 labels), val and test (1 each),
    built targets, benchmark ``rig-v1``, a body net ``body`` (lags 1, 2) and a segmenter ``seg``; the setup's defaults are those two."""

    lab, personal = root / "lab", root / "mine"
    library.write_setup(lab, "rig", name="Rig", fps=20.0, pixel_size_um=2.0, defaults={})
    libraries = Libraries(lab=lab, personal=personal)
    base = library.create_dataset(libraries, "base", setup="lab:rig", scope="lab")
    write_json(base.root / "splits.json", {"rec-train": "train", "rec-val": "val", "rec-test": "test"})
    for recording, frame in (("rec-train", 10), ("rec-train", 20), ("rec-val", 5), ("rec-test", 7)):
        centerline, inputs = label_inputs(frame % 3)
        record = base.save(recording=recording, frame=frame, origin="spread", **inputs)
        store_targets(libraries, record, centerline)
    library.write_benchmark(lab, "rig-v1", setup="lab:rig", datasets=["lab:base"], records=library.labels(libraries, ["lab:base"], "test"))
    torch.manual_seed(0)
    save_weights(BodyFieldModule(lags=(1, 2), pretrained=False), root / "body.ckpt")
    save_weights(SegmentationModule(pretrained=False), root / "seg.ckpt")
    library.write_model(lab, "body", {"name": "body", "kind": "body_net", "setup": "lab:rig", "outputs": BODY_OUTPUTS,
                                      "inputs": library.make_inputs([1, 2], fps=20.0, pixel_size_um=2.0)}, root / "body.ckpt")
    library.write_model(lab, "seg", {"name": "seg", "kind": "segmenter", "setup": "lab:rig", "outputs": ["mask"],
                                     "inputs": library.make_inputs([], fps=20.0, pixel_size_um=2.0)}, root / "seg.ckpt")
    library.write_setup(lab, "rig", name="Rig", fps=20.0, pixel_size_um=2.0, defaults={"mask": "lab:seg", "body": "lab:body"})
    return libraries


class LibraryTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_library(self.root)
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)


class LabelDatasetTests(LibraryTestCase):
    def records(self, split="train"):
        return library.labels(self.libraries, ["lab:base"], split)

    def test_body_items_and_loss(self):
        dataset = model_training.LabelDataset(self.libraries, self.records(), kind="body_net", lags=(1, 2))
        item = dataset[0]
        self.assertEqual(tuple(item["image"].shape), (3, *SHAPE))
        self.assertGreater(float(item["image"][1].abs().max()), 0.0)  # the shifted context gives motion
        self.assertTrue(bool(item["targets"][3].any()))  # A-P is trained
        batch = model_training.collate([item, dataset[1]])
        self.assertEqual(tuple(batch["image"].shape), (2, 3, 64, 96))
        module = BodyFieldModule(lags=(1, 2), pretrained=False)
        total, parts = module.loss(module(batch["image"]), batch["targets"])
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertLess(float(parts["head_focal"]), 20.0)
        self.assertEqual(batch["key"], ["rec-train/000010", "rec-train/000020"])

    def test_poor_fit_and_mask_only_train_the_mask_only_unless_settled(self):
        record = self.records()[0]
        centerline, _ = straight_worm()
        store_targets(self.libraries, record, centerline, fit_iou=0.5)
        targets = model_training.LabelDataset(self.libraries, [record], kind="body_net")[0]["targets"]
        self.assertFalse(bool(targets[3].any()))
        self.assertTrue(bool(torch.isnan(targets[4]).all()))
        self.assertTrue(bool(targets[0].any()))  # the mask still trains
        mine = library.create_dataset(self.libraries, "mine", setup="lab:rig", extends="lab:base")
        _, inputs = label_inputs(record.frame % 3)
        settled = mine.save(recording=record.recording, frame=record.frame, origin="fix", orientation="manual", head_xy=[10.0, 32.0], **inputs)
        store_targets(self.libraries, settled, centerline, fit_iou=0.5)
        self.assertEqual(settled.status, "complete")
        self.assertTrue(bool(model_training.LabelDataset(self.libraries, [settled], kind="body_net")[0]["targets"][3].any()))
        masked = mine.save(recording=record.recording, frame=record.frame, origin="fix", mask_only=True, **inputs)
        store_targets(self.libraries, masked, centerline)
        self.assertFalse(bool(model_training.LabelDataset(self.libraries, [masked], kind="body_net")[0]["targets"][3].any()))

    def test_segmenter_items_augment_and_pad(self):
        dataset = model_training.LabelDataset(self.libraries, self.records(), kind="segmenter", augment=True, crop_size=48)
        item = dataset[0]
        self.assertEqual(tuple(item["image"].shape), (1, 48, 48))
        self.assertEqual(set(item), {"key", "image", "mask", "valid"})
        batch = model_training.collate([item, model_training.LabelDataset(self.libraries, self.records("val"), kind="segmenter")[0]])
        self.assertEqual(tuple(batch["mask"].shape), (2, 64, 96))
        self.assertEqual(int(batch["valid"][0].sum()), 48 * 48)  # padding is excluded

    def test_missing_targets_are_an_error(self):
        mine = library.create_dataset(self.libraries, "new", setup="lab:rig")
        _, inputs = label_inputs()
        record = mine.save(recording="rec-train", frame=99, origin="spread", **inputs)
        with self.assertRaises(FileNotFoundError):
            model_training.LabelDataset(self.libraries, [record], kind="body_net")[0]


class PlanTests(LibraryTestCase):
    def test_kind_lags_and_name_follow_the_start(self):
        request = model_training.TrainRequest(setup="lab:rig", datasets=["lab:base"], start_from="lab:body")
        plan = model_training.plan(self.libraries, request)
        self.assertEqual((plan.kind, plan.lags, len(plan.train), len(plan.val)), ("body_net", (1, 2), 2, 1))
        self.assertEqual(plan.request.name, "base-2")
        self.assertEqual(plan.params["patience"], 15)
        segmenter = model_training.plan(self.libraries, model_training.TrainRequest(setup="lab:rig", datasets=["lab:base"], start_from="lab:seg"))
        self.assertEqual((segmenter.kind, segmenter.lags, segmenter.params["patience"]), ("segmenter", (), 5))
        self.assertNotIn("ap_weight", segmenter.params)
        scratch = model_training.plan(self.libraries, model_training.TrainRequest(setup="lab:rig", datasets=["lab:base"], context="short"),
                                      taken=["base-2"])
        self.assertEqual((scratch.kind, scratch.lags, scratch.request.name), ("body_net", (1, 4, 16), "base-2-2"))

    def test_refusals(self):
        bad = [
            ({"datasets": []}, "at least one dataset"),
            ({"params": {"max_epochs": 1.5}}, "integer"),
            ({"params": {"learning_rate": -1}}, "must lie"),
            ({"params": {"warmup": 3}}, "unknown training parameters"),
            ({"context": "long"}, "temporal context"),
            ({"name": "bad name"}, "invalid library id"),
        ]
        for change, message in bad:
            request = model_training.TrainRequest(**{"setup": "lab:rig", "datasets": ["lab:base"], **change})
            with self.assertRaisesRegex(ValueError, message):
                model_training.plan(self.libraries, request)
        library.create_setup(self.libraries, "other", name="Other")
        with self.assertRaisesRegex(ValueError, "not of mine:other"):
            model_training.plan(self.libraries, model_training.TrainRequest(setup="mine:other", datasets=["lab:base"]))
        write_json(library.Dataset(self.libraries, "lab:base").root / "splits.json", {"rec-train": "train", "rec-val": "train", "rec-test": "test"})
        with self.assertRaisesRegex(ValueError, "no validation labels"):
            model_training.plan(self.libraries, model_training.TrainRequest(setup="lab:rig", datasets=["lab:base"]))


class InferenceTests(LibraryTestCase):
    def test_predict_matches_sequence_and_lags_convert(self):
        model = load_model(self.libraries, "lab:body", device="cpu")
        self.assertEqual((model.lags, model.max_lag, model.outputs), ((1, 2), 2, tuple(BODY_OUTPUTS)))
        label = library.labels(self.libraries, ["lab:base"], "test")[0].load()
        out = model.predict(label.context, label.context_valid)
        self.assertEqual(out.mask.shape, SHAPE)
        self.assertTrue(np.all((out.ap >= 0) & (out.ap <= 1)))
        sequence = model.predict_sequence(label.context, label.context_valid, [MAX_LAG])[0]
        np.testing.assert_allclose(sequence.head, out.head, atol=1e-6)
        self.assertEqual(model.predict_probability_batch(label.context).shape, (5, *SHAPE))
        self.assertIs(load_model(self.libraries, "lab:body", device="cpu"), model)  # cached
        with self.assertRaisesRegex(ValueError, "needs ±2"):
            model.predict(label.context[1:4])
        segmenter = load_model(self.libraries, "lab:seg", device="cpu")
        only = segmenter.predict(label.image)
        self.assertIsNone(only.ap)
        self.assertEqual(only.mask.shape, SHAPE)
        card = library.get_card(self.libraries, "lab:body")
        self.assertEqual(lags_at(card, 40.0), (2, 4))
        self.assertEqual(lags_at(card, 20.0), (1, 2))


class ScoringTests(unittest.TestCase):
    def test_perfect_prediction_and_worst_share(self):
        centerline, mask = straight_worm()
        targets = render_body_targets(mask, centerline, np.full(50, 8.0))
        head = np.zeros(SHAPE, np.float32)
        head[32, 10] = 1
        outputs = Outputs(mask=mask.astype(np.float32), ap=np.nan_to_num(targets.ap).astype(np.float32), head=head)
        label = type("Label", (), {"mask": mask.astype(np.uint8), "record": type("R", (), {
            "identity": {"dataset": "lab:d", "recording": "r", "frame": 1, "revision": 1, "sha256": "x"}, "split": "test",
            "status": "auto"})()})()
        arrays = {"ap": targets.ap, "head_xy": targets.head_xy, "tail_xy": targets.tail_xy}
        row = model_eval.score_label(outputs, label, ({"fit_iou": 0.97}, arrays), True)
        self.assertEqual((row["iou"], row["head_tail_correct"], row["missed_px"], row["extra_px"]), (1.0, True, 0, 0))
        self.assertAlmostEqual(row["ap_error"], 0.0, places=6)
        rows = [{"iou": v, "head_tail_correct": None, "ap_error": None} for v in np.linspace(0.5, 1.0, 40)]
        summary = model_eval.summarize(rows)
        self.assertEqual((summary["labels"], summary["worst5_count"]), (40, 2))
        self.assertAlmostEqual(summary["iou_worst5"], (0.5 + 0.5 + 0.5 / 39) / 2)
        self.assertIsNone(summary["head_tail_correct"])
        self.assertEqual(model_eval.summarize(rows[:3])["worst5_count"], 1)


class TrainEndToEndTests(LibraryTestCase):
    def test_one_cpu_epoch_fine_tunes_evaluates_and_writes_the_card(self):
        reports = []
        request = model_training.TrainRequest(
            setup="lab:rig", datasets=["lab:base"], start_from="lab:body", name="tuned", notes="first try",
            params={"max_epochs": 1, "batch_size": 2, "crop_size": 64},
        )
        card = model_training.train(self.libraries, request, device="cpu", progress=lambda f, m, r=None: reports.append((f, m, r)))
        self.assertEqual(card.ref, "mine:tuned")
        self.assertEqual((card.kind, card.parent, card.notes, card.outputs), ("body_net", "lab:body", "first try", tuple(BODY_OUTPUTS)))
        self.assertEqual(card.inputs["lags_frames"], [1, 2])
        self.assertEqual(card.inputs["lags_s"], [0.05, 0.1])
        self.assertEqual((card.inputs["fps"], card.inputs["pixel_size_um"]), (20.0, 2.0))
        self.assertEqual(card.trained_on[0]["counts"], {"train": 2, "val": 1, "test": 0})
        self.assertEqual(card.hparams["max_epochs"], 1)
        directory = self.libraries.personal / "models" / "tuned"
        self.assertEqual(sorted(p.name for p in directory.iterdir()), ["evaluations", "model.json", "training", "weights.ckpt"])
        self.assertTrue({"metrics.csv", "labels.json", "run.json"} <= {p.name for p in (directory / "training").iterdir()})
        used = json.loads((directory / "training" / "labels.json").read_text())
        self.assertEqual((len(used["train"]), len(used["val"])), (2, 1))
        curve = model_training.read_curve(directory / "training" / "metrics.csv")
        self.assertEqual(len(curve), 1)
        self.assertIsNotNone(curve[0]["val_loss"])
        self.assertFalse((self.libraries.personal / model_training.RUNS_DIR / "tuned").exists())
        evaluation = library.evaluations(self.libraries, "mine:tuned")["lab:rig-v1"]
        self.assertEqual((evaluation["labels"], evaluation["worst5_count"]), (1, 1))
        self.assertEqual(len(evaluation["rows"]), 1)
        self.assertTrue((directory / "evaluations" / "lab.rig-v1" / evaluation["worst"][0]["file"]).is_file())
        phases = [r[2]["phase"] for r in reports if r[2]]
        self.assertEqual((phases[0], phases[-1]), ("train", "done"))
        self.assertEqual(reports[-1][0], 1.0)
        self.assertEqual(reports[-1][2]["curve"][0]["epoch"], 1)
        with self.assertRaisesRegex(ValueError, "already exists"):
            model_training.plan(self.libraries, model_training.TrainRequest(setup="lab:rig", datasets=["lab:base"], name="tuned"))
        loaded = load_model(self.libraries, "mine:tuned", device="cpu")
        self.assertEqual(loaded.lags, (1, 2))

    def test_segmenter_fine_tune(self):
        request = model_training.TrainRequest(setup="lab:rig", datasets=["lab:base"], start_from="lab:seg",
                                              params={"max_epochs": 1, "batch_size": 2, "crop_size": 64})
        card = model_training.train(self.libraries, request, device="cpu")
        self.assertEqual((card.kind, card.outputs, card.inputs["lags_frames"]), ("segmenter", ("mask",), []))
        evaluation = library.evaluations(self.libraries, card.ref)["lab:rig-v1"]
        self.assertIsNone(evaluation["head_tail_correct"])
        self.assertIsNotNone(evaluation["iou_mean"])


class EvaluateTests(LibraryTestCase):
    def test_lab_model_evaluation_goes_to_the_personal_library(self):
        result = model_eval.evaluate(self.libraries, "lab:body", "lab:rig-v1", device="cpu")
        self.assertEqual(result["labels"], 1)
        self.assertEqual(result["head_tail_labels"], 1)
        self.assertIsNotNone(result["ap_error"])
        path = self.libraries.personal / "evaluations" / "lab.body" / "lab.rig-v1.json"
        self.assertTrue(path.is_file())
        self.assertEqual(len(list((path.parent / "lab.rig-v1").glob("worst1-rec-test-000007.png"))), 1)
        self.assertEqual(model_eval.missing_evaluations(self.libraries, "lab:rig"), [("lab:seg", "lab:rig-v1")])


class CommandLineTests(LibraryTestCase):
    def run_script(self, *argv):
        env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
        result = subprocess.run([sys.executable, *argv], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        return result.stdout

    def library_flags(self):
        return ["--lab-library", str(self.libraries.lab), "--library", str(self.libraries.personal), "--device", "cpu"]

    def test_evaluate_and_train_scripts(self):
        output = self.run_script("scripts/evaluate_model.py", *self.library_flags(), "--model", "lab:seg")
        self.assertIn("lab:seg on lab:rig-v1", output)
        self.assertIn("lab:rig-v1", library.evaluations(self.libraries, "lab:seg"))
        output = self.run_script("scripts/train.py", *self.library_flags(), "--setup", "lab:rig", "--dataset", "lab:base",
                                 "--start-from", "lab:seg", "--max-epochs", "1", "--batch-size", "2", "--crop-size", "64", "--name", "cli")
        self.assertEqual(json.loads(output.strip().splitlines()[-1])["model"], "mine:cli")
        self.assertEqual(library.get_card(self.libraries, "mine:cli").hparams["max_epochs"], 1)


if __name__ == "__main__":
    unittest.main()
