"""scripts/publish.py: copying personal setups, labels, datasets and models into a lab library, in temporary directories."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from worm_pose_gen import library
from worm_pose_gen.library import Libraries

sys.path.insert(0, str(Path(__file__).parent))
from test_library import label_inputs  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish.py"
spec = importlib.util.spec_from_file_location("publish", SCRIPT)
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class PublishTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "lab").mkdir()
        self.libraries = Libraries(lab=root / "lab", personal=root / "mine")
        library.create_setup(self.libraries, "rig2", name="Rig 2", fps=30.0)
        library.create_dataset(self.libraries, "rig2-labels", setup="mine:rig2", splits={"rec": "train"})
        library.Collection(self.libraries, "mine:rig2").save(recording="rec", frame=1, origin="spread", **label_inputs(1))
        weights = root / "best.ckpt"
        weights.write_bytes(b"weights")
        library.create_model(self.libraries, "rig2-ft", {
            "name": "rig2-ft", "kind": "body_net", "setup": "mine:rig2", "outputs": ["mask", "ap", "head", "tail", "overlap"],
            "inputs": library.make_inputs([1], fps=30.0, pixel_size_um=None),
            "trained_on": library.trained_on("mine:rig2-labels", library.labels(self.libraries, "mine:rig2-labels")),
        }, weights)
        library.save_evaluation(self.libraries, "mine:rig2-ft", "lab:v1", {"iou_mean": 0.9})
        library.save_evaluation(self.libraries, "mine:rig2-ft", "mine:rig2-labels-b1", {"iou_mean": 0.8})

    def run_publish(self, *argv):
        publish.main([*argv, "--lab", str(self.libraries.lab), "--library", str(self.libraries.personal)])

    def test_publish_in_dependency_order_and_freeze(self):
        with self.assertRaisesRegex(SystemExit, "mine:rig2 is not in the lab library"):
            self.run_publish("model", "mine:rig2-ft")
        self.run_publish("setup", "mine:rig2")
        self.run_publish("labels", "mine:rig2")
        self.run_publish("labels", "mine:rig2")  # nothing new: copied once
        self.run_publish("dataset", "mine:rig2-labels")
        self.run_publish("model", "mine:rig2-ft", "--default", "body", "--reason", "first model for rig 2")
        card = library.get_card(self.libraries, "lab:rig2-ft")
        self.assertEqual((card.setup, card.trained_on[0]["dataset"]), ("lab:rig2", "lab:rig2-labels"))
        self.assertEqual(library.weights_path(self.libraries, "lab:rig2-ft").read_bytes(), b"weights")
        self.assertEqual(set(library.evaluations(self.libraries, "lab:rig2-ft")), {"lab:v1"})  # personal-benchmark scores stay personal
        self.assertEqual(library.get_setup(self.libraries, "lab:rig2").defaults, {"body": "lab:rig2-ft"})
        self.assertEqual(library.defaults_log(self.libraries, "lab:rig2")[0]["reason"], "first model for rig 2")
        published = library.labels(self.libraries, "lab:rig2-labels")
        self.assertEqual([(r.scope, r.setup, r.revision, r.split, r.sha256) for r in published],
                         [("lab", "lab:rig2", 1, "train", r.sha256) for r in library.labels(self.libraries, "mine:rig2-labels")])
        np.testing.assert_array_equal(published[0].load().mask, label_inputs(1)["mask"])
        self.assertEqual(json.loads((self.libraries.lab / "datasets" / "rig2-labels" / "dataset.json").read_text())["setup"], "lab:rig2")
        # The personal items are copied, not moved; a published id is never replaced.
        self.assertTrue((self.libraries.personal / "models" / "rig2-ft" / "weights.ckpt").exists())
        with self.assertRaisesRegex(SystemExit, "already has model rig2-ft"):
            self.run_publish("model", "mine:rig2-ft")
        self.run_publish("model", "mine:rig2-ft", "--as", "rig2-ft-v2")
        self.assertEqual(library.get_card(self.libraries, "lab:rig2-ft-v2").name, "rig2-ft")
        with self.assertRaises(SystemExit):
            self.run_publish("dataset", "lab:rig2-labels")


if __name__ == "__main__":
    unittest.main()
