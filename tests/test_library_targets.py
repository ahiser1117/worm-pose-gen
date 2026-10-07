"""The body-target cache of the library: built from a label alone, keyed by its revision and builder, oriented by the person's choice."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from worm_pose_gen import body_fields, library
from worm_pose_gen.batch_fit import PRESETS
from worm_pose_gen.library import Libraries, targets


SHAPE = (96, 400)
MAX_LAG = 2


def worm_label():
    """A straight worm along y=48 from x=50 to x=350, thicker at the left end."""

    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    half_width = np.interp(xx, [50, 350], [9.0, 5.0])
    mask = ((np.abs(yy - 48) <= half_width) & (xx >= 50) & (xx <= 350)).astype(np.uint8)
    image = np.where(mask == 1, 60, 200).astype(np.uint8)
    context = np.repeat(image[None], 2 * MAX_LAG + 1, axis=0)
    nose = np.full((2 * MAX_LAG + 1, 2), np.nan)
    nose[MAX_LAG] = (345.0, 48.0)  # the acquisition says the head is at the right end
    valid = np.zeros(2 * MAX_LAG + 1, bool)
    valid[MAX_LAG] = True
    return dict(image=image, image_raw=image, mask=mask, context=context, context_valid=np.ones(2 * MAX_LAG + 1, bool),
                nose_xy=nose, nose_valid=valid)


def fast_fit_config():
    """The fast schedule: the reference one the builder uses takes minutes on a CPU."""

    return replace(PRESETS["fast"], length_bounds_px=body_fields.FIT_LENGTH_BOUNDS_PX)


class TargetCacheTests(unittest.TestCase):
    @mock.patch.object(body_fields, "fit_config", fast_fit_config)
    def test_build_once_per_revision_with_nose_then_manual_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library.write_setup(root / "mine", "rig", name="Rig")
            libraries = Libraries(lab=None, personal=root / "mine")
            dataset = library.create_dataset(libraries, "d", setup="mine:rig")
            auto = dataset.save(recording="rec", frame=7, origin="spread", **worm_label())
            self.assertIsNone(library.cached_meta(libraries, auto))
            meta = library.build_targets(libraries, auto, device="cpu")
            self.assertEqual((meta["has_body"], meta["orientation"], meta["fit_method"]), (True, "nose", "independent"))
            self.assertGreater(meta["fit_iou"], 0.85)
            self.assertFalse(meta["self_contact"])
            self.assertTrue(meta["in_view"])
            self.assertEqual(meta["label"], auto.identity)
            stored, arrays = library.load_targets(libraries, auto)
            self.assertEqual(stored, meta)
            self.assertGreater(arrays["head_xy"][0], 300)  # head at the nose end
            self.assertLess(float(np.nanmean(arrays["ap"][:, 300:])), float(np.nanmean(arrays["ap"][:, :100])))
            built = (libraries.personal / "cache" / "body_targets" / "none" / f"{auto.sha256}.json").stat().st_mtime_ns
            self.assertEqual(library.build_targets(libraries, auto, device="cpu"), meta)  # cached: not refit
            self.assertEqual((libraries.personal / "cache" / "body_targets" / "none" / f"{auto.sha256}.json").stat().st_mtime_ns, built)
            # A person puts the head at the left end: a new revision, new targets, the old ones untouched.
            flipped = dataset.save(recording="rec", frame=7, origin="fix", orientation="manual", head_xy=[52.0, 48.0], **worm_label())
            meta = library.build_targets(libraries, flipped, device="cpu")
            self.assertEqual(meta["orientation"], "manual")
            _, arrays = library.load_targets(libraries, flipped)
            self.assertLess(arrays["head_xy"][0], 100)
            self.assertEqual(library.cached_meta(libraries, auto)["orientation"], "nose")
            rows = sorted(p.suffix for p in (libraries.personal / "cache" / "body_targets" / "none").iterdir())
            self.assertEqual(rows, [".json", ".json", ".npz", ".npz"])

    @mock.patch.object(body_fields, "fit_config", fast_fit_config)
    def test_a_new_default_mask_model_means_new_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = Libraries(lab=None, personal=root / "mine")
            library.write_setup(libraries.personal, "rig", name="Rig")
            weights = root / "w.ckpt"
            weights.write_bytes(b"w")
            library.create_model(libraries, "seg", {"name": "seg", "kind": "segmenter", "setup": "mine:rig", "outputs": ["mask"],
                                                    "inputs": library.make_inputs([], fps=20.0, pixel_size_um=1.0)}, weights)
            record = library.create_dataset(libraries, "d", setup="mine:rig").save(recording="rec", frame=7, origin="spread", **worm_label())
            self.assertIsNone(library.target_builder(libraries, "mine:rig"))
            library.build_targets(libraries, record, device="cpu")
            self.assertEqual(library.cached_meta(libraries, record)["builder"], None)
            library.set_default(libraries, "mine:rig", "mask", "mine:seg", reason="first model")
            self.assertEqual(library.target_builder(libraries, "mine:rig"), "mine:seg")
            self.assertIsNone(library.cached_meta(libraries, record))  # built by another model: missing until rebuilt
            self.assertIsNone(library.load_targets(libraries, record))
            self.assertIsNotNone(library.cached_meta(libraries, record, None))
            # The builder's mask model segments the context frames for a chain fit; here every frame is the label's.
            segmenter = mock.Mock()
            segmenter.predict_probability_batch.side_effect = lambda frames, batch_size=8: np.stack([worm_label()["mask"]] * len(frames)).astype(np.float32)
            meta = library.build_targets(libraries, record, segmenter=segmenter, device="cpu")
            self.assertEqual(meta["builder"], "mine:seg")
            self.assertEqual(library.cached_meta(libraries, record), meta)
            self.assertTrue((libraries.personal / "cache" / "body_targets" / "mine.seg" / f"{record.sha256}.npz").is_file())

    def test_job_command_names_the_label_revisions(self):
        libraries = Libraries(lab=Path("/lab"), personal=Path("/mine"))
        identity = {"dataset": "mine:d", "recording": "rec", "frame": 7, "revision": 2, "sha256": "x"}
        command = targets.targets_command(libraries, [identity])
        self.assertEqual(command[1:5], ["-m", "worm_pose_gen.library.targets", "--personal", "/mine"])
        self.assertEqual(json.loads(command[command.index("--labels") + 1]), [{"dataset": "mine:d", "recording": "rec", "frame": 7, "revision": 2}])
        self.assertEqual(command[-2:], ["--lab", "/lab"])


if __name__ == "__main__":
    unittest.main()
