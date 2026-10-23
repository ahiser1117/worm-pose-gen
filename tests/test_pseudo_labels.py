"""Self-training pseudo-labels: frame selection on synthetic workspace states, target building, items, library sets, the sampler, one epoch and its recipe."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from worm_pose_gen import library, model_training, pseudo_labels
from worm_pose_gen.body_net import _AP_VALID, _HEAD, _MASK, _TAIL, _VALID, BodyFieldModule
from worm_pose_gen.pose_run import render_tube
from worm_pose_gen.pseudo_labels import IGNORE, PseudoLabelDataset, SelectionParams, select_rows

from tests.test_model_training import SHAPE, LibraryTestCase, straight_worm


N = 200
PARAMS = SelectionParams(window=5, clear_stride=4, bridged_stride=2, max_stretch=30)


def synthetic_state(n=N):
    """A straight worm drifting one pixel per row, every row a clean independent fit."""

    x = np.linspace(100.0, 400.0, 50)
    centerlines = np.stack([np.stack((x + 0.5 * t, np.full(50, 200.0)), 1) for t in range(n)])
    state = {
        "frame_index": np.arange(1000, 1000 + n), "fitted": np.ones(n, bool), "iou": np.full(n, 0.97),
        "centerline_xy": centerlines, "width_px": np.full(n, 40.0), "body_length_px": np.full(n, 300.0),
        "pose_jump_px": np.r_[np.nan, np.full(n - 1, 0.5)],
    }
    for flag in ("low_iou", "area_deficit", "area_excess", "self_contact", "holes", "fragments", "length_deviation",
                 "pose_jump", "edge_inside"):
        state[f"flag_{flag}"] = np.zeros(n, bool)
    return state, np.full(n, "independent_fit", dtype="<U32")


class SelectionTests(unittest.TestCase):
    def test_clean_rows_are_kept_at_the_stride_away_from_the_ends(self):
        state, algorithm = synthetic_state()
        selection = select_rows(state, algorithm, [], PARAMS, image_shape=(480, 640))
        frames = state["frame_index"][selection["rows"]]
        self.assertTrue(np.all(frames % 4 == 0))
        self.assertEqual(set(selection["kind"]), {"clear"})
        self.assertEqual((frames.min(), frames.max()), (1008, 1192))  # rows 5..194 have a full window
        self.assertEqual(selection["counts"]["orientation_flips"], 0)

    def test_a_flip_a_jump_or_a_length_change_spoils_its_window(self):
        for spoil in ("flip", "jump", "length"):
            state, algorithm = synthetic_state()
            if spoil == "flip":
                state["centerline_xy"][100:] = state["centerline_xy"][100:, ::-1]
            elif spoil == "jump":
                state["pose_jump_px"][100] = 60.0
            else:
                state["body_length_px"][100:] = 330.0
            rows = select_rows(state, algorithm, [], PARAMS, image_shape=(480, 640))["rows"]
            self.assertFalse(np.any((rows >= 95) & (rows <= 104)), spoil)  # pair (99, 100) lies in their windows
            self.assertTrue(np.any(rows == 92) and np.any(rows == 108), spoil)

    def test_failure_flags_and_low_iou_drop_clear_rows_but_pose_flags_do_not(self):
        state, algorithm = synthetic_state()
        state["flag_fragments"][40] = True
        state["iou"][44] = 0.94
        state["flag_self_contact"][48] = True
        rows = select_rows(state, algorithm, [], PARAMS)["rows"]
        self.assertNotIn(40, rows)
        self.assertNotIn(44, rows)
        self.assertIn(48, rows)

    def test_bridged_rows_need_chain_fits_and_anchored_short_stretches(self):
        state, algorithm = synthetic_state()
        for first, last in ((60, 70), (120, 130), (150, 190)):
            algorithm[first : last + 1] = "chain_forward"
            state["iou"][first : last + 1] = 0.91
            state["flag_area_excess"][first : last + 1] = True  # a contact; allowed inside a bridged stretch
        state["iou"][119] = 0.85  # the second stretch has a poor anchor
        state["flag_edge_inside"][66] = True
        selection = select_rows(state, algorithm, [(60, 70), (120, 130), (150, 190)], PARAMS)
        bridged = selection["rows"][selection["kind"] == "bridged"]
        self.assertEqual(bridged.tolist(), [60, 62, 64, 68, 70])  # 66 carries an edge flag; 150-190 is too long
        self.assertEqual(selection["counts"]["kept_bridged"], 5)
        clear = selection["rows"][selection["kind"] == "clear"]
        self.assertFalse(np.any((clear >= 60) & (clear <= 70)))

    def test_excluded_rows_are_never_kept(self):
        state, algorithm = synthetic_state()
        excluded = np.zeros(N, bool)
        excluded[:100] = True
        rows = select_rows(state, algorithm, [], PARAMS, excluded=excluded)["rows"]
        self.assertTrue(np.all(rows >= 100))


class TargetTests(unittest.TestCase):
    def test_mask_target_ignores_the_edge_band_and_corrects_far_disagreement(self):
        tube = np.zeros((60, 80), bool)
        tube[20:30, 10:70] = True
        clean = tube.copy()
        clean[19, 10:70] = True  # one pixel wider: the width model's call, ignored
        clean[45:55, 30:40] = True  # a blob far from the body: background
        clean[24:26, 40:42] = False  # a hole inside the body: worm
        target = pseudo_labels.mask_target(clean, tube, band_px=3.0)
        self.assertTrue(np.all(target[19, 10:70] == IGNORE))
        self.assertTrue(np.all(target[45:55, 30:40] == 0))
        self.assertTrue(np.all(target[24:26, 40:42] == 1))
        self.assertTrue(np.all(target[tube & (clean == tube)] == 1))

    def test_item_targets_follow_the_pose_and_leave_the_tail_out(self):
        centerline, mask = straight_worm()
        width = np.full(len(centerline), 8.0)
        tube = render_tube(centerline, width, *SHAPE, device="cpu")
        item = pseudo_labels.render_item_targets(mask, centerline, width, band_px=2.0)
        self.assertTrue(np.all(item["mask"][tube & mask] == 1))
        rows = pseudo_labels.target_rows(item)
        self.assertTrue(np.isnan(rows[_TAIL]).all())
        peak = np.unravel_index(np.argmax(rows[_HEAD]), SHAPE)
        self.assertLess(abs(peak[1] - centerline[0, 0]) + abs(peak[0] - centerline[0, 1]), 2.0)
        self.assertTrue(np.all(rows[_VALID][item["mask"] == IGNORE] == 0))
        ap = item["ap"][rows[_AP_VALID] > 0]
        self.assertLess(float(ap.min()), 0.1)
        self.assertGreater(float(ap.max()), 0.9)
        self.assertTrue(np.all(rows[_MASK] == (item["mask"] == 1)))


def write_items(directory: Path, count: int = 3, offsets=pseudo_labels.CONTEXT_OFFSETS) -> None:
    """Pseudo-label items of the straight worm, as :func:`pseudo_labels.build_items` writes them."""

    centerline, mask = straight_worm()
    width = np.full(len(centerline), 8.0)
    targets = pseudo_labels.render_item_targets(mask, centerline, width, band_px=2.0)
    image = np.where(mask, 60, 200).astype(np.uint8)
    frames = np.stack([np.roll(image, int(np.sign(o)) + o // 4, axis=1) for o in offsets])
    entries = []
    directory.mkdir(parents=True, exist_ok=True)
    for k in range(count):
        name = f"ws_{k:06d}.npz"
        np.savez_compressed(directory / name, frames=frames, offsets=np.asarray(offsets),
                            valid=np.ones(len(frames), bool), centerline_xy=centerline, width_profile=width, **targets)
        entries.append({"file": name, "workspace": "ws", "recording": "/data/rec-x.h5", "frame": k, "kind": "clear" if k else "bridged"})
    pseudo_labels.write_index(directory, entries, SelectionParams(), offsets=offsets)


class DatasetTests(unittest.TestCase):
    def test_items_match_the_label_layout_and_train(self):
        with tempfile.TemporaryDirectory() as root:
            write_items(Path(root) / "items")
            dataset = PseudoLabelDataset([Path(root) / "items"], lags=(1, 4, 16), augment=True, crop_size=64)
            item = dataset[0]
            self.assertEqual(tuple(item["image"].shape), (4, 64, 64))
            self.assertEqual(tuple(item["targets"].shape), (8, 64, 64))
            self.assertGreater(float(item["image"][3].abs().max()), 0.0)  # lag 16 sees motion
            batch = model_training.collate([item, dataset[1]])
            module = BodyFieldModule(lags=(1, 4, 16), pretrained=False)
            total, parts = module.loss(module(batch["image"]), batch["targets"])
            self.assertTrue(torch.isfinite(total))
            self.assertEqual(float(parts["tail_focal"]), 0.0)
            self.assertEqual(dataset.counts(), {"counts": {"pseudo": 3, "bridged": 1, "clear": 2}, "recordings": 1})
            with self.assertRaisesRegex(ValueError, "offsets"):
                PseudoLabelDataset([Path(root) / "items"], lags=(2,))


class PseudoLabelSetTests(LibraryTestCase):
    def test_a_set_is_a_library_item_of_its_setup(self):
        pseudo = library.create_pseudo_labels(self.libraries, "self", setup="lab:rig", description="confident fits")
        write_items(pseudo.root, count=4)
        self.assertEqual([p.ref for p in library.list_pseudo_labels(self.libraries, "lab:rig")], ["mine:self"])
        summary = library.get_pseudo_labels(self.libraries, "mine:self").summary()
        self.assertEqual((summary["items"], summary["counts"], summary["recordings"]), (4, {"bridged": 1, "clear": 3}, 1))
        with self.assertRaises(FileExistsError):
            library.create_pseudo_labels(self.libraries, "self", setup="lab:rig")
        with self.assertRaisesRegex(LookupError, "unknown pseudo-label set"):
            library.get_pseudo_labels(self.libraries, "lab:self")


class TrainWithPseudoLabelsTests(LibraryTestCase):
    def test_one_cpu_epoch_with_pseudo_labels_writes_them_on_the_card(self):
        pseudo = library.create_pseudo_labels(self.libraries, "self", setup="lab:rig")
        write_items(pseudo.root, count=4, offsets=(-2, -1, 0, 1, 2))  # the fixture's body net has lags 1 and 2
        with self.assertRaisesRegex(ValueError, "pseudo-label fraction"):
            model_training.plan(self.libraries, model_training.TrainRequest(
                setup="lab:rig", dataset="lab:base", start_from="lab:body", pseudo_labels=["mine:self"]))
        with self.assertRaisesRegex(LookupError, "unknown pseudo-label set"):
            model_training.plan(self.libraries, model_training.TrainRequest(
                setup="lab:rig", dataset="lab:base", start_from="lab:body", pseudo_labels=["mine:other"], pseudo_fraction=0.5))
        request = model_training.TrainRequest(
            setup="lab:rig", dataset="lab:base", start_from="lab:body", name="selftrained", notes="pseudo",
            params={"max_epochs": 1, "batch_size": 2, "crop_size": 64, "learning_rate": 1e-4}, pseudo_labels=["mine:self"],
            pseudo_fraction=0.5, fiber_share=0.5,
        )
        card = model_training.train(self.libraries, request, device="cpu", num_workers=0)
        self.assertEqual(card.trained_on[0]["counts"], {"train": 2, "val": 1, "test": 0})
        self.assertEqual(card.trained_on[1], {"pseudo_labels": ["mine:self"], "counts": {"pseudo": 4, "bridged": 1, "clear": 3},
                                              "recordings": 1})
        self.assertEqual(card.hparams["pseudo_fraction"], 0.5)
        self.assertEqual(card.hparams["pseudo_labels"], ["mine:self"])
        # The recipe repeats the run: its settings, the set and the shares; from scratch, the context its lags are.
        recipe = model_training.recipe(card)
        self.assertEqual({k: recipe[k] for k in ("pseudo_labels", "pseudo_fraction", "fiber_share", "context")},
                         {"pseudo_labels": ["mine:self"], "pseudo_fraction": 0.5, "fiber_share": 0.5, "context": None})
        self.assertEqual((recipe["params"]["learning_rate"], recipe["params"]["max_epochs"]), (1e-4, 1))
        argv = model_training.command(self.libraries, request, model_training.plan(
            self.libraries, model_training.TrainRequest(setup="lab:rig", dataset="lab:base", start_from="lab:body", name="other",
                                                        pseudo_labels=["mine:self"], pseudo_fraction=0.5)))
        self.assertEqual(argv[argv.index("--pseudo-labels") + 1], "mine:self")


if __name__ == "__main__":
    unittest.main()
