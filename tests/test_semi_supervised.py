import itertools
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import h5py
import lightning
import numpy as np
import torch

from worm_pose_gen.body_net import OUTPUTS, load_body_net
from worm_pose_gen.semi_supervised import (
    SemiBodyFieldModule, SemiConfig, UnlabeledWindows, apply_transform, consistency_terms, export_body_checkpoint, ramp,
    undo_transform,
)
from worm_pose_gen.unlabeled_clips import ClipStore, build_store, plan_segments


SHAPE = (64, 160)
LENGTH = 80.0
SPEED = 3.0


def body_logits(t: float, *, swapped: bool = False, head_offset: float = 0.0) -> torch.Tensor:
    """``[5, H, W]`` confident logits of a straight 12-px-wide body moving right by ``SPEED`` px per frame (head on the right)."""

    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]].astype(np.float64)
    tail_x = 20.0 + SPEED * t
    head_x = tail_x + LENGTH
    inside = (np.abs(yy - 32) <= 6) & (xx >= tail_x) & (xx <= head_x)
    ap = np.clip((head_x - xx) / LENGTH, 0.0, 1.0)  # 0 at the head
    if swapped:
        ap = 1.0 - ap
    logits = np.zeros((len(OUTPUTS), *SHAPE))
    logits[OUTPUTS.index("mask")] = np.where(inside, 20.0, -20.0)
    logits[OUTPUTS.index("ap")] = np.log(np.clip(ap, 1e-4, 1 - 1e-4) / np.clip(1 - ap, 1e-4, 1))
    head = (head_x + head_offset, 32.0) if not swapped else (tail_x, 32.0)
    tail = (tail_x, 32.0) if not swapped else (head_x, 32.0)
    for name, (x, y) in (("head", head), ("tail", tail)):
        heat = np.exp(-0.5 * (((xx - round(x)) ** 2 + (yy - y) ** 2) / 10.0**2))
        logits[OUTPUTS.index(name)] = np.log(np.clip(heat, 1e-6, 1 - 1e-6) / np.clip(1 - heat, 1e-6, 1))
    logits[OUTPUTS.index("overlap")] = -20.0
    return torch.as_tensor(logits, dtype=torch.float32)


def window(**student_options) -> tuple[torch.Tensor, torch.Tensor]:
    teacher = torch.stack([body_logits(t) for t in (-1.0, 0.0, 1.0)])[None]
    return body_logits(0.0, **student_options)[None], teacher


CONFIG = SemiConfig(store="-", min_area=100, tail_temporal=0.1, tail_teacher=0.1)


class ConsistencyLossTests(unittest.TestCase):
    def test_consistent_moving_body_costs_nothing(self):
        terms, stats = consistency_terms(*window(), CONFIG)
        self.assertEqual(float(stats["items"]), 1.0)
        self.assertGreater(float(stats["sandwich_px"]), 500)
        self.assertGreater(float(stats["ap_temporal_px"]), 500)
        for name in ("mask_sandwich", "mask_teacher", "ap_temporal", "ap_teacher"):
            self.assertLess(float(terms[name]), 1e-4, name)

    def test_head_tail_swap_is_penalized(self):
        consistent, _ = consistency_terms(*window(), CONFIG)
        swapped, _ = consistency_terms(*window(swapped=True), CONFIG)
        self.assertGreater(float(swapped["ap_temporal"]), 0.3)
        self.assertGreater(float(swapped["head_temporal"]), 10 * float(consistent["head_temporal"]))
        self.assertGreater(float(swapped["tail_temporal"]), 10 * float(consistent["tail_temporal"]))

    def test_flickering_mask_is_penalized(self):
        student, teacher = window()
        student[0, OUTPUTS.index("mask"), 30:34, 60:80] = -20.0  # a hole the neighbours do not have
        terms, _ = consistency_terms(student, teacher, CONFIG)
        self.assertGreater(float(terms["mask_sandwich"]), 0.1)

    def test_head_jump_is_penalized_and_small_motion_is_not(self):
        near, _ = consistency_terms(*window(head_offset=1.0), CONFIG)
        far, _ = consistency_terms(*window(head_offset=-40.0), CONFIG)
        self.assertGreater(float(far["head_temporal"]), 5 * float(near["head_temporal"]))

    def test_zero_weight_hands_pixels_to_the_teacher_term(self):
        student, teacher = window()
        student[0, OUTPUTS.index("mask"), 30:34, 60:80] = -20.0
        terms, _ = consistency_terms(student, teacher, replace(CONFIG, mask_sandwich=0.0))
        self.assertEqual(float(terms["mask_sandwich"]), 0.0)
        self.assertGreater(float(terms["mask_teacher"]), 0.1)

    def test_empty_window_is_ignored(self):
        student, teacher = window()
        teacher[:, :, OUTPUTS.index("mask")] = -20.0
        terms, stats = consistency_terms(student, teacher, CONFIG)
        self.assertEqual(float(stats["items"]), 0.0)
        for value in terms.values():
            self.assertEqual(float(value), 0.0)


class TransformTests(unittest.TestCase):
    def test_undo_inverts_every_flip_and_rotation(self):
        x = torch.arange(2 * 6 * 6, dtype=torch.float32).reshape(2, 6, 6)
        for fx, fy, k in itertools.product((False, True), (False, True), (0, 1)):
            torch.testing.assert_close(undo_transform(apply_transform(x, fx, fy, k), fx, fy, k), x)

    def test_ramp(self):
        config = SemiConfig(store="-", weight=2.0, start_epoch=1.0, ramp_epochs=4.0)
        self.assertEqual(ramp(0.5, config), 0.0)
        self.assertLess(ramp(1.5, config), 0.2)
        self.assertEqual(ramp(5.0, config), 2.0)


class ExportTests(unittest.TestCase):
    def test_exported_checkpoint_loads_as_a_plain_body_net(self):
        module = SemiBodyFieldModule(semi={"store": "-"}, lags=(1, 4), pretrained=False, heatmap_weight=0.2)
        with torch.no_grad():
            for parameter in module.teacher.parameters():
                parameter.add_(1.0)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "semi.ckpt"
            torch.save({"epoch": 3, "global_step": 30, "pytorch-lightning_version": lightning.__version__,
                        "state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, source)
            for network in ("student", "teacher"):
                target = export_body_checkpoint(source, Path(tmp) / f"{network}.ckpt", network=network)
                loaded = load_body_net(target, device="cpu")
                self.assertEqual(loaded.lags, (1, 4))
                self.assertEqual(loaded.hparams.heatmap_weight, 0.2)
                expected = getattr(module, "network" if network == "student" else "teacher").state_dict()
                for key, value in loaded.network.state_dict().items():
                    torch.testing.assert_close(value, expected[key])


def write_recording(path: Path, frames: int = 120) -> None:
    rng = np.random.default_rng(0)
    data = rng.integers(180, 230, size=(frames, 48, 64), dtype=np.uint8)
    data[:, 20:28, 10:50] = 80  # a dark worm
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=data, chunks=(1, 48, 64))


class ClipStoreTests(unittest.TestCase):
    def test_plan_avoids_excluded_frames_and_recordings(self):
        excluded = lambda recording, frame: Path(recording).stem == "a" and 100 <= frame < 200
        planned = plan_segments([("/x/a.h5", 1000), ("/x/b.h5", 1000), ("/y/a.h5", 1000), ("/x/c.h5", 1000)],
                                per_recording=5, length=40, seed=0, excluded=excluded, excluded_recordings=["c"])
        self.assertEqual(sorted({Path(s.recording).parent.name + Path(s.recording).stem for s in planned}), ["xa", "xb"])
        for segment in planned:
            if Path(segment.recording).stem == "a":
                self.assertTrue(segment.start + segment.length <= 100 or segment.start >= 200)
        starts = sorted(s.start for s in planned if Path(s.recording).stem == "b")
        self.assertTrue(all(b - a >= 40 for a, b in zip(starts, starts[1:])))

    def test_default_exclusion_rejects_test_recordings(self):
        from worm_pose_gen.temporal_eval import excluded

        planned = plan_segments([("/x/2024-06-18-12.h5", 20000)], per_recording=3, length=40, seed=0, excluded=excluded)
        self.assertEqual(planned, [])

    def test_store_round_trip_and_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            recording = Path(tmp) / "rec.h5"
            write_recording(recording)
            broken = Path(tmp) / "broken.h5"
            broken.write_bytes(b"not an hdf5 file")
            written = build_store(Path(tmp) / "store", [str(recording), str(broken)], per_recording=2, length=40, seed=1,
                                  excluded=lambda r, f: False, workers=1, log=lambda m: None)
            self.assertEqual(len(written), 2)
            store = ClipStore(Path(tmp) / "store")
            self.assertEqual(store.recordings(), ["rec"])
            self.assertEqual(store.frames(0).shape, (40, 48, 64))
            self.assertEqual(list(store.centres(0, 1, 16)), list(range(17, 23)))
            windows = UnlabeledWindows(store, (1, 4, 16), crop=32, augment=False)
            self.assertEqual(len(windows), 12)
            item = windows[5]
            self.assertEqual(tuple(item["teacher"].shape), (3, 4, 32, 32))
            torch.testing.assert_close(item["student"], item["teacher"][1])
            augmented = UnlabeledWindows(store, (1, 4, 16), crop=32, seed=3)[5]
            self.assertEqual(tuple(augmented["student"].shape), (4, 32, 32))
            torch.testing.assert_close(UnlabeledWindows(store, (1, 4, 16), crop=32, seed=3)[5]["student"], augmented["student"])


if __name__ == "__main__":
    unittest.main()
