import tempfile
import unittest
from pathlib import Path

import numpy as np

from worm_pose_gen import synthetic_clips as sc
from worm_pose_gen.body_net import body_target_rows
from worm_pose_gen.body_targets import render_body_targets, self_contact
from worm_pose_gen.segmenter import IGNORE_LABEL


SHAPE = (160, 200)


def straight(length=120.0, x0=40.0, y=80.0, n=100):
    return np.column_stack((np.linspace(x0, x0 + length, n), np.full(n, y)))


def width_profile(n=100, diameter=14.0):
    s = np.linspace(0, 1, n)
    return diameter * np.clip(np.minimum(s, 1 - s) / 0.1, 0.15, 1.0)


def loop(n=100):
    """A centerline that crosses itself once (a curl of more than a full turn)."""

    angle = np.linspace(-0.2, 2.4 * np.pi, n - 1)
    step = 4.5
    curve = np.vstack((np.zeros(2), np.cumsum(step * np.column_stack((np.cos(angle), np.sin(angle))), axis=0)))
    # Pull the turns apart radially so the tail crosses over the head segment.
    curve[:, 1] += np.linspace(0, 30, n)
    return curve - curve.mean(0) + np.array([100.0, 80.0])


class FakeSource:
    """Duck-typed :class:`synthetic_clips.Source`: a slowly moving straight worm on a textured background."""

    name = "fake"
    path = Path("/fake")

    def __init__(self, frames=60):
        rng = np.random.default_rng(3)
        self.frame_index = np.arange(frames)
        self.centerline = np.stack([straight(x0=30.0 + 0.5 * t) for t in range(frames)])
        self.width = np.tile(width_profile(), (frames, 1))
        self.good = np.ones(frames, dtype=bool)
        self.background = 180 + rng.normal(0, 3, (frames, *SHAPE))

    def mask(self, row):
        return sc.body_layers(self.centerline[row], self.width[row], SHAPE).mask_in(SHAPE)

    def frames(self, rows):
        out = []
        for row in rows:
            image = self.background[row].copy()
            image[self.mask(row)] *= 0.8
            out.append(np.clip(image, 0, 255).astype(np.uint8))
        return np.stack(out)


def texture():
    along, across = sc.TEXTURE_SHAPE
    u = np.linspace(-sc.U_MAX, sc.U_MAX, across)
    profile = np.where(np.abs(u) < 1, 0.8 - 0.1 * (np.abs(u) > 0.7), 1.0)
    # A body of the test width profile (90 % of the length), then a 3-px whip.
    s = np.linspace(0, 1, along)
    width = np.where(s <= 0.9, np.interp(s / 0.9, np.linspace(0, 1, 100), width_profile()), 3.0)
    return sc.Texture(np.tile(profile, (along, 1)).astype(np.float32), width.astype(np.float32), 0.9, "test")


class BodyLayersTest(unittest.TestCase):
    def test_straight_tube_coordinates(self):
        centerline, width = straight(), np.full(100, 14.0)
        layers = sc.body_layers(centerline, width, SHAPE)
        mask = layers.mask_in(SHAPE)
        # The tube covers the pixels within a radius of the centerline.
        self.assertTrue(mask[80, 100] and mask[86, 100] and not mask[88, 100])
        u = layers.u[0, 80 - layers.y0 + 3, 100 - layers.x0]
        self.assertAlmostEqual(abs(float(u)), 3 / 7, places=3)
        self.assertAlmostEqual(float(layers.s[0, 80 - layers.y0, 100 - layers.x0]), 60 / 120, places=2)
        # A straight body never has a second layer.
        self.assertFalse(np.isfinite(layers.coverage[1]).any())

    def test_overlap_where_the_tube_crosses_itself(self):
        centerline, width = loop(), width_profile()
        self.assertTrue(self_contact(centerline, width))
        layers = sc.body_layers(centerline, width, SHAPE)
        mask = layers.mask_in(SHAPE)
        targets = render_body_targets(mask, centerline, width)
        both = np.zeros(SHAPE, dtype=bool)
        box = (slice(layers.y0, layers.y0 + layers.s.shape[1]), slice(layers.x0, layers.x0 + layers.s.shape[2]))
        both[box] = (layers.coverage > 0).all(0)
        self.assertGreater(targets.overlap.sum(), 20)
        np.testing.assert_array_equal(targets.overlap, both)


class PoseTransformTest(unittest.TestCase):
    def test_identity_keeps_the_pose_and_gain_bends_it(self):
        s = (np.arange(99) + 0.5) / 99
        angle = 0.6 * np.sin(2 * np.pi * 1.5 * s) + 0.3
        centerline = np.vstack((np.zeros(2), np.cumsum(3.0 * np.column_stack((np.cos(angle), np.sin(angle))), axis=0))) + 100
        length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
        same = sc.transform_pose(centerline, length, sc.PoseTransform())
        self.assertLess(np.abs(same - centerline).max(), 1.0)
        bent = sc.transform_pose(centerline, length, sc.PoseTransform(gain=2.0, scale=1.1))
        np.testing.assert_allclose(bent.mean(0), centerline.mean(0), atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(np.diff(bent, axis=0), axis=1).sum(), 1.1 * length, places=3)
        self.assertLess(sc.min_bend_radius(bent), sc.min_bend_radius(centerline))

    def test_real_offsets_strictly_increase(self):
        self.assertEqual(sc.real_offsets((1, 4, 16), 1.0), [1, 4, 16])
        self.assertEqual(sc.real_offsets((1, 4, 16), 0.25), [1, 2, 4])
        self.assertEqual(sc.real_offsets((1, 4, 16), 2.0), [2, 8, 32])


class ItemTest(unittest.TestCase):
    def make(self, seed, want_contact=False):
        source = FakeSource()
        return sc.make_item(source, [texture()], 35, (1, 4, 16), np.random.default_rng(seed), want_contact=want_contact, tries=200)

    def test_targets_match_the_rendered_worm(self):
        item = self.make(0)
        self.assertIsNotNone(item)
        self.assertEqual(item["frames"].shape, (7, *SHAPE))
        worm = item["mask"] == 1
        tube = sc.body_layers(item["centerline_xy"], item["width_profile"], SHAPE).mask_in(SHAPE)
        # The mask is the tube plus the tail whip beyond its end.
        self.assertFalse((tube & ~worm).any())
        whip = worm & ~tube
        self.assertGreater(whip.sum(), 10)
        tail = item["centerline_xy"][-1]
        yy, xx = np.nonzero(whip)
        self.assertLess(np.hypot(xx - tail[0], yy - tail[1]).min(), 3.0)
        # Ignored pixels are the filled real worm away from the synthetic one.
        self.assertFalse(((item["mask"] == IGNORE_LABEL) & worm).any())
        ap = item["ap"].astype(np.float32)
        self.assertTrue(np.isfinite(ap[worm & ~item["overlap"]]).all())
        self.assertFalse(np.isfinite(ap[~worm]).any())
        # Whip pixels take the A-P of the tube's tail end, as hand labels do.
        self.assertGreater(float(ap[whip].min()), 0.95)
        np.testing.assert_allclose(item["head_xy"], item["centerline_xy"][0], atol=1e-4)
        # The worm is darker than its surroundings in the rendered frame.
        self.assertLess(item["frames"][3][tube].mean(), item["frames"][3][~tube & (item["mask"] != IGNORE_LABEL)].mean() - 10)

    def test_whip_measured_and_sampled_from_the_mask(self):
        centerline = straight(length=100.0, x0=40.0)
        width = np.full(100, 14.0)
        mask = sc.body_layers(centerline, width, SHAPE).mask_in(SHAPE)
        mask[79:82, 140:156] = True  # a 3-px whip of 15 px beyond the tail end at x = 140
        whip, _ = sc.measure_whip(mask, centerline)
        self.assertAlmostEqual(whip, 15.0, delta=3.0)
        frame = np.where(mask, 120.0, 180.0)
        texture = sc.sample_texture(frame, np.full(SHAPE, 180.0), centerline, width, mask)
        self.assertAlmostEqual(texture.body_fraction, 99 / (99 + whip), places=2)
        along = np.linspace(0, 1, len(texture.width))
        on_whip = along > texture.body_fraction + 0.05
        self.assertTrue(np.all(texture.width[on_whip] < 6.0))
        middle = np.argmin(np.abs(along - 0.5 * (texture.body_fraction + 1.0)))
        self.assertLess(float(texture.transmission[middle, sc.TEXTURE_SHAPE[1] // 2]), 0.75)

    def test_deterministic_with_a_seed(self):
        first, second, other = self.make(5), self.make(5), self.make(6)
        for key in ("frames", "mask", "overlap", "centerline_xy"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(np.array_equal(first["frames"], other["frames"]))

    def test_dataset_serves_model_inputs(self):
        item = self.make(1)
        with tempfile.TemporaryDirectory() as directory:
            sc.save_item(Path(directory) / "item_000000.npz", item)
            dataset = sc.SyntheticClipDataset(directory, (1, 4, 16), augment=False)
            sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (4, *SHAPE))
        self.assertEqual(tuple(sample["targets"].shape), (8, *SHAPE))
        rows = body_target_rows(item["mask"], {"ap": item["ap"].astype(np.float32), "overlap": item["overlap"],
                                               "head_xy": item["head_xy"], "tail_xy": item["tail_xy"],
                                               "diameter_px": float(item["diameter_px"])})
        np.testing.assert_array_equal(sample["targets"].numpy(), rows)
        # The lag-1 channel is the difference of the +1 and -1 frames.
        from worm_pose_gen.segmenter import INPUT_STD

        expected = (item["frames"][4].astype(np.float32) - item["frames"][2]) / (255.0 * INPUT_STD)
        np.testing.assert_allclose(sample["image"][1].numpy(), expected, atol=1e-5)


class MixedSamplerTest(unittest.TestCase):
    def test_every_real_item_once_and_the_synthetic_share(self):
        sampler = sc.MixedSampler(real=30, synthetic=50, fraction=0.25, seed=0)
        epoch = list(sampler)
        self.assertEqual(len(epoch), len(sampler))
        self.assertEqual(sorted(i for i in epoch if i < 30), list(range(30)))
        self.assertEqual(sum(i >= 30 for i in epoch), 10)
        # Without replacement: five epochs of ten see all fifty.
        seen = {i for i in epoch if i >= 30}
        for _ in range(4):
            seen |= {i for i in sampler if i >= 30}
        self.assertEqual(len(seen), 50)


if __name__ == "__main__":
    unittest.main()
