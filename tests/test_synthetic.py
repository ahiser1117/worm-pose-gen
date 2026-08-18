import unittest

import torch

from worm_pose_gen.geometry import in_fov_mask
from worm_pose_gen.synthetic import (
    PARAMETER_PROFILES,
    SyntheticConfig,
    anatomical_crop_transform,
    generate_synthetic_pose,
    moving_crop_sequence,
    original_to_render,
    render_to_original,
)


class SyntheticTests(unittest.TestCase):
    def test_seeded_generation_contract(self) -> None:
        config = SyntheticConfig()
        for seed in range(20260818, 20260838):
            pose = generate_synthetic_pose(seed, config)
            centerline = pose["centerline_xy"]
            self.assertEqual(centerline.shape, (100, 2))
            self.assertTrue(bool(torch.isfinite(centerline).all()))
            self.assertTrue(300 <= float(pose["body_length"]) <= 600)
            self.assertTrue(0.25 <= float(pose["bend_amplitude"]) <= 0.55)
            self.assertTrue(bool(pose["in_fov_mask"].all()))
            torch.testing.assert_close(
                render_to_original(original_to_render(centerline, config), config),
                centerline,
                atol=1e-10,
                rtol=0,
            )
            self.assertTrue(
                torch.equal(
                    in_fov_mask(centerline, 732, 968),
                    in_fov_mask(original_to_render(centerline, config), 192, 256),
                )
            )
        a = generate_synthetic_pose(20260818)
        b = generate_synthetic_pose(20260818)
        torch.testing.assert_close(a["centerline_xy"], b["centerline_xy"])

    def test_held_out_profile_is_disjoint(self) -> None:
        development = PARAMETER_PROFILES["development"]
        held_out = PARAMETER_PROFILES["held_out"]
        self.assertLess(development.bend_amplitude_range[1], held_out.bend_amplitude_range[0])
        band_counts = [0, 0]
        for seed in range(20270000, 20270128):
            pose = generate_synthetic_pose(seed, profile="held_out")
            length = float(pose["body_length"])
            self.assertTrue(250 <= length <= 299 or 601 <= length <= 700)
            self.assertTrue(0.65 <= float(pose["bend_amplitude"]) <= 0.90)
            band_counts[int(pose["length_band_index"])] += 1
        self.assertGreater(min(band_counts), 0)

    def test_exact_head_and_tail_censoring(self) -> None:
        config = SyntheticConfig()
        for seed in (20260818, 20260991, 20270000):
            profile = "held_out" if seed >= 20270000 else "development"
            source = generate_synthetic_pose(seed, config, profile=profile)["centerline_xy"]
            for fraction in (0.05, 0.10, 0.20, 0.30, 0.40):
                for end in ("head", "tail"):
                    transform, camera, support = anatomical_crop_transform(source, fraction, end)
                    self.assertEqual(int((~support).sum()), round(100 * fraction))
                    self.assertTrue(torch.equal(support, in_fov_mask(camera, 732, 968)))
                    self.assertTrue(
                        torch.equal(support, in_fov_mask(original_to_render(camera), 192, 256))
                    )
                    error = (transform.to_source(camera) - source).abs().max()
                    self.assertLessEqual(float(error), 1e-10)

    def test_temporally_coherent_moving_crop(self) -> None:
        source = generate_synthetic_pose(20260818)["centerline_xy"]
        for hidden_end in ("head", "tail"):
            sequence = moving_crop_sequence(source, hidden_end=hidden_end, num_frames=21)
            cameras = sequence["centerline_camera_xy"]
            support = sequence["support_mask"]
            counts = sequence["hidden_count"]
            self.assertEqual((int(counts[0]), int(counts[-1])), (5, 40))
            self.assertTrue(bool(torch.all(counts[1:] >= counts[:-1])))
            self.assertTrue(torch.equal(support, in_fov_mask(cameras, 732, 968)))
            offsets = torch.stack([item.offset for item in sequence["transforms"]])
            torch.testing.assert_close(
                offsets[2:] - 2 * offsets[1:-1] + offsets[:-2],
                torch.zeros_like(offsets[2:]),
                atol=5e-13,
                rtol=0,
            )
            for transform, camera in zip(sequence["transforms"], cameras, strict=True):
                torch.testing.assert_close(transform.to_source(camera), source, atol=1e-10, rtol=0)

    def test_invalid_length_contract(self) -> None:
        with self.assertRaises(ValueError):
            generate_synthetic_pose(1, SyntheticConfig(min_length=200))
        with self.assertRaises(ValueError):
            generate_synthetic_pose(1, profile="test")


if __name__ == "__main__":
    unittest.main()
