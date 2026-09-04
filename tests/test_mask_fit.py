from __future__ import annotations

import unittest

import numpy as np
import torch

from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import (
    MaskFitConfig,
    crop_window,
    default_width_template,
    fill_narrow_holes,
    fit_mask,
    hard_iou,
    init_from_centerline,
    init_from_moments,
    init_from_skeleton,
    measure_width_template,
    render_tube_segments,
    standard_initializations,
)


def _synthetic_case(seed: int = 0, height: int = 160, width: int = 220):
    rng = np.random.default_rng(seed)
    shape = np.convolve(rng.normal(0.0, 0.5, 16), [0.25, 0.5, 0.25], "same")
    latent = np.concatenate((shape, [0.4, 150.0], [width / 2, height / 2]))
    curve = decode_centerline(latent)
    template = default_width_template()
    rendered = render_tube_segments(
        torch.as_tensor(curve, dtype=torch.float32)[None],
        torch.as_tensor(12.0 * template, dtype=torch.float32)[None],
        height,
        width,
    )[0]
    return latent, curve, template, (rendered >= 0.5).numpy()


SMALL = MaskFitConfig(
    stage_downsample=(2, 1, 1),
    stage_steps=(60, 60, 40),
    stage_lr_scale=(1.0, 0.3, 0.05),
    crop_padding=16,
    length_bounds_px=(60.0, 300.0),
    width_bounds_px=(4.0, 30.0),
    default_length_px=150.0,
    default_width_px=12.0,
    moment_arc_curvatures=(0.0, 0.01, -0.01),
)


class MaskFitTests(unittest.TestCase):
    def test_default_width_template_is_unit_peak_and_tapered(self) -> None:
        template = default_width_template(100)
        self.assertEqual(template.shape, (100,))
        self.assertAlmostEqual(float(template.max()), 1.0)
        self.assertLess(template[0], 0.1)
        self.assertLess(template[-1], 0.1)
        self.assertTrue(np.all(template > 0))

    def test_segment_renderer_length_gradient_matches_finite_difference(self) -> None:
        # The sample-based renderer carried a consistent shortening bias; the
        # segment renderer must agree with central differences on log length.
        latent, _, template, mask = _synthetic_case(seed=3)
        target = torch.as_tensor(mask, dtype=torch.float32)
        width = torch.as_tensor(12.0 * template, dtype=torch.float32)[None]

        def energy(log_length: torch.Tensor) -> torch.Tensor:
            values = torch.as_tensor(latent, dtype=torch.float32).clone()
            values[17] = torch.exp(log_length)
            from worm_pose_gen.latent import decode_centerline_torch

            curve = decode_centerline_torch(values)[None]
            rendered = render_tube_segments(curve, width, *mask.shape)[0]
            intersection = (rendered * target).sum()
            return 1 - 2 * intersection / (rendered.sum() + target.sum())

        base = torch.tensor(float(np.log(150.0)), requires_grad=True)
        energy(base).backward()
        analytic = float(base.grad)
        delta = 2e-3
        finite = float(
            (energy(base.detach() + delta) - energy(base.detach() - delta)) / (2 * delta)
        )
        self.assertAlmostEqual(analytic, finite, delta=0.02 * max(1.0, abs(finite)) + 5e-3)

    def test_crop_window_is_padded_and_aligned(self) -> None:
        mask = np.zeros((90, 120), dtype=bool)
        mask[40:50, 30:80] = True
        crop = crop_window(mask, padding=10, multiple=4)
        self.assertLessEqual(crop.x0, 20)
        self.assertLessEqual(crop.y0, 30)
        self.assertEqual(crop.width % 4, 0)
        self.assertEqual(crop.height % 4, 0)
        self.assertGreaterEqual(crop.x1, 80)
        self.assertGreaterEqual(crop.y1, 50)

    def test_initializations_cover_reference_skeleton_and_moments(self) -> None:
        _, curve, _, mask = _synthetic_case()
        starts = standard_initializations(mask, reference_centerline_xy=curve, config=SMALL)
        names = [start.name for start in starts]
        self.assertEqual(names[0], "reference")
        self.assertIn("skeleton_longest_path", names)
        self.assertEqual(sum(name.startswith("moments") for name in names), 3)
        for start in starts:
            self.assertEqual(start.latent.shape, (20,))
            self.assertTrue(np.isfinite(start.latent).all())
            self.assertGreater(start.width_px, 0)

    def test_moment_start_uses_principal_axis(self) -> None:
        mask = np.zeros((80, 200), dtype=bool)
        mask[35:45, 20:180] = True
        start = init_from_moments(mask, config=SMALL)
        self.assertIsNotNone(start)
        assert start is not None
        angle = start.latent[16] % np.pi
        self.assertTrue(min(angle, np.pi - angle) < 0.05)
        self.assertAlmostEqual(start.latent[18], 99.5, delta=0.5)
        self.assertAlmostEqual(start.latent[19], 39.5, delta=0.5)

    def test_skeleton_start_returns_none_for_tiny_mask(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[20:22, 20:24] = True
        self.assertIsNone(init_from_skeleton(mask, config=SMALL))

    def test_fit_recovers_synthetic_pose_from_crude_starts(self) -> None:
        latent, curve, template, mask = _synthetic_case(seed=1)
        starts = standard_initializations(mask, config=SMALL)
        result = fit_mask(mask, starts, width_template=template, config=SMALL, device="cpu")
        self.assertGreater(result.records[result.best_index]["final_iou"], 0.9)
        forward = np.linalg.norm(result.centerline_xy - curve, axis=1)
        reverse = np.linalg.norm(result.centerline_xy[::-1] - curve, axis=1)
        self.assertLess(min(np.median(forward), np.median(reverse)), 2.0)
        self.assertAlmostEqual(result.body_length_px, 150.0, delta=8.0)
        self.assertAlmostEqual(result.width_px, 12.0, delta=1.5)
        self.assertEqual(result.points_in_fov, 100)
        self.assertEqual(result.energy_history.shape, (sum(SMALL.stage_steps), len(starts)))

    def test_reference_start_is_not_degraded(self) -> None:
        latent, curve, template, mask = _synthetic_case(seed=2)
        start = init_from_centerline(curve, mask, name="reference", config=SMALL)
        result = fit_mask(mask, [start], width_template=template, config=SMALL, device="cpu")
        record = result.records[0]
        self.assertLessEqual(record["final_soft_dice_energy"], record["initial_soft_dice_energy"] + 1e-4)
        self.assertGreaterEqual(record["final_iou"], record["initial_iou"] - 0.01)

    def test_measure_width_template_normalizes_scale(self) -> None:
        _, curve, template, mask = _synthetic_case(seed=4)
        measured, scales = measure_width_template([mask, mask], [curve, curve])
        self.assertEqual(measured.shape, (100,))
        self.assertEqual(len(scales), 2)
        self.assertAlmostEqual(float(np.median(measured[20:80])), 1.0, delta=0.05)
        self.assertLess(measured[0], 0.4)

    def test_fill_narrow_holes_keeps_wide_cavities_and_border_background(self) -> None:
        mask = np.zeros((60, 80), dtype=bool)
        mask[10:50, 10:70] = True
        mask[20:23, 20:24] = False  # narrow texture hole
        mask[25:45, 40:60] = False  # wide enclosed cavity, like a coil interior
        mask[30:50, 10:14] = False  # notch open to the border side
        filled, added = fill_narrow_holes(mask, radius=4, device="cpu")
        self.assertEqual(added, 12)
        self.assertTrue(filled[20:23, 20:24].all())
        self.assertFalse(filled[25:45, 40:60].any())
        self.assertFalse(filled[30:50, 10:14].any())
        untouched, none_added = fill_narrow_holes(mask, radius=0, device="cpu")
        self.assertEqual(none_added, 0)
        self.assertTrue(np.array_equal(untouched, mask))

    def test_hard_iou_handles_empty_union(self) -> None:
        empty = np.zeros((4, 4), dtype=bool)
        self.assertEqual(hard_iou(empty, empty), 0.0)
        full = np.ones((4, 4), dtype=bool)
        self.assertEqual(hard_iou(full, full), 1.0)

    def test_fit_rejects_empty_mask_and_misaligned_schedule(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            fit_mask(np.zeros((20, 20), dtype=bool), [], config=SMALL, device="cpu")
        _, _, _, mask = _synthetic_case()
        start = init_from_moments(mask, config=SMALL)
        assert start is not None
        with self.assertRaisesRegex(ValueError, "align"):
            fit_mask(
                mask,
                [start],
                config=MaskFitConfig(stage_downsample=(2, 1), stage_steps=(1,), stage_lr_scale=(1.0, 1.0)),
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
