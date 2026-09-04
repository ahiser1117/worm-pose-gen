from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from worm_pose_gen.latent import cubic_bspline_basis, decode_centerline
from worm_pose_gen.mask_fit import (
    CropWindow,
    Initialization,
    MaskFitConfig,
    MaskFitResult,
    _MaskFitState,
    default_width_template,
    fit_mask,
    hard_iou,
    orient_tail_last,
    render_tube_segments,
    reverse_initialization,
    reverse_result,
    standard_initializations,
    taper_asymmetry,
)


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
# Thin, long taper at the end of the body (the tail), blunt start (the head).
TAIL_LAST = np.array([0.25, 0.2, 0.1, -0.05, -0.35, -0.9])


def _asymmetric_case(seed: int = 1, height: int = 160, width: int = 220):
    rng = np.random.default_rng(seed)
    shape = np.convolve(rng.normal(0.0, 0.4, 16), [0.25, 0.5, 0.25], "same")
    latent = np.concatenate((shape, [0.3, 150.0], [width / 2, height / 2]))
    curve = decode_centerline(latent)
    template = default_width_template()
    correction = cubic_bspline_basis(100, len(TAIL_LAST)) @ TAIL_LAST
    profile = 12.0 * template * np.exp(correction - correction.mean())
    rendered = render_tube_segments(
        torch.as_tensor(curve, dtype=torch.float32)[None],
        torch.as_tensor(profile, dtype=torch.float32)[None],
        height,
        width,
    )[0]
    return latent, curve, template, profile, (rendered >= 0.5).numpy()


def _fake_result(curve: np.ndarray, profile: np.ndarray, shape: np.ndarray) -> MaskFitResult:
    return MaskFitResult(
        best_index=0,
        initializations=[],
        records=[{"name": "x", "final_iou": 1.0}],
        latent=np.zeros(20),
        width_px=12.0,
        width_profile=profile,
        centerline_xy=curve,
        crop=CropWindow(0, 8, 0, 8, 8, 8),
        rendered_hard_mask=np.zeros((8, 8), dtype=bool),
        energy_history=np.zeros((0, 1)),
        points_in_fov=100,
        body_length_px=150.0,
        width_shape=shape,
    )


class WidthModelTests(unittest.TestCase):
    def test_reverse_initialization_round_trips(self) -> None:
        latent, curve, _, _, _ = _asymmetric_case()
        start = Initialization("s", latent, 12.0, TAIL_LAST)
        reversed_start = reverse_initialization(start)
        np.testing.assert_allclose(decode_centerline(reversed_start.latent), curve[::-1], atol=1e-9)
        np.testing.assert_array_equal(reversed_start.width_shape, TAIL_LAST[::-1])
        again = reverse_initialization(reversed_start)
        np.testing.assert_allclose(decode_centerline(again.latent), curve, atol=1e-9)
        np.testing.assert_array_equal(again.width_shape, TAIL_LAST)

    def test_width_correction_is_mean_centered_and_mirror_symmetric(self) -> None:
        latent, _, template, _, _ = _asymmetric_case()
        start = Initialization("s", latent, 12.0, TAIL_LAST)
        state = _MaskFitState([start, reverse_initialization(start)], MaskFitConfig(), torch.device("cpu"))
        correction = state.log_width_correction().detach().numpy()
        np.testing.assert_allclose(correction.mean(1), 0.0, atol=1e-6)
        np.testing.assert_allclose(correction[1], correction[0][::-1], atol=1e-6)
        diameter = state.diameter(torch.as_tensor(template, dtype=torch.float32)).detach().numpy()
        np.testing.assert_allclose(diameter[0], 12.0 * template * np.exp(correction[0]), rtol=1e-5)
        self.assertLess(taper_asymmetry(diameter[0]), -0.3)
        self.assertGreater(taper_asymmetry(diameter[1]), 0.3)

    def test_symmetric_configuration_has_no_correction(self) -> None:
        latent, _, template, _, mask = _asymmetric_case()
        config = replace(SMALL, width_coefficients=0, stage_steps=(5, 5, 5))
        start = Initialization("s", latent, 12.0)
        state = _MaskFitState([start], config, torch.device("cpu"))
        self.assertEqual(tuple(state.width_shape.shape), (1, 0))
        self.assertEqual(len(state.optimizer().param_groups), 5)
        result = fit_mask(mask, [start], width_template=template, config=config, device="cpu")
        self.assertEqual(result.width_shape.size, 0)
        np.testing.assert_allclose(result.width_profile / result.width_px, template, rtol=1e-5)
        with self.assertRaises(ValueError):
            _MaskFitState([start], replace(SMALL, width_coefficients=3), torch.device("cpu"))
        with self.assertRaises(ValueError):
            _MaskFitState([Initialization("s", latent, 12.0, np.zeros(4))], SMALL, torch.device("cpu"))

    def test_orient_tail_last_reverses_thin_first_fits(self) -> None:
        _, curve, _, profile, _ = _asymmetric_case()
        kept, flipped = orient_tail_last(_fake_result(curve, profile, TAIL_LAST))
        self.assertFalse(flipped)
        np.testing.assert_array_equal(kept.centerline_xy, curve)
        oriented, flipped = orient_tail_last(_fake_result(curve[::-1].copy(), profile[::-1].copy(), TAIL_LAST[::-1]))
        self.assertTrue(flipped)
        np.testing.assert_array_equal(oriented.centerline_xy, curve)
        np.testing.assert_array_equal(oriented.width_profile, profile)
        np.testing.assert_array_equal(oriented.width_shape, TAIL_LAST)
        np.testing.assert_allclose(decode_centerline(oriented.latent), curve, atol=1e-6)
        twice = reverse_result(reverse_result(oriented))
        np.testing.assert_array_equal(twice.centerline_xy, curve)

    def test_asymmetric_fit_beats_symmetric_and_labels_the_tail(self) -> None:
        _, curve, template, profile, mask = _asymmetric_case()
        starts = standard_initializations(mask, config=SMALL)
        asymmetric = fit_mask(mask, starts, width_template=template, config=SMALL, device="cpu")
        symmetric = fit_mask(
            mask, starts, width_template=template, config=replace(SMALL, width_coefficients=0), device="cpu"
        )
        target = mask[asymmetric.crop.y0 : asymmetric.crop.y1, asymmetric.crop.x0 : asymmetric.crop.x1]
        iou_asymmetric = hard_iou(asymmetric.rendered_hard_mask, target)
        target = mask[symmetric.crop.y0 : symmetric.crop.y1, symmetric.crop.x0 : symmetric.crop.x1]
        iou_symmetric = hard_iou(symmetric.rendered_hard_mask, target)
        self.assertGreater(iou_asymmetric, iou_symmetric + 0.02)
        self.assertGreater(iou_asymmetric, 0.9)
        oriented, _ = orient_tail_last(asymmetric, config=SMALL)
        self.assertLess(taper_asymmetry(oriented.width_profile), -0.2)
        # The thin end of the fit sits at the true tail, not the head.
        self.assertLess(np.linalg.norm(oriented.centerline_xy[-1] - curve[-1]), 12.0)
        self.assertGreater(np.linalg.norm(oriented.centerline_xy[-1] - curve[0]), 40.0)


if __name__ == "__main__":
    unittest.main()
