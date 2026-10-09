from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from worm_pose_gen.batch_fit import BatchFitConfig, BodyFieldEvidence, batch_windows, field_energies, fit_masks, plan_groups
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import (
    CropWindow,
    Initialization,
    MaskFitConfig,
    crop_window,
    default_width_template,
    fit_mask,
    hard_iou,
    orientation_pair,
    render_tube_segments,
    standard_initializations,
)

from tests.slow import slow


SMALL = BatchFitConfig(
    stage_downsample=(2, 1),
    stage_steps=(60, 60),
    stage_lr_scale=(1.0, 0.3),
    stage_point_stride=(2, 1),
    crop_padding=16,
    crop_multiple=8,
    compile_renderer=False,
    length_bounds_px=(60.0, 300.0),
    width_bounds_px=(4.0, 30.0),
    default_length_px=150.0,
    default_width_px=12.0,
    moment_arc_curvatures=(0.0, 0.01, -0.01),
)

REFERENCE = MaskFitConfig(
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


def _render(latent: np.ndarray, height: int, width: int, body_width: float = 12.0) -> np.ndarray:
    curve = decode_centerline(latent)
    template = default_width_template()
    rendered = render_tube_segments(
        torch.as_tensor(curve, dtype=torch.float32)[None],
        torch.as_tensor(body_width * template, dtype=torch.float32)[None],
        height,
        width,
    )[0]
    return (rendered >= 0.5).numpy()


def _latent(seed: int, centroid: tuple[float, float], length: float = 150.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = np.convolve(rng.normal(0.0, 0.5, 16), [0.25, 0.5, 0.25], "same")
    return np.concatenate((shape, [0.4, length], centroid))


def body_evidence(curve: np.ndarray, mask: np.ndarray) -> BodyFieldEvidence:
    """What a perfect body-field network says about a head-first ``curve``: each mask pixel's arc position, and the ends."""

    ys, xs = np.nonzero(mask)
    pixels = np.stack((xs, ys), 1).astype(np.float64)
    nearest = np.argmin(np.linalg.norm(pixels[:, None, :] - curve[None], axis=-1), axis=1)
    ap = np.full(mask.shape, np.nan, dtype=np.float32)
    ap[ys, xs] = nearest / (len(curve) - 1)
    return BodyFieldEvidence(ap, np.asarray(curve[0], dtype=np.float64), np.asarray(curve[-1], dtype=np.float64))


class BatchFitTests(unittest.TestCase):
    def test_batch_windows_cover_crops_inside_camera(self) -> None:
        crops = [CropWindow(10, 50, 20, 44, 100, 120), CropWindow(60, 116, 4, 100, 100, 120)]
        windows, height, width = batch_windows(crops)
        self.assertEqual((height, width), (96, 56))
        for crop, window in zip(crops, windows, strict=True):
            self.assertLessEqual(window.x0, crop.x0)
            self.assertGreaterEqual(window.x1, crop.x1)
            self.assertLessEqual(window.y0, crop.y0)
            self.assertGreaterEqual(window.y1, crop.y1)
            self.assertGreaterEqual(window.x0, 0)
            self.assertLessEqual(window.x1, 120)
            self.assertGreaterEqual(window.y0, 0)
            self.assertLessEqual(window.y1, 100)

    def test_batch_windows_overhang_when_larger_than_camera(self) -> None:
        crops = [CropWindow(0, 120, 0, 100, 100, 120), CropWindow(0, 8, 0, 8, 100, 120)]
        windows, height, width = batch_windows(crops)
        self.assertEqual((height, width), (100, 120))
        self.assertEqual((windows[1].x0, windows[1].y0), (0, 0))
        crops = [CropWindow(0, 128, 0, 104, 100, 120)]
        windows, _, _ = batch_windows(crops)
        self.assertLess(windows[0].x0, 0)
        self.assertGreater(windows[0].x1, 120)

    def test_plan_groups_respects_budget_and_covers_every_frame(self) -> None:
        crops = [CropWindow(0, 64, 0, 64, 200, 200)] * 5 + [CropWindow(0, 128, 0, 128, 200, 200)] * 2
        config = BatchFitConfig(row_pixel_budget=64 * 64 * 12, max_rows=1000)
        groups = plan_groups(crops, [4] * len(crops), config)
        seen = sorted(i for g in groups for i in g)
        self.assertEqual(seen, list(range(len(crops))))
        for group in groups:
            rows = 4 * len(group)
            height = max(crops[i].height for i in group)
            width = max(crops[i].width for i in group)
            if len(group) > 1:
                self.assertLessEqual(rows * height * width, config.row_pixel_budget)
        self.assertGreater(len(groups), 1)

    @slow
    def test_fits_several_frames_including_a_clipped_body(self) -> None:
        height, width = 160, 220
        truths = [
            _latent(0, (width / 2, height / 2)),
            _latent(1, (width / 2 - 20, height / 2 + 10)),
            _latent(2, (width - 25.0, height / 2)),  # runs off the right edge
        ]
        masks = [_render(t, height, width) for t in truths]
        self.assertTrue(masks[2][:, -1].any(), "third body should touch the camera edge")
        starts = [standard_initializations(m, config=SMALL) for m in masks]
        results = fit_masks(masks, starts, config=SMALL, device="cpu")
        self.assertEqual(len(results), 3)
        for truth, mask, result in zip(truths, masks, results, strict=True):
            crop = result.crop
            self.assertGreaterEqual(crop.x0, 0)
            self.assertLessEqual(crop.x1, width)
            self.assertEqual(result.rendered_hard_mask.shape, (crop.height, crop.width))
            iou = hard_iou(result.rendered_hard_mask, mask[crop.y0 : crop.y1, crop.x0 : crop.x1])
            self.assertGreater(iou, 0.85)
            self.assertEqual(result.energy_history.shape[1], len(result.initializations))
            self.assertTrue(np.isfinite(result.records[result.best_index]["final_iou"]))
        self.assertEqual(results[0].points_in_fov, 100)
        self.assertLess(results[2].points_in_fov, 100)
        self.assertEqual(results[2].crop.x1, width)
        # The fitted length of the clipped body is not shorter than the visible part.
        visible = np.sum(masks[2].any(axis=0))
        self.assertGreater(results[2].body_length_px, 0.8 * visible)

    @slow
    def test_matches_single_frame_fitter(self) -> None:
        height, width = 160, 220
        truth = _latent(5, (width / 2, height / 2))
        mask = _render(truth, height, width)
        starts = standard_initializations(mask, config=SMALL)
        batched = fit_masks([mask], [starts], config=SMALL, device="cpu")[0]
        single = fit_mask(mask, starts, config=REFERENCE, device="cpu")
        crop = batched.crop
        batched_iou = hard_iou(batched.rendered_hard_mask, mask[crop.y0 : crop.y1, crop.x0 : crop.x1])
        single_iou = single.records[single.best_index]["final_iou"]
        self.assertGreater(batched_iou, single_iou - 0.03)
        truth_curve = decode_centerline(truth)
        fitted = batched.centerline_xy
        forward = np.linalg.norm(fitted - truth_curve, axis=1).mean()
        backward = np.linalg.norm(fitted[::-1] - truth_curve, axis=1).mean()
        self.assertLess(min(forward, backward), 2.0)

    @slow
    def test_frames_with_different_crop_sizes_keep_input_order(self) -> None:
        height, width = 160, 220
        small = _render(_latent(3, (width / 2, height / 2), length=80.0), height, width, body_width=8.0)
        large = _render(_latent(4, (width / 2, height / 2), length=200.0), height, width)
        masks = [large, small, large]
        starts = [standard_initializations(m, config=SMALL) for m in masks]
        config = BatchFitConfig(**{**SMALL.__dict__, "row_pixel_budget": 1, "max_rows": 1000})
        results = fit_masks(masks, starts, config=config, device="cpu")
        self.assertEqual(len(results), 3)
        self.assertLess(results[1].body_length_px, results[0].body_length_px)
        self.assertLess(results[1].body_length_px, results[2].body_length_px)
        self.assertEqual(crop_window(small, SMALL.crop_padding, 8).height, results[1].crop.height)

    def test_temporal_prior_pulls_the_fit_toward_the_reference(self) -> None:
        height, width = 120, 200
        latent = np.concatenate((np.zeros(16), [0.0, 140.0], [width / 2, height / 2]))
        curve = decode_centerline(latent)
        template = default_width_template()
        rendered = render_tube_segments(
            torch.as_tensor(curve, dtype=torch.float32)[None], torch.as_tensor(12.0 * template, dtype=torch.float32)[None], height, width
        )[0]
        mask = (rendered >= 0.5).numpy()
        reference = curve + (0.0, 6.0)
        start = [Initialization("s", latent, 12.0)]
        free = fit_masks([mask], [start], config=SMALL, device="cpu", references=[reference])[0]
        # Measured pull toward a reference 6 px away: 1.2 px at weight 0.05, 3.9 at 0.2, 5.6 at 1.0.
        pulled_config = replace(SMALL, temporal_prior_weight=0.2, temporal_prior_sigma_px=4.0)
        pulled = fit_masks([mask], [start], config=pulled_config, device="cpu", references=[reference])[0]
        offset_free = float(np.mean(free.centerline_xy[:, 1] - curve[:, 1]))
        offset_pulled = float(np.mean(pulled.centerline_xy[:, 1] - curve[:, 1]))
        self.assertLess(abs(offset_free), 1.0)
        self.assertGreater(offset_pulled, 2.5)
        self.assertGreater(pulled.records[0]["final_energy"], pulled.records[0]["final_soft_dice_energy"])
        self.assertGreater(free.records[0]["final_coverage"], 0.9)
        self.assertLessEqual(free.records[0]["final_coverage"], 1.0)
        with self.assertRaisesRegex(ValueError, "references"):
            fit_masks([mask], [start], config=SMALL, device="cpu", references=[])

    @slow
    def test_body_field_evidence_is_recorded_and_scores_any_pose_as_the_fit_does(self) -> None:
        height, width = 160, 220
        latent = _latent(6, (width / 2, height / 2))
        mask = _render(latent, height, width)
        curve = decode_centerline(latent)
        evidence = body_evidence(curve, mask)
        config = replace(SMALL, field_ap_weight=0.01, field_end_weight=0.01)
        starts = list(orientation_pair(Initialization("s", latent, 12.0), config=config))
        result = fit_masks([mask], [starts], config=config, device="cpu", fields=[evidence])[0]
        best = result.records[result.best_index]
        # The evidence picks the head-first orientation and its energy is part of the total.
        self.assertLess(float(np.linalg.norm(result.centerline_xy[0] - curve[0])), 5.0)
        self.assertGreater(best["final_field_energy"], 0.0)
        self.assertGreaterEqual(best["final_energy"], best["final_soft_dice_energy"] + best["final_field_energy"])
        # Evaluated outside the fit, the same pose costs the same; its mirror costs far more.
        scored, mirrored = field_energies([result.centerline_xy, result.centerline_xy[::-1]], [evidence, evidence], config)
        self.assertAlmostEqual(scored, best["final_field_energy"], delta=1e-4 + 1e-3 * best["final_field_energy"])
        self.assertGreater(mirrored, best["final_field_energy"] + 1.0)
        self.assertEqual(field_energies([result.centerline_xy], [None], config).tolist(), [0.0])
        self.assertEqual(field_energies([result.centerline_xy], [evidence], SMALL).tolist(), [0.0])
        # Without evidence (or without weights) the record says 0.
        for record in fit_masks([mask], [starts], config=config, device="cpu", fields=[None])[0].records:
            self.assertEqual(record["final_field_energy"], 0.0)
        for record in fit_masks([mask], [starts], config=SMALL, device="cpu", fields=[evidence])[0].records:
            self.assertEqual(record["final_field_energy"], 0.0)

    def test_a_fixed_length_lays_every_start_and_keeps_it(self) -> None:
        height, width = 160, 220
        latent = _latent(6, (width / 2, height / 2))
        mask = _render(latent, height, width)
        config = replace(SMALL, length_bounds_px=None, length_prior_px=150.0, length_fixed=True)
        # A start 30 px short is laid on along the mask to the prior's length before the fit, and keeps it.
        short = Initialization("short", np.concatenate((latent[:17], [120.0], latent[18:])), 12.0)
        result = fit_masks([mask], [[short]], config=config, device="cpu")[0]
        self.assertAlmostEqual(result.body_length_px, 150.0, delta=0.5)
        self.assertAlmostEqual(float(result.latent[17]), 150.0, places=3)
        self.assertGreater(result.records[0]["final_iou"], 0.85)
        with self.assertRaisesRegex(ValueError, "length_prior_px"):
            fit_masks([mask], [[short]], config=replace(SMALL, length_fixed=True), device="cpu")

    def test_only_the_head_scores_the_ends(self) -> None:
        height, width = 160, 220
        latent = _latent(6, (width / 2, height / 2))
        mask = _render(latent, height, width)
        curve = decode_centerline(latent)
        evidence = body_evidence(curve, mask)
        config = replace(SMALL, field_end_weight=0.01)
        # The tail does not score a fit (people trace it inconsistently); the head does, in units of 10 px.
        moved = replace(evidence, tail_xy=evidence.tail_xy + 40.0)
        self.assertEqual(field_energies([curve], [moved], config).tolist(), field_energies([curve], [evidence], config).tolist())
        shifted = replace(evidence, head_xy=evidence.head_xy + [30.0, 40.0])
        self.assertAlmostEqual(float(field_energies([curve], [shifted], config)[0]), 0.01 * 25.0, places=4)

    def test_the_ap_term_scores_the_front_half(self) -> None:
        height, width = 160, 220
        latent = _latent(6, (width / 2, height / 2))
        mask = _render(latent, height, width)
        curve = decode_centerline(latent)
        evidence = body_evidence(curve, mask)
        config = replace(SMALL, field_ap_weight=0.01)
        plain = float(field_energies([curve], [evidence], config)[0])
        # A label whose tail stops short stretches the back half's A-P; the term does not see it.
        stretched = replace(evidence, ap=np.where(evidence.ap > 0.55, np.minimum(evidence.ap * 1.2, 1.0), evidence.ap).astype(np.float32))
        self.assertAlmostEqual(float(field_energies([curve], [stretched], config)[0]), plain, places=6)
        shifted = replace(evidence, ap=np.clip(evidence.ap + 0.1, 0.0, 1.0).astype(np.float32))
        self.assertGreater(float(field_energies([curve], [shifted], config)[0]), plain + 0.02)

    def test_rejects_misaligned_inputs(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 5:25] = True
        with self.assertRaises(ValueError):
            fit_masks([mask], [], config=SMALL, device="cpu")
        with self.assertRaises(ValueError):
            fit_masks([mask], [[]], config=SMALL, device="cpu")
        with self.assertRaises(ValueError):
            fit_masks([np.zeros((32, 32), dtype=bool)], [standard_initializations(mask, config=SMALL)], config=SMALL, device="cpu")


if __name__ == "__main__":
    unittest.main()
