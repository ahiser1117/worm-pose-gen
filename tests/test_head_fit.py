"""Head constraints must hold even when overlap favors another body branch."""

from dataclasses import replace
import unittest

import numpy as np
import torch

from worm_pose_gen.batch_fit import fit_masks
from worm_pose_gen.head_fit import HeadConstraint, HeadPriors
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import Initialization, reverse_initialization
from tests.test_batch_fit import SMALL, _render


class HeadFitTests(unittest.TestCase):
    def test_projection_respects_both_camera_and_motion_at_edges(self):
        previous = np.asarray([[-2, 40], [50, 50], [0, 0]], dtype=float)
        constraints = [HeadConstraint(previous_xy=p, max_step_px=5) for p in previous]
        priors = HeadPriors(constraints, torch.full((3, 2), 100.))
        projected = priors.project(torch.tensor([[80., -10.], [-50., 300.], [10., 10.]]))
        self.assertTrue(bool(((projected >= 0) & (projected <= 99)).all()))
        self.assertTrue(np.all(np.linalg.norm(projected.numpy() - previous, axis=1) <= 5.00001))
        with self.assertRaisesRegex(ValueError, "too far outside"):
            HeadPriors([HeadConstraint(previous_xy=np.array([-20, 0]), max_step_px=5)], torch.full((1, 2), 100.))

    def test_tracking_selects_head_orientation_and_keeps_latent_consistent(self):
        latent = np.concatenate((np.zeros(16), [0., 110., 100., 60.]))
        start = Initialization("correct", latent, 12.)
        reversed_start = reverse_initialization(start, config=SMALL)
        mask = _render(latent, 120, 200)
        head = decode_centerline(latent)[0]
        result = fit_masks([mask], [[reversed_start, start]], config=SMALL, device="cpu",
                           head_constraints=[HeadConstraint(tracking_xy=head, sigma_px=6)])[0]
        self.assertLess(np.linalg.norm(result.centerline_xy[0] - head), 3.)
        self.assertEqual(result.initializations[result.best_index].name, "correct")
        np.testing.assert_allclose(decode_centerline(result.latent), result.centerline_xy, atol=1e-4)

    def test_motion_limit_overrides_distracting_tracking_point(self):
        latent = np.concatenate((np.zeros(16), [0., 110., 100., 60.]))
        head = decode_centerline(latent)[0]
        shifted = latent.copy()
        shifted[-1] += 25
        mask = _render(shifted, 120, 200)
        start = Initialization("previous", latent, 12.)
        config = replace(SMALL, stage_steps=(25, 25))
        free = fit_masks([mask], [[start]], config=config, device="cpu")[0]
        result = fit_masks([mask], [[start]], config=config, device="cpu",
                           head_constraints=[HeadConstraint(tracking_xy=head + [0, 25], previous_xy=head,
                                                            tracking_weight=3., max_step_px=3., sigma_px=4.)])[0]
        self.assertGreater(np.linalg.norm(free.centerline_xy[0] - head), 6.)
        self.assertLessEqual(np.linalg.norm(result.centerline_xy[0] - head), 3.0001)
        np.testing.assert_allclose(decode_centerline(result.latent), result.centerline_xy, atol=1e-4)

    def test_constraints_follow_masks_through_batch_groups(self):
        latent = np.concatenate((np.zeros(16), [0., 110., 100., 60.]))
        mask = _render(latent, 120, 200)
        start = Initialization("s", latent, 12.)
        config = replace(SMALL, stage_steps=(2, 2), row_pixel_budget=1)
        heads = [np.array([45., 40.]), np.array([45., 80.])]
        results = fit_masks([mask, mask], [[start], [start]], config=config, device="cpu",
                            head_constraints=[HeadConstraint(previous_xy=p, max_step_px=0.) for p in heads])
        for result, head in zip(results, heads, strict=True):
            np.testing.assert_allclose(result.centerline_xy[0], head, atol=1e-4)
        with self.assertRaisesRegex(ValueError, "head_constraints must align"):
            fit_masks([mask], [[start]], config=config, head_constraints=[])


if __name__ == "__main__":
    unittest.main()
