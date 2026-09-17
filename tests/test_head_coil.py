"""A head constraint must survive an ambiguous self-crossing body mask."""
from dataclasses import replace
import unittest

import numpy as np

from worm_pose_gen.batch_fit import fit_masks
from worm_pose_gen.head_fit import HeadConstraint
from worm_pose_gen.latent import decode_centerline, encode_centerline
from worm_pose_gen.mask_fit import Initialization
from tests.test_batch_fit import SMALL, _render


def _proper_crossings(curve):
    """Count transverse crossings between non-neighboring polyline segments."""
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    count = 0
    for i in range(len(curve) - 1):
        a, b = curve[i:i + 2]
        for j in range(i + 2, len(curve) - 1):
            c, d = curve[j:j + 2]
            count += (cross(b - a, c - a) * cross(b - a, d - a) < 0
                      and cross(d - c, a - c) * cross(d - c, b - c) < 0)
    return count


class HeadCoilTests(unittest.TestCase):
    def test_crossing_branch_cue_cannot_move_head_beyond_previous_head_disk(self):
        # Figure eight with distinct endpoints and a transverse interior crossing.
        t = np.linspace(-.6, 2 * np.pi - .2, 100)
        curve = np.column_stack((64 + 27 * np.sin(t), 64 + 22 * np.sin(2 * t)))
        latent = encode_centerline(curve, 16)
        previous = decode_centerline(latent)
        self.assertGreater(_proper_crossings(previous), 0, 'fixture must really self-intersect')
        head = previous[0]
        # The false observation lies on a remote part of this SAME coiled body.
        distractor = previous[np.argmax(np.linalg.norm(previous - head, axis=1))]
        self.assertGreater(np.linalg.norm(distractor - head), 30.)
        mask = _render(latent, 128, 128, body_width=8.)
        start = Initialization('previous_coil', latent, 8.)
        config = replace(SMALL, stage_steps=(8, 8), default_width_px=8.)
        common = dict(tracking_xy=distractor, previous_xy=head, tracking_weight=3.,
                      previous_weight=.5, sigma_px=4.)
        free = fit_masks([mask], [[start]], config=config, device='cpu',
                         references=[previous],
                         head_constraints=[HeadConstraint(**common)])[0]
        constrained = fit_masks([mask], [[start]], config=config, device='cpu',
                                references=[previous],
                                head_constraints=[HeadConstraint(**common, max_step_px=2.)])[0]
        self.assertGreater(np.linalg.norm(free.centerline_xy[0] - head), 5.)
        self.assertLessEqual(np.linalg.norm(constrained.centerline_xy[0] - head), 2.0001)
        self.assertGreater(np.linalg.norm(constrained.centerline_xy[-1] - head), 4.)
        # The returned latent must retain the same constrained head/orientation.
        np.testing.assert_allclose(decode_centerline(constrained.latent),
                                   constrained.centerline_xy, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
