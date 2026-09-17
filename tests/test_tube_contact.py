"""Self-contact must combine body coverage without carving out the wider tube."""

import unittest

import torch

from worm_pose_gen.anchors import render_centerline_mask
from worm_pose_gen.mask_fit import render_tube_segments
from worm_pose_gen.renderer import render_worm


def _sample_mask(points, diameters):
    return render_worm(points, diameters, 64, 64)["tube_mask"]


def _segment_mask(points, diameters):
    return render_tube_segments(points, diameters, 64, 64)


def _numpy_mask(points, diameters):
    return torch.stack([
        torch.from_numpy(render_centerline_mask(p.numpy(), w.numpy(), (64, 64))).float()
        for p, w in zip(points, diameters, strict=True)
    ])


class TubeContactTests(unittest.TestCase):
    def test_straight_tubes_match_capsule_coverage(self) -> None:
        x = torch.arange(12.0, 53.0)
        points = torch.stack((x, torch.full_like(x, 32.0)), -1)[None].repeat(2, 1, 1)
        diameters = torch.tensor([[6.0], [20.0]]).expand(2, len(x))
        y_grid, x_grid = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
        dx = (12.0 - x_grid).clamp_min(0) + (x_grid - 52.0).clamp_min(0)
        distance = torch.sqrt(dx.square() + (y_grid - 32.0).square() + torch.finfo(torch.float32).eps)
        expected = torch.sigmoid((diameters[:, :1, None] / 2 - distance) / 0.8)
        for render in (_sample_mask, _segment_mask):
            with self.subTest(renderer=render.__name__):
                torch.testing.assert_close(render(points, diameters), expected)

    def test_nose_contact_preserves_midbody_coverage(self) -> None:
        for render in (_sample_mask, _segment_mask, _numpy_mask):
            # A narrow nose approaches and overlaps a horizontal, wider body.
            # Contact, including a hidden nose, must not subtract body pixels.
            for nose_y in (29.5, 30.0, 32.0, 40.0):
                with self.subTest(renderer=render.__name__, nose_y=nose_y):
                    points = torch.tensor([[
                        [32.0, nose_y], [32.0, 15.0], [8.0, 15.0],
                        [8.0, 40.0], [32.0, 40.0], [56.0, 40.0],
                    ]])
                    diameters = torch.tensor([[1.0, 4.0, 8.0, 20.0, 20.0, 20.0]])
                    body = render(points[:, 3:], diameters[:, 3:])
                    touching = render(points, diameters)
                    self.assertTrue(bool((touching >= body - 1e-6).all()))
                    # This pixel is two pixels inside the thick body's edge.
                    self.assertGreater(float(touching[0, 32, 32]), 0.9)
                    torch.testing.assert_close(
                        touching, render(points.flip(1), diameters.flip(1))
                    )

    def test_covered_body_pixel_does_not_push_nose_away(self) -> None:
        for render in (_sample_mask, _segment_mask):
            with self.subTest(renderer=render.__name__):
                points = torch.tensor([[
                    [32.0, 30.0], [32.0, 15.0], [8.0, 15.0],
                    [8.0, 40.0], [32.0, 40.0], [56.0, 40.0],
                ]], requires_grad=True)
                diameters = torch.tensor(
                    [[1.0, 4.0, 8.0, 20.0, 20.0, 20.0]], requires_grad=True
                )
                loss = (1.0 - render(points, diameters)[0, 32, 32]).square()
                loss.backward()
                self.assertTrue(bool(torch.isfinite(points.grad).all()))
                self.assertTrue(bool(torch.isfinite(diameters.grad).all()))
                self.assertEqual(float(points.grad[:, :2].abs().sum()), 0.0)
                self.assertEqual(float(diameters.grad[:, :2].abs().sum()), 0.0)
                self.assertGreater(float(points.grad[:, 3:].abs().sum()), 0.0)
                self.assertGreater(float(diameters.grad[:, 3:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
