"""Numerical regression for the scalar-coordinate polyline renderer."""

import unittest

import torch

from worm_pose_gen.mask_fit import render_tube_segments


def vector_reference(points, diameter, height, width):
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=points.dtype), torch.arange(width, dtype=points.dtype), indexing="ij"
    )
    pixels = torch.stack((xx, yy), -1).reshape(1, -1, 1, 2)
    start = points[:, None, :-1]
    segment = (points[:, 1:] - points[:, :-1])[:, None]
    t = (((pixels - start) * segment).sum(-1) / segment.square().sum(-1).clamp_min(1e-6)).clamp(0, 1)
    distance = ((pixels - (start + t[..., None] * segment)).square().sum(-1) + torch.finfo(points.dtype).eps).sqrt()
    widths = (1 - t) * diameter[:, None, :-1] + t * diameter[:, None, 1:]
    return torch.sigmoid((0.5 * widths - distance).amax(-1) / 0.8).reshape(len(points), height, width)


class RendererRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and the GPU compiler")
    def test_compiled_cuda_gradients_stay_finite_on_recorded_geometry(self):
        from worm_pose_gen.latent import decode_centerline_torch
        from worm_pose_gen.mask_fit import default_width_template

        # Two initialization latents from a real frame that produced NaNs in
        # every parameter on the first compiled scalar-renderer backward.
        # Store geometry rather than a recording dependency or image fixture.
        latents = torch.tensor([
            [-.2538983524, .1529210657, .4097109735, .5979629159,
             .1188794076, -.3603157103, -.5210646391, -.5954734683,
             -.3104858100, .05767707154, .5076470375, .5522603989,
             .02274555527, -.1212742925, -.3597614467, -.3187085092,
             .4711856246, 609.5786133, 682.0573120, 459.3647156],
            [0.] * 16 + [-2.715449333, 600.0000610, 700.8553467, 467.8057556],
        ], dtype=torch.float32, device="cuda")
        index = torch.cat((torch.arange(0, 100, 2, device="cuda"), torch.tensor([99], device="cuda")))
        offset = torch.tensor([380., 283.], device="cuda")
        points = (decode_centerline_torch(latents)[:, index] - offset + .5) / 4 - .5
        template = torch.as_tensor(default_width_template(), dtype=torch.float32, device="cuda")
        widths = torch.tensor([51., 48.9849968], device="cuda")[:, None] * template[None, index] / 4
        compiled = torch.compile(render_tube_segments, dynamic=True)
        with torch.no_grad():
            self.assertTrue(torch.isfinite(compiled(points, widths, 88, 144)).all().item())
        expected_points, expected_widths = points.clone().requires_grad_(), widths.clone().requires_grad_()
        actual_points, actual_widths = points.clone().requires_grad_(), widths.clone().requires_grad_()
        expected = render_tube_segments(expected_points, expected_widths, 88, 144)
        actual = compiled(actual_points, actual_widths, 88, 144)
        weights = torch.linspace(.1, 1, actual.numel(), device="cuda").reshape_as(actual)
        expected_grads = torch.autograd.grad((expected * weights).sum(), (expected_points, expected_widths))
        actual_grads = torch.autograd.grad((actual * weights).sum(), (actual_points, actual_widths))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            self.assertTrue(torch.isfinite(actual_grad).all().item())
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-3, atol=5e-3)

    def test_outputs_and_gradients_match_vector_reference(self):
        # Varying widths, self-contact, off-camera points and a zero-length
        # segment exercise the geometric cases that affect fitting gradients.
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                points = torch.tensor([
                    [[-2, 5], [8, 5], [8, 5], [9, 14], [2, 14], [3, 6]],
                    [[7, -5], [12, 3], [6, 10], [11, 15], [18, 8], [26, 5]],
                ], dtype=dtype)
                widths = torch.tensor([[2, 6, 7, 5, 3, 1], [1, 3, 6, 7, 5, 2]], dtype=dtype)
                reference_points = points.clone().requires_grad_()
                reference_widths = widths.clone().requires_grad_()
                reference = vector_reference(reference_points, reference_widths, 20, 24)
                points.requires_grad_()
                widths.requires_grad_()
                actual = render_tube_segments(points, widths, 20, 24)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                weights = torch.linspace(0.1, 1, actual.numel(), dtype=dtype).reshape_as(actual)
                expected_grads = torch.autograd.grad((reference * weights).sum(), (reference_points, reference_widths))
                actual_grads = torch.autograd.grad((actual * weights).sum(), (points, widths))
                for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
