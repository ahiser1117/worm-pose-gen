import unittest
from unittest import mock

import numpy as np
import torch

from worm_pose_gen.body_net import OUTPUTS
from worm_pose_gen.body_proposal import OVERLAP_THRESHOLD, FieldPrediction, field_evidence, propose_trace, trace_start
from worm_pose_gen.body_targets import point_heatmap, render_body_targets
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import MaskFitConfig
from worm_pose_gen.pipeline import FitParams, fit_setup, network_trace_start
from worm_pose_gen.recording_prior import RecordingPrior

SHAPE = (300, 420)
CORNERS = np.array([[80, 150], [340, 150], [340, 250], [240, 250], [240, 60]], float)


def looped_body():
    """A realistic-size body that turns back and crosses its first run at (240, 150)."""

    pieces = [np.linspace(a, b, 60, endpoint=False) for a, b in zip(CORNERS[:-1], CORNERS[1:])]
    centerline = np.concatenate(pieces + [CORNERS[-1:]])
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    points = np.stack((xx.ravel(), yy.ravel()), 1).astype(float)
    distance = np.full(len(points), np.inf)
    for a, b in zip(centerline[:-1], centerline[1:]):
        t = np.clip(((points - a) @ (b - a)) / max(float((b - a) @ (b - a)), 1e-9), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(points - (a + t[:, None] * (b - a)), axis=1))
    mask = (distance <= 14).reshape(SHAPE)
    return centerline, mask


def prediction_from(centerline, mask, noise=0.0, seed=0):
    targets = render_body_targets(mask, centerline, np.full(len(centerline), 28.0))
    ap = np.nan_to_num(targets.ap, nan=0.5).astype(np.float32)
    if noise:
        ap = np.clip(ap + np.random.default_rng(seed).normal(0, noise, ap.shape), 0, 1).astype(np.float32)
    return FieldPrediction(
        mask=mask.astype(np.float32), ap=ap,
        head=point_heatmap(SHAPE, centerline[0], 6.0), tail=point_heatmap(SHAPE, centerline[-1], 6.0),
        overlap=targets.overlap.astype(np.float32),
    )


class ProposeTraceTests(unittest.TestCase):
    def test_trace_follows_the_body_through_its_crossing(self):
        centerline, mask = looped_body()
        trace = propose_trace(prediction_from(centerline, mask, noise=0.03), mask)
        np.testing.assert_allclose(trace[0], CORNERS[0], atol=2)
        np.testing.assert_allclose(trace[-1], CORNERS[-1], atol=2)
        # Every point lies near the true midline, and the walk advances along it.
        nearest = [int(np.argmin(np.linalg.norm(centerline - p, axis=1))) for p in trace]
        self.assertLess(max(float(np.linalg.norm(centerline[i] - p)) for i, p in zip(nearest, trace)), 10.0)
        self.assertEqual(nearest, sorted(nearest))

    def test_an_end_out_of_view_is_left_out(self):
        centerline, mask = looped_body()
        prediction = prediction_from(centerline, mask)
        prediction = FieldPrediction(prediction.mask, prediction.ap, prediction.head, np.zeros_like(prediction.tail), prediction.overlap)
        trace = propose_trace(prediction, mask)
        self.assertGreater(np.linalg.norm(trace[-1] - CORNERS[-1]), 5.0)
        self.assertLess(np.linalg.norm(trace[0] - CORNERS[0]), 2.0)

    def test_an_end_off_the_mask_is_left_out(self):
        centerline, mask = looped_body()
        prediction = prediction_from(centerline, mask)
        # The tail heatmap peaks on a fragment the cleanup removed, 40 px past the mask's tail end.
        fragment = CORNERS[-1] - [0, 54]
        prediction = FieldPrediction(prediction.mask, prediction.ap, prediction.head, point_heatmap(SHAPE, fragment, 6.0), prediction.overlap)
        trace = propose_trace(prediction, mask)
        self.assertTrue(all(mask[int(y), int(x)] for x, y in trace))
        self.assertGreater(float(np.linalg.norm(trace[-1] - fragment)), 30.0)
        evidence = field_evidence(prediction, mask)
        self.assertIsNone(evidence.tail_xy)
        np.testing.assert_allclose(evidence.head_xy, CORNERS[0], atol=1)

    def test_no_body_gives_no_trace(self):
        centerline, mask = looped_body()
        self.assertIsNone(propose_trace(prediction_from(centerline, mask), np.zeros(SHAPE, bool)))

    def test_a_body_cut_by_the_image_edge_is_traced_to_the_edge(self):
        centerline, mask = looped_body()
        prediction = prediction_from(centerline, mask)
        # Without the top 100 rows the last run, going up from (240, 250), leaves the image and the tail with it.
        clipped = FieldPrediction(**{name: getattr(prediction, name)[100:] for name in OUTPUTS})
        trace = propose_trace(clipped, mask[100:])
        np.testing.assert_allclose(trace[0], CORNERS[0] - [0, 100], atol=2)
        self.assertLess(abs(trace[-1, 0] - 240), 10.0)
        self.assertLess(trace[-1, 1], 40.0)
        self.assertTrue(all(mask[100:][int(y), int(x)] for x, y in trace))


class TraceStartTests(unittest.TestCase):
    def test_a_trace_stopping_short_of_the_image_edge_is_continued_off_camera(self):
        centerline, mask = looped_body()
        prediction = prediction_from(centerline, mask)
        # Without the top 109 rows the last run leaves the image, and its last band point lies about 20 px below the edge.
        cut = 109
        clipped = FieldPrediction(**{name: getattr(prediction, name)[cut:] for name in OUTPUTS})
        self.assertGreater(propose_trace(clipped, mask[cut:])[-1, 1], 15.0)
        config = MaskFitConfig()
        curve = decode_centerline(trace_start(clipped, mask[cut:], config=config, length_px=650.0).latent, config.coefficients)
        self.assertAlmostEqual(float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()), 650.0, delta=10.0)
        self.assertLess(curve[-1, 1], -50.0)  # the tail, off the top edge
        np.testing.assert_allclose(curve[0], CORNERS[0] - [0, cut], atol=3)

    def test_with_a_length_the_body_is_laid_from_the_head(self):
        centerline, mask = looped_body()
        config = MaskFitConfig()
        prediction = prediction_from(centerline, mask)
        start = trace_start(prediction, mask, config=config, length_px=500.0, width_px=28.0)
        curve = decode_centerline(start.latent, config.coefficients)
        self.assertEqual(start.latent[config.coefficients + 1], 500.0)
        np.testing.assert_allclose(curve[0], CORNERS[0], atol=3)
        # The 650 px body is cut 500 px from the head, on its last run (which starts 460 px along it,
        # at y=250), well short of the tail (y=60); the trace cuts corners, so a little further along.
        self.assertLess(abs(curve[-1, 0] - 240.0), 8.0)
        self.assertTrue(150.0 < curve[-1, 1] < 230.0)
        # The tail is an outcome, not an input: a tail heatmap elsewhere on the body changes nothing.
        moved = FieldPrediction(prediction.mask, prediction.ap, prediction.head, point_heatmap(SHAPE, np.array([340.0, 200.0]), 6.0), prediction.overlap)
        np.testing.assert_array_equal(trace_start(moved, mask, config=config, length_px=500.0, width_px=28.0).latent, start.latent)

    def test_a_short_trace_is_continued_along_the_mask(self):
        centerline, mask = looped_body()
        config = MaskFitConfig()
        prediction = prediction_from(centerline, mask)
        # A-P saturates over the last 150 px (a label whose tail stopped short): the bands end there, the body does not.
        ap = np.minimum(prediction.ap * 650 / 500, 1.0).astype(np.float32)
        short = FieldPrediction(prediction.mask, ap, prediction.head, prediction.tail, prediction.overlap)
        self.assertGreater(float(np.linalg.norm(propose_trace(short, mask, tail=False)[-1] - CORNERS[-1])), 50.0)
        curve = decode_centerline(trace_start(short, mask, config=config, length_px=640.0, width_px=28.0).latent, config.coefficients)
        self.assertLess(float(np.linalg.norm(curve[-1] - CORNERS[-1])), 15.0)

    def test_without_the_head_the_trace_runs_to_the_tail(self):
        centerline, mask = looped_body()
        config = MaskFitConfig()
        prediction = prediction_from(centerline, mask)
        headless = FieldPrediction(prediction.mask, prediction.ap, np.zeros_like(prediction.head), prediction.tail, prediction.overlap)
        curve = decode_centerline(trace_start(headless, mask, config=config, length_px=500.0, width_px=28.0).latent, config.coefficients)
        np.testing.assert_allclose(curve[-1], CORNERS[-1], atol=3)
        self.assertGreater(float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()), 600.0)

    def test_a_known_width_replaces_the_measurement(self):
        centerline, mask = looped_body()
        config = MaskFitConfig()
        prediction = prediction_from(centerline, mask)
        self.assertAlmostEqual(trace_start(prediction, mask, config=config).width_px, 28.0, delta=3.0)
        shape = np.linspace(-0.1, 0.1, config.width_coefficients)
        with mock.patch("worm_pose_gen.mask_fit.estimate_width_along_normals", side_effect=AssertionError("measured")):
            start = trace_start(prediction, mask, config=config, width_px=21.0, width_shape=shape)
        self.assertEqual(start.width_px, 21.0)
        np.testing.assert_array_equal(start.width_shape, shape)

    def test_the_pipeline_start_takes_the_recording_priors_width(self):
        centerline, mask = looped_body()
        coefficients = MaskFitConfig().width_coefficients
        prior = RecordingPrior(
            length_px=700.0, log_length_sigma=0.05, width_px=24.0, log_width_sigma=0.05,
            width_shape=tuple(np.linspace(-0.2, 0.2, coefficients)), width_shape_sigma=(0.1,) * coefficients,
            frames_used=12, frames_candidates=12, selection={},
        )
        start = network_trace_start(prediction_from(centerline, mask), mask, fit_setup(FitParams(), prior))
        self.assertEqual(start.width_px, 24.0)
        np.testing.assert_array_equal(start.width_shape, prior.width_shape)


class FieldEvidenceTests(unittest.TestCase):
    def test_evidence_is_the_field_on_the_body_off_crossings(self):
        centerline, mask = looped_body()
        prediction = prediction_from(centerline, mask)
        evidence = field_evidence(prediction, mask)
        np.testing.assert_array_equal(evidence.ap, np.where(mask & (prediction.overlap < OVERLAP_THRESHOLD), prediction.ap, np.nan))
        np.testing.assert_allclose(evidence.head_xy, CORNERS[0], atol=1)
        np.testing.assert_allclose(evidence.tail_xy, CORNERS[-1], atol=1)
        self.assertTrue(np.isnan(field_evidence(prediction, np.zeros(SHAPE, bool)).ap).all())

    def test_an_end_at_the_image_edge_does_not_score_fits(self):
        centerline, mask = looped_body()
        prediction = prediction_from(centerline, mask)
        # Without the top 50 rows the tail sits 10 px below the edge, where a body leaving the camera fires the tail heatmap too.
        clipped = FieldPrediction(**{name: getattr(prediction, name)[50:] for name in OUTPUTS})
        evidence = field_evidence(clipped, mask[50:])
        self.assertIsNone(evidence.tail_xy)
        np.testing.assert_allclose(evidence.head_xy, CORNERS[0] - [0, 50], atol=1)


class StubModule(torch.nn.Module):
    """Returns logits of a fixed prediction, like a trained body-field network would."""

    def __init__(self, prediction):
        super().__init__()
        maps = np.stack([getattr(prediction, name) for name in OUTPUTS]).clip(1e-4, 1 - 1e-4)
        self.logits = torch.as_tensor(np.log(maps / (1 - maps)), dtype=torch.float32)
        self.lags = ()
        self.device = torch.device("cpu")
        self.checkpoint_path = "stub.ckpt"

    def forward(self, images):
        return self.logits[None]


if __name__ == "__main__":
    unittest.main()
