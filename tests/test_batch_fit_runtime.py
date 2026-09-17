"""Batch fitting keeps per-row histories and winners across cached stages."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

from worm_pose_gen.batch_fit import BatchFitConfig, _point_index, fit_masks
from worm_pose_gen.mask_fit import Initialization


class BatchFitRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.config = BatchFitConfig(
            n_points=16, coefficients=4, crop_padding=8, crop_multiple=4,
            stage_downsample=(2, 1, 1), stage_steps=(2, 3, 2),
            stage_lr_scale=(1., .5, .1), stage_point_stride=(2, 1, 1),
            compile_renderer=False, length_bounds_px=(10., 70.),
            width_bounds_px=(2., 15.), default_length_px=40., default_width_px=8.,
        )
        self.mask = np.zeros((40, 64), dtype=bool)
        self.mask[16:24, 12:52] = True
        self.starts = [
            Initialization('center', np.array([0., 0., 0., 0., 0., 40., 32., 20.]), 8.),
            Initialization('offset', np.array([.1, .1, .1, .1, 0., 42., 32., 22.]), 9.),
        ]

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_point_indices_include_last_point_without_duplicates(self):
        for n in (16, 17):
            for stride in (0, 1, 2, 5, 20):
                expected = list(range(0, n, max(stride, 1)))
                if expected[-1] != n - 1:
                    expected.append(n - 1)
                self.assertEqual(_point_index(n, stride, torch.device('cpu')).tolist(), expected)

    def test_cached_repeated_stages_preserve_independent_row_histories(self):
        batched = fit_masks([self.mask, self.mask], [self.starts, self.starts[::-1]],
                            config=self.config, device='cpu')
        for result, starts in zip(batched, (self.starts, self.starts[::-1])):
            single = fit_masks([self.mask], [starts], config=self.config, device='cpu')[0]
            self.assertEqual(result.energy_history.shape, (7, 2))
            np.testing.assert_allclose(result.energy_history, single.energy_history, atol=1e-6)
            np.testing.assert_allclose(result.centerline_xy, single.centerline_xy, atol=1e-5)
            np.testing.assert_array_equal(result.rendered_hard_mask, single.rendered_hard_mask)
            self.assertEqual(result.best_index, single.best_index)
            self.assertEqual(result.initializations[result.best_index].name,
                             batched[0].initializations[batched[0].best_index].name)

    def test_empty_schedule_has_correct_history_shape(self):
        config = replace(self.config, stage_steps=(0, 0, 0))
        result = fit_masks([self.mask], [self.starts], config=config, device='cpu')[0]
        self.assertEqual(result.energy_history.shape, (0, 2))
        self.assertTrue(np.isfinite(result.centerline_xy).all())

    def test_whole_energy_capture_reuses_graphs_across_groups(self):
        # The eager backend verifies capture, gradients and guard reuse without
        # requiring a C++/CUDA compiler in the unit-test environment.
        import torch._dynamo
        torch._dynamo.reset()
        graphs = []
        original_compile = torch.compile

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        def compile_for_test(function, **kwargs):
            return original_compile(function, backend=backend, **kwargs)

        config = replace(self.config, row_pixel_budget=1)
        expected = fit_masks([self.mask, self.mask], [self.starts, self.starts[::-1]],
                             config=config, device='cpu')
        try:
            with patch('worm_pose_gen.batch_fit.torch.compile', side_effect=compile_for_test):
                actual = fit_masks([self.mask, self.mask], [self.starts, self.starts[::-1]],
                                   config=replace(config, compile_energy=True), device='cpu')
            # One graph per training stage and one finest-stage no-grad graph;
            # creating a new fit state for the second group must reuse them.
            self.assertEqual(len(graphs), 3)
            for eager, compiled in zip(expected, actual):
                np.testing.assert_array_equal(eager.energy_history, compiled.energy_history)
                np.testing.assert_array_equal(eager.centerline_xy, compiled.centerline_xy)
                np.testing.assert_array_equal(eager.rendered_hard_mask, compiled.rendered_hard_mask)
        finally:
            torch._dynamo.reset()

    def test_energy_specialization_limit_falls_back_and_continues_groups(self):
        from torch._dynamo.exc import FailOnRecompileLimitHit
        from worm_pose_gen import batch_fit
        from worm_pose_gen.mask_fit import render_tube_segments

        config = replace(self.config, row_pixel_budget=1)
        expected = fit_masks([self.mask, self.mask], [self.starts, self.starts[::-1]],
                             config=config, device='cpu')
        calls = []
        renderer_flags = []

        def compile_for_test(function, **kwargs):
            # Let the first two evaluations run to verify fallback mid-fit
            # preserves the optimizer state, history and stage schedule.
            group_calls = 0
            def compiled(*args):
                nonlocal group_calls
                group_calls += 1
                calls.append(group_calls)
                if group_calls == 3:
                    raise FailOnRecompileLimitHit('test specialization budget')
                return function(*args)
            return compiled

        def renderer_for_test(enabled):
            renderer_flags.append(enabled)
            return render_tube_segments

        with patch('worm_pose_gen.batch_fit.torch.compile', side_effect=compile_for_test), \
             patch('worm_pose_gen.batch_fit.get_renderer', side_effect=renderer_for_test), \
             patch('worm_pose_gen.batch_fit._ENERGY_COMPILE_FALLBACKS', 0), \
             warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            actual = fit_masks([self.mask, self.mask], [self.starts, self.starts[::-1]],
                               config=replace(config, compile_energy=True, compile_renderer=True), device='cpu')
            self.assertEqual(batch_fit._ENERGY_COMPILE_FALLBACKS, 2)
        self.assertEqual(calls, [1, 2, 3, 1, 2, 3])
        self.assertEqual(renderer_flags, [False, True, False, True])
        fallback_warnings = [w for w in emitted if 'specialization limit' in str(w.message)]
        self.assertEqual(len(fallback_warnings), 1)
        self.assertIs(fallback_warnings[0].category, RuntimeWarning)
        for eager, recovered in zip(expected, actual):
            np.testing.assert_array_equal(eager.energy_history, recovered.energy_history)
            np.testing.assert_array_equal(eager.centerline_xy, recovered.centerline_xy)
            np.testing.assert_array_equal(eager.rendered_hard_mask, recovered.rendered_hard_mask)

    def test_energy_compile_other_errors_still_surface(self):
        def broken_compile(function, **kwargs):
            def compiled(*args):
                raise RuntimeError('unrelated compiler failure')
            return compiled
        with patch('worm_pose_gen.batch_fit.torch.compile', side_effect=broken_compile):
            with self.assertRaisesRegex(RuntimeError, 'unrelated compiler failure'):
                fit_masks([self.mask], [self.starts],
                          config=replace(self.config, compile_energy=True), device='cpu')

    def test_temporal_and_head_fits_bypass_whole_energy_compilation(self):
        from worm_pose_gen.head_fit import HeadConstraint
        from worm_pose_gen.mask_fit import render_tube_segments

        reference = np.column_stack((np.linspace(12, 52, self.config.n_points), np.full(self.config.n_points, 20.)))
        cases = [
            {'references': [reference]},
            {'references': [None]},
            {'head_constraints': [HeadConstraint(tracking_xy=np.array([12., 20.]))]},
            {'head_constraints': [None]},
        ]
        for arguments in cases:
            with self.subTest(arguments=list(arguments)):
                config = replace(self.config, temporal_prior_weight=.01)
                expected = fit_masks([self.mask], [self.starts], config=config, device='cpu', **arguments)[0]
                with patch('worm_pose_gen.batch_fit.torch.compile') as compiler, \
                     patch('worm_pose_gen.batch_fit.get_renderer', return_value=render_tube_segments) as renderer:
                    actual = fit_masks([self.mask], [self.starts], device='cpu', **arguments,
                                       config=replace(config, compile_energy=True, compile_renderer=True))[0]
                compiler.assert_not_called()
                renderer.assert_called_once_with(True)
                np.testing.assert_array_equal(expected.energy_history, actual.energy_history)
                np.testing.assert_array_equal(expected.centerline_xy, actual.centerline_xy)

    def test_refit_schedules_disable_whole_energy_compilation(self):
        from worm_pose_gen.propagation import slow_schedule, warm_schedule
        config = replace(self.config, compile_energy=True, compile_renderer=True)
        for schedule in (warm_schedule(config), slow_schedule(config, self.config)):
            self.assertFalse(schedule.compile_energy)
            self.assertTrue(schedule.compile_renderer)


class HardWinnerRenderTests(unittest.TestCase):
    def test_cropped_hard_render_matches_full_raster(self):
        from worm_pose_gen.batch_fit import _render_hard_winners
        from worm_pose_gen.mask_fit import render_tube_segments
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            generator = torch.Generator().manual_seed(104)
            centerline = torch.randn((5, 12, 2), generator=generator).cumsum(1) * 2 + 25
            centerline[1, :, 0] -= 25  # censored on the left
            centerline[2, :, 1] += 100  # entirely outside the camera
            centerline[3, :, 0] += 43  # censored on the right
            diameter = torch.rand((5, 12), generator=generator) * 12 + 1
            for softness in (.2, 2.5):
                full = render_tube_segments(centerline, diameter, 64, 72, edge_softness=softness)
                for threshold in (0., 1., .5, .01, .99, 1e-100, 1 - 1e-10):
                    for rows in (1, 3, 8):
                        actual = _render_hard_winners(centerline, diameter, 64, 72,
                            edge_softness=softness, threshold=threshold, chunk_rows=rows)
                        torch.testing.assert_close(actual, full >= threshold, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous)
