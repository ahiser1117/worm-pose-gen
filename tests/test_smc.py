import math
import unittest

import torch

from worm_pose_gen.smc import (
    effective_sample_size,
    normalize_log_weights,
    propagate_position_velocity,
    resample_with_genealogy,
    systematic_resample,
    trace_genealogy,
    trace_genealogy_path,
)


class SMCPrimitiveTests(unittest.TestCase):
    def test_stable_normalization_and_ess(self) -> None:
        values = torch.tensor([10_000.0, 9_999.0, -torch.inf], dtype=torch.float64)
        normalized = normalize_log_weights(values)
        torch.testing.assert_close(normalized.exp().sum(), torch.tensor(1.0, dtype=torch.float64))
        self.assertTrue(bool(torch.isfinite(normalized[:2]).all()))
        self.assertTrue(bool(torch.isneginf(normalized[2])))
        expected = 1.0 / normalized.exp().square().sum()
        torch.testing.assert_close(effective_sample_size(values), expected)

    def test_positive_infinite_weights_split_mass(self) -> None:
        values = torch.tensor([0.0, torch.inf, -3.0, torch.inf])
        probability = normalize_log_weights(values).exp()
        torch.testing.assert_close(probability, torch.tensor([0.0, 0.5, 0.0, 0.5]))

    def test_systematic_resampling_is_seeded_and_preserves_two_modes(self) -> None:
        # Both separated modes own substantial mass, so one systematic grid
        # cannot eliminate the lower-probability mode.
        particles = torch.cat((torch.full((32,), -10.0), torch.full((32,), 10.0)))
        weights = torch.cat((torch.full((32,), 0.7 / 32), torch.full((32,), 0.3 / 32)))
        first = systematic_resample(weights, seed=41)
        second = systematic_resample(weights, seed=41)
        self.assertTrue(torch.equal(first, second))
        selected = particles[first]
        self.assertTrue(bool((selected < 0).any()))
        self.assertTrue(bool((selected > 0).any()))

    def test_resampling_returns_genealogy_and_uniform_log_weights(self) -> None:
        particles = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        log_weights = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        selected, ancestors, reset = resample_with_genealogy(
            particles, log_weights, seed=9
        )
        torch.testing.assert_close(selected, particles[ancestors])
        torch.testing.assert_close(reset, torch.full((4,), -math.log(4)))
        self.assertEqual(ancestors.dtype, torch.long)

    def test_gaussian_propagation_is_seeded(self) -> None:
        position = torch.zeros(20, 2)
        velocity = torch.full((20, 2), 0.5)
        first = propagate_position_velocity(
            position, velocity, dt=0.2,
            position_noise_std=0.1, velocity_noise_std=0.05, seed=123,
        )
        second = propagate_position_velocity(
            position, velocity, dt=0.2,
            position_noise_std=0.1, velocity_noise_std=0.05, seed=123,
        )
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        self.assertFalse(bool(torch.equal(first[0], position + 0.2 * velocity)))

    def test_backward_genealogy_path(self) -> None:
        ancestors = torch.tensor([[0, 0, 2], [1, 1, 0]], dtype=torch.long)
        indices = trace_genealogy(ancestors, 2)
        torch.testing.assert_close(indices, torch.tensor([0, 0, 2]))
        history = torch.tensor([
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ])
        torch.testing.assert_close(
            trace_genealogy_path(history, ancestors, 2),
            torch.tensor([10.0, 20.0, 32.0]),
        )

    def test_seeded_filtering_is_deterministic(self) -> None:
        def run(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
            generator = torch.Generator().manual_seed(seed)
            position = torch.linspace(-3, 3, 64)
            velocity = torch.zeros_like(position)
            genealogy = []
            history = [position.clone()]
            for observation in (0.3, 0.8, 1.4):
                position, velocity = propagate_position_velocity(
                    position, velocity, dt=1.0,
                    position_noise_std=0.25, velocity_noise_std=0.08,
                    generator=generator,
                )
                log_weights = -0.5 * ((position - observation) / 0.35).square()
                state = torch.stack((position, velocity), dim=-1)
                state, ancestors, _ = resample_with_genealogy(
                    state, log_weights, generator=generator
                )
                position, velocity = state.unbind(-1)
                genealogy.append(ancestors)
                history.append(position.clone())
            return torch.stack(history), torch.stack(genealogy)

        first = run(20260819)
        second = run(20260819)
        torch.testing.assert_close(first[0], second[0])
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertFalse(bool(torch.equal(first[0], run(20260820)[0])))


if __name__ == "__main__":
    unittest.main()
