"""Dependency-minimal PyTorch primitives for sequential Monte Carlo.

These functions are deliberately generic: they know nothing about worm data,
rendering, observations, or experiment configuration. Genealogy convention:
``ancestor_indices[t, child]`` is the particle index at time ``t`` that
produced ``child`` at time ``t + 1``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def normalize_log_weights(log_weights: Tensor, dim: int = -1) -> Tensor:
    """Return normalized log weights without exponentiating first.

    Finite values and ``-inf`` (zero mass) are accepted. If one or more entries
    are ``+inf``, probability is divided equally among those entries. A slice
    containing only ``-inf`` has no probability measure and is rejected.
    """

    if not isinstance(log_weights, Tensor) or log_weights.ndim == 0:
        raise ValueError("log_weights must be a tensor with at least one dimension")
    if not log_weights.is_floating_point():
        raise TypeError("log_weights must have floating dtype")
    if bool(torch.isnan(log_weights).any()):
        raise ValueError("log_weights must not contain NaN")
    axis = dim % log_weights.ndim
    positive = torch.isposinf(log_weights)
    has_positive = positive.any(dim=axis, keepdim=True)
    all_negative = torch.isneginf(log_weights).all(dim=axis, keepdim=True)
    if bool((all_negative & ~has_positive).any()):
        raise ValueError("a log-weight slice has zero total mass")

    ordinary = log_weights - torch.logsumexp(log_weights, dim=axis, keepdim=True)
    positive_count = positive.sum(dim=axis, keepdim=True).clamp_min(1)
    positive_only = torch.where(
        positive,
        -torch.log(positive_count.to(dtype=log_weights.dtype)),
        torch.full_like(log_weights, -torch.inf),
    )
    return torch.where(has_positive, positive_only, ordinary)


def effective_sample_size(log_weights: Tensor, dim: int = -1) -> Tensor:
    """Compute ESS = ``1 / sum(w**2)`` from unnormalized log weights."""

    normalized = normalize_log_weights(log_weights, dim=dim)
    return torch.exp(-torch.logsumexp(2.0 * normalized, dim=dim))


def _generator(
    device: torch.device, *, generator: torch.Generator | None, seed: int | None
) -> torch.Generator | None:
    if generator is not None and seed is not None:
        raise ValueError("pass either generator or seed, not both")
    if seed is None:
        return generator
    result = torch.Generator(device=device)
    result.manual_seed(seed)
    return result


def systematic_resample(
    weights: Tensor,
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> Tensor:
    """Return systematic-resampling ancestor indices for one particle set.

    ``weights`` need not sum exactly to one, but must be finite, nonnegative,
    one-dimensional, and have positive total mass. Supplying ``seed`` makes the
    sole random offset deterministic without changing PyTorch's global RNG.
    """

    if not isinstance(weights, Tensor) or weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a nonempty one-dimensional tensor")
    if not weights.is_floating_point():
        raise TypeError("weights must have floating dtype")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("weights must be finite and nonnegative")
    total = weights.sum()
    if not bool(total > 0):
        raise ValueError("weights must have positive total mass")
    probabilities = weights / total
    count = len(weights)
    rng = _generator(weights.device, generator=generator, seed=seed)
    offset = torch.rand((), dtype=weights.dtype, device=weights.device, generator=rng) / count
    positions = offset + torch.arange(count, dtype=weights.dtype, device=weights.device) / count
    cumulative = torch.cumsum(probabilities, dim=0)
    cumulative[-1] = 1.0  # Prevent roundoff from putting the last point out of range.
    # ``right=True`` avoids selecting a zero-mass bin if the random grid lands
    # exactly on a repeated cumulative-weight boundary.
    return torch.searchsorted(cumulative, positions, right=True).clamp_max(count - 1).to(torch.long)


def resample_with_genealogy(
    particles: Tensor,
    log_weights: Tensor,
    *,
    particle_dim: int = 0,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Resample particles and return particles, ancestors, and uniform logs.

    The returned ancestor indices are suitable for one row of a genealogy
    table. ``log_weights`` is one-dimensional; arbitrary trailing particle
    state dimensions are supported through ``particle_dim``.
    """

    if not isinstance(particles, Tensor) or particles.ndim == 0:
        raise ValueError("particles must have at least one dimension")
    axis = particle_dim % particles.ndim
    if log_weights.ndim != 1 or particles.shape[axis] != len(log_weights):
        raise ValueError("log_weights must match the selected particle dimension")
    normalized = normalize_log_weights(log_weights)
    ancestors = systematic_resample(
        normalized.exp(), generator=generator, seed=seed
    )
    selected = torch.index_select(particles, axis, ancestors)
    uniform = torch.full_like(log_weights, -math.log(len(log_weights)))
    return selected, ancestors, uniform


def propagate_position_velocity(
    position: Tensor,
    velocity: Tensor,
    *,
    dt: float | Tensor = 1.0,
    position_noise_std: float | Tensor = 0.0,
    velocity_noise_std: float | Tensor = 0.0,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply a constant-velocity Gaussian transition to a latent state.

    ``position`` and ``velocity`` may have any shared floating shape. Independent
    Gaussian innovations are added to predicted position and velocity. Standard
    deviations and ``dt`` broadcast against the state.
    """

    if position.shape != velocity.shape or not position.is_floating_point() or not velocity.is_floating_point():
        raise ValueError("position and velocity must share a floating tensor shape")
    if position.device != velocity.device or position.dtype != velocity.dtype:
        raise ValueError("position and velocity must share device and dtype")
    delta = torch.as_tensor(dt, dtype=position.dtype, device=position.device)
    position_std = torch.as_tensor(
        position_noise_std, dtype=position.dtype, device=position.device
    )
    velocity_std = torch.as_tensor(
        velocity_noise_std, dtype=position.dtype, device=position.device
    )
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("dt must be finite")
    if not bool(torch.isfinite(position_std).all()) or bool((position_std < 0).any()):
        raise ValueError("position_noise_std must be finite and nonnegative")
    if not bool(torch.isfinite(velocity_std).all()) or bool((velocity_std < 0).any()):
        raise ValueError("velocity_noise_std must be finite and nonnegative")
    rng = _generator(position.device, generator=generator, seed=seed)
    position_noise = torch.randn(
        position.shape, dtype=position.dtype, device=position.device, generator=rng
    ) * position_std
    velocity_noise = torch.randn(
        velocity.shape, dtype=velocity.dtype, device=velocity.device, generator=rng
    ) * velocity_std
    return position + delta * velocity + position_noise, velocity + velocity_noise


def trace_genealogy(ancestor_indices: Tensor, terminal_indices: Tensor | int) -> Tensor:
    """Trace terminal particle indices backward through an SMC genealogy.

    For ``ancestor_indices`` of shape ``[T, N]``, returns indices of shape
    ``[T+1]`` for one terminal or ``[T+1, K]`` for K terminals.
    """

    if ancestor_indices.ndim != 2 or ancestor_indices.dtype != torch.long:
        raise ValueError("ancestor_indices must be a long tensor with shape [T,N]")
    steps, particles = ancestor_indices.shape
    if particles == 0:
        raise ValueError("genealogy must contain particles")
    terminal = torch.as_tensor(
        terminal_indices, dtype=torch.long, device=ancestor_indices.device
    )
    scalar = terminal.ndim == 0
    current = terminal.reshape(1) if scalar else terminal.reshape(-1)
    if bool(((current < 0) | (current >= particles)).any()):
        raise IndexError("terminal particle index is out of range")
    path = torch.empty(
        (steps + 1, len(current)), dtype=torch.long, device=ancestor_indices.device
    )
    path[-1] = current
    for time in range(steps - 1, -1, -1):
        current = ancestor_indices[time].index_select(0, current)
        path[time] = current
    return path[:, 0] if scalar else path


def trace_genealogy_path(
    particle_history: Tensor,
    ancestor_indices: Tensor,
    terminal_index: int,
) -> Tensor:
    """Gather one state trajectory using :func:`trace_genealogy` indices."""

    if particle_history.ndim < 2:
        raise ValueError("particle_history must have shape [T+1,N,...]")
    if particle_history.shape[:2] != (
        ancestor_indices.shape[0] + 1,
        ancestor_indices.shape[1],
    ):
        raise ValueError("particle history and genealogy shapes are inconsistent")
    indices = trace_genealogy(ancestor_indices, terminal_index)
    times = torch.arange(len(indices), device=particle_history.device)
    if indices.device != particle_history.device:
        indices = indices.to(particle_history.device)
    return particle_history[times, indices]


# Descriptive aliases kept for callers that prefer explicit algorithm names.
systematic_resampling = systematic_resample
gaussian_propagate_position_velocity = propagate_position_velocity
backward_trace_genealogy = trace_genealogy
