"""Joint temporal smoothing of fixed-body chains under a first-order motion prior.

A chain is the fixed body of ``fixed_body`` in the pipeline's own pose space:
``S`` equal links of one recording length whose tangent angles are a cubic
B-spline of ``K`` coefficients (``latent.cubic_bspline_basis``), so a frame
is its head position plus ``K`` angle coefficients and the result encodes
exactly as a pose latent.  The worm is overdamped at its scale, so the prior
bounds *rates*, not accelerations: each pair of consecutive frames pays a
quadratic penalty on the change of head position and of every coefficient,
standardised by scales measured on trusted frames (``motion_scales``) and
divided by the source-frame gap.  The data term pulls each frame toward
equal arc-distance targets sampled from its current pose
(``fixed_body.chain_targets``) with a per-frame weight, so a frame the
pipeline does not trust is shaped by its neighbours instead of smearing its
error into them.  All frames are solved together (``smooth_chains``) by
Levenberg-Marquardt: the normal matrix is block-tridiagonal in time with
``2 + K`` variables per frame, so one iteration is a block Thomas sweep and
a whole recording takes seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .latent import cubic_bspline_basis

MAD_TO_SIGMA = 1.4826
MIN_CALIBRATION_PAIRS = 10


@dataclass(frozen=True)
class MotionScales:
    """Typical per-source-frame motion of trusted frames: robust sigmas of pair differences divided by sqrt(gap)."""

    head_px: float
    coefficients: np.ndarray
    pairs: int

    def to_dict(self) -> dict[str, Any]:
        return {"head_px": self.head_px, "coefficients": [float(v) for v in self.coefficients], "pairs": self.pairs}


def chain_angles(points: np.ndarray) -> np.ndarray:
    """Unwrapped tangent angles ``[S]`` of an ordered chain ``[S + 1, 2]``."""

    delta = np.diff(np.asarray(points, dtype=float), axis=0)
    return np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))


def encode_chain(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """``[2 + K]`` head and angle coefficients of a chain (least squares on its tangent angles)."""

    return np.concatenate((np.asarray(points[0], dtype=float), np.linalg.lstsq(basis, chain_angles(points), rcond=None)[0]))


def decode_chains(x: np.ndarray, basis: np.ndarray, step: float) -> np.ndarray:
    """Chains ``[T, S + 1, 2]`` from ``[T, 2 + K]`` heads and coefficients."""

    theta = x[:, 2:] @ basis.T
    links = step * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    return x[:, None, :2] + np.concatenate((np.zeros((len(x), 1, 2)), np.cumsum(links, axis=1)), axis=1)


def initial_chain(curve: np.ndarray, length: float, segments: int) -> np.ndarray:
    """Equal-link chain along ``curve`` from its head at absolute arc distances; a shorter curve continues straight."""

    curve = np.asarray(curve, dtype=float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-9]
    step = length / segments
    distances = np.arange(segments + 1) * step
    supported = int(np.sum(distances <= arc[-1] + 1e-9))
    points = np.empty((segments + 1, 2))
    points[:supported] = np.column_stack([np.interp(distances[:supported], arc[keep], curve[keep, axis]) for axis in (0, 1)])
    if supported < 2:
        direction = curve[-1] - curve[0]
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.array([1., 0.])
        points[0] = curve[0]
        supported = 1
    else:
        direction = points[supported - 1] - points[supported - 2]
        direction /= max(np.linalg.norm(direction), 1e-12)
    for i in range(supported, segments + 1):
        points[i] = points[i - 1] + step * direction
    return points


def motion_scales(chains: np.ndarray, frames: np.ndarray, basis: np.ndarray, *, max_gap: int) -> MotionScales:
    """Robust per-source-frame motion of consecutive fully observed chains ``[M, S + 1, 2]`` at source ``frames``.

    Pairs further apart than ``max_gap`` source frames are skipped.  Each
    difference is divided by the square root of its gap (a random walk over
    the skipped frames) and the sigma is the median absolute difference
    scaled to a Gaussian, per coefficient.  ``ValueError`` below
    ``MIN_CALIBRATION_PAIRS``.
    """

    frames = np.asarray(frames, dtype=np.int64)
    x = np.asarray([encode_chain(points, basis) for points in chains]) if len(chains) else np.zeros((0, 2 + basis.shape[1]))
    gaps = np.diff(frames)
    use = (gaps >= 1) & (gaps <= max_gap)
    pairs = int(use.sum())
    if pairs < MIN_CALIBRATION_PAIRS:
        raise ValueError(
            f"The motion prior needs at least {MIN_CALIBRATION_PAIRS} consecutive pairs of trusted, fully visible frames to calibrate; found {pairs}. "
            "Review more frames or widen the workspace."
        )
    rate = (x[1:][use] - x[:-1][use]) / np.sqrt(gaps[use].astype(float))[:, None]
    head = max(MAD_TO_SIGMA * float(np.median(np.abs(rate[:, :2]))), 0.1)
    coefficients = np.maximum(MAD_TO_SIGMA * np.median(np.abs(rate[:, 2:]), axis=0), 1e-3)
    return MotionScales(head, coefficients, pairs)


@dataclass
class SmoothingProblem:
    """``T`` chain nodes in time order.

    ``targets`` ``[T, S + 1, 2]`` and ``weights`` ``[T, S + 1]`` (zero where a
    node has no target at that link), ``gaps`` ``[T - 1]`` source frames
    between consecutive nodes, ``fixed`` ``[T]`` nodes that keep their
    initial chain (anchors), ``initial`` ``[T, S + 1, 2]`` chains, ``step``
    the link length and ``coefficients`` the angle basis size.
    """

    targets: np.ndarray
    weights: np.ndarray
    gaps: np.ndarray
    fixed: np.ndarray
    initial: np.ndarray
    step: float
    coefficients: int


def _solve_block_tridiagonal(diagonal: np.ndarray, coupling: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``x`` of the symmetric block-tridiagonal system with diagonal blocks ``[T, D, D]``, couplings ``[T - 1, D]`` (diagonal blocks between t-1 and t) and ``rhs`` ``[T, D]``."""

    count = len(diagonal)
    reduced = np.empty_like(diagonal)
    carried = np.empty_like(rhs)
    reduced[0], carried[0] = diagonal[0], rhs[0]
    for t in range(1, count):
        # The coupling is diagonal: M = C_t inv(A'_{t-1}) is a row scaling of the inverse.
        inverse = np.linalg.inv(reduced[t - 1])
        m = coupling[t - 1][:, None] * inverse
        reduced[t] = diagonal[t] - m * coupling[t - 1][None, :]
        carried[t] = rhs[t] - m @ carried[t - 1]
    x = np.empty_like(rhs)
    x[-1] = np.linalg.solve(reduced[-1], carried[-1])
    for t in range(count - 2, -1, -1):
        x[t] = np.linalg.solve(reduced[t], carried[t] - coupling[t] * x[t + 1])
    return x


def smooth_chains(
    problem: SmoothingProblem, scales: MotionScales, *, data_sigma_px: float = 1.0, tolerance: float = 2.0,
    bend_weight: float = 1e-3, maxiter: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Chains ``[T, S + 1, 2]`` minimising data misfit plus the standardised motion prior, and solver information.

    The prior sigmas are ``tolerance`` times the calibrated scales.  A small
    ``bend_weight`` on the squared bend between links suppresses zigzags
    where no data or neighbour constrains the body.
    """

    targets, weights, fixed = np.asarray(problem.targets, dtype=float), np.asarray(problem.weights, dtype=float), np.asarray(problem.fixed, dtype=bool)
    count, links = targets.shape[0], targets.shape[1] - 1
    if count == 0:
        return problem.initial.copy(), {"iterations": 0, "converged": True, "energy": 0.0}
    basis = cubic_bspline_basis(links, problem.coefficients)
    step = float(problem.step)
    dimension = 2 + problem.coefficients
    x = np.asarray([encode_chain(points, basis) for points in problem.initial])
    inverse_data = 1.0 / data_sigma_px ** 2
    prior = np.empty(dimension)
    prior[:2] = 1.0 / (tolerance * scales.head_px) ** 2
    prior[2:] = 1.0 / (tolerance * np.asarray(scales.coefficients, dtype=float)) ** 2
    pair_scale = (prior[None, :] / np.asarray(problem.gaps, dtype=float)[:, None]) if count > 1 else np.zeros((0, dimension))
    difference = np.diff(np.eye(links), axis=0) @ basis
    ridge = bend_weight * difference.T @ difference

    def energy(x: np.ndarray, chains: np.ndarray) -> float:
        value = inverse_data * float((weights[..., None] * (chains - targets) ** 2).sum())
        value += float(np.einsum("tk,kl,tl->", x[:, 2:], ridge, x[:, 2:]))
        if count > 1:
            value += float((pair_scale * np.diff(x, axis=0) ** 2).sum())
        return value

    chains = decode_chains(x, basis, step)
    current = energy(x, chains)
    damping, iterations, converged = 1e-3, 0, False
    for iterations in range(1, maxiter + 1):
        theta = x[:, 2:] @ basis.T
        derivative = np.stack((-np.sin(theta), np.cos(theta)), axis=-1)  # [T, S, 2]
        # Jacobian of point i with respect to the coefficients: step * sum over links before i.
        jac = step * np.cumsum(derivative[:, :, :, None] * basis[None, :, None, :], axis=1)
        jac = np.concatenate((np.zeros((count, 1, 2, problem.coefficients)), jac), axis=1)  # [T, S + 1, 2, K]
        residual = chains - targets
        w = inverse_data * weights
        diagonal = np.zeros((count, dimension, dimension))
        rhs = np.zeros((count, dimension))
        diagonal[:, 0, 0] = diagonal[:, 1, 1] = w.sum(axis=1)
        cross = np.einsum("ti,tiak->tak", w, jac)
        diagonal[:, :2, 2:] = cross
        diagonal[:, 2:, :2] = cross.transpose(0, 2, 1)
        diagonal[:, 2:, 2:] = np.einsum("ti,tiak,tial->tkl", w, jac, jac) + ridge[None]
        rhs[:, :2] = -np.einsum("ti,tia->ta", w, residual)
        rhs[:, 2:] = -np.einsum("ti,tiak,tia->tk", w, jac, residual) - x[:, 2:] @ ridge
        coupling = np.zeros((max(count - 1, 0), dimension))
        if count > 1:
            delta = np.diff(x, axis=0)
            diagonal[1:] += pair_scale[:, :, None] * np.eye(dimension)[None]
            diagonal[:-1] += pair_scale[:, :, None] * np.eye(dimension)[None]
            coupling = -pair_scale
            rhs[1:] -= pair_scale * delta
            rhs[:-1] += pair_scale * delta
            touching = fixed[1:] | fixed[:-1]
            coupling[touching] = 0.0
        diagonal[fixed] = np.eye(dimension)[None]
        rhs[fixed] = 0.0
        accepted = False
        while damping < 1e8:
            damped = diagonal + damping * np.eye(dimension)[None] * np.diagonal(diagonal, axis1=1, axis2=2)[:, None, :]
            update = _solve_block_tridiagonal(damped, coupling, rhs)
            update[fixed] = 0.0
            trial = x + update
            trial_chains = decode_chains(trial, basis, step)
            value = energy(trial, trial_chains)
            if np.isfinite(value) and value <= current:
                improvement = current - value
                x, chains, current = trial, trial_chains, value
                damping = max(damping / 3.0, 1e-9)
                accepted = True
                if improvement <= 1e-9 * max(current, 1.0) or np.abs(update).max() < 1e-7:
                    converged = True
                break
            damping *= 10.0
        if not accepted or converged:
            converged = converged or not accepted
            break
    chains[fixed] = problem.initial[fixed]
    return chains, {"iterations": iterations, "converged": converged, "energy": float(current)}
