"""Fixed low-dimensional intrinsic pose representation selected by EXP-SMC-003."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


FloatArray = NDArray[np.float64]


def cubic_bspline_basis(samples: int = 99, coefficients: int = 16) -> FloatArray:
    """Return a uniform clamped cubic B-spline design matrix."""

    degree = 3
    if samples < 2 or coefficients < degree + 1:
        raise ValueError("cubic splines need at least two samples and four coefficients")
    internal = np.linspace(0.0, 1.0, coefficients - degree + 1)[1:-1]
    knots = np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))
    position = np.linspace(0.0, 1.0, samples)
    basis = np.zeros((samples, len(knots) - 1), dtype=np.float64)
    for index in range(len(knots) - 1):
        basis[:, index] = (position >= knots[index]) & (position < knots[index + 1])
    basis[-1] = 0
    basis[-1, coefficients - 1] = 1
    for order in range(1, degree + 1):
        updated = np.zeros((samples, len(knots) - order - 1), dtype=np.float64)
        for index in range(updated.shape[1]):
            left = knots[index + order] - knots[index]
            right = knots[index + order + 1] - knots[index + 1]
            if left > 0:
                updated[:, index] += (position - knots[index]) / left * basis[:, index]
            if right > 0:
                updated[:, index] += (
                    (knots[index + order + 1] - position) / right * basis[:, index + 1]
                )
        basis = updated
    result = basis[:, :coefficients]
    if not np.allclose(result.sum(1), 1.0, atol=1e-12):
        raise RuntimeError("B-spline basis lost partition of unity")
    return result


def encode_centerline(centerline_xy: NDArray[np.generic], coefficients: int = 16) -> FloatArray:
    """Encode a uniformly sampled centerline as shape, rotation, length, and xy."""

    points = np.asarray(centerline_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape != (100, 2) or not np.isfinite(points).all():
        raise ValueError("centerline_xy must be finite with shape [100,2]")
    difference = np.diff(points, axis=0)
    length = float(np.linalg.norm(difference, axis=1).sum())
    if length <= 0:
        raise ValueError("centerline length must be positive")
    angle = np.unwrap(np.arctan2(difference[:, 1], difference[:, 0]))
    rotation = float(angle.mean())
    basis = cubic_bspline_basis(99, coefficients)
    shape = np.linalg.lstsq(basis, angle - rotation, rcond=None)[0]
    return np.concatenate((shape, [rotation, length], points.mean(0)))


def decode_centerline(latent: NDArray[np.generic], coefficients: int = 16) -> FloatArray:
    """Decode one latent vector to 100 uniformly stepped xy points."""

    values = np.asarray(latent, dtype=np.float64)
    if values.shape != (coefficients + 4,) or not np.isfinite(values).all():
        raise ValueError(f"latent must be finite with shape [{coefficients + 4}]")
    angle = cubic_bspline_basis(99, coefficients) @ values[:coefficients] + values[-4]
    difference = values[-3] / 99 * np.column_stack((np.cos(angle), np.sin(angle)))
    curve = np.vstack((np.zeros(2), np.cumsum(difference, axis=0)))
    return curve - curve.mean(0) + values[-2:]


def decode_centerline_torch(
    latent: Tensor, basis: Tensor | None = None, coefficients: int = 16
) -> Tensor:
    """Differentiably decode ``[D]`` or ``[B,D]`` latent tensors."""

    squeeze = latent.ndim == 1
    values = latent.unsqueeze(0) if squeeze else latent
    if values.ndim != 2 or values.shape[1] != coefficients + 4:
        raise ValueError(f"latent must have shape [{coefficients + 4}] or [B,{coefficients + 4}]")
    if basis is None:
        basis = torch.as_tensor(
            cubic_bspline_basis(99, coefficients), dtype=values.dtype, device=values.device
        )
    else:
        basis = basis.to(dtype=values.dtype, device=values.device)
    angle = values[:, :coefficients] @ basis.T + values[:, -4, None]
    step = values[:, -3, None] / 99
    difference = step[:, :, None] * torch.stack((torch.cos(angle), torch.sin(angle)), dim=2)
    origin = torch.zeros(len(values), 1, 2, dtype=values.dtype, device=values.device)
    curve = torch.cat((origin, difference.cumsum(1)), dim=1)
    curve = curve - curve.mean(1, keepdim=True) + values[:, None, -2:]
    return curve.squeeze(0) if squeeze else curve


def orient_to_reference(
    centerline_xy: NDArray[np.generic], reference_xy: NDArray[np.generic]
) -> tuple[FloatArray, bool]:
    """Choose endpoint order giving the smaller pointwise distance to a reference."""

    current = np.asarray(centerline_xy, dtype=np.float64)
    reference = np.asarray(reference_xy, dtype=np.float64)
    if current.shape != (100, 2) or reference.shape != (100, 2):
        raise ValueError("both centerlines must have shape [100,2]")
    forward = np.mean(np.linalg.norm(current - reference, axis=1))
    reverse = np.mean(np.linalg.norm(current[::-1] - reference, axis=1))
    return (current[::-1].copy(), True) if reverse < forward else (current.copy(), False)


def unwrap_latent_rotation(current: FloatArray, previous: FloatArray) -> FloatArray:
    """Shift global rotation by 2pi to the branch closest to the previous state."""

    result = np.asarray(current, dtype=np.float64).copy()
    reference = float(np.asarray(previous)[-4])
    result[-4] += 2 * np.pi * np.round((reference - result[-4]) / (2 * np.pi))
    return result
