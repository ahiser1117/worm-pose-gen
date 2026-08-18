"""Differentiable geometry for 2-D worm centerlines.

Coordinates are pixel-center coordinates: ``(0, 0)`` is the center of the
upper-left pixel, x points right, and y points down.  Angles therefore increase
clockwise in a displayed image and are always wrapped to ``[-pi, pi)``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def wrap_angle(angle: Tensor | float) -> Tensor:
    """Wrap radians to the half-open interval ``[-pi, pi)``."""

    value = torch.as_tensor(angle)
    return torch.remainder(value + torch.pi, 2 * torch.pi) - torch.pi


def in_fov_mask(centerline_xy: Tensor, image_height: int, image_width: int) -> Tensor:
    """Return centerline-point membership in half-open image bounds.

    This is geometric point membership, not full-tube visibility or usable
    image support.
    """

    if centerline_xy.shape[-1] != 2:
        raise ValueError("centerline_xy must have final dimension 2")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    x, y = centerline_xy.unbind(dim=-1)
    return (x >= 0) & (x < image_width) & (y >= 0) & (y < image_height)


def tangent_angles(centerline_xy: Tensor) -> Tensor:
    """Estimate point tangents using one-sided endpoints and centered interiors."""

    if centerline_xy.shape[-1] != 2 or centerline_xy.shape[-2] < 2:
        raise ValueError("centerline_xy must have shape [..., N>=2, 2]")
    differences = torch.empty_like(centerline_xy)
    differences[..., 0, :] = centerline_xy[..., 1, :] - centerline_xy[..., 0, :]
    differences[..., -1, :] = centerline_xy[..., -1, :] - centerline_xy[..., -2, :]
    if centerline_xy.shape[-2] > 2:
        differences[..., 1:-1, :] = centerline_xy[..., 2:, :] - centerline_xy[..., :-2, :]
    return wrap_angle(torch.atan2(differences[..., 1], differences[..., 0]))


def curvature_from_angles(tangent_angle: Tensor, body_length: Tensor | float) -> Tensor:
    """Finite-difference curvature in radians per original-image pixel.

    Samples are assumed uniformly spaced over ``body_length``. Endpoints use
    one-sided differences and interiors use centered differences. Wrapped
    angle increments avoid discontinuities at ``-pi/pi``. Positive curvature
    turns clockwise because y points down.
    """

    if tangent_angle.shape[-1] < 2:
        raise ValueError("at least two tangent samples are required")
    length = torch.as_tensor(body_length, dtype=tangent_angle.dtype, device=tangent_angle.device)
    if bool(torch.any(length <= 0)):
        raise ValueError("body_length must be positive")
    n = tangent_angle.shape[-1]
    ds = length / (n - 1)
    increments = wrap_angle(tangent_angle[..., 1:] - tangent_angle[..., :-1])
    scaled = increments / ds[..., None]
    result = torch.empty_like(tangent_angle)
    result[..., 0] = scaled[..., 0]
    result[..., -1] = scaled[..., -1]
    if n > 2:
        result[..., 1:-1] = 0.5 * (scaled[..., :-1] + scaled[..., 1:])
    return result


def curvature(centerline_xy: Tensor) -> Tensor:
    """Estimate signed curvature using centerline arc length in pixel units."""

    angles = tangent_angles(centerline_xy)
    segment_length = torch.linalg.vector_norm(
        centerline_xy[..., 1:, :] - centerline_xy[..., :-1, :], dim=-1
    )
    if bool(torch.any(segment_length <= 0)):
        raise ValueError("consecutive centerline points must be distinct")
    increments = wrap_angle(angles[..., 1:] - angles[..., :-1])
    result = torch.empty_like(angles)
    result[..., 0] = increments[..., 0] / segment_length[..., 0]
    result[..., -1] = increments[..., -1] / segment_length[..., -1]
    if angles.shape[-1] > 2:
        span = segment_length[..., :-1] + segment_length[..., 1:]
        result[..., 1:-1] = (increments[..., :-1] + increments[..., 1:]) / span
    return result


def reconstruct_centerline(
    anchor_xy: Tensor,
    tangent_angle: Tensor,
    body_length: Tensor | float,
    *,
    anchor_index: int = 0,
) -> Tensor:
    """Integrate uniformly sampled point tangents into a centerline.

    Trapezoidal integration is used between tangent samples. ``anchor_xy`` is
    placed at ``anchor_index``; the common intrinsic representation uses zero.
    The operation is differentiable with respect to anchor, angles, and length.
    """

    if tangent_angle.shape[-1] < 2:
        raise ValueError("at least two tangent samples are required")
    n = tangent_angle.shape[-1]
    if not -n <= anchor_index < n:
        raise IndexError("anchor_index is outside the centerline")
    anchor_index %= n
    length = torch.as_tensor(body_length, dtype=tangent_angle.dtype, device=tangent_angle.device)
    if bool(torch.any(length <= 0)):
        raise ValueError("body_length must be positive")
    unit = torch.stack((torch.cos(tangent_angle), torch.sin(tangent_angle)), dim=-1)
    step = 0.5 * (unit[..., :-1, :] + unit[..., 1:, :])
    step = step * (length / (n - 1))[..., None, None]
    zero = torch.zeros_like(step[..., :1, :])
    offsets = torch.cat((zero, torch.cumsum(step, dim=-2)), dim=-2)
    offsets = offsets - offsets[..., anchor_index : anchor_index + 1, :]
    return anchor_xy[..., None, :] + offsets


def tangent_from_basis(
    global_orientation: Tensor | float, coefficients: Tensor, basis: Tensor
) -> Tensor:
    """Evaluate ``theta(s) = phi + sum_j c_j B_j(s)``.

    ``coefficients`` has shape ``[..., K]`` and ``basis`` has shape ``[N, K]``.
    The returned angles have shape ``[..., N]``.
    """

    if basis.ndim != 2 or coefficients.shape[-1] != basis.shape[-1]:
        raise ValueError("basis must be [N, K] matching coefficients [..., K]")
    phi = torch.as_tensor(
        global_orientation, dtype=coefficients.dtype, device=coefficients.device
    )
    return wrap_angle(phi[..., None] + coefficients @ basis.transpose(-1, -2))


def reconstruct_from_coefficients(
    anchor_xy: Tensor,
    global_orientation: Tensor | float,
    body_length: Tensor | float,
    coefficients: Tensor,
    basis: Tensor,
    *,
    anchor_index: int = 0,
) -> tuple[Tensor, Tensor]:
    """Evaluate a compact tangent basis and reconstruct its centerline."""

    angles = tangent_from_basis(global_orientation, coefficients, basis)
    return (
        reconstruct_centerline(anchor_xy, angles, body_length, anchor_index=anchor_index),
        angles,
    )


def resample_centerline(centerline_xy: Tensor, num_points: int) -> Tensor:
    """Linearly resample a polyline at uniform arc-length positions.

    Supports arbitrary leading batch dimensions and preserves gradients through
    the selected interpolation segments.
    """

    if centerline_xy.shape[-1] != 2 or centerline_xy.shape[-2] < 2:
        raise ValueError("centerline_xy must have shape [..., N>=2, 2]")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    original_shape = centerline_xy.shape
    flat = centerline_xy.reshape(-1, original_shape[-2], 2)
    segment = torch.linalg.vector_norm(flat[:, 1:] - flat[:, :-1], dim=-1)
    if bool(torch.any(segment <= 0)):
        raise ValueError("consecutive centerline points must be distinct")
    cumulative = torch.cat((torch.zeros_like(segment[:, :1]), torch.cumsum(segment, 1)), 1)
    fraction = torch.linspace(0, 1, num_points, dtype=flat.dtype, device=flat.device)
    target = cumulative[:, -1:] * fraction
    upper = torch.searchsorted(cumulative.contiguous(), target.contiguous(), right=True)
    upper = upper.clamp(1, original_shape[-2] - 1)
    lower = upper - 1
    s0 = torch.gather(cumulative, 1, lower)
    s1 = torch.gather(cumulative, 1, upper)
    p0 = torch.gather(flat, 1, lower[..., None].expand(-1, -1, 2))
    p1 = torch.gather(flat, 1, upper[..., None].expand(-1, -1, 2))
    weight = ((target - s0) / (s1 - s0))[..., None]
    result = p0 + weight * (p1 - p0)
    return result.reshape(*original_shape[:-2], num_points, 2)


def canonicalize_orientation(
    centerline_xy: Tensor,
    index0_head_probability: Tensor | float,
    *,
    tangent_angle: Tensor | None = None,
    curvature_values: Tensor | None = None,
    **body_fields: Tensor,
) -> dict[str, Any]:
    """Export the more probable head-to-tail orientation.

    ``index0_head_probability`` refers to the input ordering. Samples below
    0.5 are reversed. The returned ``head_tail_probability`` is confidence that
    exported index 0 is the head and is therefore in ``[0.5, 1]``. Tangents
    gain pi when reversed and curvature changes sign. Extra body fields are
    simply reversed along their final (body) dimension.
    """

    probability = torch.as_tensor(
        index0_head_probability, dtype=centerline_xy.dtype, device=centerline_xy.device
    )
    if bool(torch.any((probability < 0) | (probability > 1))):
        raise ValueError("index0_head_probability must lie in [0, 1]")
    reverse = probability < 0.5

    def choose(value: Tensor, reversed_value: Tensor) -> Tensor:
        condition = reverse
        while condition.ndim < value.ndim:
            condition = condition.unsqueeze(-1)
        return torch.where(condition, reversed_value, value)

    result: dict[str, Any] = {
        "centerline_xy": choose(centerline_xy, centerline_xy.flip(-2)),
        "head_tail_probability": torch.maximum(probability, 1 - probability),
        "reversed": reverse,
    }
    if tangent_angle is not None:
        result["tangent_angle"] = choose(
            wrap_angle(tangent_angle), wrap_angle(tangent_angle.flip(-1) + torch.pi)
        )
    if curvature_values is not None:
        result["curvature"] = choose(curvature_values, -curvature_values.flip(-1))
    for name, value in body_fields.items():
        result[name] = choose(value, value.flip(-1))
    return result
