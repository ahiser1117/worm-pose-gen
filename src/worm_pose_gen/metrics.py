"""Evaluation metrics and controlled support/FOV censoring utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .geometry import in_fov_mask, wrap_angle


def circular_angle_error(prediction: Tensor, target: Tensor) -> Tensor:
    """Absolute shortest angular error in radians."""

    return torch.abs(wrap_angle(prediction - target))


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(dtype=torch.bool, device=values.device)
    try:
        selected = torch.masked_select(values, torch.broadcast_to(mask, values.shape))
    except RuntimeError as error:
        raise ValueError("mask is not broadcastable to metric values") from error
    if selected.numel() == 0:
        raise ValueError("metric mask selects no values")
    return selected.mean()


def circular_angle_mae(
    prediction: Tensor, target: Tensor, mask: Tensor | None = None, *, degrees: bool = False
) -> Tensor:
    """Masked circular MAE, in radians by default or degrees on request."""

    value = _masked_mean(circular_angle_error(prediction, target), mask)
    return torch.rad2deg(value) if degrees else value


def point_errors(prediction_xy: Tensor, target_xy: Tensor) -> Tensor:
    """Per-point Euclidean centerline errors in pixels."""

    if prediction_xy.shape != target_xy.shape or prediction_xy.shape[-1] != 2:
        raise ValueError("prediction_xy and target_xy must share shape [..., N, 2]")
    return torch.linalg.vector_norm(prediction_xy - target_xy, dim=-1)


def masked_point_mae(
    prediction_xy: Tensor,
    target_xy: Tensor,
    mask: Tensor | None = None,
    *,
    normalization: Tensor | float | None = None,
) -> Tensor:
    """Mean point distance, optionally normalized by width or body length."""

    error = point_errors(prediction_xy, target_xy)
    if normalization is not None:
        scale = torch.as_tensor(normalization, dtype=error.dtype, device=error.device)
        if bool(torch.any(scale <= 0)):
            raise ValueError("normalization must be positive")
        while scale.ndim < error.ndim:
            scale = scale.unsqueeze(-1)
        error = error / scale
    return _masked_mean(error, mask)


def binary_brier_score(probability: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared probability error for image-support/orientation targets."""

    if bool(torch.any((probability < 0) | (probability > 1))):
        raise ValueError("probabilities must lie in [0, 1]")
    return _masked_mean((probability - target.to(probability.dtype)).square(), mask)


def binary_calibration_bins(
    probability: Tensor, target: Tensor, *, num_bins: int = 10, mask: Tensor | None = None
) -> dict[str, Tensor]:
    """Return fixed-width reliability-bin counts, confidence, and accuracy."""

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if bool(torch.any((probability < 0) | (probability > 1))):
        raise ValueError("probabilities must lie in [0, 1]")
    p, y = probability.reshape(-1), target.to(probability.dtype).reshape(-1)
    if mask is not None:
        selected = torch.broadcast_to(mask, probability.shape).reshape(-1).bool()
        p, y = p[selected], y[selected]
    bin_index = torch.clamp((p * num_bins).long(), max=num_bins - 1)
    count = torch.bincount(bin_index, minlength=num_bins)
    confidence_sum = torch.zeros(num_bins, dtype=p.dtype, device=p.device).scatter_add_(0, bin_index, p)
    accuracy_sum = torch.zeros_like(confidence_sum).scatter_add_(0, bin_index, y)
    denominator = count.clamp_min(1).to(p.dtype)
    nan = torch.full_like(confidence_sum, torch.nan)
    confidence = torch.where(count > 0, confidence_sum / denominator, nan)
    accuracy = torch.where(count > 0, accuracy_sum / denominator, nan)
    return {"count": count, "confidence": confidence, "accuracy": accuracy}


def expected_calibration_error(bins: dict[str, Tensor]) -> Tensor:
    """Count-weighted absolute calibration gap from ``binary_calibration_bins``."""

    count = bins["count"]
    if int(count.sum()) == 0:
        raise ValueError("calibration bins contain no samples")
    occupied = count > 0
    weight = count[occupied] / count.sum()
    return (weight * torch.abs(bins["confidence"][occupied] - bins["accuracy"][occupied])).sum()


def anatomical_support_mask(
    num_points: int, hidden_fraction: float, *, hidden_end: str = "tail", device: torch.device | None = None
) -> Tensor:
    """Controlled anatomical support mask for synthetic head/tail censoring.

    The exact hidden count is ``round(hidden_fraction * num_points)``. This
    utility defines evaluator support independently of a model prediction.
    """

    if num_points < 2 or not 0 <= hidden_fraction <= 1:
        raise ValueError("num_points >= 2 and hidden_fraction in [0, 1] are required")
    if hidden_end not in {"head", "tail"}:
        raise ValueError("hidden_end must be 'head' or 'tail'")
    hidden = round(hidden_fraction * num_points)
    support = torch.ones(num_points, dtype=torch.bool, device=device)
    if hidden:
        if hidden_end == "head":
            support[:hidden] = False
        else:
            support[-hidden:] = False
    return support


def support_regions(support_mask: Tensor, boundary_points: int = 1) -> dict[str, Tensor]:
    """Split a 1-D support target into visible, hidden, and boundary strata."""

    if support_mask.ndim != 1 or boundary_points < 0:
        raise ValueError("support_mask must be 1-D and boundary_points nonnegative")
    support = support_mask.bool()
    transitions = torch.zeros_like(support)
    transitions[1:] |= support[1:] != support[:-1]
    transitions[:-1] |= support[1:] != support[:-1]
    boundary = transitions.clone()
    for _ in range(boundary_points):
        expanded = boundary.clone()
        expanded[1:] |= boundary[:-1]
        expanded[:-1] |= boundary[1:]
        boundary = expanded
    return {"visible": support & ~boundary, "hidden": ~support & ~boundary, "boundary": boundary}


@dataclass(frozen=True)
class FOVCropTransform:
    """Axis-aligned half-open crop, preserving exact original-coordinate mapping."""

    x0: int
    y0: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("crop dimensions must be positive")

    def to_crop(self, points_xy: Tensor) -> Tensor:
        offset = points_xy.new_tensor((self.x0, self.y0))
        return points_xy - offset

    def to_original(self, points_xy: Tensor) -> Tensor:
        offset = points_xy.new_tensor((self.x0, self.y0))
        return points_xy + offset

    def support_mask(self, original_points_xy: Tensor) -> Tensor:
        return in_fov_mask(self.to_crop(original_points_xy), self.height, self.width)

    def as_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "width": self.width, "height": self.height}
