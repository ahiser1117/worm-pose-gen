"""Training losses for orientation-ambiguous centerlines."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .geometry import tangent_angles, wrap_angle


def symmetric_point_loss(
    prediction_xy: Tensor, target_xy: Tensor, *, reduction: str = "mean"
) -> Tensor:
    """Smooth-L1 loss against the better of target forward/reverse order.

    The minimum is selected independently for each batch item. This encodes the
    experiment's unvalidated head/tail truth rather than silently assigning an
    anatomical orientation.
    """

    if prediction_xy.shape != target_xy.shape or prediction_xy.shape[-1] != 2:
        raise ValueError("prediction and target must have identical [..., N, 2] shape")
    forward = F.smooth_l1_loss(prediction_xy, target_xy, reduction="none").mean((-2, -1))
    reverse = F.smooth_l1_loss(
        prediction_xy, target_xy.flip(-2), reduction="none"
    ).mean((-2, -1))
    result = torch.minimum(forward, reverse)
    if reduction == "none":
        return result
    if reduction == "mean":
        return result.mean()
    if reduction == "sum":
        return result.sum()
    raise ValueError(f"unsupported reduction {reduction!r}")


def symmetric_tangent_loss(
    prediction_xy: Tensor, target_xy: Tensor, *, reduction: str = "mean"
) -> Tensor:
    """Circular tangent-angle loss with forward/reverse symmetry."""

    predicted = tangent_angles(prediction_xy)
    target = tangent_angles(target_xy)
    forward = (1.0 - torch.cos(wrap_angle(predicted - target))).mean(-1)
    reversed_target = wrap_angle(target.flip(-1) + torch.pi)
    reverse = (1.0 - torch.cos(wrap_angle(predicted - reversed_target))).mean(-1)
    result = torch.minimum(forward, reverse)
    if reduction == "none":
        return result
    if reduction == "mean":
        return result.mean()
    if reduction == "sum":
        return result.sum()
    raise ValueError(f"unsupported reduction {reduction!r}")


def proposal_loss(
    output: dict[str, Tensor],
    target_xy: Tensor,
    image_support_target: Tensor,
    *,
    image_height: int = 192,
    image_width: int = 256,
) -> dict[str, Tensor]:
    """Return jointly orientation-selected total and component losses.

    Tangents are evaluated in the original 968x732 pixel geometry. The same
    per-example orientation is used for points, tangents, and anatomical image
    support; support is reversed whenever geometry is reversed.
    """

    scale = target_xy.new_tensor((image_width, image_height))
    prediction = output["centerline_xy"]
    forward_point = F.smooth_l1_loss(prediction / scale, target_xy / scale, reduction="none").mean((-2, -1))
    reverse_point = F.smooth_l1_loss(prediction / scale, target_xy.flip(-2) / scale, reduction="none").mean((-2, -1))

    original_scale = target_xy.new_tensor((968 / image_width, 732 / image_height))
    predicted_angle = tangent_angles(prediction * original_scale)
    target_angle = tangent_angles(target_xy * original_scale)
    forward_angle = (1 - torch.cos(wrap_angle(predicted_angle - target_angle))).mean(-1)
    reverse_angle = (
        1 - torch.cos(wrap_angle(predicted_angle - wrap_angle(target_angle.flip(-1) + torch.pi)))
    ).mean(-1)

    target_support = image_support_target.to(target_xy.dtype)
    forward_support = F.binary_cross_entropy_with_logits(
        output["image_support_logits"], target_support, reduction="none"
    ).mean(-1)
    reverse_support = F.binary_cross_entropy_with_logits(
        output["image_support_logits"], target_support.flip(-1), reduction="none"
    ).mean(-1)
    forward_total = forward_point + 0.10 * forward_angle + 0.05 * forward_support
    reverse_total = reverse_point + 0.10 * reverse_angle + 0.05 * reverse_support
    reverse = reverse_total < forward_total

    def select(forward: Tensor, backward: Tensor) -> Tensor:
        return torch.where(reverse, backward, forward).mean()

    point = select(forward_point, reverse_point)
    angle = select(forward_angle, reverse_angle)
    support = select(forward_support, reverse_support)
    total = select(forward_total, reverse_total)
    return {
        "loss": total,
        "point_loss": point,
        "angle_loss": angle,
        "support_loss": support,
        "reversed_fraction": reverse.to(target_xy.dtype).mean(),
    }
