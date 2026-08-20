"""Mask-space observation energies for generative pose inference."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _batched(values: Tensor) -> tuple[Tensor, bool]:
    if values.ndim == 2:
        return values.unsqueeze(0), True
    if values.ndim == 3:
        return values, False
    raise ValueError("mask tensors must have shape [H,W] or [B,H,W]")


def balanced_soft_bce_energy(rendered: Tensor, probability: Tensor) -> Tensor:
    """Foreground/background-balanced Bernoulli cross-entropy per batch item."""

    prediction, squeeze = _batched(rendered)
    target, _ = _batched(
        probability.to(dtype=prediction.dtype, device=prediction.device)
    )
    target = torch.broadcast_to(target, prediction.shape)
    if bool(torch.any((prediction < 0) | (prediction > 1))):
        raise ValueError("rendered occupancy must lie in [0,1]")
    if bool(torch.any((target < 0) | (target > 1))):
        raise ValueError("probability must lie in [0,1]")
    eps = torch.finfo(prediction.dtype).eps
    clipped = prediction.clamp(eps, 1 - eps)
    foreground_mass = target.sum((-2, -1)).clamp_min(eps)
    background_mass = (1 - target).sum((-2, -1)).clamp_min(eps)
    foreground = -(target * torch.log(clipped)).sum((-2, -1)) / foreground_mass
    background = -(
        (1 - target) * torch.log1p(-clipped)
    ).sum((-2, -1)) / background_mass
    result = 0.5 * (foreground + background)
    return result.squeeze(0) if squeeze else result


def soft_dice_energy(rendered: Tensor, target_mask: Tensor) -> Tensor:
    """One minus soft Dice overlap per batch item."""

    prediction, squeeze = _batched(rendered)
    target, _ = _batched(target_mask.to(dtype=prediction.dtype, device=prediction.device))
    target = torch.broadcast_to(target, prediction.shape)
    eps = torch.finfo(prediction.dtype).eps
    intersection = (prediction * target).sum((-2, -1))
    denominator = prediction.sum((-2, -1)) + target.sum((-2, -1))
    result = 1 - (2 * intersection + eps) / (denominator + eps)
    return result.squeeze(0) if squeeze else result


def signed_distance_from_mask(mask: Tensor, *, chunk_pixels: int = 4096) -> Tensor:
    """Pixel-center distance to a morphological boundary band, positive inside."""

    values = mask.to(dtype=torch.bool)
    if values.ndim != 2:
        raise ValueError("mask must have shape [H,W]")
    if chunk_pixels < 1:
        raise ValueError("chunk_pixels must be positive")
    if not bool(values.any()) or bool(values.all()):
        raise ValueError("mask must contain foreground and background")
    floating = values.to(dtype=torch.float32)
    dilated = F.max_pool2d(floating[None, None], 3, stride=1, padding=1)[0, 0]
    eroded = -F.max_pool2d(-floating[None, None], 3, stride=1, padding=1)[0, 0]
    boundary = (dilated - eroded) > 0
    boundary_yx = torch.nonzero(boundary, as_tuple=False).to(dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.arange(values.shape[0], device=values.device),
        torch.arange(values.shape[1], device=values.device),
        indexing="ij",
    )
    pixels = torch.stack((yy, xx), -1).reshape(-1, 2).to(dtype=torch.float32)
    distances: list[Tensor] = []
    for start in range(0, len(pixels), chunk_pixels):
        distances.append(
            torch.cdist(pixels[start : start + chunk_pixels], boundary_yx).min(1).values
        )
    magnitude = torch.cat(distances).reshape(values.shape)
    return torch.where(values, magnitude, -magnitude)


def signed_distance_energy(
    rendered: Tensor,
    target_signed_distance: Tensor,
    *,
    edge_softness: float,
    clip_distance: float = 20.0,
) -> Tensor:
    """Smooth-L1 energy between rendered and observed clipped signed distances."""

    prediction, squeeze = _batched(rendered)
    target, _ = _batched(
        target_signed_distance.to(dtype=prediction.dtype, device=prediction.device)
    )
    target = torch.broadcast_to(target, prediction.shape)
    if edge_softness <= 0 or clip_distance <= 0:
        raise ValueError("edge_softness and clip_distance must be positive")
    eps = torch.finfo(prediction.dtype).eps
    rendered_distance = edge_softness * torch.logit(prediction.clamp(eps, 1 - eps))
    first = rendered_distance.clamp(-clip_distance, clip_distance) / clip_distance
    second = target.clamp(-clip_distance, clip_distance) / clip_distance
    result = F.smooth_l1_loss(first, second, reduction="none").mean((-2, -1))
    return result.squeeze(0) if squeeze else result


def hybrid_mask_energy(
    rendered: Tensor,
    target_mask: Tensor,
    target_signed_distance: Tensor,
    *,
    edge_softness: float,
    signed_distance_weight: float = 0.25,
    clip_distance: float = 20.0,
) -> Tensor:
    """Dice plus a small clipped signed-distance term."""

    if signed_distance_weight < 0:
        raise ValueError("signed_distance_weight must be non-negative")
    return soft_dice_energy(rendered, target_mask) + signed_distance_weight * signed_distance_energy(
        rendered,
        target_signed_distance,
        edge_softness=edge_softness,
        clip_distance=clip_distance,
    )
