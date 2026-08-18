"""Small differentiable, batched soft renderer for worm-shaped tubes."""

from __future__ import annotations

import torch
from torch import Tensor

from .geometry import in_fov_mask


def render_worm(
    centerline_xy: Tensor,
    width: Tensor | float,
    image_height: int,
    image_width: int,
    *,
    foreground: Tensor | float = 0.22,
    background: Tensor | float = 0.78,
    edge_softness: float = 0.8,
    illumination_gradient: Tensor | None = None,
    noise: Tensor | None = None,
    image_support_target: Tensor | None = None,
) -> dict[str, Tensor]:
    """Render grayscale image, soft tube mask, and anatomical support targets.

    Inputs are ``[B,N,2]`` (or ``[N,2]``). Width may be scalar, ``[B]``,
    ``[N]``, or ``[B,N]`` and denotes full tube diameter in raster pixels.
    The distance field is evaluated against centerline samples, which is smooth
    and sufficient at the dense 100-point sampling used here. Pixels outside
    the FOV are never instantiated, providing mandatory image-loss censoring.
    """

    if image_height <= 0 or image_width <= 0 or edge_softness <= 0:
        raise ValueError("positive image dimensions and edge_softness are required")
    squeeze = centerline_xy.ndim == 2
    points = centerline_xy.unsqueeze(0) if squeeze else centerline_xy
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("centerline_xy must be [N,2] or [B,N,2]")
    batch, count, _ = points.shape
    diameter = torch.as_tensor(width, dtype=points.dtype, device=points.device)
    if diameter.ndim == 0:
        diameter = diameter.expand(batch, count)
    elif diameter.ndim == 1 and diameter.shape[0] == count:
        diameter = diameter.unsqueeze(0).expand(batch, -1)
    elif diameter.ndim == 1 and diameter.shape[0] == batch:
        diameter = diameter[:, None].expand(-1, count)
    else:
        diameter = torch.broadcast_to(diameter, (batch, count))
    if bool(torch.any(diameter <= 0)):
        raise ValueError("width must be positive")

    yy, xx = torch.meshgrid(
        torch.arange(image_height, dtype=points.dtype, device=points.device),
        torch.arange(image_width, dtype=points.dtype, device=points.device), indexing="ij"
    )
    pixels = torch.stack((xx, yy), -1).reshape(1, -1, 1, 2)
    squared = (pixels - points[:, None, :, :]).square().sum(-1)
    # Nearest sample also selects its local anatomical diameter. The min is
    # piecewise differentiable and yields useful finite pose/width gradients.
    min_squared, nearest = squared.min(-1)
    local_diameter = torch.gather(diameter, 1, nearest)
    distance = torch.sqrt(min_squared + torch.finfo(points.dtype).eps)
    mask = torch.sigmoid((0.5 * local_diameter - distance) / edge_softness)
    mask = mask.reshape(batch, image_height, image_width)

    fg = torch.as_tensor(foreground, dtype=points.dtype, device=points.device).reshape(-1, 1, 1)
    bg = torch.as_tensor(background, dtype=points.dtype, device=points.device).reshape(-1, 1, 1)
    if fg.shape[0] == 1:
        fg = fg.expand(batch, -1, -1)
    if bg.shape[0] == 1:
        bg = bg.expand(batch, -1, -1)
    image = bg + (fg - bg) * mask
    if illumination_gradient is not None:
        gradient = torch.broadcast_to(illumination_gradient, (batch, 2))
        normalized_x = xx / max(image_width - 1, 1) - 0.5
        normalized_y = yy / max(image_height - 1, 1) - 0.5
        image = image + gradient[:, 0, None, None] * normalized_x + gradient[:, 1, None, None] * normalized_y
    if noise is not None:
        image = image + torch.broadcast_to(noise, image.shape)
    geometric_support = in_fov_mask(points, image_height, image_width)
    if image_support_target is None:
        support_target = geometric_support
    else:
        support_target = torch.broadcast_to(
            image_support_target.to(dtype=torch.bool, device=points.device), geometric_support.shape
        )
    result = {
        "image": image.clamp(0, 1),
        "tube_mask": mask,
        "in_fov_mask": geometric_support,
        "image_support_target": support_target,
        "observable_pixel_mask": torch.ones_like(mask, dtype=torch.bool),
    }
    if squeeze:
        result = {name: value.squeeze(0) for name, value in result.items()}
    return result
