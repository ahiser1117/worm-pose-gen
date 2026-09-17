"""Head-specific priors and feasible head positions for the batched fitter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class HeadConstraint:
    """Image XY targets for centerline point 0, with an optional hard step limit.

    The step limit is already scaled to the elapsed source frames by the
    caller. A missing tracking observation contributes no tracking penalty.
    """

    tracking_xy: np.ndarray | None = None
    previous_xy: np.ndarray | None = None
    tracking_weight: float = 0.15
    previous_weight: float = 0.5
    sigma_px: float = 10.0
    max_step_px: float | None = None
    keep_in_frame: bool = True


class HeadPriors:
    """Vectorized penalties and projection, one constraint per optimizer row."""

    def __init__(self, constraints: Sequence[HeadConstraint | None], camera_size: Tensor):
        device, dtype = camera_size.device, camera_size.dtype
        tracking, previous, tracking_scale, previous_scale, radii, inside = [], [], [], [], [], []
        for constraint in constraints:
            c = constraint or HeadConstraint(tracking_weight=0, previous_weight=0, keep_in_frame=False)
            for name, value in (("sigma_px", c.sigma_px), ("tracking_weight", c.tracking_weight), ("previous_weight", c.previous_weight)):
                if not math.isfinite(value) or value < 0 or (name == "sigma_px" and value == 0):
                    raise ValueError(f"head {name} must be finite and {'positive' if name == 'sigma_px' else 'non-negative'}")
            if c.max_step_px is not None and (not math.isfinite(c.max_step_px) or c.max_step_px < 0 or c.previous_xy is None):
                raise ValueError("head max_step_px requires a previous head and a finite non-negative distance")
            for name, point, output in (("tracking_xy", c.tracking_xy, tracking), ("previous_xy", c.previous_xy, previous)):
                xy = np.zeros(2) if point is None else np.asarray(point, dtype=np.float32)
                if xy.shape != (2,) or not np.isfinite(xy).all():
                    raise ValueError(f"head {name} must be a finite XY point")
                output.append(xy)
            tracking_scale.append(0 if c.tracking_xy is None else c.tracking_weight / c.sigma_px**2)
            previous_scale.append(0 if c.previous_xy is None else c.previous_weight / c.sigma_px**2)
            radii.append(float("inf") if c.max_step_px is None else c.max_step_px)
            inside.append(c.keep_in_frame)
        def tensor(values):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)
        self.tracking = tensor(tracking)
        self.previous = tensor(previous)
        self.tracking_scale = tensor(tracking_scale)
        self.previous_scale = tensor(previous_scale)
        self.radius = tensor(radii)
        self.inside = torch.as_tensor(inside, dtype=torch.bool, device=device)
        self.upper = camera_size - 1
        nearest = self.previous.clamp_min(0).minimum(self.upper)
        impossible = self.inside & (torch.linalg.vector_norm(nearest - self.previous, dim=1) > self.radius + 1e-5)
        if bool(impossible.any()):
            raise ValueError("previous head is too far outside the image for the head movement limit; choose an in-frame anchor or increase the limit")

    def energy(self, head: Tensor) -> Tensor:
        return ((head - self.tracking).square().sum(1) * self.tracking_scale
                + (head - self.previous).square().sum(1) * self.previous_scale)

    def project(self, head: Tensor) -> Tensor:
        """A point in the camera/step-disk intersection, including edge anchors.

        Start in the camera rectangle. If outside the movement disk, move
        along the segment to the rectangle's nearest point to the previous
        head. Both ends are in the rectangle, so the disk intersection is
        feasible even when the previous head was outside the image.
        """
        boxed = head.clamp_min(0).minimum(self.upper)
        point = torch.where(self.inside[:, None], boxed, head)
        nearest = self.previous.clamp_min(0).minimum(self.upper)
        base = torch.where(self.inside[:, None], nearest, self.previous)
        bounded = torch.isfinite(self.radius)
        outside = bounded & (torch.linalg.vector_norm(point - self.previous, dim=1) > self.radius)
        direction = point - base
        offset = base - self.previous
        a = direction.square().sum(1).clamp_min(1e-12)
        b = (offset * direction).sum(1)
        radius = torch.where(bounded, self.radius, torch.zeros_like(self.radius))
        c = offset.square().sum(1) - radius.square()
        t = ((-b + (b.square() - a * c).clamp_min(0).sqrt()) / a).clamp(0, 1)
        return torch.where(outside[:, None], base + t[:, None] * direction, point)
