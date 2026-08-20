"""Training-free differentiable centerline refinement for EXP-008.

The optimizer updates an intrinsic pose around an arbitrary initialization:
anchor translation, global rotation, log length, and a smooth tangent-angle
basis.  The image objective is evaluated only on instantiated camera pixels,
so anatomy outside the FOV is naturally censored rather than penalized as
missing foreground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .geometry import reconstruct_centerline
from .model import smooth_tangent_basis
from .renderer import render_worm


RefinementObjective = Literal["pixel", "pixel_gradient", "tube_likelihood"]


@dataclass(frozen=True)
class RefinementConfig:
    image_height: int = 192
    image_width: int = 256
    anchor_index: int = 50
    coefficients: int = 16
    edge_softness: float = 0.8
    # Adam's first update is approximately the learning rate regardless of
    # gradient scale.  These are therefore explicit per-step trust radii, small
    # enough not to throw an already-good proposal out of the image basin.
    anchor_learning_rate: float = 0.35
    angle_learning_rate: float = 0.02
    length_learning_rate: float = 0.005
    shape_learning_rate: float = 0.02
    update_regularization: float = 1e-4


class RefinablePose(nn.Module):
    """Intrinsic pose updates around a supplied initialization."""

    def __init__(
        self,
        initial_anchor_xy: Tensor,
        initial_tangent_angle: Tensor,
        initial_body_length: Tensor,
        *,
        config: RefinementConfig = RefinementConfig(),
    ) -> None:
        super().__init__()
        if initial_tangent_angle.ndim != 2 or initial_tangent_angle.shape[1] < 2:
            raise ValueError("initial_tangent_angle must have shape [B,N>=2]")
        batch, count = initial_tangent_angle.shape
        if initial_anchor_xy.shape != (batch, 2) or initial_body_length.shape != (batch,):
            raise ValueError("initial anchor/length shapes must be [B,2] and [B]")
        if config.anchor_index < 0 or config.anchor_index >= count:
            raise ValueError("anchor_index lies outside the body")
        if bool(torch.any(initial_body_length <= 0)):
            raise ValueError("initial body lengths must be positive")
        self.config = config
        self.register_buffer("initial_anchor_xy", initial_anchor_xy.detach().clone())
        self.register_buffer("initial_tangent_angle", initial_tangent_angle.detach().clone())
        self.register_buffer("initial_body_length", initial_body_length.detach().clone())
        self.register_buffer(
            "basis",
            smooth_tangent_basis(count, config.coefficients).to(
                dtype=initial_tangent_angle.dtype, device=initial_tangent_angle.device
            ),
        )
        self.anchor_delta = nn.Parameter(torch.zeros_like(initial_anchor_xy))
        self.angle_delta = nn.Parameter(torch.zeros(batch, dtype=initial_tangent_angle.dtype, device=initial_tangent_angle.device))
        self.log_length_delta = nn.Parameter(torch.zeros_like(initial_body_length))
        self.shape_delta = nn.Parameter(
            torch.zeros(batch, config.coefficients, dtype=initial_tangent_angle.dtype, device=initial_tangent_angle.device)
        )

    def pose(self) -> dict[str, Tensor]:
        tangent = (
            self.initial_tangent_angle
            + self.angle_delta[:, None]
            + self.shape_delta @ self.basis.transpose(0, 1)
        )
        length = self.initial_body_length * torch.exp(self.log_length_delta)
        anchor = self.initial_anchor_xy + self.anchor_delta
        centerline = reconstruct_centerline(
            anchor, tangent, length, anchor_index=self.config.anchor_index
        )
        return {
            "centerline_xy": centerline,
            "tangent_angle": tangent,
            "body_length": length,
            "anchor_xy": anchor,
        }

    def regularization(self) -> Tensor:
        return self.config.update_regularization * (
            self.angle_delta.square().mean()
            + self.log_length_delta.square().mean()
            + self.shape_delta.square().mean()
            + 1e-2 * self.anchor_delta.square().mean()
        )

    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            [
                {"params": [self.anchor_delta], "lr": self.config.anchor_learning_rate},
                {"params": [self.angle_delta], "lr": self.config.angle_learning_rate},
                {"params": [self.log_length_delta], "lr": self.config.length_learning_rate},
                {"params": [self.shape_delta], "lr": self.config.shape_learning_rate},
            ]
        )


def _robust_residual(first: Tensor, second: Tensor) -> Tensor:
    return torch.sqrt((first - second).square() + 1e-6)


def refinement_image_loss(
    rendered_image: Tensor,
    target_image: Tensor,
    *,
    objective: RefinementObjective,
    rendered_mask: Tensor | None = None,
    target_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate one of the preregistered EXP-008 image objectives."""

    if rendered_image.shape != target_image.shape or rendered_image.ndim != 3:
        raise ValueError("rendered_image and target_image must share shape [B,H,W]")
    pixel = _robust_residual(rendered_image, target_image)
    if objective == "pixel":
        return pixel.mean()
    if objective == "pixel_gradient":
        dx = _robust_residual(
            rendered_image[:, :, 1:] - rendered_image[:, :, :-1],
            target_image[:, :, 1:] - target_image[:, :, :-1],
        ).mean()
        dy = _robust_residual(
            rendered_image[:, 1:, :] - rendered_image[:, :-1, :],
            target_image[:, 1:, :] - target_image[:, :-1, :],
        ).mean()
        return 0.5 * pixel.mean() + 0.25 * (dx + dy)
    if objective == "tube_likelihood":
        if rendered_mask is None or target_mask is None:
            raise ValueError("tube_likelihood requires rendered_mask and target_mask")
        if rendered_mask.shape != rendered_image.shape or target_mask.shape != target_image.shape:
            raise ValueError("tube masks must match image shapes")
        weight = torch.maximum(rendered_mask.detach(), target_mask.detach()) + 0.05
        return (pixel * weight).sum() / weight.sum()
    raise ValueError(f"unknown refinement objective {objective!r}")


def refine_pose(
    initial_anchor_xy: Tensor,
    initial_tangent_angle: Tensor,
    initial_body_length: Tensor,
    width: Tensor,
    target_image: Tensor,
    *,
    target_mask: Tensor | None = None,
    objective: RefinementObjective = "pixel",
    steps: int = 5,
    record_steps: tuple[int, ...] = (0, 1, 3, 5, 10),
    config: RefinementConfig = RefinementConfig(),
) -> tuple[RefinablePose, dict[int, dict[str, Tensor | float]]]:
    """Optimize a batch and retain detached poses at requested step counts."""

    if steps < 0 or any(value < 0 or value > steps for value in record_steps):
        raise ValueError("record_steps must lie in [0, steps]")
    module = RefinablePose(
        initial_anchor_xy, initial_tangent_angle, initial_body_length, config=config
    )
    target_image = target_image.detach()
    target_mask = target_mask.detach() if target_mask is not None else None
    width = width.detach()
    optimizer = module.optimizer()
    history: dict[int, dict[str, Tensor | float]] = {}

    def snapshot(step: int, loss: float | None = None) -> None:
        pose = module.pose()
        history[step] = {
            name: value.detach().clone() for name, value in pose.items()
        }
        history[step]["loss"] = float("nan") if loss is None else loss

    if 0 in record_steps:
        snapshot(0)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pose = module.pose()
        rendered = render_worm(
            pose["centerline_xy"], width, config.image_height, config.image_width,
            edge_softness=config.edge_softness,
        )
        loss = refinement_image_loss(
            rendered["image"], target_image, objective=objective,
            rendered_mask=rendered["tube_mask"], target_mask=target_mask,
        ) + module.regularization()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite refinement loss")
        loss.backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in module.parameters()
        ):
            raise RuntimeError("non-finite refinement gradient")
        optimizer.step()
        if step in record_steps:
            snapshot(step, float(loss.detach()))
    return module, history
