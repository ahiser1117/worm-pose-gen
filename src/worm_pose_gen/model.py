"""Compact Lightning proposal models for EXP-0004 and EXP-0007.

Inference batch contract: ``model(images)`` accepts float images shaped
``[B, 1, 192, 256]`` in the closed interval [0, 1]. It returns a dictionary
containing ``centerline_xy`` in 192x256 pixel coordinates, normalized
``centerline_normalized_xy``, learned ``image_support_probability`` for each of
100 body points, and the separately computed geometric ``in_fov_mask``.
"""

from __future__ import annotations

import math
from typing import Literal

import lightning as L
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import in_fov_mask, reconstruct_from_coefficients, tangent_angles, wrap_angle
from .losses import proposal_loss


Variant = Literal["coordinate", "intrinsic"]


def smooth_tangent_basis(num_points: int = 100, coefficients: int = 16) -> Tensor:
    """Deterministic zero-mean orthonormal cosine basis [N, K]."""

    s = (torch.arange(num_points, dtype=torch.float32) + 0.5) / num_points
    frequencies = torch.arange(1, coefficients + 1, dtype=torch.float32)
    return math.sqrt(2.0 / num_points) * torch.cos(math.pi * s[:, None] * frequencies)


class SmallEncoder(nn.Module):
    """Shared encoder used unchanged by both representation variants."""

    def __init__(self, pool_output: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        if len(pool_output) != 2 or any(value < 1 for value in pool_output):
            raise ValueError("pool_output must contain two positive integers")
        feature_shape = (12, 16)
        if any(size % output for size, output in zip(feature_shape, pool_output, strict=True)):
            raise ValueError("pool_output must evenly divide the fixed 12x16 feature map")
        channels = (1, 24, 48, 96, 128)
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(),
                    nn.Conv2d(output_channels, output_channels, 3, padding=1, groups=output_channels),
                    nn.SiLU(),
                ]
            )
        # The fixed 192x256 input is 12x16 after four stride-2 blocks. A fixed
        # average pool preserves the configured spatial map and, unlike
        # adaptive_avg_pool2d_backward, has a deterministic CUDA backward path.
        kernel = tuple(size // output for size, output in zip(feature_shape, pool_output, strict=True))
        self.network = nn.Sequential(*blocks, nn.AvgPool2d(kernel), nn.Flatten())
        self.output_features = 128 * pool_output[0] * pool_output[1]
        self.pool_output = pool_output

    def forward(self, image: Tensor) -> Tensor:
        return self.network(image)


class WormProposalModule(L.LightningModule):
    """LightningModule comparing coordinate and 16-coefficient intrinsic heads."""

    def __init__(
        self,
        variant: Variant = "coordinate",
        *,
        learning_rate: float = 3e-4,
        image_height: int = 192,
        image_width: int = 256,
        num_points: int = 100,
        intrinsic_coefficients: int = 16,
        anchor_index: int = 50,
        encoder_pool_output: tuple[int, int] = (2, 2),
        model_seed: int | None = None,
        data_seed: int | None = None,
        training_order_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if variant not in ("coordinate", "intrinsic"):
            raise ValueError("variant must be 'coordinate' or 'intrinsic'")
        if intrinsic_coefficients != 16:
            raise ValueError("EXP-0004 requires exactly 16 intrinsic coefficients")
        self.save_hyperparameters()
        self.variant = variant
        encoder_pool_output = tuple(int(value) for value in encoder_pool_output)
        self.encoder = SmallEncoder(encoder_pool_output)
        representation_size = 2 * num_points if variant == "coordinate" else 2 + 1 + 2 + 16
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_features, 256),
            nn.SiLU(),
            nn.Linear(256, representation_size + num_points),
        )
        self.register_buffer("tangent_basis", smooth_tangent_basis(num_points, 16))

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1:] != (
            1,
            self.hparams.image_height,
            self.hparams.image_width,
        ):
            raise ValueError("images must have shape [B, 1, 192, 256]")
        raw = self.head(self.encoder(images))
        representation_size = raw.shape[-1] - self.hparams.num_points
        representation = raw[:, :representation_size]
        support_logits = raw[:, representation_size:]
        scale = images.new_tensor((self.hparams.image_width, self.hparams.image_height))
        result: dict[str, Tensor] = {"image_support_logits": support_logits}
        if self.variant == "coordinate":
            # Values are normalized by raster dimensions, but deliberately not
            # clipped: censored anatomy may legitimately lie outside [0, 1].
            normalized = 0.5 + representation.reshape(-1, self.hparams.num_points, 2)
        else:
            anchor = torch.sigmoid(representation[:, :2]) * scale
            # Positive and useful from initialization; not capped to the FOV diagonal.
            length = 20.0 + F.softplus(representation[:, 2]) * 160.0
            orientation_vector = F.normalize(representation[:, 3:5], dim=-1, eps=1e-6)
            orientation = torch.atan2(orientation_vector[:, 1], orientation_vector[:, 0])
            coefficients = torch.tanh(representation[:, 5:21]) * math.pi
            centerline, tangent = reconstruct_from_coefficients(
                anchor,
                orientation,
                length,
                coefficients,
                self.tangent_basis,
                anchor_index=self.hparams.anchor_index,
            )
            normalized = centerline / scale
            result.update(
                {
                    "anchor_xy": anchor,
                    "body_length": length,
                    "global_orientation_vector": orientation_vector,
                    "tangent_coefficients": coefficients,
                    "tangent_angle": tangent,
                }
            )
        centerline = normalized * scale
        result.update(
            {
                "centerline_normalized_xy": normalized,
                "centerline_xy": centerline,
                "image_support_probability": torch.sigmoid(support_logits),
                "in_fov_mask": in_fov_mask(
                    centerline, self.hparams.image_height, self.hparams.image_width
                ),
            }
        )
        return result

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        values = proposal_loss(
            self(batch["image"]),
            batch["centerline_xy"],
            batch["image_support_target"],
            image_height=self.hparams.image_height,
            image_width=self.hparams.image_width,
        )
        for name, value in values.items():
            self.log(
                f"{stage}_{name}",
                value,
                on_step=stage == "train",
                on_epoch=True,
                add_dataloader_idx=False,
            )
        if not bool(torch.isfinite(values["loss"])):
            raise RuntimeError("non-finite proposal loss")
        return values["loss"]

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        namespace = "val_proxy_candidate" if dataloader_idx == 0 else "val_tier_c"
        loss = self._shared_step(batch, namespace)
        if dataloader_idx == 1:
            prediction = self(batch["image"])["centerline_xy"]
            target = batch["centerline_xy"]
            original_scale = target.new_tensor(
                (968 / self.hparams.image_width, 732 / self.hparams.image_height)
            )
            predicted_angle = tangent_angles(prediction * original_scale)
            target_angle = tangent_angles(target * original_scale)
            forward = wrap_angle(predicted_angle - target_angle).abs().mean(-1)
            reverse = wrap_angle(
                predicted_angle - wrap_angle(target_angle.flip(-1) + torch.pi)
            ).abs().mean(-1)
            angle_mae_degrees = torch.minimum(forward, reverse).mean() * 180 / torch.pi
            self.log(
                "val_tier_c_angle_mae_degrees",
                angle_mae_degrees,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            fully_visible = batch["image_support_target"].all(-1)
            if bool(fully_visible.any()):
                fully_visible_angle = torch.minimum(forward, reverse)[fully_visible]
                self.log(
                    "val_tier_c_fully_visible_angle_mae_degrees",
                    fully_visible_angle.mean() * 180 / torch.pi,
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(fully_visible.sum()),
                    add_dataloader_idx=False,
                )
            self.log(
                "val_tier_c_fully_visible_count",
                fully_visible.sum().to(torch.float32),
                on_step=False,
                on_epoch=True,
                reduce_fx="sum",
                add_dataloader_idx=False,
            )
        return loss

    def on_after_backward(self) -> None:
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in self.parameters()
        ):
            raise RuntimeError("non-finite proposal gradient")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
