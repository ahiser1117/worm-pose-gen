"""Localization-preserving proposal models for scientific EXP-003."""

from __future__ import annotations

import math
from typing import Literal

import lightning as L
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import in_fov_mask, reconstruct_from_coefficients
from .losses import proposal_loss
from .model import smooth_tangent_basis


SpatialVariant = Literal["dense_centerline_field", "anchored_intrinsic_grid"]


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class SpatialBackbone(nn.Module):
    """Small U-Net-like backbone retaining 12x16 and 48x64 feature grids."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(1, 24, stride=2)       # 96x128
        self.encoder2 = ConvBlock(24, 48, stride=2)      # 48x64
        self.encoder3 = ConvBlock(48, 96, stride=2)      # 24x32
        self.encoder4 = ConvBlock(96, 128, stride=2)     # 12x16
        self.decoder3 = ConvBlock(128 + 96, 96)
        self.decoder2 = ConvBlock(96 + 48, 64)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        feature1 = self.encoder1(image)
        feature2 = self.encoder2(feature1)
        feature3 = self.encoder3(feature2)
        grid = self.encoder4(feature3)
        decoded3 = self.decoder3(torch.cat((
            F.interpolate(grid, size=feature3.shape[-2:], mode="bilinear", align_corners=False),
            feature3,
        ), dim=1))
        dense = self.decoder2(torch.cat((
            F.interpolate(decoded3, size=feature2.shape[-2:], mode="bilinear", align_corners=False),
            feature2,
        ), dim=1))
        return grid, dense


def spatial_soft_argmax(
    logits: Tensor, *, image_height: int, image_width: int, temperature: float
) -> tuple[Tensor, Tensor]:
    """Decode one ordered coordinate per heatmap channel."""

    if logits.ndim != 4 or temperature <= 0:
        raise ValueError("logits must be [B,N,H,W] and temperature must be positive")
    batch, points, height, width = logits.shape
    probability = torch.softmax(logits.reshape(batch, points, -1) / temperature, dim=-1)
    x = (torch.arange(width, device=logits.device, dtype=logits.dtype) + 0.5) * (
        image_width / width
    )
    y = (torch.arange(height, device=logits.device, dtype=logits.dtype) + 0.5) * (
        image_height / height
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    return probability @ coordinates, probability


def symmetric_dense_heatmap_loss(
    logits: Tensor,
    target_xy: Tensor,
    support: Tensor,
    *,
    image_height: int,
    image_width: int,
) -> Tensor:
    """Head-tail-symmetric visible-point spatial cross entropy."""

    batch, points, height, width = logits.shape
    if target_xy.shape != (batch, points, 2) or support.shape != (batch, points):
        raise ValueError("dense heatmap targets do not match logits")
    x = torch.floor(target_xy[..., 0] * width / image_width).to(torch.long)
    y = torch.floor(target_xy[..., 1] * height / image_height).to(torch.long)
    valid = support.bool() & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    index = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
    log_probability = torch.log_softmax(logits.reshape(batch, points, -1), dim=-1)

    def direction(target_index: Tensor, target_valid: Tensor) -> Tensor:
        selected = -torch.gather(log_probability, -1, target_index.unsqueeze(-1)).squeeze(-1)
        denominator = target_valid.sum(-1).clamp_min(1)
        return (selected * target_valid).sum(-1) / denominator

    forward = direction(index, valid)
    reverse = direction(index.flip(-1), valid.flip(-1))
    return torch.minimum(forward, reverse).mean()


class SpatialPoseModule(L.LightningModule):
    """Dense-heatmap or grid-anchored single-worm proposal module."""

    def __init__(
        self,
        variant: SpatialVariant,
        *,
        learning_rate: float = 3e-4,
        image_height: int = 192,
        image_width: int = 256,
        num_points: int = 100,
        intrinsic_coefficients: int = 16,
        anchor_index: int = 50,
        heatmap_temperature: float = 0.25,
        selection_temperature: float = 0.5,
        model_seed: int | None = None,
        data_seed: int | None = None,
        training_order_sha256: str | None = None,
        exclusion_manifest_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if variant not in ("dense_centerline_field", "anchored_intrinsic_grid"):
            raise ValueError("unsupported EXP-003 spatial variant")
        if (image_height, image_width, num_points, intrinsic_coefficients) != (192, 256, 100, 16):
            raise ValueError("initial EXP-003 run requires 192x256, 100 points, and 16 coefficients")
        self.save_hyperparameters()
        self.variant = variant
        self.backbone = SpatialBackbone()
        if variant == "dense_centerline_field":
            self.heatmap_head = nn.Conv2d(64, num_points, 1)
            self.support_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_points))
        else:
            candidate_values = 1 + 2 + 1 + 2 + intrinsic_coefficients + num_points
            self.candidate_head = nn.Conv2d(128, candidate_values, 1)
            self.register_buffer(
                "tangent_basis", smooth_tangent_basis(num_points, intrinsic_coefficients)
            )

    def _dense_forward(self, grid: Tensor, dense: Tensor) -> dict[str, Tensor]:
        logits = self.heatmap_head(dense)
        centerline, probability = spatial_soft_argmax(
            logits,
            image_height=self.hparams.image_height,
            image_width=self.hparams.image_width,
            temperature=self.hparams.heatmap_temperature,
        )
        support_logits = self.support_head(grid)
        peak = probability.amax(-1).mean(-1)
        return {
            "centerline_xy": centerline,
            "dense_heatmap_logits": logits,
            "dense_heatmap_peak_probability": peak,
            "selection_score": peak,
            "image_support_logits": support_logits,
        }

    def _anchored_forward(self, grid: Tensor) -> dict[str, Tensor]:
        raw = self.candidate_head(grid).permute(0, 2, 3, 1)
        batch, height, width, _ = raw.shape
        confidence = raw[..., 0]
        offset = torch.tanh(raw[..., 1:3])
        length = 20.0 + F.softplus(raw[..., 3]) * 160.0
        orientation_vector = F.normalize(raw[..., 4:6], dim=-1, eps=1e-6)
        orientation = torch.atan2(orientation_vector[..., 1], orientation_vector[..., 0])
        coefficient_end = 6 + self.hparams.intrinsic_coefficients
        coefficients = torch.tanh(raw[..., 6:coefficient_end]) * math.pi
        support_candidates = raw[..., coefficient_end:]
        x = (torch.arange(width, device=grid.device, dtype=grid.dtype) + 0.5) * (
            self.hparams.image_width / width
        )
        y = (torch.arange(height, device=grid.device, dtype=grid.dtype) + 0.5) * (
            self.hparams.image_height / height
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((xx, yy), dim=-1)
        cell = grid.new_tensor((self.hparams.image_width / width, self.hparams.image_height / height))
        anchor = base + 0.5 * cell * offset
        candidates, _ = reconstruct_from_coefficients(
            anchor.reshape(-1, 2),
            orientation.reshape(-1),
            length.reshape(-1),
            coefficients.reshape(-1, self.hparams.intrinsic_coefficients),
            self.tangent_basis,
            anchor_index=self.hparams.anchor_index,
        )
        candidates = candidates.reshape(batch, height * width, self.hparams.num_points, 2)
        confidence_flat = confidence.reshape(batch, -1)
        support_flat = support_candidates.reshape(batch, height * width, self.hparams.num_points)
        probability = torch.softmax(
            confidence_flat / self.hparams.selection_temperature, dim=-1
        )
        soft_centerline = torch.einsum("bc,bcnk->bnk", probability, candidates)
        soft_support = torch.einsum("bc,bcn->bn", probability, support_flat)
        selected = confidence_flat.argmax(-1)
        batch_index = torch.arange(batch, device=grid.device)
        hard_centerline = candidates[batch_index, selected]
        hard_support = support_flat[batch_index, selected]
        centerline = soft_centerline if self.training else hard_centerline
        support_logits = soft_support if self.training else hard_support
        return {
            "centerline_xy": centerline,
            "soft_centerline_xy": soft_centerline,
            "candidate_centerline_xy": candidates,
            "anchor_confidence_logits": confidence_flat,
            "anchor_confidence_probability": probability,
            "selected_cell_index": selected,
            "selection_score": torch.softmax(confidence_flat, dim=-1).amax(-1),
            "image_support_logits": support_logits,
        }

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        expected = (1, self.hparams.image_height, self.hparams.image_width)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"images must have shape [B,{expected[0]},{expected[1]},{expected[2]}]")
        grid, dense = self.backbone(images)
        result = (
            self._dense_forward(grid, dense)
            if self.variant == "dense_centerline_field"
            else self._anchored_forward(grid)
        )
        centerline = result["centerline_xy"]
        scale = images.new_tensor((self.hparams.image_width, self.hparams.image_height))
        result.update({
            "centerline_normalized_xy": centerline / scale,
            "image_support_probability": torch.sigmoid(result["image_support_logits"]),
            "in_fov_mask": in_fov_mask(
                centerline, self.hparams.image_height, self.hparams.image_width
            ),
        })
        return result

    def _additional_spatial_loss(self, output: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
        target = batch["centerline_xy"]
        support = batch["image_support_target"]
        if self.variant == "dense_centerline_field":
            return 0.02 * symmetric_dense_heatmap_loss(
                output["dense_heatmap_logits"], target, support,
                image_height=self.hparams.image_height,
                image_width=self.hparams.image_width,
            )
        midpoint = 0.5 * (target[:, self.hparams.anchor_index - 1] + target[:, self.hparams.anchor_index])
        height, width = 12, 16
        x = torch.floor(midpoint[:, 0] * width / self.hparams.image_width).long()
        y = torch.floor(midpoint[:, 1] * height / self.hparams.image_height).long()
        valid = (
            support[:, self.hparams.anchor_index - 1].bool()
            & support[:, self.hparams.anchor_index].bool()
            & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        )
        if not bool(valid.any()):
            return output["anchor_confidence_logits"].sum() * 0.0
        target_cell = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
        return 0.02 * F.cross_entropy(
            output["anchor_confidence_logits"][valid], target_cell[valid]
        )

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        output = self(batch["image"])
        values = proposal_loss(
            output,
            batch["centerline_xy"],
            batch["image_support_target"],
            image_height=self.hparams.image_height,
            image_width=self.hparams.image_width,
        )
        spatial = self._additional_spatial_loss(output, batch)
        total = values["loss"] + spatial
        self.log(f"{stage}_loss", total, on_step=stage == "train", on_epoch=True)
        self.log(f"{stage}_spatial_loss", spatial, on_step=stage == "train", on_epoch=True)
        for name, value in values.items():
            if name != "loss":
                self.log(f"{stage}_{name}", value, on_step=stage == "train", on_epoch=True)
        if not bool(torch.isfinite(total)):
            raise RuntimeError("non-finite EXP-003 loss")
        return total

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        namespace = "val_proxy_candidate" if dataloader_idx == 0 else "val_tier_c"
        return self._shared_step(batch, namespace)

    def on_after_backward(self) -> None:
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in self.parameters()
        ):
            raise RuntimeError("non-finite EXP-003 gradient")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )
