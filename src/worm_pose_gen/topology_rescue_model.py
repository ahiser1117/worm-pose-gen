"""Topology-safe soft-anchored intrinsic model for controlled EXP-003B."""

from __future__ import annotations

import math

import lightning as L
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import in_fov_mask, reconstruct_from_coefficients
from .losses import proposal_loss
from .model import smooth_tangent_basis
from .spatial_model import SpatialBackbone


class SoftAnchoredIntrinsicModule(L.LightningModule):
    """Localize softly on a grid, then decode exactly one ordered intrinsic curve."""

    def __init__(
        self,
        *,
        learning_rate: float = 3e-4,
        image_height: int = 192,
        image_width: int = 256,
        num_points: int = 100,
        intrinsic_coefficients: int = 16,
        anchor_index: int = 50,
        selection_temperature: float = 0.5,
        model_seed: int | None = None,
        data_seed: int | None = None,
        training_order_sha256: str | None = None,
        exclusion_manifest_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if (image_height, image_width, num_points, intrinsic_coefficients) != (192, 256, 100, 16):
            raise ValueError("EXP-003B requires 192x256, 100 points, and 16 coefficients")
        self.save_hyperparameters()
        self.backbone = SpatialBackbone()
        self.anchor_head = nn.Conv2d(128, 1, 1)
        output_values = 2 + 1 + 2 + intrinsic_coefficients + num_points
        self.pose_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, output_values),
        )
        self.register_buffer(
            "tangent_basis", smooth_tangent_basis(num_points, intrinsic_coefficients)
        )

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        expected = (1, self.hparams.image_height, self.hparams.image_width)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"images must have shape [B,{expected[0]},{expected[1]},{expected[2]}]")
        grid, _ = self.backbone(images)
        batch, channels, height, width = grid.shape
        anchor_logits = self.anchor_head(grid).reshape(batch, -1)
        anchor_probability = torch.softmax(
            anchor_logits / self.hparams.selection_temperature, dim=-1
        )
        pooled = torch.einsum(
            "bp,bcp->bc", anchor_probability, grid.reshape(batch, channels, -1)
        )
        raw = self.pose_head(pooled)
        x = (torch.arange(width, device=grid.device, dtype=grid.dtype) + 0.5) * (
            self.hparams.image_width / width
        )
        y = (torch.arange(height, device=grid.device, dtype=grid.dtype) + 0.5) * (
            self.hparams.image_height / height
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        cell_centers = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        visible_anchor = anchor_probability @ cell_centers
        scale = images.new_tensor((self.hparams.image_width, self.hparams.image_height))
        anatomical_anchor = visible_anchor + torch.tanh(raw[:, :2]) * (0.5 * scale)
        body_length = 20.0 + F.softplus(raw[:, 2]) * 160.0
        orientation_vector = F.normalize(raw[:, 3:5], dim=-1, eps=1e-6)
        orientation = torch.atan2(orientation_vector[:, 1], orientation_vector[:, 0])
        coefficient_end = 5 + self.hparams.intrinsic_coefficients
        coefficients = torch.tanh(raw[:, 5:coefficient_end]) * math.pi
        support_logits = raw[:, coefficient_end:]
        centerline, tangent = reconstruct_from_coefficients(
            anatomical_anchor,
            orientation,
            body_length,
            coefficients,
            self.tangent_basis,
            anchor_index=self.hparams.anchor_index,
        )
        return {
            "centerline_xy": centerline,
            "centerline_normalized_xy": centerline / scale,
            "image_support_logits": support_logits,
            "image_support_probability": torch.sigmoid(support_logits),
            "in_fov_mask": in_fov_mask(
                centerline, self.hparams.image_height, self.hparams.image_width
            ),
            "anchor_heatmap_logits": anchor_logits,
            "anchor_probability": anchor_probability,
            "visible_anchor_xy": visible_anchor,
            "anatomical_anchor_xy": anatomical_anchor,
            "body_length": body_length,
            "global_orientation_vector": orientation_vector,
            "tangent_coefficients": coefficients,
            "tangent_angle": tangent,
            "selection_score": anchor_probability.amax(-1),
        }

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        output = self(batch["image"])
        values = proposal_loss(
            output,
            batch["centerline_xy"],
            batch["image_support_target"],
            image_height=self.hparams.image_height,
            image_width=self.hparams.image_width,
        )
        target = batch["centerline_xy"]
        support = batch["image_support_target"].bool()
        target_anchor = 0.5 * (target[:, self.hparams.anchor_index - 1] + target[:, self.hparams.anchor_index])
        anchor_visible = support[:, self.hparams.anchor_index - 1] & support[:, self.hparams.anchor_index]
        height, width = 12, 16
        x = torch.floor(target_anchor[:, 0] * width / self.hparams.image_width).long()
        y = torch.floor(target_anchor[:, 1] * height / self.hparams.image_height).long()
        anchor_visible &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if bool(anchor_visible.any()):
            target_cell = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
            anchor_loss = F.cross_entropy(
                output["anchor_heatmap_logits"][anchor_visible], target_cell[anchor_visible]
            )
        else:
            anchor_loss = output["anchor_heatmap_logits"].sum() * 0.0
        target_length = torch.linalg.vector_norm(target[:, 1:] - target[:, :-1], dim=-1).sum(-1)
        length_loss = F.smooth_l1_loss(
            torch.log(output["body_length"]), torch.log(target_length)
        )
        total = values["loss"] + 0.02 * anchor_loss + 0.05 * length_loss
        batch_size = int(batch["image"].shape[0])
        self.log(f"{stage}_loss", total, on_step=stage == "train", on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_anchor_loss", anchor_loss, on_step=stage == "train", on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_length_loss", length_loss, on_step=stage == "train", on_epoch=True, batch_size=batch_size)
        for name, value in values.items():
            if name != "loss":
                self.log(f"{stage}_{name}", value, on_step=stage == "train", on_epoch=True, batch_size=batch_size)
        if not bool(torch.isfinite(total)):
            raise RuntimeError("non-finite EXP-003B loss")
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
            raise RuntimeError("non-finite EXP-003B gradient")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )
