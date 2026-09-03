"""Small pretrained worm/background segmenter and its Lightning module.

The network is a U-Net whose encoder is torchvision's ImageNet-pretrained
ResNet-18 with the stem collapsed to one input channel.  Skip connections at
strides 2 through 32 plus the raw input give full-resolution logits, which is
what thin tails need.  About 14M parameters; a 732 x 968 frame runs in a few
milliseconds on the project GPU.

Inputs are grayscale frames (uint8 or float in [0, 255]); the module handles
normalization and padding to a multiple of 32.  Labels are ``0`` background,
``1`` worm, and ``255`` ignore; ignored pixels contribute to neither the loss
nor the metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision


IGNORE_LABEL = 255
# ImageNet statistics collapsed to one channel.
INPUT_MEAN = 0.449
INPUT_STD = 0.226
PAD_MULTIPLE = 32


def normalize_frame(frame: NDArray[np.generic] | Tensor) -> Tensor:
    """Map a ``[H,W]`` grayscale frame in [0, 255] to a normalized ``[1,H,W]`` tensor."""

    values = torch.as_tensor(np.asarray(frame) if not isinstance(frame, Tensor) else frame)
    if values.ndim != 2:
        raise ValueError("frame must have shape [H,W]")
    scaled = values.to(dtype=torch.float32) / 255.0
    return ((scaled - INPUT_MEAN) / INPUT_STD).unsqueeze(0)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class ResNet18UNet(nn.Module):
    """ResNet-18 encoder (ImageNet weights) with a light U-Net decoder."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        stem = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            stem.weight.copy_(backbone.conv1.weight.sum(dim=1, keepdim=True))
        self.stem = nn.Sequential(stem, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.up4 = _UpBlock(512, 256, 256)
        self.up3 = _UpBlock(256, 128, 128)
        self.up2 = _UpBlock(128, 64, 64)
        self.up1 = _UpBlock(64, 64, 32)
        self.up0 = _UpBlock(32, 1, 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def encoder_parameters(self) -> list[nn.Parameter]:
        modules = (self.stem, self.layer1, self.layer2, self.layer3, self.layer4)
        return [parameter for module in modules for parameter in module.parameters()]

    def decoder_parameters(self) -> list[nn.Parameter]:
        modules = (self.up4, self.up3, self.up2, self.up1, self.up0, self.head)
        return [parameter for module in modules for parameter in module.parameters()]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("input must have shape [B,1,H,W]")
        height, width = x.shape[-2:]
        pad_h = (-height) % PAD_MULTIPLE
        pad_w = (-width) % PAD_MULTIPLE
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
        s2 = self.stem(padded)
        s4 = self.layer1(self.pool(s2))
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)
        d = self.up4(s32, s16)
        d = self.up3(d, s8)
        d = self.up2(d, s4)
        d = self.up1(d, s2)
        d = self.up0(d, padded)
        logits = self.head(d)
        return logits[..., :height, :width]


def masked_binary_metrics(
    probability: Tensor, target: Tensor, valid: Tensor, threshold: float = 0.5
) -> dict[str, Tensor]:
    """IoU, Dice, precision, and recall over valid pixels, per batch item.

    A frame whose label and prediction are both empty scores 1 on every
    metric (an empty frame correctly left empty).  A label that is empty but
    a prediction that is not scores 0, as the standard formulas give.
    """

    prediction = (probability >= threshold) & valid.bool()
    truth = (target >= 0.5) & valid.bool()
    dims = tuple(range(1, prediction.ndim))
    tp = (prediction & truth).sum(dims).float()
    fp = (prediction & ~truth).sum(dims).float()
    fn = (~prediction & truth).sum(dims).float()
    eps = 1e-6
    both_empty = (tp + fp + fn) == 0
    metrics = {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }
    return {name: torch.where(both_empty, torch.ones_like(value), value) for name, value in metrics.items()}


class SegmentationModule(L.LightningModule):
    """Binary worm segmentation trained with masked BCE plus soft Dice."""

    def __init__(
        self,
        pretrained: bool = True,
        learning_rate: float = 3e-4,
        encoder_learning_rate_scale: float = 0.25,
        weight_decay: float = 1e-4,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network = ResNet18UNet(pretrained=pretrained)

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)

    def loss(self, logits: Tensor, target: Tensor, valid: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        logits = logits.squeeze(1)
        weight = valid.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none")
        bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
        probability = torch.sigmoid(logits) * weight
        truth = target.to(logits.dtype) * weight
        dims = tuple(range(1, probability.ndim))
        intersection = (probability * truth).sum(dims)
        denominator = probability.sum(dims) + truth.sum(dims)
        dice = 1 - (2 * intersection + 1.0) / (denominator + 1.0)
        total = bce + self.hparams.dice_weight * dice.mean()
        return total, {"bce": bce.detach(), "dice_loss": dice.mean().detach()}

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        logits = self(batch["image"])
        total, parts = self.loss(logits, batch["mask"], batch["valid"])
        metrics = masked_binary_metrics(torch.sigmoid(logits.squeeze(1)), batch["mask"], batch["valid"])
        batch_size = batch["image"].shape[0]
        self.log(f"{stage}_loss", total, prog_bar=True, batch_size=batch_size, on_epoch=True, on_step=stage == "train")
        for name, value in parts.items():
            self.log(f"{stage}_{name}", value, batch_size=batch_size, on_epoch=True, on_step=False)
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", value.mean(), prog_bar=name == "iou", batch_size=batch_size, on_epoch=True, on_step=False)
        return total

    def training_step(self, batch: dict[str, Tensor], batch_index: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_index: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_index: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> Any:
        lr = float(self.hparams.learning_rate)
        optimizer = torch.optim.AdamW(
            [
                {"params": self.network.encoder_parameters(), "lr": lr * float(self.hparams.encoder_learning_rate_scale)},
                {"params": self.network.decoder_parameters(), "lr": lr},
            ],
            weight_decay=float(self.hparams.weight_decay),
        )
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.02)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @torch.no_grad()
    def predict_probability(self, frame: NDArray[np.generic] | Tensor) -> NDArray[np.float32]:
        """Worm probability for one ``[H,W]`` frame, on the module's device."""

        was_training = self.training
        self.eval()
        try:
            image = normalize_frame(frame).unsqueeze(0).to(self.device)
            logits = self(image)
        finally:
            self.train(was_training)
        return torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)


def load_segmenter(
    checkpoint_path: str | Path, device: torch.device | str | None = None
) -> SegmentationModule:
    """Load a trained module for inference without downloading pretrained weights."""

    resolved = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    module = SegmentationModule.load_from_checkpoint(
        str(checkpoint_path), map_location=resolved, pretrained=False
    )
    module.to(resolved)
    module.eval()
    module.freeze()
    return module
