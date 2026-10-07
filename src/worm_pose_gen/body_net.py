"""Body-field network: worm mask, A-P field, head/tail heatmaps, and overlap from a frame and its motion.

The input is the flat-fielded frame plus one symmetric difference channel per
lag (:mod:`temporal_context`), fused at the first convolution of the same
ResNet-18 U-Net the segmenter uses, so the lag set costs almost nothing at
inference.  An empty lag set is the single-frame baseline.

Output channels (:data:`OUTPUTS`):

``mask``     worm logit, trained like the segmenter (masked BCE + soft Dice)
``ap``       logit of the normalized arc position (sigmoid: 0 head, 1 tail),
             L1 on mask pixels with a defined target
``head``     heatmap logit, penalty-reduced focal loss against a Gaussian
``tail``     the same for the tail
``overlap``  logit that a mask pixel is covered twice (a crossing), BCE on
             mask pixels

Targets are a label's mask and its built body targets
(:mod:`library.targets`); :mod:`model_training` reads them from the library
and decides which labels train the body (:func:`model_eval.body_used`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import lightning as L
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn.functional as F

from .body_targets import point_heatmap
from .segmenter import IGNORE_LABEL, ResNet18UNet, masked_binary_metrics


OUTPUTS = ("mask", "ap", "head", "tail", "overlap")
# Heatmap Gaussian sigma as a fraction of the median body diameter.
HEATMAP_SIGMA_DIAMETERS = 0.25
# A heatmap pixel at least this high counts as the peak in the focal loss.
HEATMAP_PEAK = 0.95
HEATMAP_PRIOR_LOGIT = -4.6
# Full-resolution decoder width shared by the five outputs (the segmenter uses 16).
FINAL_CHANNELS = 32


def _augment(
    frames: NDArray[np.float32], targets: NDArray[np.float32], rng: np.random.Generator, crop: int | None
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """The segmenter's augmentation applied jointly to a frame stack and a target stack.

    ``targets[0]`` is the worm mask (used to bias crops toward the body).
    Gain and offset are shared by every frame, like a change of illumination
    (the offset cancels in the differences); pixel noise is independent per
    frame, as camera noise is.
    """

    height, width = frames.shape[-2:]
    if crop is not None and (height > crop or width > crop):
        size_h, size_w = min(crop, height), min(crop, width)
        foreground = np.argwhere(targets[0] == 1)
        if len(foreground) and rng.random() < 0.5:
            center_y, center_x = foreground[rng.integers(len(foreground))]
            y0 = int(np.clip(center_y - size_h // 2 + rng.integers(-size_h // 4, size_h // 4 + 1), 0, height - size_h))
            x0 = int(np.clip(center_x - size_w // 2 + rng.integers(-size_w // 4, size_w // 4 + 1), 0, width - size_w))
        else:
            y0 = int(rng.integers(0, height - size_h + 1))
            x0 = int(rng.integers(0, width - size_w + 1))
        frames = frames[:, y0 : y0 + size_h, x0 : x0 + size_w]
        targets = targets[:, y0 : y0 + size_h, x0 : x0 + size_w]
    if rng.random() < 0.5:
        frames, targets = frames[..., ::-1], targets[..., ::-1]
    if rng.random() < 0.5:
        frames, targets = frames[..., ::-1, :], targets[..., ::-1, :]
    if frames.shape[-1] == frames.shape[-2] and rng.random() < 0.5:
        frames, targets = np.rot90(frames, axes=(-2, -1)), np.rot90(targets, axes=(-2, -1))
    gain = float(rng.uniform(0.8, 1.2))
    offset = float(rng.uniform(-20.0, 20.0))
    frames = np.clip(frames * gain + offset, 0.0, 255.0)
    if rng.random() < 0.5:
        sigma = float(rng.uniform(1.0, 6.0))
        frames = np.clip(frames + rng.normal(0.0, sigma, frames.shape), 0.0, 255.0)
    return np.ascontiguousarray(frames, dtype=np.float32), np.ascontiguousarray(targets, dtype=np.float32)


# Rows of the target stack a body-field item carries (:func:`body_target_rows`).
_MASK, _VALID, _AP, _AP_VALID, _HEAD, _TAIL, _OVERLAP, _OVERLAP_VALID = range(8)
TARGET_ROWS = 8


def body_target_rows(mask: NDArray[np.uint8], targets: dict[str, NDArray[Any]] | None) -> NDArray[np.float32]:
    """The ``[8,H,W]`` target stack of a label: its mask, and its body targets where they train the body.

    ``mask`` is the label (0, 1, 255 excluded); ``targets`` the arrays of its
    built body targets (:func:`library.load_targets`), or ``None`` when the
    label trains the mask only, in which case the A-P and overlap rows are
    invalid and the heatmaps NaN (they say nothing either way).
    """

    shape = mask.shape
    rows = np.zeros((TARGET_ROWS, *shape), dtype=np.float32)
    rows[_MASK] = mask == 1
    rows[_VALID] = mask != IGNORE_LABEL
    if targets is None:
        rows[_HEAD] = rows[_TAIL] = np.nan
        return rows
    ap = targets["ap"].astype(np.float32)
    body = (mask == 1) & np.isfinite(ap)
    rows[_AP] = np.where(body, ap, 0.0)
    rows[_AP_VALID] = body
    sigma = HEATMAP_SIGMA_DIAMETERS * float(targets["diameter_px"])
    rows[_HEAD] = point_heatmap(shape, targets["head_xy"], sigma)
    rows[_TAIL] = point_heatmap(shape, targets["tail_xy"], sigma)
    rows[_OVERLAP] = targets["overlap"] & (mask == 1)
    rows[_OVERLAP_VALID] = mask == 1
    return rows


def heatmap_focal_loss(logits: Tensor, target: Tensor, alpha: float = 2.0, beta: float = 4.0) -> Tensor:
    """CenterNet's penalty-reduced focal loss per batch item; NaN targets are skipped.

    Pixels at the Gaussian peak are positives; every other pixel is a
    negative whose penalty fades as the target approaches the peak.  Each
    item is normalized by its number of positives (at least one).
    """

    known = torch.isfinite(target)
    target = torch.where(known, target, torch.zeros_like(target))
    probability = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = (target >= HEATMAP_PEAK) & known
    negative = (~positive) & known
    dims = tuple(range(1, logits.ndim))
    positive_loss = (-((1 - probability) ** alpha) * torch.log(probability) * positive).sum(dims)
    negative_loss = (-((1 - target) ** beta) * probability**alpha * torch.log(1 - probability) * negative).sum(dims)
    return (positive_loss + negative_loss) / positive.sum(dims).clamp_min(1)


def peak_xy(logits: Tensor) -> Tensor:
    """``[B,2]`` (x, y) of each map's maximum."""

    flat = logits.flatten(1).argmax(1)
    width = logits.shape[-1]
    return torch.stack(((flat % width).float(), (flat // width).float()), dim=1)


class BodyFieldModule(L.LightningModule):
    def __init__(
        self,
        lags: Sequence[int] = (),
        pretrained: bool = True,
        learning_rate: float = 3e-4,
        encoder_learning_rate_scale: float = 0.25,
        weight_decay: float = 1e-4,
        dice_weight: float = 1.0,
        ap_weight: float = 1.0,
        heatmap_weight: float = 0.1,
        overlap_weight: float = 1.0,
        plateau_patience: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lags = tuple(int(lag) for lag in lags)
        self.network = ResNet18UNet(
            pretrained=pretrained, in_channels=1 + len(self.lags), out_channels=len(OUTPUTS), final_channels=FINAL_CHANNELS,
        )
        # Heatmaps start near zero everywhere (CenterNet's prior of 0.01).  At
        # an even start the focal loss sums every pixel's negative term over a
        # handful of peak pixels and swamps the other outputs by four orders.
        with torch.no_grad():
            for name in ("head", "tail"):
                self.network.head.bias[OUTPUTS.index(name)] = HEATMAP_PRIOR_LOGIT

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)

    def loss(self, logits: Tensor, targets: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        logits = logits.float()
        mask, valid = targets[:, _MASK], targets[:, _VALID]
        mask_logits = logits[:, 0]
        bce = F.binary_cross_entropy_with_logits(mask_logits, mask, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        probability = torch.sigmoid(mask_logits) * valid
        truth = mask * valid
        intersection = (probability * truth).sum((1, 2))
        dice = (1 - (2 * intersection + 1.0) / (probability.sum((1, 2)) + truth.sum((1, 2)) + 1.0)).mean()

        ap_valid = targets[:, _AP_VALID]
        ap_error = (torch.sigmoid(logits[:, 1]) - targets[:, _AP]).abs()
        ap = (ap_error * ap_valid).sum() / ap_valid.sum().clamp_min(1.0)

        head = heatmap_focal_loss(logits[:, 2], targets[:, _HEAD]).mean()
        tail = heatmap_focal_loss(logits[:, 3], targets[:, _TAIL]).mean()

        overlap_valid = targets[:, _OVERLAP_VALID]
        overlap = F.binary_cross_entropy_with_logits(logits[:, 4], targets[:, _OVERLAP], reduction="none")
        overlap = (overlap * overlap_valid).sum() / overlap_valid.sum().clamp_min(1.0)

        hp = self.hparams
        total = (
            bce + hp.dice_weight * dice + hp.ap_weight * ap
            + hp.heatmap_weight * (head + tail) + hp.overlap_weight * overlap
        )
        parts = {"bce": bce, "dice_loss": dice, "ap_l1": ap, "head_focal": head, "tail_focal": tail, "overlap_bce": overlap}
        return total, {k: v.detach() for k, v in parts.items()}

    @staticmethod
    def endpoint_metrics(logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
        """Peak errors in pixels and orientation, for items whose end lies in the image."""

        result: dict[str, Tensor] = {}
        truth = {}
        for name, channel, row in (("head", 2, _HEAD), ("tail", 3, _TAIL)):
            target = targets[:, row]
            present = torch.isfinite(target).all(dim=(1, 2)) & (torch.nan_to_num(target).amax(dim=(1, 2)) >= HEATMAP_PEAK)
            truth[name] = (peak_xy(torch.nan_to_num(target, nan=-1.0)), present)
            error = (peak_xy(logits[:, channel]) - truth[name][0]).norm(dim=1)
            result[f"{name}_error_px"] = error[present]
        both = truth["head"][1] & truth["tail"][1]
        predicted_head = peak_xy(logits[:, 2])
        to_head = (predicted_head - truth["head"][0]).norm(dim=1)
        to_tail = (predicted_head - truth["tail"][0]).norm(dim=1)
        result["orientation_correct"] = (to_head < to_tail).float()[both]
        return result

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        logits = self(batch["image"])
        targets = batch["targets"]
        total, parts = self.loss(logits, targets)
        batch_size = batch["image"].shape[0]
        self.log(f"{stage}_loss", total, prog_bar=True, batch_size=batch_size, on_epoch=True, on_step=stage == "train")
        for name, value in parts.items():
            self.log(f"{stage}_{name}", value, batch_size=batch_size, on_epoch=True, on_step=False)
        if stage != "train":
            metrics = masked_binary_metrics(torch.sigmoid(logits[:, 0].float()), targets[:, _MASK], targets[:, _VALID])
            self.log(f"{stage}_iou", metrics["iou"].mean(), prog_bar=True, batch_size=batch_size, on_epoch=True)
            ap_valid = targets[:, _AP_VALID]
            if ap_valid.sum() > 0:
                ap_error = (torch.sigmoid(logits[:, 1].float()) - targets[:, _AP]).abs()
                self.log(f"{stage}_ap_mae", (ap_error * ap_valid).sum() / ap_valid.sum(), batch_size=batch_size, on_epoch=True)
            for name, values in self.endpoint_metrics(logits.float(), targets).items():
                if values.numel():
                    self.log(f"{stage}_{name}", values.mean(), batch_size=values.numel(), on_epoch=True)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(self.hparams.plateau_patience), min_lr=lr * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss", "strict": False},
        }


def load_body_net(checkpoint_path: str | Path, device: torch.device | str | None = None) -> BodyFieldModule:
    resolved = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    module = BodyFieldModule.load_from_checkpoint(str(checkpoint_path), map_location=resolved, pretrained=False)
    module.to(resolved)
    module.eval()
    module.freeze()
    module.checkpoint_path = str(checkpoint_path)
    return module

