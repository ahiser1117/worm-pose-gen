"""Semi-supervised training of the body-field network: temporal consistency on unlabeled video against a mean teacher.

The supervised loss on hand labels (:meth:`body_net.BodyFieldModule.loss`)
is unchanged.  Each step also draws unlabeled windows of three consecutive
frames ``t-1, t, t+1`` (each with its own lag channels) from a clip store
(:mod:`unlabeled_clips`).  An exponential moving average of the student (the
*teacher*, Tarvainen & Valpola 2017) predicts all three frames from the clean
crop; the student predicts frame ``t`` from an augmented crop (gain, offset,
noise, and a flip/rotation that is undone on its outputs).  The student is
pulled toward targets built from the teacher (:func:`consistency_terms`):

``mask_sandwich``  (a) where the teacher's masks at ``t-1`` and ``t+1`` are
                   confident and agree, the student's mask at ``t`` should agree
                   with them (the flicker the temporal evaluation counts)
``ap_temporal``    (b) on pixels inside all three teacher masks (no crossing),
                   the student's A-P at ``t`` should lie within ``ap_margin`` of
                   the mean of the teacher's at ``t±1``; a head/tail swap
                   costs up to 1 per pixel.  Pixels where the teacher's own
                   neighbours differ by more than ``ap_neighbour_max`` (the
                   teacher flipped between them) are skipped.
``head_temporal``  (c) when the teacher's head peaks at ``t±1`` are confident
                   and within a body width of each other, the student's head
                   heatmap at ``t`` is trained (focal loss) toward a Gaussian at
                   their midpoint; the tail likewise with its own (default
                   zero) weight, since the tail is the end people cannot label
``*_teacher``      (d) everywhere a temporal target does not apply, the
                   un-transformed student matches the teacher at ``t`` on
                   confident pixels (mask, A-P, confident end peaks): flip and
                   rotation equivariance plus the usual mean-teacher term

The teacher predicts whole frames, and only the largest connected component
of its mask counts as worm; its other positives (agar tracks the worm left,
debris, image borders) are negative targets.  The first runs, which took
every confident teacher positive as a target, self-trained partly segmented
tracks into confident false positives; ignoring the other components instead
let background blobs grow unchecked on another project's recording.

The unlabeled loss ramps up from zero (``ramp_epochs``) after an optional
delay.  Validation, early stopping and the learning-rate plateau are on the
supervised validation loss alone, as for supervised training, of the network
that becomes the model (``export``: the teacher by default, whose averaged
weights validate lower and more steadily than the student's); both networks'
validation losses are logged.

The trained checkpoint is converted back to a plain
:class:`body_net.BodyFieldModule` checkpoint (:func:`export_body_checkpoint`),
so a semi-supervised model loads with :func:`body_net.load_body_net` and runs
in the pipeline and the app like any other.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import inspect
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn.functional as F

from .body_net import HEATMAP_SIGMA_DIAMETERS, OUTPUTS, BodyFieldModule, heatmap_focal_loss, peak_xy
from .body_proposal import END_THRESHOLD
from .segmenter import INPUT_MEAN, INPUT_STD
from .temporal_eval import BODY_WIDTH_PX
from .unlabeled_clips import ClipStore


_MASK, _AP, _HEAD, _TAIL, _OVERLAP = (OUTPUTS.index(n) for n in ("mask", "ap", "head", "tail", "overlap"))
# A dark block this far below the frame's median brightness is worm (the crop is centred on one, usually).
_DARK_LEVEL = 20.0
_BLOCK = 8


@dataclass
class SemiConfig:
    """Settings of the unlabeled part of a run (the card keeps them under ``hparams.semi``)."""

    store: str = ""                      # clip store directory (unlabeled_clips.build_store)
    unlabeled_batch: int = 4             # windows per step, beside the labeled batch
    crop: int = 512
    workers: int = 6
    weight: float = 1.0                  # weight of the whole unlabeled loss after ramp-up
    start_epoch: float = 0.0             # epochs before the unlabeled loss starts
    ramp_epochs: float = 5.0             # sigmoid ramp-up length (epochs)
    ema_decay: float = 0.995             # teacher decay (warmed up as 1 - 1/(step + 1) early on)
    confidence: float = 0.9              # teacher mask probability (or 1 - it) a pixel needs to count
    min_area: int = 8000                 # teacher worm pixels at t (in the crop) a window needs to count
    mask_sandwich: float = 1.0           # (a)
    ap_temporal: float = 1.0             # (b)
    ap_margin: float = 0.02
    ap_neighbour_max: float = 0.1
    head_temporal: float = 0.1           # (c)
    tail_temporal: float = 0.0
    mask_teacher: float = 1.0            # (d)
    ap_teacher: float = 1.0
    head_teacher: float = 0.1
    tail_teacher: float = 0.0
    export: str = "teacher"              # which network becomes the model (and is validated): teacher or student

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemiConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown semi-supervised settings: {unknown}")
        config = cls(**data)
        if not config.store:
            raise ValueError("semi-supervised training needs a clip store")
        if config.export not in ("student", "teacher"):
            raise ValueError("export must be student or teacher")
        return config


# --------------------------------------------------------------------------- data


def _input_stack(frames: NDArray[np.float32], centre: int, lags: Sequence[int]) -> NDArray[np.float32]:
    """``[1 + lags, H, W]`` network input of frame ``centre`` of a float stack (as :class:`model_training.LabelDataset` builds it)."""

    image = (frames[centre] / 255.0 - INPUT_MEAN) / INPUT_STD
    channels = [image] + [(frames[centre + lag] - frames[centre - lag]) / (255.0 * INPUT_STD) for lag in lags]
    return np.stack(channels).astype(np.float32)


def apply_transform(x: Tensor, flip_x: bool, flip_y: bool, rotations: int) -> Tensor:
    """Flip the last axis, then the second last, then rotate by ``rotations`` quarter turns."""

    if flip_x:
        x = torch.flip(x, (-1,))
    if flip_y:
        x = torch.flip(x, (-2,))
    return torch.rot90(x, rotations, (-2, -1)) if rotations else x


def undo_transform(x: Tensor, flip_x: bool, flip_y: bool, rotations: int) -> Tensor:
    if rotations:
        x = torch.rot90(x, -rotations, (-2, -1))
    if flip_y:
        x = torch.flip(x, (-2,))
    return torch.flip(x, (-1,)) if flip_x else x


class UnlabeledWindows:
    """Training windows of a clip store (a torch ``Dataset``; draw it with a sampler over a large index space).

    Item ``i`` is window ``i mod windows`` with a crop and augmentation drawn
    from ``(seed, i)``: ``frames`` holds the whole uint8 frames at
    :attr:`offsets` from ``t`` (what the teacher's inputs at ``t-1, t, t+1``
    read, :func:`teacher_inputs`), ``origin`` the crop's top-left corner and
    ``student`` the augmented input of ``t`` in the crop after the
    flip/rotation ``transform`` (flip x, flip y, quarter turns).  The crop is
    centred near a dark (worm) block of frame ``t`` most of the time.  The
    teacher sees whole frames so that which mask component is the worm is
    decided on the whole frame, not on a crop that may cut the body apart.
    """

    def __init__(self, store: ClipStore, lags: Sequence[int], *, crop: int = 512, seed: int = 0, augment: bool = True) -> None:
        self.store = store
        self.lags = tuple(int(lag) for lag in lags)
        self.reach = max(self.lags, default=0) + 1
        self.crop = crop
        self.seed = seed
        self.augment = augment
        self.offsets = sorted({o + sign * lag for o in (-1, 0, 1) for lag in (0, *self.lags) for sign in (-1, 1)})
        self.windows = [(s, c) for s in range(len(store)) for c in store.centres(s, 1, max(self.lags, default=0))]
        if not self.windows:
            raise ValueError(f"{store.root}: no segment is long enough for windows of ±1 frame with ±{max(self.lags, default=0)} lags")

    def __len__(self) -> int:
        return len(self.windows)

    def _crop_origin(self, frame: NDArray[np.uint8], rng: np.random.Generator) -> tuple[int, int]:
        height, width = frame.shape
        size_h, size_w = min(self.crop, height), min(self.crop, width)
        rows, cols = height // _BLOCK, width // _BLOCK
        blocks = frame[: rows * _BLOCK, : cols * _BLOCK].reshape(rows, _BLOCK, cols, _BLOCK).mean((1, 3))
        dark = np.argwhere(blocks < np.median(blocks) - _DARK_LEVEL)
        if len(dark) and rng.random() < 0.9:
            by, bx = dark[rng.integers(len(dark))]
            cy, cx = (by + 0.5) * _BLOCK, (bx + 0.5) * _BLOCK
            y0 = int(np.clip(cy - size_h // 2 + rng.integers(-size_h // 4, size_h // 4 + 1), 0, height - size_h))
            x0 = int(np.clip(cx - size_w // 2 + rng.integers(-size_w // 4, size_w // 4 + 1), 0, width - size_w))
            return y0, x0
        return int(rng.integers(0, height - size_h + 1)), int(rng.integers(0, width - size_w + 1))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        segment, centre = self.windows[index % len(self.windows)]
        rng = np.random.default_rng((self.seed, int(index)))
        frames = self.store.frames(segment)
        first = centre - self.reach
        whole = np.asarray(frames[first : centre + self.reach + 1])
        y0, x0 = self._crop_origin(whole[self.reach], rng)
        size = min(self.crop, frames.shape[1]), min(self.crop, frames.shape[2])
        window = whole[:, y0 : y0 + size[0], x0 : x0 + size[1]].astype(np.float32)
        middle = self.reach
        student = window
        flip_x = flip_y = False
        rotations = 0
        if self.augment:
            student = np.clip(window * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-20.0, 20.0)), 0.0, 255.0)
            if rng.random() < 0.5:
                student = np.clip(student + rng.normal(0.0, float(rng.uniform(1.0, 6.0)), student.shape), 0.0, 255.0)
            flip_x, flip_y = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
            rotations = int(rng.random() < 0.5) if size[0] == size[1] else 0
        student_input = apply_transform(torch.as_tensor(_input_stack(student.astype(np.float32), middle, self.lags)), flip_x, flip_y, rotations)
        return {
            "frames": torch.as_tensor(np.ascontiguousarray(whole[[middle + o for o in self.offsets]])),
            "origin": torch.tensor([y0, x0]), "student": student_input.contiguous(),
            "transform": torch.tensor([int(flip_x), int(flip_y), rotations]),
        }


def teacher_inputs(frames: Tensor, offsets: Sequence[int], lags: Sequence[int]) -> Tensor:
    """``[B, 3, 1 + lags, H, W]`` network inputs of ``t-1, t, t+1`` from uint8 frames ``[B, len(offsets), H, W]`` at ``offsets`` from ``t``."""

    position = {offset: k for k, offset in enumerate(offsets)}
    f = frames.float()
    windows = []
    for o in (-1, 0, 1):
        channels = [(f[:, position[o]] / 255.0 - INPUT_MEAN) / INPUT_STD]
        channels += [(f[:, position[o + lag]] - f[:, position[o - lag]]) / (255.0 * INPUT_STD) for lag in lags]
        windows.append(torch.stack(channels, 1))
    return torch.stack(windows, 1)


def unlabeled_loader(config: SemiConfig, lags: Sequence[int], *, seed: int, pin_memory: bool) -> Any:
    from torch.utils.data import DataLoader, RandomSampler

    windows = UnlabeledWindows(ClipStore(config.store), lags, crop=config.crop, seed=seed)
    # Indices from a space much larger than the windows, so each draw gets its own crop and augmentation.
    sampler = RandomSampler(range(10**8), replacement=True, num_samples=10**8)
    return DataLoader(windows, batch_size=config.unlabeled_batch, sampler=sampler, num_workers=config.workers,
                      persistent_workers=config.workers > 0, pin_memory=pin_memory, drop_last=True)


# --------------------------------------------------------------------------- losses


def gaussian_targets(xy: Tensor, ok: Tensor, shape: tuple[int, int], sigma: float) -> Tensor:
    """``[B, H, W]`` unit-peak Gaussians at the rounded points ``xy`` (``[B, 2]``, x then y); NaN where not ``ok``."""

    height, width = shape
    point = xy.round()
    gx = torch.exp(-0.5 * ((torch.arange(width, device=xy.device)[None] - point[:, :1]) / sigma) ** 2)
    gy = torch.exp(-0.5 * ((torch.arange(height, device=xy.device)[None] - point[:, 1:]) / sigma) ** 2)
    maps = gy[:, :, None] * gx[:, None, :]
    return torch.where(ok[:, None, None], maps, torch.full_like(maps, float("nan")))


def _mean_over(values: Tensor, gate: Tensor) -> Tensor:
    count = gate.sum()
    return (values * gate).sum() / count if count > 0 else values.sum() * 0.0


def _end_terms(student: Tensor, teacher: Tensor, items: Tensor, use_temporal: bool, sigma: float) -> tuple[Tensor, Tensor, Tensor]:
    """Focal losses of one end's heatmap: toward the midpoint of the teacher's ``t±1`` peaks, else toward its ``t`` peak."""

    height, width = student.shape[-2:]
    probability = torch.sigmoid(teacher)
    peaks = [peak_xy(teacher[:, k]) for k in range(3)]
    values = [probability[:, k].flatten(1).amax(1) for k in range(3)]
    temporal = (
        items & (values[0] >= END_THRESHOLD) & (values[2] >= END_THRESHOLD) & ((peaks[0] - peaks[2]).norm(dim=1) < BODY_WIDTH_PX)
    ) if use_temporal else torch.zeros_like(items)
    at_t = items & ~temporal & (values[1] >= END_THRESHOLD)
    zero = student.sum() * 0.0
    loss_temporal = heatmap_focal_loss(student, gaussian_targets(0.5 * (peaks[0] + peaks[2]), temporal, (height, width), sigma))
    loss_teacher = heatmap_focal_loss(student, gaussian_targets(peaks[1], at_t, (height, width), sigma))
    return (
        loss_temporal[temporal].mean() if temporal.any() else zero,
        loss_teacher[at_t].mean() if at_t.any() else zero,
        temporal.float().mean(),
    )


def largest_component(masks: Tensor) -> Tensor:
    """Each ``[..., H, W]`` boolean map reduced to its largest 8-connected component."""

    from scipy import ndimage

    flat = masks.reshape(-1, *masks.shape[-2:]).cpu().numpy()
    kept = np.zeros_like(flat)
    for k, mask in enumerate(flat):
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
        if count:
            kept[k] = labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
    return torch.as_tensor(kept, device=masks.device).reshape(masks.shape)


def consistency_terms(
    student: Tensor, teacher: Tensor, config: SemiConfig, worm: Tensor | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """The unlabeled loss terms (module docstring) and gate statistics.

    ``student`` holds the student's logits at ``t`` in the teacher's frame
    (``[B, 5, H, W]``, transform undone), ``teacher`` the teacher's logits at
    ``t-1, t, t+1`` (``[B, 3, 5, H, W]``), ``worm`` which of the teacher's
    mask pixels belong to the worm (``[B, 3, H, W]``; by default the largest
    component of each mask).  A term whose weight is zero opens no gate, so
    the teacher-at-``t`` term covers its pixels instead.
    """

    student, teacher = student.float(), teacher.float()
    c = config.confidence
    p = torch.sigmoid(teacher[:, :, _MASK])
    # Only the worm component is worm: other teacher positives (agar tracks,
    # debris, image borders) are confident negatives, so self-training
    # neither reinforces them nor lets them drift.
    inside = largest_component(p > 0.5) if worm is None else worm
    confident = ((p > c) & inside) | (p < 1 - c) | ((p > 0.5) & ~inside)
    items = inside[:, 1].sum((1, 2)) >= config.min_area
    item_pixels = items[:, None, None]

    s_mask = student[:, _MASK]
    if config.mask_sandwich > 0:
        sandwich = confident[:, 0] & confident[:, 2] & (inside[:, 0] == inside[:, 2]) & item_pixels
    else:
        sandwich = torch.zeros_like(inside[:, 0])
    at_t = confident[:, 1] & ~sandwich & item_pixels
    target = torch.where(sandwich, inside[:, 0], inside[:, 1]).float()
    bce = F.binary_cross_entropy_with_logits(s_mask, target, reduction="none")
    area = inside[:, 1].sum((1, 2)).clamp_min(config.min_area).float()

    def per_area(gate: Tensor) -> Tensor:
        per_item = (bce * gate).sum((1, 2)) / area
        return per_item[items].mean() if items.any() else bce.sum() * 0.0

    ap = torch.sigmoid(teacher[:, :, _AP])
    s_ap = torch.sigmoid(student[:, _AP])
    body = ((p > c) & inside).all(1) & (torch.sigmoid(teacher[:, 1, _OVERLAP]) < 0.5) & item_pixels
    if config.ap_temporal > 0:
        ap_gate = body & ((ap[:, 0] - ap[:, 2]).abs() < config.ap_neighbour_max)
    else:
        ap_gate = torch.zeros_like(body)
    middle = 0.5 * (ap[:, 0] + ap[:, 2])
    ap_teacher_gate = (p[:, 1] > c) & inside[:, 1] & (torch.sigmoid(teacher[:, 1, _OVERLAP]) < 0.5) & item_pixels & ~ap_gate

    sigma = HEATMAP_SIGMA_DIAMETERS * BODY_WIDTH_PX
    head_temporal, head_teacher, head_fraction = _end_terms(student[:, _HEAD], teacher[:, :, _HEAD], items, config.head_temporal > 0, sigma)
    tail_temporal, tail_teacher, tail_fraction = _end_terms(student[:, _TAIL], teacher[:, :, _TAIL], items, config.tail_temporal > 0, sigma)

    terms = {
        "mask_sandwich": per_area(sandwich),
        "mask_teacher": per_area(at_t),
        "ap_temporal": _mean_over(F.relu((s_ap - middle).abs() - config.ap_margin), ap_gate),
        "ap_teacher": _mean_over((s_ap - ap[:, 1]).abs(), ap_teacher_gate),
        "head_temporal": head_temporal, "head_teacher": head_teacher,
        "tail_temporal": tail_temporal, "tail_teacher": tail_teacher,
    }
    stats = {
        "items": items.float().mean(), "head_temporal_items": head_fraction, "tail_temporal_items": tail_fraction,
        "sandwich_px": sandwich.float().sum() / items.sum().clamp_min(1), "ap_temporal_px": ap_gate.float().sum() / items.sum().clamp_min(1),
    }
    return terms, stats


def weighted_total(terms: dict[str, Tensor], config: SemiConfig) -> Tensor:
    return sum(float(getattr(config, name)) * value for name, value in terms.items())


def ramp(epoch: float, config: SemiConfig) -> float:
    """The unlabeled weight at a (fractional) epoch: zero before ``start_epoch``, then a sigmoid ramp to ``weight``."""

    progress = epoch - config.start_epoch
    if progress < 0:
        return 0.0
    if config.ramp_epochs <= 0 or progress >= config.ramp_epochs:
        return config.weight
    return config.weight * math.exp(-5.0 * (1.0 - progress / config.ramp_epochs) ** 2)


# --------------------------------------------------------------------------- module


class SemiBodyFieldModule(BodyFieldModule):
    """:class:`body_net.BodyFieldModule` with a mean teacher and the unlabeled loss (module docstring).

    Trains on batches ``{"labeled": <body-field batch>, "unlabeled": <UnlabeledWindows batch>}``.
    """

    def __init__(self, semi: dict[str, Any] | None = None, **body: Any) -> None:
        super().__init__(**body)
        self.semi = SemiConfig.from_dict(semi or {"store": "-"})
        self.save_hyperparameters()
        self.teacher = copy.deepcopy(self.network)
        self.teacher.requires_grad_(False)
        self._offsets = sorted({o + sign * lag for o in (-1, 0, 1) for lag in (0, *self.lags) for sign in (-1, 1)})

    @classmethod
    def from_body(cls, module: BodyFieldModule, config: SemiConfig) -> "SemiBodyFieldModule":
        """A semi-supervised module starting from ``module``'s weights and settings (the teacher starts as a copy)."""

        semi = cls(semi=asdict(config), **{k: module.hparams[k] for k in body_hparam_names() if k in module.hparams})
        semi.network.load_state_dict(module.network.state_dict())
        semi.teacher.load_state_dict(module.network.state_dict())
        return semi

    def train(self, mode: bool = True) -> "SemiBodyFieldModule":
        super().train(mode)
        self.teacher.eval()  # batch norm on its running statistics, no dropout
        return self

    def _epoch(self, batch_index: int) -> float:
        steps = max(1, int(self.trainer.num_training_batches)) if self._trainer is not None else 1
        return self.current_epoch + batch_index / steps

    def training_step(self, batch: dict[str, Any], batch_index: int) -> Tensor:
        supervised = self._step(batch["labeled"], "train")
        weight = ramp(self._epoch(batch_index), self.semi)
        unlabeled = batch["unlabeled"]
        size = unlabeled["student"].shape[0]
        self.log("train_semi_weight", weight, batch_size=size, on_step=False, on_epoch=True)
        if weight <= 0:
            return supervised
        with torch.no_grad():
            inputs = teacher_inputs(unlabeled["frames"], self._offsets, self.lags)
            whole = self.teacher(inputs.flatten(0, 1)).unflatten(0, (size, 3))
            worm_whole = largest_component(torch.sigmoid(whole[:, :, _MASK].float()) > 0.5)
        student = self(unlabeled["student"])
        student = torch.stack([undo_transform(student[k], *(int(v) for v in unlabeled["transform"][k])) for k in range(size)])
        height, width = student.shape[-2:]
        crops = [(int(y), int(x)) for y, x in unlabeled["origin"]]
        teacher = torch.stack([whole[k, ..., y : y + height, x : x + width] for k, (y, x) in enumerate(crops)])
        worm = torch.stack([worm_whole[k, ..., y : y + height, x : x + width] for k, (y, x) in enumerate(crops)])
        terms, stats = consistency_terms(student, teacher, self.semi, worm)
        total = weighted_total(terms, self.semi)
        self.log("train_semi_loss", total.detach(), batch_size=size, on_step=False, on_epoch=True)
        for name, value in {**terms, **stats}.items():
            self.log(f"train_semi_{name}", value.detach(), batch_size=size, on_step=False, on_epoch=True)
        return supervised + weight * total

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_index: int) -> None:
        decay = min(float(self.semi.ema_decay), 1.0 - 1.0 / (self.global_step + 1))
        with torch.no_grad():
            for teacher, student in zip(self.teacher.parameters(), self.network.parameters()):
                teacher.mul_(decay).add_(student.detach(), alpha=1.0 - decay)
            for teacher, student in zip(self.teacher.buffers(), self.network.buffers()):
                if teacher.dtype.is_floating_point:
                    teacher.mul_(decay).add_(student.detach(), alpha=1.0 - decay)
                else:
                    teacher.copy_(student)

    def validation_step(self, batch: dict[str, Tensor], batch_index: int) -> None:
        """``val_*`` of the network that becomes the model (``export``), so early stopping selects it; both losses are logged too."""

        logits = {"student": self.network(batch["image"]), "teacher": self.teacher(batch["image"])}
        self._step(batch, "val", logits=logits[self.semi.export])
        for name, value in logits.items():
            total, _ = self.loss(value, batch["targets"])
            self.log(f"val_{name}_loss", total, batch_size=batch["image"].shape[0], on_epoch=True)


def body_hparam_names() -> list[str]:
    return [name for name in inspect.signature(BodyFieldModule.__init__).parameters if name != "self"]


def export_body_checkpoint(source: Path, target: Path, network: str = "student") -> Path:
    """Rewrite a :class:`SemiBodyFieldModule` checkpoint as a plain :class:`BodyFieldModule` one (no teacher, no optimizer)."""

    checkpoint = torch.load(str(source), map_location="cpu", weights_only=False)
    prefix = "network." if network == "student" else "teacher."
    state = {"network." + key[len(prefix):]: value for key, value in checkpoint["state_dict"].items() if key.startswith(prefix)}
    names = body_hparam_names()
    exported = {
        "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
        "pytorch-lightning_version": checkpoint["pytorch-lightning_version"], "state_dict": state,
        "hyper_parameters": {k: v for k, v in checkpoint["hyper_parameters"].items() if k in names},
    }
    torch.save(exported, str(target))
    return target
