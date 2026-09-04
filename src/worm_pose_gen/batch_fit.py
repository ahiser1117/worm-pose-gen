"""Fit the tube model to many masks at once.

``mask_fit.fit_mask`` optimizes the starts of one frame as a batch.  Whole
recordings need the batch to span frames as well, so this module lays several
frames' crops on a common raster, renders every (frame, start) row in one
call, and compiles the renderer, which is memory-bound in eager mode.

Each frame keeps its own window on the shared raster.  Window pixels outside
the camera carry zero weight in the energy (censored, exactly as ``fit_mask``
never instantiates them); window pixels inside the camera but outside the
frame's own padded crop are ordinary background.  Frames are grouped so that a
group's raster is the maximum crop of its members and the rows-times-pixels
product stays under a memory budget.

The default schedule stops at downsample 2.  The body is 40--50 px wide, so a
2 px raster still resolves the edge, and the full-resolution stage dominated
the runtime of the reference schedule.  Final overlaps are still measured on
a full-resolution render.  ``PRESETS`` holds the measured speed/accuracy
trade-offs; ``reference`` reproduces ``fit_mask``'s result exactly.  The width
model (scale, template, and log-space asymmetry correction) is the one in
``mask_fit._MaskFitState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from .mask_fit import (
    CropWindow,
    Initialization,
    MaskFitConfig,
    MaskFitResult,
    _MaskFitState,
    crop_window,
    default_width_template,
    hard_iou,
    render_tube_segments,
    signed_edge_distance,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class BatchFitConfig(MaskFitConfig):
    """``MaskFitConfig`` plus batching controls; the schedule stops at downsample 2."""

    stage_downsample: tuple[int, ...] = (4, 2)
    stage_steps: tuple[int, ...] = (60, 100)
    # The downsample-2 stage is the last one, so it keeps a higher rate than
    # the reference schedule's middle stage; on the 30-frame set this closed
    # a third of the gap to the 550-step reference at no cost.
    stage_lr_scale: tuple[float, ...] = (1.0, 0.6)
    crop_padding: int = 32
    # Centerline points used for rendering at each stage; a stride of 2 halves
    # the segment count where the raster cannot resolve the difference.
    stage_point_stride: tuple[int, ...] = (2, 1)
    crop_multiple: int = 32
    compile_renderer: bool = True
    # Rows (frame x start) times raster pixels per optimization group.
    row_pixel_budget: int = 4_000_000
    max_rows: int = 256
    # Rows per chunk for the no-grad full-resolution render at the end.
    final_render_rows: int = 8


# Measured on the 30-frame stress set (docs/POSE_PIPELINE_PLAN.md, step 1):
# median IoU deficit against the 550-step ``fit_mask`` reference and cost per
# frame on an RTX 6000 Ada with two starts (skeleton and straight moments).
#   fast       -0.008 IoU   0.25 s/frame
#   balanced   -0.006 IoU   0.46 s/frame
#   reference   0.000 IoU   4.9 s/frame (all starts, padding 64)
PRESETS: dict[str, "BatchFitConfig"] = {
    "fast": BatchFitConfig(),
    "balanced": BatchFitConfig(stage_steps=(100, 200), within_stage_decay=0.03),
    "reference": BatchFitConfig(
        stage_downsample=(4, 2, 1, 1),
        stage_steps=(150, 100, 150, 150),
        stage_lr_scale=(1.0, 0.5, 0.25, 0.05),
        stage_point_stride=(1, 1, 1, 1),
        crop_padding=64,
    ),
}

Renderer = Callable[..., Tensor]
_COMPILED: dict[str, Renderer] = {}


def get_renderer(compile_renderer: bool) -> Renderer:
    """The soft-tube renderer, compiled once per process when requested."""

    if not compile_renderer:
        return render_tube_segments
    if "fn" not in _COMPILED:
        try:
            _COMPILED["fn"] = torch.compile(render_tube_segments, dynamic=True)
        except Exception as error:  # pragma: no cover - depends on the toolchain
            warnings.warn(f"torch.compile unavailable ({error}); rendering eagerly", stacklevel=2)
            _COMPILED["fn"] = render_tube_segments
    return _COMPILED["fn"]


def _place(start: int, size: int, target: int, limit: int) -> int:
    """Origin of a ``target``-long window covering ``[start, start+size)`` in ``[0, limit)``."""

    if target >= limit:
        return (limit - target) // 2
    origin = start - (target - size) // 2
    return min(max(origin, 0), limit - target)


def batch_windows(crops: Sequence[CropWindow]) -> tuple[list[CropWindow], int, int]:
    """Place every crop inside a window of the common (max) size; may overhang the camera."""

    height = max(c.height for c in crops)
    width = max(c.width for c in crops)
    windows = []
    for c in crops:
        x0 = _place(c.x0, c.width, width, c.image_width)
        y0 = _place(c.y0, c.height, height, c.image_height)
        windows.append(CropWindow(x0, x0 + width, y0, y0 + height, c.image_height, c.image_width))
    return windows, height, width


def plan_groups(
    crops: Sequence[CropWindow], rows_per_frame: Sequence[int], config: BatchFitConfig
) -> list[list[int]]:
    """Group frame indices by crop size under the row-pixel and row budgets."""

    if len(crops) != len(rows_per_frame):
        raise ValueError("crops and rows_per_frame must align")
    order = sorted(range(len(crops)), key=lambda i: (crops[i].height * crops[i].width, crops[i].height))
    groups: list[list[int]] = []
    current: list[int] = []
    rows = height = width = 0
    for index in order:
        c = crops[index]
        new_rows = rows + rows_per_frame[index]
        new_height, new_width = max(height, c.height), max(width, c.width)
        fits = new_rows * new_height * new_width <= config.row_pixel_budget and new_rows <= config.max_rows
        if current and not fits:
            groups.append(current)
            current, rows, height, width = [], 0, 0, 0
            new_rows, new_height, new_width = rows_per_frame[index], c.height, c.width
        current.append(index)
        rows, height, width = new_rows, new_height, new_width
    if current:
        groups.append(current)
    return groups


def _downsample_batch(values: Tensor, factor: int) -> Tensor:
    if factor == 1:
        return values
    *lead, height, width = values.shape
    return values.reshape(*lead, height // factor, factor, width // factor, factor).mean((-3, -1))


def _point_index(n_points: int, stride: int, device: torch.device) -> Tensor:
    index = torch.arange(0, n_points, max(1, stride), device=device)
    if int(index[-1]) != n_points - 1:
        index = torch.cat((index, torch.tensor([n_points - 1], device=device)))
    return index


def _window_targets(
    masks: Sequence[BoolArray], windows: Sequence[CropWindow], height: int, width: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Hard target, signed edge distance (-inf off camera), and validity per frame."""

    n = len(masks)
    target = torch.zeros((n, height, width), dtype=torch.float32, device=device)
    valid = torch.zeros((n, height, width), dtype=torch.float32, device=device)
    distance = torch.full((n, height, width), -float("inf"), dtype=torch.float32, device=device)
    for f, (mask, w) in enumerate(zip(masks, windows, strict=True)):
        ix0, ix1 = max(w.x0, 0), min(w.x1, w.image_width)
        iy0, iy1 = max(w.y0, 0), min(w.y1, w.image_height)
        local = torch.as_tensor(mask[iy0:iy1, ix0:ix1], dtype=torch.float32, device=device)
        rows = slice(iy0 - w.y0, iy1 - w.y0)
        cols = slice(ix0 - w.x0, ix1 - w.x0)
        target[f, rows, cols] = local
        valid[f, rows, cols] = 1.0
        if bool(local.all()):
            distance[f, rows, cols] = float("inf")
        else:
            distance[f, rows, cols] = signed_edge_distance(local >= 0.5)
    return target, distance, valid


def fit_masks(
    masks: Sequence[NDArray[np.generic]],
    initializations: Sequence[Sequence[Initialization]],
    *,
    width_template: NDArray[np.generic] | None = None,
    config: BatchFitConfig = BatchFitConfig(),
    device: torch.device | str | None = None,
) -> list[MaskFitResult]:
    """Fit every mask from its own starts; results follow the input order."""

    if len(masks) != len(initializations):
        raise ValueError("masks and initializations must align")
    if not masks:
        return []
    if not (
        len(config.stage_downsample)
        == len(config.stage_steps)
        == len(config.stage_lr_scale)
        == len(config.stage_point_stride)
    ):
        raise ValueError("stage_downsample, stage_steps, stage_lr_scale, and stage_point_stride must align")
    if not 0 < config.within_stage_decay <= 1:
        raise ValueError("within_stage_decay must lie in (0, 1]")
    binaries = []
    for index, (mask, starts) in enumerate(zip(masks, initializations, strict=True)):
        binary = np.asarray(mask, dtype=bool)
        if binary.ndim != 2 or not binary.any():
            raise ValueError(f"mask {index} must be a non-empty 2-D boolean image")
        if not starts:
            raise ValueError(f"mask {index} has no initializations")
        binaries.append(binary)
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    template = (
        default_width_template(config.n_points)
        if width_template is None
        else np.asarray(width_template, dtype=np.float64)
    )
    if template.shape != (config.n_points,) or np.any(template <= 0):
        raise ValueError("width_template must be positive with shape [n_points]")
    template_t = torch.as_tensor(template, dtype=torch.float32, device=resolved_device)
    multiple = max(max(config.stage_downsample), config.crop_multiple)
    crops = [crop_window(b, config.crop_padding, multiple) for b in binaries]
    results: list[MaskFitResult | None] = [None] * len(binaries)
    for group in plan_groups(crops, [len(s) for s in initializations], config):
        fitted = _fit_group(
            [binaries[i] for i in group],
            [initializations[i] for i in group],
            [crops[i] for i in group],
            template_t,
            config,
            resolved_device,
        )
        for index, result in zip(group, fitted, strict=True):
            results[index] = result
    return [r for r in results if r is not None]


def _fit_group(
    masks: Sequence[BoolArray],
    initializations: Sequence[Sequence[Initialization]],
    crops: Sequence[CropWindow],
    template: Tensor,
    config: BatchFitConfig,
    device: torch.device,
) -> list[MaskFitResult]:
    renderer = get_renderer(config.compile_renderer)
    windows, height, width = batch_windows(crops)
    target, distance, valid = _window_targets(masks, windows, height, width, device)
    starts_flat = [s for starts in initializations for s in starts]
    frame_of_row = torch.as_tensor(
        [f for f, starts in enumerate(initializations) for _ in starts], device=device
    )
    row_windows = [windows[int(f)] for f in frame_of_row.tolist()]
    offsets = torch.as_tensor(
        [[w.x0, w.y0] for w in row_windows], dtype=torch.float32, device=device
    )
    # Window edges strictly inside the camera constrain the body; edges at or
    # beyond the camera edge are censoring boundaries.
    edge_lo = torch.as_tensor([[w.x0, w.y0] for w in row_windows], dtype=torch.float32, device=device)
    edge_hi = torch.as_tensor([[w.x1 - 1, w.y1 - 1] for w in row_windows], dtype=torch.float32, device=device)
    lo_active = torch.as_tensor([[w.x0 > 0, w.y0 > 0] for w in row_windows], dtype=torch.float32, device=device)
    hi_active = torch.as_tensor(
        [[w.x1 < w.image_width, w.y1 < w.image_height] for w in row_windows], dtype=torch.float32, device=device
    )
    camera_size = torch.as_tensor(
        [[w.image_width, w.image_height] for w in row_windows], dtype=torch.float32, device=device
    )

    state = _MaskFitState(starts_flat, config, device)
    optimizer = state.optimizer()
    base_rates = [group["lr"] for group in optimizer.param_groups]
    finest = min(config.stage_downsample)
    eps = torch.finfo(torch.float32).eps
    stage_cache: dict[int, tuple[Tensor, Tensor]] = {}

    def stage_arrays(factor: int) -> tuple[Tensor, Tensor]:
        if factor not in stage_cache:
            soft = torch.sigmoid(distance / (config.edge_softness * factor))
            stage_cache[factor] = (_downsample_batch(soft, factor), _downsample_batch(valid, factor))
        return stage_cache[factor]

    def regularization(centerline: Tensor) -> Tensor:
        c = config
        smooth = c.shape_smoothness * (state.shape[:, 1:] - state.shape[:, :-1]).square().mean(1)
        below = ((edge_lo[:, None, :] - centerline).clamp_min(0).square() * lo_active[:, None, :]).sum(-1)
        above = ((centerline - edge_hi[:, None, :]).clamp_min(0).square() * hi_active[:, None, :]).sum(-1)
        # Points past the camera edge are censored: no data, no escape penalty.
        inside_camera = ((centerline >= 0) & (centerline < camera_size[:, None, :])).all(-1).to(centerline.dtype)
        escape = ((below + above) * inside_camera).mean(1)
        return smooth + state.size_regularization() + c.crop_escape_weight * escape + state.width_prior()

    def render_rows(centerline: Tensor, diameter: Tensor, factor: int, stride: int, fn: Renderer) -> Tensor:
        index = _point_index(centerline.shape[1], stride, device)
        local = (centerline[:, index] - offsets[:, None, :] + 0.5) / factor - 0.5
        return fn(
            local, diameter[:, index] / factor, height // factor, width // factor, edge_softness=config.edge_softness
        )

    def energy(factor: int, stride: int) -> tuple[Tensor, Tensor]:
        centerline = state.centerline()
        diameter = state.diameter(template)
        rendered = render_rows(centerline, diameter, factor, stride, renderer)
        soft, weight = stage_arrays(factor)
        soft = soft[frame_of_row]
        weight = weight[frame_of_row]
        intersection = (rendered * soft * weight).sum((1, 2))
        denominator = (rendered * weight).sum((1, 2)) + (soft * weight).sum((1, 2))
        dice = 1 - (2 * intersection + eps) / (denominator + eps)
        return dice, dice + regularization(centerline)

    finest_stride = config.stage_point_stride[config.stage_downsample.index(finest)]
    with torch.no_grad():
        initial_dice, _ = energy(finest, finest_stride)
    best_loss = torch.full((len(starts_flat),), float("inf"), device=device)
    best_snapshot = state.parameter_snapshot()
    history: list[list[float]] = []
    for factor, steps, scale, stride in zip(
        config.stage_downsample, config.stage_steps, config.stage_lr_scale, config.stage_point_stride, strict=True
    ):
        for step in range(steps):
            progress = step / max(steps - 1, 1)
            decay = 1.0 + (config.within_stage_decay - 1.0) * progress
            for group, base in zip(optimizer.param_groups, base_rates, strict=True):
                group["lr"] = base * scale * decay
            optimizer.zero_grad(set_to_none=True)
            dice, loss = energy(factor, stride)
            total = loss.sum()
            if not bool(torch.isfinite(total)):
                raise RuntimeError("non-finite batch mask-fit energy")
            if factor == finest:
                improved = loss.detach() < best_loss
                if bool(improved.any()):
                    rows = torch.nonzero(improved, as_tuple=False).squeeze(1)
                    current = state.parameter_snapshot()
                    for kept, now in zip(best_snapshot, current, strict=True):
                        kept[rows] = now[rows]
                    best_loss = torch.where(improved, loss.detach(), best_loss)
            total.backward()
            optimizer.step()
            history.append([float(v) for v in dice.detach().cpu()])
    if bool(torch.isfinite(best_loss).any()):
        rows = torch.nonzero(torch.isfinite(best_loss), as_tuple=False).squeeze(1)
        state.restore_rows(best_snapshot, rows)

    with torch.no_grad():
        final_dice, final_loss = energy(finest, finest_stride)
        centerline = state.centerline()
        width_scale = state.log_width.exp()
        diameter = state.diameter(template)
        # Winner per frame by total energy (overlap plus priors), then one
        # full-resolution render of the winners.
        winners: list[int] = []
        row_start = 0
        for starts in initializations:
            block = final_loss[row_start : row_start + len(starts)]
            winners.append(row_start + int(torch.argmin(block)))
            row_start += len(starts)
        winner_rows = torch.as_tensor(winners, device=device)
        hard = torch.zeros((len(winners), height, width), dtype=torch.bool, device=device)
        for chunk_start in range(0, len(winners), max(1, config.final_render_rows)):
            chunk = winner_rows[chunk_start : chunk_start + config.final_render_rows]
            index = _point_index(centerline.shape[1], 1, device)
            local = centerline[chunk][:, index] - offsets[chunk][:, None, :]
            rendered = render_tube_segments(
                local, diameter[chunk][:, index], height, width, edge_softness=config.edge_softness
            )
            hard[chunk_start : chunk_start + len(chunk)] = rendered >= config.hard_threshold
        hard_np = hard.cpu().numpy()
        centerline_np = centerline.cpu().numpy().astype(np.float64)
        diameter_np = diameter.cpu().numpy().astype(np.float64)
        latent_np = state.latent().cpu().numpy().astype(np.float64)
        width_np = width_scale.cpu().numpy().astype(np.float64)
        shape_np = state.width_shape.detach().cpu().numpy().astype(np.float64)
        initial_np = initial_dice.cpu().numpy()
        final_np = final_dice.cpu().numpy()
        loss_np = final_loss.cpu().numpy()
    history_np = np.asarray(history, dtype=np.float64)

    results: list[MaskFitResult] = []
    row_start = 0
    for f, (mask, starts, w) in enumerate(zip(masks, initializations, windows, strict=True)):
        rows = slice(row_start, row_start + len(starts))
        best_row = winners[f]
        ix0, ix1 = max(w.x0, 0), min(w.x1, w.image_width)
        iy0, iy1 = max(w.y0, 0), min(w.y1, w.image_height)
        clipped = CropWindow(ix0, ix1, iy0, iy1, w.image_height, w.image_width)
        rendered_hard = hard_np[f, iy0 - w.y0 : iy1 - w.y0, ix0 - w.x0 : ix1 - w.x0]
        target_np = mask[iy0:iy1, ix0:ix1]
        iou = hard_iou(rendered_hard, target_np)
        records: list[dict[str, float | str | int]] = []
        for k, start in enumerate(starts):
            row = row_start + k
            records.append(
                {
                    "name": start.name,
                    "initial_soft_dice_energy": float(initial_np[row]),
                    "final_soft_dice_energy": float(final_np[row]),
                    "final_energy": float(loss_np[row]),
                    "final_iou": iou if row == best_row else float("nan"),
                }
            )
        curve = centerline_np[best_row]
        in_fov = int(
            np.sum(
                (curve[:, 0] >= 0) & (curve[:, 0] < w.image_width) & (curve[:, 1] >= 0) & (curve[:, 1] < w.image_height)
            )
        )
        results.append(
            MaskFitResult(
                best_index=best_row - row_start,
                initializations=list(starts),
                records=records,
                latent=latent_np[best_row],
                width_px=float(width_np[best_row]),
                width_profile=diameter_np[best_row],
                centerline_xy=curve,
                crop=clipped,
                rendered_hard_mask=np.ascontiguousarray(rendered_hard),
                energy_history=history_np[:, rows],
                points_in_fov=in_fov,
                body_length_px=float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()),
                width_shape=shape_np[best_row],
            )
        )
        row_start += len(starts)
    return results
