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
from typing import Any, Callable, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn.functional as F

from .head_fit import HeadConstraint, HeadPriors
from .mask_fit import (
    CropWindow,
    Initialization,
    MaskFitConfig,
    MaskFitResult,
    _MaskFitState,
    bend_penalty,
    crop_window,
    default_width_template,
    hard_coverage,
    hard_iou,
    render_tube_segments,
    separation_penalty,
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
    # Opt-in for independent fits without temporal references or head
    # constraints: fuse geometry, overlap and priors with the renderer.
    # Independent of compile_renderer (CUDA or CPU); requires a working
    # compiler and costs more compilation up front. Specialization-limit errors
    # fall back to the normal renderer; other compiler errors still surface.
    # Enable with fit overrides {"compile_energy": true} after benchmarking.
    compile_energy: bool = False
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
# Diagnostic count of groups that exceeded Dynamo's specialization budget.
_ENERGY_COMPILE_FALLBACKS = 0
# Per CUDA device: the side stream steps are recorded on, the memory pool
# every recording shares (a recording is dropped before the next is made),
# and a one-kernel recording that holds the pool: PyTorch releases a pool
# with its last graph and then refuses to record into it again.
_CAPTURE: dict[torch.device, tuple[torch.cuda.Stream, Any, torch.cuda.CUDAGraph]] = {}


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


def _record_step(step: Callable[[], Tensor], device: torch.device) -> tuple[torch.cuda.CUDAGraph, Tensor]:
    """Record one optimization step's CUDA work as a graph; returns it and the step's output, which every replay rewrites.

    An eager step launches hundreds of small kernels from Python; for all but
    the largest groups the launches, not the GPU, set the fit's speed.  A
    replay launches them all at once and computes exactly what the eager step
    computes.
    """

    if device not in _CAPTURE:
        stream, pool, keeper = torch.cuda.Stream(device), torch.cuda.graph_pool_handle(), torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            keeper.capture_begin(pool=pool)
            torch.zeros(1, device=device)
            keeper.capture_end()
        _CAPTURE[device] = (stream, pool, keeper)
    stream, pool, _ = _CAPTURE[device]
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        # Other threads (the app's request handlers) may use the GPU meanwhile.
        graph.capture_begin(pool=pool, capture_error_mode="thread_local")
        try:
            output = step()
        finally:
            graph.capture_end()
    return graph, output


@dataclass(frozen=True)
class BodyFieldEvidence:
    """Body-field network predictions for one frame that a fit is scored against.

    ``ap`` is the full-image A-P field (0 head, 1 tail), NaN where it says
    nothing: off the body and on predicted crossings.  ``head_xy`` and
    ``tail_xy`` are the predicted ends, ``None`` when that end is not in view.
    """

    ap: NDArray[np.float32]
    head_xy: NDArray[np.float64] | None
    tail_xy: NDArray[np.float64] | None


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
    stride = max(1, stride)
    index = torch.arange(0, n_points, stride, device=device)
    if (n_points - 1) % stride:
        index = torch.cat((index, torch.tensor([n_points - 1], device=device)))
    return index


def _window_targets(
    masks: Sequence[BoolArray], windows: Sequence[CropWindow], height: int, width: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Hard target, signed edge distance (-inf off camera), and validity per frame."""

    n = len(masks)
    # Masks and the exact distance transform live on the host. Assemble the
    # whole group here, then transfer each array once instead of sending every
    # crop to the GPU and back for its distance transform and validity checks.
    target = np.zeros((n, height, width), dtype=np.float32)
    valid = np.zeros_like(target)
    distance = np.full_like(target, -float("inf"))
    for f, (mask, w) in enumerate(zip(masks, windows, strict=True)):
        ix0, ix1 = max(w.x0, 0), min(w.x1, w.image_width)
        iy0, iy1 = max(w.y0, 0), min(w.y1, w.image_height)
        local = mask[iy0:iy1, ix0:ix1]
        rows = slice(iy0 - w.y0, iy1 - w.y0)
        cols = slice(ix0 - w.x0, ix1 - w.x0)
        target[f, rows, cols] = local
        valid[f, rows, cols] = 1.0
        if local.all():
            distance[f, rows, cols] = float("inf")
        else:
            distance[f, rows, cols] = signed_edge_distance(torch.from_numpy(local)).numpy()
    return tuple(torch.from_numpy(array).to(device=device) for array in (target, distance, valid))


@torch.no_grad()
def _render_hard_winners(
    centerline: Tensor, diameter: Tensor, height: int, width: int, *,
    edge_softness: float, threshold: float, chunk_rows: int,
) -> Tensor:
    """Render only pixels that can reach the hard threshold in each chunk.

    A segment lies inside its endpoints' bounding box. Expanding that box by
    the largest radius plus the sigmoid's threshold margin contains every
    potentially occupied pixel. Keep pixel coordinates in the original raster
    so cropping does not change floating-point distances at the boundary.
    """

    hard = torch.zeros((len(centerline), height, width), dtype=torch.bool, device=centerline.device)
    effective_threshold = float(torch.tensor(threshold, dtype=centerline.dtype, device="cpu"))
    bounded = 0.0 < effective_threshold < 1.0 and edge_softness > 0
    if bounded:
        bounds = torch.cat((centerline.amin(1), centerline.amax(1), diameter.amax(1)[:, None]), 1).cpu().numpy()
        # One extra pixel makes the bound conservative against rounding in
        # interpolated radii and in the float32 sigmoid near the threshold.
        margin = max(0.0, -edge_softness * (np.log(effective_threshold) - np.log1p(-effective_threshold))) + 1.0
    rows = max(1, chunk_rows)
    for start in range(0, len(centerline), rows):
        stop = min(start + rows, len(centerline))
        x0, y0, x1, y1 = 0, 0, width, height
        if bounded and np.isfinite(bounds[start:stop]).all():
            block = bounds[start:stop]
            radius = max(0.0, float(block[:, 4].max()) * 0.5) + margin
            x0 = max(0, min(width, int(np.floor(block[:, 0].min() - radius))))
            y0 = max(0, min(height, int(np.floor(block[:, 1].min() - radius))))
            x1 = max(0, min(width, int(np.ceil(block[:, 2].max() + radius)) + 1))
            y1 = max(0, min(height, int(np.ceil(block[:, 3].max() + radius)) + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        rendered = render_tube_segments(
            centerline[start:stop], diameter[start:stop], y1 - y0, x1 - x0,
            edge_softness=edge_softness, pixel_origin_xy=(x0, y0),
        )
        hard[start:stop, y0:y1, x0:x1] = rendered >= threshold
    return hard


def _field_tensors(
    fields: Sequence[BodyFieldEvidence | None], windows: Sequence[CropWindow], height: int, width: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per frame: A-P crop and its validity on the batch window raster, ends and their presence."""

    n = len(fields)
    ap = np.zeros((n, height, width), dtype=np.float32)
    valid = np.zeros_like(ap)
    ends = np.zeros((n, 2, 2), dtype=np.float32)
    has_end = np.zeros((n, 2), dtype=np.float32)
    for f, (evidence, w) in enumerate(zip(fields, windows, strict=True)):
        if evidence is None:
            continue
        ix0, ix1 = max(w.x0, 0), min(w.x1, w.image_width)
        iy0, iy1 = max(w.y0, 0), min(w.y1, w.image_height)
        local = np.asarray(evidence.ap, dtype=np.float32)[iy0:iy1, ix0:ix1]
        rows, cols = slice(iy0 - w.y0, iy1 - w.y0), slice(ix0 - w.x0, ix1 - w.x0)
        known = np.isfinite(local)
        ap[f, rows, cols] = np.where(known, local, 0.0)
        valid[f, rows, cols] = known
        for k, point in enumerate((evidence.head_xy, evidence.tail_xy)):
            if point is not None and np.all(np.isfinite(point)):
                ends[f, k], has_end[f, k] = point, 1.0
    return tuple(torch.from_numpy(a).to(device) for a in (ap, valid, ends, has_end))


def _field_penalty(
    centerline: Tensor, origin: Tensor, ap: Tensor, valid: Tensor, ends: Tensor, has_end: Tensor, config: MaskFitConfig
) -> Tensor:
    """Per row: A-P agreement of the midline points and distance of the ends to the predicted ones.

    ``ap`` and ``valid`` are per-row rasters whose pixel (0, 0) is image pixel
    ``origin`` (x, y); ``ends`` holds the predicted head and tail of each row
    and ``has_end`` whether each is in view.
    """

    total = torch.zeros(centerline.shape[0], dtype=centerline.dtype, device=centerline.device)
    if config.field_ap_weight > 0:
        height, width = ap.shape[-2:]
        local = centerline - origin[:, None, :]
        # grid_sample's normalized coordinates with align_corners=False.
        grid = torch.stack((2 * (local[..., 0] + 0.5) / width - 1, 2 * (local[..., 1] + 0.5) / height - 1), -1)[:, None]
        sampled = F.grid_sample(ap[:, None], grid, align_corners=False)[:, 0, 0]
        known = (F.grid_sample(valid[:, None], grid, align_corners=False)[:, 0, 0] > 0.999).to(centerline.dtype)
        position = torch.linspace(0.0, 1.0, centerline.shape[1], device=centerline.device)
        squared = ((sampled - position[None, :]) / config.field_ap_sigma).square()
        total = total + config.field_ap_weight * (squared * known).sum(1) / known.sum(1).clamp_min(1.0)
    if config.field_end_weight > 0:
        squared = (torch.stack((centerline[:, 0], centerline[:, -1]), 1) - ends).square().sum(-1) / config.field_end_sigma_px**2
        total = total + config.field_end_weight * (squared * has_end).sum(1)
    return total


def field_energies(
    centerlines: Sequence[NDArray[np.generic]], fields: Sequence[BodyFieldEvidence | None], config: MaskFitConfig
) -> FloatArray:
    """The evidence energy ``fit_masks`` adds for each centerline against its frame's evidence (0 without either).

    This scores poses the fitter did not produce, such as the mirror of a
    fit.  Each frame's A-P field is read on the centerline's bounding box
    plus a pixel, which holds every pixel the bilinear sampling of a point
    in the image reads; the evidence is NaN off the body, so the fitter's
    window, which holds the whole mask, reads the same values.
    """

    if len(centerlines) != len(fields):
        raise ValueError("fields must align with centerlines")
    out = np.zeros(len(centerlines))
    if config.field_ap_weight <= 0 and config.field_end_weight <= 0:
        return out
    cpu = torch.device("cpu")
    for k, (curve, evidence) in enumerate(zip(centerlines, fields, strict=True)):
        if evidence is None:
            continue
        points = np.ascontiguousarray(curve, dtype=np.float64)
        x0, y0 = (int(v) - 1 for v in np.floor(points.min(0)))
        x1, y1 = (int(v) + 2 for v in np.ceil(points.max(0)))
        window = CropWindow(x0, x1, y0, y1, *evidence.ap.shape)
        ap, valid, ends, has_end = _field_tensors([evidence], [window], window.height, window.width, cpu)
        origin = torch.tensor([[x0, y0]], dtype=torch.float32)
        centerline = torch.as_tensor(points[None], dtype=torch.float32)
        out[k] = float(_field_penalty(centerline, origin, ap, valid, ends, has_end, config)[0])
    return out


def fit_masks(
    masks: Sequence[NDArray[np.generic]],
    initializations: Sequence[Sequence[Initialization]],
    *,
    width_template: NDArray[np.generic] | None = None,
    config: BatchFitConfig = BatchFitConfig(),
    device: torch.device | str | None = None,
    references: Sequence[NDArray[np.generic] | None] | None = None,
    head_constraints: Sequence[HeadConstraint | None] | None = None,
    fields: Sequence[BodyFieldEvidence | None] | None = None,
) -> list[MaskFitResult]:
    """Fit every mask from its own starts; results follow the input order.

    ``references`` gives, per mask, a centerline ``[n_points, 2]`` the fit is
    pulled toward by the temporal prior (``config.temporal_prior_weight``),
    or ``None`` for no pull on that frame.
    ``fields`` gives, per mask, body-field network evidence the fit is
    scored against (``config.field_ap_weight`` / ``field_end_weight``), or
    ``None`` for none on that frame.
    ``head_constraints`` adds tracking/previous-head penalties and projects
    every optimizer iterate into the permitted head movement and image bounds.
    """

    if len(masks) != len(initializations):
        raise ValueError("masks and initializations must align")
    if references is not None and len(references) != len(masks):
        raise ValueError("references must align with masks")
    if head_constraints is not None and len(head_constraints) != len(masks):
        raise ValueError("head_constraints must align with masks")
    if fields is not None and len(fields) != len(masks):
        raise ValueError("fields must align with masks")
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
            references=None if references is None else [references[i] for i in group],
            head_constraints=None if head_constraints is None else [head_constraints[i] for i in group],
            fields=None if fields is None else [fields[i] for i in group],
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
    references: Sequence[NDArray[np.generic] | None] | None = None,
    head_constraints: Sequence[HeadConstraint | None] | None = None,
    fields: Sequence[BodyFieldEvidence | None] | None = None,
) -> list[MaskFitResult]:
    # Whole-energy compilation is measured only for independent fits. The
    # changing references/head constraints in sequential fitting can generate
    # unstable specialized GPU kernels, so those calls retain normal rendering.
    use_fields = fields is not None and any(f is not None for f in fields) and (
        config.field_ap_weight > 0 or config.field_end_weight > 0
    )
    compile_energy = config.compile_energy and references is None and head_constraints is None and not use_fields
    renderer = get_renderer(config.compile_renderer and not compile_energy)
    windows, height, width = batch_windows(crops)
    target, distance, valid = _window_targets(masks, windows, height, width, device)
    starts_flat = [s for starts in initializations for s in starts]
    frame_indices = [f for f, starts in enumerate(initializations) for _ in starts]
    frame_of_row = torch.as_tensor(frame_indices, device=device)
    row_windows = [windows[f] for f in frame_indices]
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
    # Temporal prior: per row, the reference centerline of its frame and a
    # mask of the reference points inside the camera (off-camera body is
    # unconstrained and must not pull); rows without a reference get zero.
    use_temporal = config.temporal_prior_weight > 0 and references is not None and any(r is not None for r in references)
    if use_temporal:
        reference_rows = []
        reference_masks = []
        for f, starts in enumerate(initializations):
            ref = references[f]
            w = windows[f]
            for _ in starts:
                if ref is None:
                    reference_rows.append(np.zeros((config.n_points, 2), dtype=np.float32))
                    reference_masks.append(np.zeros(config.n_points, dtype=np.float32))
                else:
                    points = np.asarray(ref, dtype=np.float32)
                    if points.shape != (config.n_points, 2):
                        raise ValueError("reference centerlines must have shape [n_points, 2]")
                    inside = (points[:, 0] >= 0) & (points[:, 0] < w.image_width) & (points[:, 1] >= 0) & (points[:, 1] < w.image_height)
                    reference_rows.append(points)
                    reference_masks.append(inside.astype(np.float32))
        reference_t = torch.as_tensor(np.stack(reference_rows), device=device)
        reference_mask_t = torch.as_tensor(np.stack(reference_masks), device=device)
        temporal_scale = config.temporal_prior_weight / float(config.temporal_prior_sigma_px) ** 2
        # Rows without a reference have no deliberate crossings: NaN compares false.
        has_reference = reference_mask_t.sum(1, keepdim=True)[..., None] > 0
        crossing_reference_t = torch.where(has_reference, reference_t, torch.full_like(reference_t, float("nan")))

    if use_fields:
        field_ap, field_valid, field_ends, field_has_end = _field_tensors(fields, windows, height, width, device)

    def field_energy(centerline: Tensor) -> Tensor:
        return _field_penalty(
            centerline, offsets, field_ap[frame_of_row], field_valid[frame_of_row], field_ends[frame_of_row],
            field_has_end[frame_of_row], config,
        )

    state = _MaskFitState(starts_flat, config, device)
    head_priors = None if head_constraints is None else HeadPriors(
        [head_constraints[f] for f, starts in enumerate(initializations) for _ in starts], camera_size,
    )

    def constrain_head() -> None:
        if head_priors is not None:
            with torch.no_grad():
                head = state.centerline()[:, 0]
                state.centroid.add_(head_priors.project(head) - head)

    constrain_head()
    optimizer = state.optimizer()
    base_rates = [group["lr"] for group in optimizer.param_groups]
    finest = min(config.stage_downsample)
    eps = torch.finfo(torch.float32).eps
    stage_cache: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
    point_indices = {
        stride: slice(None) if stride <= 1 else _point_index(config.n_points, stride, device)
        for stride in set(config.stage_point_stride)
    }

    def stage_arrays(factor: int) -> tuple[Tensor, Tensor, Tensor]:
        if factor not in stage_cache:
            soft = torch.sigmoid(distance / (config.edge_softness * factor))
            soft = _downsample_batch(soft, factor)[frame_of_row]
            weight = _downsample_batch(valid, factor)[frame_of_row]
            stage_cache[factor] = (soft, weight, (soft * weight).sum((1, 2)))
        return stage_cache[factor]

    def regularization(centerline: Tensor, diameter: Tensor) -> Tensor:
        c = config
        smooth = c.shape_smoothness * (state.shape[:, 1:] - state.shape[:, :-1]).square().mean(1)
        below = ((edge_lo[:, None, :] - centerline).clamp_min(0).square() * lo_active[:, None, :]).sum(-1)
        above = ((centerline - edge_hi[:, None, :]).clamp_min(0).square() * hi_active[:, None, :]).sum(-1)
        # Points past the camera edge are censored: no data, no escape penalty.
        inside_camera = ((centerline >= 0) & (centerline < camera_size[:, None, :])).all(-1).to(centerline.dtype)
        escape = ((below + above) * inside_camera).mean(1)
        total = smooth + state.size_regularization() + c.crop_escape_weight * escape + state.width_prior()
        total = total + bend_penalty(centerline, state.log_length.exp(), state.log_width.exp(), c)
        total = total + separation_penalty(centerline, diameter, c, crossing_reference_t if use_temporal else None)
        if use_temporal:
            squared = ((centerline - reference_t).square().sum(-1) * reference_mask_t).sum(1) / reference_mask_t.sum(1).clamp_min(1.0)
            total = total + temporal_scale * squared
        if head_priors is not None:
            total = total + head_priors.energy(centerline[:, 0])
        if use_fields:
            total = total + field_energy(centerline)
        return total

    def render_rows(centerline: Tensor, diameter: Tensor, factor: int, stride: int, fn: Renderer) -> Tensor:
        index = point_indices[stride]
        local = (centerline[:, index] - offsets[:, None, :] + 0.5) / factor - 0.5
        return fn(
            local, diameter[:, index] / factor, height // factor, width // factor, edge_softness=config.edge_softness
        )

    def energy(factor: int, stride: int) -> tuple[Tensor, Tensor]:
        centerline = state.centerline()
        diameter = state.diameter(template)
        rendered = render_rows(centerline, diameter, factor, stride, renderer)
        soft, weight, target_mass = stage_arrays(factor)
        intersection = (rendered * soft * weight).sum((1, 2))
        denominator = (rendered * weight).sum((1, 2)) + target_mass
        dice = 1 - (2 * intersection + eps) / (denominator + eps)
        return dice, dice + regularization(centerline, diameter)

    if compile_energy:
        # Mutating this dictionary after capture would invalidate Dynamo's
        # guards. All fixed data is ready before tracing the two stage shapes.
        for factor in set(config.stage_downsample):
            stage_arrays(factor)
        from torch._dynamo.exc import FailOnRecompileLimitHit

        eager_energy = energy
        compiled_energy = torch.compile(eager_energy, dynamic=True, fullgraph=True)

        def evaluate_energy(factor: int, stride: int) -> tuple[Tensor, Tensor]:
            nonlocal compiled_energy, renderer
            global _ENERGY_COMPILE_FALLBACKS
            if compiled_energy is not None:
                try:
                    return compiled_energy(factor, stride)
                except FailOnRecompileLimitHit:
                    # Different group shapes and configurations can exhaust
                    # Dynamo's shared guard cache. Keep this group's
                    # optimizer and schedule, returning to the normal renderer
                    # path for all its remaining evaluations.
                    compiled_energy = None
                    renderer = get_renderer(config.compile_renderer)
                    _ENERGY_COMPILE_FALLBACKS += 1
                    if _ENERGY_COMPILE_FALLBACKS == 1:
                        warnings.warn(
                            "whole-energy compilation exceeded its specialization limit; "
                            "continuing affected groups with the normal renderer",
                            RuntimeWarning, stacklevel=2,
                        )
            return eager_energy(factor, stride)

        energy = evaluate_energy

    finest_stride = config.stage_point_stride[config.stage_downsample.index(finest)]
    with torch.no_grad():
        initial_dice, _ = energy(finest, finest_stride)
    best_loss = torch.full((len(starts_flat),), float("inf"), device=device)
    best_snapshot = state.parameter_snapshot()
    # Checked once after the schedule: a check per step would make the host
    # wait for every step's kernels.
    finite = torch.ones((), dtype=torch.bool, device=device)
    # Keep the small history on the fitting device and transfer it once. A
    # host copy inside every iteration forces CUDA to finish every queued op.
    history = torch.empty((sum(config.stage_steps), len(starts_flat)), device=device)
    history_step = 0

    def step_gradients(factor: int, stride: int) -> Tensor:
        """The energy, the best state at the finest stage, and the gradients of one step; returns the soft Dice."""

        dice, loss = energy(factor, stride)
        total = loss.sum()
        finite.logical_and_(torch.isfinite(total))
        if factor == finest:
            improved = loss.detach() < best_loss
            with torch.no_grad():
                for kept, now in zip(best_snapshot, state.parameters(), strict=True):
                    selected = improved.reshape(-1, *([1] * (now.ndim - 1)))
                    torch.where(selected, now, kept, out=kept)
                torch.where(improved, loss.detach(), best_loss, out=best_loss)
        total.backward()
        return dice.detach()

    # On CUDA every step after a stage's first replays a recording of it; the
    # first runs eagerly, so the renderer is compiled for the stage's raster
    # outside the recording.  The optimizer steps eagerly, since its learning
    # rate and bias correction change every step.  The crossing exemption of
    # the separation penalty reads its reach on the host, so fits that use it
    # stay eager.
    record = device.type == "cuda" and not (use_temporal and config.separation_weight > 0)
    for factor, steps, scale, stride in zip(
        config.stage_downsample, config.stage_steps, config.stage_lr_scale, config.stage_point_stride, strict=True
    ):
        graph = None
        for step in range(steps):
            progress = step / max(steps - 1, 1)
            decay = 1.0 + (config.within_stage_decay - 1.0) * progress
            for group, base in zip(optimizer.param_groups, base_rates, strict=True):
                group["lr"] = base * scale * decay
            if graph is None:
                optimizer.zero_grad(set_to_none=True)
                dice = step_gradients(factor, stride)
            else:
                graph.replay()
            optimizer.step()
            constrain_head()
            history[history_step].copy_(dice)
            history_step += 1
            if record and graph is None and step + 1 < steps:
                # The recording's backward writes fresh gradient tensors, which every replay overwrites.
                optimizer.zero_grad(set_to_none=True)
                graph, dice = _record_step(lambda: step_gradients(factor, stride), device)
    if not bool(finite):
        raise RuntimeError("non-finite batch mask-fit energy")
    if bool(torch.isfinite(best_loss).any()):
        rows = torch.nonzero(torch.isfinite(best_loss), as_tuple=False).squeeze(1)
        state.restore_rows(best_snapshot, rows)

    with torch.no_grad():
        final_dice, final_loss = energy(finest, finest_stride)
        centerline = state.centerline()
        final_field = field_energy(centerline) if use_fields else torch.zeros_like(final_dice)
        width_scale = state.log_width.exp()
        diameter = state.diameter(template)
        # Winner per frame by total energy (overlap plus priors), then one
        # full-resolution render of the winners.
        winners: list[int] = []
        loss_np = final_loss.cpu().numpy()
        row_start = 0
        for starts in initializations:
            block = loss_np[row_start : row_start + len(starts)]
            winners.append(row_start + int(np.argmin(block)))
            row_start += len(starts)
        winner_rows = torch.as_tensor(winners, device=device)
        hard = _render_hard_winners(
            centerline[winner_rows] - offsets[winner_rows, None, :], diameter[winner_rows],
            height, width, edge_softness=config.edge_softness,
            threshold=config.hard_threshold, chunk_rows=config.final_render_rows,
        )
        hard_np = hard.cpu().numpy()
        centerline_np = centerline.cpu().numpy().astype(np.float64)
        diameter_np = diameter.cpu().numpy().astype(np.float64)
        latent_np = state.latent().cpu().numpy().astype(np.float64)
        width_np = width_scale.cpu().numpy().astype(np.float64)
        shape_np = state.width_shape.detach().cpu().numpy().astype(np.float64)
        initial_np = initial_dice.cpu().numpy()
        final_np = final_dice.cpu().numpy()
        field_np = final_field.cpu().numpy()
    history_np = history.cpu().numpy().astype(np.float64)

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
        coverage = hard_coverage(rendered_hard, target_np)
        records: list[dict[str, float | str | int]] = []
        for k, start in enumerate(starts):
            row = row_start + k
            records.append(
                {
                    "name": start.name,
                    "initial_soft_dice_energy": float(initial_np[row]),
                    "final_soft_dice_energy": float(final_np[row]),
                    # The body-field evidence part of ``final_energy`` (0 without evidence).
                    "final_field_energy": float(field_np[row]),
                    "final_energy": float(loss_np[row]),
                    "final_iou": iou if row == best_row else float("nan"),
                    "final_coverage": coverage if row == best_row else float("nan"),
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
