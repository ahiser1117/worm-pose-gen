"""Midline traces proposed by the body-field network.

The network (:mod:`body_net`) predicts, for a frame and its temporal
context, the worm mask, the A-P field, head and tail heatmaps, and where the
body crosses itself.  A trace like a hand-clicked one follows from them: the
predicted head, then one point per band of predicted A-P values across the
labeled mask (the centre of the band's patch of body, skipping pixels
predicted as a crossing), then the predicted tail.  Because the A-P field
changes across a contact line, a band picks out one limb even where two
touch, which is what a skeleton of the mask cannot do.  The trace is then fit
like a hand trace (:func:`body_fields.trace_fit`), or the fields score an
ordinary fit (:func:`field_evidence`, :class:`batch_fit.BodyFieldEvidence`)
and the trace starts it (:func:`trace_start`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
import torch
from torch import Tensor
import torch.nn.functional as F

from .batch_fit import BodyFieldEvidence
from .body_net import OUTPUTS, BodyFieldModule
from .mask_fit import (
    Initialization, MaskFitConfig, at_length, extend_start_to_length, init_from_centerline, lay_centerline,
)
from .segmenter import INPUT_MEAN, INPUT_STD


FloatArray = NDArray[np.float64]

TRACE_LEVELS = 20
# A heatmap peak below this means the end is not in view.
END_THRESHOLD = 0.3
# An end predicted this close to the image edge does not score fits: where
# the body leaves the camera the tail heatmap fires (edge_0528: peaks of
# 0.3-0.47 on 31 frames, switching on and off), and an end term pinning the
# fit's tail there pulls the whole body into view.
END_BORDER_PX = 20.0
# A peak farther than this from the cleaned mask is on a part the cleanup
# removed (tail_reentry_0623: a re-entering tail tip cut off as its own
# component, 20-180 px away), and an end pulled there drags the fit off the
# body; such an end is not used.  Real tails lie up to 16 px off reviewed
# masks, which miss the thin tip.
END_MASK_PX = 20.0
# The head heatmap's modes kept per frame for the head track (:mod:`head_track`):
# local maxima at least ``HEAD_MODE_RADIUS_PX`` apart (the maximum of a
# square of twice that plus one; about a body width, 44 px at nir-flv), the
# strongest ``HEAD_MODES`` of those reaching ``HEAD_MODE_FLOOR``.
HEAD_MODES = 4
HEAD_MODE_RADIUS_PX = 22
HEAD_MODE_FLOOR = 0.02
# Pixels predicted as a crossing above this are left out of the bands.
OVERLAP_THRESHOLD = 0.5
MIN_BAND_PIXELS = 20
# A band's real cross-section is its largest patch; patches much smaller are
# A-P noise along a contact line and are not followed.
MIN_PATCH_FRACTION = 0.5
# A band's point is its pixel nearest the band's centroid among the most
# central ones (distance to the mask edge at least this share of the band's maximum).
CENTRAL_FRACTION = 0.7


# A heatmap's maximum: (x, y, value).
Peak = tuple[float, float, float]
Box = tuple[slice, slice]


def heatmap_modes(heatmaps: Tensor, k: int = HEAD_MODES, radius: int = HEAD_MODE_RADIUS_PX, floor: float = HEAD_MODE_FLOOR) -> Tensor:
    """``[B, k, 3]`` (x, y, value) modes of ``[B, H, W]`` heatmaps, strongest first, NaN past the last.

    A mode is a pixel equal to the maximum of the square of half-width
    ``radius`` around it (a separable max-pool) and at least ``floor``.  The
    network's half-precision outputs often tie between neighbouring pixels;
    of tied maxima within one square only the first in raster order is a
    mode, so the first mode is the pixel ``np.argmax`` finds.
    """

    size = 2 * radius + 1
    pooled = F.max_pool2d(heatmaps[:, None], (1, size), 1, (0, radius))
    pooled = F.max_pool2d(pooled, (size, 1), 1, (radius, 0))[:, 0]
    candidates = torch.where((heatmaps >= pooled) & (heatmaps >= floor), heatmaps, torch.full_like(heatmaps, -1.0))
    values, index = candidates.flatten(1).topk(4 * k, 1)
    width = heatmaps.shape[-1]
    x, y = index % width, index // width
    near = ((x[:, :, None] - x[:, None, :]).abs() <= radius) & ((y[:, :, None] - y[:, None, :]).abs() <= radius)
    tied = near & (values[:, :, None] == values[:, None, :]) & (index[:, None, :] < index[:, :, None])
    values = torch.where(tied.any(-1), -1.0, values)
    values, order = values.topk(k, 1)
    index = index.gather(1, order)
    modes = torch.stack(((index % width).double(), (index // width).double(), values.double()), -1)
    return torch.where((values >= floor)[..., None], modes, torch.full_like(modes, float("nan")))


def in_view(peak: Peak | NDArray[np.float64]) -> tuple[float, float] | None:
    """A peak's (x, y) when it reaches ``END_THRESHOLD``."""

    return (float(peak[0]), float(peak[1])) if float(peak[2]) >= END_THRESHOLD else None


@dataclass(frozen=True)
class FieldPrediction:
    """Per-pixel network outputs for one frame, as probabilities (A-P as a value in [0, 1]).

    The trace and the evidence read a prediction only through :meth:`end`,
    :meth:`ap_in` and :meth:`crossing_in`, which :class:`FieldCrop` (the part
    a workspace keeps) answers too.
    """

    mask: NDArray[np.float32]
    ap: NDArray[np.float32]
    head: NDArray[np.float32]
    tail: NDArray[np.float32]
    overlap: NDArray[np.float32]

    def peak(self, name: str) -> Peak:
        """The maximum of the ``head`` or ``tail`` heatmap."""

        heatmap = getattr(self, name)
        index = int(np.argmax(heatmap))
        y, x = np.unravel_index(index, heatmap.shape)
        return float(x), float(y), float(heatmap.flat[index])

    def head_modes(self) -> NDArray[np.float64]:
        return heatmap_modes(torch.from_numpy(np.ascontiguousarray(self.head, dtype=np.float32))[None])[0].numpy()

    def end(self, name: str) -> tuple[float, float] | None:
        """The ``head`` or ``tail`` in view: the heatmap's maximum when it reaches ``END_THRESHOLD``."""

        return in_view(self.peak(name))

    def ap_in(self, box: Box) -> NDArray[np.float32]:
        return self.ap[box]

    def crossing_in(self, box: Box) -> NDArray[np.bool_]:
        """Pixels predicted as a crossing (overlap at least ``OVERLAP_THRESHOLD``)."""

        return self.overlap[box] >= OVERLAP_THRESHOLD


# The stored region of a frame's fields reaches this far past its mask, so a
# mask edit that moves the edge a little still lies inside it.
CROP_MARGIN_PX = 2


@dataclass(frozen=True)
class FieldCrop:
    """What the fit reads of a prediction (:mod:`field_store` keeps one per workspace row).

    The A-P field (float32, as predicted: rounding it moves fits) and the
    crossings on ``support``, the frame's mask grown by
    ``CROP_MARGIN_PX`` within its box ``(y0, y1, x0, x1)``, the head
    heatmap's modes (``heads``, :func:`heatmap_modes`) and the tail
    heatmap's peak.  ``head`` is the head the fit uses: the strongest mode
    when it reaches ``END_THRESHOLD`` (:func:`crop_fields`), or the
    recording's head track (:meth:`with_head`, :mod:`head_track`); ``None``
    is no head in view.  :meth:`covers` says whether a mask lies on the
    support, which is where the trace and the evidence read the fields;
    elsewhere A-P is NaN and nothing is a crossing.
    """

    shape: tuple[int, int]
    box: tuple[int, int, int, int]
    support: NDArray[np.bool_]
    ap: NDArray[np.float32]
    crossing: NDArray[np.bool_]
    heads: NDArray[np.float64]
    tail: Peak
    head: tuple[float, float] | None

    def head_modes(self) -> NDArray[np.float64]:
        return self.heads

    def end(self, name: str) -> tuple[float, float] | None:
        return self.head if name == "head" else in_view(self.tail)

    def with_head(self, xy: NDArray[np.float64] | Sequence[float]) -> FieldCrop:
        """This crop with ``xy`` as its head (NaN: no head in view)."""

        x, y = (float(v) for v in xy)
        return replace(self, head=(x, y) if np.isfinite(x) and np.isfinite(y) else None)

    def _in(self, values: NDArray[Any], box: Box, fill: Any) -> NDArray[Any]:
        rows, cols = (range(*s.indices(n)) for s, n in zip(box, self.shape))
        y0, y1, x0, x1 = self.box
        out = np.full((len(rows), len(cols)), fill, dtype=values.dtype)
        top, bottom, left, right = max(rows.start, y0), min(rows.stop, y1), max(cols.start, x0), min(cols.stop, x1)
        if top < bottom and left < right:
            out[top - rows.start : bottom - rows.start, left - cols.start : right - cols.start] = values[top - y0 : bottom - y0, left - x0 : right - x0]
        return out

    def ap_in(self, box: Box) -> NDArray[np.float32]:
        return self._in(self.ap, box, np.nan)

    def crossing_in(self, box: Box) -> NDArray[np.bool_]:
        return self._in(self.crossing, box, False)

    def covers(self, mask: NDArray[np.generic]) -> bool:
        body = np.asarray(mask, dtype=bool)
        if body.shape != self.shape:
            return False
        y0, y1, x0, x1 = self.box
        inside = body[y0:y1, x0:x1]
        return int(np.count_nonzero(inside & self.support)) == int(np.count_nonzero(body))


def crop_fields(prediction: FieldPrediction, mask: NDArray[np.generic]) -> FieldCrop:
    """The part of ``prediction`` the fit of ``mask`` reads (an empty region and the ends for an empty mask)."""

    body = np.asarray(mask, dtype=bool)
    height, width = body.shape
    box = _box(body) or (slice(0, 0), slice(0, 0))
    margin = CROP_MARGIN_PX if body.any() else 0
    y0, y1 = max(box[0].start - margin, 0), min(box[0].stop + margin, height)
    x0, x1 = max(box[1].start - margin, 0), min(box[1].stop + margin, width)
    region = (slice(y0, y1), slice(x0, x1))
    support = ndimage.binary_dilation(body[region], np.ones((3, 3), bool), iterations=margin) if margin else body[region]
    ap = np.where(support, prediction.ap_in(region), np.nan).astype(np.float32)
    return FieldCrop(
        shape=(height, width), box=(y0, y1, x0, x1), support=support, ap=ap,
        crossing=prediction.crossing_in(region) & support, heads=prediction.head_modes(), tail=prediction.peak("tail"),
        head=prediction.end("head"),
    )


# Body-field evidence weights for ordinary fits (``MaskFitConfig.field_*``).
# On the 66 reviewed held-out frames (scripts/evaluate_field_fitting.py) the
# trace start plus these terms put all 32 hand-traced contacts and coils
# within half a body width of the hand trace (15 without) with every head at
# the right end; an A-P weight of 0.01 or more pulled fits off the mask edges.
FIELD_AP_WEIGHT = 0.001
FIELD_END_WEIGHT = 0.01


def network_inputs(frames: Mapping[int, Tensor], targets: Sequence[int], lags: Sequence[int]) -> Tensor:
    """The network's ``[B, 1 + len(lags), H, W]`` input stacks of frames ``targets``: the normalized frame, then one difference per lag.

    ``frames`` maps a frame's index to it (uint8 ``[H, W]``, on the device
    the stacks are built on); the difference of lag ``lag`` is
    ``frame[t + lag] - frame[t - lag]``, and a lag reaching a frame missing
    from ``frames`` (one that does not exist) gets a zero channel, as in
    training (:func:`temporal_context.difference_channels`).  The arithmetic
    is float32, as the network was trained on.
    """

    centre = torch.stack([frames[f] for f in targets]).float()
    # Divisors as tensors on the device: CUDA divides by a host scalar as a
    # multiplication by its reciprocal, which rounds differently from NumPy.
    full, std, step = torch.tensor([255.0, INPUT_STD, 255.0 * INPUT_STD], dtype=torch.float32, device=centre.device)
    channels = [(centre / full - INPUT_MEAN) / std]
    zero = torch.zeros_like(frames[targets[0]])
    for lag in lags:
        present = torch.tensor([f + lag in frames and f - lag in frames for f in targets], device=centre.device)
        later = torch.stack([frames.get(f + lag, zero) for f in targets]).float()
        earlier = torch.stack([frames.get(f - lag, zero) for f in targets]).float()
        channels.append(torch.where(present[:, None, None], (later - earlier) / step, 0.0))
    return torch.stack(channels, 1)


@torch.inference_mode()
def _probabilities(module: BodyFieldModule, inputs: Tensor) -> Tensor:
    """The network's ``[B, 5, H, W]`` sigmoids (float32) on its device."""

    with torch.autocast(device_type=module.device.type, enabled=module.device.type == "cuda"):
        logits = module(inputs.to(module.device)).float()
    return torch.sigmoid(logits)


def _predictions(probability: Tensor) -> list[FieldPrediction]:
    maps = probability.cpu().numpy()
    return [FieldPrediction(**{name: m[k] for k, name in enumerate(OUTPUTS)}) for m in maps]


@dataclass(frozen=True)
class FrameFields:
    """What the pipeline reads of a frame's prediction: the mask probability, A-P, crossings, the head's modes and the tail's peak.

    The mask and A-P are the network's float32 sigmoids, as in a
    :class:`FieldPrediction`; the head heatmap is reduced to its modes
    (:func:`heatmap_modes`), the tail heatmap to its peak and the overlap to
    its crossings on the device, so a frame moves less than half of a full
    prediction to the host.  It answers :meth:`peak`, :meth:`head_modes`,
    :meth:`end`, :meth:`ap_in` and :meth:`crossing_in` as a
    :class:`FieldPrediction` does.
    """

    mask: NDArray[np.float32]
    ap: NDArray[np.float32]
    crossing: NDArray[np.bool_]
    heads: NDArray[np.float64]
    tail: Peak

    def peak(self, name: str) -> Peak:
        return tuple(float(v) for v in self.heads[0]) if name == "head" else self.tail

    def head_modes(self) -> NDArray[np.float64]:
        return self.heads

    def end(self, name: str) -> tuple[float, float] | None:
        return in_view(self.peak(name))

    def ap_in(self, box: Box) -> NDArray[np.float32]:
        return self.ap[box]

    def crossing_in(self, box: Box) -> NDArray[np.bool_]:
        return self.crossing[box]


def _to_host(values: Tensor) -> Tensor:
    """A copy of ``values`` on the host, started without waiting (into pinned memory) when they are on a GPU."""

    if not values.is_cuda:
        return values.cpu()
    host = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
    return host.copy_(values, non_blocking=True)


class _FieldTransfer:
    """A batch's :class:`FrameFields` on their way to the host.

    The head's modes, the tail's peak and the crossings are found on the device and the copies are only
    started, so the host can read the next batch's frames while the device
    finishes this one; :meth:`fields` waits for them.
    """

    def __init__(self, probability: Tensor) -> None:
        width = probability.shape[-1]
        # The tail's first maximum, as np.argmax finds it in a FieldPrediction.
        value, index = probability[:, OUTPUTS.index("tail")].flatten(1).max(1)
        tail = torch.stack(((index % width).double(), (index // width).double(), value.double()), -1)
        heads = heatmap_modes(probability[:, OUTPUTS.index("head")])
        maps = probability[:, : OUTPUTS.index("ap") + 1]
        crossing = probability[:, OUTPUTS.index("overlap")] >= OVERLAP_THRESHOLD
        self.host = [_to_host(t) for t in (maps, crossing, heads, tail)]
        self.done = None
        if probability.is_cuda:
            self.done = torch.cuda.Event()
            self.done.record()

    def fields(self) -> list[FrameFields]:
        if self.done is not None:
            self.done.synchronize()
        maps, crossing, heads, tails = (t.numpy() for t in self.host)
        return [
            FrameFields(mask=m[0], ap=m[1], crossing=c, heads=h, tail=tuple(float(v) for v in t))
            for m, c, h, t in zip(maps, crossing, heads, tails)
        ]


def predict_fields(module: BodyFieldModule, context: NDArray[np.uint8], valid: NDArray[np.bool_]) -> FieldPrediction:
    """Run the network on the centre of a context stack (as stored in a body-field record)."""

    centre = context.shape[0] // 2
    if module.lags and max(module.lags) > centre:
        raise ValueError(f"the model needs lags up to {max(module.lags)}; the context reaches {centre}")
    frames = {k: torch.as_tensor(context[k]).to(module.device) for k in range(len(context)) if valid[k]}
    frames[centre] = torch.as_tensor(context[centre]).to(module.device)
    return _predictions(_probabilities(module, network_inputs(frames, [centre], module.lags)))[0]


# Frames run through the network at once.  At 732x968 on an RTX 6000 Ada the
# forward pass costs 5.0 ms/frame at 8 and 5.3 at 16 (4.7 and 4.9 with
# channels_last, cudnn.benchmark no different), and the outputs depend on the
# batch size in the fourth decimal (cuDNN's kernel choice), so it stays at the
# batch the fits were tuned with.
PREDICT_BATCH = 8


class RecordingFieldPredictor:
    """Body-field predictions for frames of one recording, from their flat-fielded neighbours.

    The frames are flat-fielded and stacked into the network's inputs on the
    module's device.  The frames of a call stay there for the next: the
    slabs a stage predicts one after another share their neighbours, which
    are then not read and corrected again.
    """

    def __init__(self, module: BodyFieldModule, frames: Any, *, batch_size: int = PREDICT_BATCH) -> None:
        self.module = module
        self.frames = frames
        self.batch_size = batch_size
        self._corrected: dict[int, Tensor] = {}

    def _probabilities(self, frame_indices: Sequence[int]) -> Iterator[Tensor]:
        """The sigmoids of these source frames, a batch at a time; neighbours outside the recording give zero channels.

        A batch's frames are read when the batch is asked for, so the reading
        overlaps the device's work on the batch before.
        """

        lags = self.module.lags
        targets = [int(f) for f in frame_indices]
        frames: dict[int, Tensor] = {}
        for start in range(0, len(targets), self.batch_size):
            batch = targets[start : start + self.batch_size]
            needed = sorted({f + o for f in batch for o in (0, *lags, *(-lag for lag in lags)) if 0 <= f + o < self.frames.total} - frames.keys())
            frames.update((f, self._corrected[f]) for f in needed if f in self._corrected)
            new = [f for f in needed if f not in frames]
            if new:
                frames.update(zip(new, self.frames.corrected_on(new, self.module.device)))
            yield _probabilities(self.module, network_inputs(frames, batch, lags))
        self._corrected = frames

    def predict(self, frame_indices: Sequence[int]) -> list[FrameFields]:
        """What the pipeline reads of each source frame's prediction, in order."""

        results: list[FrameFields] = []
        pending: _FieldTransfer | None = None
        for probability in self._probabilities(frame_indices):
            transfer = _FieldTransfer(probability)
            if pending is not None:
                results.extend(pending.fields())
            pending = transfer
        if pending is not None:
            results.extend(pending.fields())
        return results

    def predict_full(self, frame_indices: Sequence[int]) -> list[FieldPrediction]:
        """Every output map of each source frame's prediction (the app's views of the raw channels)."""

        return [prediction for probability in self._probabilities(frame_indices) for prediction in _predictions(probability)]


def _end(point: tuple[float, float] | None, body: NDArray[np.bool_]) -> FloatArray | None:
    """An end in view (a prediction's ``end(name)``) as (x, y); ``None`` without one or more than ``END_MASK_PX`` off ``body``."""

    if point is None:
        return None
    x, y = int(point[0]), int(point[1])
    reach = int(np.ceil(END_MASK_PX))
    top, left = max(y - reach, 0), max(x - reach, 0)
    ys, xs = np.nonzero(body[top : y + reach + 1, left : x + reach + 1])
    if not len(ys) or float(np.hypot(ys + top - y, xs + left - x).min()) > END_MASK_PX:
        return None
    return np.array([x, y], dtype=np.float64)


def head_candidates(
    prediction: FieldPrediction | FrameFields | FieldCrop, mask: NDArray[np.bool_], nose_xy: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A frame's head candidates for :mod:`head_track`: the head modes ``[K, 3]`` and the nose ``[2]``, NaN where more than ``END_MASK_PX`` off ``mask``.

    The end rule of :func:`field_evidence` applied before the track, so the
    track picks among heads the fit can use.
    """

    body = np.asarray(mask, dtype=bool)
    modes = np.array(prediction.head_modes(), dtype=np.float64)
    for mode in modes:
        if np.isfinite(mode).all() and _end((mode[0], mode[1]), body) is None:
            mode[:] = np.nan
    nose = np.full(2, np.nan) if nose_xy is None else np.array(nose_xy, dtype=np.float64)
    if np.isfinite(nose).all() and _end((nose[0], nose[1]), body) is None:
        nose[:] = np.nan
    return modes, nose


def _box(mask: NDArray[np.bool_]) -> tuple[slice, slice] | None:
    """Rows and columns of the mask's bounding box plus one pixel; ``None`` for an empty mask.

    The extra pixel is background (or the box meets the image edge), so the
    distance transform of the crop finds the same nearest background pixels
    as that of the whole image.
    """

    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        return None
    return slice(max(int(rows[0]) - 1, 0), int(rows[-1]) + 2), slice(max(int(cols[0]) - 1, 0), int(cols[-1]) + 2)


def _scored(point: FloatArray | None, shape: tuple[int, ...]) -> FloatArray | None:
    """``point`` unless it lies within ``END_BORDER_PX`` of the image edge."""

    height, width = shape[:2]
    return None if point is None or min(point[0], point[1], width - 1 - point[0], height - 1 - point[1]) <= END_BORDER_PX else point


def propose_trace(
    prediction: FieldPrediction | FieldCrop, mask: NDArray[np.bool_], *, levels: int = TRACE_LEVELS, tail: bool = True,
) -> FloatArray | None:
    """Head-first trace points over ``mask`` from a prediction; ``None`` when it gives fewer than three.

    Each A-P band contributes one point from one connected patch of mask
    pixels.  Patches under half the size of the band's largest are A-P noise
    along a contact line and are skipped; of the rest, the walk takes the one
    nearest the previous point (the largest for the first band without a
    head), so it stays on the limb it is following.  The point is the
    patch's most central pixel (by distance to the mask edge) nearest its
    centroid, which stays on the body even where a band bends.
    Points closer than a third of a body width to the previous one are
    dropped (except the tail, which replaces its neighbour).  An end whose heatmap stays below ``END_THRESHOLD`` is left
    out, so a trace whose body leaves the camera stops at the last band
    (which :func:`trace_start` continues off camera); so is an end whose
    peak lies more than ``END_MASK_PX`` off the mask.  Without ``tail`` the
    trace stops at the last band either way.
    """

    body = np.asarray(mask, dtype=bool)
    box = _box(body)
    if box is None:
        return None
    head = _end(prediction.end("head"), body)
    tail = _end(prediction.end("tail"), body) if tail else None
    # Everything below works on the mask's box; ``origin`` is its corner in the image.
    origin = np.array([box[1].start, box[0].start])
    body = body[box]
    ys, xs = np.nonzero(body & ~prediction.crossing_in(box))
    if not len(ys):
        return None
    depth = ndimage.distance_transform_edt(body)
    diameter = 2.0 * float(depth.max())
    points: list[FloatArray] = [] if head is None else [head]
    # Band k holds edges[k] <= A-P < edges[k + 1], the last one closed at 1.
    edges = np.linspace(0.0, 1.0, levels + 1)
    ap = prediction.ap_in(box)[ys, xs]
    level = np.searchsorted(edges, ap, side="right") - 1
    level[ap == edges[-1]] = levels - 1
    order = np.argsort(level, kind="stable")  # pixels grouped by band, in raster order within one
    bounds = np.searchsorted(level[order], np.arange(levels + 1))
    for k in range(levels):
        members = order[bounds[k] : bounds[k + 1]]
        if len(members) < MIN_BAND_PIXELS:  # no patch can be large enough
            continue
        # The band's patches, labeled on the band's own box.
        by, bx = ys[members], xs[members]
        top, left = int(by.min()), int(bx.min())
        patch = np.zeros((int(by.max()) - top + 1, int(bx.max()) - left + 1), dtype=bool)
        patch[by - top, bx - left] = True
        components, count = ndimage.label(patch, structure=np.ones((3, 3)))
        labels = components[by - top, bx - left]
        sizes = np.bincount(labels)[1:]
        largest = int(sizes.max())
        if largest < MIN_BAND_PIXELS:
            continue
        candidates = [i + 1 for i in range(count) if sizes[i] >= MIN_PATCH_FRACTION * largest]
        # Image coordinates are summed before dividing, as a centre of mass over the whole image is.
        centroids = np.stack((np.bincount(labels, bx + origin[0]), np.bincount(labels, by + origin[1])), 1)[1:] / sizes[:, None]
        if points:
            chosen = min(candidates, key=lambda c: float(np.linalg.norm(centroids[c - 1] - points[-1])))
        else:
            chosen = max(candidates, key=lambda c: sizes[c - 1])
        inside = labels == chosen
        py, px = by[inside], bx[inside]
        central = depth[py, px] >= CENTRAL_FRACTION * depth[py, px].max()
        pixels = (np.stack((px[central], py[central]), 1) + origin).astype(np.float64)
        points.append(pixels[np.argmin(np.linalg.norm(pixels - centroids[chosen - 1], axis=1))])
    if tail is not None:
        points.append(tail)
    kept: list[FloatArray] = []
    for point in points:
        if not kept or np.linalg.norm(point - kept[-1]) >= diameter / 3.0:
            kept.append(point)
    if tail is not None and kept[-1] is not points[-1]:
        kept[-1] = tail  # the tail is the end; it replaces a band point too close to it
    return np.stack(kept) if len(kept) >= 3 else None


def field_evidence(prediction: FieldPrediction | FieldCrop, mask: NDArray[np.bool_]) -> BodyFieldEvidence:
    """What a fit of ``mask`` is scored against: the A-P field on the body (not on crossings) and the ends in view on it, away from the image edge."""

    body = np.asarray(mask, dtype=bool)
    ap = np.full(body.shape, np.nan, dtype=np.float32)
    box = _box(body)
    if box is not None:
        ap[box] = np.where(body[box] & ~prediction.crossing_in(box), prediction.ap_in(box), np.nan)
    return BodyFieldEvidence(
        ap=ap, head_xy=_scored(_end(prediction.end("head"), body), body.shape), tail_xy=_scored(_end(prediction.end("tail"), body), body.shape),
    )


def trace_start(
    prediction: FieldPrediction | FieldCrop,
    mask: NDArray[np.bool_],
    *,
    config: MaskFitConfig,
    length_px: float | None = None,
    width_px: float | None = None,
    width_shape: NDArray[np.float64] | None = None,
) -> Initialization | None:
    """A head-first starting pose along the proposed trace (``None`` without one).

    With ``length_px`` (the recording prior's length) and the head in view
    (passing the end rules of :func:`field_evidence`), the body is laid from
    the head along the trace's bands, without the predicted tail, at exactly
    that length (:func:`mask_fit.lay_centerline`): a trace zigzagging
    between the turns of a spiral is cut there, a short one continues past
    its last band point along the mask, or off camera where the body leaves
    the image.  The
    tail's position is an outcome: people place it inconsistently when they
    trace labels, so the network's tail is not a reliable end.

    Without the head, the trace runs from its first band (or the head) to
    the tail and is lengthened off camera to ``length_px`` as the standard
    starts are (:func:`mask_fit.extend_start_to_length`: an end within 80 px
    of mask pixels on the border).  The last band point of a clipped body
    lies a median 23 px from the edge (90th percentile 60 px, on edge_0528),
    so a test of that point alone left most such traces short, and their
    fits squeezed the whole body into view.

    ``width_px`` and ``width_shape`` (the recording prior's) take the place
    of the width measured across the mask along the trace.
    """

    body = np.asarray(mask, dtype=bool)
    anchored = length_px is not None and _scored(_end(prediction.end("head"), body), body.shape) is not None
    trace = propose_trace(prediction, body, tail=not anchored)
    if trace is None:
        return None
    if anchored:
        # A last band point behind the one before (A-P noise where the field
        # flattens toward the tail) would turn the continuation back.
        while len(trace) > 3 and float(np.dot(trace[-1] - trace[-2], trace[-2] - trace[-3])) < 0:
            trace = trace[:-1]
        # Laid as a polyline, before the latent's smoothing can bend its end.
        trace = lay_centerline(trace, body, length_px, width_px or 2.0 * float(ndimage.distance_transform_edt(body).max()))
    start = init_from_centerline(trace, body, name="network_trace", width_px=width_px, config=config)
    if width_shape is not None:
        start = replace(start, width_shape=np.asarray(width_shape, dtype=np.float64))
    if length_px is None:
        return start
    if anchored:
        return at_length(start, length_px, trace[0], config=config)
    return extend_start_to_length(start, body, length_px, config=config)
