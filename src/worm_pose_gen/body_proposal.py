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
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
import torch

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
# Pixels predicted as a crossing above this are left out of the bands.
OVERLAP_THRESHOLD = 0.5
MIN_BAND_PIXELS = 20
# A band's real cross-section is its largest patch; patches much smaller are
# A-P noise along a contact line and are not followed.
MIN_PATCH_FRACTION = 0.5
# A band's point is its pixel nearest the band's centroid among the most
# central ones (distance to the mask edge at least this share of the band's maximum).
CENTRAL_FRACTION = 0.7


@dataclass(frozen=True)
class FieldPrediction:
    """Per-pixel network outputs for one frame, as probabilities (A-P as a value in [0, 1])."""

    mask: NDArray[np.float32]
    ap: NDArray[np.float32]
    head: NDArray[np.float32]
    tail: NDArray[np.float32]
    overlap: NDArray[np.float32]


# Body-field evidence weights for ordinary fits (``MaskFitConfig.field_*``).
# On the 66 reviewed held-out frames (scripts/evaluate_field_fitting.py) the
# trace start plus these terms put all 32 hand-traced contacts and coils
# within half a body width of the hand trace (15 without) with every head at
# the right end; an A-P weight of 0.01 or more pulled fits off the mask edges.
FIELD_AP_WEIGHT = 0.001
FIELD_END_WEIGHT = 0.01


def network_inputs(
    centre: NDArray[np.uint8], pairs: Sequence[tuple[NDArray[np.uint8] | None, NDArray[np.uint8] | None]],
) -> NDArray[np.float32]:
    """The network's input stack: the normalized frame, then one difference per lag.

    ``pairs`` holds ``(frame[t + lag], frame[t - lag])`` per lag, ``None`` for
    a frame that does not exist; such a lag gets a zero channel, as in
    training (:func:`temporal_context.difference_channels`).
    """

    channels = [(centre.astype(np.float32) / 255.0 - INPUT_MEAN) / INPUT_STD]
    for later, earlier in pairs:
        if later is None or earlier is None:
            channels.append(np.zeros(centre.shape, dtype=np.float32))
        else:
            channels.append((later.astype(np.float32) - earlier.astype(np.float32)) / (255.0 * INPUT_STD))
    return np.stack(channels)


@torch.inference_mode()
def _run(module: BodyFieldModule, inputs: NDArray[np.float32]) -> list[FieldPrediction]:
    tensor = torch.as_tensor(inputs).to(module.device)
    with torch.autocast(device_type=module.device.type, enabled=module.device.type == "cuda"):
        logits = module(tensor).float()
    maps = torch.sigmoid(logits).cpu().numpy()
    return [FieldPrediction(**{name: m[k] for k, name in enumerate(OUTPUTS)}) for m in maps]


def predict_fields(module: BodyFieldModule, context: NDArray[np.uint8], valid: NDArray[np.bool_]) -> FieldPrediction:
    """Run the network on the centre of a context stack (as stored in a body-field record)."""

    centre = context.shape[0] // 2
    if module.lags and max(module.lags) > centre:
        raise ValueError(f"the model needs lags up to {max(module.lags)}; the context reaches {centre}")
    pairs = [
        (context[centre + lag] if valid[centre + lag] else None, context[centre - lag] if valid[centre - lag] else None)
        for lag in module.lags
    ]
    return _run(module, network_inputs(context[centre], pairs)[None])[0]


class RecordingFieldPredictor:
    """Body-field predictions for frames of one recording, from their flat-fielded neighbours."""

    def __init__(self, module: BodyFieldModule, frames: Any, *, batch_size: int = 8) -> None:
        self.module = module
        self.frames = frames
        self.batch_size = batch_size

    def predict(self, frame_indices: Sequence[int]) -> list[FieldPrediction]:
        """One prediction per source frame index; neighbours outside the recording give zero channels."""

        lags = self.module.lags
        targets = [int(f) for f in frame_indices]
        needed = sorted({f + o for f in targets for o in (0, *lags, *(-lag for lag in lags)) if 0 <= f + o < self.frames.total})
        corrected, _, _ = self.frames.corrected(needed)
        frame = dict(zip(needed, corrected))
        results: list[FieldPrediction] = []
        for start in range(0, len(targets), self.batch_size):
            batch = targets[start : start + self.batch_size]
            inputs = np.stack([network_inputs(frame[f], [(frame.get(f + lag), frame.get(f - lag)) for lag in lags]) for f in batch])
            results.extend(_run(self.module, inputs))
        return results


def _end(heatmap: NDArray[np.float32], body: NDArray[np.bool_]) -> FloatArray | None:
    """The heatmap's peak (x, y); ``None`` below ``END_THRESHOLD`` or more than ``END_MASK_PX`` off ``body``."""

    peak = int(np.argmax(heatmap))
    if float(heatmap.flat[peak]) < END_THRESHOLD:
        return None
    y, x = (int(v) for v in np.unravel_index(peak, heatmap.shape))
    reach = int(np.ceil(END_MASK_PX))
    top, left = max(y - reach, 0), max(x - reach, 0)
    ys, xs = np.nonzero(body[top : y + reach + 1, left : x + reach + 1])
    if not len(ys) or float(np.hypot(ys + top - y, xs + left - x).min()) > END_MASK_PX:
        return None
    return np.array([x, y], dtype=np.float64)


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
    prediction: FieldPrediction, mask: NDArray[np.bool_], *, levels: int = TRACE_LEVELS, tail: bool = True,
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
    head = _end(prediction.head, body)
    tail = _end(prediction.tail, body) if tail else None
    # Everything below works on the mask's box; ``origin`` is its corner in the image.
    origin = np.array([box[1].start, box[0].start])
    body = body[box]
    ys, xs = np.nonzero(body & (prediction.overlap[box] < OVERLAP_THRESHOLD))
    if not len(ys):
        return None
    depth = ndimage.distance_transform_edt(body)
    diameter = 2.0 * float(depth.max())
    points: list[FloatArray] = [] if head is None else [head]
    # Band k holds edges[k] <= A-P < edges[k + 1], the last one closed at 1.
    edges = np.linspace(0.0, 1.0, levels + 1)
    ap = prediction.ap[box][ys, xs]
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


def field_evidence(prediction: FieldPrediction, mask: NDArray[np.bool_]) -> BodyFieldEvidence:
    """What a fit of ``mask`` is scored against: the A-P field on the body (not on crossings) and the ends in view on it, away from the image edge."""

    body = np.asarray(mask, dtype=bool)
    ap = np.full(body.shape, np.nan, dtype=np.float32)
    box = _box(body)
    if box is not None:
        ap[box] = np.where(body[box] & (prediction.overlap[box] < OVERLAP_THRESHOLD), prediction.ap[box], np.nan)
    return BodyFieldEvidence(
        ap=ap, head_xy=_scored(_end(prediction.head, body), body.shape), tail_xy=_scored(_end(prediction.tail, body), body.shape),
    )


def trace_start(
    prediction: FieldPrediction,
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
    anchored = length_px is not None and _scored(_end(prediction.head, body), body.shape) is not None
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
