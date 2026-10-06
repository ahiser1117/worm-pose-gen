"""Midline traces proposed by the body-field network.

The network (:mod:`body_net`) predicts, for a frame and its temporal
context, the worm mask, the A-P field, head and tail heatmaps, and where the
body crosses itself.  A trace like a hand-clicked one follows from them: the
predicted head, then one point per band of predicted A-P values across the
labeled mask (the centre of the band's patch of body, skipping pixels
predicted as a crossing), then the predicted tail.  Because the A-P field
changes across a contact line, a band picks out one limb even where two
touch, which is what a skeleton of the mask cannot do.  The trace is then fit
like a hand trace (:func:`body_fields.trace_fit`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
import torch

from .body_net import OUTPUTS, BodyFieldModule
from .segmenter import INPUT_MEAN, INPUT_STD
from .temporal_context import difference_channels


FloatArray = NDArray[np.float64]

TRACE_LEVELS = 20
# A heatmap peak below this means the end is not in view.
END_THRESHOLD = 0.3
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


@torch.inference_mode()
def predict_fields(module: BodyFieldModule, context: NDArray[np.uint8], valid: NDArray[np.bool_]) -> FieldPrediction:
    """Run the network on the centre of a context stack (as stored in a body-field record)."""

    centre = context.shape[0] // 2
    if module.lags and max(module.lags) > centre:
        raise ValueError(f"the model needs lags up to {max(module.lags)}; the context reaches {centre}")
    frame = (context[centre].astype(np.float32) / 255.0 - INPUT_MEAN) / INPUT_STD
    inputs = np.concatenate((frame[None], difference_channels(context.astype(np.float32), valid, module.lags)))
    tensor = torch.as_tensor(inputs)[None].to(module.device)
    with torch.autocast(device_type=module.device.type, enabled=module.device.type == "cuda"):
        logits = module(tensor)[0].float()
    maps = torch.sigmoid(logits).cpu().numpy()
    return FieldPrediction(**{name: maps[k] for k, name in enumerate(OUTPUTS)})


def _end(heatmap: NDArray[np.float32]) -> FloatArray | None:
    if float(heatmap.max()) < END_THRESHOLD:
        return None
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return np.array([x, y], dtype=np.float64)


def propose_trace(prediction: FieldPrediction, mask: NDArray[np.bool_], *, levels: int = TRACE_LEVELS) -> FloatArray | None:
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
    (which :func:`body_fields.extend_trace` continues off camera).
    """

    body = np.asarray(mask, dtype=bool)
    domain = body & (prediction.overlap < OVERLAP_THRESHOLD)
    if not domain.any():
        return None
    depth = ndimage.distance_transform_edt(body)
    diameter = 2.0 * float(depth.max())
    head, tail = _end(prediction.head), _end(prediction.tail)
    points: list[FloatArray] = [] if head is None else [head]
    edges = np.linspace(0.0, 1.0, levels + 1)
    for k in range(levels):
        upper = prediction.ap <= edges[k + 1] if k == levels - 1 else prediction.ap < edges[k + 1]
        band = domain & (prediction.ap >= edges[k]) & upper
        components, count = ndimage.label(band, structure=np.ones((3, 3)))
        if not count:
            continue
        sizes = np.bincount(components.ravel())[1:]
        largest = int(sizes.max())
        if largest < MIN_BAND_PIXELS:
            continue
        candidates = [i + 1 for i in range(count) if sizes[i] >= MIN_PATCH_FRACTION * largest]
        centroids = {c: np.array(ndimage.center_of_mass(band, components, c)[::-1]) for c in candidates}
        if points:
            chosen = min(candidates, key=lambda c: float(np.linalg.norm(centroids[c] - points[-1])))
        else:
            chosen = max(candidates, key=lambda c: sizes[c - 1])
        ys, xs = np.nonzero(components == chosen)
        central = depth[ys, xs] >= CENTRAL_FRACTION * depth[ys, xs].max()
        pixels = np.stack((xs[central], ys[central]), 1).astype(np.float64)
        points.append(pixels[np.argmin(np.linalg.norm(pixels - centroids[chosen], axis=1))])
    if tail is not None:
        points.append(tail)
    kept: list[FloatArray] = []
    for point in points:
        if not kept or np.linalg.norm(point - kept[-1]) >= diameter / 3.0:
            kept.append(point)
    if tail is not None and kept[-1] is not points[-1]:
        kept[-1] = tail  # the tail is the end; it replaces a band point too close to it
    return np.stack(kept) if len(kept) >= 3 else None
