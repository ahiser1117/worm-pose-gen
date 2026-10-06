"""Body-field records of a segmentation store: build, load, review and correct them.

For each hand-labeled frame ``<root>/body_fields/<sample_id>.npz`` holds:

``context`` / ``context_valid``
    the flat-fielded frames ``t - max_lag .. t + max_lag`` from the source
    recording (:func:`temporal_context.read_context`); the centre frame is
    checked against the stored sample image.
``centerline_xy`` / ``width_profile`` / ``fit_iou``
    the tube fitted to the hand mask with the ``reference`` schedule and every
    standard start, oriented head first.  A single-frame fit cannot find a
    coiled or self-touching body (the starts have the wrong topology), so a
    frame fitting below :data:`CHAIN_IOU` or touching itself is also fitted
    by a chain (:func:`chain_fit`): from the nearest clear context frame on
    each side, frame by frame toward it, as the head-tracked fit does.  The
    better of the fits is kept; ``fit_method`` is ``independent`` or
    ``chain``, with ``chain_anchor_offset`` and ``independent_fit_iou``.
``ap`` / ``overlap`` / ``head_xy`` / ``tail_xy`` / ``diameter_px``
    the targets rendered from that tube (:mod:`body_targets`).
``orientation``
    how the head was chosen: ``nose`` (acquisition nose landmark on this
    frame), ``nose_nearby`` (the nearest valid landmark within the context;
    ``nose_offset`` says which frame), ``taper`` (no landmark; the thinner
    end is the tail), or ``manual`` (flipped by hand, :func:`flip`).
    ``orientation_margin`` is ``(d_tail - d_head) / diameter`` for the nose
    cases.

``meta`` is a JSON string.  Besides the build fields it may hold ``review``
(:data:`REVIEW_STATES`; absent means ``unreviewed``) and ``reviewed_at``
(UTC ISO), written by :func:`set_review`.  A rejected sample trains the mask
only.  The record is *stale* when the store's mask revision differs from
``meta["mask_revision"]``: the mask was edited after the targets were built,
and :func:`build` must refit it.  Building replaces the whole record, so a
rebuild starts unreviewed and re-derives the orientation.

A frame whose label holds no worm stores the context only (``has_body``
false).  Every write goes to a temporary file that replaces the record, so a
reader never sees a partial file; edits hold ``body_fields/.lock`` across
their read-modify-write.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch

from .batch_fit import PRESETS, BatchFitConfig, fit_masks
from .body_targets import render_body_targets, self_contact
from .classical import resample_centerline
from .head_fit import HeadConstraint
from .head_tracking import HeadTracking, read_head_tracking
from .label_app import RecordingSource
from .mask_fit import (
    MaskFitResult, decode_centerline, default_width_template, hard_iou, init_from_centerline, orient_tail_last,
    reverse_initialization, reverse_result,
)
from .pipeline import initializations_for
from .pose_run import clean_mask, render_tube
from .propagation import warm_initialization, warm_schedule
from .segmenter import SegmentationModule
from .segmentation_dataset import SampleRecord, SegmentationStore
from .temporal_context import MAX_LAG, read_context
from .workspace import utc_now


FloatArray = NDArray[np.float64]

FIELDS_DIR = "body_fields"
REVIEW_DIR = "review"
FIT_PRESET = "reference"
# The fitter's default length bounds once ended at 750 px and cut off the
# longest animals (in 2023-08-22-01 most whole-body fits sat at 751 px with
# their tails short).  Targets must follow the label, so the bound here only
# rules out absurd fits.
FIT_LENGTH_BOUNDS_PX = (250.0, 1000.0)


def fit_config() -> BatchFitConfig:
    """The schedule body-field fits use: the reference preset with wider length bounds."""

    return replace(PRESETS[FIT_PRESET], length_bounds_px=FIT_LENGTH_BOUNDS_PX)
REVIEW_STATES = ("unreviewed", "accepted", "rejected")


def fields_dir(root: str | Path) -> Path:
    return Path(root) / FIELDS_DIR


def field_path(root: str | Path, sample_id: str) -> Path:
    return fields_dir(root) / f"{sample_id}.npz"


def read_meta(path: Path) -> dict[str, Any]:
    with np.load(path) as archive:
        return json.loads(str(archive["meta"]))


def load(path: Path, names: Sequence[str] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The arrays of a record (all of them, or only ``names`` that it has) and its meta."""

    with np.load(path) as archive:
        wanted = [n for n in archive.files if n != "meta" and (names is None or n in names)]
        arrays = {name: np.asarray(archive[name]) for name in wanted}
        meta = json.loads(str(archive["meta"]))
    return arrays, meta


def save(path: Path, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """Write a record atomically: a temporary file in the same directory replaces ``path``."""

    temporary = path.with_suffix(".npz.partial")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, meta=json.dumps(meta), **arrays)
    os.replace(temporary, path)


@contextmanager
def locked(root: str | Path) -> Iterator[None]:
    directory = fields_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def review_status(meta: dict[str, Any]) -> str:
    return str(meta.get("review", "unreviewed"))


def is_stale(meta: dict[str, Any], record: SampleRecord) -> bool:
    """Whether the hand mask changed after these targets were built."""

    return meta.get("mask_revision") != record.revision


def is_current(path: Path, revision: int, max_lag: int) -> bool:
    if not path.exists():
        return False
    meta = read_meta(path)
    return meta.get("mask_revision") == revision and meta.get("max_lag") == max_lag


# --------------------------------------------------------------------------- edits


def _edit(root: str | Path, sample_id: str, change: Callable[[dict[str, np.ndarray], dict[str, Any]], None]) -> dict[str, Any]:
    path = field_path(root, sample_id)
    with locked(root):
        if not path.exists():
            raise FileNotFoundError(f"{sample_id} has no body fields; build them first")
        arrays, meta = load(path)
        change(arrays, meta)
        save(path, meta, arrays)
    return meta


def flip(root: str | Path, sample_id: str) -> dict[str, Any]:
    """Swap head and tail: reverse the tube, swap the end points, ``ap -> 1 - ap``; orientation becomes ``manual``.

    The review PNG, when there is one, is redrawn to match.
    """

    def change(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        if not meta.get("has_body"):
            raise ValueError(f"{sample_id} has no body to flip")
        arrays["centerline_xy"] = arrays["centerline_xy"][::-1].copy()
        arrays["width_profile"] = arrays["width_profile"][::-1].copy()
        arrays["head_xy"], arrays["tail_xy"] = arrays["tail_xy"], arrays["head_xy"]
        arrays["ap"] = (1 - arrays["ap"]).astype(arrays["ap"].dtype)
        meta["orientation"] = "manual"
        if "orientation_margin" in meta:
            meta["orientation_margin"] = -meta["orientation_margin"]

    meta = _edit(root, sample_id, change)
    review = fields_dir(root) / REVIEW_DIR / f"{sample_id}.png"
    if review.exists():
        arrays, _ = load(field_path(root, sample_id), ("context", "ap", "overlap", "head_xy", "tail_xy"))
        body = np.isfinite(arrays["ap"]) | arrays["overlap"]
        centre = arrays["context"][meta["max_lag"]]
        review_image(centre, arrays["ap"], arrays["overlap"], arrays["head_xy"], arrays["tail_xy"], body_box(body)).save(review)
    return meta


def set_review(root: str | Path, sample_id: str, status: str) -> dict[str, Any]:
    if status not in REVIEW_STATES:
        raise ValueError(f"unknown review status {status!r}; expected one of {REVIEW_STATES}")

    def change(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        meta["review"] = status
        meta["reviewed_at"] = utc_now()

    return _edit(root, sample_id, change)


# --------------------------------------------------------------------------- build


def choose_nose(nose_xy: np.ndarray, nose_valid: np.ndarray, max_lag: int) -> tuple[np.ndarray | None, int | None]:
    """The landmark of the centre frame, else the valid one nearest in time."""

    for distance in range(max_lag + 1):
        for offset in ((0,) if distance == 0 else (-distance, distance)):
            if nose_valid[max_lag + offset]:
                return nose_xy[max_lag + offset], offset
    return None, None


def body_box(mask: NDArray[np.bool_], pad: int = 40) -> tuple[int, int, int, int]:
    """``(x0, x1, y0, y1)`` around the mask, padded and clipped to the image."""

    yy, xx = np.nonzero(mask)
    return (max(0, xx.min() - pad), min(mask.shape[1], xx.max() + pad + 1),
            max(0, yy.min() - pad), min(mask.shape[0], yy.max() + pad + 1))


def review_image(image: np.ndarray, ap: np.ndarray, overlap: np.ndarray, head: np.ndarray, tail: np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    """A-P coloured over the frame, overlap in white, head green, tail red; cropped to ``box`` and doubled."""

    import matplotlib

    rgb = np.repeat(image[..., None], 3, axis=2).astype(np.float32) / 255.0
    body = np.isfinite(ap)
    colours = matplotlib.colormaps["viridis"](np.nan_to_num(ap))[..., :3]
    rgb[body] = 0.35 * rgb[body] + 0.65 * colours[body]
    rgb[overlap] = 1.0
    for point, colour in ((head, (0.0, 1.0, 0.0)), (tail, (1.0, 0.0, 0.0))):
        x, y = int(round(point[0])), int(round(point[1]))
        if not (0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]):
            continue  # an end off camera has no marker
        y0, y1, x0, x1 = max(0, y - 3), min(rgb.shape[0], y + 4), max(0, x - 3), min(rgb.shape[1], x + 4)
        rgb[y0:y1, x0:x1] = colour
    x0, x1, y0, y1 = box
    crop = rgb[y0:y1, x0:x1]
    picture = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
    return picture.resize((picture.width * 2, picture.height * 2), Image.NEAREST)


CHAIN_IOU = 0.93
# A context frame anchors a chain when its independent fit is this good and untangled.
ANCHOR_IOU = 0.93
MIN_WORM_PIXELS = 500
# The head-tracked fit's defaults (algorithms.HeadTrackedFit).
CHAIN_TRACKING_WEIGHT = 0.2
CHAIN_PREVIOUS_HEAD_WEIGHT = 0.5
CHAIN_HEAD_SIGMA_PX = 6.0
CHAIN_MAX_HEAD_STEP_PX = 8.0
CHAIN_PREVIOUS_POSE_WEIGHT = 0.005


def _nose_first(result: MaskFitResult, nose: NDArray[np.generic], config: BatchFitConfig) -> MaskFitResult:
    if np.linalg.norm(result.centerline_xy[-1] - nose) < np.linalg.norm(result.centerline_xy[0] - nose):
        return reverse_result(result, config=config)
    return result


def chain_fit(
    centre_mask: NDArray[np.bool_],
    context: NDArray[np.uint8],
    valid: NDArray[np.bool_],
    tracking: HeadTracking,
    segmenter: SegmentationModule,
    *,
    config: BatchFitConfig,
    template: NDArray[np.generic],
    device: torch.device,
) -> tuple[MaskFitResult, int] | None:
    """Fit the centre frame by tracking the body in from the nearest clear context frame.

    The context frames are segmented, fitted independently (fast schedule,
    starts oriented to that frame's nose), and the nearest frame on each side
    that fits at :data:`ANCHOR_IOU` without touching itself anchors a chain.
    Each step starts from the previous pose under the head-tracked fit's
    penalties (nose, previous head, bounded head step, weak pull toward the
    previous pose); the last step fits ``centre_mask``.  Returns the better
    centre fit of the two sides and its anchor offset, or ``None`` when
    neither side has an anchor.
    """

    centre = len(context) // 2
    probability = segmenter.predict_probability_batch(context, batch_size=8)
    masks = [clean_mask(p, 0.5, 2, device, fill_holes=False)[0] for p in probability]
    masks[centre] = centre_mask
    usable = [
        k for k in range(len(context))
        if k != centre and valid[k] and tracking.valid[k] and masks[k].sum() >= MIN_WORM_PIXELS
    ]
    if not usable:
        return None
    fast = replace(PRESETS["fast"], length_bounds_px=FIT_LENGTH_BOUNDS_PX)

    def oriented(start, nose):
        curve = decode_centerline(start.latent, config.coefficients)
        return reverse_initialization(start, config=config) if np.linalg.norm(curve[-1] - nose) < np.linalg.norm(curve[0] - nose) else start

    independent = fit_masks(
        [masks[k] for k in usable],
        [[oriented(s, tracking.xy[k]) for s in initializations_for(masks[k], config)] for k in usable],
        width_template=template, config=fast, device=device,
    )
    anchors = [
        k for k, r in zip(usable, independent, strict=True)
        if r.records[r.best_index]["final_iou"] >= ANCHOR_IOU and not self_contact(r.centerline_xy, r.width_profile)
    ]
    fits = {k: _nose_first(r, tracking.xy[k], config) for k, r in zip(usable, independent, strict=True)}
    warm = warm_schedule(config)
    best: tuple[MaskFitResult, int] | None = None
    for side in (-1, 1):
        candidates = [k for k in anchors if (k - centre) * side > 0]
        if not candidates:
            continue
        anchor = min(candidates, key=lambda k: abs(k - centre))
        previous, previous_k = fits[anchor], anchor
        path = [k for k in range(anchor - side, centre, -side) if valid[k] and masks[k].sum() >= MIN_WORM_PIXELS] + [centre]
        for k in path:
            step_config = replace(
                warm, temporal_prior_weight=CHAIN_PREVIOUS_POSE_WEIGHT,
                temporal_prior_sigma_px=max(1.0, 0.5 * previous.width_px), length_prior_px=previous.body_length_px,
            )
            head = HeadConstraint(
                tracking_xy=tracking.xy[k] if tracking.valid[k] else None, previous_xy=previous.centerline_xy[0],
                tracking_weight=CHAIN_TRACKING_WEIGHT, previous_weight=CHAIN_PREVIOUS_HEAD_WEIGHT,
                sigma_px=CHAIN_HEAD_SIGMA_PX, max_step_px=CHAIN_MAX_HEAD_STEP_PX * abs(k - previous_k),
            )
            start = warm_initialization(previous.latent, previous.width_px, previous.width_shape, "previous_pose")
            previous = fit_masks(
                [masks[k]], [[start]], width_template=template, config=step_config, device=device,
                references=[previous.centerline_xy], head_constraints=[head],
            )[0]
            previous_k = k
        if best is None or previous.records[previous.best_index]["final_iou"] > best[0].records[best[0].best_index]["final_iou"]:
            best = (previous, anchor - centre)
    return best


# A traced midline: the clicked points, head first, become the start and a
# per-point pull of the fit (:func:`trace_fit`).
TRACE_BORDER_PX = 8.0
TRACE_SIGMA_WIDTHS = 0.5
TRACE_POSE_WEIGHT = 0.02
TRACE_HEAD_WEIGHT = 0.2
TRACE_HEAD_SIGMA_PX = 6.0
# Non-interpenetration for trace fits (``MaskFitConfig.separation_*``): on 25
# traced records, weight 1 at full separation cut the median overlap from 263
# to 99 px (mean 438 to 116) for 0.002 median IoU; weight 5 cost up to 0.06.
TRACE_SEPARATION_WEIGHT = 1.0
TRACE_SEPARATION_FRACTION = 1.0


def extend_trace(trace_xy: NDArray[np.generic], image_shape: tuple[int, int], length_px: float | None) -> FloatArray:
    """The trace, continued straight off camera to ``length_px`` when its tail end is at the image border.

    The fit pulls each of its points toward the same-numbered point of the
    resampled trace, so a trace that stops where the body leaves the camera
    would squeeze the whole body into the visible part.  The continuation
    lies off camera, where the fit ignores the pull, and only fixes how the
    points are spread along the body.
    """

    points = np.asarray(trace_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("a trace needs at least two (x, y) points")
    height, width = image_shape
    end = points[-1]
    at_border = min(end[0], end[1], width - 1 - end[0], height - 1 - end[1]) <= TRACE_BORDER_PX
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if not at_border or length_px is None or length_px <= length:
        return points
    direction = points[-1] - points[-2]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    return np.vstack((points, end + direction * (length_px - length)))


def trace_fit(
    mask: NDArray[np.bool_],
    trace_xy: NDArray[np.generic],
    *,
    length_px: float | None,
    config: BatchFitConfig,
    template: NDArray[np.generic],
    device: torch.device,
    as_drawn: bool = False,
) -> tuple[FloatArray, FloatArray, float]:
    """Centerline, width profile, and mask IoU of the body along a traced midline.

    The trace (head first) is extended off camera to ``length_px`` when it
    ends at the border (:func:`extend_trace`).  By default the tube starts on
    the trace and is fit to ``mask`` while each point is pulled toward the
    trace at half a body width and the head toward the first click, so the
    fit keeps the traced route through a crossing and finds the body's edges
    and width itself.  The length bounds do not apply, since the trace sets
    the length.  ``as_drawn`` keeps the trace as the midline, with a
    template width scaled to the mask.
    """

    path = extend_trace(trace_xy, mask.shape, length_px)
    start = init_from_centerline(path, mask, name="trace", config=config)
    reference = resample_centerline(path, config.n_points)
    if as_drawn:
        profile = template * start.width_px
        tube = render_tube(reference, profile, *mask.shape, device=device)
        return reference, profile, hard_iou(tube, mask)
    # The trace sets the length; no bound may shorten it.
    traced_config = replace(
        config, length_bounds_px=None, separation_weight=TRACE_SEPARATION_WEIGHT,
        separation_fraction=TRACE_SEPARATION_FRACTION, temporal_prior_weight=TRACE_POSE_WEIGHT,
        temporal_prior_sigma_px=max(1.0, TRACE_SIGMA_WIDTHS * start.width_px),
    )

    head = HeadConstraint(
        tracking_xy=np.asarray(trace_xy[0], dtype=np.float64), tracking_weight=TRACE_HEAD_WEIGHT,
        previous_weight=0.0, sigma_px=TRACE_HEAD_SIGMA_PX,
    )
    result = fit_masks(
        [mask], [[start]], width_template=template, config=traced_config, device=device,
        references=[reference], head_constraints=[head],
    )[0]
    return result.centerline_xy, result.width_profile, float(result.records[result.best_index]["final_iou"])


TRACE_METHODS = ("traced", "trace_as_drawn")
# Whole bodies this well fit set a recording's typical length (for traces that leave the camera).
LENGTH_REFERENCE_IOU = 0.95


def _inside(points: NDArray[np.generic], shape: tuple[int, int]) -> bool:
    height, width = shape
    return bool(np.all((points[:, 0] >= 0) & (points[:, 0] <= width - 1) & (points[:, 1] >= 0) & (points[:, 1] <= height - 1)))


def recording_length(store: SegmentationStore, record: SampleRecord) -> float | None:
    """Median length of the whole, well-fit, automatically fitted bodies from the same recording file.

    Records count when their fit covers the mask well (fit IoU at least
    ``LENGTH_REFERENCE_IOU``), the whole body is in view, and the record is
    current.  Recordings are matched by file name, since the store holds
    labels of one recording under more than one path and recording id.
    ``None`` without any such record.
    """

    name = Path(record.source_path).name
    lengths = []
    for other in store.records():
        path = field_path(store.root, other.sample_id)
        if Path(other.source_path).name != name or not path.exists():
            continue
        arrays, meta = load(path, ("centerline_xy",))
        if (not meta.get("has_body") or meta.get("fit_iou", 0.0) < LENGTH_REFERENCE_IOU
                or meta.get("fit_method") in TRACE_METHODS or is_stale(meta, other)):
            continue
        curve = arrays["centerline_xy"]
        if _inside(curve, (other.image_height, other.image_width)):
            lengths.append(float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()))
    return float(np.median(lengths)) if lengths else None


def _set_body(
    meta: dict[str, Any], arrays: dict[str, np.ndarray], mask: NDArray[np.bool_],
    centerline: NDArray[np.generic], width_profile: NDArray[np.generic], fit_iou: float,
) -> Any:
    """Render the targets of a head-first tube into the record; returns them."""

    targets = render_body_targets(mask, centerline, width_profile)
    meta["fit_iou"] = float(fit_iou)
    meta["overlap_px"] = int(targets.overlap.sum())
    arrays.update(
        centerline_xy=np.asarray(centerline, dtype=np.float64), width_profile=np.asarray(width_profile, dtype=np.float64),
        ap=targets.ap.astype(np.float16), overlap=targets.overlap,
        head_xy=targets.head_xy, tail_xy=targets.tail_xy, diameter_px=np.float64(targets.diameter_px),
    )
    if "nose_xy" in arrays:
        d_head = float(np.linalg.norm(targets.head_xy - arrays["nose_xy"]))
        d_tail = float(np.linalg.norm(targets.tail_xy - arrays["nose_xy"]))
        meta["orientation_margin"] = (d_tail - d_head) / targets.diameter_px
    return targets


def _traced_body(
    meta: dict[str, Any], arrays: dict[str, np.ndarray], mask: NDArray[np.bool_], trace_xy: NDArray[np.generic],
    *, as_drawn: bool, length_px: float | None, config: BatchFitConfig, template: NDArray[np.generic], device: torch.device,
) -> Any:
    centerline, profile, iou = trace_fit(
        mask, trace_xy, length_px=length_px, config=config, template=template, device=device, as_drawn=as_drawn,
    )
    meta.update(
        fit_method="trace_as_drawn" if as_drawn else "traced", orientation="manual",
    )
    arrays["trace_xy"] = np.asarray(trace_xy, dtype=np.float64)
    return _set_body(meta, arrays, mask, centerline, profile, iou)


def apply_trace(
    store: SegmentationStore,
    sample_id: str,
    trace_xy: NDArray[np.generic],
    *,
    as_drawn: bool = False,
    commit: bool = False,
    device: Any = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Refit a sample's body along a traced midline (clicked points, head first).

    Returns the new record's meta and arrays.  With ``commit`` it replaces
    the stored record (and its review PNG, when there is one); otherwise it
    is a preview and nothing is written.  The trace is kept in the record
    (``trace_xy``), so a rebuild after a mask edit refits along it.  A
    committed trace is a reviewed correction: ``review`` becomes
    ``accepted`` and ``auto_fit_iou`` keeps the replaced fit's overlap.
    """

    record = store.get(sample_id)
    if record is None:
        raise KeyError(sample_id)
    path = field_path(store.root, sample_id)
    if not path.exists():
        raise FileNotFoundError(f"{sample_id} has no body fields; build them first")
    _, label, _ = store.load(sample_id)
    mask = label == 1
    if not mask.any():
        raise ValueError(f"{sample_id} has no worm in its mask")
    points = np.asarray(trace_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("a trace needs at least two finite (x, y) points")
    arrays, meta = load(path)
    previous_iou = meta.get("auto_fit_iou", meta.get("fit_iou"))
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = fit_config()
    targets = _traced_body(
        meta, arrays, mask, points, as_drawn=as_drawn, length_px=recording_length(store, record),
        config=config, template=default_width_template(config.n_points), device=device,
    )
    meta.update(has_body=True, mask_revision=record.revision, auto_fit_iou=previous_iou, review="accepted", reviewed_at=utc_now())
    if commit:
        with locked(store.root):
            save(path, meta, arrays)
        review_path = fields_dir(store.root) / REVIEW_DIR / f"{sample_id}.png"
        if review_path.exists():
            review_image(arrays["context"][meta["max_lag"]], targets.ap, targets.overlap, targets.head_xy, targets.tail_xy, body_box(mask)).save(review_path)
    return meta, arrays


def build(
    store: SegmentationStore,
    records: Sequence[SampleRecord],
    *,
    max_lag: int = MAX_LAG,
    device: Any = None,
    review: bool = False,
    segmenter: SegmentationModule | None = None,
    on_built: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    """Build the records of ``records`` (all of them; skipping current ones is the caller's choice).

    Frames are read and fitted per source recording, so each recording opens
    once.  Without ``segmenter`` no chain fits are tried.  ``on_built``
    receives each written meta.  Returns how many samples
    took each orientation (``no_body`` for empty labels).
    """

    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = fit_config()
    template = default_width_template(config.n_points)
    out_dir = fields_dir(store.root)
    review_dir = out_dir / REVIEW_DIR
    out_dir.mkdir(exist_ok=True)
    if review:
        review_dir.mkdir(exist_ok=True)

    by_source: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_path].append(record)
    summary: dict[str, int] = defaultdict(int)
    for source_path, group in sorted(by_source.items()):
        source = RecordingSource(Path(source_path), store.root / "flat_fields")
        try:
            pending = []
            for record in group:
                image, label, _ = store.load(record.sample_id)
                context, valid = read_context(source, record.frame_index, max_lag)
                if not np.array_equal(context[max_lag], image):
                    raise RuntimeError(f"{record.sample_id}: recording frame differs from the stored sample image")
                frames = np.clip(np.arange(-max_lag, max_lag + 1) + record.frame_index, 0, source.frame_count - 1)
                tracking = read_head_tracking(source_path, frames, image.shape)
                pending.append((record, image, label == 1, context, valid, tracking))
        finally:
            source.close()

        # A stored trace is a human correction: the rebuild refits along it.
        traces: dict[str, tuple[np.ndarray, bool]] = {}
        for record, *_ in pending:
            path = field_path(store.root, record.sample_id)
            if path.exists():
                previous, previous_meta = load(path, ("trace_xy",))
                if "trace_xy" in previous and previous_meta.get("fit_method") in TRACE_METHODS:
                    traces[record.sample_id] = (previous["trace_xy"], previous_meta["fit_method"] == "trace_as_drawn")
        bodies = [p for p in pending if p[2].any() and p[0].sample_id not in traces]
        results = fit_masks(
            [p[2] for p in bodies],
            [initializations_for(p[2], config) for p in bodies],
            width_template=template, config=config, device=device,
        ) if bodies else []
        fitted = {p[0].sample_id: r for p, r in zip(bodies, results, strict=True)}

        for record, image, mask, context, valid, tracking in pending:
            meta: dict[str, Any] = {
                "sample_id": record.sample_id, "mask_revision": record.revision, "max_lag": max_lag,
                "fit_preset": FIT_PRESET, "has_body": record.sample_id in fitted,
            }
            arrays: dict[str, np.ndarray] = {"context": context, "context_valid": valid}
            result = fitted.get(record.sample_id)
            if record.sample_id in traces and mask.any():
                trace, as_drawn = traces[record.sample_id]
                nose, offset = choose_nose(tracking.xy, tracking.valid, max_lag)
                if nose is not None:
                    arrays["nose_xy"] = np.asarray(nose, dtype=np.float64)
                    meta["nose_offset"] = offset
                targets = _traced_body(
                    meta, arrays, mask, trace, as_drawn=as_drawn, length_px=recording_length(store, record),
                    config=config, template=template, device=device,
                )
                meta.update(has_body=True, review="accepted", reviewed_at=utc_now())
                summary["traced"] += 1
                if review:
                    review_image(image, targets.ap, targets.overlap, targets.head_xy, targets.tail_xy, body_box(mask)).save(
                        review_dir / f"{record.sample_id}.png"
                    )
            elif result is not None:
                meta["fit_method"] = "independent"
                independent_iou = float(result.records[result.best_index]["final_iou"])
                tangled = independent_iou < CHAIN_IOU or self_contact(result.centerline_xy, result.width_profile)
                if segmenter is not None and tangled:
                    chained = chain_fit(mask, context, valid, tracking, segmenter, config=config, template=template, device=device)
                    if chained is not None and chained[0].records[chained[0].best_index]["final_iou"] > independent_iou:
                        result = chained[0]
                        meta.update(fit_method="chain", chain_anchor_offset=chained[1], independent_fit_iou=independent_iou)
                nose, offset = choose_nose(tracking.xy, tracking.valid, max_lag)
                if nose is not None:
                    d_head = float(np.linalg.norm(result.centerline_xy[0] - nose))
                    d_tail = float(np.linalg.norm(result.centerline_xy[-1] - nose))
                    if d_tail < d_head:
                        result = reverse_result(result, config=config)
                    meta["orientation"] = "nose" if offset == 0 else "nose_nearby"
                    meta["nose_offset"] = offset
                    arrays["nose_xy"] = np.asarray(nose, dtype=np.float64)
                else:
                    result, _ = orient_tail_last(result, config=config)
                    meta["orientation"] = "taper"
                targets = _set_body(
                    meta, arrays, mask, result.centerline_xy, result.width_profile,
                    float(result.records[result.best_index]["final_iou"]),
                )
                summary[meta["orientation"]] += 1
                if review:
                    review_image(image, targets.ap, targets.overlap, targets.head_xy, targets.tail_xy, body_box(mask)).save(
                        review_dir / f"{record.sample_id}.png"
                    )
            else:
                summary["no_body"] += 1
            with locked(store.root):
                save(field_path(store.root, record.sample_id), meta, arrays)
            if on_built is not None:
                on_built(meta)
    return dict(summary)
