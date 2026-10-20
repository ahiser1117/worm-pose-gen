"""One label revision: what a person decided about one frame, stored with everything needed to train on it.

A revision is an immutable ``.npz`` file
(``labels/<setup>/<recording>/<frame:06d>/<revision:04d>.npz`` in a library,
:mod:`.collection`) with:

``image`` / ``image_raw``
    the frame as the network sees it (flat-fielded when the setup says so)
    and as recorded, uint8 ``[H,W]``.
``mask``
    0 background, 1 worm; 255 marks pixels excluded from the loss, which
    only labels migrated from the old stores have.
``context`` / ``context_valid``
    the frames ``t - 16 .. t + 16`` (preprocessed like ``image``, which is
    the centre one) and whether each was readable, so a label never needs
    its recording again (raw recordings are on stores that corrupt, and other
    users may not be able to read them).
``nose_xy`` / ``nose_valid``
    the acquisition's nose landmark on each context frame (NaN and false
    where there is none); the automatic orientation and chain fits use it.
``trace_xy`` (optional)
    a traced midline, head first; the body is fit along it, from its first
    point to its last unless the meta's ``trace_extend`` asks for it to be
    continued off camera to the recording's typical body length.
``head_xy`` (optional)
    the head end a person chose without tracing (orientation ``manual``).
``meta``
    JSON: ``recording``, ``frame``, ``revision``, ``orientation`` (``auto``
    or ``manual``), ``mask_only`` (the body is unclear and the label trains
    the mask only), ``origin`` (``spread``, ``fix`` or ``migrated``),
    ``author``, ``saved_at``, ``source_path``, ``dataset_path``,
    ``trace_extend`` (with a trace: continue it off camera when it ends at
    the border) and ``trace_length_px`` (the body length it was continued
    to, as the Labeling page showed it; ``None`` for the recording's
    length from its labels), and for migrated labels where they came from
    (``migrated_from``).

A label's *status* is ``mask_only``; else ``complete`` when a person
settled its body (a trace or a manual orientation) or the frame holds no
worm; else ``auto`` (the body comes from the automatic orientation and
nobody has confirmed it: only migrated, never-reviewed labels).

Anything computed from a label (the fitted tube, A-P field, heatmaps,
``fit_iou``) is a rebuildable cache keyed by the revision's ``sha256``
(:mod:`.targets`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


MAX_LAG = 16
IGNORE = 255
ORIGINS = ("spread", "fix", "migrated")
ORIENTATIONS = ("auto", "manual")
STATUSES = ("mask_only", "complete", "auto")
# Origins a benchmark may hold: labels placed to sample a recording, not to fix a failure.
BENCHMARK_ORIGINS = ("spread", "migrated")


def label_key(recording: str, frame: int) -> str:
    return f"{recording}/{int(frame):06d}"


def label_status(*, mask_only: bool, empty: bool, has_trace: bool, orientation: str) -> str:
    if mask_only:
        return "mask_only"
    if empty or has_trace or orientation == "manual":
        return "complete"
    return "auto"


@dataclass(frozen=True)
class LabelRecord:
    """One label revision as the index lists it; :meth:`load` reads its arrays.

    ``scope`` is the library that holds it (``lab`` or ``mine``) and
    ``setup`` the setup whose collection it belongs to; ``split`` is the
    recording's split in the dataset it was read through (``None`` read from
    the collection, or not included).
    """

    scope: str
    setup: str
    recording: str
    frame: int
    revision: int
    split: str | None
    sha256: str
    saved_at: str
    author: str
    origin: str
    orientation: str
    mask_only: bool
    has_trace: bool
    empty: bool
    foreground_fraction: float
    ignore_fraction: float
    height: int
    width: int
    source_path: str
    dataset_path: str
    path: str = field(repr=False, default="")

    @property
    def key(self) -> str:
        return label_key(self.recording, self.frame)

    @property
    def status(self) -> str:
        return label_status(mask_only=self.mask_only, empty=self.empty, has_trace=self.has_trace, orientation=self.orientation)

    @property
    def identity(self) -> dict[str, Any]:
        """What a benchmark or a training run records to name exactly this revision."""

        return {"scope": self.scope, "setup": self.setup, "recording": self.recording, "frame": self.frame,
                "revision": self.revision, "sha256": self.sha256}

    def load(self) -> "Label":
        return read_label(Path(self.path), self)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("path")
        result["status"] = self.status
        return result


@dataclass(frozen=True)
class Label:
    record: LabelRecord
    image: NDArray[np.uint8]
    image_raw: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    context: NDArray[np.uint8]
    context_valid: NDArray[np.bool_]
    nose_xy: NDArray[np.float64]
    nose_valid: NDArray[np.bool_]
    trace_xy: NDArray[np.float64] | None
    head_xy: NDArray[np.float64] | None
    meta: dict[str, Any]

    @property
    def max_lag(self) -> int:
        return len(self.context) // 2


def read_label(path: Path, record: LabelRecord) -> Label:
    with np.load(path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    meta = json.loads(str(arrays.pop("meta")))
    return Label(
        record=record, image=arrays["image"], image_raw=arrays["image_raw"], mask=arrays["mask"],
        context=arrays["context"], context_valid=arrays["context_valid"].astype(bool),
        nose_xy=arrays["nose_xy"].astype(np.float64), nose_valid=arrays["nose_valid"].astype(bool),
        trace_xy=arrays.get("trace_xy"), head_xy=arrays.get("head_xy"), meta=meta,
    )


def _uint8(values: NDArray[np.generic], name: str, shape: tuple[int, ...] | None = None) -> NDArray[np.uint8]:
    array = np.asarray(values)
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    return array


def _points(values: Any, name: str) -> NDArray[np.float64] | None:
    if values is None:
        return None
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} needs at least two finite (x, y) points")
    return points


def encode_label(
    *,
    recording: str,
    frame: int,
    revision: int,
    image: NDArray[np.generic],
    image_raw: NDArray[np.generic],
    mask: NDArray[np.generic],
    context: NDArray[np.generic],
    context_valid: NDArray[np.generic],
    nose_xy: NDArray[np.generic] | None,
    nose_valid: NDArray[np.generic] | None,
    orientation: str,
    head_xy: Any,
    trace_xy: Any,
    mask_only: bool,
    origin: str,
    trace_extend: bool = False,
    trace_length_px: float | None = None,
    author: str,
    saved_at: str,
    source_path: str,
    dataset_path: str,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Validate a label and return its ``.npz`` bytes and the index entry describing it.

    A trace makes the orientation ``manual`` (its first point is the head);
    a ``manual`` orientation without a trace needs ``head_xy``.
    """

    image = _uint8(image, "image")
    if image.ndim != 2:
        raise ValueError("image must have shape [H,W]")
    shape = image.shape
    image_raw = _uint8(image_raw, "image_raw", shape)
    labels = np.asarray(mask)
    if labels.dtype == bool:
        labels = labels.astype(np.uint8)
    labels = _uint8(labels, "mask", shape)
    if not np.isin(labels, (0, 1, IGNORE)).all():
        raise ValueError("mask values must be 0, 1 or 255")
    stack = _uint8(context, "context")
    if stack.ndim != 3 or stack.shape[1:] != shape or len(stack) % 2 != 1:
        raise ValueError(f"context must have shape [2L+1,H,W] with [H,W] = {shape}")
    centre = len(stack) // 2
    if not np.array_equal(stack[centre], image):
        raise ValueError("the centre context frame must be the label's image")
    valid = np.asarray(context_valid, dtype=bool)
    if valid.shape != (len(stack),) or not valid[centre]:
        raise ValueError("context_valid needs one entry per context frame, and the centre frame is valid")
    nose = np.full((len(stack), 2), np.nan) if nose_xy is None else np.asarray(nose_xy, dtype=np.float64)
    nose_ok = np.zeros(len(stack), dtype=bool) if nose_valid is None else np.asarray(nose_valid, dtype=bool)
    if nose.shape != (len(stack), 2) or nose_ok.shape != (len(stack),):
        raise ValueError("nose_xy and nose_valid need one entry per context frame")
    if origin not in ORIGINS:
        raise ValueError(f"unknown label origin {origin!r}; expected one of {ORIGINS}")
    trace = _points(trace_xy, "trace_xy")
    head = None if head_xy is None else np.asarray(head_xy, dtype=np.float64)
    if trace is not None:
        orientation, head = "manual", None
    if orientation not in ORIENTATIONS:
        raise ValueError(f"unknown orientation {orientation!r}; expected one of {ORIENTATIONS}")
    if orientation == "manual" and trace is None and (head is None or head.shape != (2,) or not np.all(np.isfinite(head))):
        raise ValueError("a manual orientation without a trace needs head_xy (x, y)")
    if orientation == "auto":
        head = None

    meta = {
        **(extra_meta or {}),
        "recording": recording, "frame": int(frame), "revision": int(revision), "orientation": orientation,
        "mask_only": bool(mask_only), "origin": origin, "author": author, "saved_at": saved_at,
        "source_path": str(source_path), "dataset_path": str(dataset_path),
        "trace_extend": bool(trace_extend) and trace is not None,
        "trace_length_px": float(trace_length_px) if trace_extend and trace is not None and trace_length_px else None,
    }
    arrays: dict[str, Any] = dict(
        image=image, image_raw=image_raw, mask=labels, context=stack, context_valid=valid,
        nose_xy=nose, nose_valid=nose_ok,
    )
    if trace is not None:
        arrays["trace_xy"] = trace
    if head is not None:
        arrays["head_xy"] = head
    buffer = io.BytesIO()
    np.savez_compressed(buffer, meta=json.dumps(meta), **arrays)
    data = buffer.getvalue()
    entry = {
        "revision": int(revision), "sha256": hashlib.sha256(data).hexdigest(), "saved_at": saved_at, "author": author,
        "origin": origin, "orientation": orientation, "mask_only": bool(mask_only), "has_trace": trace is not None,
        "empty": not bool((labels == 1).any()), "foreground_fraction": float((labels == 1).mean()),
        "ignore_fraction": float((labels == IGNORE).mean()), "height": int(shape[0]), "width": int(shape[1]),
        "source_path": str(source_path), "dataset_path": str(dataset_path),
    }
    return data, entry


def write_immutable(path: Path, data: bytes) -> None:
    """Write ``data`` to a new file at ``path``; an existing revision is never replaced."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # fails if the revision exists
    finally:
        temporary.unlink(missing_ok=True)
