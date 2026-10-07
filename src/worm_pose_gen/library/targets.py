"""Body-field targets of a label: a rebuildable cache in the personal library, keyed by label revision.

The targets (the head-first tube fitted to the hand mask, the rendered A-P
field, overlap, head and tail, ``fit_iou``, self-contact) are computed from
the label alone by :func:`body_fields.fit_targets`, the builder the old
``body_fields/`` records came from: a traced label is refit along its trace,
a manual orientation puts the head at the end nearest the chosen point, and
otherwise the acquisition nose or the taper decides.  Since a revision never
changes, its ``sha256`` names its targets for good:
``<personal>/cache/body_targets/<sha256>.npz`` holds the arrays and
``<sha256>.json`` the meta, which listings read without opening the arrays.
Every user builds targets in their own library, lab labels included, since
the lab library is read-only.

Building needs the fitter (a GPU makes it seconds instead of tens of
seconds); a tangled frame is also chain-fit through its context frames when
a mask model is given (``segmenter``: anything with
``predict_probability_batch``, today the setup's mask default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .datasets import Dataset
from .labels import LabelRecord
from .roots import Libraries, read_json, write_json
from ..workspace import utc_now


CACHE_DIR = Path("cache") / "body_targets"


def cache_paths(libraries: Libraries, sha256: str) -> tuple[Path, Path]:
    directory = libraries.personal / CACHE_DIR
    return directory / f"{sha256}.npz", directory / f"{sha256}.json"


def cached_meta(libraries: Libraries, record: LabelRecord) -> dict[str, Any] | None:
    """The meta of a label's built targets, or ``None`` when they are not built yet."""

    return read_json(cache_paths(libraries, record.sha256)[1])


def load_targets(libraries: Libraries, record: LabelRecord) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    """The meta and arrays of a label's targets (``centerline_xy``, ``width_profile``, ``ap``, ``overlap``,
    ``head_xy``, ``tail_xy``, ``diameter_px``; none for a label without a worm), or ``None`` when not built."""

    arrays_path, meta_path = cache_paths(libraries, record.sha256)
    meta = read_json(meta_path)
    if meta is None:
        return None
    with np.load(arrays_path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return meta, arrays


def write_targets(libraries: Libraries, sha256: str, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """Store targets; the arrays first, so a meta file always has its arrays."""

    arrays_path, meta_path = cache_paths(libraries, sha256)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = arrays_path.with_name(f".{arrays_path.name}.partial")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(arrays_path)
    write_json(meta_path, meta)


def describe_body(meta: dict[str, Any], centerline: np.ndarray, width_profile: np.ndarray, shape: tuple[int, int]) -> None:
    """Add what listings and the length reference need: self-contact, body length, and whether it is all in view."""

    from ..body_targets import self_contact

    height, width = shape
    meta["self_contact"] = self_contact(centerline, width_profile)
    meta["body_length_px"] = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    meta["in_view"] = bool(np.all((centerline[:, 0] >= 0) & (centerline[:, 0] <= width - 1)
                                  & (centerline[:, 1] >= 0) & (centerline[:, 1] <= height - 1)))


def recording_length(libraries: Libraries, record: LabelRecord) -> float | None:
    """Median length of the whole, well-fit, untraced bodies among the built labels of the same recording.

    It sets the length of a body that leaves the camera
    (:func:`body_fields.mark_exits`, :func:`body_fields.extend_trace`), as
    :func:`body_fields.recording_length` does for the old store.
    """

    from ..body_fields import LENGTH_REFERENCE_IOU, TRACE_METHODS

    lengths = []
    for other in Dataset(libraries, record.dataset).labels():
        if other.recording != record.recording or other.sha256 == record.sha256:
            continue
        meta = cached_meta(libraries, other)
        if (meta and meta.get("has_body") and meta.get("in_view") and meta.get("fit_iou", 0.0) >= LENGTH_REFERENCE_IOU
                and meta.get("fit_method") not in TRACE_METHODS):
            lengths.append(float(meta["body_length_px"]))
    return float(np.median(lengths)) if lengths else None


def build_targets(
    libraries: Libraries, record: LabelRecord, *, segmenter: Any = None, device: Any = None, force: bool = False,
) -> dict[str, Any]:
    """Fit and store a label's targets unless they are already built; returns their meta."""

    if not force:
        meta = cached_meta(libraries, record)
        if meta is not None:
            return meta
    import torch

    from .. import body_fields
    from ..head_tracking import HeadTracking
    from ..mask_fit import default_width_template

    label = record.load()
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = body_fields.fit_config()
    count = len(label.context)
    tracking = HeadTracking(
        xy=label.nose_xy.astype(np.float32), valid=label.nose_valid, confidence=np.full(count, np.nan, np.float32),
        frame_indices=np.arange(count) - count // 2 + record.frame, source_frame_ids=np.full(count, -1),
        source_timestamps=np.full(count, -1), provenance={"source": "label"},
    )
    mask = label.mask == 1
    built, arrays, targets = body_fields.fit_targets(
        mask, label.context, label.context_valid, tracking, config=config,
        template=default_width_template(config.n_points), device=device,
        length_px=lambda: recording_length(libraries, record),
        trace=None if label.trace_xy is None else (label.trace_xy, False),
        head_xy=label.head_xy, segmenter=segmenter,
    )
    meta = {
        **built, "label": record.identity, "built_at": utc_now(), "fit_preset": body_fields.FIT_PRESET,
        "max_lag": label.max_lag,
    }
    if targets is not None:
        describe_body(meta, arrays["centerline_xy"], arrays["width_profile"], mask.shape)
    write_targets(libraries, record.sha256, meta, arrays)
    return meta
