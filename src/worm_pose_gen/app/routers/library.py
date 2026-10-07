"""The library for the Workspace, Labeling and Training pages: setups, recordings, datasets, labels, benchmarks, models.

Everything is read from both libraries (:mod:`worm_pose_gen.library`) and
named by reference (``lab:<id>``, ``mine:<id>``); references go in paths as
they are (``/api/library/datasets/lab:nir-labels``).  The writes are the
ones the pages need: create a personal setup or dataset, register a
recording to a setup, save a label revision, freeze a benchmark, and set a
default model.  A write that names a lab item is refused with 403, since
the app never writes the lab library.

Images go to the browser as PNG data URLs: a mask as 0 background, 255
worm, 128 excluded from the loss (:func:`label_app.mask_to_png_values`), and
the A-P field as 0 (undefined) or ``1 + round(254 * ap)``.
"""

from __future__ import annotations

from contextlib import contextmanager
import binascii
from typing import Any, Iterator

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException

from . import get_state
from ... import library
from ...label_app import data_url, decode_mask_data_url, mask_to_png_values
from ...recordings import list_recordings
from ..state import AppState, NotFound, _integer

router = APIRouter(prefix="/api/library")

SORTS = ("", "fit_iou")


@contextmanager
def _found() -> Iterator[None]:
    """A missing library item is a 404 and a write to the lab library a 403."""

    try:
        yield
    except NotFound:
        raise
    except PermissionError as error:
        raise HTTPException(403, str(error)) from error
    except LookupError as error:
        if isinstance(error, KeyError):
            raise
        raise NotFound(str(error)) from error


def _libraries(app: AppState) -> library.Libraries:
    return app.libraries


@router.get("")
def roots(app: AppState = Depends(get_state)) -> dict[str, Any]:
    libraries = _libraries(app)
    return {"lab": None if libraries.lab is None else str(libraries.lab), "lab_available": "lab" in libraries.scopes(),
            "personal": str(libraries.personal)}


# --------------------------------------------------------------------------- setups


@router.get("/setups")
def setups(app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"setups": [setup.to_dict() for setup in library.list_setups(_libraries(app))]}


@router.post("/setups")
def create_setup(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    fields = {key: payload[key] for key in ("name", "description", "video", "pixel_size_um", "fps", "recording_roots") if key in payload}
    if not fields.get("name"):
        raise ValueError("'name' is required")
    with _found():
        return library.create_setup(_libraries(app), str(payload.get("id") or ""), **fields).to_dict()


@router.get("/setups/{ref}")
def setup(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return {**library.get_setup(_libraries(app), ref).to_dict(), "defaults_log": library.defaults_log(_libraries(app), ref)}


@router.post("/setups/{ref}/defaults")
def set_default(ref: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        updated = library.set_default(
            _libraries(app), ref, str(payload.get("role") or ""), str(payload.get("model") or ""), reason=str(payload.get("reason") or ""),
        )
        return {**updated.to_dict(), "defaults_log": library.defaults_log(_libraries(app), ref)}


@router.get("/setups/{ref}/recordings")
def setup_recordings(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The recordings under the setup's roots and those registered to it, with frame counts and readability."""

    libraries = _libraries(app)
    with _found():
        setup = library.get_setup(libraries, ref)
        sources = library.recording_sources(libraries, ref)
    infos = list_recordings(
        sources, poses_root=None, workspaces_root=app.config.workspaces_root, prior_cache=app.config.prior_cache,
        cache=app.config.recordings_cache, dataset=str(setup.video["dataset_path"]),
    )
    registered = library.setups.registered_recordings(libraries)
    rows = []
    for info in infos:
        owner = registered.get(info.path, {}).get("setup")
        if owner is None or owner == ref:  # a file under the roots registered to another setup belongs there
            rows.append({**info.to_dict(), "id": library.recording_id(info.path), "registered": info.path in registered})
    return {"setup": ref, "recordings": rows}


@router.get("/recording-setup")
def recording_setup(path: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"path": path, "id": library.recording_id(path), "setup": library.setup_for_recording(_libraries(app), path)}


@router.post("/recordings")
def register_recording(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return library.register_recording(_libraries(app), str(payload.get("path") or ""), str(payload.get("setup") or ""))


# --------------------------------------------------------------------------- datasets and labels


def _dataset(app: AppState, ref: str) -> library.Dataset:
    with _found():
        return library.Dataset(_libraries(app), ref)


@router.get("/datasets")
def datasets(setup: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return {"datasets": [d.summary() for d in library.list_datasets(_libraries(app), setup or None)]}


@router.post("/datasets")
def create_dataset(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        dataset = library.create_dataset(
            _libraries(app), str(payload.get("id") or ""), setup=str(payload.get("setup") or ""),
            extends=payload.get("extends") or None, name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
        )
        return dataset.summary()


@router.get("/datasets/{ref}")
def dataset(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return _dataset(app, ref).summary()


def _row(app: AppState, record: library.LabelRecord) -> dict[str, Any]:
    meta = library.cached_meta(_libraries(app), record)
    return {**record.to_dict(), "targets": "built" if meta is not None else "missing",
            "fit_iou": None if meta is None else meta.get("fit_iou"),
            "self_contact": None if meta is None else meta.get("self_contact")}


@router.get("/datasets/{ref}/labels")
def dataset_labels(ref: str, recording: str = "", split: str = "", status: str = "", contact: str = "", sort: str = "",
                   app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The dataset's labels (inherited ones included), filtered; ``sort=fit_iou`` puts the lowest body fit IoU first."""

    if status and status not in library.STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {library.STATUSES}")
    if contact not in ("", "yes", "no"):
        raise ValueError("contact must be yes or no")
    if sort not in SORTS:
        raise ValueError(f"unknown sort {sort!r}; expected one of {SORTS}")
    with _found():
        records = library.labels(_libraries(app), [ref], split or None, recording=recording or None, status=status or None)
    rows = [_row(app, record) for record in records]
    if contact:
        rows = [row for row in rows if row["self_contact"] is (contact == "yes")]
    if sort == "fit_iou":
        rows.sort(key=lambda row: (row["fit_iou"] is None, row["fit_iou"] if row["fit_iou"] is not None else 0.0))
    return {"dataset": ref, "total": len(rows), "labels": rows}


def _point(xy: np.ndarray) -> list[float] | None:
    return xy.tolist() if np.all(np.isfinite(xy)) else None


def _targets_layers(app: AppState, record: library.LabelRecord) -> dict[str, Any] | None:
    built = library.load_targets(_libraries(app), record)
    if built is None:
        return None
    meta, arrays = built
    result: dict[str, Any] = {"meta": meta}
    if meta.get("has_body"):
        ap = arrays["ap"].astype(np.float32)
        encoded = np.where(np.isfinite(ap), 1 + np.round(254 * np.clip(np.nan_to_num(ap), 0, 1)), 0).astype(np.uint8)
        result.update(
            ap=data_url(encoded), overlap=data_url(arrays["overlap"].astype(np.uint8) * 255),
            centerline_xy=arrays["centerline_xy"].tolist(), width_profile=arrays["width_profile"].tolist(),
            head_xy=_point(arrays["head_xy"]), tail_xy=_point(arrays["tail_xy"]), diameter_px=float(arrays["diameter_px"]),
        )
    return result


def _label(app: AppState, ref: str, recording: str, frame: int, revision: int | None) -> library.LabelRecord:
    with _found():
        return _dataset(app, ref).get(recording, frame, revision)


@router.get("/datasets/{ref}/labels/{recording}/{frame}")
def label(ref: str, recording: str, frame: int, revision: int | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """A label as the dataset sees it (its own, else inherited), with its images, human fields and built targets."""

    record = _label(app, ref, recording, frame, revision)
    loaded = record.load()
    owner = _dataset(app, record.dataset)
    return {
        "label": _row(app, record), "revisions": [r.to_dict() for r in owner.revisions(recording, frame)],
        "width": record.width, "height": record.height, "meta": loaded.meta,
        "image": data_url(loaded.image), "image_raw": data_url(loaded.image_raw), "mask": data_url(mask_to_png_values(loaded.mask)),
        "orientation": record.orientation, "mask_only": record.mask_only,
        "head_xy": None if loaded.head_xy is None else loaded.head_xy.tolist(),
        "trace_xy": None if loaded.trace_xy is None else loaded.trace_xy.tolist(),
        "max_lag": loaded.max_lag, "context_valid": loaded.context_valid.tolist(),
        "nose_xy": [xy.tolist() if ok else None for xy, ok in zip(loaded.nose_xy, loaded.nose_valid)],
        "targets": _targets_layers(app, record),
    }


@router.get("/datasets/{ref}/labels/{recording}/{frame}/context")
def label_context(ref: str, recording: str, frame: int, revision: int | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    loaded = _label(app, ref, recording, frame, revision).load()
    return {"max_lag": loaded.max_lag, "valid": loaded.context_valid.tolist(), "frames": [data_url(f) for f in loaded.context]}


@router.post("/datasets/{ref}/labels")
def save_label(ref: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    """Save a new revision of a frame's label into a personal dataset.

    With ``path`` the frame, its context and its nose landmarks are read
    from that recording (which must belong to the dataset's setup);
    without it they are taken from the label the dataset already has (its
    own or inherited), so an edit never needs the recording.  The body
    fields are ``orientation`` (``auto`` or ``manual`` with ``head_xy``),
    ``trace_xy`` (head first) and ``mask_only``; ``origin`` is ``spread`` or
    ``fix``; ``expected_revision`` is the dataset's own revision the editor
    started from (0 for none).
    """

    libraries = _libraries(app)
    frame = _integer(payload, "frame")
    origin = str(payload.get("origin") or "")
    if origin not in ("spread", "fix"):
        raise ValueError("'origin' must be spread or fix")
    with _found():
        dataset = _dataset(app, ref)
        if not dataset.writable:
            raise PermissionError(f"{ref} is in the lab library, which is read-only")
        if payload.get("path"):
            path = str(payload["path"])
            owner = library.setup_for_recording(libraries, path)
            if owner != dataset.setup:
                raise ValueError(f"{path} belongs to {owner or 'no setup'}, not to {dataset.setup}; register it first")
            from ...library.capture import read_label_inputs

            setup = library.get_setup(libraries, dataset.setup)
            inputs = read_label_inputs(path, frame, video=setup.video, flat_field_cache=app.config.dataset_root / "flat_fields")
            recording = library.recording_id(path)
        else:
            recording = str(payload.get("recording") or "")
            existing = dataset.get(recording, frame).load()
            inputs = {
                "image": existing.image, "image_raw": existing.image_raw, "context": existing.context,
                "context_valid": existing.context_valid, "nose_xy": existing.nose_xy, "nose_valid": existing.nose_valid,
                "source_path": existing.record.source_path, "dataset_path": existing.record.dataset_path,
            }
        try:
            mask = decode_mask_data_url(str(payload.get("mask") or ""), inputs["image"].shape)
        except (OSError, binascii.Error) as error:
            raise ValueError("mask must contain a readable base64 PNG image") from error
        expected = payload.get("expected_revision")
        record = dataset.save(
            recording=recording, frame=frame, mask=mask, origin=origin,
            orientation=str(payload.get("orientation") or "auto"), head_xy=payload.get("head_xy"),
            trace_xy=payload.get("trace_xy"), mask_only=bool(payload.get("mask_only")),
            expected_revision=None if expected is None else int(expected), **inputs,
        )
    return {"label": _row(app, record)}


# --------------------------------------------------------------------------- benchmarks and models


@router.get("/benchmarks")
def benchmarks(setup: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return {"benchmarks": [b.summary() for b in library.list_benchmarks(_libraries(app), setup or None)]}


@router.post("/benchmarks")
def freeze_benchmark(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        frozen = library.freeze_benchmark(
            _libraries(app), str(payload.get("dataset") or ""), payload.get("id") or None,
            description=str(payload.get("description") or ""),
        )
        return frozen.summary()


@router.get("/benchmarks/{ref}")
def benchmark(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        found = library.get_benchmark(_libraries(app), ref)
    return {**found.summary(), "entries": list(found.entries)}


def _model(app: AppState, card: library.ModelCard) -> dict[str, Any]:
    return {**card.to_dict(), "evaluations": library.evaluations(_libraries(app), card.ref)}


@router.get("/models")
def models(setup: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return {"models": [_model(app, card) for card in library.list_models(_libraries(app), setup or None)]}


@router.get("/models/{ref}")
def model(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    libraries = _libraries(app)
    with _found():
        card = library.get_card(libraries, ref)
        training = library.training_dir(libraries, ref)
    return {**_model(app, card), "training_files": sorted(p.name for p in training.iterdir()) if training.is_dir() else []}
