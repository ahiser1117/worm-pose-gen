"""User corpus browsing and editing (the old label store; models train from the library, :mod:`.training`)."""
from __future__ import annotations

from dataclasses import asdict
import binascii
from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state
from ..state import AppState, NotFound, _integer
from ...corpus import CorpusStore, recording_identity
from ...label_app import data_url, decode_mask_data_url, mask_to_png_values
from ...pipeline import workspace_dataset, workspace_lock

router = APIRouter(prefix="/api")


def _store(app: AppState) -> CorpusStore:
    return CorpusStore(app.config.corpus_root)


def _sample(store, sample_id):
    record = store.get(sample_id)
    if record is None:
        raise NotFound(f"unknown corpus label {sample_id!r}")
    return record


@router.get("/corpus")
def corpus(source: str = "", split: str = "", recording: str = "", q: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    store = _store(app)
    with store.locked():
        records = store.records()
        filtered = store.filtered(source=source, split=split, recording=recording, q=q)
        recordings = {recording_identity(r.source_path, r.dataset_path):
                      {"id": recording_identity(r.source_path, r.dataset_path), "path": r.source_path,
                       "dataset": r.dataset_path} for r in records}
        return {"root": str(store.root), "counts": store.counts(),
                "filtered_counts": {s: sum(r.split == s for r in filtered) for s in ("train", "val", "test")},
                "facets": {"sources": sorted({r.label_source for r in records}),
                           "splits": ["train", "val", "test"], "recordings": list(recordings.values())},
                "samples": [asdict(r) for r in filtered]}


@router.post("/corpus/labels")
def save_label(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    name, frame = str(payload.get("workspace") or ""), _integer(payload, "frame")
    view = app.view(name)
    app.check_writable(name)
    with workspace_lock(view.workspace, timeout=2):
        row = view.workspace.row_of(frame)
        if payload.get("mask_revision") is not None and payload["mask_revision"] != view.workspace.mask_revision(row):
            raise ValueError("workspace mask changed before corpus save; reload before saving")
        label = view.workspace.get_override_mask(row)
        if label is None:
            raise ValueError("edit and save this workspace mask before adding it to the corpus")
        if view.source is None:
            raise ValueError(view.source_error or "recording is unavailable")
        raw, corrected = view.source.corrected(frame)
        record = _store(app).save_frame(view.workspace.recording, workspace_dataset(view.workspace), frame,
                                       corrected, label, image_raw=raw, split=payload.get("split"), revision=payload.get("revision"))
    return {"sample": asdict(record), "counts": _store(app).counts(), "root": str(app.config.corpus_root)}


@router.get("/corpus/labels/{sample_id}")
def get_label(sample_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    store = _store(app)
    with store.locked():
        _sample(store, sample_id)
        image, label, record = store.load(sample_id)
        label_url = data_url(mask_to_png_values(label))
        return {"sample": asdict(record), "image": data_url(image), "image_raw": data_url(store.load_raw(sample_id)),
                "label": label_url, "mask": label_url, "encoding": {"background": 0, "worm": 255, "ignore": 128}}


@router.put("/corpus/labels/{sample_id}")
@router.post("/corpus/labels/{sample_id}")
def update_label(sample_id: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    store = _store(app)
    record = _sample(store, sample_id)
    try:
        mask = decode_mask_data_url(str(payload.get("mask") or payload.get("label") or ""), (record.image_height, record.image_width))
    except (OSError, binascii.Error) as error:
        raise ValueError("mask must contain a readable base64 PNG image") from error
    updated = store.update_label(sample_id, mask, payload.get("revision"))
    return {"sample": asdict(updated), "counts": store.counts()}


@router.delete("/corpus/labels/{sample_id}")
def delete_label(sample_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    store = _store(app)
    if not store.delete(sample_id):
        raise NotFound(f"unknown corpus label {sample_id!r}")
    return {"deleted": sample_id, "counts": store.counts()}
