"""Body-field targets of the corpus store: browse, view with their temporal context, flip, review, rebuild.

The records live in ``<corpus_root>/body_fields`` (:mod:`worm_pose_gen.body_fields`),
so the masks they were built from are the corpus labels the Labels screen
edits; a label saved after its targets were built makes them stale until a
rebuild job refits the tube.  Layers go to the browser as PNG data URLs: the
A-P field as ``0`` (undefined: off the mask or on an overlap) or
``1 + round(254 * ap)``, the mask and overlap as 0/255.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np
from fastapi import APIRouter, Body, Depends

from . import get_state, query_float
from ... import body_fields
from ...body_targets import self_contact
from ...corpus import CorpusStore
from ...jobs import REPO_ROOT, JobSpec
from ...label_app import data_url, mask_to_png_values
from ...segmentation_dataset import SampleRecord
from ..state import AppState, NotFound

router = APIRouter(prefix="/api/body-fields")

JOB_KIND = "body_fields"
STATUSES = ("current", "stale", "missing")
# Summary of a record keyed by path, valid while its (mtime_ns, size) holds.
_summaries: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
_summaries_lock = threading.Lock()
# One trace fit at a time: they share the app's device.
_trace_lock = threading.Lock()


def _store(app: AppState) -> CorpusStore:
    return CorpusStore(app.config.corpus_root)


def _record(store: CorpusStore, sample_id: str) -> SampleRecord:
    record = store.get(sample_id)
    if record is None:
        raise NotFound(f"unknown corpus label {sample_id!r}")
    return record


def _fields_summary(path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _summaries_lock:
        cached = _summaries.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    arrays, meta = body_fields.load(path, ("centerline_xy", "width_profile"))
    summary = {
        "meta": meta,
        "self_contact": bool(meta.get("has_body")) and self_contact(arrays["centerline_xy"], arrays["width_profile"]),
    }
    with _summaries_lock:
        _summaries[path] = (stamp, summary)
    return summary


def _active_jobs(app: AppState) -> dict[str, str]:
    """Sample id -> id of its queued or running rebuild job."""

    return {job.spec.params.get("sample_id"): job.id for job in app.runner.list()
            if job.spec.kind == JOB_KIND and not job.finished}


def _row(store: CorpusStore, record: SampleRecord, jobs: dict[str, str]) -> dict[str, Any]:
    row = {"sample_id": record.sample_id, "recording": record.recording, "frame_index": record.frame_index,
           "split": record.split, "source_path": record.source_path, "mask_revision": record.revision,
           "job": jobs.get(record.sample_id)}
    summary = _fields_summary(body_fields.field_path(store.root, record.sample_id))
    if summary is None:
        return {**row, "status": "missing", "has_body": None, "orientation": None, "fit_iou": None, "fit_method": None,
                "overlap_px": None, "self_contact": None, "review": None, "reviewed_at": None}
    meta = summary["meta"]
    return {**row, "status": "stale" if body_fields.is_stale(meta, record) else "current",
            "fields_revision": meta.get("mask_revision"), "has_body": bool(meta.get("has_body")),
            "orientation": meta.get("orientation"), "fit_iou": meta.get("fit_iou"), "fit_method": meta.get("fit_method"),
            "auto_fit_iou": meta.get("auto_fit_iou"),
            "overlap_px": meta.get("overlap_px"), "orientation_margin": meta.get("orientation_margin"),
            "nose_offset": meta.get("nose_offset"), "self_contact": summary["self_contact"],
            "review": body_fields.review_status(meta), "reviewed_at": meta.get("reviewed_at")}


def _keep(row: dict[str, Any], split: str, orientation: str, review: str, contact: str, status: str, method: str,
          min_iou: float | None, max_iou: float | None) -> bool:
    iou = row["fit_iou"]
    return ((not split or row["split"] == split)
            and (not method or row["fit_method"] == method)
            and (not orientation or row["orientation"] == orientation)
            and (not review or row["review"] == review)
            and (not contact or row["self_contact"] is (contact == "yes"))
            and (not status or row["status"] == status)
            and (min_iou is None or (iou is not None and iou >= min_iou))
            and (max_iou is None or (iou is not None and iou <= max_iou)))


@router.get("")
def samples(split: str = "", orientation: str = "", review: str = "", contact: str = "", status: str = "", method: str = "",
            min_iou: str | None = None, max_iou: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    if review and review not in body_fields.REVIEW_STATES:
        raise ValueError(f"unknown review status {review!r}; expected one of {body_fields.REVIEW_STATES}")
    if contact not in ("", "yes", "no"):
        raise ValueError("contact must be yes or no")
    if status and status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
    store, jobs = _store(app), _active_jobs(app)
    rows = [_row(store, record, jobs) for record in store.records()]
    low, high = query_float(min_iou), query_float(max_iou)
    shown = [row for row in rows if _keep(row, split, orientation, review, contact, status, method, low, high)]
    count = lambda key, values: {value: sum(row[key] == value for row in rows) for value in values}
    return {"root": str(store.root), "total": len(rows), "samples": shown,
            "counts": {"status": count("status", STATUSES), "review": count("review", body_fields.REVIEW_STATES)},
            "facets": {"splits": ["train", "val", "test"], "statuses": list(STATUSES),
                       "reviews": list(body_fields.REVIEW_STATES),
                       "orientations": sorted({row["orientation"] for row in rows if row["orientation"]}),
                       "methods": sorted({row["fit_method"] for row in rows if row["fit_method"]})}}


def _detail(app: AppState, sample_id: str) -> dict[str, Any]:
    store = _store(app)
    record = _record(store, sample_id)
    image, label, _ = store.load(sample_id)
    result: dict[str, Any] = {"sample": _row(store, record, _active_jobs(app)), "record": asdict(record),
                              "width": int(image.shape[1]), "height": int(image.shape[0]),
                              "image": data_url(image), "mask": data_url(mask_to_png_values(label)), "meta": None}
    path = body_fields.field_path(store.root, sample_id)
    if not path.exists():
        return result
    arrays, meta = body_fields.load(path, LAYER_ARRAYS)
    return {**result, **_layers(meta, arrays)}


LAYER_ARRAYS = ("context_valid", "centerline_xy", "width_profile", "ap", "overlap", "head_xy", "tail_xy", "nose_xy",
                "diameter_px", "trace_xy")


def _point(xy: np.ndarray) -> list[float] | None:
    """A point as a list, or ``None`` for an end off camera (stored as NaN; NaN is not JSON)."""

    return xy.tolist() if np.all(np.isfinite(xy)) else None


def _layers(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """A record's meta and drawable layers: the A-P field, overlap, tube, ends, nose and stored trace."""

    result: dict[str, Any] = {"meta": meta, "max_lag": meta["max_lag"], "context_valid": arrays["context_valid"].tolist()}
    if meta.get("has_body"):
        ap = arrays["ap"].astype(np.float32)
        encoded = np.where(np.isfinite(ap), 1 + np.round(254 * np.clip(np.nan_to_num(ap), 0, 1)), 0).astype(np.uint8)
        result.update(
            ap=data_url(encoded), overlap=data_url(arrays["overlap"].astype(np.uint8) * 255),
            centerline_xy=arrays["centerline_xy"].tolist(), width_profile=arrays["width_profile"].tolist(),
            head_xy=_point(arrays["head_xy"]), tail_xy=_point(arrays["tail_xy"]),
            diameter_px=float(arrays["diameter_px"]),
            nose_xy=arrays["nose_xy"].tolist() if "nose_xy" in arrays else None,
            trace_xy=arrays["trace_xy"].tolist() if "trace_xy" in arrays else None,
        )
    return result


@router.get("/{sample_id}")
def detail(sample_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return _detail(app, sample_id)


@router.get("/{sample_id}/context")
def context(sample_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The flat-fielded context frames ``t - max_lag .. t + max_lag`` and which of them are real."""

    store = _store(app)
    _record(store, sample_id)
    path = body_fields.field_path(store.root, sample_id)
    if not path.exists():
        raise NotFound(f"{sample_id} has no body fields; rebuild it first")
    arrays, meta = body_fields.load(path, ("context", "context_valid"))
    return {"max_lag": meta["max_lag"], "valid": arrays["context_valid"].tolist(),
            "frames": [data_url(frame) for frame in arrays["context"]]}


def _check_editable(app: AppState, sample_id: str, *, built: bool = True) -> None:
    store = _store(app)
    _record(store, sample_id)
    if built and not body_fields.field_path(store.root, sample_id).exists():
        raise NotFound(f"{sample_id} has no body fields; rebuild it first")
    job = _active_jobs(app).get(sample_id)
    if job is not None:
        raise ValueError(f"rebuild job {job} is pending for {sample_id}; wait for it to finish")


@router.post("/{sample_id}/flip")
def flip(sample_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    _check_editable(app, sample_id)
    body_fields.flip(app.config.corpus_root, sample_id)
    return _detail(app, sample_id)


@router.post("/{sample_id}/review")
def review(sample_id: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    _check_editable(app, sample_id)
    meta = body_fields.set_review(app.config.corpus_root, sample_id, str(payload.get("status") or ""))
    return {"sample": _row(_store(app), _record(_store(app), sample_id), _active_jobs(app)), "meta": meta}


@router.post("/{sample_id}/trace")
def trace(sample_id: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    """Refit along a traced midline (``points``, head first): a preview, or with ``commit`` the new record.

    Runs in the request, a few seconds on a GPU (about 10 s on a CPU), on the
    app's device; fits are serialized.  The preview returns the same layers
    as ``GET /{id}`` without writing anything.
    """

    _check_editable(app, sample_id)
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError("'points' must be a list of [x, y] pairs, head first")
    with _trace_lock:
        meta, arrays = body_fields.apply_trace(_store(app), sample_id, np.asarray(points, dtype=np.float64),
                                               as_drawn=bool(payload.get("as_drawn")), commit=bool(payload.get("commit")),
                                               device=app.device)
    result = {**_layers(meta, arrays), "committed": bool(payload.get("commit"))}
    if result["committed"]:
        result["sample"] = _row(_store(app), _record(_store(app), sample_id), _active_jobs(app))
    return result


@router.post("/{sample_id}/rebuild")
def rebuild(sample_id: str, payload: dict[str, Any] = Body(default={}), app: AppState = Depends(get_state)) -> dict[str, Any]:
    """Queue a job that refits the tube to the current mask and rewrites the record (unreviewed, orientation re-derived)."""

    _check_editable(app, sample_id, built=False)
    root = Path(app.config.corpus_root).resolve()
    command = [sys.executable, str(REPO_ROOT / "scripts" / "build_body_fields.py"), "--dataset-root", str(root),
               "--samples", sample_id, "--force"]
    if (body_fields.fields_dir(root) / body_fields.REVIEW_DIR).is_dir():
        command.append("--review")
    spec = JobSpec(kind=JOB_KIND, params={"sample_id": sample_id, "root": str(root)},
                   gpus=1 if app.config.gpus else 0, label=f"Rebuild body fields {sample_id}")
    spec.gpu = payload.get("gpu")
    return app.runner.submit(spec, command).to_dict()
