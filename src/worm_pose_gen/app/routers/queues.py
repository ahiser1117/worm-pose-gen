"""Labeling queues (``app/queues.py``) and the Relabel round trip with the Workspace.

- ``GET /api/queues?setup=``: the queues, newest first, without entries
  (``{id, kind, name, setup, state, progress: {total, saved, remaining},
  first_unsaved, complete, workspace, job, ...}``).
- ``POST /api/queues``: ``{kind: "relabel", workspace, frames: [frame, ...]}``
  (the Workspace's Relabel) or ``{kind: "spread", setup, recordings:
  [path, ...], frames: N, dataset?}`` (New queue: starts a Find-frames job,
  ``job`` names it); answers with the queue's summary, ``id`` among it.
- ``GET /api/queues/{id}``: the summary with ``entries`` (``{path,
  recording, frame, uncertainty?, saved}``).
- ``DELETE /api/queues/{id}``.
- ``POST /api/queues/{id}/stitch`` ``{params?}``: a finished Relabel queue's
  labels become mask overrides in its workspace and a stitch job starts;
  answers ``{job, preview, plan}`` like ``/api/workspaces/{ws}/fixes/stitch``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state
from .. import queues
from ..state import AppState

router = APIRouter(prefix="/api/queues")


@router.get("")
def list_queues(setup: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"queues": [queues.summary(q) for q in app.queues.list(setup or None)]}


@router.post("")
def create(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    kind = payload.get("kind")
    if kind == "relabel":
        queue = queues.relabel_queue(app, str(payload.get("workspace") or ""), payload.get("frames"))
    elif kind == "spread":
        queue = queues.spread_queue(app, str(payload.get("setup") or ""), payload.get("recordings"), payload.get("frames"),
                                    payload.get("dataset") or None)
    else:
        raise ValueError("'kind' must be relabel or spread")
    return queues.summary(queue)


@router.get("/{queue_id}")
def get_queue(queue_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return queues.detail(app.queues.get(queue_id))


@router.delete("/{queue_id}")
def delete_queue(queue_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    app.queues.delete(queue_id)
    return {"deleted": queue_id}


@router.post("/{queue_id}/stitch")
def stitch(queue_id: str, payload: dict[str, Any] = Body(default={}), app: AppState = Depends(get_state)) -> dict[str, Any]:
    params = (payload or {}).get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("'params' must be an object of parameter values")
    return queues.stitch(app, queue_id, params)
