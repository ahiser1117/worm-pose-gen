"""Phase 3: the algorithm registry, region proposals, candidate sets and the outcome log.

- ``GET /api/algorithms``: the registry (``algorithms.list_algorithms``).
- ``GET /api/workspaces/{name}/region?frame=`` proposes the region around a
  frame with anchors; ``?first=&last=`` (frames) proposes anchors for a
  region the user typed.
- ``GET /api/workspaces/{name}/candidates`` lists the workspace's candidate
  sets; ``GET .../candidates/{id}`` is one set with every candidate;
  ``POST .../candidates/{id}/accept`` makes its path the poses (an
  ``accept_path`` edit); ``DELETE .../candidates/{id}`` discards it.
- ``GET /api/outcomes?workspace=&algorithm=``: past region runs with their
  metrics before and after and whether they were accepted.

Region jobs are submitted through ``POST /api/jobs`` with ``kind: region``
(``routers/jobs``).  Frames in every request; rows only appear beside the
frames in responses.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state
from ... import algorithms
from .. import regions
from ..state import AppState

router = APIRouter(prefix="/api")


@router.get("/algorithms")
def list_algorithms() -> list[dict[str, Any]]:
    return algorithms.list_algorithms()


@router.get("/workspaces/{name}/region")
def region(name: str, frame: int | None = None, first: int | None = None, last: int | None = None, pad: int = 2, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return regions.propose(app.view(name), frame, first, last, pad=pad)


@router.get("/workspaces/{name}/candidates")
def list_candidates(name: str, app: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    return regions.list_sets(app.view(name))


@router.get("/workspaces/{name}/candidates/{set_id}")
def get_candidates(name: str, set_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    view = app.view(name)
    return regions.set_payload(view, regions.load_set(view, set_id))


@router.post("/workspaces/{name}/candidates/{set_id}/accept")
def accept_candidates(name: str, set_id: str, payload: dict[str, Any] = Body(default={}), app: AppState = Depends(get_state)) -> dict[str, Any]:
    view = app.view(name)
    app.check_writable(name)  # an accept is an edit: not while a job writes the workspace (409)
    return regions.accept(view, set_id, payload or {}, app.viewer.segmenters, app.device)


@router.delete("/workspaces/{name}/candidates/{set_id}")
def discard_candidates(name: str, set_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return regions.discard(app.view(name), set_id)


@router.get("/outcomes")
def outcomes(workspace: str | None = None, algorithm: str | None = None, app: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    return regions.outcomes(app.config.workspaces_root, workspace or None, algorithm or None)
