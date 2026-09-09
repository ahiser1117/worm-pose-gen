"""Manual interventions on a workspace: the segment around a frame, the edit log, and applying an edit.

``POST /api/workspaces/{name}/edits`` takes one of

- ``{"kind": "pick_hypothesis", "frame", "index", "mirrored"?, "note"?}``
- ``{"kind": "flip", "frame", "scope": "frame" | "segment", "note"?}``
- ``{"kind": "flip", "frames": [...], "note"?}``
- ``{"kind": "undo", "edit"?: id, "frame"?}``

and answers with the ``EditResult``, the refreshed light frame payload, a
``series_patch`` of the rows the edit touched, the provenance block and the
edit list, so the browser updates in place without reloading the workspace.
A ``ValueError`` from the edit (unknown frame, bad index, nothing to undo)
is a 400 through the app's error handlers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state
from ..state import AppState

router = APIRouter(prefix="/api/workspaces")


@router.get("/{name}/segment")
def segment(name: str, frame: int, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.view(name).segment(frame)


@router.get("/{name}/edits")
def list_edits(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    view = app.view(name)
    entries = view.edits()
    return {"edits": entries, "count": len(entries)}


@router.post("/{name}/edits")
def apply_edit(name: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.apply_edit(name, payload)
