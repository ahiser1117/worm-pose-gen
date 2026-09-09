"""Reversible workspace mask overrides, using the labeler's PNG encoding."""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Body, Depends
from . import get_state
from ..state import AppState
from ... import edits
from ...label_app import decode_mask_data_url

router = APIRouter(prefix="/api/workspaces")


@router.get("/{name}/mask")
def get_mask(name: str, frame: int, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.view(name).mask_payload(frame, app.viewer.segmenters, app.device)


def _edit(name: str, frame: int, encoded: str | None, revision: str | None, app: AppState) -> dict[str, Any]:
    app.check_writable(name)
    view = app.view(name)
    view.refresh()
    workspace = view.workspace
    shape = workspace.image_shape
    if shape is None:
        raise ValueError("recording image shape is unavailable")
    labels = None if encoded is None else decode_mask_data_url(encoded, shape)
    result = edits.set_mask(workspace, workspace.row_of(frame), labels, revision=revision)
    view.invalidate()
    response = view.edit_response(result, frame, app.viewer.segmenters, app.device)
    response["mask"] = view.mask_payload(frame, app.viewer.segmenters, app.device)
    return response


@router.post("/{name}/mask")
def set_mask(name: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return _edit(name, int(payload["frame"]), str(payload["mask"]), payload.get("revision"), app)


@router.delete("/{name}/mask")
def clear_mask(name: str, frame: int, revision: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return _edit(name, frame, None, revision, app)
