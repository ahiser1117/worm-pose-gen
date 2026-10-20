"""Edit mask on the Workspace page: a frame's mask to paint on, and saving it as a reversible override (``app/images`` PNG encoding).

``GET /api/workspaces/{name}/mask?frame=`` answers with the frame, its
current mask (the override, else the stored mask) and the network's
proposal; ``POST`` ``{frame, mask, revision}`` saves the override as one
``set_mask`` edit (Undo in the fixes list removes it) and answers like an
edit, plus the frame's new mask payload.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state
from ... import edits
from ..images import decode_mask_data_url
from ..state import AppState

router = APIRouter(prefix="/api/workspaces")


@router.get("/{name}/mask")
def get_mask(name: str, frame: int, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.view(name).mask_payload(frame, app.segmenters, app.device)


@router.post("/{name}/mask")
def set_mask(name: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    frame = int(payload["frame"])
    app.check_writable(name)
    view = app.view(name)
    view.refresh()
    workspace = view.workspace
    shape = workspace.image_shape
    if shape is None:
        raise ValueError("recording image shape is unavailable")
    labels = decode_mask_data_url(str(payload["mask"]), shape)
    result = edits.set_mask(workspace, workspace.row_of(frame), labels, revision=payload.get("revision"))
    view.invalidate()
    response = view.edit_response(result, frame, app.segmenters, app.device)
    response["mask"] = view.mask_payload(frame, app.segmenters, app.device)
    return response
