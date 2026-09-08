"""The viewer's endpoints: catalog state, run payloads, frames, poses, starts and review notes.

``/api/run`` and ``/api/frame`` take run directory names as before and also
accept a workspace name, so the existing browser code opens either.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state, query_flag, query_float
from ..state import AppState

router = APIRouter(prefix="/api")


@router.get("/state")
def state(rescan: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.state_payload(query_flag(rescan))


@router.get("/run")
def run(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.run_or_workspace_payload(name)


@router.get("/frame")
def frame(run: str, frame: int, detail: str = "full", threshold: str | None = None, raw: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.frame_payload(run, frame, query_float(threshold), query_flag(raw), detail)


@router.get("/pose")
def pose(run: str, frame: int, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.pose_payload(run, frame)


@router.get("/starts")
def starts(run: str, frame: int, threshold: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.starts_payload(run, frame, query_float(threshold))


@router.get("/notes")
def notes(app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"notes": app.viewer.notes.load(), "path": str(app.viewer.notes.path)}


@router.post("/note")
def add_note(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"notes": app.add_note(payload)}


@router.post("/note/delete")
def delete_note(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"notes": app.viewer.notes.delete(int(payload["index"]))}
