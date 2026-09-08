"""Workspaces: create, import a run, list, open, frames, snapshots and the edit log."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state, query_flag, query_float
from ..state import AppState

router = APIRouter(prefix="/api/workspaces")


@router.get("")
def list_all(app: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    return app.workspace_rows()


@router.post("")
def create(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.create_workspace(payload).info()


@router.post("/import")
def import_run(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.import_workspace(payload).info()


@router.get("/{name}")
def open_workspace(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.view(name).payload(app.catalog_entries())


@router.get("/{name}/frame")
def frame(name: str, frame: int, detail: str = "full", threshold: str | None = None, raw: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.view(name).frame(frame, app.viewer.segmenters, query_float(threshold), app.device, raw=query_flag(raw), detail=detail)


@router.get("/{name}/pose")
def pose(name: str, frame: int, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.view(name).pose(frame)


@router.get("/{name}/starts")
def starts(name: str, frame: int, threshold: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.view(name).starts(frame, app.viewer.segmenters, query_float(threshold), app.device)


@router.post("/{name}/snapshot")
def snapshot(name: str, payload: dict[str, Any] = Body(default={}), app: AppState = Depends(get_state)) -> dict[str, Any]:
    workspace = app.workspace(name)
    path = workspace.snapshot(str(payload.get("label") or "snapshot"))
    return {"path": str(path), "name": path.name, "snapshots": workspace.snapshots()}


@router.get("/{name}/edits")
def edits(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"edits": app.workspace(name).edits()}
