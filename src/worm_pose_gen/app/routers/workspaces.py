"""Workspaces: create, import a run, list, open, frames and snapshots (the edits live in ``routers/edits``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse
from urllib.parse import quote

from ...pipeline import workspace_lock
from ..exporting import export_workspace, exported_file

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
    app.check_writable(name)
    with workspace_lock(workspace, timeout=0):
        path = workspace.snapshot(str(payload.get("label") or "snapshot"))
    return {"path": str(path), "name": path.name, "snapshots": workspace.snapshots()}


@router.post("/{name}/export")
def export(name: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    result = export_workspace(app, name, str(payload.get("name") or ""))
    result["download_url"] = f"/api/workspaces/{quote(name, safe='')}/exports/{quote(result['snapshot'], safe='')}/{quote(result['name'], safe='')}.parquet"
    return result


@router.get("/{name}/exports/{snapshot_name}/{filename}")
def download(name: str, snapshot_name: str, filename: str, app: AppState = Depends(get_state)):
    try:
        path = exported_file(app.workspace(name), snapshot_name, filename)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, filename=filename, media_type="application/vnd.apache.parquet")
