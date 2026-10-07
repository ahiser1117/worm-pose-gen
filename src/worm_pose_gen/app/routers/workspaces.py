"""Workspaces: create, import a run, list, open, frames, network predictions, snapshots and exports (the edits live in ``routers/edits``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse
from urllib.parse import quote

from ...pipeline import workspace_lock
from ..exporting import export_workspace, exported_file, list_exports

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


@router.get("/{name}/network-fields")
def network_fields(name: str, frame: int, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The body-field network's A-P field, crossings and head/tail on one frame (``app/network_fields.py``)."""
    return app.network_fields.frame(name, frame)


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


def _with_urls(name: str, export: dict[str, Any]) -> dict[str, Any]:
    base = f"/api/workspaces/{quote(name, safe='')}/exports/{quote(export['name'], safe='')}"
    return {**export, "download_url": f"{base}/{quote(export['table'], safe='')}", "metadata_url": f"{base}/export.json"}


@router.post("/{name}/export")
def export(name: str, payload: dict[str, Any] = Body(default={}), app: AppState = Depends(get_state)) -> dict[str, Any]:
    """Export the workspace now (named ``<recording>_<UTC time>``); the body may give ``pixel_size_um``, ``fps`` and ``setup``."""
    return _with_urls(name, export_workspace(app, name, payload))


@router.get("/{name}/exports")
def exports(name: str, app: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    return [_with_urls(name, export) for export in list_exports(app.workspace(name))]


@router.get("/{name}/exports/{export_name}/{filename}")
def download(name: str, export_name: str, filename: str, app: AppState = Depends(get_state)):
    try:
        path = exported_file(app.workspace(name), export_name, filename)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    media_type = "application/json" if path.suffix == ".json" else "application/vnd.apache.parquet"
    return FileResponse(path, filename=filename, media_type=media_type)
