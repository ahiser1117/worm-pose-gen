"""A workspace's payloads: the whole workspace (developer tools), one frame, the network's fields on a frame, and exports.

Analyse makes workspaces (``routers/analysis``); the fixes and the issues
live in ``routers/fixes`` and mask edits in ``routers/masks``.  A frame's
full detail runs the segmenter for the probability layers only in developer
mode (``--dev``): analysts see the stored mask.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse
from urllib.parse import quote

from ..exporting import export_workspace, exported_file, list_exports

from . import get_state, query_flag
from ..state import AppState

router = APIRouter(prefix="/api/workspaces")


@router.get("/{name}")
def open_workspace(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.view(name).payload()


@router.get("/{name}/frame")
def frame(name: str, frame: int, detail: str = "full", raw: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.view(name).frame(frame, app.segmenters, app.device, raw=query_flag(raw), detail=detail, segment=app.config.dev)


@router.get("/{name}/network-fields")
def network_fields(name: str, frame: int, outputs: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The body-field network's A-P field, crossings and head/tail on one frame, with ``outputs=1`` every raw output channel (``app/network_fields.py``)."""
    return app.network_fields.frame(name, frame, outputs=query_flag(outputs))


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
