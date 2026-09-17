"""Browsing the HDF5 roots: the recording catalog and frame thumbnails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from . import get_state, query_flag
from ..state import AppState
from ...pipeline import workspace_dataset

router = APIRouter(prefix="/api/recordings")


@router.get("")
def recordings(rescan: str | None = None, app: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    return [info.to_dict() for info in app.recordings(query_flag(rescan))]


@router.get("/datasets")
def datasets(path: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.datasets(path)


@router.post("/register")
def register(payload: dict[str, Any], app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.register_recording(payload).to_dict()


def preparation_source(app: AppState, path: str | None = None, dataset: str | None = None, workspace: str | None = None, run: str | None = None):
    if workspace:
        work = app.workspace(workspace)
        recording, dataset = work.recording, workspace_dataset(work)
    elif run:
        # Resolve through the existing run catalog, not an arbitrary file path.
        entry = app.viewer.catalog.get(run)
        if entry is None:
            raise ValueError("Unknown run")
        recording = Path(entry["recording_path"])
    else:
        recording = app.recording_path(path or "")
    source, error = app.viewer._source(str(recording.resolve()), dataset or app.recording_dataset(recording))
    if source is None:
        raise ValueError(error or "Cannot read recording")
    return source


@router.get("/preparation")
def preparation(path: str | None = None, dataset: str | None = None, workspace: str | None = None, run: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return dict(preparation_source(app, path, dataset, workspace, run).preparation)


@router.post("/prepare")
def prepare(payload: dict[str, Any], app: AppState = Depends(get_state)) -> dict[str, Any]:
    source = preparation_source(app, **{key: payload.get(key) for key in ("path", "dataset", "workspace", "run")})
    source.flat_field()
    return dict(source.preparation)


@router.post("/unregister")
def unregister(payload: dict[str, Any], app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"removed": app.unregister_recording(payload)}


@router.get("/thumbnail")
def thumbnail(path: str, frame: int = 0, scale: float = 0.25, app: AppState = Depends(get_state)) -> Response:
    return Response(content=app.thumbnail(path, frame, scale), media_type="image/png")


files_router = APIRouter(prefix="/api/files")


@files_router.get("")
def browse(path: str | None = None, all: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The file explorer: directories and HDF5 files under ``path`` (every file with ``all=1``)."""

    return app.browse(path, query_flag(all))
