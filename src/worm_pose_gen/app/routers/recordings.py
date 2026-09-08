"""Browsing the HDF5 roots: the recording catalog and frame thumbnails."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from . import get_state, query_flag
from ..state import AppState

router = APIRouter(prefix="/api/recordings")


@router.get("")
def recordings(rescan: str | None = None, app: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    return [info.to_dict() for info in app.recordings(query_flag(rescan))]


@router.get("/thumbnail")
def thumbnail(path: str, frame: int = 0, scale: float = 0.25, app: AppState = Depends(get_state)) -> Response:
    return Response(content=app.thumbnail(path, frame, scale), media_type="image/png")
