"""The browser UI from ``worm_pose_gen.app_ui``: ``/`` and ``/static/<file>``."""

from __future__ import annotations

import importlib.resources
import mimetypes

from fastapi import APIRouter
from fastapi.responses import Response

from ..state import NotFound

router = APIRouter()
UI_PACKAGE = "worm_pose_gen.app_ui"


def static_response(name: str) -> Response:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise NotFound(f"no static file {name!r}")
    resource = importlib.resources.files(UI_PACKAGE).joinpath(name)
    if not resource.is_file():
        raise NotFound(f"no static file {name!r}")
    media_type, _ = mimetypes.guess_type(name)
    media_type = media_type or "application/octet-stream"
    if media_type.startswith("text/") or media_type in ("application/javascript", "application/json"):
        media_type += "; charset=utf-8"
    # The UI has no build step; revalidate so a server update is picked up on reload.
    return Response(content=resource.read_bytes(), media_type=media_type, headers={"Cache-Control": "no-cache"})


@router.get("/", include_in_schema=False)
@router.get("/index.html", include_in_schema=False)
def index() -> Response:
    return static_response("index.html")


@router.get("/static/{name}", include_in_schema=False)
def static_file(name: str) -> Response:
    return static_response(name)
