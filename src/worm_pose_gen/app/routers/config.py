"""``GET /api/config``: what the browser shell needs before it shows a page."""

from __future__ import annotations

import getpass
from typing import Any

from fastapi import APIRouter, Depends

from ...settings import DATA_DIR
from ..state import AppState
from . import get_state

router = APIRouter(prefix="/api")


@router.get("/config")
def config(app: AppState = Depends(get_state)) -> dict[str, Any]:
    libraries = app.libraries
    return {
        "dev": app.config.dev,
        # Whether the server's own models run on a GPU: without one, Labeling proposes bodies only when asked.
        "gpu": app.device.type == "cuda",
        "user": getpass.getuser(),
        # Where this checkout keeps what the user makes (settings.json); the app's own roots are below.
        "data_dir": str(DATA_DIR),
        "workspaces": str(app.config.workspaces_root),
        "libraries": {
            "lab": None if libraries.lab is None else str(libraries.lab),
            "lab_available": "lab" in libraries.scopes(),
            "personal": str(libraries.personal),
        },
    }
