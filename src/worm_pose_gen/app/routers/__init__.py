"""API routers of the pose app, one per resource; each takes the shared ``AppState`` through ``get_state``."""

from __future__ import annotations

from fastapi import Request

from ..state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def query_flag(value: str | None) -> bool:
    return str(value or "0").lower() in ("1", "true", "yes")


def query_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)
