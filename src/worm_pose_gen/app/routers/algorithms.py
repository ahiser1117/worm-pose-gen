"""``GET /api/algorithms``: the algorithm registry (``algorithms.list_algorithms``), for the developer's Refit override."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ... import algorithms

router = APIRouter(prefix="/api")


@router.get("/algorithms")
def list_algorithms() -> list[dict[str, Any]]:
    return algorithms.list_algorithms()
