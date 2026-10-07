"""The Workspace page's own endpoints: the Recordings screen, Analyse, a workspace's status and its kymograph (``app/analysis.py``).

- ``GET /api/home?setup=<ref>``: the setups, and the chosen setup's
  recordings (the first setup's when none is named), each with the status
  of its workspace or ``null``.
- ``POST /api/analyse`` ``{path, models?: {mask, body}, run_on?, slurm?,
  gpu?, stages?, first?, last?, step?}``: analyse a recording; answers
  ``{workspace, job, status}``.  The job is placed like any other
  (``routers/jobs.place``); a workspace with a job queued or running is busy
  (409).
- ``GET /api/workspaces/{name}/status``: models, analysis progress, state
  and issue counts, without loading the arrays.
- ``POST /api/workspaces/{name}/opened``: remember when the workspace was
  last opened (the Recordings screen shows it).
- ``GET /api/workspaces/{name}/kymograph``: curvature along the body over
  time as one PNG.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state
from .. import analysis
from ..state import AppState
from .jobs import place

router = APIRouter(prefix="/api")


@router.get("/home")
def home(setup: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    return analysis.home(app, setup or None)


@router.post("/analyse")
def analyse(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    name, spec, command = analysis.analysis_job(app, payload)
    record = app.runner.submit(place(app, spec, payload), command)
    return {"workspace": name, "job": record.to_dict(), "status": analysis.status(app, name)}


@router.get("/workspaces/{name}/status")
def status(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return analysis.status(app, name)


@router.post("/workspaces/{name}/opened")
def opened(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"name": name, "last_opened": analysis.mark_opened(app, name)}


@router.get("/workspaces/{name}/kymograph")
def kymograph(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return analysis.kymograph(app, name)
