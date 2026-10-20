"""The Workspace page's own endpoints: the Recordings screen, opening a recording, Analyse, a workspace's status and its kymograph (``app/analysis.py``).

- ``GET /api/home?setup=<ref>``: the setups, and the chosen setup's
  recordings (the first setup's when none is named), each with the status
  of its workspace or ``null``.
- ``POST /api/recordings/open`` ``{path, first?, last?, step?}``: the
  recording's workspace, made (not analysed) when it has none; answers
  ``{workspace, status}``.
- ``GET /api/workspaces/{name}/pipeline-settings``: the pipeline settings
  Analyse offers for the workspace's setup, each ``{key, label, type,
  default, help, stages, choices? | min?, max?}``.
- ``POST /api/workspaces/{name}/analyse`` ``{models?: {mask, body},
  mask_source?, pipeline?: {key: value}, run_on?, slurm?, gpu?, stages?}``:
  analyse the workspace;
  answers ``{workspace, job, status}``.  The job is placed like any other
  (``routers/jobs.place``) and cancelled like any other (``POST
  /api/jobs/{id}/cancel``, its id is the status's ``analysis.id``); a
  workspace with a job queued or running is busy (409).
- ``GET /api/workspaces/{name}/status``: models, analysis progress (overall and per stage), state
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


@router.post("/recordings/open")
def open_recording(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    name = analysis.open_recording(app, payload)
    return {"workspace": name, "status": analysis.status(app, name)}


@router.get("/workspaces/{name}/pipeline-settings")
def pipeline_settings(name: str, app: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    return analysis.pipeline_settings(analysis.workspace_setup(app, app.workspace(name)))


@router.post("/workspaces/{name}/analyse")
def analyse(name: str, payload: dict[str, Any] = Body(default={}), app: AppState = Depends(get_state)) -> dict[str, Any]:
    spec, command = analysis.analysis_job(app, name, payload)
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
