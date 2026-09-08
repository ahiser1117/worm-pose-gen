"""The job queue: submit a stage over a workspace, list, inspect, cancel, read logs; the stage parameter schemas.

A stage job's command comes from ``pipeline.stage_command`` so the process
that runs it is the same ``python -m worm_pose_gen.pipeline`` a script would
start; the runner adds ``WORM_POSE_PROGRESS_FILE`` and ``WORM_POSE_JOB_ID``
to its environment, which is how progress and provenance find their way back.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state
from ... import pipeline
from ...jobs import STATES, JobRecord, JobSpec
from ..state import AppState, NotFound

router = APIRouter(prefix="/api")


def _record(app: AppState, job_id: str) -> JobRecord:
    try:
        return app.runner.get(job_id)
    except KeyError as error:
        raise NotFound(f"unknown job {job_id!r}") from error


def stage_job(app: AppState, payload: dict[str, Any], stage: str) -> tuple[JobSpec, list[str]]:
    if stage not in pipeline.STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {pipeline.STAGES}")
    name = str(payload.get("workspace") or "")
    workspace = app.workspace(name)
    params = dict(payload.get("params") or {})
    spec = JobSpec(
        kind=str(payload.get("kind") or "stage"), params={"stage": stage, "params": params}, workspace=name,
        frames=[int(v) for v in workspace.info.frames], label=str(payload.get("label") or f"{stage} on {name}"),
    )
    return spec, pipeline.stage_command(workspace.path, stage, params)


def command_job(payload: dict[str, Any]) -> tuple[JobSpec, list[str]]:
    command = payload.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise ValueError("a command job needs 'command': a non-empty list of strings")
    spec = JobSpec(kind="command", params={"command": command}, workspace=payload.get("workspace"), label=str(payload.get("label") or command[0]))
    return spec, command


@router.get("/jobs")
def list_jobs(state: str | None = None, app: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    if state not in (None, "") and state not in STATES:
        raise ValueError(f"unknown job state {state!r}; expected one of {STATES}")
    return [record.to_dict() for record in app.runner.list(state or None)]


@router.post("/jobs")
def submit(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    kind = str(payload.get("kind") or "stage")
    if kind == "stage":
        spec, command = stage_job(app, payload, str(payload.get("stage") or ""))
    elif kind == "export":
        spec, command = stage_job(app, payload, "export")
    elif kind == "command":
        spec, command = command_job(payload)
    else:
        raise ValueError(f"unknown job kind {kind!r}; expected stage, export or command")
    return app.runner.submit(spec, command).to_dict()


@router.get("/jobs/{job_id}")
def get_job(job_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return _record(app, job_id).to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    _record(app, job_id)
    return app.runner.cancel(job_id).to_dict()


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str, tail: int | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    _record(app, job_id)
    return {"id": job_id, "tail": tail, "log": app.runner.log(job_id, tail)}


@router.get("/stages")
def stages() -> list[dict[str, Any]]:
    return [{"name": stage, "params": pipeline.stage_schema(stage)} for stage in pipeline.STAGES]
