"""The job queue: submit a job, list, inspect, cancel, retry, read logs; where jobs can run.

A ``stage`` job runs one pipeline stage over a workspace; its command comes
from ``pipeline.stage_command``, so the process is the same ``python -m
worm_pose_gen.pipeline`` a script would start, and the runner adds
``WORM_POSE_PROGRESS_FILE`` and ``WORM_POSE_JOB_ID`` to its environment,
which is how progress and provenance find their way back.  ``train`` and
``evaluate`` jobs (the Training page, :mod:`.training`) run
``model_training`` and ``model_eval`` the same way; a ``command`` job runs
any argv.  Analyse (``routers/analysis``), the fixes (``routers/fixes``) and
Labeling (``routers/labeling``, ``routers/queues``) submit their own jobs.

Every job is placed by the request: ``run_on`` (``local`` or ``slurm``;
omitted, this machine when it has GPUs for jobs, else SLURM), ``slurm``
(``{"partition", "time"}``, defaults from the host's table) and, for a local
job, ``gpu`` (null lets the queue choose).  ``GET /api/compute`` reports the
choices: what ``compute.detect_compute`` found at startup.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state
from ... import pipeline
from ...jobs import STATES, JobRecord, JobSpec
from .training import evaluation_job, training_job
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
    if stage in ("segment", "prior", "fit"):
        # The models and mask source the workspace was analysed with (Analyse records them).
        for key, default in (("mask_source", "segmenter"), ("checkpoint", None), ("body_net", None)):
            params.setdefault(key, workspace.info.settings.get(key, default))
        params.setdefault("dataset_root", str(app.config.dataset_root))
    spec = JobSpec(
        kind=str(payload.get("kind") or "stage"), params={"stage": stage, "params": params}, workspace=name,
        frames=[int(v) for v in workspace.info.frames], label=str(payload.get("label") or f"{stage} on {name}"),
    )
    return spec, pipeline.stage_command(workspace.path, stage, params)


def default_run_on(app: AppState) -> str | None:
    """This machine when it has GPUs for jobs, else SLURM when available, else None (nothing can run)."""

    if app.config.gpus:
        return "local"
    return "slurm" if app.compute.slurm.available else None


def compute_payload(app: AppState) -> dict[str, Any]:
    """Where jobs can run: this machine's job GPUs, SLURM with its partitions and this host's defaults, and the default choice.

    ``can_run`` is false, with ``reason``, when neither exists; the UI then
    disables training and analysis.
    """

    found = app.compute
    names = {gpu.index: gpu for gpu in found.gpus}
    gpus = [{"index": index, "name": names[index].name if index in names else None,
             "memory_gb": names[index].memory_gb if index in names else None} for index in app.config.gpus]
    local_reason = None if gpus else f"no GPU on {found.host} is enabled for jobs"
    slurm_reason = None if found.slurm.available else f"SLURM is not available ({found.slurm.reason})"
    default = default_run_on(app)
    return {
        "host": found.host,
        "local": {"available": bool(gpus), "reason": local_reason, "gpus": gpus, "max_concurrent": app.runner.max_concurrent},
        "slurm": {
            "available": found.slurm.available, "reason": slurm_reason,
            "partitions": [asdict(partition) for partition in found.slurm.partitions],
            "defaults": asdict(found.slurm.defaults),
        },
        "default_run_on": default,
        "can_run": default is not None,
        "reason": None if default is not None else f"{local_reason}, and {slurm_reason}",
    }


def place(app: AppState, spec: JobSpec, payload: dict[str, Any], previous: JobRecord | None = None) -> JobSpec:
    """Set where ``spec`` runs from a request; a retry (``previous``) keeps the previous placement for what the request omits.

    A retry on the same local machine keeps the previous job's GPU unless
    the request gives ``gpu`` (null opts back into automatic choice).
    """

    if previous is None:
        spec.run_on = str(payload.get("run_on") or default_run_on(app) or "local")
        spec.slurm = payload.get("slurm")
        spec.gpu = payload.get("gpu")
    else:
        spec.run_on = str(payload.get("run_on") or previous.spec.run_on)
        same = spec.run_on == previous.spec.run_on
        spec.slurm = payload["slurm"] if "slurm" in payload else (previous.spec.slurm if same else None)
        if "gpu" in payload:
            spec.gpu = payload["gpu"]
        elif same and spec.run_on == "local":
            spec.gpu = previous.spec.gpu if previous.spec.gpu is not None else previous.gpu
        else:
            spec.gpu = None
    if spec.slurm is not None and not isinstance(spec.slurm, dict):
        raise ValueError("'slurm' must be an object {\"partition\", \"time\"} or null")
    return spec


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
    elif kind == "train":
        spec, command = training_job(app, payload)
    elif kind == "evaluate":
        spec, command = evaluation_job(app, payload)
    elif kind == "command":
        spec, command = command_job(payload)
    else:
        raise ValueError(f"unknown job kind {kind!r}; expected stage, train, evaluate or command")
    return app.runner.submit(place(app, spec, payload), command).to_dict()


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, payload: dict[str, Any] = Body(default={}), app: AppState = Depends(get_state)) -> dict[str, Any]:
    previous = _record(app, job_id)
    if not previous.finished:
        raise ValueError("wait for the job to finish or cancel it before retrying")
    spec = place(app, deepcopy(previous.spec), payload, previous)
    return app.runner.submit(spec, list(previous.command)).to_dict()


@router.get("/compute")
def compute(app: AppState = Depends(get_state)) -> dict[str, Any]:
    return compute_payload(app)


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
