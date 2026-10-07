"""Analyse: one workspace per recording, run with its setup's models (``docs/APP_SIMPLIFICATION.md`` section 2).

A recording belongs to a setup (:func:`library.setup_for_recording`), and
the setup names the default model of each role (``mask``: the segmenter
whose masks the pipeline fits; ``body``: the body-field network that scores
the fits and gives the A-P field).  **Analyse** takes those models, or the
ones the user picked instead, resolves each reference to its weights, makes
the recording's workspace if it has none (always the whole recording; a
developer may give a range), records the setup and the model references in
the workspace settings (the header and the export read them back), and
submits one job of kind ``analyse`` that runs the default stages in one
process (``pipeline.stages_command``), so its progress spans the whole
analysis and it survives the browser closing.  Analysing again replaces the
poses; the user's mask edits stay.

**Status** answers what the Recordings screen and the Workspace header show
without loading a workspace's arrays: the models, the analysis job and its
progress, and the state (``not_analysed``, ``queued``, ``analysing``,
``failed``, ``analysed``, ``issues``, ``reviewed``, ``exported``).  The issue
counts come from the summary the Issues panel last wrote
(:func:`inspection.cached_issue_summary`); until the workspace is opened
after an analysis they are unknown and the state is ``analysed``.

The **kymograph** is the curvature along the body over time: one column per
workspace row, one row per midline point head to tail, the signed curvature
times the body length (dimensionless, so worms of any size and pixel size
compare) clipped to ``±KYMOGRAPH_RANGE``.  It goes to the browser as one
8-bit PNG: 0 where the frame has no pose, else ``1 + round(254 * (kL +
R) / 2R)`` (128 is straight).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import library, pipeline
from ..jobs import FINISHED_STATES, JobRecord, JobSpec
from .images import data_url
from ..library.setups import ROLE_OUTPUTS
from ..recordings import RecordingInfo, list_recordings
from ..workspace import Workspace, _write_json_atomic, list_workspaces, utc_now
from .exporting import list_exports
from .inspection import cached_issue_summary
from .state import AppState, NotFound

ANALYSE_JOB_KIND = "analyse"
# The model kind that can fill each role today: masks still come from the
# segmenter until a body-field net matches it (section 4, decision 1).
ROLE_KINDS = {"mask": "segmenter", "body": "body_net"}
LAST_OPENED_FILE = "last_opened.json"
KYMOGRAPH_RANGE = 15.0


# ---------------------------------------------------------------------------
# Models and the analysis job


def resolve_models(libraries: library.Libraries, setup: library.Setup, requested: dict[str, Any] | None = None) -> dict[str, dict[str, Any] | None]:
    """The model of each role, ``{ref, name, weights}`` or ``None``: a role named in ``requested`` takes that reference (``None``: no model), else the setup's default.

    A mask model is required (a body model is not: without one the fit
    orients by the body's taper); a model of the wrong kind or without the
    role's outputs is refused.
    """

    requested = dict(requested or {})
    unknown = sorted(set(requested) - set(ROLE_KINDS))
    if unknown:
        raise ValueError(f"unknown model roles {unknown}; expected {sorted(ROLE_KINDS)}")
    models: dict[str, dict[str, Any] | None] = {}
    for role, kind in ROLE_KINDS.items():
        ref = requested[role] if role in requested else setup.defaults.get(role)
        if not ref:
            models[role] = None
            continue
        card = library.get_card(libraries, str(ref))
        if card.kind != kind:
            raise ValueError(f"{ref} is a {card.kind} model; the {role} model must be a {kind}")
        missing = [output for output in ROLE_OUTPUTS[role] if output not in card.outputs]
        if missing:
            raise ValueError(f"{ref} cannot be the {role} model: it has no {', '.join(missing)} output")
        models[role] = {"ref": card.ref, "name": card.name, "weights": str(library.weights_path(libraries, card.ref))}
    if models["mask"] is None:
        raise ValueError(f"no mask model: setup {setup.ref} has no default one; choose one")
    return models


def analysis_params(app: AppState, setup: library.Setup, models: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """The parameters every stage of an analysis reads (``run_all``'s ``'*'``): the models' weights and the setup's video settings."""

    body = models.get("body")
    return {
        "checkpoint": models["mask"]["weights"],  # type: ignore[index]
        "body_net": None if body is None else body["weights"],
        "dataset_root": str(app.config.dataset_root),
        "flat_field": bool(setup.video.get("flat_field", True)),
    }


def analysis_command(workspace_path: Path, stages: list[str], params: dict[str, Any]) -> list[str]:
    return pipeline.stages_command(workspace_path, stages, params)


def _stages(payload: dict[str, Any]) -> list[str]:
    stages = payload.get("stages") or list(pipeline.DEFAULT_STAGES)
    if not isinstance(stages, list) or not all(isinstance(stage, str) for stage in stages):
        raise ValueError("'stages' must be a list of stage names")
    unknown = [stage for stage in stages if stage not in pipeline.STAGES or stage == "export"]
    if unknown:
        raise ValueError(f"cannot analyse with stages {unknown}; choose from {[s for s in pipeline.STAGES if s != 'export']}")
    return [stage for stage in pipeline.STAGES if stage in stages]


def active_jobs(app: AppState, name: str) -> list[JobRecord]:
    return [record for record in app.runner.list() if record.spec.workspace == name and record.state not in FINISHED_STATES]


def analysis_job(app: AppState, payload: dict[str, Any]) -> tuple[str, JobSpec, list[str]]:
    """Analyse the recording at ``payload["path"]``: its workspace (made when missing), settings and the job; returns ``(workspace, spec, argv)``.

    ``models`` (``{role: ref}``) replaces the setup's defaults; ``stages``,
    and for a new workspace ``first``, ``last`` and ``step``, are the
    developer's.  A workspace with a queued or running job is busy.
    """

    libraries = app.libraries
    recording = app.recording_path(str(payload.get("path") or ""))
    setup_ref = library.setup_for_recording(libraries, recording)
    if setup_ref is None:
        raise ValueError(f"{recording} does not belong to a setup; add it to one first")
    setup = library.get_setup(libraries, setup_ref)
    models = resolve_models(libraries, setup, payload.get("models"))
    stages = _stages(payload)
    name = app.workspace_of_recording(recording)
    if name is None:
        name = app.create_workspace({key: payload.get(key) for key in ("first", "last", "step")} | {"recording": str(recording)}).name
    busy = active_jobs(app, name)
    if busy:
        raise pipeline.WorkspaceBusy(f"{busy[0].spec.label or busy[0].spec.kind} is {busy[0].state} on {name}; wait for it or cancel it")
    workspace = Workspace.open(app.workspace_path(name))
    settings = workspace.info.settings
    settings.update({
        "setup": setup_ref,
        "models": {role: None if model is None else model["ref"] for role, model in models.items()},
        "checkpoint": models["mask"]["weights"],  # type: ignore[index]
        "body_net": None if models["body"] is None else models["body"]["weights"],
    })
    with pipeline.workspace_lock(workspace, timeout=0):
        workspace.save_info()
    app.view(name).invalidate()
    params = analysis_params(app, setup, models)
    spec = JobSpec(
        kind=ANALYSE_JOB_KIND, params={"stages": stages, "params": params, "setup": setup_ref, "models": settings["models"]},
        workspace=name, frames=[int(v) for v in workspace.info.frames], label=f"Analyse {recording.stem}",
    )
    return name, spec, analysis_command(workspace.path, stages, params)


# ---------------------------------------------------------------------------
# Status


def _model_names(libraries: library.Libraries, refs: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    out: dict[str, dict[str, Any] | None] = {}
    for role in ROLE_KINDS:
        ref = refs.get(role)
        if not ref:
            out[role] = None
            continue
        try:
            card = library.get_card(libraries, str(ref))
            out[role] = {"ref": card.ref, "name": card.name, "outputs": list(card.outputs)}
        except (LookupError, ValueError):
            out[role] = {"ref": str(ref), "name": str(ref), "outputs": [], "missing": True}
    return out


def workspace_body_net(workspace: Any) -> Path | None:
    """The body-field network the workspace was analysed with (its settings, else its fit's), when the file exists."""

    path = workspace.info.settings.get("body_net") or (pipeline.read_summary(workspace).get("fit_params") or {}).get("body_net")
    return Path(path) if path and Path(path).is_file() else None


def _job(record: JobRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "id": record.id, "kind": record.spec.kind, "label": record.spec.label, "state": record.state, "progress": record.progress,
        "message": record.message, "error": record.error, "run_on": record.spec.run_on, "slurm_state": record.slurm_state,
        "finished_at": record.finished_at,
    }


def last_opened(app: AppState) -> dict[str, str]:
    path = app.config.workspaces_root / LAST_OPENED_FILE
    try:
        return dict(json.loads(path.read_text()))
    except (OSError, ValueError):
        return {}


def mark_opened(app: AppState, name: str) -> str:
    app.workspace_path(name)  # validates the name
    if not app.has_workspace(name):
        raise NotFound(f"unknown workspace {name!r}")
    opened = last_opened(app)
    opened[name] = utc_now()
    _write_json_atomic(app.config.workspaces_root / LAST_OPENED_FILE, opened)
    return opened[name]


def status(app: AppState, name: str, jobs: list[JobRecord] | None = None, opened: dict[str, str] | None = None) -> dict[str, Any]:
    """What the Recordings screen and the Workspace header show of workspace ``name`` (see the module docstring)."""

    if not app.has_workspace(name):
        raise NotFound(f"unknown workspace {name!r}")
    workspace = Workspace.open(app.workspace_path(name))
    settings = workspace.info.settings
    libraries = app.libraries
    setup_ref = settings.get("setup") or library.setup_for_recording(libraries, workspace.recording)
    setup = None
    if setup_ref:
        try:
            found = library.get_setup(libraries, setup_ref)
            setup = {"ref": found.ref, "name": found.name, "pixel_size_um": found.pixel_size_um, "fps": found.fps}
        except (LookupError, ValueError):
            setup = {"ref": setup_ref, "name": setup_ref, "pixel_size_um": None, "fps": None, "missing": True}
    records = [r for r in (app.runner.list() if jobs is None else jobs) if r.spec.workspace == name]
    active = [r for r in records if r.state not in FINISHED_STATES]
    analysis = next((r for r in records if r.spec.kind == ANALYSE_JOB_KIND), None)
    analysed = bool(pipeline.read_summary(workspace).get("fit_config"))
    issues = cached_issue_summary(app.view(name)) if analysed and not active else None
    exports = list_exports(workspace)
    if analysis is not None and not analysis.finished:
        state = "queued" if analysis.state == "queued" or analysis.slurm_state == "PENDING" else "analysing"
    elif not analysed:
        state = "failed" if analysis is not None and analysis.state == "failed" else "not_analysed"
    elif issues is None:
        state = "analysed"
    elif issues["unreviewed"]:
        state = "issues"
    else:
        state = "exported" if exports else "reviewed"
    return {
        "name": name, "recording": str(workspace.recording), "recording_id": workspace.recording.stem,
        "frames": [int(v) for v in workspace.info.frames], "step": int(workspace.info.step), "frame_count": workspace.n,
        "image_shape": workspace.info.image_shape, "setup": setup,
        "models": _model_names(libraries, settings.get("models") or {}),
        "has_body_model": workspace_body_net(workspace) is not None,
        "state": state, "analysed": analysed, "issues": issues,
        "analysis": _job(analysis), "active_jobs": [_job(r) for r in active],
        "exports": len(exports), "last_export": exports[0]["created_at"] if exports else None,
        "last_opened": (last_opened(app) if opened is None else opened).get(name),
    }


def home(app: AppState, setup_ref: str | None) -> dict[str, Any]:
    """The Recordings screen: the setups, and the chosen setup's recordings with their workspace's status (newest workspace per recording)."""

    libraries = app.libraries
    setups = library.list_setups(libraries)
    payload: dict[str, Any] = {"setups": [s.to_dict() for s in setups], "setup": None, "recordings": []}
    if not setups:
        return payload
    setup = library.get_setup(libraries, setup_ref) if setup_ref else setups[0]
    payload["setup"] = setup.to_dict()
    infos = library_recordings(app, setup)
    workspaces: dict[str, str] = {}
    for info in reversed(list_workspaces(app.config.workspaces_root)):  # oldest first, so the newest wins
        workspaces[str(Path(info.recording).resolve())] = info.name
    jobs, opened = app.runner.list(), last_opened(app)
    rows = []
    for info in infos:
        name = workspaces.get(str(Path(info.path).resolve()))
        rows.append({
            "id": library.recording_id(info.path), "path": info.path, "frames": info.frames, "readable": info.readable, "error": info.error,
            "workspace": name, "status": None if name is None else status(app, name, jobs, opened),
        })
    payload["recordings"] = rows
    return payload


def library_recordings(app: AppState, setup: library.Setup) -> list[RecordingInfo]:
    """The setup's recordings (under its roots or registered to it, not registered to another setup)."""

    libraries = app.libraries
    infos = list_recordings(library.recording_sources(libraries, setup.ref), cache=app.config.recordings_cache, dataset=str(setup.video["dataset_path"]))
    registered = library.setups.registered_recordings(libraries)
    return [info for info in infos if registered.get(info.path, {}).get("setup") in (None, setup.ref)]


# ---------------------------------------------------------------------------
# The kymograph


def curvature_kymograph(centerline_xy: np.ndarray, body_length_px: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """Signed curvature times body length, ``(points, rows)``, NaN where a row has no pose."""

    curves = np.asarray(centerline_xy, dtype=np.float64)
    d1 = np.gradient(curves, axis=1)
    d2 = np.gradient(d1, axis=1)
    speed = np.linalg.norm(d1, axis=2)
    cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = cross / np.where(speed > 0, speed**3, np.nan) * np.asarray(body_length_px, dtype=np.float64)[:, None]
    curvature[~np.asarray(fitted, dtype=bool)] = np.nan
    return curvature.T


def encode_kymograph(values: np.ndarray, limit: float = KYMOGRAPH_RANGE) -> np.ndarray:
    scaled = 1 + np.round(254 * (np.clip(np.nan_to_num(values), -limit, limit) + limit) / (2 * limit))
    return np.where(np.isfinite(values), scaled, 0).astype(np.uint8)


def kymograph(app: AppState, name: str) -> dict[str, Any]:
    run = app.view(name).run
    arrays = run.arrays
    values = curvature_kymograph(arrays["centerline_xy"], arrays["body_length_px"], arrays["fitted"])
    return {
        "rows": int(values.shape[1]), "points": int(values.shape[0]), "range": KYMOGRAPH_RANGE,
        "frames": [int(run.frame_index[0]), int(run.frame_index[-1])], "step": int(run.frame_index[1] - run.frame_index[0]) if len(run.frame_index) > 1 else 1,
        "image": data_url(encode_kymograph(values)),
    }
