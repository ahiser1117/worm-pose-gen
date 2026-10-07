"""The Training page and the model picker: models with their benchmark numbers, training runs, evaluations.

The picker (``GET /api/training/models``) is one row per model of a setup,
lab and personal mixed: its inputs (lags in frames and seconds), outputs and
what a missing output means in plain words, what it was trained on, which
default roles it fills, and its numbers on one benchmark (``benchmark``,
default the setup's first lab benchmark), or the evaluation job filling
them in.  A setup without models lists the lab models of the other setups
instead (``others: true``), with their numbers on their own benchmark: how a
new microscope picks its first default.

Jobs go through ``POST /api/jobs`` like every other job:

- ``{"kind": "train", "setup", "datasets": [refs], "start_from": ref | null,
  "context": "none" | "short", "params": {...}, "name", "notes"}`` checks
  the request (:func:`model_training.plan`) and queues
  ``python -m worm_pose_gen.model_training`` (prepare targets, train,
  evaluate on every benchmark, write the card);
- ``{"kind": "evaluate", "model", "benchmark"}`` queues one
  ``python -m worm_pose_gen.model_eval``.

Both take ``run_on`` and ``slurm`` like the other jobs.
``POST /api/training/evaluations`` queues an evaluation for every model of
the setup that lacks one on a benchmark, skipping pairs already queued; the
picker calls it when it shows "Not evaluated", and the Datasets tab after
freezing a benchmark.  ``GET /api/training/runs`` lists a setup's training
jobs that are queued, running, or failed and not yet dismissed, with the
live loss curve each job reports.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse

from . import get_state
from ... import library, model_eval, model_training
from ...jobs import JobRecord, JobSpec
from ...library.roots import read_json, write_json
from ...library.setups import ROLE_OUTPUTS
from ..state import AppState, NotFound

router = APIRouter(prefix="/api/training")

# What a model without an output cannot do, in the picker's words.
MISSING_OUTPUTS = (
    (("head", "tail"), "no head/tail: orientation from body taper only"),
    (("ap",), "no A-P field: coils and contacts are fit from the mask alone"),
    (("overlap",), "no crossing map: self-contacts are not marked"),
)
DISMISSED_FILE = "dismissed_training.json"


@contextmanager
def _found() -> Iterator[None]:
    try:
        yield
    except NotFound:
        raise
    except LookupError as error:
        if isinstance(error, KeyError):
            raise
        raise NotFound(str(error)) from error


def missing_outputs(outputs: tuple[str, ...] | list[str]) -> list[str]:
    return [text for names, text in MISSING_OUTPUTS if any(name not in outputs for name in names)]


def _jobs(app: AppState, kind: str) -> list[JobRecord]:
    return [r for r in app.runner.list() if r.spec.kind == kind]


def _pending_evaluations(app: AppState) -> dict[tuple[str, str], JobRecord]:
    return {(r.spec.params.get("model"), r.spec.params.get("benchmark")): r for r in _jobs(app, "evaluate") if not r.finished}


def _job_summary(record: JobRecord) -> dict[str, Any]:
    return {"id": record.id, "state": record.state, "progress": record.progress, "message": record.message,
            "error": record.error, "result": record.result, "label": record.spec.label, "params": record.spec.params,
            "created_at": record.created_at, "run_on": record.spec.run_on}


def _trained_on(card: library.ModelCard) -> dict[str, Any]:
    entries = list(card.trained_on)
    return {
        "labels": sum(int(e.get("counts", {}).get("train", 0)) + int(e.get("counts", {}).get("val", 0)) for e in entries),
        "recordings": sum(int(e.get("recordings") or 0) for e in entries),
        "datasets": [e.get("dataset") for e in entries],
    }


def _evaluation(found: dict[str, Any] | None) -> dict[str, Any] | None:
    if found is None:
        return None
    keys = ("labels", "iou_mean", "iou_worst5", "worst5_count", "head_tail_correct", "head_tail_labels", "ap_error", "ap_labels",
            "evaluated_at", "benchmark")
    return {key: found.get(key) for key in keys}


def model_row(app: AppState, card: library.ModelCard, setup: library.Setup, benchmark: str | None,
              pending: dict[tuple[str, str], JobRecord]) -> dict[str, Any]:
    """One picker row."""

    evaluations = library.evaluations(app.libraries, card.ref)
    job = pending.get((card.ref, benchmark))
    return {
        "ref": card.ref, "name": card.name, "kind": card.kind, "scope": card.ref.split(":", 1)[0], "setup": card.setup,
        "inputs": card.inputs, "outputs": list(card.outputs), "missing": missing_outputs(card.outputs),
        "roles": [role for role, needed in ROLE_OUTPUTS.items() if all(name in card.outputs for name in needed)],
        "default_for": sorted(role for role, ref in setup.defaults.items() if ref == card.ref),
        "trained_on": _trained_on(card), "parent": card.parent, "author": card.author, "created_at": card.created_at,
        "notes": card.notes, "benchmark": benchmark,
        "evaluation": _evaluation(evaluations.get(benchmark)) if benchmark else None,
        "evaluating": None if job is None else _job_summary(job),
    }


def default_benchmark(benchmarks: list[library.Benchmark]) -> str | None:
    """The setup's lab benchmark (the newest lab one), else its first personal one."""

    lab = [b.ref for b in benchmarks if b.ref.startswith("lab:")]
    if lab:
        return sorted(lab)[-1]
    return benchmarks[0].ref if benchmarks else None


@router.get("/schema")
def schema() -> dict[str, Any]:
    """The Train form: every parameter with its per-kind default, and the from-scratch temporal contexts."""

    return {"parameters": list(model_training.PARAMETERS), "contexts": {k: list(v) for k, v in model_training.CONTEXTS.items()},
            "kinds": list(model_training.KINDS)}


@router.get("/models")
def models(setup: str, benchmark: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    libraries = app.libraries
    with _found():
        found = library.get_setup(libraries, setup)
        benchmarks = library.list_benchmarks(libraries, setup)
        if benchmark and benchmark not in [b.ref for b in benchmarks]:
            raise ValueError(f"{benchmark} is not a benchmark of {setup}")
        chosen = benchmark or default_benchmark(benchmarks)
        cards = library.list_models(libraries, setup)
        pending = _pending_evaluations(app)
        rows = [model_row(app, card, found, chosen, pending) for card in cards]
        others = False
        if not cards:
            others = True
            for card in library.list_models(libraries):
                if card.ref.startswith("lab:") and card.setup != setup:
                    own = library.get_setup(libraries, card.setup)
                    rows.append({**model_row(app, card, found, default_benchmark(library.list_benchmarks(libraries, card.setup)), pending),
                                 "setup_name": own.name})
    rows.sort(key=lambda row: (not row["default_for"], row["scope"] != "lab", row["created_at"] or ""), reverse=False)
    return {"setup": found.to_dict(), "benchmark": chosen, "benchmarks": [b.summary() for b in benchmarks], "models": rows,
            "others": others}


def _labels_used(path: Path) -> list[dict[str, Any]]:
    """The label revisions a model trained on, from ``training/labels.json`` (written by training or by the migration)."""

    data = read_json(path)
    if isinstance(data, dict):
        return [{**entry, "split": split} for split, entries in data.items() for entry in entries]
    if isinstance(data, list):
        return [{**entry["label"], "split": entry.get("old_split") or entry.get("split_now")} for entry in data]
    return []


@router.get("/models/{ref}")
def model_details(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """Details: the card, loss curves, the run record, the labels used by recording, and every evaluation with its worst frames."""

    libraries = app.libraries
    with _found():
        card = library.get_card(libraries, ref)
        training = library.training_dir(libraries, ref)
        setup = library.get_setup(libraries, card.setup)
    used = _labels_used(training / "labels.json")
    by_recording: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in used:
        row = by_recording.setdefault((entry["dataset"], entry["recording"]),
                                      {"dataset": entry["dataset"], "recording": entry["recording"], "train": 0, "val": 0})
        if entry.get("split") in ("train", "val"):
            row[entry["split"]] += 1
    evaluations = {}
    for benchmark, found in library.evaluations(libraries, ref).items():
        evaluations[benchmark] = {
            **_evaluation(found), "rows": found.get("rows", []),
            "worst": [{**w, "url": f"/api/training/models/{ref}/overlays/{benchmark}/{w['file']}"} for w in found.get("worst", [])],
        }
    return {
        "card": card.to_dict(), "missing": missing_outputs(card.outputs), "curve": model_training.read_curve(training / "metrics.csv"),
        "run": read_json(training / "run.json"), "labels": sorted(by_recording.values(), key=lambda r: (r["dataset"], r["recording"])),
        "label_count": len(used), "evaluations": evaluations,
        "defaults_log": [e for e in library.defaults_log(libraries, setup.ref) if e.get("model") == ref],
    }


@router.get("/models/{ref}/overlays/{benchmark}/{name}")
def overlay(ref: str, benchmark: str, name: str, app: AppState = Depends(get_state)) -> FileResponse:
    if "/" in name or name.startswith(".") or not name.endswith(".png"):
        raise NotFound(f"no overlay {name!r}")
    # The personal evaluation replaces a published one, as in ``library.evaluations``.
    with _found():
        stem = library.roots.ref_filename(benchmark)
        directories = library.models.evaluation_dirs(app.libraries, ref)
    for directory in reversed(directories):
        if (directory / f"{stem}.json").is_file():
            path = directory / stem / name
            if path.is_file():
                return FileResponse(path, media_type="image/png")
            break
    raise NotFound(f"no overlay {name!r}")


@router.post("/plan")
def check_plan(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    """Check a Train form without starting it: the kind it trains, label counts, and the generated name."""

    with _found():
        request = model_training.TrainRequest.from_dict(payload)
        return model_training.plan(app.libraries, request, taken=_taken_names(app)).summary()


def _taken_names(app: AppState) -> list[str]:
    return [str(r.spec.params.get("name")) for r in _jobs(app, "train") if not r.finished]


def training_job(app: AppState, payload: dict[str, Any]) -> tuple[JobSpec, list[str]]:
    """``kind: train``: check the request and build the job."""

    request = model_training.TrainRequest.from_dict({k: v for k, v in payload.items() if k not in ("kind", "run_on", "slurm", "gpu")})
    with _found():
        planned = model_training.plan(app.libraries, request, taken=_taken_names(app))
    params = {**planned.summary(), "setup": request.setup, "datasets": list(request.datasets), "start_from": request.start_from,
              "context": request.context, "notes": request.notes}
    spec = JobSpec(kind="train", params=params, label=f"Train {planned.request.name}")
    return spec, model_training.command(app.libraries, request, planned)


def evaluation_job(app: AppState, payload: dict[str, Any]) -> tuple[JobSpec, list[str]]:
    """``kind: evaluate``: one model on one benchmark."""

    model, benchmark = str(payload.get("model") or ""), str(payload.get("benchmark") or "")
    with _found():
        card = library.get_card(app.libraries, model)
        found = library.get_benchmark(app.libraries, benchmark)
    spec = JobSpec(kind="evaluate", params={"model": model, "benchmark": benchmark, "setup": found.setup},
                   label=f"Evaluate {card.name} on {benchmark}")
    return spec, model_eval.command(app.libraries, model, benchmark)


@router.post("/evaluations")
def fill_evaluations(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    """Queue an evaluation for every (model, benchmark) of the setup without one; pairs already queued are skipped."""

    from .jobs import place

    setup = str(payload.get("setup") or "")
    with _found():
        library.get_setup(app.libraries, setup)
        missing = model_eval.missing_evaluations(app.libraries, setup, payload.get("benchmark") or None)
    pending = _pending_evaluations(app)
    queued = []
    for model, benchmark in missing:
        if (model, benchmark) in pending:
            continue
        spec, command = evaluation_job(app, {"model": model, "benchmark": benchmark})
        queued.append(app.runner.submit(place(app, spec, payload), command).to_dict())
    return {"queued": queued, "pending": len(pending)}


def _dismissed(app: AppState) -> set[str]:
    return set(read_json(app.runner.jobs_dir / DISMISSED_FILE, []) or [])


@router.get("/runs")
def runs(setup: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The setup's training jobs to show above the Models table: queued, running, and failed ones not dismissed."""

    dismissed = _dismissed(app)
    shown = [
        _job_summary(r) for r in _jobs(app, "train")
        if r.spec.params.get("setup") == setup and (not r.finished or (r.state == "failed" and r.id not in dismissed))
    ]
    return {"runs": shown}


@router.post("/runs/{job_id}/dismiss")
def dismiss(job_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    try:
        record = app.runner.get(job_id)
    except KeyError as error:
        raise NotFound(f"unknown job {job_id!r}") from error
    if not record.finished:
        raise HTTPException(409, "cancel the run before dismissing it")
    write_json(app.runner.jobs_dir / DISMISSED_FILE, sorted(_dismissed(app) | {job_id}))
    return {"dismissed": job_id}
