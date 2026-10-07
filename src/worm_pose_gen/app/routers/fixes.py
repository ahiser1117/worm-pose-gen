"""The Issues panel and its fixes (``docs/APP_SIMPLIFICATION.md`` section 2), frames in every request.

- ``GET /api/workspaces/{name}/issues``: the issues with their reasons and
  state, the counts, and the review token ``revision``.
- ``POST .../issues/review`` ``{first, last, revision}``: Looks OK on those
  frames; answers with the issues.
- ``POST .../fixes/flip`` ``{first, last}`` or ``{frame}``: Flip head/tail.
- ``POST .../fixes/refit`` ``{first, last, algorithm?, params?}``: starts a
  refit job; answers ``{job, preview, plan}``.
- ``GET .../fixes/keyframes?first=&last=&spacing=``: Relabel's keyframes.
- ``POST .../fixes/stitch`` ``{keyframes: [{frame, centerline_xy,
  width_profile}], params?}``: starts a stitch job; answers like refit.
- ``GET .../fixes/previews`` and ``GET .../fixes/previews/{id}``: finished
  previews, and one with its before/after poses per frame.
- ``POST .../fixes/previews/{id}/keep`` ``{frame?}`` and
  ``DELETE .../fixes/previews/{id}``: Keep or Discard.
- ``GET .../fixes``: the fixes in force; ``POST .../fixes/{edit}/undo``
  ``{frame?}`` undoes one.

Flip, Keep and Undo answer like the edits (``WorkspaceView.edit_response``)
plus the fixes list; ``frame`` names the frame whose payload comes back.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, StrictInt

from . import get_state
from .. import fixes as fix_ops
from ..inspection import issues, review_issue
from ..state import AppState

router = APIRouter(prefix="/api/workspaces")


class ReviewRequest(BaseModel):
    first: StrictInt
    last: StrictInt
    revision: str


@router.get("/{name}/issues")
def get_issues(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return issues(app.view(name))


@router.post("/{name}/issues/review")
def review(name: str, payload: ReviewRequest, app: AppState = Depends(get_state)) -> dict[str, Any]:
    app.check_writable(name)
    return review_issue(app.view(name), payload.first, payload.last, payload.revision)


@router.post("/{name}/fixes/flip")
def flip(name: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    view = app.view(name)
    app.check_writable(name)
    return fix_ops.flip(view, payload, app.viewer.segmenters, app.device)


@router.post("/{name}/fixes/refit")
def refit(name: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    spec, command, answer = fix_ops.refit_job(app.view(name), payload)
    return {"job": app.runner.submit(spec, command).to_dict(), **answer}


@router.get("/{name}/fixes/keyframes")
def keyframes(name: str, first: int, last: int, spacing: int | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return fix_ops.keyframes(app.view(name), first, last, spacing)


@router.post("/{name}/fixes/stitch")
def stitch(name: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    spec, command, answer = fix_ops.stitch_job(app.view(name), payload)
    return {"job": app.runner.submit(spec, command).to_dict(), **answer}


@router.get("/{name}/fixes/previews")
def list_previews(name: str, app: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    return fix_ops.list_previews(app.view(name))


@router.get("/{name}/fixes/previews/{preview_id}")
def get_preview(name: str, preview_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return fix_ops.preview_payload(app.view(name), preview_id)


@router.post("/{name}/fixes/previews/{preview_id}/keep")
def keep(name: str, preview_id: str, payload: dict[str, Any] = Body(default={}), app: AppState = Depends(get_state)) -> dict[str, Any]:
    view = app.view(name)
    app.check_writable(name)
    return fix_ops.keep(view, preview_id, payload or {}, app.viewer.segmenters, app.device)


@router.delete("/{name}/fixes/previews/{preview_id}")
def discard(name: str, preview_id: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return fix_ops.discard(app.view(name), preview_id)


@router.get("/{name}/fixes")
def list_fixes(name: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return fix_ops.fixes_payload(app.view(name))


@router.post("/{name}/fixes/{edit_id}/undo")
def undo(name: str, edit_id: str, payload: dict[str, Any] = Body(default={}), app: AppState = Depends(get_state)) -> dict[str, Any]:
    view = app.view(name)
    app.check_writable(name)
    return fix_ops.undo(view, edit_id, payload or {}, app.viewer.segmenters, app.device)
