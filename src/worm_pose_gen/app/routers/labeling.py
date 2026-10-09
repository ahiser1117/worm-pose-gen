"""The Labeling page's frame API (``app/labeling.py``): open, context, mask proposal and refinement, body proposal and fit, save.

Every request names the setup and the entry: ``{"setup", "entry":
{"recording", "frame", "path"?}, "queue"?}``.  Labels are saved into the
setup's label collection.  ``"models": {"mask"?, "body"?}`` chooses the
models of the open, network and proposal requests in place of the setup's
defaults.

- ``POST /api/labeling/open``: the frame, its mask and where it came
  from, the label (if any) with its body and targets, the models used and the defaults.
- ``POST /api/labeling/context``: the context frames t-16..t+16.
- ``POST /api/labeling/network``: the mask model's worm probability.
- ``POST /api/labeling/refine`` ``{mask, width, height, method}``:
  ``fill_holes``, ``largest``, ``grow`` or ``shrink``.
- ``POST /api/labeling/proposal`` ``{..., mask}``: the body model's proposal.
- ``POST /api/labeling/fit`` ``{..., mask, trace_xy, extend?}``: the body along a
  trace (``extend``: continued off camera to the body length when it ends at the border).
- ``POST /api/labeling/save`` ``{..., mask, trace_xy?, trace_extend?, trace_length_px?, head_xy?,
  mask_only, expected_revision}``: a new label revision, its targets job, the queue.

Masks travel as PNG data URLs (0 background, 255 worm, 128 excluded).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from . import get_state
from ..state import AppState

router = APIRouter(prefix="/api/labeling")


@router.post("/open")
def open_frame(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.labeling.open(payload)


@router.post("/context")
def context(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.labeling.context(payload)


@router.post("/network")
def network(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.labeling.network(payload)


@router.post("/refine")
def refine(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.labeling.refine(payload)


@router.post("/proposal")
def proposal(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.labeling.proposal(payload)


@router.post("/fit")
def fit(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.labeling.fit(payload)


@router.post("/save")
def save(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    return app.labeling.save(payload)
