"""Workspace inspection and explicit human review."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, StrictInt
from . import get_state
from ..state import AppState
from ..inspection import inspection, mark_reviewed

router = APIRouter(prefix="/api/workspaces")


class ReviewRequest(BaseModel):
    first: StrictInt
    last: StrictInt
    revision: str


@router.get("/{name}/inspection")
def get_inspection(name: str, app: AppState = Depends(get_state)) -> dict:
    return inspection(app.view(name))


@router.post("/{name}/inspection/review")
def review(name: str, payload: ReviewRequest, app: AppState = Depends(get_state)) -> dict:
    app.check_writable(name)
    return mark_reviewed(app.view(name), payload.first, payload.last, payload.revision)
