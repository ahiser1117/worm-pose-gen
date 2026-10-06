"""Mask labeling: frame drafts, proposals, refinement, corpus saves, and the label groups Paint walks."""
from fastapi import APIRouter, Body, Depends
from . import get_state
from ..state import AppState

router = APIRouter(prefix='/api/labeling')


@router.post('/frame')
def frame(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.frame(payload.get('target'), payload.get('group_id'))


@router.post('/proposals')
def proposals(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.proposals(payload)


@router.post('/refine')
def refine(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.refine(payload)


@router.post('/save')
def save(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.save(payload)


@router.get('/groups')
def groups(app: AppState = Depends(get_state)):
    return app.labeling.list_groups()


@router.post('/groups')
def open_group(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.open_group(payload)


@router.get('/groups/{group_id}')
def group(group_id: str, app: AppState = Depends(get_state)):
    return app.labeling.group(group_id)


@router.delete('/groups/{group_id}')
def close_group(group_id: str, app: AppState = Depends(get_state)):
    return app.labeling.close_group(group_id)
