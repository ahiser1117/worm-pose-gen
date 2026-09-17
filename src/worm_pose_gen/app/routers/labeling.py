"""Draft mask tools and corpus labeling without implicit workspace mutation."""
from fastapi import APIRouter, Body, Depends
from . import get_state
from ..state import AppState

router = APIRouter(prefix='/api/labeling')


@router.post('/frame')
def frame(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.frame(payload.get('target'))


@router.post('/proposals')
def proposals(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.proposals(payload)


@router.post('/refine')
def refine(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.refine(payload)


@router.post('/save')
def save(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.save(payload)


@router.post('/next')
def next_frame(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.next(payload)


@router.post('/manifests')
def load_manifest(payload: dict = Body(...), app: AppState = Depends(get_state)):
    return app.labeling.load_manifest(payload['path'])


@router.get('/manifests')
def manifests(app: AppState = Depends(get_state)):
    return {'manifests': app.labeling.list_manifests()}
