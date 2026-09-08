"""The pose app: a FastAPI front end for running the pose pipeline and auditing its results.

It serves the viewer's browser UI and API (runs, frames, poses, notes) and
adds what ``docs/APP_PLAN.md`` calls Phase 1: a catalog of the HDF5
recordings under the configured roots, workspaces that hold a recording
range's masks, poses, hypotheses and provenance, and a job queue that runs
pipeline stages over a workspace on the local GPUs.  Everything the UI does
goes through these endpoints, so a script can drive the same work headless.

Errors come back as ``{"error": ...}`` with 400 for a bad request (unknown
frame, bad parameter), 404 for a missing run, workspace, job or file
(``NotFound``), and 500 for anything unexpected, as the stdlib viewer did.
Only the exceptions the endpoints raise for bad input map to 400: a
``RuntimeError`` (a CUDA out-of-memory is one) or a ``TypeError`` is a
server fault and reaches the 500 handler.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..pose_viewer import DEFAULT_CHECKPOINT, DEFAULT_NOTES, DEFAULT_RUNS_ROOT
from ..recordings import DEFAULT_RECORDING_ROOTS
from ..segmentation_dataset import DEFAULT_DATASET_ROOT
from ..workspace import DEFAULT_WORKSPACES_ROOT
from .config import AppConfig
from .state import AppState, NotFound
from .routers import jobs, recordings, static, viewer, workspaces

__all__ = ["AppConfig", "AppState", "NotFound", "create_app", "main", "parse_args"]

BAD_REQUEST_ERRORS = (ValueError, KeyError, IndexError, FileExistsError)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFound)
    async def not_found(request: Request, error: NotFound) -> JSONResponse:
        return _error(404, str(error))

    @app.exception_handler(RequestValidationError)
    async def bad_parameters(request: Request, error: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}" for e in error.errors())
        return _error(400, problems or "invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        return _error(error.status_code, str(error.detail))

    for kind in BAD_REQUEST_ERRORS:
        @app.exception_handler(kind)
        async def bad_request(request: Request, error: Exception) -> JSONResponse:
            message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
            return _error(400, str(message))

    @app.exception_handler(Exception)
    async def failure(request: Request, error: Exception) -> JSONResponse:
        return _error(500, f"{type(error).__name__}: {error}")


def create_app(config: AppConfig | None = None) -> FastAPI:
    """The FastAPI application; its ``AppState`` starts the job runner on startup and stops it on shutdown."""

    config = config or AppConfig()
    state = AppState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state.start()
        try:
            yield
        finally:
            state.close()

    app = FastAPI(title="worm-pose app", lifespan=lifespan)
    app.state.app_state = state
    install_error_handlers(app)
    app.include_router(viewer.router)
    app.include_router(recordings.router)
    app.include_router(workspaces.router)
    app.include_router(jobs.router)
    app.include_router(static.router)
    return app


def _gpu_list(text: str | None) -> tuple[int, ...]:
    if text is None:
        import torch

        return tuple(range(torch.cuda.device_count())) if torch.cuda.is_available() else ()
    return tuple(int(part) for part in text.split(",") if part.strip() != "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--workspaces-root", type=Path, default=DEFAULT_WORKSPACES_ROOT, help="where workspaces (and the job records) live")
    parser.add_argument("--recording-root", action="append", type=Path, dest="recording_roots", help="HDF5 root to browse (repeatable)")
    parser.add_argument("--poses-root", type=Path, default=DEFAULT_RUNS_ROOT, help="directory of fit_recording.py run directories")
    parser.add_argument("--run", action="append", type=Path, dest="runs", help="extra run directory to serve (repeatable)")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="where the flat field cache lives")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="segmenter for on-demand probability maps")
    parser.add_argument("--gpus", default=None, help="comma-separated GPU ids for jobs (default: all visible)")
    parser.add_argument("--max-concurrent", type=int, default=None, help="jobs running at once (default: one per GPU)")
    parser.add_argument("--device", default=None, help="device of the server's own segmenter (default: the first job GPU)")
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES, help="JSON file review notes are appended to")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        host=args.host, port=args.port, workspaces_root=args.workspaces_root,
        recording_roots=tuple(args.recording_roots) if args.recording_roots else tuple(DEFAULT_RECORDING_ROOTS),
        poses_root=args.poses_root, dataset_root=args.dataset_root, checkpoint=args.checkpoint, notes=args.notes,
        gpus=_gpu_list(args.gpus), device=args.device, max_concurrent=args.max_concurrent, extra_runs=tuple(args.runs or ()),
    )


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = parse_args(argv)
    config = config_from_args(args)
    app = create_app(config)
    state: AppState = app.state.app_state
    print(f"pose app at http://{config.host}:{config.port}/", flush=True)
    print(f"{len(state.viewer.catalog)} runs, workspaces in {config.workspaces_root}, jobs on gpus {list(config.gpus)}, device {state.device}", flush=True)
    for path, error in state.viewer.catalog_errors.items():
        print(f"skipped {path}: {error}", flush=True)
    uvicorn.run(app, host=config.host, port=config.port, log_level=args.log_level)
