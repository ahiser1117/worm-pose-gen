"""The pose app: a FastAPI server and its browser UI (``worm_pose_gen.app_ui``) for analysing recordings, labeling frames and training models.

Three pages (``docs/APP_SIMPLIFICATION.md``):

- **Workspace**: the Recordings screen (a setup's recordings with their
  status; Analyse runs the pipeline over a whole recording with the setup's
  default models as one job), and one workspace per recording with its
  issues, the four fixes (Flip, Refit, Edit mask, Relabel), the fixes list
  with Undo, the curvature kymograph and Export.
- **Labeling**: one frame's label (mask, then body) saved into the user's
  dataset, from queues: a workspace's Relabel keyframes, frames a search
  picked, or existing labels.
- **Training**: the model picker, the datasets with their benchmarks, and
  the Train form; models train and are evaluated by ``train`` and
  ``evaluate`` jobs.

Models, datasets, setups and benchmarks live in the lab and personal
libraries (``worm_pose_gen.library``); jobs run on this machine's GPUs or
through SLURM (``worm_pose_gen.jobs``).  ``--dev`` also shows the research
diagnostics.  Everything the UI does goes through these endpoints, so a
script can drive the same work headless.

Errors come back as ``{"error": ...}`` with 400 for a bad request (unknown
frame, bad parameter), 404 for a missing workspace, job, queue or file
(``NotFound``), 409 for a workspace a job is writing, and 500 for anything
unexpected.  Only the exceptions the endpoints raise for bad input map to
400: a ``RuntimeError`` (a CUDA out-of-memory is one) or a ``TypeError`` is
a server fault and reaches the 500 handler.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import fnmatch
import getpass
import socket
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..compute import local_gpus
from ..pipeline import WorkspaceBusy
from ..settings import CACHE_DIR, DATA_DIR
from ..workspace import DEFAULT_WORKSPACES_ROOT
from .config import AppConfig
from .state import AppState, NotFound
from .routers import algorithms, analysis as analysis_routes, config as config_routes, fixes as fixes_routes, jobs, library
from .routers import labeling as labeling_routes, masks, queues, recordings, static, training as training_routes, workspaces

__all__ = ["AppConfig", "AppState", "NotFound", "create_app", "main", "parse_args"]

BAD_REQUEST_ERRORS = (ValueError, KeyError, IndexError, FileExistsError)
ENGAGING_LOGIN = "orcd-login.mit.edu"


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

    @app.exception_handler(WorkspaceBusy)
    async def busy(request: Request, error: WorkspaceBusy) -> JSONResponse:
        return _error(409, str(error))

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
    app.include_router(recordings.router)
    app.include_router(recordings.files_router)
    app.include_router(workspaces.router)
    app.include_router(masks.router)
    app.include_router(labeling_routes.router)
    app.include_router(queues.router)
    app.include_router(fixes_routes.router)
    app.include_router(analysis_routes.router)
    app.include_router(algorithms.router)
    app.include_router(jobs.router)
    app.include_router(library.router)
    app.include_router(training_routes.router)
    app.include_router(config_routes.router)
    app.include_router(static.router)
    return app


DEFAULT_PORT = 8768


def free_port(host: str, start: int = DEFAULT_PORT, tries: int = 100) -> int:
    """``start``, or the first port above it nobody listens on (another user's app may hold it on a shared machine)."""

    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"no free port in {start}-{start + tries - 1}; pass --port")


def tunnel_command(port: int, host: str | None = None, user: str | None = None) -> str:
    """The ``ssh`` command that forwards ``localhost:<port>`` on the user's computer to this server.

    The server listens on this machine's localhost only, so on an Engaging
    compute node the tunnel jumps through the login node to the node itself.
    """

    host = (host or socket.gethostname()).split(".")[0]
    user = user or getpass.getuser()
    if fnmatch.fnmatchcase(host, "node[0-9]*"):
        return f"ssh -N -L {port}:localhost:{port} -J {user}@{ENGAGING_LOGIN} {user}@{host}"
    if fnmatch.fnmatchcase(host, "orcd-*"):
        return f"ssh -N -L {port}:localhost:{port} {user}@{host}.mit.edu"
    return f"ssh -N -L {port}:localhost:{port} {user}@{host}"


def _gpu_list(text: str | None) -> tuple[int, ...]:
    if text is None:
        return tuple(gpu.index for gpu in local_gpus())
    return tuple(int(part) for part in text.split(",") if part.strip() != "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help=f"default: {DEFAULT_PORT}, or the next free port above it")
    parser.add_argument("--workspaces-root", type=Path, default=DEFAULT_WORKSPACES_ROOT, help="where workspaces, queues and the job records live (default: workspaces/ in the data directory)")
    parser.add_argument("--dataset-root", type=Path, default=CACHE_DIR, help="where the flat field cache lives (<dataset-root>/flat_fields; default: cache/ in the data directory)")
    parser.add_argument("--gpus", default=None, help="comma-separated GPU ids for jobs (default: all visible)")
    parser.add_argument("--max-concurrent", type=int, default=None, help="jobs running at once (default: one per GPU)")
    parser.add_argument("--device", default=None, help="device of the server's own models (default: the job GPU with the least memory in use)")
    parser.add_argument("--lab-library", type=Path, default=None, help="the lab's model library, read-only (default: by host name)")
    parser.add_argument("--library", type=Path, default=None, help="your personal library (default: library/ in the data directory, settings.json)")
    parser.add_argument("--dev", action="store_true", help="developer mode: also show the research diagnostics")
    parser.add_argument("--queue", type=Path, default=None,
                        help="a labeling manifest (scripts/build_labeling_manifest.py) to open as a Labeling queue; its recordings must belong to one setup")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        host=args.host, port=free_port(args.host) if args.port is None else args.port, workspaces_root=args.workspaces_root, dataset_root=args.dataset_root,
        lab_library=args.lab_library, library=args.library, dev=args.dev,
        gpus=_gpu_list(args.gpus), device=args.device, max_concurrent=args.max_concurrent,
    )


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = parse_args(argv)
    config = config_from_args(args)
    app = create_app(config)
    state: AppState = app.state.app_state
    page = ""
    if args.queue is not None:
        from .queues import manifest_queue

        page = f"#labeling/queue/{manifest_queue(state, args.queue)['id']}"
    slurm = state.compute.slurm
    print(f"data in {DATA_DIR} (workspaces {config.workspaces_root}), jobs on gpus {list(config.gpus)}, server device {state.device}", flush=True)
    lab = state.libraries.lab
    print(f"libraries: lab {lab}{'' if 'lab' in state.libraries.scopes() else ' (not readable from here)'}, personal {state.libraries.personal}", flush=True)
    print(f"SLURM: {'available' if slurm.available else slurm.reason}; defaults {slurm.defaults}", flush=True)
    print(f"\npose app at http://{config.host}:{config.port}/{page}", flush=True)
    print(f"From your own computer: {tunnel_command(config.port)}\nthen open http://localhost:{config.port}/{page}\n", flush=True)
    uvicorn.run(app, host=config.host, port=config.port, log_level=args.log_level)
