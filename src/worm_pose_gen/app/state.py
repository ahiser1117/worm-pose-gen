"""The application's shared state: the run catalog, workspaces, recordings and the job queue.

One instance per server, attached to the FastAPI app.  It owns the viewer's
``ViewerState`` (run directories, recordings opened read-only, segmenters
loaded once), a ``WorkspaceView`` per opened workspace, and the ``JobRunner``
whose background thread starts stage processes on the configured GPUs.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import threading
from typing import Any

import torch

from ..jobs import JobRunner, LocalGPUBackend
from ..pose_viewer import ViewerState
from ..pipeline import workspace_dataset
from ..recordings import DATASET_PATH, RecordingInfo, RecordingRegistry, default_video_dataset, hdf5_datasets, list_directory, list_recordings, probe_recording, thumbnail_png
from ..workspace import Workspace, list_workspaces
from .config import AppConfig
from .workspace_view import WorkspaceView


class NotFound(LookupError):
    """A run, workspace, job or file the request named does not exist."""


def _validate_name(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"invalid workspace name {name!r}")
    return name


def _integer(payload: dict[str, Any], key: str, default: int | None = None) -> int:
    """An integer field of a request body; a missing or malformed value is a bad request, not a server fault."""

    value = payload.get(key)
    if value is None or value == "":
        if default is None:
            raise ValueError(f"'{key}' is required")
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"'{key}' must be an integer, got {value!r}") from error


class AppState:
    """Everything the endpoints share; see the module docstring."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        config.workspaces_root.mkdir(parents=True, exist_ok=True)
        runs = [p for p in config.extra_runs] + [p for p in ViewerState.discover(config.poses_root) if p not in config.extra_runs]
        self.viewer = ViewerState(
            runs, dataset_root=config.dataset_root, checkpoint=config.checkpoint, device=config.viewer_device, notes=config.notes,
            runs_root=config.poses_root,
        )
        self.runner = JobRunner(config.jobs_root, LocalGPUBackend(list(config.gpus)), max_concurrent=config.max_concurrent)
        self._views: dict[str, WorkspaceView] = {}
        self._lock = threading.Lock()
        self._recordings_lock = threading.Lock()
        self.registry = RecordingRegistry(config.workspaces_root / "recordings_registry.json")

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.runner.recover()
        self.runner.start(interval=self.config.job_interval)

    def close(self) -> None:
        self.runner.stop()
        self.viewer.close()

    @property
    def device(self) -> torch.device:
        return self.viewer.device

    # ---------------------------------------------------------------- workspaces

    def workspace_path(self, name: str) -> Path:
        return self.config.workspaces_root / _validate_name(name)

    def workspace(self, name: str) -> Workspace:
        return self.view(name).workspace

    def view(self, name: str) -> WorkspaceView:
        """The (cached) view of a workspace; ``NotFound`` when there is no such directory."""

        path = self.workspace_path(name)
        with self._lock:
            view = self._views.get(name)
        if view is None or not path.exists():
            if not (path / "workspace.json").exists():
                with self._lock:
                    self._views.pop(name, None)
                raise NotFound(f"unknown workspace {name!r}")
            workspace = Workspace.open(path)
            source, error = self.viewer._source(str(workspace.recording), workspace_dataset(workspace))
            view = WorkspaceView(workspace, source, error)
            with self._lock:
                view = self._views.setdefault(name, view)
        return view

    def has_workspace(self, name: str) -> bool:
        try:
            return (self.workspace_path(name) / "workspace.json").exists()
        except ValueError:
            return False

    def workspace_names(self) -> list[str]:
        return [info.name for info in list_workspaces(self.config.workspaces_root)]

    def workspace_rows(self) -> list[dict[str, Any]]:
        """``WorkspaceInfo`` plus summary of every workspace, newest first."""

        rows = []
        for name in self.workspace_names():
            try:
                rows.append(self.view(name).info())
            except (NotFound, OSError, ValueError, KeyError) as error:
                rows.append({"name": name, "path": str(self.workspace_path(name)), "kind": "workspace", "error": f"{type(error).__name__}: {error}"})
        return rows

    def catalog_entries(self) -> list[dict[str, Any]]:
        """Run and workspace catalog rows together, for the compatible-runs lists."""

        entries = list(self.viewer.catalog.values())
        for name in self.workspace_names():
            try:
                entries.append(self.view(name).entry())
            except (NotFound, OSError, ValueError, KeyError):
                continue
        return entries

    def create_workspace(self, payload: dict[str, Any]) -> WorkspaceView:
        name = _validate_name(str(payload["name"]))
        recording = self.recording_path(str(payload["recording"]))
        if not recording.is_file():
            raise ValueError(f"recording {recording} does not exist")
        settings = dict(payload.get("settings") or {})
        # A registered recording brings its dataset name; /img_nir needs no setting.
        dataset = settings.get("dataset") or self.registry.dataset_of(recording) or DATASET_PATH
        if dataset != DATASET_PATH:
            settings["dataset"] = dataset
        Workspace.create(
            self.config.workspaces_root, name, recording, _integer(payload, "first"), _integer(payload, "last"), _integer(payload, "step", 1),
            settings=settings or None,
        )
        return self.view(name)

    def import_workspace(self, payload: dict[str, Any]) -> WorkspaceView:
        run_dir = self.resolve_run(str(payload["run"]))
        name = payload.get("name")
        workspace = Workspace.import_run(self.config.workspaces_root, run_dir, None if name in (None, "") else _validate_name(str(name)))
        return self.view(workspace.info.name)

    def resolve_run(self, run: str) -> Path:
        """A run directory by catalog name, by name under the poses root, or by path."""

        candidates = []
        if run in self.viewer.catalog:
            candidates.append(Path(self.viewer.catalog[run]["path"]))
        if "/" not in run:
            candidates.append(self.config.poses_root / run)
        candidates.append(Path(run))
        for candidate in candidates:
            if (candidate / "summary.json").exists() and (candidate / "poses.npz").exists():
                return candidate
        raise NotFound(f"unknown run {run!r}")

    # ---------------------------------------------------------------- recordings

    def recordings(self, rescan: bool = False) -> list[RecordingInfo]:
        cache = self.config.recordings_cache
        # One listing at a time: a rescan unlinks the cache another listing may be writing.
        with self._recordings_lock:
            if rescan and cache is not None and cache.exists():
                cache.unlink()
            return list_recordings(
                self.config.recording_roots, poses_root=self.config.poses_root, workspaces_root=self.config.workspaces_root,
                prior_cache=self.config.prior_cache, cache=cache, registry=self.registry,
            )

    def recording_path(self, path: str) -> Path:
        """``path`` as a recording under one of the configured roots or registered by hand; ``NotFound`` for anything else."""

        recording = Path(path).expanduser()
        try:
            resolved = recording.resolve()
        except OSError as error:
            raise NotFound(f"no recording at {path}") from error
        if self.registry.dataset_of(resolved) is not None:
            return resolved
        if not any(resolved.is_relative_to(root.resolve()) for root in self.config.recording_roots):
            raise NotFound(f"{path} is not under the configured recording roots and is not a registered recording")
        return recording

    def recording_dataset(self, path: Path) -> str:
        return self.registry.dataset_of(path) or DATASET_PATH

    # ----------------------------------------------------------------- file explorer

    def browse(self, path: str | None) -> dict[str, Any]:
        """Directories and HDF5 files under ``path`` (the first recording root when none is given), with the roots as shortcuts."""

        start = Path(path).expanduser() if path else (self.config.recording_roots[0] if self.config.recording_roots else Path.home())
        try:
            listing = list_directory(start)
        except FileNotFoundError as error:
            raise NotFound(str(error)) from error
        except (NotADirectoryError, PermissionError) as error:
            raise ValueError(str(error)) from error
        listing["shortcuts"] = [{"name": p.name or str(p), "path": str(p)} for p in (*self.config.recording_roots, Path.home()) if p.exists()]
        registered = {str(p) for p in self.registry.paths()}
        for entry in listing["entries"]:
            entry["registered"] = entry["path"] in registered
        return listing

    def datasets(self, path: str) -> dict[str, Any]:
        """The datasets of one HDF5 file and the one a recording would read by default."""

        file = Path(path).expanduser()
        if not file.is_file():
            raise NotFound(f"no file at {path}")
        try:
            datasets = hdf5_datasets(file)
        except OSError as error:
            raise ValueError(f"{path}: not an HDF5 file ({error})") from error
        return {"path": str(file.resolve()), "datasets": datasets, "default": default_video_dataset(datasets), "registered": self.registry.dataset_of(file) is not None}

    def register_recording(self, payload: dict[str, Any]) -> RecordingInfo:
        """Add a recording by path (any readable HDF5 file) with the dataset holding its frames."""

        file = Path(str(payload["path"])).expanduser()
        if not file.is_file():
            raise NotFound(f"no file at {file}")
        dataset = str(payload.get("dataset") or "").strip()
        if not dataset:
            dataset = default_video_dataset(hdf5_datasets(file)) or ""
            if not dataset:
                raise ValueError("this file has several video-like datasets; name the one to use")
        facts = probe_recording(file, dataset)
        if not facts["readable"]:
            raise ValueError(f"{file} cannot be read as a recording with dataset {dataset}: {facts['error']}")
        self.registry.add(file, dataset)
        rec = next((r for r in self.recordings(rescan=False) if Path(r.path).resolve() == file.resolve()), None)
        if rec is None:
            raise RuntimeError("the registered recording did not appear in the catalog")
        return rec

    def unregister_recording(self, payload: dict[str, Any]) -> bool:
        return self.registry.remove(Path(str(payload["path"])))

    def thumbnail(self, path: str, frame: int, scale: float) -> bytes:
        recording = self.recording_path(path)
        if not recording.is_file():
            raise NotFound(f"no recording at {path}")
        try:
            # Thumbnails never exceed the frame: ``scale`` above 1 would build a gigapixel image in the server.
            return thumbnail_png(
                recording, frame, min(float(scale), 1.0), dataset_root=self.config.dataset_root, dataset=self.recording_dataset(recording)
            )
        except OSError as error:
            raise ValueError(f"{recording.name} cannot be read as a recording") from error

    # -------------------------------------------------------------------- viewer

    def is_run(self, name: str) -> bool:
        return name in self.viewer.catalog

    def run_or_workspace_payload(self, name: str) -> dict[str, Any]:
        if self.is_run(name):
            return self.viewer.run_payload(name)
        return self.view(name).payload(self.catalog_entries())

    def frame_payload(self, name: str, frame: int, threshold: float | None, raw: bool, detail: str) -> dict[str, Any]:
        if self.is_run(name):
            return self.viewer.frame_payload(name, frame, threshold, raw, detail)
        return self.view(name).frame(frame, self.viewer.segmenters, threshold, self.device, raw=raw, detail=detail)

    def pose_payload(self, name: str, frame: int) -> dict[str, Any]:
        if self.is_run(name):
            return self.viewer.pose_payload(name, frame)
        return self.view(name).pose(frame)

    def starts_payload(self, name: str, frame: int, threshold: float | None) -> dict[str, Any]:
        if self.is_run(name):
            return self.viewer.starts_payload(name, frame, threshold)
        return self.view(name).starts(frame, self.viewer.segmenters, threshold, self.device)

    def state_payload(self, rescan: bool = False) -> dict[str, Any]:
        added = self.viewer.rescan() if rescan else 0
        return {
            **self.viewer.state(),
            "added": added,
            "workspaces": self.workspace_rows(),
            "workspaces_root": str(self.config.workspaces_root),
            "recording_roots": [str(p) for p in self.config.recording_roots],
            "poses_root": str(self.config.poses_root),
            "gpus": list(self.config.gpus),
            "jobs_running": len(self.runner.list("running")),
            "jobs_queued": len(self.runner.list("queued")),
        }

    # --------------------------------------------------------------------- notes

    def add_note(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """A review note on a run (as the viewer does) or on a workspace (``workspace`` names one explicitly)."""

        name = str(payload.get("workspace") or payload.get("run") or "")
        if not payload.get("workspace") and self.is_run(name):
            return self.viewer.add_note(payload)
        view = self.view(name)
        note = {
            "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "run": name,
            "workspace": name,
            "recording": view.workspace.recording.stem,
            "frame_index": int(payload["frame_index"]),
            "tags": [str(t) for t in payload.get("tags", [])],
            "comment": str(payload.get("comment", "")),
        }
        return self.viewer.notes.add(note)
