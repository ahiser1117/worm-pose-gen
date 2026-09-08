"""Configuration of the pose app: roots, hardware, and where its own files go.

One dataclass so ``create_app`` can be called from tests with temporary
directories and from ``main`` with the command line; every path has the
project's default so a bare ``worm-pose-app`` serves the lab's recordings,
runs and workspaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..pose_viewer import DEFAULT_CHECKPOINT, DEFAULT_NOTES, DEFAULT_RUNS_ROOT
from ..recordings import DEFAULT_PRIOR_CACHE, DEFAULT_RECORDING_ROOTS
from ..segmentation_dataset import DEFAULT_DATASET_ROOT
from ..workspace import DEFAULT_WORKSPACES_ROOT


@dataclass
class AppConfig:
    """Everything the application needs to start."""

    host: str = "127.0.0.1"
    port: int = 8768
    workspaces_root: Path = DEFAULT_WORKSPACES_ROOT
    recording_roots: tuple[Path, ...] = tuple(DEFAULT_RECORDING_ROOTS)
    poses_root: Path = DEFAULT_RUNS_ROOT
    dataset_root: Path = DEFAULT_DATASET_ROOT
    checkpoint: Path | None = DEFAULT_CHECKPOINT
    prior_cache: Path | None = DEFAULT_PRIOR_CACHE
    notes: Path = DEFAULT_NOTES
    gpus: tuple[int, ...] = (0,)
    device: str | None = None
    jobs_root: Path | None = None
    recordings_cache: Path | None = None
    max_concurrent: int | None = None
    job_interval: float = 1.0
    extra_runs: tuple[Path, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.workspaces_root = Path(self.workspaces_root)
        self.recording_roots = tuple(Path(p) for p in self.recording_roots)
        self.poses_root = Path(self.poses_root)
        self.dataset_root = Path(self.dataset_root)
        self.checkpoint = None if self.checkpoint is None else Path(self.checkpoint)
        self.prior_cache = None if self.prior_cache is None else Path(self.prior_cache)
        self.notes = Path(self.notes)
        self.gpus = tuple(int(g) for g in self.gpus)
        self.jobs_root = self.workspaces_root if self.jobs_root is None else Path(self.jobs_root)
        self.recordings_cache = self.workspaces_root / "recordings_index.json" if self.recordings_cache is None else Path(self.recordings_cache)
        self.extra_runs = tuple(Path(p) for p in self.extra_runs)

    @property
    def viewer_device(self) -> str | None:
        """The device the server's own segmenter runs on: the first job GPU unless told otherwise."""

        if self.device is not None:
            return self.device
        if self.gpus:
            import torch

            if torch.cuda.is_available() and self.gpus[0] < torch.cuda.device_count():
                return f"cuda:{self.gpus[0]}"
        return None
