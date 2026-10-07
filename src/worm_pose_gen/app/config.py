"""Configuration of the pose app: roots, hardware, and where its own files go.

One dataclass so ``create_app`` can be called from tests with temporary
directories and from ``main`` with the command line.  Recordings and models
come from the libraries (a setup's recording roots and default models), so
the configuration only says where the libraries, the workspaces and the
flat-field cache are, and which hardware jobs and the server's own models
use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..library.roots import default_lab_root, default_personal_root
from ..segmentation_dataset import DEFAULT_DATASET_ROOT
from ..workspace import DEFAULT_WORKSPACES_ROOT


@dataclass
class AppConfig:
    """Everything the application needs to start."""

    host: str = "127.0.0.1"
    port: int = 8768
    workspaces_root: Path = DEFAULT_WORKSPACES_ROOT
    # Where the per-recording flat fields are cached (``<dataset_root>/flat_fields``).
    dataset_root: Path = DEFAULT_DATASET_ROOT
    gpus: tuple[int, ...] = (0,)
    device: str | None = None
    jobs_root: Path | None = None
    recordings_cache: Path | None = None
    max_concurrent: int | None = None
    job_interval: float = 1.0
    # The lab library (read-only; None: the host's, from library.LAB_LIBRARY_BY_HOST) and the personal one (None: the host default).
    lab_library: Path | None = None
    library: Path | None = None
    # Developer mode: the UI also shows the research diagnostics (docs/APP_SIMPLIFICATION.md).
    dev: bool = False

    def __post_init__(self) -> None:
        self.workspaces_root = Path(self.workspaces_root)
        self.dataset_root = Path(self.dataset_root)
        self.gpus = tuple(int(g) for g in self.gpus)
        self.jobs_root = self.workspaces_root if self.jobs_root is None else Path(self.jobs_root)
        self.recordings_cache = self.workspaces_root / "recordings_index.json" if self.recordings_cache is None else Path(self.recordings_cache)
        self.lab_library = default_lab_root() if self.lab_library is None else Path(self.lab_library)
        self.library = default_personal_root() if self.library is None else Path(self.library)

    @property
    def server_device(self) -> str | None:
        """The device of the server's own models (frame layers, Labeling proposals): the first job GPU unless told otherwise."""

        if self.device is not None:
            return self.device
        if self.gpus:
            import torch

            if torch.cuda.is_available() and self.gpus[0] < torch.cuda.device_count():
                return f"cuda:{self.gpus[0]}"
        return None
