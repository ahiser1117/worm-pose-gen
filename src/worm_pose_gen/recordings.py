"""Catalog of the HDF5 recordings under the configured roots.

The app's recording browser needs, for every ``.h5`` file under the data
roots, its frame count and image size, whether it can actually be read (some
recordings hold chunks compressed with an HDF5 filter plugin that is not
installed, and a few files are not HDF5 at all), whether a recording prior is
cached for it, and which pose runs and workspaces already refer to it.

Opening every file to read its shape is slow over network storage, so the
per-file facts are cached in one JSON index keyed by path; an entry is reused
while the file's size and modification time are unchanged and refreshed
otherwise.  Runs, workspaces and cached priors are cheap directory scans and
are recomputed on every call so the catalog never lags behind them.
Thumbnails are flat-fielded when the field cache holds the recording's field
and are otherwise the raw frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import h5py
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .segmentation_dataset import DEFAULT_DATASET_ROOT


DEFAULT_RECORDING_ROOTS = (Path("/store1/shared/all_data_raw/prj_aversion"),)
DEFAULT_POSES_ROOT = Path("/temp_data4/alex/external_artifacts/poses")
DEFAULT_WORKSPACES_ROOT = Path("/temp_data4/alex/external_artifacts/workspaces")
DEFAULT_PRIOR_CACHE = Path("/temp_data4/alex/external_artifacts/recording_priors")
DATASET_PATH = "/img_nir"
CACHE_VERSION = 1


@dataclass
class RecordingInfo:
    """One recording file as the browser shows it."""

    name: str
    path: str
    frames: int | None
    height: int | None
    width: int | None
    size_bytes: int
    modified_at: str
    readable: bool
    error: str | None
    prior_cached: bool
    runs: list[str]
    workspaces: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


def probe_frames(frame_count: int) -> tuple[int, ...]:
    """First, middle and last frame: one recording reads its first chunk but not the later ones."""

    if frame_count <= 0:
        return ()
    return tuple(sorted({0, frame_count // 2, frame_count - 1}))


def probe_recording(path: Path) -> dict[str, Any]:
    """Shape and readability of one file: open it, read the dataset shape, then a few frames.

    Reading frames is what tells an installed-filter recording from one whose
    chunks need a plugin we do not have; the shape alone reads fine for both.
    """

    facts: dict[str, Any] = {"frames": None, "height": None, "width": None, "readable": False, "error": None}
    try:
        with h5py.File(path, "r") as handle:
            if DATASET_PATH not in handle:
                raise KeyError(f"no dataset {DATASET_PATH}")
            dataset = handle[DATASET_PATH]
            if dataset.ndim != 3:
                raise ValueError(f"expected a [T,H,W] dataset, got shape {tuple(dataset.shape)}")
            facts["frames"], facts["height"], facts["width"] = (int(v) for v in dataset.shape)
            for frame in probe_frames(facts["frames"]):
                try:
                    np.asarray(dataset[frame])
                except OSError as error:
                    raise OSError(f"frame {frame} not readable: {error}") from error
            facts["readable"] = True
    except Exception as error:  # noqa: BLE001 - any failure makes the file unreadable, and we want its message
        facts["error"] = f"{type(error).__name__}: {error}".strip()
    return facts


def _file_stamp(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


class RecordingIndex:
    """The JSON cache of per-file facts; ``cache is None`` keeps it in memory only."""

    def __init__(self, cache: Path | None) -> None:
        self.cache = None if cache is None else Path(cache)
        self.entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if self.cache is not None and self.cache.exists():
            try:
                data = json.loads(self.cache.read_text())
                if data.get("version") == CACHE_VERSION:
                    self.entries = dict(data.get("entries") or {})
            except (OSError, ValueError):
                self.entries = {}

    def facts(self, path: Path) -> dict[str, Any]:
        stamp = _file_stamp(path)
        entry = self.entries.get(str(path))
        if entry is None or entry.get("mtime_ns") != stamp["mtime_ns"] or entry.get("size") != stamp["size"]:
            entry = {**stamp, **probe_recording(path)}
            self.entries[str(path)] = entry
            self._dirty = True
        return entry

    def save(self) -> None:
        if self.cache is None or not self._dirty:
            return
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        # A unique temporary: concurrent listings (the app serves them from a
        # thread pool) must not replace or unlink each other's half-written file.
        with tempfile.NamedTemporaryFile("w", dir=self.cache.parent, prefix=self.cache.name + ".", suffix=".tmp", delete=False) as handle:
            json.dump({"version": CACHE_VERSION, "entries": self.entries}, handle, indent=1)
        os.replace(handle.name, self.cache)
        self._dirty = False


def find_recordings(roots: Iterable[Path], pattern: str = "*.h5") -> list[Path]:
    """Every file matching ``pattern`` under the roots that exist, sorted by name then path."""

    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            if root.match(pattern):
                found.add(root)
            continue
        if not root.is_dir():
            continue
        found.update(p for p in root.rglob(pattern) if p.is_file())
    return sorted(found, key=lambda p: (p.stem, str(p)))


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _references(root: Path | None, manifest: str) -> dict[str, list[str]]:
    """Directory names under ``root`` grouped by the recording their ``manifest`` names.

    Keyed both by the recording's path string and by its stem so a moved
    recording still finds its runs.
    """

    refs: dict[str, list[str]] = {}
    if root is None or not Path(root).is_dir():
        return refs
    for child in sorted(Path(root).iterdir()):
        data = _load_json(child / manifest) if child.is_dir() else None
        recording = None if data is None else data.get("recording")
        if not isinstance(recording, str) or not recording:
            continue
        for key in {recording, Path(recording).stem}:
            refs.setdefault(key, []).append(child.name)
    return refs


def _lookup(refs: dict[str, list[str]], path: Path) -> list[str]:
    names = list(refs.get(str(path), []))
    names.extend(n for n in refs.get(path.stem, []) if n not in names)
    return names


def prior_is_cached(path: Path, prior_cache: Path | None) -> bool:
    """A recording prior exists for the recording at any coefficient count."""

    if prior_cache is None or not Path(prior_cache).is_dir():
        return False
    return any(Path(prior_cache).glob(f"{path.stem}_k*.json"))


def list_recordings(
    roots: Iterable[Path] = DEFAULT_RECORDING_ROOTS,
    *,
    pattern: str = "*.h5",
    poses_root: Path | None = DEFAULT_POSES_ROOT,
    workspaces_root: Path | None = DEFAULT_WORKSPACES_ROOT,
    prior_cache: Path | None = DEFAULT_PRIOR_CACHE,
    cache: Path | None = None,
) -> list[RecordingInfo]:
    """Catalog the recordings under ``roots``, reusing ``cache`` for files whose mtime and size are unchanged."""

    index = RecordingIndex(cache)
    runs = _references(poses_root, "summary.json")
    workspaces = _references(workspaces_root, "workspace.json")
    infos = []
    for path in find_recordings(roots, pattern):
        try:
            entry = index.facts(path)
            stamp = _file_stamp(path)
        except OSError as error:
            # Vanished or unstat-able between listing and probing: report it rather than drop it.
            entry = {"frames": None, "height": None, "width": None, "readable": False, "error": f"{type(error).__name__}: {error}"}
            stamp = {"mtime_ns": 0, "size": 0}
        infos.append(
            RecordingInfo(
                name=path.stem,
                path=str(path),
                frames=entry["frames"],
                height=entry["height"],
                width=entry["width"],
                size_bytes=int(stamp["size"]),
                modified_at=_iso_utc(stamp["mtime_ns"] / 1e9),
                readable=bool(entry["readable"]),
                error=entry["error"],
                prior_cached=prior_is_cached(path, prior_cache),
                runs=_lookup(runs, path),
                workspaces=_lookup(workspaces, path),
            )
        )
    index.save()
    return infos


def _read_frame(path: Path, frame: int) -> NDArray[np.uint8]:
    with h5py.File(path, "r") as handle:
        dataset = handle[DATASET_PATH]
        if not 0 <= frame < dataset.shape[0]:
            raise IndexError(f"frame {frame} out of range for {dataset.shape[0]} frames")
        return np.asarray(dataset[int(frame)], dtype=np.uint8)


def thumbnail_frame(path: Path, frame: int, dataset_root: Path = DEFAULT_DATASET_ROOT) -> NDArray[np.uint8]:
    """The frame flat-fielded through the cached field when one exists, else raw."""

    path = Path(path)
    field_cache = Path(dataset_root) / "flat_fields"
    if (field_cache / f"{path.stem}.npz").exists():
        from .label_app import RecordingSource  # deferred: label_app imports torch

        source = RecordingSource(path, field_cache)
        try:
            _, corrected = source.corrected(int(frame))
        finally:
            source.close()
        return corrected
    return _read_frame(path, int(frame))


def thumbnail_png(path: Path, frame: int, scale: float = 0.25, dataset_root: Path = DEFAULT_DATASET_ROOT) -> bytes:
    """PNG bytes of one frame scaled by ``scale`` (at least 1x1 pixel)."""

    if not scale > 0:
        raise ValueError("scale must be positive")
    image = Image.fromarray(thumbnail_frame(path, frame, dataset_root), mode="L")
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    if size != image.size:
        image = image.resize(size, Image.Resampling.BOX if scale < 1 else Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
