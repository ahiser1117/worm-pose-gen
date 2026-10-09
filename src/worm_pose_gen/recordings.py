"""The HDF5 recordings: catalog, frame thumbnails, and frames read and flat-fielded.

The Recordings screen and Labeling's New queue need, for every ``.h5`` file
of a setup (under its roots or registered to it), its frame count and image
size and whether it can actually be read (some recordings hold chunks
compressed with an HDF5 filter plugin that is not installed, and a few
files are not HDF5 at all).  Opening every file to read its shape is slow
over network storage, so the per-file facts are cached in one JSON index
keyed by path; an entry is reused while the file's size and modification
time are unchanged and refreshed otherwise.  Thumbnails are flat-fielded
when the field cache holds the recording's field and are otherwise the raw
frame.

``RecordingSource`` reads one recording's frames read-only and flat-fields
them with a per-recording correction fitted once from frames spread over
the recording and cached on disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterable

import h5py
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .flat_field import FlatField, apply_flat_field, estimate_flat_field
from .segmentation_dataset import DEFAULT_DATASET_ROOT
from .videos import VIDEO_SUFFIXES


DATASET_PATH = "/img_nir"
CACHE_VERSION = 1
FLAT_FIELD_SAMPLE_COUNT = 64
HDF5_SUFFIXES = (".h5", ".hdf5")


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
    # The HDF5 dataset holding the frames.
    dataset: str = DATASET_PATH

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_directory(path: Path, *, all_files: bool = False) -> dict[str, Any]:
    """Directories, HDF5 files (kind "h5") and convertible videos (kind "video") directly under ``path``, for the file explorer.

    Hidden entries are skipped; unreadable subdirectories are listed but
    flagged.  With ``all_files`` every other regular file is listed too
    (kind "file"), for videos stored under other names.
    """

    directory = Path(path).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"{directory} does not exist")
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    directory = directory.resolve()
    entries: list[dict[str, Any]] = []
    try:
        children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as error:
        raise PermissionError(f"{directory}: permission denied") from error
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir():
                entries.append({"name": child.name, "path": str(child), "kind": "dir", "size_bytes": None, "modified_at": _iso_utc(child.stat().st_mtime), "readable": os.access(child, os.R_OK | os.X_OK)})
            elif child.is_file():
                kind = _file_kind(child.suffix.lower()) or ("file" if all_files else None)
                if kind is None:
                    continue
                stat = child.stat()
                entries.append({"name": child.name, "path": str(child), "kind": kind, "size_bytes": int(stat.st_size), "modified_at": _iso_utc(stat.st_mtime), "readable": os.access(child, os.R_OK)})
        except OSError:
            continue
    parent = None if directory.parent == directory else str(directory.parent)
    return {"path": str(directory), "parent": parent, "entries": entries, "all_files": all_files, "suffixes": [*HDF5_SUFFIXES, *VIDEO_SUFFIXES]}


def _file_kind(suffix: str) -> str | None:
    return "h5" if suffix in HDF5_SUFFIXES else "video" if suffix in VIDEO_SUFFIXES else None


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


def probe_frames(frame_count: int) -> tuple[int, ...]:
    """First, middle and last frame: one recording reads its first chunk but not the later ones."""

    if frame_count <= 0:
        return ()
    return tuple(sorted({0, frame_count // 2, frame_count - 1}))


def probe_recording(path: Path, dataset: str = DATASET_PATH) -> dict[str, Any]:
    """Shape and readability of one file: open it, read the dataset shape, then a few frames.

    Reading frames is what tells an installed-filter recording from one whose
    chunks need a plugin we do not have; the shape alone reads fine for both.
    """

    facts: dict[str, Any] = {"frames": None, "height": None, "width": None, "readable": False, "error": None}
    try:
        with h5py.File(path, "r") as handle:
            if dataset not in handle:
                raise KeyError(f"no dataset {dataset}")
            data = handle[dataset]
            if data.ndim != 3:
                raise ValueError(f"expected a [T,H,W] dataset, got shape {tuple(data.shape)}")
            facts["frames"], facts["height"], facts["width"] = (int(v) for v in data.shape)
            for frame in probe_frames(facts["frames"]):
                try:
                    np.asarray(data[frame])
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

    def facts(self, path: Path, dataset: str = DATASET_PATH) -> dict[str, Any]:
        stamp = _file_stamp(path)
        key = str(path) if dataset == DATASET_PATH else f"{path}#{dataset}"
        entry = self.entries.get(key)
        if entry is None or entry.get("mtime_ns") != stamp["mtime_ns"] or entry.get("size") != stamp["size"]:
            entry = {**stamp, **probe_recording(path, dataset)}
            self.entries[key] = entry
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
    """Every file matching ``pattern`` under the roots that exist, resolved, sorted by name then path.

    Resolving makes a file reached through two roots (one a symlink to the other) one recording.
    """

    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            if root.match(pattern):
                found.add(root.resolve())
            continue
        if not root.is_dir():
            continue
        found.update(p.resolve() for p in root.rglob(pattern) if p.is_file())
    return sorted(found, key=lambda p: (p.stem, str(p)))


def list_recordings(roots: Iterable[Path], *, pattern: str = "*.h5", cache: Path | None = None, dataset: str = DATASET_PATH) -> list[RecordingInfo]:
    """Catalog the recordings under ``roots`` (directories, or recording files themselves), reusing ``cache`` for unchanged files.

    ``dataset`` is the HDF5 dataset holding the frames (a setup's video dataset).
    """

    index = RecordingIndex(cache)
    infos = []
    for path in find_recordings(roots, pattern):
        try:
            entry = index.facts(path, dataset)
            stamp = _file_stamp(path)
        except OSError as error:
            # Vanished or unstat-able between listing and probing: report it rather than drop it.
            entry = {"frames": None, "height": None, "width": None, "readable": False, "error": f"{type(error).__name__}: {error}"}
            stamp = {"mtime_ns": 0, "size": 0}
        infos.append(
            RecordingInfo(
                name=path.stem, path=str(path), frames=entry["frames"], height=entry["height"], width=entry["width"],
                size_bytes=int(stamp["size"]), modified_at=_iso_utc(stamp["mtime_ns"] / 1e9), readable=bool(entry["readable"]),
                error=entry["error"], dataset=dataset,
            )
        )
    index.save()
    return infos


def _read_frame(path: Path, frame: int, dataset: str = DATASET_PATH) -> NDArray[np.uint8]:
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"no dataset {dataset}")
        data = handle[dataset]
        if not 0 <= frame < data.shape[0]:
            raise IndexError(f"frame {frame} out of range for {data.shape[0]} frames")
        values = np.asarray(data[int(frame)])
        if values.dtype != np.uint8:
            # 16-bit cameras: scale by the frame's own range so the thumbnail is visible.
            low, high = float(values.min()), float(values.max())
            values = ((values - low) / max(high - low, 1.0) * 255.0).astype(np.uint8)
        return values


def thumbnail_frame(path: Path, frame: int, dataset_root: Path = DEFAULT_DATASET_ROOT, dataset: str = DATASET_PATH) -> NDArray[np.uint8]:
    """The frame flat-fielded through the cached field when one exists, else raw."""

    path = Path(path)
    field_cache = Path(dataset_root) / "flat_fields"
    if (field_cache / f"{path.stem}.npz").exists():
        source = RecordingSource(path, field_cache, dataset=dataset)
        try:
            _, corrected = source.corrected(int(frame))
        finally:
            source.close()
        return corrected
    return _read_frame(path, int(frame), dataset)


def thumbnail_png(
    path: Path, frame: int, scale: float = 0.25, dataset_root: Path = DEFAULT_DATASET_ROOT, dataset: str = DATASET_PATH
) -> bytes:
    """PNG bytes of one frame scaled by ``scale`` (at least 1x1 pixel)."""

    if not scale > 0:
        raise ValueError("scale must be positive")
    image = Image.fromarray(thumbnail_frame(path, frame, dataset_root, dataset), mode="L")
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    if size != image.size:
        image = image.resize(size, Image.Resampling.BOX if scale < 1 else Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


class RecordingSource:
    """One read-only recording with a lazily fitted, disk-cached flat field."""

    def __init__(self, path: Path, cache_dir: Path, dataset: str = DATASET_PATH) -> None:
        self.path = Path(path)
        self.name = self.path.stem
        self.cache_dir = Path(cache_dir)
        self.dataset_path = dataset
        self._lock = threading.Lock()
        self._field_lock = threading.Lock()
        self._handle: h5py.File | None = None
        self._field: FlatField | None = None
        with h5py.File(self.path, "r") as handle:
            if dataset not in handle:
                raise KeyError(f"{self.path}: no dataset {dataset}")
            dataset = handle[dataset]
            if dataset.ndim != 3:
                raise ValueError(f"{self.path}: expected a [T,H,W] dataset")
            self.frame_count = int(dataset.shape[0])
            self.shape = (int(dataset.shape[1]), int(dataset.shape[2]))

    def _dataset(self) -> h5py.Dataset:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle[self.dataset_path]

    def read(self, frame_index: int) -> NDArray[np.uint8]:
        if not 0 <= frame_index < self.frame_count:
            raise IndexError("frame index out of range")
        with self._lock:
            return np.asarray(self._dataset()[int(frame_index)], dtype=np.uint8)

    def flat_field(self) -> FlatField:
        # Light/full previews and label proposals can request the same field concurrently.
        with self._field_lock:
            return self._prepare_flat_field()

    def _prepare_flat_field(self) -> FlatField:
        if self._field is not None:
            return self._field
        cache = self.cache_dir / f"{self.name}.npz"
        if cache.exists():
            with np.load(cache) as archive:
                self._field = FlatField(
                    illumination=np.asarray(archive["illumination"], dtype=np.float64),
                    dark_level=float(archive["dark_level"]),
                    reference_level=float(archive["reference_level"]),
                    gain=np.asarray(archive["gain"], dtype=np.float64),
                )
            return self._field
        indices = np.linspace(0, self.frame_count - 1, min(FLAT_FIELD_SAMPLE_COUNT, self.frame_count), dtype=np.int64)
        frames = []
        with self._lock:
            dataset = self._dataset()
            for i in indices:
                try:
                    frames.append(np.asarray(dataset[int(i)], dtype=np.uint8))
                except OSError:
                    # Some recordings hold chunks compressed with an HDF5 filter plugin that is not installed.
                    continue
        if len(frames) < min(8, len(indices)):
            raise OSError(f"only {len(frames)} of {len(indices)} calibration frames of {self.name} are readable")
        calibration = np.stack(frames)
        field = estimate_flat_field(
            calibration, temporal_quantile=0.8, spatial_radius=31, smoothing_passes=2, min_gain=0.5, max_gain=2.5
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Every writer owns its temporary file, including separate server/job processes.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, prefix=f"{cache.name}.", suffix=".partial", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle, illumination=field.illumination, dark_level=field.dark_level,
                    reference_level=field.reference_level, gain=field.gain,
                )
            temporary.replace(cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._field = field
        return field

    def corrected(self, frame_index: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        raw = self.read(frame_index)
        corrected = apply_flat_field(raw, self.flat_field(), clip=(0.0, 255.0))
        return raw, np.clip(np.rint(corrected), 0, 255).astype(np.uint8)

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
