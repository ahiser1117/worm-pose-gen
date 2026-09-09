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
    # The HDF5 dataset holding the frames and whether the file was added by hand
    # (through the file explorer) rather than found under a root.
    dataset: str = DATASET_PATH
    registered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VIDEO_DTYPES = ("uint8", "uint16")


def hdf5_datasets(path: Path) -> list[dict[str, Any]]:
    """Every dataset in an HDF5 file with shape and dtype; ``video`` marks 3-D unsigned-integer ones."""

    found: list[dict[str, Any]] = []

    def visit(name: str, node: Any) -> None:
        if isinstance(node, h5py.Dataset):
            shape = tuple(int(v) for v in node.shape)
            found.append({
                "name": "/" + name.lstrip("/"), "shape": list(shape), "dtype": str(node.dtype),
                "video": len(shape) == 3 and str(node.dtype) in VIDEO_DTYPES and shape[0] >= 1 and min(shape[1:]) >= 8,
            })

    with h5py.File(path, "r") as handle:
        handle.visititems(visit)
    # The conventional name first, then the video candidates, then the rest.
    found.sort(key=lambda d: (d["name"] != DATASET_PATH, not d["video"], d["name"]))
    return found


def default_video_dataset(datasets: list[dict[str, Any]]) -> str | None:
    """The dataset a new recording should use: ``/img_nir`` when present, else the single video candidate."""

    names = {d["name"] for d in datasets}
    if DATASET_PATH in names:
        return DATASET_PATH
    videos = [d["name"] for d in datasets if d["video"]]
    return videos[0] if len(videos) == 1 else None


def list_directory(path: Path, *, suffixes: tuple[str, ...] = (".h5", ".hdf5"), all_files: bool = False) -> dict[str, Any]:
    """Directories and HDF5 files directly under ``path``, for the file explorer.

    Hidden entries are skipped; unreadable subdirectories are listed but
    flagged.  With ``all_files`` every regular file is listed (kind "file"
    unless its suffix is an HDF5 one), for videos stored under other names.
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
            elif child.is_file() and (all_files or child.suffix.lower() in suffixes):
                stat = child.stat()
                kind = "h5" if child.suffix.lower() in suffixes else "file"
                entries.append({"name": child.name, "path": str(child), "kind": kind, "size_bytes": int(stat.st_size), "modified_at": _iso_utc(stat.st_mtime), "readable": os.access(child, os.R_OK)})
        except OSError:
            continue
    parent = None if directory.parent == directory else str(directory.parent)
    return {"path": str(directory), "parent": parent, "entries": entries, "all_files": all_files, "suffixes": list(suffixes)}


class RecordingRegistry:
    """Recordings added by hand, with the dataset to read; a JSON file under the workspaces root."""

    def __init__(self, path: Path | None) -> None:
        self.path = None if path is None else Path(path)
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            data = _load_json(self.path)
            if data and isinstance(data.get("recordings"), dict):
                self.entries = {str(k): dict(v) for k, v in data["recordings"].items()}

    def add(self, path: Path, dataset: str = DATASET_PATH) -> dict[str, Any]:
        entry = {"dataset": dataset, "added_at": _iso_utc(datetime.now(tz=timezone.utc).timestamp())}
        self.entries[str(Path(path).expanduser().resolve())] = entry
        self.save()
        return entry

    def remove(self, path: Path) -> bool:
        removed = self.entries.pop(str(Path(path).expanduser().resolve()), None) is not None
        if removed:
            self.save()
        return removed

    def dataset_of(self, path: Path) -> str | None:
        entry = self.entries.get(str(Path(path).expanduser().resolve()))
        return None if entry is None else str(entry.get("dataset") or DATASET_PATH)

    def paths(self) -> list[Path]:
        return [Path(p) for p in self.entries]

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp", delete=False) as handle:
            json.dump({"version": 1, "recordings": self.entries}, handle, indent=1)
        os.replace(handle.name, self.path)


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
    registry: RecordingRegistry | None = None,
) -> list[RecordingInfo]:
    """Catalog the recordings under ``roots`` plus the registered ones, reusing ``cache`` for unchanged files."""

    index = RecordingIndex(cache)
    runs = _references(poses_root, "summary.json")
    workspaces = _references(workspaces_root, "workspace.json")
    infos = []
    found = find_recordings(roots, pattern)
    registered: dict[Path, str] = {}
    if registry is not None:
        for path in registry.paths():
            registered[path] = registry.dataset_of(path) or DATASET_PATH
            if path not in found:
                found.append(path)
    found.sort(key=lambda p: (p.stem, str(p)))
    for path in found:
        dataset = registered.get(path, DATASET_PATH)
        try:
            entry = index.facts(path, dataset)
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
                dataset=dataset,
                registered=path in registered,
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
        from .label_app import RecordingSource  # deferred: label_app imports torch

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
