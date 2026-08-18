"""Versioned, streamed and atomic HDF5 pose output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from numpy.typing import ArrayLike


SCHEMA_VERSION = "1.0.0"
GROUP_PATH = "/worm_pose"


class IncompleteOutputError(ValueError):
    """Raised when a partial or not-complete output is encountered."""


@dataclass(frozen=True)
class SourceIdentity:
    configured_path: str
    resolved_path: str
    dataset_path: str
    size_bytes: int
    mtime_ns: int

    @classmethod
    def from_path(
        cls, source_path: str | os.PathLike[str], dataset_path: str
    ) -> SourceIdentity:
        configured = Path(source_path)
        resolved = configured.resolve(strict=True)
        stat = resolved.stat()
        return cls(
            configured_path=str(configured),
            resolved_path=str(resolved),
            dataset_path=dataset_path,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )


@dataclass(frozen=True)
class OutputProvenance:
    source: SourceIdentity
    checkpoint_sha256: str
    config_sha256: str
    git_commit: str
    package_versions: Mapping[str, str]
    geometry_convention: str
    image_height: int | None = None
    image_width: int | None = None


def sha256_file(path: str | os.PathLike[str], block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


_FRAME_DATASETS: dict[str, tuple[np.dtype, str]] = {
    "centerline_xy": (np.dtype("float32"), "original_image_pixel"),
    "tangent_angle": (np.dtype("float32"), "radian"),
    "curvature": (np.dtype("float32"), "radian_per_original_image_pixel"),
    "in_fov_mask": (np.dtype("bool"), "boolean"),
    "image_support_probability": (np.dtype("float32"), "probability"),
    "angle_uncertainty": (np.dtype("float32"), "radian"),
    "head_tail_probability": (np.dtype("float32"), "probability"),
    "quality_score": (np.dtype("float32"), "unitless"),
    "frame_index": (np.dtype("int64"), "source_frame_index"),
    "timestamp": (np.dtype("float64"), "second"),
}


class PoseHDF5Writer:
    """Append complete prediction batches, then validate and publish atomically.

    The predictable ``<output>.partial`` path makes interrupted runs visible and
    intentionally non-resumable.  It is never silently removed or reused.
    """

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        *,
        body_points: int,
        provenance: OutputProvenance,
        chunk_frames: int = 64,
        compression_level: int = 4,
        overwrite: bool = False,
    ) -> None:
        if body_points < 2:
            raise ValueError("body_points must be at least 2")
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        if not 0 <= compression_level <= 9:
            raise ValueError("compression_level must be in [0, 9]")
        self.output_path = Path(output_path)
        self.partial_path = self.output_path.with_name(self.output_path.name + ".partial")
        self.body_points = body_points
        self.provenance = provenance
        self.chunk_frames = chunk_frames
        self.compression_level = compression_level
        self.overwrite = overwrite
        self._file: h5py.File | None = None
        self._group: h5py.Group | None = None
        self._published = False

    def __enter__(self) -> PoseHDF5Writer:
        self.open()
        return self

    def open(self) -> None:
        if self._file is not None:
            raise RuntimeError("writer is already open")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(f"completed output already exists: {self.output_path}")
        if self.partial_path.exists():
            raise IncompleteOutputError(
                f"incomplete output exists and resume is unsupported: {self.partial_path}"
            )
        self._file = h5py.File(self.partial_path, "x")
        group = self._file.create_group(GROUP_PATH)
        self._group = group
        group.attrs["schema_version"] = SCHEMA_VERSION
        group.attrs["complete"] = False
        group.attrs["body_points"] = self.body_points
        self._write_provenance(group)
        self._create_datasets(group)
        self._file.flush()

    def _write_provenance(self, group: h5py.Group) -> None:
        metadata = group.create_group("provenance")
        source = asdict(self.provenance.source)
        for name, value in source.items():
            metadata.attrs[f"source_{name}"] = value
        metadata.attrs["checkpoint_sha256"] = self.provenance.checkpoint_sha256
        metadata.attrs["config_sha256"] = self.provenance.config_sha256
        metadata.attrs["git_commit"] = self.provenance.git_commit
        metadata.attrs["package_versions_json"] = json.dumps(
            dict(self.provenance.package_versions), sort_keys=True, separators=(",", ":")
        )
        metadata.attrs["geometry_convention"] = self.provenance.geometry_convention
        if self.provenance.image_height is not None:
            metadata.attrs["image_height"] = self.provenance.image_height
        if self.provenance.image_width is not None:
            metadata.attrs["image_width"] = self.provenance.image_width

    def _create_datasets(self, group: h5py.Group) -> None:
        body = self.body_points
        shapes = {
            "centerline_xy": ((0, body, 2), (None, body, 2)),
            "tangent_angle": ((0, body), (None, body)),
            "curvature": ((0, body), (None, body)),
            "in_fov_mask": ((0, body), (None, body)),
            "image_support_probability": ((0, body), (None, body)),
            "angle_uncertainty": ((0, body), (None, body)),
            "head_tail_probability": ((0,), (None,)),
            "quality_score": ((0,), (None,)),
            "frame_index": ((0,), (None,)),
            "timestamp": ((0,), (None,)),
        }
        for name, (shape, maxshape) in shapes.items():
            tail = shape[1:]
            chunks = (self.chunk_frames, *tail)
            dtype, units = _FRAME_DATASETS[name]
            dataset = group.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                chunks=chunks,
                compression="gzip",
                compression_opts=self.compression_level,
                shuffle=True,
                dtype=dtype,
            )
            dataset.attrs["units"] = units
            dataset.attrs["missing_value"] = (
                "NaN only when all timestamps are unavailable"
                if name == "timestamp"
                else "none; missing/non-finite frames are rejected"
            )
            dataset.attrs["axis_order"] = (
                "frame,body,xy" if name == "centerline_xy" else
                "frame,body" if len(shape) == 2 else "frame"
            )

    @property
    def frame_count(self) -> int:
        self._require_open()
        assert self._group is not None
        return int(self._group["frame_index"].shape[0])

    def append(self, **values: ArrayLike) -> None:
        """Append one batch; every core dataset must be present."""

        self._require_open()
        missing = set(_FRAME_DATASETS) - set(values)
        extra = set(values) - set(_FRAME_DATASETS)
        if missing or extra:
            raise ValueError(f"dataset keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        arrays = {name: np.asarray(values[name]) for name in _FRAME_DATASETS}
        batch_size = self._validate_batch(arrays)
        if batch_size == 0:
            return
        assert self._group is not None
        start = self.frame_count
        stop = start + batch_size
        for name, array in arrays.items():
            dataset = self._group[name]
            dataset.resize(stop, axis=0)
            dataset[start:stop] = array.astype(dataset.dtype, copy=False)
        assert self._file is not None
        self._file.flush()

    def _validate_batch(self, arrays: Mapping[str, np.ndarray]) -> int:
        centerline = arrays["centerline_xy"]
        if centerline.ndim != 3:
            raise ValueError("centerline_xy must have shape [frames, body, 2]")
        batch = centerline.shape[0]
        expected = {
            "centerline_xy": (batch, self.body_points, 2),
            "tangent_angle": (batch, self.body_points),
            "curvature": (batch, self.body_points),
            "in_fov_mask": (batch, self.body_points),
            "image_support_probability": (batch, self.body_points),
            "angle_uncertainty": (batch, self.body_points),
            "head_tail_probability": (batch,),
            "quality_score": (batch,),
            "frame_index": (batch,),
            "timestamp": (batch,),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {arrays[name].shape}")
        for name in (
            "centerline_xy", "tangent_angle", "curvature",
            "image_support_probability", "angle_uncertainty",
            "head_tail_probability", "quality_score",
        ):
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"{name} contains a missing or non-finite value")
        self._validate_ranges(arrays)
        self._validate_mapping(arrays["frame_index"], arrays["timestamp"])
        self._validate_fov(arrays)
        return batch

    @staticmethod
    def _validate_ranges(arrays: Mapping[str, np.ndarray]) -> None:
        for name in ("image_support_probability", "head_tail_probability"):
            if np.any((arrays[name] < 0) | (arrays[name] > 1)):
                raise ValueError(f"{name} must lie in [0, 1]")
        if np.any(arrays["head_tail_probability"] < 0.5):
            raise ValueError("canonical head_tail_probability must lie in [0.5, 1]")
        if np.any(arrays["angle_uncertainty"] < 0):
            raise ValueError("angle_uncertainty must be non-negative")
        angles = arrays["tangent_angle"]
        if np.any((angles < -np.pi) | (angles >= np.pi)):
            raise ValueError("tangent_angle must be wrapped to [-pi, pi)")

    def _validate_mapping(self, indices: np.ndarray, timestamps: np.ndarray) -> None:
        if not np.issubdtype(indices.dtype, np.integer):
            if not np.all(np.isfinite(indices)) or not np.all(indices == np.floor(indices)):
                raise TypeError("frame_index must contain integers")
        current_indices = np.asarray(indices, dtype=np.int64)
        assert self._group is not None
        if self.frame_count and len(current_indices):
            previous = int(self._group["frame_index"][-1])
            if current_indices[0] <= previous:
                raise ValueError("frame_index must be strictly increasing across batches")
        if len(current_indices) > 1 and np.any(np.diff(current_indices) <= 0):
            raise ValueError("frame_index must be strictly increasing")
        finite = np.isfinite(timestamps)
        if np.any(finite) and not np.all(finite):
            raise ValueError("timestamps must be all finite or all unavailable (NaN)")
        if self.frame_count and len(timestamps):
            previous_timestamp = float(self._group["timestamp"][-1])
            if bool(np.isfinite(previous_timestamp)) != bool(np.all(finite)):
                raise ValueError("timestamp availability cannot change across batches")
        if np.all(finite) and len(timestamps):
            if self.frame_count:
                previous_timestamp = float(self._group["timestamp"][-1])
                if np.isfinite(previous_timestamp) and timestamps[0] <= previous_timestamp:
                    raise ValueError("finite timestamps must be strictly increasing across batches")
            if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
                raise ValueError("finite timestamps must be strictly increasing")

    def _validate_fov(self, arrays: Mapping[str, np.ndarray]) -> None:
        height = self.provenance.image_height
        width = self.provenance.image_width
        if (height is None) != (width is None):
            raise ValueError("image_height and image_width must be supplied together")
        if height is None or width is None:
            return
        xy = arrays["centerline_xy"]
        expected = (
            (xy[..., 0] >= 0) & (xy[..., 0] < width)
            & (xy[..., 1] >= 0) & (xy[..., 1] < height)
        )
        if not np.array_equal(arrays["in_fov_mask"].astype(bool), expected):
            raise ValueError("in_fov_mask disagrees with half-open image bounds")

    def finish(self) -> Path:
        """Validate, mark complete, close, and atomically publish the file."""

        self._require_open()
        assert self._file is not None and self._group is not None
        if self.frame_count == 0:
            raise ValueError("refusing to publish an output with no frames")
        self._validate_stored()
        self._group.attrs["complete"] = True
        self._group.attrs["frame_count"] = self.frame_count
        self._file.flush()
        self._file.close()
        self._file = None
        self._group = None
        try:
            if self.output_path.exists() and not self.overwrite:
                raise FileExistsError(f"output appeared before publish: {self.output_path}")
            os.replace(self.partial_path, self.output_path)
        except BaseException:
            # The complete partial remains available for diagnosis.
            raise
        self._published = True
        validate_output(self.output_path)
        return self.output_path

    def _validate_stored(self) -> None:
        assert self._group is not None
        _validate_group_contents(self._group, require_complete=False)

    def abort(self) -> None:
        """Close without deleting the partial, preserving evidence of interruption."""

        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self._group = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None and not self._published:
            try:
                self.finish()
            except BaseException:
                self.abort()
                raise
        elif not self._published:
            self.abort()

    def _require_open(self) -> None:
        if self._file is None or self._group is None:
            raise RuntimeError("writer is not open")


def validate_output(path: str | os.PathLike[str], *, require_complete: bool = True) -> int:
    """Validate structural completion and return the stored frame count."""

    output = Path(path)
    if output.name.endswith(".partial") and require_complete:
        raise IncompleteOutputError("partial output paths are never accepted as complete")
    with h5py.File(output, "r") as handle:
        if GROUP_PATH not in handle:
            raise ValueError(f"missing group {GROUP_PATH}")
        group = handle[GROUP_PATH]
        if group.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported or missing schema_version")
        if require_complete and not bool(group.attrs.get("complete", False)):
            raise IncompleteOutputError("output has no completion marker")
        return _validate_group_contents(group, require_complete=require_complete)


def _validate_group_contents(group: h5py.Group, *, require_complete: bool) -> int:
    """Validate a group in frame chunks so validation memory is bounded."""

    missing = set(_FRAME_DATASETS) - set(group.keys())
    if missing:
        raise ValueError(f"missing output datasets: {sorted(missing)}")
    provenance_names = {
        "source_configured_path", "source_resolved_path", "source_dataset_path",
        "source_size_bytes", "source_mtime_ns", "checkpoint_sha256",
        "config_sha256", "git_commit", "package_versions_json",
        "geometry_convention",
    }
    if "provenance" not in group:
        raise ValueError("missing output provenance")
    provenance = group["provenance"].attrs
    missing_provenance = provenance_names - set(provenance.keys())
    if missing_provenance:
        raise ValueError(f"missing provenance attributes: {sorted(missing_provenance)}")
    counts = {name: group[name].shape[0] for name in _FRAME_DATASETS}
    if len(set(counts.values())) != 1:
        raise ValueError(f"output dataset frame counts differ: {counts}")
    count = int(next(iter(counts.values())))
    if require_complete and int(group.attrs.get("frame_count", -1)) != count:
        raise ValueError("completion frame_count does not match datasets")
    body = int(group.attrs.get("body_points", -1))
    expected_shapes = {
        "centerline_xy": (count, body, 2),
        "tangent_angle": (count, body),
        "curvature": (count, body),
        "in_fov_mask": (count, body),
        "image_support_probability": (count, body),
        "angle_uncertainty": (count, body),
        "head_tail_probability": (count,), "quality_score": (count,),
        "frame_index": (count,), "timestamp": (count,),
    }
    for name, expected_shape in expected_shapes.items():
        dataset = group[name]
        if dataset.shape != expected_shape:
            raise ValueError(f"{name} has shape {dataset.shape}, expected {expected_shape}")
        if np.dtype(dataset.dtype) != _FRAME_DATASETS[name][0]:
            raise TypeError(f"{name} has dtype {dataset.dtype}, expected {_FRAME_DATASETS[name][0]}")
        if dataset.chunks is None or dataset.compression != "gzip":
            raise ValueError(f"{name} is not chunked with gzip compression")

    chunk_frames = max(1, int(group["frame_index"].chunks[0]))
    previous_index: int | None = None
    previous_timestamp: float | None = None
    timestamp_mode: bool | None = None  # True means finite, False unavailable.
    height = provenance.get("image_height")
    width = provenance.get("image_width")
    if (height is None) != (width is None):
        raise ValueError("provenance image dimensions must be supplied together")
    for start in range(0, count, chunk_frames):
        stop = min(count, start + chunk_frames)
        arrays = {name: np.asarray(group[name][start:stop]) for name in _FRAME_DATASETS}
        for name in (
            "centerline_xy", "tangent_angle", "curvature",
            "image_support_probability", "angle_uncertainty",
            "head_tail_probability", "quality_score",
        ):
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"stored {name} contains missing/non-finite values")
        for name in ("image_support_probability", "head_tail_probability"):
            if np.any((arrays[name] < 0) | (arrays[name] > 1)):
                raise ValueError(f"stored {name} lies outside [0, 1]")
        if np.any(arrays["head_tail_probability"] < 0.5):
            raise ValueError("stored head_tail_probability lies outside [0.5, 1]")
        if np.any(arrays["angle_uncertainty"] < 0):
            raise ValueError("stored angle_uncertainty is negative")
        angles = arrays["tangent_angle"]
        if np.any((angles < -np.pi) | (angles >= np.pi)):
            raise ValueError("stored tangent_angle is outside [-pi, pi)")
        indices = arrays["frame_index"]
        if len(indices):
            if previous_index is not None and int(indices[0]) <= previous_index:
                raise ValueError("stored frame_index is not strictly increasing")
            if len(indices) > 1 and np.any(np.diff(indices) <= 0):
                raise ValueError("stored frame_index is not strictly increasing")
            previous_index = int(indices[-1])
        timestamps = arrays["timestamp"]
        finite = np.isfinite(timestamps)
        if np.any(finite) and not np.all(finite):
            raise ValueError("stored timestamps mix finite and unavailable values")
        mode = bool(np.all(finite))
        if len(timestamps):
            if timestamp_mode is not None and mode != timestamp_mode:
                raise ValueError("stored timestamp availability changes between chunks")
            timestamp_mode = mode
            if mode:
                if previous_timestamp is not None and timestamps[0] <= previous_timestamp:
                    raise ValueError("stored timestamps are not strictly increasing")
                if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
                    raise ValueError("stored timestamps are not strictly increasing")
                previous_timestamp = float(timestamps[-1])
        if height is not None and width is not None:
            xy = arrays["centerline_xy"]
            expected_fov = (
                (xy[..., 0] >= 0) & (xy[..., 0] < int(width))
                & (xy[..., 1] >= 0) & (xy[..., 1] < int(height))
            )
            if not np.array_equal(arrays["in_fov_mask"], expected_fov):
                raise ValueError("stored in_fov_mask disagrees with half-open image bounds")
    return count


def open_completed_output(path: str | os.PathLike[str]) -> h5py.File:
    """Open a validated completed output read-only."""

    validate_output(path)
    return h5py.File(path, "r")
