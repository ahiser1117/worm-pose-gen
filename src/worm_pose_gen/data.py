"""Bounded, process-safe access to frame datasets in HDF5 recordings.

The dataset path and expected layout are deliberately caller supplied.  This
module never guesses which dataset in a recording contains images.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np
from numpy.typing import NDArray


PaddingMode = Literal["error", "edge", "constant"]


@dataclass(frozen=True)
class FrameDatasetInfo:
    """Validated, immutable description of a source frame dataset."""

    source_path: Path
    resolved_source_path: Path
    dataset_path: str
    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def frame_count(self) -> int:
        return self.shape[0]

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return self.shape[1:]


@dataclass(frozen=True)
class FrameWindow:
    """A temporal window and its mapping back to source frame indices.

    ``valid_mask`` reports whether each requested index existed in the source.
    Edge-padded positions repeat the clipped source index explicitly but remain
    marked false, so consumers cannot mistake padding for a real observation.
    """

    frames: NDArray[np.generic]
    source_indices: NDArray[np.int64]
    valid_mask: NDArray[np.bool_]


class HDF5FrameSource:
    """Lazy, read-only, process-local access to one explicit HDF5 dataset.

    Parameters make layout assumptions executable rather than implicit.
    ``max_frames_per_read`` is a hard bound on every public read operation.
    A pickled instance contains configuration only; an open HDF5 handle is
    never serialized or shared with a child process.
    """

    def __init__(
        self,
        source_path: str | os.PathLike[str],
        dataset_path: str,
        *,
        expected_frame_shape: Sequence[int] | None = None,
        allowed_dtypes: Sequence[str | np.dtype] | None = None,
        expected_ndim: int | None = None,
        max_frames_per_read: int = 256,
    ) -> None:
        if not dataset_path or not dataset_path.startswith("/"):
            raise ValueError("dataset_path must be an explicit absolute HDF5 path")
        if max_frames_per_read < 1:
            raise ValueError("max_frames_per_read must be positive")
        self.source_path = Path(source_path)
        self.dataset_path = dataset_path
        self.expected_frame_shape = (
            tuple(int(v) for v in expected_frame_shape)
            if expected_frame_shape is not None
            else None
        )
        self.allowed_dtypes = (
            tuple(np.dtype(value) for value in allowed_dtypes)
            if allowed_dtypes is not None
            else None
        )
        self.expected_ndim = expected_ndim
        self.max_frames_per_read = max_frames_per_read
        self._file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None
        self._owner_pid: int | None = None
        self._info: FrameDatasetInfo | None = None

    def _open(self) -> h5py.Dataset:
        pid = os.getpid()
        if self._file is not None and self._owner_pid != pid:
            # Do not call into an inherited HDF5 handle after fork.
            self._file = None
            self._dataset = None
            self._owner_pid = None
        if self._file is None:
            handle = h5py.File(self.source_path, "r")
            try:
                if self.dataset_path not in handle:
                    raise KeyError(
                        f"dataset {self.dataset_path!r} not found in {self.source_path}"
                    )
                obj = handle[self.dataset_path]
                if not isinstance(obj, h5py.Dataset):
                    raise TypeError(f"{self.dataset_path!r} is not an HDF5 dataset")
                self._validate_dataset(obj)
            except BaseException:
                handle.close()
                raise
            self._file = handle
            self._dataset = obj
            self._owner_pid = pid
            self._info = FrameDatasetInfo(
                source_path=self.source_path,
                resolved_source_path=self.source_path.resolve(strict=True),
                dataset_path=self.dataset_path,
                shape=tuple(obj.shape),
                dtype=np.dtype(obj.dtype),
            )
        assert self._dataset is not None
        return self._dataset

    def _validate_dataset(self, dataset: h5py.Dataset) -> None:
        if dataset.ndim < 1:
            raise ValueError("frame dataset must have a leading frame dimension")
        if self.expected_ndim is not None and dataset.ndim != self.expected_ndim:
            raise ValueError(
                f"expected dataset ndim {self.expected_ndim}, got {dataset.ndim}"
            )
        if self.expected_frame_shape is not None:
            actual = tuple(dataset.shape[1:])
            if actual != self.expected_frame_shape:
                raise ValueError(
                    f"expected frame shape {self.expected_frame_shape}, got {actual}"
                )
        if self.allowed_dtypes is not None and np.dtype(dataset.dtype) not in self.allowed_dtypes:
            raise TypeError(
                f"dataset dtype {dataset.dtype} is not one of {self.allowed_dtypes}"
            )

    @property
    def info(self) -> FrameDatasetInfo:
        self._open()
        assert self._info is not None
        return self._info

    def __len__(self) -> int:
        return self.info.frame_count

    def read_frame(self, frame_index: int) -> NDArray[np.generic]:
        dataset = self._open()
        index = self._normalize_index(frame_index)
        return np.asarray(dataset[index])

    def read_slice(self, start: int, stop: int) -> NDArray[np.generic]:
        """Read a contiguous half-open interval without accepting strides."""

        if start < 0 or stop < start or stop > len(self):
            raise IndexError(f"invalid frame interval [{start}, {stop})")
        count = stop - start
        self._check_read_count(count)
        return np.asarray(self._open()[start:stop])

    def read_indices(self, frame_indices: Sequence[int]) -> NDArray[np.generic]:
        """Read explicit indices, preserving order and duplicates.

        h5py fancy-index restrictions are avoided with contiguous run reads.
        This also makes discontinuities explicit at the call site.
        """

        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("frame_indices must be one-dimensional")
        self._check_read_count(len(indices))
        if len(indices) == 0:
            return np.empty((0, *self.info.frame_shape), dtype=self.info.dtype)
        normalized = [self._normalize_index(int(index)) for index in indices]
        dataset = self._open()
        return np.stack([np.asarray(dataset[index]) for index in normalized], axis=0)

    def read_window(
        self,
        center_index: int,
        *,
        before: int,
        after: int,
        padding: PaddingMode = "error",
        constant_value: int | float = 0,
    ) -> FrameWindow:
        """Read ``[center-before, center+after]`` with explicit boundary policy."""

        if before < 0 or after < 0:
            raise ValueError("before and after must be non-negative")
        if padding not in ("error", "edge", "constant"):
            raise ValueError(f"unsupported padding mode: {padding!r}")
        requested = np.arange(
            center_index - before, center_index + after + 1, dtype=np.int64
        )
        self._check_read_count(len(requested))
        frame_count = len(self)
        if frame_count == 0:
            raise IndexError("cannot read a window from an empty dataset")
        valid = (requested >= 0) & (requested < frame_count)
        if padding == "error" and not np.all(valid):
            raise IndexError("window extends outside the recording")
        clipped = np.clip(requested, 0, frame_count - 1)
        frames = self.read_indices(clipped)
        source_indices = clipped
        if padding == "constant" and not np.all(valid):
            frames = frames.copy()
            frames[~valid] = constant_value
            source_indices = requested.copy()
            source_indices[~valid] = -1
        return FrameWindow(frames, source_indices, valid.astype(np.bool_))

    def _normalize_index(self, index: int) -> int:
        # Negative indexing is rejected so source mappings remain unambiguous.
        if index < 0 or index >= len(self):
            raise IndexError(f"frame index {index} is outside [0, {len(self)})")
        return index

    def _check_read_count(self, count: int) -> None:
        if count > self.max_frames_per_read:
            raise ValueError(
                f"read of {count} frames exceeds max_frames_per_read="
                f"{self.max_frames_per_read}"
            )

    @property
    def is_open(self) -> bool:
        return self._file is not None and self._owner_pid == os.getpid()

    def close(self) -> None:
        if self._file is not None and self._owner_pid == os.getpid():
            self._file.close()
        self._file = None
        self._dataset = None
        self._owner_pid = None

    def __enter__(self) -> HDF5FrameSource:
        self._open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_dataset"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
