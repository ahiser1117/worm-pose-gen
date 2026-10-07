"""The old on-disk store of worm segmentation labels (``segmentation_v1`` and the app corpus).

One sample is one ``.npz`` file under ``<root>/samples/`` holding the frame
the network sees (``image``, flat-fielded uint8), the original frame
(``image_raw``), and the label (``mask``: 0 background, 1 worm, 255 ignore).
``<root>/index.json`` records provenance and the split of every sample.

A sample's split is assigned once, when it is first saved, to whichever of
train, validation, and test is furthest below its 80/10/10 target, so the
proportions hold even for a small set.  The assignment is pledged in
``<root>/splits.json``, which is append-only: re-saving a refined label,
deleting the sample, or labeling the same frame again later all keep the
pledged split, so a frame that has ever been validation or test can never
enter the training set.

Models are no longer trained from this store: its labels were migrated into
the library (``scripts/migrate_to_library.py``), which training and
evaluation read (:mod:`model_training`, :mod:`model_eval`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .library.datasets import SPLITS, assign_split
from .segmenter import IGNORE_LABEL


DEFAULT_DATASET_ROOT = Path(
    "/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/segmentation_v1"
)


def make_sample_id(recording: str, frame_index: int) -> str:
    return f"{recording}_f{int(frame_index):06d}"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    recording: str
    frame_index: int
    split: str
    source_path: str
    label_source: str
    saved_at: str
    image_height: int
    image_width: int
    foreground_fraction: float
    ignore_fraction: float
    flat_fielded: bool
    revision: int = 1
    dataset_path: str = "/img_nir"


def _validate_mask(mask: NDArray[np.generic]) -> NDArray[np.uint8]:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must have shape [H,W]")
    if values.dtype == bool:
        return values.astype(np.uint8)
    allowed = np.isin(values, (0, 1, IGNORE_LABEL))
    if not allowed.all():
        raise ValueError("mask values must be 0, 1, or 255")
    return values.astype(np.uint8)


class SegmentationStore:
    """Append-or-replace store of labeled frames with an atomic JSON index."""

    def __init__(self, root: str | Path = DEFAULT_DATASET_ROOT) -> None:
        self.root = Path(root)
        self.samples_dir = self.root / "samples"
        self.index_path = self.root / "index.json"
        self.splits_path = self.root / "splits.json"
        self._lock = threading.Lock()

    def _read_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text())

    def _write_index(self, index: dict[str, dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(index, indent=1, sort_keys=True))
        os.replace(temporary, self.index_path)

    def _read_splits(self, index: dict[str, dict[str, Any]] | None = None) -> dict[str, str]:
        """The pledged split of every sample ever saved (deleted ones included).

        A store written before the registry existed is seeded from its index.
        """

        if self.splits_path.exists():
            return json.loads(self.splits_path.read_text())
        index = self._read_index() if index is None else index
        return {sample_id: str(value["split"]) for sample_id, value in index.items()}

    def _write_splits(self, splits: dict[str, str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.splits_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(splits, indent=1, sort_keys=True))
        os.replace(temporary, self.splits_path)

    def pledged_split(self, recording: str, frame_index: int) -> str | None:
        """The split this frame will land in, or has ever been in; None if unseen."""

        return self._read_splits().get(make_sample_id(recording, frame_index))

    def sample_path(self, sample_id: str) -> Path:
        return self.samples_dir / f"{sample_id}.npz"

    def has(self, recording: str, frame_index: int) -> bool:
        return make_sample_id(recording, frame_index) in self._read_index()

    def get(self, sample_id: str) -> SampleRecord | None:
        record = self._read_index().get(sample_id)
        return None if record is None else SampleRecord(**record)

    def records(self, split: str | None = None) -> list[SampleRecord]:
        rows = [SampleRecord(**value) for value in self._read_index().values()]
        if split is not None:
            if split not in SPLITS:
                raise ValueError(f"unknown split {split!r}")
            rows = [row for row in rows if row.split == split]
        return sorted(rows, key=lambda row: row.sample_id)

    def counts(self) -> dict[str, int]:
        result = {name: 0 for name in SPLITS}
        for record in self.records():
            result[record.split] += 1
        return result

    def save(
        self,
        recording: str,
        frame_index: int,
        image: NDArray[np.generic],
        mask: NDArray[np.generic],
        *,
        source_path: str,
        label_source: str,
        image_raw: NDArray[np.generic] | None = None,
        flat_fielded: bool = True,
        split: str | None = None,
        dataset_path: str = "/img_nir",
        revision: int | None = None,
    ) -> SampleRecord:
        """Write one sample atomically; a repeat save replaces the label.

        ``split`` pledges a new sample to that split instead of the balanced
        assignment (a recording whose animal should appear only in validation
        or test).  A sample that already has a pledge keeps it.
        """

        if split is not None and split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")

        frame = np.asarray(image)
        if frame.ndim != 2:
            raise ValueError("image must have shape [H,W]")
        label = _validate_mask(mask)
        if label.shape != frame.shape:
            raise ValueError("mask and image shapes must match")
        frame_u8 = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
        raw_u8 = (
            frame_u8 if image_raw is None else np.clip(np.rint(np.asarray(image_raw)), 0, 255).astype(np.uint8)
        )
        if raw_u8.shape != frame_u8.shape:
            raise ValueError("raw image and corrected image shapes must match")
        sample_id = make_sample_id(recording, frame_index)
        valid = label != IGNORE_LABEL
        with self._lock:
            index = self._read_index()
            previous = index.get(sample_id)
            splits = self._read_splits(index)
            if sample_id in splits:
                split = splits[sample_id]
            else:
                if split is None:
                    # Balance against every pledge ever made, not just the samples
                    # currently present, so deletions cannot skew later assignments.
                    counts = {name: 0 for name in SPLITS}
                    for value in splits.values():
                        counts[value] += 1
                    split = assign_split(counts)
                splits[sample_id] = split
                self._write_splits(splits)
            record = SampleRecord(
                sample_id=sample_id,
                recording=str(recording),
                frame_index=int(frame_index),
                split=split,
                source_path=str(source_path),
                label_source=str(label_source),
                saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                image_height=int(frame_u8.shape[0]),
                image_width=int(frame_u8.shape[1]),
                foreground_fraction=float((label == 1).sum() / label.size),
                ignore_fraction=float((~valid).sum() / label.size),
                flat_fielded=bool(flat_fielded),
                revision=revision if revision is not None else (int(previous["revision"]) + 1 if previous else 1),
                dataset_path=dataset_path,
            )
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            path = self.sample_path(sample_id)
            temporary = path.with_suffix(".npz.partial")
            with open(temporary, "wb") as handle:
                np.savez_compressed(handle, image=frame_u8, image_raw=raw_u8, mask=label)
            os.replace(temporary, path)
            index[sample_id] = asdict(record)
            self._write_index(index)
        return record

    def load(self, sample_id: str) -> tuple[NDArray[np.uint8], NDArray[np.uint8], SampleRecord]:
        record = self.get(sample_id)
        if record is None:
            raise KeyError(sample_id)
        with np.load(self.sample_path(sample_id)) as archive:
            return (
                np.asarray(archive["image"], dtype=np.uint8),
                np.asarray(archive["mask"], dtype=np.uint8),
                record,
            )

    def load_raw(self, sample_id: str) -> NDArray[np.uint8]:
        with np.load(self.sample_path(sample_id)) as archive:
            return np.asarray(archive["image_raw"], dtype=np.uint8)

    def delete(self, sample_id: str) -> bool:
        """Remove a sample's files and index entry; its split pledge is kept."""

        with self._lock:
            index = self._read_index()
            if sample_id not in index:
                return False
            del index[sample_id]
            self._write_index(index)
            path = self.sample_path(sample_id)
            if path.exists():
                path.unlink()
        return True
