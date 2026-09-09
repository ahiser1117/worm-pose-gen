"""Versioned user corpus, compatible with the original segmentation store.

Live samples retain the legacy index/samples layout. Every app save also
keeps an immutable revision; training copies exact bytes while holding the
corpus lock, so later edits or deletions cannot change a queued run.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .segmentation_dataset import SegmentationStore, SampleRecord, make_sample_id
from .workspace import _write_json_atomic, utc_now


def recording_identity(path: str | Path, dataset: str = "/img_nir") -> str:
    canonical = json.dumps([str(Path(path).expanduser().resolve()), "/" + dataset.strip("/")])
    return "rec_" + hashlib.sha256(canonical.encode()).hexdigest()[:24]


class CorpusStore(SegmentationStore):
    def _identities(self) -> dict[str, str]:
        path = self.root / "identities.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _remember(self, record: SampleRecord) -> None:
        identities = self._identities()
        key = make_sample_id(recording_identity(record.source_path, record.dataset_path), record.frame_index)
        identities[key] = record.recording
        _write_json_atomic(self.root / "identities.json", identities)

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".corpus.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _archive(self, record: SampleRecord) -> None:
        directory = self.root / "revisions" / record.sample_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{record.revision:08d}.npz"
        if not target.exists():
            temporary = target.with_suffix(".npz.partial")
            shutil.copy2(self.sample_path(record.sample_id), temporary)
            os.replace(temporary, target)
        if not target.with_suffix(".json").exists():
            _write_json_atomic(target.with_suffix(".json"), asdict(record))

    def _save(self, recording, frame_index, image, mask, **kwargs) -> SampleRecord:
        sample_id = make_sample_id(recording, frame_index)
        previous = self.get(sample_id)
        if previous is not None:
            self._archive(previous)
        revisions = self.root / "revisions" / sample_id
        revision = max([0, *[int(p.stem) for p in revisions.glob("*.npz")]]) + 1
        record = super().save(recording, frame_index, image, mask, revision=revision, **kwargs)
        self._archive(record)
        self._remember(record)
        return record

    def save(self, recording, frame_index, image, mask, **kwargs) -> SampleRecord:
        with self.locked():
            return self._save(recording, frame_index, image, mask, **kwargs)

    def save_frame(self, source_path, dataset_path, frame_index, image, mask, *, image_raw=None, split=None) -> SampleRecord:
        """Canonical file/dataset identity; adopt matching legacy labels in place."""
        source_path = str(Path(source_path).expanduser().resolve())
        dataset_path = "/" + dataset_path.strip("/")
        recording = recording_identity(source_path, dataset_path)
        with self.locked():
            recording = self._identities().get(make_sample_id(recording, frame_index), recording)
            # Existing stores used a recording basename. Reuse only an exact
            # source-path/dataset match, preserving its original split pledge.
            for record in self.records():
                if (str(Path(record.source_path).resolve()) == source_path
                        and record.dataset_path == dataset_path and record.frame_index == frame_index):
                    recording = record.recording
                    break
            return self._save(recording, frame_index, image, mask, image_raw=image_raw,
                              source_path=source_path, dataset_path=dataset_path,
                              label_source="manual:workspace", flat_fielded=True, split=split)

    def update_label(self, sample_id: str, mask: np.ndarray, revision: int | None = None) -> SampleRecord:
        with self.locked():
            image, _, record = self.load(sample_id)
            if revision is not None and int(revision) != record.revision:
                raise ValueError("label changed since it was opened; reload before saving")
            return self._save(record.recording, record.frame_index, image, mask,
                              image_raw=self.load_raw(sample_id), source_path=record.source_path,
                              dataset_path=record.dataset_path, label_source="manual:corpus",
                              flat_fielded=record.flat_fielded, split=record.split)

    def delete(self, sample_id: str) -> bool:
        with self.locked():
            previous = self.get(sample_id)
            if previous is not None:
                self._archive(previous)
                self._remember(previous)
            return super().delete(sample_id)

    def snapshot(self, destination: Path) -> dict[str, Any]:
        """Copy live records under the write lock; no hardlinks or live references."""
        with self.locked():
            records = self.records()
            if not any(r.split == "train" for r in records) or not any(r.split == "val" for r in records):
                raise ValueError("fine-tuning needs at least one train and one validation label")
            destination.mkdir(parents=True, exist_ok=False)
            (destination / "samples").mkdir()
            index, entries = {}, []
            for record in records:
                target = destination / "samples" / f"{record.sample_id}.npz"
                shutil.copy2(self.sample_path(record.sample_id), target)
                index[record.sample_id] = asdict(record)
                entries.append({**asdict(record), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            _write_json_atomic(destination / "index.json", index)
            _write_json_atomic(destination / "splits.json", self._read_splits())
            manifest = {"source_root": str(self.root.resolve()), "created_at": utc_now(), "samples": entries,
                        "counts": {s: sum(r.split == s for r in records) for s in ("train", "val", "test")}}
            _write_json_atomic(destination / "manifest.json", manifest)
            return manifest
