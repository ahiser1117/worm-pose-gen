"""A setup's label collection: every label saved for the setup's recordings, in the lab and personal libraries.

Each library holds its part of a setup's collection under
``labels/<setup>/`` (the setup reference as a file name, ``lab.nir-flv``)::

    index.json                                   "<recording>/<frame:06d>" -> [revision entries, oldest first]
    <recording>/<frame:06d>/<revision:04d>.npz   one revision (:mod:`.labels`)

The app saves into the personal part; the lab part is published there by
the developer (``scripts/publish.py labels``).  A frame's label is its
newest revision in either part (by ``saved_at``, the lab's on a tie, as
when a personal label was published), so a user's edit of a lab label is a
new personal revision and nothing is copied.

The collection holds every recording with a label; which of them a model
trains, validates or is tested on is a dataset's choice (:mod:`.datasets`).
Every revision stays on disk, so a training run or a benchmark names exact
revisions (:attr:`LabelRecord.identity`).  Writes hold the part's lock; a
revision file is written once and never replaced.
"""

from __future__ import annotations

from collections import Counter
import getpass
from pathlib import Path
from typing import Any

import numpy as np

from .labels import LabelRecord, encode_label, label_key, write_immutable
from .roots import Libraries, check_id, locked, read_json, ref_filename, write_json
from ..workspace import utc_now


LABELS_DIR = "labels"
ENTRY_FIELDS = (
    "revision", "sha256", "saved_at", "author", "origin", "orientation", "mask_only", "has_trace", "empty",
    "foreground_fraction", "ignore_fraction", "height", "width", "source_path", "dataset_path",
)


def collection_dir(root: Path, setup_ref: str) -> Path:
    return Path(root) / LABELS_DIR / ref_filename(setup_ref)


class Collection:
    """The labels of one setup, read from both libraries; saves go to the personal one."""

    def __init__(self, libraries: Libraries, setup_ref: str) -> None:
        from .setups import get_setup

        self.libraries = libraries
        self.setup = get_setup(libraries, setup_ref).ref

    def root(self, scope: str) -> Path:
        return collection_dir(self.libraries.root(scope), self.setup)

    def _index(self, scope: str) -> dict[str, list[dict[str, Any]]]:
        return read_json(self.root(scope) / "index.json", {}) or {}

    def label_path(self, scope: str, recording: str, frame: int, revision: int) -> Path:
        return self.root(scope) / recording / f"{int(frame):06d}" / f"{int(revision):04d}.npz"

    def _record(self, scope: str, recording: str, frame: int, entry: dict[str, Any]) -> LabelRecord:
        return LabelRecord(
            scope=scope, setup=self.setup, recording=recording, frame=int(frame), split=None,
            path=str(self.label_path(scope, recording, frame, entry["revision"])), **{k: entry[k] for k in ENTRY_FIELDS},
        )

    def _all(self) -> dict[str, list[LabelRecord]]:
        """Every revision of every frame, oldest first (the lab's last on a tie, so it is current)."""

        found: dict[str, list[LabelRecord]] = {}
        for scope in self.libraries.scopes():
            for key, entries in self._index(scope).items():
                recording, frame = key.rsplit("/", 1)
                found.setdefault(key, []).extend(self._record(scope, recording, int(frame), entry) for entry in entries)
        for records in found.values():
            records.sort(key=lambda r: (r.saved_at, r.scope == "lab", r.revision))
        return found

    def labels(self) -> list[LabelRecord]:
        """The current label of every labeled frame, sorted by recording and frame."""

        return sorted((records[-1] for records in self._all().values()), key=lambda r: (r.recording, r.frame))

    def recordings(self) -> dict[str, int]:
        """Recording -> its number of labeled frames."""

        return dict(sorted(Counter(r.recording for r in self.labels()).items()))

    def revisions(self, recording: str, frame: int) -> list[LabelRecord]:
        """Every revision of a frame's label in either library, oldest first."""

        return self._all().get(label_key(recording, frame), [])

    def get(self, recording: str, frame: int, revision: int | None = None, scope: str | None = None) -> LabelRecord:
        """A frame's current label, or the named revision of the library ``scope``."""

        records = self.revisions(recording, frame)
        if revision is None:
            if records:
                return records[-1]
            raise LookupError(f"{self.setup} has no label of {label_key(recording, frame)}")
        for record in records:
            if record.revision == int(revision) and record.scope == scope:
                return record
        raise LookupError(f"{self.setup} has no revision {scope}:{revision} of {label_key(recording, frame)}")

    def own_revision(self, recording: str, frame: int) -> int:
        """The newest personal revision of a frame (0 for none): what a save's ``expected_revision`` is checked against."""

        entries = self._index("mine").get(label_key(recording, frame), [])
        return int(entries[-1]["revision"]) if entries else 0

    def save(
        self,
        *,
        recording: str,
        frame: int,
        image: np.ndarray,
        image_raw: np.ndarray,
        mask: np.ndarray,
        context: np.ndarray,
        context_valid: np.ndarray,
        origin: str,
        nose_xy: np.ndarray | None = None,
        nose_valid: np.ndarray | None = None,
        orientation: str = "auto",
        head_xy: Any = None,
        trace_xy: Any = None,
        trace_extend: bool = False,
        trace_length_px: float | None = None,
        mask_only: bool = False,
        author: str | None = None,
        source_path: str = "",
        dataset_path: str = "/img_nir",
        expected_revision: int | None = None,
        saved_at: str | None = None,
        extra_meta: dict[str, Any] | None = None,
        scope: str = "mine",
    ) -> LabelRecord:
        """Write a new revision of a frame's label (see :func:`.labels.encode_label` for the arrays).

        ``expected_revision`` is the personal revision the editor started
        from (:meth:`own_revision`; 0 for none, as when a lab label is
        edited); a different one means someone saved in between, and the
        save is refused.  Only scripts that build the lab library pass
        ``scope="lab"``.
        """

        check_id(recording)
        author = author or getpass.getuser()
        root = self.root(scope)
        with locked(root):
            index = self._index(scope)
            key = label_key(recording, frame)
            entries = index.get(key, [])
            current = entries[-1]["revision"] if entries else 0
            if expected_revision is not None and int(expected_revision) != current:
                raise ValueError(f"{key} changed since it was opened (revision {current}); reload before saving")
            revision = current + 1
            data, entry = encode_label(
                recording=recording, frame=frame, revision=revision, image=image, image_raw=image_raw, mask=mask,
                context=context, context_valid=context_valid, nose_xy=nose_xy, nose_valid=nose_valid,
                orientation=orientation, head_xy=head_xy, trace_xy=trace_xy, trace_extend=trace_extend,
                trace_length_px=trace_length_px, mask_only=mask_only, origin=origin,
                author=author, saved_at=saved_at or utc_now(), source_path=source_path, dataset_path=dataset_path,
                extra_meta=extra_meta,
            )
            write_immutable(self.label_path(scope, recording, frame, revision), data)
            index[key] = [*entries, entry]
            write_json(root / "index.json", index)
        return self._record(scope, recording, frame, entry)
