"""Datasets: which split each recording of a setup's label collection goes to.

``datasets/<id>/`` holds::

    dataset.json    {"setup": ref, "name", "description", "author", "created_at"}
    splits.json     recording -> "train" | "val" | "test"

A dataset holds no labels: they are the setup's collection's
(:mod:`.collection`), and the dataset chooses, recording by recording,
whether a recording's labels train, validate or test a model, or are not
included.  A recording missing from ``splits.json`` is not included, so a
new dataset includes nothing and a recording labeled for the first time
joins no dataset until someone chooses its split.  A recording's labels
added later follow its split.

**Splits are per recording.** Frames of one recording are not independent,
and the question a split answers is whether a model works on the next
recording.

A training run reads the dataset's labels as they are when it starts and
records the exact revisions and their splits (:func:`trained_on`), so a
split changed later does not change what a model was trained on.  Only
personal datasets are writable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import getpass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

from .collection import Collection
from .labels import LabelRecord
from .roots import Libraries, check_id, locked, make_ref, parse_ref, read_json, write_json
from ..workspace import utc_now


DATASETS_DIR = "datasets"
SPLITS = ("train", "val", "test")


def dataset_dir(root: Path, dataset_id: str) -> Path:
    return Path(root) / DATASETS_DIR / check_id(dataset_id)


def create_dataset(
    libraries: Libraries, dataset_id: str, *, setup: str, name: str = "", description: str = "", author: str | None = None,
    splits: dict[str, str] | None = None, scope: str = "mine",
) -> "Dataset":
    """A new dataset with every recording not included unless ``splits`` says otherwise.

    It is personal unless a script passes ``scope="lab"`` with a writable lab root.
    """

    from .setups import get_setup

    ref = make_ref(scope, dataset_id)
    directory = dataset_dir(libraries.root(scope), dataset_id)
    if (directory / "dataset.json").exists():
        raise FileExistsError(f"dataset {ref} already exists")
    get_setup(libraries, setup)
    for recording, split in (splits or {}).items():
        check_id(recording)
        check_split(split)
    write_json(directory / "dataset.json", {
        "setup": setup, "name": name or dataset_id, "description": description,
        "author": author or getpass.getuser(), "created_at": utc_now(),
    })
    write_json(directory / "splits.json", dict(splits or {}))
    return Dataset(libraries, ref, writable=True)


def check_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return split


class Dataset:
    """A dataset as one library holds it; its labels come from its setup's collection.

    Only personal datasets are writable; scripts that build a lab library
    pass ``writable=True``.
    """

    def __init__(self, libraries: Libraries, ref: str, *, writable: bool | None = None) -> None:
        self.libraries = libraries
        self.ref = ref
        self.scope, self.id = parse_ref(ref)
        self.root = dataset_dir(libraries.root(self.scope), self.id)
        self.info = read_json(self.root / "dataset.json")
        if self.info is None:
            raise LookupError(f"unknown dataset {ref}")
        self.writable = self.scope == "mine" if writable is None else writable

    @property
    def setup(self) -> str:
        return str(self.info["setup"])

    def collection(self) -> Collection:
        return Collection(self.libraries, self.setup)

    def splits(self) -> dict[str, str]:
        """Recording -> split, for the recordings the dataset includes."""

        return dict(read_json(self.root / "splits.json", {}) or {})

    def set_splits(self, changes: dict[str, str | None]) -> dict[str, str]:
        """Put recordings in a split, or out of the dataset with ``None``; returns the new splits."""

        if not self.writable:
            raise PermissionError(f"{self.ref} is in the lab library, which is read-only")
        for recording, split in changes.items():
            check_id(recording)
            if split is not None:
                check_split(split)
        with locked(self.root):
            splits = self.splits()
            for recording, split in changes.items():
                if split is None:
                    splits.pop(recording, None)
                else:
                    splits[recording] = split
            write_json(self.root / "splits.json", splits)
        return splits

    def labels(self) -> list[LabelRecord]:
        """The collection's labels of the included recordings, each with its recording's split."""

        splits = self.splits()
        return [replace(r, split=splits[r.recording]) for r in self.collection().labels() if r.recording in splits]

    def summary(self) -> dict[str, Any]:
        """Counts for the Datasets tab: every recording of the collection with its labels and split, totals by split, readiness."""

        splits = self.splits()
        recordings: dict[str, dict[str, Any]] = {}
        for record in self.collection().labels():
            row = recordings.setdefault(record.recording, {
                "recording": record.recording, "split": splits.get(record.recording), "labels": 0, "statuses": Counter(),
            })
            row["labels"] += 1
            row["statuses"][record.status] += 1
        for row in recordings.values():
            row["statuses"] = dict(row["statuses"])
        rows = sorted(recordings.values(), key=lambda row: row["recording"])
        by_split = {split: sum(row["labels"] for row in rows if row["split"] == split) for split in SPLITS}
        recordings_by_split = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
        statuses: Counter[str] = Counter()
        for row in rows:
            if row["split"] is not None:
                statuses.update(row["statuses"])
        readiness = []
        if not by_split["train"]:
            readiness.append("no training labels")
        if not recordings_by_split["val"]:
            readiness.append("no validation recording")
        if not recordings_by_split["test"]:
            readiness.append("no test recording")
        return {
            "ref": self.ref, "setup": self.setup, "name": self.info.get("name"),
            "description": self.info.get("description"), "author": self.info.get("author"),
            "created_at": self.info.get("created_at"), "writable": self.writable,
            "labels": sum(by_split.values()), "by_split": by_split, "recordings_by_split": recordings_by_split,
            "not_included": {"recordings": sum(row["split"] is None for row in rows),
                             "labels": sum(row["labels"] for row in rows if row["split"] is None)},
            "by_status": dict(statuses), "recordings": rows, "readiness": readiness,
        }


def list_datasets(libraries: Libraries, setup: str | None = None) -> list[Dataset]:
    found = []
    for scope in libraries.scopes():
        directory = libraries.root(scope) / DATASETS_DIR
        for path in sorted(directory.glob("*/dataset.json")) if directory.is_dir() else ():
            dataset = Dataset(libraries, make_ref(scope, path.parent.name))
            if setup is None or dataset.setup == setup:
                found.append(dataset)
    return found


def labels(
    libraries: Libraries,
    dataset_ref: str,
    split: str | None = None,
    *,
    recording: str | None = None,
    status: str | None = None,
) -> list[LabelRecord]:
    """The labels training or evaluation reads: a dataset's included labels, optionally filtered.

    ``labels(libs, "mine:copper", "train")`` gives the setup's labels of the
    recordings the dataset puts in ``train``; each record's
    :meth:`~LabelRecord.load` reads the arrays.
    """

    if split is not None:
        check_split(split)
    return [
        r for r in Dataset(libraries, dataset_ref).labels()
        if (split is None or r.split == split) and (recording is None or r.recording == recording)
        and (status is None or r.status == status)
    ]


def fingerprint(records: Iterable[LabelRecord]) -> str:
    """A short hash naming exactly this set of label revisions and their splits (order does not matter)."""

    lines = sorted(f"{r.scope}:{r.setup}/{r.key}@{r.revision}:{r.sha256}:{r.split}" for r in records)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def trained_on(dataset_ref: str, records: Sequence[LabelRecord]) -> list[dict[str, Any]]:
    """A model card's ``trained_on``: the dataset, and the fingerprint and counts of the revisions a run used."""

    return [{
        "dataset": dataset_ref, "fingerprint": fingerprint(records),
        "counts": {split: sum(r.split == split for r in records) for split in SPLITS},
        "recordings": len({r.recording for r in records}),
    }]
