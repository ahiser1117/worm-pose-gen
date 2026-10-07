"""Datasets: named sets of label revisions for one setup, their per-recording splits, and inheritance.

``datasets/<id>/`` holds::

    dataset.json        {"setup": ref, "extends": ref | null, "name", "description", "author", "created_at"}
    splits.json         recording -> "train" | "val" | "test", append-only
    labels/index.json   "<recording>/<frame:06d>" -> [revision entries, oldest first]
    labels/<recording>/<frame:06d>/<revision:04d>.npz   (:mod:`.labels`)

**Splits are per recording.** Frames of one recording are not independent,
and the question a split answers is whether a model works on the next
recording.  A recording's first label pledges it to the split furthest below
80/10/10 by label count (counting the labels the dataset inherits); later
labels follow it, and the pledge is never changed or removed.  A recording
that a dataset it extends has already split keeps that split.

**Inheritance.** A personal dataset can extend another dataset (usually a
lab one).  Its labels are the extended dataset's plus its own, and its own
label of a (recording, frame) overrides the inherited one: a user's edit of
a lab label is a new revision in the personal dataset.  Nothing is copied.

Every revision stays on disk, so a training run or a benchmark names exact
revisions (:attr:`LabelRecord.identity`).  Writes hold the dataset's lock;
a revision file is written once and never replaced.
"""

from __future__ import annotations

from collections import Counter
import getpass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .labels import LabelRecord, encode_label, label_key, write_immutable
from .roots import Libraries, check_id, locked, make_ref, parse_ref, read_json, write_json
from ..workspace import utc_now


DATASETS_DIR = "datasets"
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)


def assign_split(counts: dict[str, int]) -> str:
    """The split furthest below its target share after one more label."""

    total = sum(int(counts.get(name, 0)) for name in SPLITS) + 1
    deficits = [fraction * total - int(counts.get(name, 0)) for name, fraction in zip(SPLITS, SPLIT_FRACTIONS, strict=True)]
    return SPLITS[int(np.argmax(deficits))]


def dataset_dir(root: Path, dataset_id: str) -> Path:
    return Path(root) / DATASETS_DIR / check_id(dataset_id)


def create_dataset(
    libraries: Libraries, dataset_id: str, *, setup: str, extends: str | None = None, name: str = "",
    description: str = "", author: str | None = None, scope: str = "mine",
) -> "Dataset":
    """A new, empty dataset (personal unless a script passes ``scope="lab"`` with a writable lab root)."""

    from .setups import get_setup

    ref = make_ref(scope, dataset_id)
    directory = dataset_dir(libraries.root(scope), dataset_id)
    if (directory / "dataset.json").exists():
        raise FileExistsError(f"dataset {ref} already exists")
    get_setup(libraries, setup)
    if extends is not None:
        parent = Dataset(libraries, extends)
        if parent.setup != setup:
            raise ValueError(f"{ref} is for {setup} but {extends} is for {parent.setup}")
    write_json(directory / "dataset.json", {
        "setup": setup, "extends": extends, "name": name or dataset_id, "description": description,
        "author": author or getpass.getuser(), "created_at": utc_now(),
    })
    write_json(directory / "splits.json", {})
    return Dataset(libraries, ref, writable=True)


class Dataset:
    """A dataset as one library holds it, with its inherited labels resolved through ``libraries``.

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

    @property
    def extends(self) -> str | None:
        return self.info.get("extends")

    def chain(self) -> list["Dataset"]:
        """This dataset, then the one it extends, and so on."""

        chain = [self]
        while chain[-1].extends is not None:
            parent = Dataset(self.libraries, chain[-1].extends)
            if any(d.ref == parent.ref for d in chain):
                raise ValueError(f"{self.ref}: circular 'extends'")
            chain.append(parent)
        return chain

    # ------------------------------------------------------------------ splits

    def own_splits(self) -> dict[str, str]:
        return dict(read_json(self.root / "splits.json", {}) or {})

    def splits(self) -> dict[str, str]:
        """Recording -> split, inherited pledges included."""

        result: dict[str, str] = {}
        for dataset in reversed(self.chain()):
            for recording, split in dataset.own_splits().items():
                result.setdefault(recording, split)
        return result

    # ------------------------------------------------------------------ labels

    def _index(self) -> dict[str, list[dict[str, Any]]]:
        return read_json(self.root / "labels" / "index.json", {}) or {}

    def label_path(self, recording: str, frame: int, revision: int) -> Path:
        return self.root / "labels" / recording / f"{int(frame):06d}" / f"{int(revision):04d}.npz"

    def _record(self, recording: str, frame: int, entry: dict[str, Any], split: str | None) -> LabelRecord:
        return LabelRecord(
            dataset=self.ref, recording=recording, frame=int(frame), split=split,
            path=str(self.label_path(recording, frame, entry["revision"])),
            **{k: entry[k] for k in (
                "revision", "sha256", "saved_at", "author", "origin", "orientation", "mask_only", "has_trace", "empty",
                "foreground_fraction", "ignore_fraction", "height", "width", "source_path", "dataset_path",
            )},
        )

    def own_labels(self) -> list[LabelRecord]:
        """The current (newest) revision of each of this dataset's own labels."""

        splits = self.splits()
        records = []
        for key, entries in self._index().items():
            recording, frame = key.rsplit("/", 1)
            records.append(self._record(recording, int(frame), entries[-1], splits.get(recording)))
        return records

    def labels(self) -> list[LabelRecord]:
        """The dataset's labels: inherited ones, overridden by its own, sorted by recording and frame."""

        return resolve([self])

    def revisions(self, recording: str, frame: int) -> list[LabelRecord]:
        """Every revision of this dataset's own label of a frame, oldest first."""

        split = self.splits().get(recording)
        entries = self._index().get(label_key(recording, frame), [])
        return [self._record(recording, frame, entry, split) for entry in entries]

    def get(self, recording: str, frame: int, revision: int | None = None) -> LabelRecord:
        """A frame's label as this dataset sees it (its own, else inherited); a ``revision`` must be its own."""

        if revision is not None:
            for record in self.revisions(recording, frame):
                if record.revision == int(revision):
                    return record
            raise LookupError(f"{self.ref} has no revision {revision} of {label_key(recording, frame)}")
        for dataset in self.chain():
            own = dataset.revisions(recording, frame)
            if own:
                return own[-1]
        raise LookupError(f"{self.ref} has no label of {label_key(recording, frame)}")

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
        mask_only: bool = False,
        author: str | None = None,
        source_path: str = "",
        dataset_path: str = "/img_nir",
        expected_revision: int | None = None,
        saved_at: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> LabelRecord:
        """Write a new revision of a frame's label (see :func:`.labels.encode_label` for the arrays).

        ``expected_revision`` is the revision of this dataset's own label the
        editor started from (0 for none, as when a lab label is edited); a
        different current revision means someone saved in between, and the
        save is refused.  The recording's first label pledges its split.
        """

        if not self.writable:
            raise PermissionError(f"{self.ref} is read-only")
        check_id(recording)
        author = author or getpass.getuser()
        with locked(self.root):
            index = self._index()
            key = label_key(recording, frame)
            entries = index.get(key, [])
            current = entries[-1]["revision"] if entries else 0
            if expected_revision is not None and int(expected_revision) != current:
                raise ValueError(f"{key} changed since it was opened (revision {current}); reload before saving")
            revision = current + 1
            data, entry = encode_label(
                recording=recording, frame=frame, revision=revision, image=image, image_raw=image_raw, mask=mask,
                context=context, context_valid=context_valid, nose_xy=nose_xy, nose_valid=nose_valid,
                orientation=orientation, head_xy=head_xy, trace_xy=trace_xy, mask_only=mask_only, origin=origin,
                author=author, saved_at=saved_at or utc_now(), source_path=source_path, dataset_path=dataset_path,
                extra_meta=extra_meta,
            )
            splits = self.own_splits()
            if recording not in splits:
                inherited = self.splits().get(recording)
                splits[recording] = inherited or assign_split(Counter(r.split for r in self.labels()))
                write_json(self.root / "splits.json", splits)
            write_immutable(self.label_path(recording, frame, revision), data)
            index[key] = [*entries, entry]
            write_json(self.root / "labels" / "index.json", index)
        return self._record(recording, frame, entry, splits[recording])

    # ------------------------------------------------------------------ summary

    def summary(self) -> dict[str, Any]:
        """Counts for the Datasets tab: labels by split and status, recordings with their split, readiness."""

        records = self.labels()
        splits = self.splits()
        recordings: dict[str, dict[str, Any]] = {}
        for record in records:
            row = recordings.setdefault(record.recording, {"recording": record.recording, "split": splits.get(record.recording), "labels": 0, "statuses": Counter()})
            row["labels"] += 1
            row["statuses"][record.status] += 1
        for row in recordings.values():
            row["statuses"] = dict(row["statuses"])
        by_split = {split: sum(r.split == split for r in records) for split in SPLITS}
        recordings_by_split = {split: sum(row["split"] == split for row in recordings.values()) for split in SPLITS}
        readiness = []
        if not by_split["train"]:
            readiness.append("no training labels")
        if not recordings_by_split["val"]:
            readiness.append("no validation recording")
        if not recordings_by_split["test"]:
            readiness.append("no test recording")
        return {
            "ref": self.ref, "setup": self.setup, "extends": self.extends, "name": self.info.get("name"),
            "description": self.info.get("description"), "author": self.info.get("author"),
            "created_at": self.info.get("created_at"), "writable": self.writable,
            "labels": len(records), "own_labels": len(self._index()), "by_split": by_split,
            "recordings_by_split": recordings_by_split, "by_status": dict(Counter(r.status for r in records)),
            "recordings": sorted(recordings.values(), key=lambda row: row["recording"]), "readiness": readiness,
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


def resolve(datasets: Sequence[Dataset]) -> list[LabelRecord]:
    """The labels of several datasets together: each dataset's own labels override those it extends.

    Datasets are taken base first (a dataset after every dataset it
    extends), and a later dataset's label of a (recording, frame) replaces an
    earlier one.  Two datasets that put a recording in different splits
    cannot be used together.
    """

    ordered: list[Dataset] = []
    for dataset in datasets:
        for member in reversed(dataset.chain()):
            if all(d.ref != member.ref for d in ordered):
                ordered.append(member)
    chosen: dict[str, LabelRecord] = {}
    split_of: dict[str, tuple[str, str]] = {}
    for dataset in ordered:
        for recording, split in dataset.splits().items():
            previous = split_of.setdefault(recording, (split, dataset.ref))
            if previous[0] != split:
                raise ValueError(f"recording {recording} is {previous[0]} in {previous[1]} but {split} in {dataset.ref}")
        for record in dataset.own_labels():
            chosen[record.key] = record
    return sorted(chosen.values(), key=lambda r: (r.recording, r.frame))


def labels(
    libraries: Libraries,
    dataset_refs: Iterable[str],
    split: str | None = None,
    *,
    recording: str | None = None,
    status: str | None = None,
) -> list[LabelRecord]:
    """The labels training or evaluation reads: the datasets' labels resolved together, optionally filtered.

    ``labels(libs, ["mine:copper"], "train")`` gives the personal dataset's
    training labels plus those of the lab dataset it extends; each record's
    :meth:`~LabelRecord.load` reads the arrays.
    """

    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    records = resolve([Dataset(libraries, ref) for ref in dataset_refs])
    return [
        r for r in records
        if (split is None or r.split == split) and (recording is None or r.recording == recording)
        and (status is None or r.status == status)
    ]


def fingerprint(records: Iterable[LabelRecord]) -> str:
    """A short hash naming exactly this set of label revisions (order does not matter)."""

    lines = sorted(f"{r.dataset}/{r.key}@{r.revision}:{r.sha256}" for r in records)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def trained_on(records: Sequence[LabelRecord]) -> list[dict[str, Any]]:
    """A model card's ``trained_on``: per dataset, the fingerprint and counts of the revisions a run used."""

    result = []
    for dataset in sorted({r.dataset for r in records}):
        group = [r for r in records if r.dataset == dataset]
        result.append({
            "dataset": dataset, "fingerprint": fingerprint(group),
            "counts": {split: sum(r.split == split for r in group) for split in SPLITS},
            "recordings": len({r.recording for r in group}),
        })
    return result
