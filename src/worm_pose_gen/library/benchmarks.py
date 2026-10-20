"""Benchmarks: frozen lists of test-label revisions that every model of a setup is scored on.

``benchmarks/<id>.json``::

    {"setup": ref, "dataset": ref, "author", "created_at", "description",
     "labels": [{"scope", "setup", "recording", "frame", "revision", "sha256"}, ...]}

A benchmark takes the labels of a dataset's test recordings whose origin is
``spread`` or ``migrated``: frames chosen to sample recordings, not frames a
person relabeled because a model failed on them, which would make any model
look worse than it is.  It is written once and never changed; when the test
set has grown, a new benchmark is frozen (``nir-v2``) and every model is
scored on it.  A personal benchmark freezes a personal dataset's test labels
as ``mine:<dataset>-b<n>``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import getpass
from pathlib import Path
from typing import Any, Sequence

from .collection import Collection
from .datasets import Dataset, labels
from .labels import BENCHMARK_ORIGINS, LabelRecord
from .roots import Libraries, check_id, make_ref, parse_ref, read_json, write_json
from ..workspace import utc_now


BENCHMARKS_DIR = "benchmarks"


@dataclass(frozen=True)
class Benchmark:
    ref: str
    setup: str
    dataset: str
    author: str
    created_at: str
    description: str
    entries: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        return {
            "ref": self.ref, "setup": self.setup, "dataset": self.dataset, "author": self.author,
            "created_at": self.created_at, "description": self.description, "labels": len(self.entries),
            "recordings": sorted({e["recording"] for e in self.entries}),
        }


def benchmark_path(root: Path, benchmark_id: str) -> Path:
    return Path(root) / BENCHMARKS_DIR / f"{check_id(benchmark_id)}.json"


def benchmark_records(records: Sequence[LabelRecord]) -> list[LabelRecord]:
    """The records a benchmark may hold: test split, spread or migrated."""

    return [r for r in records if r.split == "test" and r.origin in BENCHMARK_ORIGINS]


def write_benchmark(
    root: Path, benchmark_id: str, *, setup: str, dataset: str, records: Sequence[LabelRecord],
    author: str | None = None, description: str = "",
) -> None:
    """Freeze ``records`` as a benchmark in a library root; an existing benchmark is never replaced."""

    path = benchmark_path(root, benchmark_id)
    if path.exists():
        raise FileExistsError(f"benchmark {benchmark_id} already exists; freeze a new one")
    chosen = benchmark_records(records)
    if not chosen:
        raise ValueError("no test labels sampled by spread or migration to freeze")
    write_json(path, {
        "setup": setup, "dataset": dataset, "author": author or getpass.getuser(), "created_at": utc_now(),
        "description": description, "labels": [r.identity for r in sorted(chosen, key=lambda r: (r.recording, r.frame))],
    })


def freeze_benchmark(
    libraries: Libraries, dataset_ref: str, benchmark_id: str | None = None, *, author: str | None = None, description: str = "",
) -> Benchmark:
    """Freeze a dataset's spread-sampled test labels as a personal benchmark.

    The id defaults to ``<dataset id>-b<n>`` with the next free ``n``.
    """

    dataset = Dataset(libraries, dataset_ref)
    if benchmark_id is None:
        n = 1
        while benchmark_path(libraries.personal, f"{dataset.id}-b{n}").exists():
            n += 1
        benchmark_id = f"{dataset.id}-b{n}"
    write_benchmark(
        libraries.personal, benchmark_id, setup=dataset.setup, dataset=dataset_ref,
        records=labels(libraries, dataset_ref, "test"), author=author, description=description,
    )
    return get_benchmark(libraries, make_ref("mine", benchmark_id))


def get_benchmark(libraries: Libraries, ref: str) -> Benchmark:
    scope, benchmark_id = parse_ref(ref)
    data = read_json(benchmark_path(libraries.root(scope), benchmark_id))
    if data is None:
        raise LookupError(f"unknown benchmark {ref}")
    return Benchmark(
        ref=ref, setup=str(data["setup"]), dataset=str(data["dataset"]), author=str(data.get("author") or ""),
        created_at=str(data.get("created_at") or ""), description=str(data.get("description") or ""),
        entries=tuple(data["labels"]),
    )


def list_benchmarks(libraries: Libraries, setup: str | None = None) -> list[Benchmark]:
    found = []
    for scope in libraries.scopes():
        directory = libraries.root(scope) / BENCHMARKS_DIR
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
            benchmark = get_benchmark(libraries, make_ref(scope, path.stem))
            if setup is None or benchmark.setup == setup:
                found.append(benchmark)
    return found


def benchmark_labels(libraries: Libraries, ref: str) -> list[LabelRecord]:
    """The exact label revisions of a benchmark, checked against their recorded hashes."""

    benchmark = get_benchmark(libraries, ref)
    collection = Collection(libraries, benchmark.setup)
    result = []
    for entry in benchmark.entries:
        record = collection.get(entry["recording"], entry["frame"], entry["revision"], entry["scope"])
        if record.sha256 != entry["sha256"]:
            raise ValueError(f"{ref}: {record.scope}:{record.key}@{record.revision} does not match the frozen label")
        result.append(replace(record, split="test"))
    return result
