"""Small helpers for leaving a durable record of training and evaluation runs.

Every training run and every evaluation writes one JSON file named by its UTC
start time.  The record carries enough to reproduce the number later: the
git revision of the code, a fingerprint of the checkpoint (path, size,
modification time, SHA-256), and the exact validation and test membership at
the time of the run (sample id, label source, revision, save time).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from typing import Any

from .segmentation_dataset import SPLITS, SegmentationStore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_slug(iso: str | None = None) -> str:
    """A filesystem-friendly form of an ISO timestamp: ``2026-09-03T18-43-04Z``."""

    stamp = datetime.fromisoformat(iso) if iso else datetime.now(timezone.utc)
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def git_revision(root: str | Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(root), capture_output=True, text=True, check=True,
        ).stdout.strip() != ""
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def checkpoint_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    """Identify a checkpoint file well enough to match runs to it later."""

    if path is None:
        return None
    file = Path(path)
    if not file.exists():
        return {"path": str(file), "exists": False}
    digest = hashlib.sha256()
    with open(file, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    stat = file.stat()
    return {
        "path": str(file.resolve()),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "sha256": digest.hexdigest(),
    }


def split_manifest(store: SegmentationStore, splits: tuple[str, ...] = SPLITS) -> dict[str, list[dict[str, Any]]]:
    """The exact membership of each split, for the run record."""

    manifest: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        manifest[split] = [
            {
                "sample_id": record.sample_id,
                "label_source": record.label_source,
                "revision": record.revision,
                "saved_at": record.saved_at,
            }
            for record in store.records(split)
        ]
    return manifest
