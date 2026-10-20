"""Small helpers for leaving a durable record of pipeline runs and evaluation scripts.

Such a run writes one JSON file named by its UTC start time, carrying enough
to reproduce the number later: the git revision of the code and a
fingerprint of the checkpoint (path, size, modification time, SHA-256).
Library models record their training on the model card instead
(:mod:`model_training`).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from typing import Any


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
