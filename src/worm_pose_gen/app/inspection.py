"""Human review is separate from automatic quality flags and bound to inputs."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from weakref import WeakKeyDictionary

import numpy as np
from typing import Any

from fastapi import HTTPException

from .workspace_view import WorkspaceView, STAMPED_FILES
from ..pipeline import workspace_lock
from ..workspace import _write_json_atomic, utc_now

_REVIEW_LOCK = threading.RLock()
_FINGERPRINT_CACHE: WeakKeyDictionary = WeakKeyDictionary()


def revision(view: WorkspaceView) -> str:
    workspace = view.workspace
    paths = [workspace.path / name for name in STAMPED_FILES]
    paths.append(workspace.recording)
    for directory in ("masks", "overrides/masks"):
        paths.extend(sorted((workspace.path / directory).rglob("*")))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}".encode())
    return digest.hexdigest()


def row_fingerprints(view: WorkspaceView, rows: list[int]) -> dict[str, str]:
    """Content identities for reviewed rows plus their immediate quality context.

    File revision remains the request concurrency token. These durable hashes
    deliberately exclude edit-log timestamps and unrelated rows, so reviewing
    one segment survives correcting another. Neighbours cover pairwise motion,
    length and orientation signals even before derived flags are recomputed.
    """
    token = revision(view)
    cached_token, cached = _FINGERPRINT_CACHE.get(view, (None, {}))
    if cached_token != token:
        cached = {}
    missing = [row for row in rows if str(row) not in cached]
    if not missing:
        return {str(row): cached[str(row)] for row in rows}
    run, workspace = view.run, view.workspace
    recording = workspace.recording
    stat = recording.stat() if recording.exists() else None
    context = dict(
        recording=str(recording),
        recording_revision=None if stat is None else [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns],
        settings=workspace.info.settings,
        frames=workspace.info.frames, step=workspace.info.step,
        config=run.summary.get("fit_config"), checkpoint=run.summary.get("checkpoint"),
        prior=run.prior, thresholds=asdict(run.thresholds),
        cleanup=run.cleanup, threshold=run.threshold,
        image_shape=run.summary.get("image_shape"),
    )
    base = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode())
    arrays = {**run.arrays, **{"provenance:" + k: v for k, v in workspace.load_provenance().items()}}
    masks: dict[int, str] = {}
    result = {}
    for row in missing:
        digest = base.copy()
        first, last = max(0, row - 1), min(workspace.n, row + 2)
        digest.update(f"{row}:{first}:{last}".encode())
        for key, values in sorted(arrays.items()):
            values = np.asarray(values)
            value = values[first:last] if values.ndim and values.shape[0] == workspace.n else values
            digest.update(f"{key}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
        for neighbour in range(first, last):
            if neighbour not in masks:
                masks[neighbour] = workspace.mask_revision(neighbour)
            digest.update(masks[neighbour].encode())
        result[str(row)] = digest.hexdigest()
    cached.update(result)
    _FINGERPRINT_CACHE[view] = (token, cached)
    return {str(row): cached[str(row)] for row in rows}


def inspection(view: WorkspaceView) -> dict[str, Any]:
    with _REVIEW_LOCK, workspace_lock(view.workspace, timeout=0):
        return _inspection(view)


def _inspection(view: WorkspaceView) -> dict[str, Any]:
    with _REVIEW_LOCK:
        run = view.run
        token = revision(view)
        path = view.workspace.path / "human_review.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        saved_rows = [r for r in saved.get("rows", []) if isinstance(r, int) and 0 <= r < view.workspace.n]
        fingerprints = row_fingerprints(view, saved_rows)
        # Legacy global-token records cannot prove an individual row unchanged.
        reviewed = {r for r in saved_rows if saved.get("row_fingerprints", {}).get(str(r)) == fingerprints[str(r)]}
        segments: list[dict[str, Any]] = []
        counts = dict(unprocessed_frames=0, flagged_frames=0, unreviewed_flagged_frames=0, reviewed_frames=len(reviewed), attention_segments=0, stale_frames=0)
        stale_rows = run.arrays.get("mask_stale")
        for row, frame in enumerate(run.frame_index):
            fitted = bool(run.arrays["fitted"][row])
            flags = [str(k) for k, value in run._flags(row).items() if value]
            stale = stale_rows is not None and bool(stale_rows[row])
            if stale:
                flags.append("mask_changed_refit_required")
                counts["stale_frames"] += 1
            kind = "unprocessed" if not fitted else "flagged" if flags else "clean"
            if kind == "unprocessed":
                counts["unprocessed_frames"] += 1
            if kind == "flagged":
                counts["flagged_frames"] += 1
                counts["unreviewed_flagged_frames"] += int(row not in reviewed)
            if kind == "clean":
                continue
            is_reviewed = row in reviewed
            if segments and segments[-1]["last"] == row - 1 and segments[-1]["kind"] == kind and segments[-1]["reviewed"] == is_reviewed:
                segment = segments[-1]
                segment["last"] = row
                segment["frames"][1] = int(frame)
                segment["reasons"] = sorted(set(segment["reasons"] + flags))
            else:
                segments.append(dict(id=f"{row}:{kind}", first=row, last=row, frames=[int(frame), int(frame)], kind=kind, reasons=flags, reviewed=is_reviewed))
        counts["attention_segments"] = sum(not s["reviewed"] for s in segments)
        return dict(revision=token, summary=counts, segments=segments, reviewed_rows=sorted(reviewed))


def mark_reviewed(view: WorkspaceView, first: int, last: int, expected_revision: str) -> dict[str, Any]:
    with _REVIEW_LOCK, workspace_lock(view.workspace, timeout=0):
        current = _inspection(view)
        if expected_revision != current["revision"]:
            raise HTTPException(409, "Workspace changed. Refresh inspection before marking reviewed.")
        if first < 0 or last < first or last >= view.workspace.n:
            raise ValueError("Review bounds must be valid inclusive workspace rows.")
        rows = sorted(set(current["reviewed_rows"]).union(range(first, last + 1)))
        _write_json_atomic(view.workspace.path / "human_review.json", dict(revision=current["revision"], rows=rows, row_fingerprints=row_fingerprints(view, rows), reviewed_at=utc_now()))
        return _inspection(view)
