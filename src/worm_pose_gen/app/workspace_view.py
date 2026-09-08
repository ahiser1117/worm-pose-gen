"""A workspace seen through the viewer's eyes.

The viewer's ``LoadedRun`` computes series, statistics, classification and
frame layers from the ``poses.npz`` dictionary and a run summary.  A
workspace holds the same arrays split across ``state.npz`` and
``hypotheses.npz``, its summary in ``summary.json`` (or the imported run's),
and the masks the fit was scored against; ``WorkspaceView`` merges them into
a ``LoadedRun`` and rebuilds it whenever a job or an edit changes the files,
so the browser always sees the current state without the server holding a
stale copy.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Iterable

import numpy as np
import torch

from ..batch_fit import BatchFitConfig
from ..label_app import RecordingSource
from ..pipeline import config_from_dict, read_summary, workspace_arrays
from ..pose_run import cleanup_options
from ..pose_viewer import LoadedRun, Segmenters, _round, compatible_entries, run_payload
from ..workspace import Workspace

STAMPED_FILES = ("state.npz", "hypotheses.npz", "provenance.npz", "summary.json", "imported_summary.json", "recording_prior.json", "edits.jsonl")
STAMPED_DIRS = ("masks", "overrides/masks", "snapshots")


def workspace_stamp(path: Path) -> tuple[int, ...]:
    """Modification times of everything a view depends on (0 when absent)."""

    def stamp(target: Path) -> int:
        try:
            return target.stat().st_mtime_ns
        except OSError:
            return 0

    return tuple(stamp(path / name) for name in STAMPED_FILES + STAMPED_DIRS)


def workspace_run(workspace: Workspace, source: RecordingSource | None, source_error: str | None) -> LoadedRun:
    """The viewer's ``LoadedRun`` over a workspace's current arrays, summary and masks."""

    summary = read_summary(workspace)
    config = config_from_dict(summary["fit_config"]) if summary.get("fit_config") else BatchFitConfig()
    arrays = workspace_arrays(workspace, config)
    arrays.update(workspace.load_hypotheses())
    if not summary.get("frame_count"):
        summary = {**summary, "frame_count": workspace.n, "frames": list(workspace.info.frames), "step": workspace.info.step}
    if workspace.image_shape is not None:
        summary.setdefault("image_shape", list(workspace.image_shape))
    return LoadedRun(workspace.path, source, source_error, arrays=arrays, summary=summary, masks=workspace.effective_mask)


def workspace_entry(workspace: Workspace, summary: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    """A catalog row for a workspace, shaped like ``pose_viewer.run_entry`` so the browser can list both."""

    cleanup = cleanup_options(summary)
    iou = stats.get("iou") or {}
    checkpoint = summary.get("checkpoint") or {}
    return {
        "name": workspace.info.name,
        "path": str(workspace.path),
        "kind": "workspace",
        "recording": workspace.recording.stem,
        "recording_path": str(workspace.recording),
        "frames": [int(v) for v in workspace.info.frames],
        "step": int(workspace.info.step),
        "frame_count": workspace.n,
        "started_at": workspace.info.created_at,
        "preset": summary.get("preset"),
        "iou_median": _round(iou.get("median")),
        "iou_min": _round(iou.get("min")),
        "frames_below_0.9": iou.get("frames_below_0.9"),
        "mask_cleanup": ("fill" if cleanup["fill_holes"] else "no fill") + " + " + ("largest" if cleanup["largest_only"] else "all components"),
        "propagated": bool(summary.get("propagation")),
        "checkpoint_sha": (checkpoint.get("sha256") or "")[:8],
        "checkpoint_path": checkpoint.get("path"),
        "git_commit": (summary.get("git") or {}).get("commit", "")[:8],
        "imported_runs": list(workspace.info.imported_runs),
    }


class WorkspaceView:
    """One workspace's ``LoadedRun``, summary and catalog entry, rebuilt when its files change."""

    def __init__(self, workspace: Workspace, source: RecordingSource | None, source_error: str | None) -> None:
        self.workspace = workspace
        self.source = source
        self.source_error = source_error
        self._lock = threading.Lock()
        self._stamp: tuple[int, ...] | None = None
        self._run: LoadedRun | None = None
        self._summary: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.workspace.info.name

    def refresh(self) -> None:
        """Reload the arrays and summary if anything on disk changed since the last look."""

        stamp = workspace_stamp(self.workspace.path)
        with self._lock:
            if stamp == self._stamp and self._run is not None:
                return
            self.workspace = Workspace.open(self.workspace.path)
            self._run = workspace_run(self.workspace, self.source, self.source_error)
            self._summary = self.workspace.summary()
            self._stamp = stamp

    @property
    def run(self) -> LoadedRun:
        self.refresh()
        assert self._run is not None
        return self._run

    def summary(self) -> dict[str, Any]:
        self.refresh()
        assert self._summary is not None
        return self._summary

    def entry(self) -> dict[str, Any]:
        return workspace_entry(self.workspace, self.run.summary, self.summary())

    def info(self) -> dict[str, Any]:
        """``WorkspaceInfo`` plus the workspace summary: the row of the workspace list."""

        self.refresh()
        return {**self.workspace.info.to_dict(), "kind": "workspace", "summary": self.summary()}

    def payload(self, others: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """The viewer's run payload for this workspace, with its info, summary and provenance counts."""

        run = self.run
        entry = self.entry()
        provenance = self.workspace.load_provenance()
        return {
            **self.workspace.info.to_dict(),
            "kind": "workspace",
            "summary": self.summary(),
            "provenance_counts": self.workspace.provenance_counts(),
            "provenance": {"algorithm": [str(a) for a in provenance["algorithm"].tolist()], "job": [str(j) for j in provenance["job"].tolist()]},
            "mask_rows": [int(r) for r in self.workspace.mask_rows().tolist()],
            "override_rows": self.workspace.override_rows(),
            **run_payload(run, entry, compatible_entries(entry, others)),
        }

    def row_provenance(self, row: int) -> dict[str, Any]:
        provenance = self.workspace.load_provenance()
        time = float(provenance["time"][row])
        return {"algorithm": str(provenance["algorithm"][row]), "job": str(provenance["job"][row]), "time": None if not np.isfinite(time) else time}

    def frame(self, frame: int, segmenters: Segmenters, threshold: float | None, device: torch.device, *, raw: bool, detail: str) -> dict[str, Any]:
        if detail not in ("full", "light"):
            raise ValueError("detail must be 'full' or 'light'")
        run = self.run
        row = run.row_of(frame)
        payload = run.frame(row, segmenters, threshold, device, raw=raw, detail=detail)
        payload["provenance"] = self.row_provenance(row)
        payload["has_stored_mask"] = self.workspace.effective_mask(row) is not None
        return payload

    def pose(self, frame: int) -> dict[str, Any]:
        run = self.run
        try:
            row = run.row_of(frame)
        except ValueError:
            return {"present": False}
        return {"present": True, "pose": run.pose(row), "stats": run.stats(row), "provenance": self.row_provenance(row)}

    def starts(self, frame: int, segmenters: Segmenters, threshold: float | None, device: torch.device) -> dict[str, Any]:
        run = self.run
        return run.starts(run.row_of(frame), segmenters, threshold, device)
