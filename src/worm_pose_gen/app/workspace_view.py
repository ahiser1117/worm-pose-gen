"""A workspace as the Workspace page sees it.

A workspace holds its per-frame arrays split across ``state.npz`` and
``hypotheses.npz``, its summary in ``summary.json``, and the masks the fit
was scored against; ``WorkspaceView`` merges them into a ``LoadedRun``
(``app/frame_view.py``) and rebuilds it whenever a job or an edit changes
the files, so the browser always sees the current state without the server
holding a stale copy.  A fix or a mask edit answers through
``edit_response``: the edit, the refreshed frame, a patch of the per-row
series, the provenance and the edit log, so the page redraws in place.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Iterable

import numpy as np
import torch

from .. import edits as edit_ops
from ..batch_fit import BatchFitConfig
from ..fixed_body import FixedBodyResult
from ..pipeline import config_from_dict, read_summary, workspace_arrays
from ..recordings import RecordingSource
from ..workspace import Workspace
from .frame_view import LoadedRun, Segmenters
from .images import data_url, mask_to_png_values

STAMPED_FILES = ("workspace.json", "state.npz", "hypotheses.npz", "provenance.npz", "summary.json", "recording_prior.json", "edits.jsonl", "fixed_body.npz")
STAMPED_DIRS = ("masks", "overrides/masks")
# The per-row series an edit can change: the pose's own numbers, the ambiguity
# signals of the row and its neighbours (a pose jump belongs to a pair of
# frames), and the path bookkeeping.  The classification and the provenance
# are added separately.
PATCHED_SERIES = (
    "fitted", "iou", "energy", "total_energy", "body_length_px", "width_px", "points_in_fov", "taper_asymmetry", "orientation_gap",
    "self_contact_px", "pose_jump_px", "length_deviation", "area_ratio", "ambiguity_score", "source", "reversed", "path_mirrored", "best_start",
    "tube_coverage", "max_bend_widths", "length_refit", "tube_area_px", "tube_area_visible_px",
)


def provenance_block(provenance: dict[str, np.ndarray], edited: np.ndarray) -> dict[str, Any]:
    """The run payload's provenance: distinct algorithm ids, a per-row index into them (-1 for none) and the manually edited rows."""

    algorithm = np.asarray(provenance["algorithm"]).astype(str)
    algorithms = sorted({str(a) for a in algorithm.tolist() if a})
    position = {name: i for i, name in enumerate(algorithms)}
    return {
        "algorithm": [str(a) for a in algorithm.tolist()],
        "job": [str(j) for j in np.asarray(provenance["job"]).astype(str).tolist()],
        "algorithms": algorithms,
        "index": [position.get(str(a), -1) for a in algorithm.tolist()],
        "edited": [int(bool(e)) for e in np.asarray(edited, dtype=bool).tolist()],
    }


def patch_window(rows: Iterable[int], n: int) -> list[int]:
    """The rows an edit of ``rows`` can have changed: the rows and their neighbours (the ambiguity signals span pairs of frames)."""

    rows = [int(r) for r in rows]
    if not rows:
        return []
    return list(range(max(min(rows) - 1, 0), min(max(rows) + 1, n - 1) + 1))


def workspace_stamp(path: Path) -> tuple[int, ...]:
    """Modification times of everything a view depends on (0 when absent)."""

    def stamp(target: Path) -> int:
        try:
            return target.stat().st_mtime_ns
        except OSError:
            return 0

    return tuple(stamp(path / name) for name in STAMPED_FILES + STAMPED_DIRS)


def workspace_run(workspace: Workspace, source: RecordingSource | None, source_error: str | None) -> LoadedRun:
    """The ``LoadedRun`` over a workspace's current arrays, summary and masks."""

    summary = dict(read_summary(workspace))
    summary["selected_checkpoint"] = workspace.info.settings.get("checkpoint")
    config = config_from_dict(summary["fit_config"]) if summary.get("fit_config") else BatchFitConfig()
    arrays = workspace_arrays(workspace, config)
    arrays.update(workspace.load_hypotheses())
    if not summary.get("frame_count"):
        summary = {**summary, "frame_count": workspace.n, "frames": list(workspace.info.frames), "step": workspace.info.step}
    if workspace.image_shape is not None:
        summary.setdefault("image_shape", list(workspace.image_shape))
    return LoadedRun(workspace.path, source, source_error, arrays=arrays, summary=summary, masks=workspace.effective_mask)


class WorkspaceView:
    """One workspace's ``LoadedRun`` and summary, rebuilt when its files change."""

    def __init__(self, workspace: Workspace, source: RecordingSource | None, source_error: str | None) -> None:
        self.workspace = workspace
        self.source = source
        self.source_error = source_error
        self._lock = threading.Lock()
        self._stamp: tuple[int, ...] | None = None
        self._run: LoadedRun | None = None
        self._summary: dict[str, Any] | None = None
        self._fixed_body: FixedBodyResult | None = None

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
            self._fixed_body = FixedBodyResult(self.workspace.path)
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

    def payload(self) -> dict[str, Any]:
        """The whole workspace for the developer tools: its info, summary, provenance and every per-frame series."""

        run = self.run
        summary = run.summary
        return {
            **self.workspace.info.to_dict(),
            "summary": self.summary(),
            "provenance_counts": self.workspace.provenance_counts(),
            "provenance": self.provenance(),
            "edits_count": len(self.workspace.edits()),
            "mask_rows": [int(r) for r in self.workspace.mask_rows().tolist()],
            "override_rows": self.workspace.override_rows(),
            "recording_readable": run.source is not None,
            "recording_error": run.source_error,
            "image_shape": None if run.image_shape is None else list(run.image_shape),
            "n_points": run.n_points,
            "threshold": run.threshold,
            "selected_checkpoint": run.segmentation_checkpoint,
            "cleanup": run.cleanup,
            "prior": run.prior,
            "thresholds": run.thresholds.__dict__,
            "stretches": [[a, b] for a, b in run.stretches],
            "propagation": summary.get("propagation"),
            "ambiguity": summary.get("ambiguity"),
            "fit_config": summary.get("fit_config"),
            "has_independent_pose": "centerline_xy_independent" in run.arrays,
            "continuity": summary.get("continuity"),
            "track_length": summary.get("track_length"),
            "series": run.series(),
        }

    def provenance(self) -> dict[str, Any]:
        """The run payload's provenance block (``provenance_block``) for the current arrays."""

        workspace = self.workspace
        return provenance_block(workspace.load_provenance(), edit_ops.edited_rows(workspace, workspace.n))

    def row_provenance(self, row: int) -> dict[str, Any]:
        provenance = self.workspace.load_provenance()
        time = float(provenance["time"][row])
        return {
            "algorithm": str(provenance["algorithm"][row]), "job": str(provenance["job"][row]), "time": None if not np.isfinite(time) else time,
            "edit": edit_ops.edit_of_row(self.workspace, row),
        }

    def edits(self) -> list[dict[str, Any]]:
        """The edit log newest first, in the ``edits.list_edits`` shape."""

        self.refresh()
        return edit_ops.list_edits(self.workspace)

    def invalidate(self) -> None:
        """Forget the cached run so the next request rebuilds it from disk (after an edit wrote the arrays)."""

        with self._lock:
            self._stamp = None
            self._run = None
            self._summary = None

    def edit_response(self, result: edit_ops.EditResult, frame: int, segmenters: Segmenters, device: torch.device) -> dict[str, Any]:
        """What the browser must update after ``result`` changed the workspace; the cached run must already be dropped.

        The response carries the ``EditResult``, the light frame payload of
        ``frame`` (its pose is the new one), a ``series_patch`` (``{series
        key: {row: value}}``) and ``flags_patch`` (``{flag name: {row:
        value}}``) for the rows the edit and its ambiguity refresh touched,
        the new provenance block, the edit list and count.
        """

        run = self.run
        rows = patch_window(result.rows, run.frame_index.shape[0])
        series = run.series()
        provenance = self.provenance()
        patch: dict[str, dict[str, Any]] = {}
        for key in (*PATCHED_SERIES, "classification"):
            values = series.get(key)
            if values is not None:
                patch[key] = {str(r): values[r] for r in rows}
        patch["mask_stale"] = {str(r): bool(run.arrays["mask_stale"][r]) for r in rows}
        patch["provenance"] = {str(r): provenance["algorithm"][r] for r in rows}
        patch["provenance_index"] = {str(r): provenance["index"][r] for r in rows}
        patch["edited"] = {str(r): provenance["edited"][r] for r in rows}
        flags = {name: {str(r): values[r] for r in rows} for name, values in series.get("flags", {}).items()}
        return {
            "edit": edit_ops.json_safe(result),
            "frame": self.frame(frame, segmenters, device, raw=False, detail="light"),
            "rows": rows,
            "frames": [int(run.frame_index[r]) for r in rows],
            "series_patch": patch,
            "flags_patch": flags,
            "provenance": provenance,
            "edits": self.edits(),
            "edits_count": len(self.workspace.edits()),
        }

    def frame(self, frame: int, segmenters: Segmenters, device: torch.device, *, raw: bool, detail: str, segment: bool = False) -> dict[str, Any]:
        """One frame's payload (``LoadedRun.frame``) with its provenance; the full detail adds the edited mask.

        A light frame (playback and scrubbing) skips every mask read; the
        stored and edited masks load when the frame rests.
        """

        if detail not in ("full", "light"):
            raise ValueError("detail must be 'full' or 'light'")
        run = self.run
        row = run.row_of(frame)
        payload = run.frame(row, segmenters, device, raw=raw, detail=detail, segment=segment)
        payload["provenance"] = self.row_provenance(row)
        payload["mask_stale"] = bool(run.arrays.get("mask_stale", np.zeros(self.workspace.n, dtype=bool))[row])
        payload["details_deferred"] = detail == "light"
        payload["fixed_body"] = self._fixed_body.frame(row) if self._fixed_body else None
        if detail == "light":
            return payload
        payload["has_stored_mask"] = self.workspace.get_mask(row) is not None
        override = self.workspace.get_override_mask(row)
        payload["has_override"] = override is not None
        payload["mask_revision"] = self.workspace.mask_revision(row)
        if override is not None:
            payload.setdefault("layers", {})["mask_override"] = data_url(mask_to_png_values(override))
            payload["layers"]["mask_final"] = data_url(np.where(override == 1, 255, 0).astype(np.uint8))
            payload["mask_final_source"] = "override"
        return payload

    def mask_payload(self, frame: int, segmenters: Segmenters, device: torch.device) -> dict[str, Any]:
        run = self.run
        row = run.row_of(frame)
        if self.source is None:
            raise ValueError(f"recording not readable: {self.source_error}")
        raw, image = self.source.corrected(frame)
        base = self.workspace.get_mask(row)
        if base is None:
            checkpoint = run.summary.get("selected_checkpoint") or (run.summary.get("checkpoint") or {}).get("path")
            probability, _ = segmenters.probability(checkpoint, image)
            if probability is not None:
                raw_mask = probability >= run.threshold
                base = run._cleaned(raw_mask, device)[2] if raw_mask.any() else raw_mask
        override = self.workspace.get_override_mask(row)
        labels = override if override is not None else np.zeros(image.shape, dtype=np.uint8) if base is None else base.astype(np.uint8)
        return {"frame": frame, "row": row, "width": image.shape[1], "height": image.shape[0],
                "image": data_url(image), "image_raw": data_url(raw), "mask": data_url(mask_to_png_values(labels)),
                "base_mask": None if base is None else data_url(mask_to_png_values(base.astype(np.uint8))),
                "has_override": override is not None, "revision": self.workspace.mask_revision(row),
                "stale": bool(run.arrays.get("mask_stale", np.zeros(self.workspace.n, dtype=bool))[row])}
