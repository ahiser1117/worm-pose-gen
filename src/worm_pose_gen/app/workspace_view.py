"""A workspace seen through the viewer's eyes.

The viewer's ``LoadedRun`` computes series, statistics, classification and
frame layers from the ``poses.npz`` dictionary and a run summary.  A
workspace holds the same arrays split across ``state.npz`` and
``hypotheses.npz``, its summary in ``summary.json`` (or the imported run's),
and the masks the fit was scored against; ``WorkspaceView`` merges them into
a ``LoadedRun`` and rebuilds it whenever a job or an edit changes the files,
so the browser always sees the current state without the server holding a
stale copy.  The Phase 2 interventions (``worm_pose_gen.edits``) run through
``WorkspaceView.edit``, which drops the cached run and answers with what the
browser must redraw: the frame, a patch of the per-row series, the provenance
and the edit log.  The Phase 3 candidate sets (``worm_pose_gen.algorithms``)
live under ``<workspace>/candidates/``; the view keeps the loaded sets cached
by their metadata's modification time and adds the sets covering a frame to
its pose payload so the browser can overlay them.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Iterable

import numpy as np
import torch

from .. import algorithms
from .. import edits as edit_ops
from ..algorithms import CandidateSet
from ..batch_fit import BatchFitConfig
from ..label_app import RecordingSource
from ..pipeline import config_from_dict, read_summary, workspace_arrays
from ..pose_run import cleanup_options
from ..pose_viewer import LoadedRun, Segmenters, _round, compatible_entries, run_payload
from ..workspace import Workspace

STAMPED_FILES = ("state.npz", "hypotheses.npz", "provenance.npz", "summary.json", "imported_summary.json", "recording_prior.json", "edits.jsonl")
STAMPED_DIRS = ("masks", "overrides/masks", "snapshots")
# The per-row series an edit can change: the pose's own numbers, the ambiguity
# signals of the row and its neighbours (a pose jump belongs to a pair of
# frames), and the path bookkeeping.  The classification and the provenance
# are added separately.
PATCHED_SERIES = (
    "fitted", "iou", "energy", "total_energy", "body_length_px", "width_px", "points_in_fov", "taper_asymmetry", "orientation_gap",
    "self_contact_px", "pose_jump_px", "length_deviation", "area_ratio", "ambiguity_score", "source", "reversed", "path_mirrored", "best_start",
    "tube_coverage", "max_bend_widths", "length_refit", "tube_area_px", "tube_area_visible_px",
)
EDIT_KINDS = ("pick_hypothesis", "flip", "undo")


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
        # Candidate sets by id with the modification time of the metadata they were loaded from.
        self._sets: dict[str, tuple[int, CandidateSet]] = {}

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
        return {
            **self.workspace.info.to_dict(),
            "kind": "workspace",
            "summary": self.summary(),
            "provenance_counts": self.workspace.provenance_counts(),
            "provenance": self.provenance(),
            "edits_count": len(self.workspace.edits()),
            "mask_rows": [int(r) for r in self.workspace.mask_rows().tolist()],
            "override_rows": self.workspace.override_rows(),
            **run_payload(run, entry, compatible_entries(entry, others)),
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

    def segment(self, frame: int) -> dict[str, Any]:
        """The segment around ``frame``: its frames and rows and whether it is a propagation stretch (``edits.segment_info``).

        Answered from the loaded run's arrays and stretches (the browser asks
        for every frame it shows), not from the files.
        """

        run = self.run
        return edit_ops.segment_info(self.workspace, run.row_of(frame), state=run.arrays, stretches=edit_ops.stretches_of(run.arrays, run.summary))

    def invalidate(self) -> None:
        """Forget the cached run so the next request rebuilds it from disk (after an edit wrote the arrays)."""

        with self._lock:
            self._stamp = None
            self._run = None
            self._summary = None

    def _apply_edit(self, payload: dict[str, Any]) -> tuple[edit_ops.EditResult, int]:
        """Run the edit ``payload`` describes; returns the result and the frame to refresh for the caller."""

        run = self.run
        workspace = self.workspace
        kind = str(payload.get("kind") or "")
        note = str(payload.get("note") or "")
        if kind == "pick_hypothesis":
            frame = int(payload["frame"])
            result = edit_ops.pick_hypothesis(
                workspace, run.row_of(frame), int(payload["index"]), mirrored=bool(payload.get("mirrored", False)), note=note,
            )
        elif kind == "flip":
            if payload.get("frames") is not None:
                frames = [int(f) for f in payload["frames"]]
                if not frames:
                    raise ValueError("'frames' is empty")
                result = edit_ops.flip_orientation(workspace, [run.row_of(f) for f in frames], note=note)
                frame = int(payload.get("frame", frames[0]))
            else:
                frame = int(payload["frame"])
                scope = str(payload.get("scope") or "frame")
                if scope == "frame":
                    result = edit_ops.flip_frame(workspace, run.row_of(frame), note=note)
                elif scope == "segment":
                    result = edit_ops.flip_segment(workspace, run.row_of(frame), note=note)
                else:
                    raise ValueError("scope must be 'frame' or 'segment'")
        elif kind == "undo":
            target = payload.get("edit")
            result = edit_ops.undo(workspace, None if target in (None, "") else str(target))
            frame = int(payload["frame"]) if payload.get("frame") not in (None, "") else int(run.frame_index[result.rows[0]])
        else:
            raise ValueError(f"unknown edit kind {kind!r}; expected one of {', '.join(EDIT_KINDS)}")
        return result, frame

    def edit(self, payload: dict[str, Any], segmenters: Segmenters, device: torch.device) -> dict[str, Any]:
        """Apply one edit and report what the browser must update.

        The response carries the ``EditResult``, the light frame payload of
        the frame the edit concerned (its pose is the new one), a
        ``series_patch`` (``{series key: {row: value}}``) and ``flags_patch``
        (``{flag name: {row: value}}``) for the rows the edit and its
        ambiguity refresh touched, the new provenance block, the edit list
        and count.  The cached run is dropped first so every payload here and
        afterwards is built from the arrays the edit wrote.
        """

        result, frame = self._apply_edit(payload)
        self.invalidate()
        return self.edit_response(result, frame, segmenters, device)

    def edit_response(self, result: edit_ops.EditResult, frame: int, segmenters: Segmenters, device: torch.device) -> dict[str, Any]:
        """What the browser must update after ``result`` changed the workspace (see ``edit``); the cached run must already be dropped."""

        run = self.run
        rows = patch_window(result.rows, run.frame_index.shape[0])
        series = run.series()
        provenance = self.provenance()
        patch: dict[str, dict[str, Any]] = {}
        for key in (*PATCHED_SERIES, "classification"):
            values = series.get(key)
            if values is not None:
                patch[key] = {str(r): values[r] for r in rows}
        patch["provenance"] = {str(r): provenance["algorithm"][r] for r in rows}
        patch["provenance_index"] = {str(r): provenance["index"][r] for r in rows}
        patch["edited"] = {str(r): provenance["edited"][r] for r in rows}
        flags = {name: {str(r): values[r] for r in rows} for name, values in series.get("flags", {}).items()}
        return {
            "edit": edit_ops.json_safe(result),
            "frame": self.frame(frame, segmenters, None, device, raw=False, detail="light"),
            "rows": rows,
            "frames": [int(run.frame_index[r]) for r in rows],
            "series_patch": patch,
            "flags_patch": flags,
            "provenance": provenance,
            "edits": self.edits(),
            "edits_count": len(self.workspace.edits()),
        }

    def frame(self, frame: int, segmenters: Segmenters, threshold: float | None, device: torch.device, *, raw: bool, detail: str) -> dict[str, Any]:
        if detail not in ("full", "light"):
            raise ValueError("detail must be 'full' or 'light'")
        run = self.run
        row = run.row_of(frame)
        payload = run.frame(row, segmenters, threshold, device, raw=raw, detail=detail)
        payload["provenance"] = self.row_provenance(row)
        payload["has_stored_mask"] = self.workspace.effective_mask(row) is not None
        self._attach_candidate_sets(payload, row)
        return payload

    def pose(self, frame: int) -> dict[str, Any]:
        run = self.run
        try:
            row = run.row_of(frame)
        except ValueError:
            return {"present": False}
        payload = {"present": True, "pose": run.pose(row), "stats": run.stats(row), "provenance": self.row_provenance(row)}
        self._attach_candidate_sets(payload, row)
        return payload

    def _attach_candidate_sets(self, payload: dict[str, Any], row: int) -> None:
        """Add the pending candidate sets covering ``row`` to the payload and to its pose (when the row has one)."""

        sets = self.candidate_sets_of_row(row)
        payload["candidate_sets"] = sets
        if isinstance(payload.get("pose"), dict):
            payload["pose"]["candidate_sets"] = sets

    # ----- candidate sets (Phase 3)

    def candidate_set(self, set_id: str) -> CandidateSet:
        """A stored candidate set, reloaded only when its metadata file changed (an accept rewrites it); ``FileNotFoundError`` when absent."""

        path = algorithms.candidate_set_path(self.workspace, set_id).with_suffix(".json")
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            with self._lock:
                self._sets.pop(str(set_id), None)
            raise FileNotFoundError(f"workspace {self.name} has no candidate set {set_id!r}") from None
        with self._lock:
            cached = self._sets.get(str(set_id))
        if cached is not None and cached[0] == stamp:
            return cached[1]
        loaded = algorithms.load_candidate_set(self.workspace, set_id)
        with self._lock:
            self._sets[str(set_id)] = (stamp, loaded)
        return loaded

    def forget_candidate_set(self, set_id: str) -> None:
        with self._lock:
            self._sets.pop(str(set_id), None)

    def candidate_sets_of_row(self, row: int) -> list[dict[str, Any]]:
        """The sets covering ``row`` not accepted on it (as a whole or row by row), newest first: ``{id, algorithm, index, mirrored, centerline_xy, iou, source}`` each.

        ``index`` is the path's candidate on this row (-1 when the path
        skipped it, then ``centerline_xy`` is ``None``); ``centerline_xy`` is
        the chosen candidate as the path presents it (mirrored applied).
        """

        out = []
        for entry in algorithms.list_candidate_sets(self.workspace):
            first, last = entry.get("rows") or (None, None)
            if first is None or last is None or not int(first) <= int(row) <= int(last):
                continue
            if entry.get("accepted") or int(row) in {int(r) for r in entry.get("accepted_rows") or []}:
                continue
            try:
                candidate_set = self.candidate_set(entry["id"])
            except (FileNotFoundError, ValueError, KeyError):
                continue
            choice = candidate_set.path_by_row.get(int(row))
            chosen = candidate_set.chosen(int(row))
            out.append(
                {
                    "id": candidate_set.id,
                    "algorithm": candidate_set.algorithm,
                    "index": -1 if choice is None else int(choice[0]),
                    "mirrored": bool(choice[1]) if choice is not None else False,
                    "centerline_xy": None if chosen is None else _round(np.round(chosen.centerline_xy, 2), 2),
                    "iou": None if chosen is None else _round(chosen.iou),
                    "source": None if chosen is None else str(chosen.source),
                    "candidates": len(candidate_set.candidates.get(int(row), [])),
                }
            )
        return out

    def starts(self, frame: int, segmenters: Segmenters, threshold: float | None, device: torch.device) -> dict[str, Any]:
        run = self.run
        return run.starts(run.row_of(frame), segmenters, threshold, device)
