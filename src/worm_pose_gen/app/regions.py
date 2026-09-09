"""Phase 3 of the app: regions between anchors, region jobs, candidate sets and the outcome log, spoken in frames.

``worm_pose_gen.algorithms`` works in workspace rows; the API works in
frames (what the viewer shows), so everything here converts on the way in
(``frame -> row`` through the workspace) and on the way out (anchors and
regions come back as ``{"row", "frame"}`` pairs or ``[first, last]`` frame
pairs).  A region job is a ``JobSpec`` of kind ``region`` whose ``params``
is the region spec ``pipeline.region_command`` turns into the argv of
``python -m worm_pose_gen.pipeline --region-run``; the job process names the
candidate set after its job id (``WORM_POSE_JOB_ID``), so the set a job made
is found under the job's id.  Accepting a set goes through
``algorithms.accept_candidates`` (one ``accept_path`` edit) and answers like
the Phase 2 edits do: the edit, the refreshed frame, a series patch and the
provenance, so the browser updates in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import algorithms, pipeline
from ..algorithms import CandidatePose, CandidateSet
from ..jobs import JobSpec
from ..pose_viewer import Segmenters, _round
from .state import NotFound
from .workspace_view import WorkspaceView

REGION_JOB_KIND = "region"


# ---------------------------------------------------------------------------
# Frames and rows


def frame_to_row(view: WorkspaceView, value: Any, name: str) -> int:
    """The row of frame ``value`` (a request field called ``name``); a missing or malformed value is a bad request."""

    if value is None or value == "":
        raise ValueError(f"'{name}' is required")
    try:
        frame = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"'{name}' must be a frame number, got {value!r}") from error
    return view.workspace.row_of(frame)


def optional_row(view: WorkspaceView, value: Any, name: str) -> int | None:
    return None if value is None or value == "" else frame_to_row(view, value, name)


def anchor_payload(view: WorkspaceView, row: int | None) -> dict[str, int] | None:
    """An anchor as the API reports it: ``{"row", "frame"}`` or ``None``."""

    if row is None:
        return None
    return {"row": int(row), "frame": int(view.workspace.frame_index[int(row)])}


def frames_of(view: WorkspaceView, first: int, last: int) -> list[int]:
    frame_index = view.workspace.frame_index
    return [int(frame_index[int(first)]), int(frame_index[int(last)])]


# ---------------------------------------------------------------------------
# Proposing a region


def propose(view: WorkspaceView, frame: int | None, first: int | None = None, last: int | None = None, *, pad: int = 2) -> dict[str, Any]:
    """The region around ``frame`` (``algorithms.propose_region``), or anchors for the frames ``first..last`` the user typed.

    Both come back as ``{"frames": [f0, f1], "rows": [a, b], "anchor_before",
    "anchor_after", "reason", "stretch"}`` with the anchors as ``{"row", "frame"}``
    pairs (``None`` when no trusted row is near enough).
    """

    view.refresh()
    workspace = view.workspace
    if first is not None or last is not None:
        if first is None or last is None:
            raise ValueError("'first' and 'last' go together")
        a, b = frame_to_row(view, first, "first"), frame_to_row(view, last, "last")
        if a > b:
            raise ValueError(f"'first' ({first}) is after 'last' ({last})")
        before, after = algorithms.propose_anchors(workspace.load_state(), a, b)
        proposal = {"first": a, "last": b, "anchor_before": before, "anchor_after": after, "reason": "anchors for the frames given", "stretch": None}
    else:
        proposal = algorithms.propose_region(workspace, frame_to_row(view, frame, "frame"), pad=pad)
    stretch = proposal.get("stretch")
    return {
        "frames": frames_of(view, proposal["first"], proposal["last"]),
        "rows": [int(proposal["first"]), int(proposal["last"])],
        "anchor_before": anchor_payload(view, proposal["anchor_before"]),
        "anchor_after": anchor_payload(view, proposal["anchor_after"]),
        "reason": str(proposal["reason"]),
        "stretch": None if stretch is None else frames_of(view, stretch[0], stretch[1]),
    }


# ---------------------------------------------------------------------------
# Region jobs


def region_spec(view: WorkspaceView, payload: dict[str, Any]) -> dict[str, Any]:
    """The region spec of a ``kind: region`` job request (frames in, rows out), validated the way the job process will.

    The algorithm must exist and its parameters resolve; the region must lie
    in the workspace with the anchors outside it on fitted rows.  The
    checks run here so a bad request fails now (a 400) rather than as a
    failed job a minute later.  The spec also records the frames of the
    region and anchors for display.
    """

    view.refresh()
    workspace = view.workspace
    algorithm = algorithms.get_algorithm(str(payload.get("algorithm") or ""))
    raw_params = payload.get("params")
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        raise ValueError("'params' must be an object of parameter values")
    params = algorithm.resolve(raw_params)  # type: ignore[attr-defined]
    first, last = frame_to_row(view, payload.get("first"), "first"), frame_to_row(view, payload.get("last"), "last")
    if first > last:
        raise ValueError(f"'first' (frame {payload.get('first')}) is after 'last' (frame {payload.get('last')})")
    before, after = optional_row(view, payload.get("anchor_before"), "anchor_before"), optional_row(view, payload.get("anchor_after"), "anchor_after")
    algorithm.check_anchors(before, after)  # type: ignore[attr-defined]
    fitted = np.asarray(workspace.load_state()["fitted"], dtype=bool)
    for name, anchor, outside in (("anchor_before", before, before is not None and before < first), ("anchor_after", after, after is not None and after > last)):
        if anchor is None:
            continue
        if not outside:
            raise ValueError(f"{name} (frame {payload.get(name)}) must lie outside the region frames {payload.get('first')}..{payload.get('last')}")
        if not bool(fitted[anchor]):
            raise ValueError(f"{name} (frame {payload.get(name)}) has no fitted pose")
    return {
        "algorithm": algorithm.id,
        "first": first,
        "last": last,
        "params": params,
        "anchor_before": before,
        "anchor_after": after,
        "frames": frames_of(view, first, last),
        "anchor_frames": {"before": None if before is None else int(payload["anchor_before"]), "after": None if after is None else int(payload["anchor_after"])},
    }


def region_job(view: WorkspaceView, payload: dict[str, Any]) -> tuple[JobSpec, list[str]]:
    """The ``JobSpec`` and argv of a region job; the candidate set takes the job's id (``WORM_POSE_JOB_ID``) since none is given here."""

    spec = region_spec(view, payload)
    frames = spec["frames"]
    label = str(payload.get("label") or f"{spec['algorithm']} on {view.name} frames {frames[0]}-{frames[1]}")
    job = JobSpec(kind=REGION_JOB_KIND, params=spec, workspace=view.name, frames=list(frames), label=label)
    return job, pipeline.region_command(view.workspace.path, spec)


# ---------------------------------------------------------------------------
# Candidate sets


def load_set(view: WorkspaceView, set_id: str) -> CandidateSet:
    try:
        return view.candidate_set(set_id)
    except FileNotFoundError as error:
        raise NotFound(str(error)) from error


def set_summary(view: WorkspaceView, entry: dict[str, Any]) -> dict[str, Any]:
    """A list entry (``CandidateSet.summary`` or ``list_candidate_sets`` row) with its anchors and frames spelled out."""

    first, last = entry.get("rows") or (None, None)
    out = dict(entry)
    if first is not None and last is not None:
        out["frames"] = frames_of(view, int(first), int(last))
    out["anchors"] = {"before": anchor_payload(view, entry.get("anchor_before")), "after": anchor_payload(view, entry.get("anchor_after"))}
    return out


def list_sets(view: WorkspaceView) -> list[dict[str, Any]]:
    view.refresh()
    return [set_summary(view, entry) for entry in algorithms.list_candidate_sets(view.workspace)]


def _candidate_payload(pose: CandidatePose, index: int, chosen: tuple[int, bool] | None) -> dict[str, Any]:
    picked = chosen is not None and chosen[0] == index
    return {
        "index": int(index),
        "centerline_xy": _round(np.round(np.asarray(pose.centerline_xy, dtype=np.float64), 2), 2),
        "energy": _round(pose.energy, 5),
        "soft_dice": _round(pose.soft_dice, 5),
        "iou": _round(pose.iou),
        "body_length_px": _round(pose.body_length_px, 1),
        "width_px": _round(pose.width_px, 2),
        "points_in_fov": int(pose.points_in_fov),
        "source": str(pose.source),
        "start": str(pose.start),
        "chosen": bool(picked),
        "mirrored": bool(picked and chosen is not None and chosen[1]),
    }


def set_payload(view: WorkspaceView, candidate_set: CandidateSet) -> dict[str, Any]:
    """The whole set: per row its candidates with the path's choice marked, the path, the set's metrics and the current state's over the same rows.

    ``per_row`` lists ``{row, frame, chosen, mirrored, chosen_centerline_xy,
    chosen_iou, candidates: [...]}`` for every row of the region (rows the
    algorithm produced nothing for have an empty list); ``chosen_iou`` per
    frame is what the timeline draws.
    """

    view.refresh()
    run = view.run
    workspace = view.workspace
    path_by_row = candidate_set.path_by_row
    per_row = []
    chosen_iou: dict[str, float | None] = {}
    for row in candidate_set.rows:
        row = int(row)
        frame = int(workspace.frame_index[row])
        choice = path_by_row.get(row)
        poses = candidate_set.candidates.get(row, [])
        chosen = candidate_set.chosen(row)
        entry = {
            "row": row,
            "frame": frame,
            "chosen": -1 if choice is None else int(choice[0]),
            "mirrored": bool(choice[1]) if choice is not None else False,
            "chosen_centerline_xy": None if chosen is None else _round(np.round(chosen.centerline_xy, 2), 2),
            "chosen_iou": None if chosen is None else _round(chosen.iou),
            "candidates": [_candidate_payload(pose, j, choice) for j, pose in enumerate(poses)],
        }
        chosen_iou[str(frame)] = entry["chosen_iou"]
        per_row.append(entry)
    current = algorithms.region_metrics(run.arrays, candidate_set.rows, workspace.image_shape)
    return {
        **set_summary(view, candidate_set.summary()),
        "recording": candidate_set.recording,
        "per_row": per_row,
        "path": [{"row": int(r), "frame": int(workspace.frame_index[int(r)]), "index": int(i), "mirrored": bool(m)} for r, i, m in candidate_set.path],
        "chosen_iou": chosen_iou,
        "current_metrics": current,
    }


def accept(view: WorkspaceView, set_id: str, payload: dict[str, Any], segmenters: Segmenters, device: torch.device) -> dict[str, Any]:
    """Accept a set's path (``algorithms.accept_candidates``) over all or some of its frames; answers like an edit does.

    ``payload`` takes ``rows`` (or ``frames``): the frames to accept, default
    every frame on the path; ``use_path`` (must be true); ``note``; and
    ``frame``, the frame whose refreshed payload to return (default the first
    accepted).  The response is the Phase 2 edit response plus the set's
    summary (now accepted) and the workspace's candidate set list.
    """

    load_set(view, set_id)  # a missing set is a 404, not a 400 from the accept
    frames = payload.get("rows", payload.get("frames"))
    rows = None
    if frames is not None:
        if not isinstance(frames, list) or not all(isinstance(f, (int, str)) and not isinstance(f, bool) for f in frames):
            raise ValueError("'rows' must be a list of frames")
        rows = [frame_to_row(view, f, "rows") for f in frames]
        if not rows:
            raise ValueError("'rows' is empty")
    use_path = payload.get("use_path", True)
    if use_path not in (True, 1, "1", "true", "True"):
        raise ValueError("only the set's path can be accepted (use_path must be true); pick other candidates per frame")
    result = algorithms.accept_candidates(view.workspace, set_id, rows=rows, use_path=True, note=str(payload.get("note") or ""))
    view.invalidate()
    frame = int(payload["frame"]) if payload.get("frame") not in (None, "") else int(view.workspace.frame_index[result.rows[0]])
    response = view.edit_response(result, frame, segmenters, device)
    response["candidate_set"] = set_summary(view, load_set(view, set_id).summary())
    response["candidate_sets"] = list_sets(view)
    return response


def discard(view: WorkspaceView, set_id: str) -> dict[str, Any]:
    """Remove a set's files; the outcome log keeps its run line."""

    load_set(view, set_id)
    removed = algorithms.delete_candidate_set(view.workspace, set_id)
    view.forget_candidate_set(set_id)
    return {"id": set_id, "removed": bool(removed), "candidate_sets": list_sets(view)}


# ---------------------------------------------------------------------------
# The outcome log


def outcomes(root: Path, workspace: str | None = None, algorithm: str | None = None) -> list[dict[str, Any]]:
    """Every region run under the workspaces root, newest first, optionally one workspace's or one algorithm's."""

    records = algorithms.outcomes(root)
    if workspace:
        records = [r for r in records if str(r.get("workspace") or "") == workspace]
    if algorithm:
        records = [r for r in records if str(r.get("algorithm") or "") == algorithm]
    return records
