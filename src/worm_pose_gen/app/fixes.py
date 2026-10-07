"""The fixes of the Workspace page spoken in frames: Flip, Refit, Relabel (keyframes and stitch), previews, the fixes list and Undo.

``worm_pose_gen.fixes`` works in workspace rows; requests and responses
here carry frames (what the viewer shows), converted on the way in and out.
A refit or a stitch is a job of kind ``fix`` whose process writes a preview
named in the request's answer; the browser waits for the job, shows the
preview's before and after, and keeps or discards it.  Everything that
changes the poses (Flip, Keep, Undo) answers like the edits do
(``WorkspaceView.edit_response``: the edit, the refreshed frame, a series
patch, the provenance and the edit log) plus the fixes list, so the page
updates in place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .. import algorithms, edits, fixes
from ..algorithms import Keyframe
from ..jobs import JobSpec
from ..pipeline import placed_rows
from ..pose_viewer import Segmenters, _round
from .regions import anchor_payload, frame_to_row, frames_of
from .state import NotFound
from .workspace_view import WorkspaceView


def _rows_of(view: WorkspaceView, payload: dict[str, Any]) -> tuple[int, int]:
    """The rows of a request's ``first``..``last`` frames, or of its single ``frame``."""

    if payload.get("first") is None and payload.get("last") is None and payload.get("frame") is not None:
        row = frame_to_row(view, payload["frame"], "frame")
        return row, row
    first, last = frame_to_row(view, payload.get("first"), "first"), frame_to_row(view, payload.get("last"), "last")
    if first > last:
        raise ValueError(f"'first' (frame {payload.get('first')}) is after 'last' (frame {payload.get('last')})")
    return first, last


def _placed(view: WorkspaceView) -> np.ndarray:
    provenance = view.workspace.load_provenance()
    return placed_rows(view.workspace.load_state(), provenance["algorithm"], provenance["job"])


def _shown_frame(view: WorkspaceView, payload: dict[str, Any], result: edits.EditResult) -> int:
    return int(payload["frame"]) if payload.get("frame") not in (None, "") else int(view.workspace.frame_index[result.rows[0]])


def changed(view: WorkspaceView, result: edits.EditResult, payload: dict[str, Any], segmenters: Segmenters, device: torch.device) -> dict[str, Any]:
    """The edit response (``WorkspaceView.edit_response``) of a fix, with the fixes list."""

    view.invalidate()
    response = view.edit_response(result, _shown_frame(view, payload, result), segmenters, device)
    response["fixes"] = fixes.fixes_list(view.workspace)
    return response


# ---------------------------------------------------------------------------
# Flip


def flip(view: WorkspaceView, payload: dict[str, Any], segmenters: Segmenters, device: torch.device) -> dict[str, Any]:
    """Flip head/tail on frames ``first..last`` (an issue) or on one ``frame``; one ``flip_orientation`` edit."""

    view.refresh()
    first, last = _rows_of(view, payload)
    frames = frames_of(view, first, last)
    note = "Flip head/tail " + (f"frame {frames[0]}" if first == last else f"frames {frames[0]}-{frames[1]}")
    result = edits.flip_orientation(view.workspace, list(range(first, last + 1)), note=note)
    return changed(view, result, payload, segmenters, device)


# ---------------------------------------------------------------------------
# Refit and stitch jobs


def _plan_payload(view: WorkspaceView, plan: fixes.RefitPlan) -> dict[str, Any]:
    return {
        "algorithm": plan.algorithm, "label": fixes.REFIT_LABELS.get(plan.algorithm) or algorithms.get_algorithm(plan.algorithm).label,  # type: ignore[attr-defined]
        "frames": frames_of(view, plan.first, plan.last), "anchors": {"before": anchor_payload(view, plan.anchor_before), "after": anchor_payload(view, plan.anchor_after)},
        "reasons": [fixes.REASONS[c] for c in plan.codes], "codes": list(plan.codes),
    }


def refit_job(view: WorkspaceView, payload: dict[str, Any]) -> tuple[JobSpec, list[str], dict[str, Any]]:
    """The job of a Refit of frames ``first..last``: the plan (``fixes.plan_refit``), validated now so a bad request is a 400.

    ``algorithm`` and ``params`` are the developer's overrides of the
    recommended algorithm and its defaults.  Returns the job, its argv and
    ``{preview, plan}`` for the answer.
    """

    view.refresh()
    first, last = _rows_of(view, payload)
    raw_params = payload.get("params") or {}
    if not isinstance(raw_params, dict):
        raise ValueError("'params' must be an object of parameter values")
    plan = fixes.plan_refit(view.workspace.load_state(), _placed(view), first, last, str(payload.get("algorithm") or "") or None)
    algorithm = algorithms.get_algorithm(plan.algorithm)
    params = algorithm.resolve(raw_params)  # type: ignore[attr-defined]
    algorithm.check_anchors(plan.anchor_before, plan.anchor_after)  # type: ignore[attr-defined]
    preview = fixes.next_preview_id(view.workspace)
    spec = {
        "preview": preview, "kind": "refit", "algorithm": plan.algorithm, "first": plan.first, "last": plan.last,
        "anchor_before": plan.anchor_before, "anchor_after": plan.anchor_after, "codes": plan.codes, "params": params,
    }
    frames = frames_of(view, plan.first, plan.last)
    job = JobSpec(kind=fixes.FIX_JOB_KIND, params=spec, workspace=view.name, frames=frames, label=f"Refit frames {frames[0]}-{frames[1]} ({plan.algorithm})")
    return job, fixes.fix_command(view.workspace.path, spec), {"preview": preview, "plan": _plan_payload(view, plan)}


def keyframes(view: WorkspaceView, first: Any, last: Any, spacing: int | None = None) -> dict[str, Any]:
    """Relabel's keyframes for frames ``first..last``: ``{frames, spacing}`` (``fixes.propose_keyframes``)."""

    view.refresh()
    a, b = _rows_of(view, {"first": first, "last": last})
    spacing = fixes.KEYFRAME_SPACING if spacing is None else int(spacing)
    rows = fixes.propose_keyframes(a, b, spacing)
    return {"frames": [int(view.workspace.frame_index[r]) for r in rows], "spacing": spacing}


def stitch_job(view: WorkspaceView, payload: dict[str, Any]) -> tuple[JobSpec, list[str], dict[str, Any]]:
    """The job of a stitch: ``keyframes`` is a list of ``{frame, centerline_xy, width_profile}``, written beside the preview for the job.

    ``centerline_xy`` is the label's body fit, head first, ``[[x, y], ...]``
    in image pixels; ``width_profile`` its diameter at each point.
    """

    view.refresh()
    given = payload.get("keyframes")
    if not isinstance(given, list) or not given:
        raise ValueError("'keyframes' must be a non-empty list of {frame, centerline_xy, width_profile}")
    parsed = []
    for k, item in enumerate(given):
        if not isinstance(item, dict) or item.get("centerline_xy") is None or item.get("width_profile") is None:
            raise ValueError(f"keyframe {k} needs frame, centerline_xy and width_profile")
        try:
            centerline = np.asarray(item["centerline_xy"], dtype=np.float64)
            profile = np.asarray(item["width_profile"], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"keyframe {k}: centerline_xy and width_profile must be numbers") from error
        parsed.append(Keyframe(frame_to_row(view, item.get("frame"), f"keyframes[{k}].frame"), centerline, profile))
    raw_params = payload.get("params") or {}
    if not isinstance(raw_params, dict):
        raise ValueError("'params' must be an object of parameter values")
    params = algorithms.get_algorithm("beam_path").resolve(raw_params)  # type: ignore[attr-defined]
    preview = fixes.next_preview_id(view.workspace)
    fixes.save_keyframes(view.workspace, preview, parsed)
    rows = sorted(k.row for k in parsed)
    spec = {"preview": preview, "kind": "stitch", "params": params}
    frames = frames_of(view, rows[0], rows[-1])
    job = JobSpec(kind=fixes.FIX_JOB_KIND, params=spec, workspace=view.name, frames=frames, label=f"Stitch frames {frames[0]}-{frames[1]} from {len(rows)} keyframes")
    plan = {
        "algorithm": algorithms.STITCH, "label": fixes.REFIT_LABELS[algorithms.STITCH], "frames": frames,
        "keyframes": [int(view.workspace.frame_index[r]) for r in rows], "anchors": {"before": None, "after": None},
    }
    return job, fixes.fix_command(view.workspace.path, spec), {"preview": preview, "plan": plan}


# ---------------------------------------------------------------------------
# Previews


def _load(view: WorkspaceView, preview_id: str) -> fixes.Preview:
    try:
        return fixes.load_preview(view.workspace, preview_id)
    except FileNotFoundError as error:
        raise NotFound(str(error)) from error


def _curve(curve: np.ndarray | None) -> list | None:
    return None if curve is None else _round(np.round(np.asarray(curve, dtype=np.float64), 2), 2)


def preview_summary(view: WorkspaceView, meta: dict[str, Any]) -> dict[str, Any]:
    """A preview's list entry: what ran where, its metrics, and why it cannot be kept any more (``stale``, else ``None``)."""

    frame_index = view.workspace.frame_index
    algorithm = str(meta["algorithm"])
    label = fixes.REFIT_LABELS.get(algorithm) or algorithms.get_algorithm(algorithm).label  # type: ignore[attr-defined]
    return {
        "id": meta["id"], "kind": meta["kind"], "algorithm": algorithm, "label": label, "job": meta.get("job") or None,
        "created_at": meta.get("created_at"), "frames": frames_of(view, int(meta["first"]), int(meta["last"])),
        "anchors": {"before": anchor_payload(view, meta.get("anchor_before")), "after": anchor_payload(view, meta.get("anchor_after"))},
        "keyframes": [int(frame_index[r]) for r in meta.get("keyframes") or []],
        "reasons": [fixes.REASONS[c] for c in meta.get("codes") or [] if c in fixes.REASONS],
        "frames_placed": len(meta.get("rows") or []), "metrics_before": meta.get("metrics_before") or {}, "metrics_after": meta.get("metrics") or {},
    }


def list_previews(view: WorkspaceView) -> list[dict[str, Any]]:
    view.refresh()
    return [preview_summary(view, meta) for meta in fixes.list_previews(view.workspace)]


def preview_payload(view: WorkspaceView, preview_id: str) -> dict[str, Any]:
    """The whole preview: its summary, ``stale`` (a reason it cannot be kept, or ``None``) and ``per_frame`` before/after poses.

    ``per_frame`` lists ``{frame, before: {centerline_xy, iou} | None,
    after: {centerline_xy, iou}}`` for every frame the fix places, head
    first in image pixels.
    """

    view.refresh()
    preview = _load(view, preview_id)
    frame_index = view.workspace.frame_index
    per_frame = []
    for row, pose in zip(preview.rows, preview.poses, strict=True):
        curve, iou = preview.before(row)
        per_frame.append({
            "frame": int(frame_index[row]),
            "before": None if curve is None else {"centerline_xy": _curve(curve), "iou": _round(iou)},
            "after": {"centerline_xy": _curve(pose.centerline_xy), "iou": _round(pose.iou)},
        })
    return {**preview_summary(view, preview.meta()), "stale": fixes.preview_problem(view.workspace, preview), "per_frame": per_frame}


def keep(view: WorkspaceView, preview_id: str, payload: dict[str, Any], segmenters: Segmenters, device: torch.device) -> dict[str, Any]:
    """Keep: install the preview as one edit (``fixes.keep``) and answer like an edit."""

    _load(view, preview_id)  # a missing preview is a 404, not a 400 from the keep
    result = fixes.keep(view.workspace, preview_id)
    response = changed(view, result, payload, segmenters, device)
    response["previews"] = list_previews(view)
    return response


def discard(view: WorkspaceView, preview_id: str) -> dict[str, Any]:
    _load(view, preview_id)
    fixes.delete_preview(view.workspace, preview_id)
    return {"id": preview_id, "removed": True, "previews": list_previews(view)}


# ---------------------------------------------------------------------------
# The fixes list


def fixes_payload(view: WorkspaceView) -> dict[str, Any]:
    view.refresh()
    entries = fixes.fixes_list(view.workspace)
    return {"fixes": entries, "count": len(entries)}


def undo(view: WorkspaceView, edit_id: str, payload: dict[str, Any], segmenters: Segmenters, device: torch.device) -> dict[str, Any]:
    view.refresh()
    result = fixes.undo_fix(view.workspace, edit_id)
    return changed(view, result, payload, segmenters, device)
