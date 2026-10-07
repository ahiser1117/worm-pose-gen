"""Issues and fixes: the review loop of the Workspace page (``docs/APP_SIMPLIFICATION.md`` sections 2 and 5D).

An **issue** is a stretch of rows that needs a person's eyes.  Every row
gets zero or more *reasons*, in plain words, from the ambiguity flags
(``ambiguity.FLAG_NAMES``), from the orientation of its neighbours, and from
whether it has a pose at all:

=============  ===========================  ==========================================
code           reason shown                 from
=============  ===========================  ==========================================
head_tail      head/tail uncertain          the shorter side of a head/tail swap
                                            between consecutive frames
coiled         coiled                       self_contact, holes, area_deficit
poor_fit       mask fits poorly             low_iou, area_excess, length_deviation,
                                            fragments
jump           sudden jump                  pose_jump
leaves_view    leaves the view              edge_inside
mask_edited    mask edited, needs refit     an edited mask whose frame has no pose yet
no_pose        no pose                      a frame that was never fitted
=============  ===========================  ==========================================

A head/tail swap is found where two consecutive fitted frames match better
with their ends exchanged.  The flags cannot see it: a worm whose head and
tail were confused on a whole stretch looks clean frame by frame, so the
rows between two swaps form segments of one orientation, and at every swap
the shorter of the two segments is the suspect (both on a tie).  The whole
segment becomes the issue, so Flip on the issue turns all of it.

Rows with a reason, and rows a person placed (``pipeline.placed_rows``:
flips, picks, kept fixes), are grouped into issues: runs of such rows
closer than ``MIN_ISSUE_FRAMES`` (8) sampled frames merge into one issue,
together with the clean rows between them.  Nothing is hidden, so a short
issue on its own stays a short issue.  An issue is

- ``unreviewed`` while one of its rows with a reason is neither reviewed nor
  placed by a person,
- ``fixed`` when that is not so and a person placed one of its rows,
- ``reviewed`` otherwise (every row with a reason marked Looks OK).

Review state is the per-row fingerprint of ``app.inspection``: any change
of a row or its neighbours (a fix nearby) drops their review, while placed
rows count as fixed through their provenance, which an Undo restores.

**Refit** runs one region algorithm on an issue, chosen by the issue's
first reason in ``REFIT_ORDER``:

- no pose, mask edited, coiled, sudden jump, leaves the view: ``beam_path``,
  the pipeline's second pass (chains from both anchors with prediction,
  temporal prior, beam and border redirects, the stored poses refit, one
  path through all candidates).  These are shape and continuity problems.
- mask fits poorly: ``slow_refit``, the current poses refit under a longer
  schedule with the length prior on the anchors' length (the cure for a
  short tube or a drifting length); its path still orients by the anchors.
- head/tail uncertain alone: ``mirror``, each pose or its reversal with the
  path following the anchors, no fitting.  The shapes are right; only the
  ends are confused.

The anchors are the nearest rows outside the issue that a person placed or
that are clean with an overlap of at least 0.9 (``choose_anchors``, the
rule of ``algorithms.propose_anchors`` with the reasons above and placed
rows added), and the refit covers every row between them, so the chains
start right at their anchors.

**Relabel** turns a stretch into keyframes (``propose_keyframes``: both ends
and evenly spaced frames at most ``KEYFRAME_SPACING`` (10) apart, a stretch
of 56 frames giving 7), which are labeled elsewhere; **stitch**
(``algorithms.stitch``) pins the labeled poses and refits every gap between
consecutive keyframes with those two as fixed anchors.

Refit and stitch run as jobs (``fix_command``) and write a **preview**
under ``<workspace>/fixes/<id>``: the poses they would install, the rows'
current poses for the before/after view, and the inputs they saw.  Keep
installs the preview as one ``edits.set_poses`` edit (provenance: the
algorithm, job ``fix:<id>``), so Undo works and the propagate stage treats
the rows as placed; Keep refuses a preview whose masks or poses changed
since it ran.  Discard deletes it.  A refit spec with ``keep`` (the refit
that follows a mask edit) is kept by the job itself as soon as it is ready,
so it lands even when no browser is waiting.  Flip goes through
``edits.flip_orientation``.  ``fixes_list`` reads the edit log back in
plain words, each fix with its Undo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from . import algorithms, edits
from .algorithms import ANCHOR_SEARCH_ROWS, STITCH, CandidatePose, CandidateSet, Keyframe, Progress
from .ambiguity import FLAG_NAMES
from .pipeline import placed_rows
from .workspace import _write_json_atomic, utc_now


# Plain words per reason code, in the order an issue lists them.
REASONS = {
    "head_tail": "head/tail uncertain",
    "coiled": "coiled",
    "poor_fit": "mask fits poorly",
    "jump": "sudden jump",
    "leaves_view": "leaves the view",
    "mask_edited": "mask edited, needs refit",
    "no_pose": "no pose",
}
# The reason each ambiguity flag gives.
FLAG_REASONS = {
    "low_iou": "poor_fit", "area_excess": "poor_fit", "length_deviation": "poor_fit", "fragments": "poor_fit",
    "self_contact": "coiled", "holes": "coiled", "area_deficit": "coiled",
    "pose_jump": "jump",
    "edge_inside": "leaves_view",
}
assert set(FLAG_REASONS) == set(FLAG_NAMES), "every ambiguity flag needs a reason"
# The first of an issue's reasons in this order picks its refit algorithm.
REFIT_ORDER = (
    ("no_pose", "beam_path"), ("mask_edited", "beam_path"), ("coiled", "beam_path"), ("jump", "beam_path"),
    ("leaves_view", "beam_path"), ("poor_fit", "slow_refit"), ("head_tail", "mirror"),
)
# A range with no reason (a stretch dragged on the timeline) refits like a shape problem.
DEFAULT_REFIT = "beam_path"
# What each recommended refit does, in plain words.
REFIT_LABELS = {
    "beam_path": "refit from the neighbouring frames",
    "slow_refit": "slower, more careful fit",
    "mirror": "match head/tail to the neighbouring frames",
    STITCH: "refit between the keyframes",
}
# Runs of issue rows closer than this many sampled frames are one issue.
MIN_ISSUE_FRAMES = 8
# Relabel: keyframes at most this many sampled frames apart.
KEYFRAME_SPACING = 10
# An anchor's overlap with its mask must be at least this (``algorithms.propose_anchors``).
ANCHOR_MIN_IOU = 0.9

FIXES_DIR = "fixes"
FIX_JOB_KIND = "fix"
FIX_JOB_PREFIX = "fix:"
# The pose fields a preview stores per row (``CandidatePose``).
_POSE_ARRAYS = ("centerline_xy", "latent", "width_px", "width_shape", "width_profile", "body_length_px", "points_in_fov", "crop", "energy", "soft_dice", "iou")

BoolArray = NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Reasons per row


def _consecutive(state: dict[str, np.ndarray]) -> BoolArray:
    """Per row, whether it and the row before are fitted and one workspace step apart (``algorithms.frame_step``)."""

    fitted = np.asarray(state["fitted"], dtype=bool)
    frames = np.asarray(state["frame_index"], dtype=np.int64)
    out = np.zeros(len(fitted), dtype=bool)
    if len(fitted) > 1:
        out[1:] = fitted[:-1] & fitted[1:] & (np.diff(frames) == algorithms.frame_step(frames))
    return out


def orientation_flips(state: dict[str, np.ndarray]) -> BoolArray:
    """Per row, whether its ends match the previous consecutive row's better exchanged (``algorithms.region_metrics``' flip test)."""

    curves = np.asarray(state["centerline_xy"], dtype=np.float64)
    out = _consecutive(state)
    if len(curves) > 1:
        a, b = curves[:-1], curves[1:]
        same = np.linalg.norm(b[:, 0] - a[:, 0], axis=1) + np.linalg.norm(b[:, -1] - a[:, -1], axis=1)
        swapped = np.linalg.norm(b[:, 0] - a[:, -1], axis=1) + np.linalg.norm(b[:, -1] - a[:, 0], axis=1)
        out[1:] &= swapped < same
    return out


def head_tail_rows(state: dict[str, np.ndarray]) -> BoolArray:
    """Rows on the shorter side of a head/tail swap: segments of one orientation, cut at swaps and at breaks in the track."""

    flips = orientation_flips(state)
    segment = np.cumsum(~_consecutive(state) | flips) - 1
    lengths = np.bincount(segment) if len(segment) else np.zeros(0, dtype=np.int64)
    suspects: set[int] = set()
    for row in np.flatnonzero(flips):
        left, right = int(segment[row - 1]), int(segment[row])
        if lengths[left] <= lengths[right]:
            suspects.add(left)
        if lengths[right] <= lengths[left]:
            suspects.add(right)
    return np.isin(segment, sorted(suspects)) & np.asarray(state["fitted"], dtype=bool)


def row_reasons(state: dict[str, np.ndarray]) -> dict[str, BoolArray]:
    """Per reason code (``REASONS``), the rows that have it."""

    n = int(len(state["frame_index"]))
    fitted = np.asarray(state["fitted"], dtype=bool)
    stale = np.asarray(state.get("mask_stale", np.zeros(n, dtype=bool)), dtype=bool)
    out = {code: np.zeros(n, dtype=bool) for code in REASONS}
    for flag, code in FLAG_REASONS.items():
        key = f"flag_{flag}"
        if key in state:
            out[code] |= np.asarray(state[key], dtype=bool) & fitted
    out["head_tail"] = head_tail_rows(state)
    out["mask_edited"] = stale & ~fitted
    out["no_pose"] = ~fitted & ~stale
    return out


def flagged_rows(reasons: dict[str, BoolArray]) -> BoolArray:
    return np.logical_or.reduce(list(reasons.values()))


def codes_of(reasons: dict[str, BoolArray], first: int, last: int) -> list[str]:
    """The reason codes present on rows ``first..last``, in ``REASONS`` order."""

    return [code for code in REASONS if reasons[code][first : last + 1].any()]


def refit_algorithm(codes: Sequence[str]) -> str:
    """The algorithm Refit runs for an issue with these reason codes (``REFIT_ORDER``)."""

    for code, algorithm in REFIT_ORDER:
        if code in codes:
            return algorithm
    return DEFAULT_REFIT


# ---------------------------------------------------------------------------
# Issues


def issue_spans(rows: BoolArray, gap: int = MIN_ISSUE_FRAMES) -> list[tuple[int, int]]:
    """Runs of true rows as inclusive ``(first, last)`` pairs, runs fewer than ``gap`` rows apart merged."""

    spans: list[tuple[int, int]] = []
    for first, last in edits._runs(rows):
        if spans and first - spans[-1][1] - 1 < gap:
            spans[-1] = (spans[-1][0], last)
        else:
            spans.append((first, last))
    return spans


def issue_report(state: dict[str, np.ndarray], placed: BoolArray, reviewed: BoolArray) -> dict[str, Any]:
    """The issues of a workspace and their counts; ``placed`` and ``reviewed`` are per-row masks (see the module docstring).

    Each issue is ``{id, frames: [f0, f1], rows: [r0, r1], frame_count,
    codes, reasons, state, refit: {algorithm, label}}``; ``id`` is
    ``"f0-f1"``.  A workspace with no pose at all is not analysed yet and
    has no issues.
    """

    frames = np.asarray(state["frame_index"], dtype=np.int64)
    fitted = np.asarray(state["fitted"], dtype=bool)
    reasons = row_reasons(state)
    flagged = flagged_rows(reasons)
    placed = np.asarray(placed, dtype=bool)
    reviewed = np.asarray(reviewed, dtype=bool)
    issues: list[dict[str, Any]] = []
    if fitted.any():
        for first, last in issue_spans(flagged | placed):
            window = slice(first, last + 1)
            codes = codes_of(reasons, first, last)
            open_rows = flagged[window] & ~placed[window] & ~reviewed[window]
            status = "unreviewed" if open_rows.any() else "fixed" if placed[window].any() else "reviewed"
            algorithm = refit_algorithm(codes)
            f0, f1 = int(frames[first]), int(frames[last])
            issues.append({
                "id": f"{f0}-{f1}", "frames": [f0, f1], "rows": [first, last], "frame_count": last - first + 1,
                "codes": codes, "reasons": [REASONS[c] for c in codes], "state": status,
                "refit": {"algorithm": algorithm, "label": REFIT_LABELS[algorithm]},
            })
    counts = {status: sum(issue["state"] == status for issue in issues) for status in ("unreviewed", "reviewed", "fixed")}
    summary = {
        "analysed": bool(fitted.any()), "issues": len(issues), **counts, "done": counts["reviewed"] + counts["fixed"],
        "frames": int(len(frames)), "frames_with_reasons": int(flagged.sum()), "frames_placed": int(placed.sum()),
    }
    return {"summary": summary, "issues": issues}


# ---------------------------------------------------------------------------
# Refit plans and keyframes


def choose_anchors(
    state: dict[str, np.ndarray], placed: BoolArray, flagged: BoolArray, first: int, last: int, *, search: int = ANCHOR_SEARCH_ROWS
) -> tuple[int | None, int | None]:
    """The nearest rows outside ``first..last`` (within ``search``) that a person placed, or that have no reason and overlap at least 0.9."""

    n = int(len(state["frame_index"]))
    fitted = np.asarray(state["fitted"], dtype=bool)
    with np.errstate(invalid="ignore"):
        clean = ~flagged & (np.asarray(state["iou"], dtype=np.float64) >= ANCHOR_MIN_IOU)
    good = fitted & (np.asarray(placed, dtype=bool) | clean)
    before = next((r for r in range(int(first) - 1, max(-1, int(first) - 1 - search), -1) if good[r]), None)
    after = next((r for r in range(int(last) + 1, min(n, int(last) + 1 + search)) if good[r]), None)
    return before, after


@dataclass
class RefitPlan:
    """What a Refit will run: the algorithm, the rows (the issue grown to its anchors), the anchors and the reasons seen there."""

    algorithm: str
    first: int
    last: int
    anchor_before: int | None
    anchor_after: int | None
    codes: list[str]


def plan_refit(state: dict[str, np.ndarray], placed: BoolArray, first: int, last: int, algorithm: str | None = None) -> RefitPlan:
    """The refit of rows ``first..last``: anchors from ``choose_anchors``, the rows between them, and the algorithm their reasons pick.

    ``algorithm`` overrides the choice (the developer's picker).
    """

    n = int(len(state["frame_index"]))
    first, last = int(first), int(last)
    if not 0 <= first <= last < n:
        raise ValueError(f"rows {first}..{last} outside 0..{n - 1}")
    reasons = row_reasons(state)
    before, after = choose_anchors(state, placed, flagged_rows(reasons), first, last)
    first = first if before is None else before + 1
    last = last if after is None else after - 1
    codes = codes_of(reasons, first, last)
    chosen = algorithms.get_algorithm(algorithm).id if algorithm else refit_algorithm(codes)
    return RefitPlan(chosen, first, last, before, after, codes)


def propose_keyframes(first: int, last: int, spacing: int = KEYFRAME_SPACING) -> list[int]:
    """Rows to label for a Relabel of ``first..last``: both ends and evenly spaced rows at most ``spacing`` apart."""

    first, last, spacing = int(first), int(last), int(spacing)
    if last < first:
        raise ValueError(f"the stretch ends ({last}) before it starts ({first})")
    if spacing < 1:
        raise ValueError("the keyframe spacing must be at least 1 frame")
    gaps = max(1, math.ceil((last - first) / spacing))
    return sorted({int(round(v)) for v in np.linspace(first, last, gaps + 1)})


# ---------------------------------------------------------------------------
# Previews


@dataclass
class Preview:
    """A refit's or stitch's result waiting for Keep or Discard.

    ``rows`` are the rows with a new pose (``poses``, ascending).
    ``watch_rows`` are those rows plus the anchors, with the pose each had
    when the job read the workspace (``watch_centerline_xy``, NaN when
    unfitted, ``watch_fitted``, ``watch_iou``): the before of the
    before/after view and what Keep checks against, with
    ``mask_revisions``.
    """

    id: str
    kind: str
    algorithm: str
    params: dict[str, Any]
    first: int
    last: int
    rows: list[int]
    poses: list[CandidatePose]
    watch_rows: list[int]
    watch_centerline_xy: np.ndarray
    watch_fitted: np.ndarray
    watch_iou: np.ndarray
    anchor_before: int | None = None
    anchor_after: int | None = None
    keyframes: list[int] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metrics_before: dict[str, Any] = field(default_factory=dict)
    mask_revisions: dict[str, str] = field(default_factory=dict)
    job: str = ""
    workspace: str = ""
    created_at: str = field(default_factory=utc_now)

    def before(self, row: int) -> tuple[np.ndarray | None, float | None]:
        """The centerline and overlap ``row`` had when the job ran (``None`` when it had no pose)."""

        k = self.watch_rows.index(int(row))
        if not bool(self.watch_fitted[k]):
            return None, None
        iou = float(self.watch_iou[k])
        return self.watch_centerline_xy[k], iou if math.isfinite(iou) else None

    def meta(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "algorithm": self.algorithm, "params": algorithms._json_safe(self.params),
            "first": self.first, "last": self.last, "anchor_before": self.anchor_before, "anchor_after": self.anchor_after,
            "keyframes": [int(r) for r in self.keyframes], "codes": list(self.codes), "metrics": algorithms._json_safe(self.metrics),
            "metrics_before": algorithms._json_safe(self.metrics_before), "mask_revisions": dict(self.mask_revisions),
            "job": self.job, "workspace": self.workspace, "created_at": self.created_at, "rows": [int(r) for r in self.rows],
        }


def fixes_dir(workspace: Any) -> Path:
    return Path(workspace.path) / FIXES_DIR


def _preview_path(workspace: Any, preview_id: str, suffix: str) -> Path:
    preview_id = str(preview_id)
    if not preview_id or "/" in preview_id or "\\" in preview_id or preview_id.startswith("."):
        raise ValueError(f"invalid preview id {preview_id!r}")
    return fixes_dir(workspace) / f"{preview_id}{suffix}"


def next_preview_id(workspace: Any) -> str:
    """``p000001``, ``p000002``, ... from a counter file beside the previews (ids are never reused)."""

    directory = fixes_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    counter = directory / "counter"
    text = counter.read_text().strip() if counter.exists() else ""
    current = (int(text) if text.isdigit() else 0) + 1
    tmp = counter.with_name("counter.tmp")
    tmp.write_text(f"{current}\n")
    os.replace(tmp, counter)
    return f"p{current:06d}"


def save_preview(workspace: Any, preview: Preview) -> None:
    arrays: dict[str, np.ndarray] = {
        "rows": np.asarray(preview.rows, dtype=np.int64),
        "watch_rows": np.asarray(preview.watch_rows, dtype=np.int64),
        "watch_centerline_xy": np.asarray(preview.watch_centerline_xy, dtype=np.float64),
        "watch_fitted": np.asarray(preview.watch_fitted, dtype=bool),
        "watch_iou": np.asarray(preview.watch_iou, dtype=np.float64),
        "source": np.asarray([p.source for p in preview.poses], dtype=algorithms.SOURCE_DTYPE),
        "start": np.asarray([p.start for p in preview.poses], dtype=algorithms.START_DTYPE),
    }
    for name in _POSE_ARRAYS:
        arrays[name] = np.asarray([getattr(p, name) for p in preview.poses])
    path = _preview_path(workspace, preview.id, ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)
    # The metadata last: a preview is listed once both files exist.
    _write_json_atomic(_preview_path(workspace, preview.id, ".json"), preview.meta())


def load_preview(workspace: Any, preview_id: str) -> Preview:
    meta_path = _preview_path(workspace, preview_id, ".json")
    if not meta_path.exists() or not meta_path.with_suffix(".npz").exists():
        raise FileNotFoundError(f"workspace {workspace.info.name} has no preview {preview_id!r}")
    meta = json.loads(meta_path.read_text())
    with np.load(meta_path.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    poses = [
        CandidatePose(
            centerline_xy=arrays["centerline_xy"][k], latent=arrays["latent"][k], width_px=float(arrays["width_px"][k]),
            width_shape=arrays["width_shape"][k], width_profile=arrays["width_profile"][k], body_length_px=float(arrays["body_length_px"][k]),
            points_in_fov=int(arrays["points_in_fov"][k]), crop=arrays["crop"][k], energy=float(arrays["energy"][k]),
            soft_dice=float(arrays["soft_dice"][k]), iou=float(arrays["iou"][k]), source=str(arrays["source"][k]), start=str(arrays["start"][k]),
        )
        for k in range(len(arrays["rows"]))
    ]
    return Preview(
        id=str(meta["id"]), kind=str(meta["kind"]), algorithm=str(meta["algorithm"]), params=dict(meta.get("params") or {}),
        first=int(meta["first"]), last=int(meta["last"]), rows=[int(r) for r in arrays["rows"].tolist()], poses=poses,
        watch_rows=[int(r) for r in arrays["watch_rows"].tolist()], watch_centerline_xy=arrays["watch_centerline_xy"],
        watch_fitted=arrays["watch_fitted"], watch_iou=arrays["watch_iou"], anchor_before=meta.get("anchor_before"),
        anchor_after=meta.get("anchor_after"), keyframes=[int(r) for r in meta.get("keyframes") or []], codes=list(meta.get("codes") or []),
        metrics=dict(meta.get("metrics") or {}), metrics_before=dict(meta.get("metrics_before") or {}),
        mask_revisions=dict(meta.get("mask_revisions") or {}), job=str(meta.get("job") or ""), workspace=str(meta.get("workspace") or ""),
        created_at=str(meta.get("created_at") or ""),
    )


def list_previews(workspace: Any) -> list[dict[str, Any]]:
    """The metadata of every stored preview, newest first."""

    directory = fixes_dir(workspace)
    out = []
    for meta_path in sorted(directory.glob("p*.json")) if directory.exists() else []:
        if meta_path.with_suffix(".npz").exists():
            out.append(json.loads(meta_path.read_text()))
    out.sort(key=lambda m: str(m.get("created_at") or ""), reverse=True)
    return out


def delete_preview(workspace: Any, preview_id: str) -> bool:
    """Remove a preview (and the keyframes it was stitched from); whether there was one."""

    found = False
    for suffix in (".json", ".npz", ".keyframes.npz"):
        path = _preview_path(workspace, preview_id, suffix)
        if path.exists():
            path.unlink()
            found = True
    return found


def preview_from(
    workspace: Any, preview_id: str, kind: str, candidate_set: CandidateSet, state: dict[str, np.ndarray], *,
    codes: Sequence[str] = (), keyframes: Sequence[int] = (), job: str = "",
) -> Preview:
    """The preview of a region result: the path's pose of every row it placed, and ``state`` (read before the run) as the before."""

    rows = sorted(candidate_set.path_by_row)
    poses = [candidate_set.chosen(row) for row in rows]  # never None: every row is on the path
    watch = sorted(set(rows) | {r for r in (candidate_set.anchor_before, candidate_set.anchor_after) if r is not None})
    index = np.asarray(watch, dtype=np.int64)
    fitted = np.asarray(state["fitted"], dtype=bool)[index]
    curves = np.asarray(state["centerline_xy"], dtype=np.float64)[index].copy()
    curves[~fitted] = np.nan
    return Preview(
        id=preview_id, kind=kind, algorithm=candidate_set.algorithm, params=dict(candidate_set.params), first=candidate_set.first,
        last=candidate_set.last, rows=rows, poses=poses, watch_rows=watch, watch_centerline_xy=curves,
        watch_fitted=fitted, watch_iou=np.asarray(state["iou"], dtype=np.float64)[index], anchor_before=candidate_set.anchor_before,
        anchor_after=candidate_set.anchor_after, keyframes=[int(r) for r in keyframes], codes=list(codes), metrics=dict(candidate_set.metrics),
        metrics_before=dict(candidate_set.metrics_before), mask_revisions=dict(candidate_set.mask_revisions), job=job,
        workspace=str(workspace.info.name),
    )


def preview_problem(workspace: Any, preview: Preview, state: dict[str, np.ndarray] | None = None) -> str | None:
    """Why the preview can no longer be kept (its masks or the poses it replaces or is anchored on changed), else ``None``."""

    try:
        algorithms.validate_mask_revisions(workspace, preview.mask_revisions, preview.rows, [preview.anchor_before, preview.anchor_after], preview.id)
    except ValueError:
        return "a mask changed on these frames since the fix ran; run it again"
    state = workspace.load_state() if state is None else state
    index = np.asarray(preview.watch_rows, dtype=np.int64)
    fitted = np.asarray(state["fitted"], dtype=bool)[index]
    curves = np.asarray(state["centerline_xy"], dtype=np.float64)[index].copy()
    curves[~fitted] = np.nan
    if not np.array_equal(fitted, preview.watch_fitted) or not np.array_equal(curves, preview.watch_centerline_xy, equal_nan=True):
        return "the poses on these frames changed since the fix ran; run it again"
    return None


def keep(workspace: Any, preview_id: str) -> edits.EditResult:
    """Install a preview's poses as one ``set_pose`` edit (job ``fix:<id>``) and delete the preview; refuses a stale one."""

    preview = load_preview(workspace, preview_id)
    if not preview.rows:
        raise ValueError(f"preview {preview_id} places no pose")
    frames = np.asarray(workspace.frame_index, dtype=np.int64)

    def check(state: dict[str, np.ndarray]) -> None:
        problem = preview_problem(workspace, preview, state)
        if problem is not None:
            raise ValueError(problem)

    fix = {
        "kind": preview.kind, "preview": preview.id, "algorithm": preview.algorithm, "codes": preview.codes,
        "anchors": [None if r is None else int(frames[r]) for r in (preview.anchor_before, preview.anchor_after)],
        "keyframes": [int(frames[r]) for r in preview.keyframes],
    }
    result = edits.set_poses(
        workspace, {row: pose.to_pose() for row, pose in zip(preview.rows, preview.poses, strict=True)},
        algorithm=preview.algorithm, job=f"{FIX_JOB_PREFIX}{preview.id}", note=_fix_title(preview.kind, preview.algorithm, len(preview.keyframes)),
        extra={"fix": fix}, check=check,
    )
    delete_preview(workspace, preview.id)
    return result


# ---------------------------------------------------------------------------
# Running a fix (inside the job)


def run_refit(
    workspace: Any, plan: RefitPlan, params: dict[str, Any] | None = None, *, preview_id: str, job: str = "",
    device: torch.device | str | None = None, progress: Progress | None = None,
) -> Preview:
    """Run a refit plan and save its preview."""

    state = workspace.load_state()
    candidate_set = algorithms.run_algorithm(
        workspace, plan.algorithm, plan.first, plan.last, params, anchor_before=plan.anchor_before, anchor_after=plan.anchor_after,
        device=device, progress=progress,
    )
    preview = preview_from(workspace, preview_id, "refit", candidate_set, state, codes=plan.codes, job=job)
    save_preview(workspace, preview)
    return preview


def run_stitch(
    workspace: Any, keyframes: Sequence[Keyframe], params: dict[str, Any] | None = None, *, preview_id: str, job: str = "",
    device: torch.device | str | None = None, progress: Progress | None = None,
) -> Preview:
    """Stitch a stretch between labeled keyframes (``algorithms.stitch``) and save its preview."""

    state = workspace.load_state()
    candidate_set = algorithms.stitch(workspace, keyframes, params, device=device, progress=progress)
    rows = sorted(int(k.row) for k in keyframes)
    reasons = row_reasons(state)
    preview = preview_from(workspace, preview_id, "stitch", candidate_set, state, codes=codes_of(reasons, rows[0], rows[-1]), keyframes=rows, job=job)
    save_preview(workspace, preview)
    return preview


def save_keyframes(workspace: Any, preview_id: str, keyframes: Sequence[Keyframe]) -> Path:
    """Write the keyframes a stitch job reads (``<id>.keyframes.npz``); validated first so a bad request fails before the job."""

    if not keyframes:
        raise ValueError("stitching needs at least one keyframe")
    rows = [int(k.row) for k in keyframes]
    if len(set(rows)) != len(rows):
        raise ValueError("two keyframes are on the same frame")
    arrays: dict[str, np.ndarray] = {"rows": np.asarray(rows, dtype=np.int64)}
    for k, keyframe in enumerate(keyframes):
        algorithms._resample_body(keyframe.centerline_xy, keyframe.width_profile, 2)  # raises on a malformed keyframe
        arrays[f"centerline_xy_{k}"] = np.asarray(keyframe.centerline_xy, dtype=np.float64)
        arrays[f"width_profile_{k}"] = np.asarray(keyframe.width_profile, dtype=np.float64)
    path = _preview_path(workspace, preview_id, ".keyframes.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_keyframes(workspace: Any, preview_id: str) -> list[Keyframe]:
    with np.load(_preview_path(workspace, preview_id, ".keyframes.npz"), allow_pickle=False) as archive:
        rows = archive["rows"].tolist()
        return [Keyframe(int(row), archive[f"centerline_xy_{k}"], archive[f"width_profile_{k}"]) for k, row in enumerate(rows)]


def fix_command(workspace_path: Path | str, spec: dict[str, Any]) -> list[str]:
    """The argv of a fix job: ``spec`` is ``{preview, kind: refit, algorithm, first, last, anchor_before, anchor_after, codes, params, keep?}`` (rows) or ``{preview, kind: stitch, params}``."""

    return [".venv/bin/python", "-m", "worm_pose_gen.fixes", "--workspace", str(workspace_path), "--run", json.dumps(algorithms._json_safe(spec))]


def run_spec(workspace: Any, spec: dict[str, Any], *, device: torch.device | str | None = None, progress: Progress | None = None) -> Preview:
    """Run a ``fix_command`` spec; the preview records the job id (``WORM_POSE_JOB_ID``)."""

    job = os.environ.get("WORM_POSE_JOB_ID", "")
    params = dict(spec.get("params") or {})
    if spec.get("kind") == "refit":
        plan = RefitPlan(str(spec["algorithm"]), int(spec["first"]), int(spec["last"]), spec.get("anchor_before"), spec.get("anchor_after"), list(spec.get("codes") or []))
        return run_refit(workspace, plan, params, preview_id=str(spec["preview"]), job=job, device=device, progress=progress)
    if spec.get("kind") == "stitch":
        keyframes = load_keyframes(workspace, str(spec["preview"]))
        return run_stitch(workspace, keyframes, params, preview_id=str(spec["preview"]), job=job, device=device, progress=progress)
    raise ValueError(f"unknown fix kind {spec.get('kind')!r}; expected refit or stitch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one fix (a refit or a stitch) over a workspace and save its preview.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run", required=True, help="JSON fix spec (fix_command)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    from .jobs import report_progress
    from .workspace import Workspace

    workspace = Workspace.open(args.workspace)
    spec = json.loads(args.run)
    report_progress(0.0, f"{spec.get('kind', 'fix')}: starting")
    preview = run_spec(workspace, spec, device=args.device, progress=report_progress)
    result = {"preview": preview.id, "kind": preview.kind, "algorithm": preview.algorithm, "rows": len(preview.rows), "metrics": algorithms._json_safe(preview.metrics)}
    if spec.get("keep"):
        # The refit after a mask edit: installed as soon as it is ready, with no one waiting to press Keep.
        result["kept"] = keep(workspace, preview.id).edit_id
        report_progress(1.0, f"{preview.kind}: kept as {result['kept']}", result)
    else:
        report_progress(1.0, f"{preview.kind}: preview {preview.id} ready", result)
    print(json.dumps(result, indent=1))
    return 0


# ---------------------------------------------------------------------------
# The fixes list


def _fix_title(kind: str, algorithm: str, keyframes: int = 0) -> str:
    if kind == STITCH:
        return f"Relabeled from {keyframes} keyframe{'s' if keyframes != 1 else ''}"
    return f"Refit: {REFIT_LABELS.get(algorithm) or algorithms.get_algorithm(algorithm).label}"  # type: ignore[attr-defined]


def _describe(entry: dict[str, Any]) -> tuple[str, str]:
    """``(kind, title)`` of an edit-log entry, in plain words."""

    kind = str(entry.get("kind"))
    payload = entry.get("payload") or {}
    fix = payload.get("fix") or {}
    if kind == "flip_orientation":
        return "flip", "Flipped head/tail"
    if kind == "set_pose" and fix:
        return ("relabel" if fix.get("kind") == STITCH else "refit"), str(payload.get("note") or "Refit")
    if kind in ("set_mask", "clear_mask"):
        return "mask", "Edited mask" if kind == "set_mask" else "Removed mask edit"
    if kind == "accept_path":
        return "refit", "Accepted another fit"
    if kind == "pick_hypothesis":
        return "pick", "Picked another fit"
    return "pose", "Placed a pose"


def _window_of(entry: dict[str, Any]) -> set[int]:
    rows = [int(r) for r in (entry.get("payload") or {}).get("rows") or []]
    return {r + d for r in rows for d in (-1, 0, 1)}


def fixes_list(workspace: Any) -> list[dict[str, Any]]:
    """The fixes in force, newest first: ``{id, kind, title, frames: [f0, f1], frame_count, time, note, undoable, blocked_by}``.

    Undone edits and the undos themselves are left out.  An edit whose
    rows (or their neighbours) a later fix in force also touched cannot be
    undone on its own: its snapshot would put back what the later fix
    changed, so ``blocked_by`` names the newest such fix, to undo first.
    """

    entries = workspace.edits()
    undone = edits._undone_ids(entries)
    live = [e for e in entries if e.get("kind") != "undo" and e["id"] not in undone]
    windows = {e["id"]: _window_of(e) for e in live}
    out = []
    for position, entry in enumerate(live):
        payload = entry.get("payload") or {}
        frames = [int(f) for f in payload.get("frames") or []]
        kind, title = _describe(entry)
        later = [e["id"] for e in live[position + 1 :] if windows[e["id"]] & windows[entry["id"]]]
        out.append({
            "id": entry["id"], "kind": kind, "title": title, "frames": [min(frames), max(frames)] if frames else None,
            "frame_count": len(frames), "time": entry.get("time"), "note": str(payload.get("note") or ""),
            "undoable": bool(payload.get("snapshot")) and not later, "blocked_by": later[-1] if later else None,
        })
    out.reverse()
    return out


def undo_fix(workspace: Any, edit_id: str) -> edits.EditResult:
    """Undo one fix of ``fixes_list``; refused while a later fix on the same frames is in force."""

    entry = next((e for e in fixes_list(workspace) if e["id"] == edit_id), None)
    if entry is None:
        raise ValueError(f"no fix {edit_id} in force")
    if entry["blocked_by"]:
        raise ValueError(f"undo {entry['blocked_by']} first: it changed the same frames after {edit_id}")
    return edits.undo(workspace, edit_id)


if __name__ == "__main__":
    raise SystemExit(main())
