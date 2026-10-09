"""Manual interventions on a workspace: place poses, flip an orientation, edit a mask, undo.

The Workspace page's fixes (``worm_pose_gen.fixes``) change a workspace only
through these edits: a kept Refit or Relabel stitch places its poses
(``set_poses``), Flip reverses the orientation of rows (``flip_orientation``),
and Edit mask saves an override mask (``set_mask``).  Every edit here

- writes the new pose into the state arrays exactly as ``pipeline.store_result``
  would (plus ``taper_asymmetry``, ``reversed`` and ``orientation_gap``),
- records provenance for the rows (``manual:flip``, ``manual:mask``, or the
  algorithm and job a caller such as a kept fix passes in),
- recomputes the ambiguity signals of the rows and their neighbours (a pose
  jump is a property of a pair of frames),
- saves a before-snapshot of every array slice it changed to
  ``<workspace>/edits/<id>.npz`` and appends one line to ``edits.jsonl``
  referencing it, so ``undo`` restores the slices and marks the edit undone
  in a new log entry (redo is out of scope).

Edits take the workspace lock the stages hold (``pipeline.workspace_lock``)
so a stage and an edit never write the same arrays at once; an edit does not
wait for a running stage, it raises ``pipeline.WorkspaceBusy`` after
``LOCK_TIMEOUT`` seconds (the browser's request would otherwise hang for the
stage's duration and then apply the user's intention to arrays the stage
rewrote).  ``pose_from_hypothesis`` turns a stored hypothesis into the pose
fields an edit writes, for the pipeline's own passes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .ambiguity import compute_ambiguity, inside_camera
from .latent import encode_centerline
from .mask_fit import max_bend_widths, taper_asymmetry
from .pipeline import (
    SOURCE_CODES,
    INDEPENDENT_COPIES,
    effective_mask_statistics,
    workspace_arrays,
    workspace_image_shape,
    workspace_lock,
    workspace_setup,
)


EDIT_KINDS = ("flip_orientation", "set_pose", "set_mask", "clear_mask", "undo")
FLIP_ALGORITHM = "manual:flip"
EDITS_DIR = "edits"
# The pose fields ``pose_from_hypothesis`` returns and ``set_poses`` accepts.
POSE_FIELDS = (
    "latent", "width_px", "width_shape", "width_profile", "centerline_xy", "body_length_px", "points_in_fov", "crop",
    "iou", "energy", "total_energy", "source", "best_start",
)
# What the per-row before/after summary of an edit records.
SUMMARY_FIELDS = ("iou", "source", "reversed", "path_index", "path_mirrored")
# The snapshot stores slices of these three dictionaries under ``<group>:<array>``.
_GROUPS = ("state", "hypotheses", "provenance")
_PROVENANCE_KEYS = ("algorithm", "job", "time")
# How long an edit waits for the workspace lock before giving up (seconds).
LOCK_TIMEOUT = 2.0

FloatArray = NDArray[np.float64]


@dataclass
class EditResult:
    """What one edit did: its log id, kind, the rows it changed and a per-row before/after summary."""

    edit_id: str
    kind: str
    rows: list[int]
    summary: dict[str, Any] = field(default_factory=dict)
    # For an undo, the id of the edit that was undone.
    undone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading and saving


@dataclass
class _Loaded:
    """A workspace's arrays with the fit configuration they were made under."""

    state: dict[str, np.ndarray]
    hypotheses: dict[str, np.ndarray]
    provenance: dict[str, np.ndarray]
    config: Any
    prior: Any
    image_shape: tuple[int, int] | None

    @property
    def n(self) -> int:
        return int(len(self.state["frame_index"]))

    def group(self, name: str) -> dict[str, np.ndarray]:
        return {"state": self.state, "hypotheses": self.hypotheses, "provenance": self.provenance}[name]


def _load(workspace: Any) -> _Loaded:
    setup = workspace_setup(workspace)
    state = workspace_arrays(workspace, setup.config)
    hypotheses = workspace.load_hypotheses()
    n = len(state["frame_index"])
    # Older hypotheses files predate the path arrays a pick writes.
    hypotheses.setdefault("path_index", np.full(n, -1, dtype=np.int64))
    hypotheses.setdefault("path_mirrored", np.zeros(n, dtype=bool))
    return _Loaded(state, hypotheses, workspace.load_provenance(), setup.config, setup.prior, workspace_image_shape(workspace))


def _save(workspace: Any, loaded: _Loaded, *, hypotheses: bool) -> None:
    workspace.save_state(loaded.state)
    if hypotheses and loaded.hypotheses:
        workspace.save_hypotheses(loaded.hypotheses)


def _row_arrays(arrays: dict[str, np.ndarray], n: int) -> list[str]:
    """The keys of the per-row arrays (first dimension ``n``); ``width_template`` and the like are skipped."""

    return [k for k, v in arrays.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n]


def _equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _check_rows(rows: Sequence[int], n: int) -> list[int]:
    out = sorted({int(r) for r in rows})
    if not out:
        raise ValueError("no rows given")
    if out[0] < 0 or out[-1] >= n:
        raise ValueError(f"rows {out[0]}..{out[-1]} outside 0..{n - 1}")
    return out


def _window(rows: Sequence[int], n: int) -> list[int]:
    """The rows plus their neighbours: the ambiguity signals an edit of ``rows`` can change."""

    window: set[int] = set()
    for row in rows:
        window.update(r for r in (row - 1, row, row + 1) if 0 <= r < n)
    return sorted(window)


# ---------------------------------------------------------------------------
# Poses


def _finite(values: np.ndarray | None) -> bool:
    return values is not None and values.size > 0 and bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def _slot(hyps: dict[str, np.ndarray], key: str, row: int, index: int) -> np.ndarray | None:
    if key not in hyps:
        return None
    return np.asarray(hyps[key][row, index])


def pose_from_hypothesis(
    arrays: dict[str, np.ndarray],
    hyps: dict[str, np.ndarray],
    row: int,
    index: int,
    mirrored: bool,
    *,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """The state fields hypothesis ``index`` of ``row`` would give the row, mirrored (traversed from the other end) on request.

    ``arrays`` is the state, ``hyps`` the hypotheses dictionary.  Fields an
    older workspace does not store are rebuilt: the latent is re-encoded from
    the centerline; width, width shape, width profile and crop come from the
    current row; the in-view count is counted against ``image_shape`` when
    given, else carried over.  Mirroring follows ``mask_fit.reverse_result``:
    centerline, width profile and width shape reversed, latent re-encoded.
    """

    row, index = int(row), int(index)
    count = int(hyps["hypotheses_count"][row]) if "hypotheses_count" in hyps else hyps["hypotheses_centerline_xy"].shape[1]
    if not 0 <= index < count:
        raise ValueError(f"row {row} has {count} hypotheses; index {index} does not exist")
    curve = np.asarray(hyps["hypotheses_centerline_xy"][row, index], dtype=np.float64)
    if not _finite(curve):
        raise ValueError(f"hypothesis {index} of row {row} has no centerline")
    coefficients = int(arrays["latent"].shape[1]) - 4
    latent = _slot(hyps, "hypotheses_latent", row, index)
    width_px = _slot(hyps, "hypotheses_width_px", row, index)
    width_shape = _slot(hyps, "hypotheses_width_shape", row, index)
    width_profile = _slot(hyps, "hypotheses_width_profile", row, index)
    body_length = _slot(hyps, "hypotheses_body_length_px", row, index)
    points_in_fov = _slot(hyps, "hypotheses_points_in_fov", row, index)
    crop = _slot(hyps, "hypotheses_crop", row, index)
    soft_dice = _slot(hyps, "hypotheses_soft_dice", row, index)
    total_energy = float(hyps["hypotheses_energy"][row, index]) if "hypotheses_energy" in hyps else float("nan")
    # The row's own profile and shape stand in for missing ones; they follow the
    # row's stored orientation, which is the hypothesis's only when the stored
    # centerline runs the same way as the candidate's (a mirrored path stores
    # the candidate reversed), so they are turned to the candidate's direction first.
    row_reversed = _row_runs_against(arrays, row, curve)
    if not _finite(width_shape) or width_shape.shape != arrays["width_shape"][row].shape:
        width_shape = np.asarray(arrays["width_shape"][row], dtype=np.float64)
        if row_reversed:
            width_shape = width_shape[::-1].copy()
    if not _finite(width_profile) or width_profile.shape != arrays["width_profile"][row].shape:
        width_profile = np.asarray(arrays["width_profile"][row], dtype=np.float64)
        if row_reversed:
            width_profile = width_profile[::-1].copy()
    if mirrored:
        curve = curve[::-1].copy()
        width_profile = width_profile[::-1].copy()
        width_shape = width_shape[::-1].copy()
        latent = None
    if not _finite(latent) or latent.shape != (coefficients + 4,):
        latent = encode_centerline(curve, coefficients)
    if crop is None or not np.any(crop):
        crop = np.asarray(arrays["crop"][row])
    if points_in_fov is None or int(points_in_fov) <= 0:
        points_in_fov = int(inside_camera(curve, image_shape).sum()) if image_shape is not None else int(arrays["points_in_fov"][row])
    source = str(hyps["hypotheses_source"][row, index]) if "hypotheses_source" in hyps else ""
    return {
        "latent": np.asarray(latent, dtype=np.float64),
        "width_px": float(width_px) if _finite(width_px) else float(arrays["width_px"][row]),
        "width_shape": np.asarray(width_shape, dtype=np.float64),
        "width_profile": np.asarray(width_profile, dtype=np.float64),
        "centerline_xy": curve,
        "body_length_px": float(body_length) if _finite(body_length) else float(latent[-3]),
        "points_in_fov": int(points_in_fov),
        "crop": np.asarray(crop, dtype=np.int64),
        "iou": float(hyps["hypotheses_iou"][row, index]) if "hypotheses_iou" in hyps else float("nan"),
        # The overlap energy without priors when stored, else the total energy (its upper bound).
        "energy": float(soft_dice) if _finite(soft_dice) else total_energy,
        "total_energy": total_energy,
        "source": int(SOURCE_CODES.get(source, 0)),
        "best_start": str(hyps["hypotheses_start"][row, index]) if "hypotheses_start" in hyps else "",
    }


def _row_runs_against(arrays: dict[str, np.ndarray], row: int, curve: np.ndarray) -> bool:
    """Whether the row's stored centerline runs the other way than ``curve`` (its ends match ``curve``'s swapped); false for an unfitted row."""

    if "fitted" in arrays and not bool(arrays["fitted"][row]):
        return False
    stored = np.asarray(arrays["centerline_xy"][row], dtype=np.float64)
    if stored.shape != curve.shape or not np.isfinite(stored).all():
        return False
    same = np.linalg.norm(stored[0] - curve[0]) + np.linalg.norm(stored[-1] - curve[-1])
    swapped = np.linalg.norm(stored[0] - curve[-1]) + np.linalg.norm(stored[-1] - curve[0])
    return bool(swapped < same)


def _complete_pose(arrays: dict[str, np.ndarray], row: int, pose: dict[str, Any]) -> dict[str, Any]:
    """A pose dictionary with every ``POSE_FIELDS`` entry, missing ones rebuilt from the centerline and the current row."""

    if "centerline_xy" not in pose:
        raise ValueError("a pose needs at least centerline_xy")
    curve = np.asarray(pose["centerline_xy"], dtype=np.float64)
    if curve.shape != tuple(arrays["centerline_xy"].shape[1:]) or not np.isfinite(curve).all():
        raise ValueError(f"centerline_xy must be finite with shape {tuple(arrays['centerline_xy'].shape[1:])}")
    coefficients = int(arrays["latent"].shape[1]) - 4
    out: dict[str, Any] = {"centerline_xy": curve}
    latent = pose.get("latent")
    out["latent"] = np.asarray(latent, dtype=np.float64) if _finite(None if latent is None else np.asarray(latent)) else encode_centerline(curve, coefficients)
    if out["latent"].shape != (coefficients + 4,):
        raise ValueError(f"latent must have shape ({coefficients + 4},)")
    for key in ("width_shape", "width_profile", "crop"):
        value = pose.get(key)
        out[key] = np.asarray(value if value is not None else arrays[key][row])
        if out[key].shape != arrays[key][row].shape:
            raise ValueError(f"{key} must have shape {arrays[key][row].shape}")
    for key, default in (("width_px", float(arrays["width_px"][row])), ("body_length_px", float(out["latent"][-3]))):
        value = pose.get(key)
        out[key] = float(value) if value is not None and math.isfinite(float(value)) else default
    value = pose.get("points_in_fov")
    out["points_in_fov"] = int(value) if value is not None else int(arrays["points_in_fov"][row])
    for key in ("iou", "energy", "total_energy"):
        value = pose.get(key)
        out[key] = float("nan") if value is None else float(value)
    out["source"] = int(pose.get("source", 0) or 0)
    out["best_start"] = str(pose.get("best_start", ""))
    for key in ("taper_asymmetry", "tube_coverage", "max_bend_widths", "reversed"):
        if key in pose:
            out[key] = pose[key]
    return out


def _write_pose(arrays: dict[str, np.ndarray], row: int, pose: dict[str, Any]) -> None:
    """Put ``pose`` into ``arrays[row]`` the way ``pipeline.store_result`` stores a fit."""

    pose = _complete_pose(arrays, row, pose)
    arrays["fitted"][row] = True
    if "mask_stale" in arrays:
        arrays["mask_stale"][row] = False
    arrays["latent"][row] = pose["latent"]
    arrays["width_px"][row] = pose["width_px"]
    arrays["width_shape"][row] = pose["width_shape"]
    arrays["width_profile"][row] = pose["width_profile"]
    arrays["centerline_xy"][row] = pose["centerline_xy"]
    arrays["taper_asymmetry"][row] = float(pose.get("taper_asymmetry", taper_asymmetry(pose["width_profile"])))
    arrays["iou"][row] = pose["iou"]
    arrays["energy"][row] = pose["energy"]
    arrays["total_energy"][row] = pose["total_energy"]
    if "field_energy" in arrays:
        # A placed pose was not scored against the body-field network's evidence.
        arrays["field_energy"][row] = np.nan
    arrays["points_in_fov"][row] = pose["points_in_fov"]
    arrays["body_length_px"][row] = pose["body_length_px"]
    arrays["crop"][row] = pose["crop"]
    arrays["source"][row] = pose["source"]
    arrays["reversed"][row] = bool(pose.get("reversed", False))
    arrays["orientation_gap"][row] = np.nan
    if "tube_coverage" in arrays:
        arrays["tube_coverage"][row] = float(pose.get("tube_coverage", float("nan")))
    if "max_bend_widths" in arrays:
        arrays["max_bend_widths"][row] = float(pose.get("max_bend_widths", max_bend_widths(pose["centerline_xy"], pose["width_px"])))
    if "best_start" in arrays:
        arrays["best_start"][row] = pose["best_start"]
    if "length_refit" in arrays:
        arrays["length_refit"][row] = False


def _reverse_row(arrays: dict[str, np.ndarray], row: int) -> None:
    """``mask_fit.reverse_result`` on a stored row: the body traversed from the other end."""

    curve = np.asarray(arrays["centerline_xy"][row], dtype=np.float64)[::-1].copy()
    arrays["centerline_xy"][row] = curve
    arrays["latent"][row] = encode_centerline(curve, int(arrays["latent"].shape[1]) - 4)
    arrays["width_profile"][row] = arrays["width_profile"][row][::-1].copy()
    arrays["width_shape"][row] = arrays["width_shape"][row][::-1].copy()
    arrays["taper_asymmetry"][row] = taper_asymmetry(arrays["width_profile"][row])
    arrays["reversed"][row] = not bool(arrays["reversed"][row])
    arrays["orientation_gap"][row] = np.nan
    if "field_energy" in arrays:
        arrays["field_energy"][row] = np.nan  # the evidence energy of the reversed body is not known


def _refresh_ambiguity(loaded: _Loaded, rows: Sequence[int]) -> list[int]:
    """Recompute the ambiguity arrays of ``rows`` and their neighbours; returns the rows rewritten.

    Only a window around the rows is computed (the pose jump of a row needs
    the row before it, so the window starts one row further back than what is
    written).  ``score_independent`` is left alone: it describes the
    independent fit that seeds the stretches, which an edit does not change.
    """

    state, n = loaded.state, loaded.n
    if "ambiguity_score" not in state or not len(rows):
        return []
    target = _window(rows, n)
    first, last = max(target[0] - 1, 0), target[-1]
    window = list(range(first, last + 1))
    sub = {k: v[window] for k, v in state.items() if k in _row_arrays(state, n)}
    prior = None if loaded.prior is None else loaded.prior.to_dict()
    computed = compute_ambiguity(sub, prior=prior, image_shape=loaded.image_shape)
    offsets = [r - first for r in target]
    for key, values in computed.items():
        if key in state and state[key].shape[0] == n:
            state[key][target] = values[offsets]
    return target


# ---------------------------------------------------------------------------
# The edit transaction: snapshot, write, log


def _edits_dir(workspace: Any) -> Path:
    return Path(workspace.path) / EDITS_DIR


def _next_edit_id(workspace: Any) -> str:
    """The id ``append_edit`` will hand out next (ids count up from the newest line of the log)."""

    newest = max((int(e["id"][1:]) for e in workspace.edits()), default=0)
    return f"e{newest + 1:06d}"


def _capture(loaded: _Loaded, window: Sequence[int]) -> dict[str, np.ndarray]:
    index = np.asarray(window, dtype=np.int64)
    return {f"{g}:{k}": loaded.group(g)[k][index].copy() for g in _GROUPS for k in _row_arrays(loaded.group(g), loaded.n)}


def _row_summary(loaded: _Loaded, row: int) -> dict[str, Any]:
    state, hyps, prov = loaded.state, loaded.hypotheses, loaded.provenance
    iou = float(state["iou"][row]) if "iou" in state else float("nan")
    return {
        "iou": None if not math.isfinite(iou) else round(iou, 4),
        "source": int(state["source"][row]) if "source" in state else 0,
        "reversed": bool(state["reversed"][row]) if "reversed" in state else False,
        "path_index": int(hyps["path_index"][row]) if "path_index" in hyps else -1,
        "path_mirrored": bool(hyps["path_mirrored"][row]) if "path_mirrored" in hyps else False,
        "algorithm": str(prov["algorithm"][row]),
        "job": str(prov["job"][row]),
    }


def _apply_provenance(workspace: Any, loaded: _Loaded, rows: Sequence[int], algorithm: str, job: str) -> None:
    now = time.time()
    workspace.set_provenance(rows, algorithm, job, now)
    index = np.asarray(rows, dtype=np.int64)
    loaded.provenance["algorithm"][index] = algorithm
    loaded.provenance["job"][index] = job
    loaded.provenance["time"][index] = now


def _restore_provenance(workspace: Any, provenance: dict[str, np.ndarray], rows: Sequence[int]) -> None:
    """Write per-row provenance back through ``set_provenance``, one call per distinct (algorithm, job, time)."""

    groups: dict[tuple[str, str, float], list[int]] = {}
    for k, row in enumerate(rows):
        stamp = float(provenance["time"][k])
        key = (str(provenance["algorithm"][k]), str(provenance["job"][k]), stamp if math.isfinite(stamp) else float("nan"))
        groups.setdefault(key, []).append(int(row))
    for (algorithm, job, stamp), group in groups.items():
        workspace.set_provenance(group, algorithm, job, stamp)


def _commit(
    workspace: Any,
    loaded: _Loaded,
    before: dict[str, np.ndarray],
    window: Sequence[int],
    rows: Sequence[int],
    kind: str,
    payload: dict[str, Any],
    summary_before: dict[int, dict[str, Any]],
    edit_id: str | None = None,
    before_save: Callable[[], None] | None = None,
) -> EditResult:
    """Save the changed slices of ``window`` as the edit's snapshot, write the arrays, log the edit under ``edit_id``.

    The provenance slices go into the snapshot all together whenever one of
    them changed: ``undo`` writes them back through ``set_provenance`` as a
    triple, and an edit that repeats the algorithm of the one before it
    changes only the job and the time.
    """

    edit_id = edit_id or _next_edit_id(workspace)
    after = _capture(loaded, window)
    changed = {key: value for key, value in before.items() if key not in after or not _equal(value, after[key])}
    if any(key.startswith("provenance:") for key in changed):
        for name in _PROVENANCE_KEYS:
            changed.setdefault(f"provenance:{name}", before[f"provenance:{name}"])
    changed["rows"] = np.asarray(window, dtype=np.int64)
    changed["edited_rows"] = np.asarray(rows, dtype=np.int64)
    directory = _edits_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / f"{edit_id}.npz"
    tmp = snapshot.with_name(f".{snapshot.name}.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **changed)
    tmp.replace(snapshot)
    if before_save is not None:
        before_save()
    _save(workspace, loaded, hypotheses=any(k.startswith("hypotheses:") for k in changed))
    frames = np.asarray(loaded.state["frame_index"], dtype=np.int64)
    summary = {
        "changes": [
            {"row": int(row), "frame": int(frames[row]), "before": summary_before[int(row)], "after": _row_summary(loaded, int(row))}
            for row in rows
        ],
        "arrays": sorted(k for k in changed if ":" in k),
    }
    record = {
        **payload,
        "rows": [int(r) for r in rows],
        "frames": [int(frames[r]) for r in rows],
        "snapshot": f"{EDITS_DIR}/{edit_id}.npz",
        "summary": summary,
    }
    logged = workspace.append_edit(kind, record)
    if logged != edit_id:
        # Another writer appended between the peek and the log (the lock makes this
        # a programming error, not a race): keep the snapshot under the id the log has.
        snapshot.rename(directory / f"{logged}.npz")
        edit_id = logged
    return EditResult(edit_id=edit_id, kind=kind, rows=[int(r) for r in rows], summary=summary)


# ---------------------------------------------------------------------------
# Public edits


def set_poses(
    workspace: Any,
    poses: dict[int, dict[str, Any]],
    *,
    algorithm: str,
    job: str,
    note: str = "",
    extra: dict[str, Any] | None = None,
    check: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> EditResult:
    """Write explicit poses (row -> the ``pose_from_hypothesis`` fields) as one edit; they are not stored hypotheses.

    ``check`` sees the state under the workspace lock before anything is
    written and raises to refuse the edit (a refit result whose inputs
    changed since it ran); ``extra`` adds fields to the log entry.
    """

    with workspace_lock(workspace, timeout=LOCK_TIMEOUT):
        loaded = _load(workspace)
        rows = _check_rows(list(poses), loaded.n)
        if check is not None:
            check(loaded.state)
        window = _window(rows, loaded.n)
        before = _capture(loaded, window)
        summary_before = {row: _row_summary(loaded, row) for row in rows}
        for row in rows:
            _write_pose(loaded.state, row, poses[row])
            if loaded.hypotheses:
                loaded.hypotheses["path_index"][row] = -1
                loaded.hypotheses["path_mirrored"][row] = False
        _refresh_ambiguity(loaded, rows)
        _apply_provenance(workspace, loaded, rows, str(algorithm), str(job))
        payload = {"algorithm": str(algorithm), "job": str(job), "note": str(note or ""), **(extra or {})}
        return _commit(workspace, loaded, before, window, rows, "set_pose", payload, summary_before)


def flip_orientation(workspace: Any, rows: Sequence[int], *, note: str = "") -> EditResult:
    """Reverse the orientation of every fitted row in ``rows``; provenance ``manual:flip``.

    Unfitted rows are skipped.  A row whose pose is a path's hypothesis has
    ``path_mirrored`` toggled so the hypotheses table keeps pointing at the pose.
    """

    with workspace_lock(workspace, timeout=LOCK_TIMEOUT):
        loaded = _load(workspace)
        wanted = _check_rows(rows, loaded.n)
        fitted = [r for r in wanted if bool(loaded.state["fitted"][r])]
        if not fitted:
            raise ValueError("none of the rows is fitted")
        window = _window(fitted, loaded.n)
        before = _capture(loaded, window)
        summary_before = {row: _row_summary(loaded, row) for row in fitted}
        for row in fitted:
            _reverse_row(loaded.state, row)
            if loaded.hypotheses and int(loaded.hypotheses["path_index"][row]) >= 0:
                loaded.hypotheses["path_mirrored"][row] = not bool(loaded.hypotheses["path_mirrored"][row])
        _refresh_ambiguity(loaded, fitted)
        edit_id = _next_edit_id(workspace)
        _apply_provenance(workspace, loaded, fitted, FLIP_ALGORITHM, f"edit:{edit_id}")
        payload = {"algorithm": FLIP_ALGORITHM, "job": f"edit:{edit_id}", "note": str(note or ""), "requested_rows": [int(r) for r in wanted]}
        return _commit(workspace, loaded, before, window, fitted, "flip_orientation", payload, summary_before, edit_id)


def set_mask(workspace: Any, row: int, labels: np.ndarray | None, *, revision: str | None = None, note: str = "") -> EditResult:
    """Reversible override edit; ignore labels are excluded from the binary geometric target.

    Clearing reveals the base mask. Corpus samples are separate explicit writes.
    """
    with workspace_lock(workspace, timeout=LOCK_TIMEOUT):
        loaded = _load(workspace)
        rows = _check_rows([row], loaded.n)
        row = rows[0]
        if hasattr(workspace, "clear_mask_cache"):
            workspace.clear_mask_cache()
        if revision is not None and revision != workspace.mask_revision(row):
            raise ValueError("mask changed since it was loaded; reload before saving")
        previous = workspace.get_override_mask(row)
        if labels is None and previous is None:
            raise ValueError("frame has no mask override to clear")
        window = _window(rows, loaded.n)
        before = _capture(loaded, window)
        before["mask:present"] = np.array(previous is not None)
        before["mask:labels"] = np.zeros((0, 0), dtype=np.uint8) if previous is None else previous
        summary_before = {row: _row_summary(loaded, row)}
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim != 2 or (workspace.image_shape is not None and labels.shape != workspace.image_shape):
                raise ValueError("override mask shape does not match frame")
            if not np.isin(labels, [0, 1, 255]).all():
                raise ValueError("override labels must be 0 background, 1 worm, or 255 ignore")
            labels = labels.astype(np.uint8)
        before["mask:after_labels"] = np.zeros((0, 0), dtype=np.uint8) if labels is None else labels
        before["mask:after_present"] = np.array(labels is not None)
        mask = workspace.get_mask(row) if labels is None else labels == 1
        loaded.state["fitted"][row] = False
        loaded.state["mask_stale"][row] = True
        # These measurements and independent baselines were scored against the
        # old mask. Keep them only in the undo snapshot, never as valid starts.
        invalid = (*POSE_FIELDS, "field_energy", "taper_asymmetry", "orientation_gap", "tube_coverage", "max_bend_widths",
                   "tube_area_px", "tube_area_visible_px", "score_independent", *(name for name, _ in INDEPENDENT_COPIES))
        for key in invalid:
            if key in loaded.state:
                loaded.state[key][row] = _blank_value(loaded.state[key].dtype, key)
        effective_mask_statistics(loaded.state, row, mask)
        for key in _row_arrays(loaded.hypotheses, loaded.n):
            loaded.hypotheses[key][row] = _blank_value(loaded.hypotheses[key].dtype, key)
        _refresh_ambiguity(loaded, rows)
        edit_id = _next_edit_id(workspace)
        stamp = time.time()
        loaded.provenance["algorithm"][row] = "manual:mask"
        loaded.provenance["job"][row] = f"edit:{edit_id}"
        loaded.provenance["time"][row] = stamp

        def write_mask() -> None:
            if labels is None:
                workspace.clear_override_mask(row)
            else:
                workspace.set_override_mask(row, labels)
            workspace.set_provenance(rows, "manual:mask", f"edit:{edit_id}", stamp)

        try:
            return _commit(workspace, loaded, before, window, rows, "clear_mask" if labels is None else "set_mask",
                           {"note": note, "algorithm": "manual:mask", "ignore_policy": "excluded_from_binary_target"},
                           summary_before, edit_id, before_save=write_mask)
        except Exception:
            # The write-ahead snapshot also survives a process crash. Recoverable
            # write errors roll back the mask and array slices before returning.
            if previous is None:
                workspace.clear_override_mask(row)
            else:
                workspace.set_override_mask(row, previous)
            index = np.asarray(window, dtype=np.int64)
            for key, values in before.items():
                group, _, array = key.partition(":")
                if group in _GROUPS:
                    _restore_slice(loaded.group(group), array, index, values)
            _save(workspace, loaded, hypotheses=True)
            _restore_provenance(workspace, {name: before[f"provenance:{name}"] for name in _PROVENANCE_KEYS}, window)
            raise


# ---------------------------------------------------------------------------
# The log and undo


def _undone_ids(entries: Sequence[dict[str, Any]]) -> set[str]:
    return {str(e["payload"].get("undoes")) for e in entries if e.get("kind") == "undo" and e.get("payload", {}).get("undoes")}


def list_edits(workspace: Any) -> list[dict[str, Any]]:
    """The edit log newest first: id, kind, time, rows (count), frames, note, undone, undoable, summary."""

    entries = workspace.edits()
    undone = _undone_ids(entries)
    out: list[dict[str, Any]] = []
    for entry in reversed(entries):
        payload = entry.get("payload") or {}
        rows = payload.get("rows") or []
        frames = payload.get("frames") or []
        kind = str(entry.get("kind", ""))
        is_undone = entry["id"] in undone
        out.append(
            {
                "id": entry["id"],
                "kind": kind,
                "time": entry.get("time"),
                "rows": len(rows),
                "frames": [int(min(frames)), int(max(frames))] if frames else None,
                "note": str(payload.get("note") or ""),
                "algorithm": payload.get("algorithm"),
                "job": payload.get("job"),
                "undone": is_undone,
                "undoable": kind != "undo" and not is_undone and bool(payload.get("snapshot")),
                "undoes": payload.get("undoes"),
                "summary": payload.get("summary") or {},
            }
        )
    return out


def undo(workspace: Any, edit_id: str | None = None) -> EditResult:
    """Restore the before-snapshot of ``edit_id`` (default: the newest edit not undone and not an undo) and log the undo.

    The snapshot's slices go back as they were, including the rows'
    provenance, so a later edit of the same rows is overwritten too: undo
    is linear when the default is used.  Undoing an undo or an edit already
    undone raises ``ValueError``.
    """

    with workspace_lock(workspace, timeout=LOCK_TIMEOUT):
        entries = workspace.edits()
        undone = _undone_ids(entries)
        by_id = {e["id"]: e for e in entries}
        if edit_id is None:
            candidates = [e for e in entries if e.get("kind") != "undo" and e["id"] not in undone and (e.get("payload") or {}).get("snapshot")]
            if not candidates:
                raise ValueError("nothing to undo")
            target = candidates[-1]
        else:
            if edit_id not in by_id:
                raise ValueError(f"no edit {edit_id}")
            target = by_id[edit_id]
            if target.get("kind") == "undo":
                raise ValueError(f"{edit_id} is an undo and cannot be undone")
            if edit_id in undone:
                raise ValueError(f"{edit_id} is already undone")
        payload = target.get("payload") or {}
        snapshot = Path(workspace.path) / str(payload.get("snapshot") or "")
        if not payload.get("snapshot") or not snapshot.exists():
            raise ValueError(f"{target['id']} has no snapshot to restore")
        with np.load(snapshot, allow_pickle=False) as archive:
            saved = {name: archive[name] for name in archive.files}
        window = [int(r) for r in saved.pop("rows").tolist()]
        rows = [int(r) for r in saved.pop("edited_rows").tolist()] if "edited_rows" in saved else window
        loaded = _load(workspace)
        before = _capture(loaded, window)
        summary_before = {row: _row_summary(loaded, row) for row in rows}
        index = np.asarray(window, dtype=np.int64)
        mask_before = saved.pop("mask:labels", None)
        mask_present = saved.pop("mask:present", None)
        saved.pop("mask:after_labels", None)
        saved.pop("mask:after_present", None)
        if mask_present is not None:
            if bool(mask_present):
                workspace.set_override_mask(rows[0], mask_before)
            else:
                workspace.clear_override_mask(rows[0])
        for key, values in saved.items():
            group_name, _, array = key.partition(":")
            _restore_slice(loaded.group(group_name), array, index, values)
        if any(k.startswith("provenance:") for k in saved):
            # Older snapshots hold only the provenance arrays that changed; the
            # rest is restored from what the rows carry now.
            provenance = {
                name: saved.get(f"provenance:{name}", loaded.provenance[name][index]) for name in _PROVENANCE_KEYS
            }
            _restore_provenance(workspace, provenance, window)
        else:
            # The snapshot predates provenance slices: at least stamp the rows as touched now.
            _apply_provenance(workspace, loaded, rows, str(loaded.provenance["algorithm"][rows[0]]), f"undo:{target['id']}")
        if mask_present is None:
            _refresh_ambiguity(loaded, rows)
        record = {"undoes": target["id"], "undone_kind": target.get("kind"), "note": f"undo {target['id']}"}
        result = _commit(workspace, loaded, before, window, rows, "undo", record, summary_before)
        result.undone = str(target["id"])
    return result


def _blank_value(dtype: np.dtype, key: str) -> Any:
    """What an empty slot of a hypotheses array holds (``pipeline.empty_hypotheses``): NaN, false, -1 for indices, 0, or an empty string."""

    if dtype.kind == "f":
        return np.nan
    if dtype.kind == "b":
        return False
    if dtype.kind in "iu":
        return -1 if key in ("hypotheses_beam", "path_index") else 0
    return ""


def _blank_like(values: np.ndarray, key: str) -> np.ndarray:
    blank = np.empty_like(values)
    blank[...] = _blank_value(values.dtype, key)
    return blank


def _restore_slice(group: dict[str, np.ndarray], array: str, index: np.ndarray, values: np.ndarray) -> None:
    """Put a saved slice back into ``group[array][index]``; a hypotheses array that grew since (more slots) takes the slice in its first slots."""

    if array not in group:
        return
    target = group[array]
    if target.shape[1:] == values.shape[1:]:
        target[index] = values.astype(target.dtype, copy=False)
        return
    if array.startswith("hypotheses_") and target.ndim >= 2 and values.ndim == target.ndim and target.shape[2:] == values.shape[2:]:
        width = min(int(values.shape[1]), int(target.shape[1]))
        target[index] = _blank_value(target.dtype, array)
        target[index, :width] = values[:, :width].astype(target.dtype, copy=False)


def edit_of_row(workspace: Any, row: int) -> dict[str, Any] | None:
    """The newest edit (not undone, not an undo) that touched ``row``, in the ``list_edits`` shape; ``None`` when none did."""

    rows_of = {e["id"]: [int(r) for r in ((e.get("payload") or {}).get("rows") or [])] for e in workspace.edits()}
    for entry in list_edits(workspace):
        if entry["kind"] == "undo" or entry["undone"]:
            continue
        if int(row) in rows_of.get(entry["id"], []):
            return entry
    return None


def edited_rows(workspace: Any, n: int) -> NDArray[np.bool_]:
    """Per row, whether a live (not undone) manual edit touched it."""

    out = np.zeros(int(n), dtype=bool)
    entries = workspace.edits()
    undone = _undone_ids(entries)
    for entry in entries:
        if entry.get("kind") == "undo" or entry["id"] in undone:
            continue
        rows = [int(r) for r in ((entry.get("payload") or {}).get("rows") or []) if 0 <= int(r) < n]
        out[rows] = True
    return out


def json_safe(result: EditResult) -> dict[str, Any]:
    """The result as JSON (NaN becomes null)."""

    return json.loads(json.dumps(result.to_dict(), default=_default), parse_constant=lambda _: None)


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
