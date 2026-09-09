"""Manual interventions on a workspace: pick a hypothesis, flip an orientation, undo.

Phase 2 of ``docs/APP_PLAN.md``: the pipeline stores candidates per frame
(``hypotheses_*`` arrays, ``pipeline.HYPOTHESIS_ARRAYS``) and one path
through them; the user corrects a frame by choosing another candidate, as it
is or mirrored, or by reversing the orientation of a frame or of the whole
propagation stretch around it.  Every edit here

- writes the new pose into the state arrays exactly as ``pipeline.store_result``
  would (plus ``taper_asymmetry``, ``reversed`` and ``orientation_gap``),
- records provenance for the rows (``manual:pick``, ``manual:flip``, or the
  algorithm and job a caller such as a Phase 3 region run passes in),
- recomputes the ambiguity signals of the rows and their neighbours (a pose
  jump is a property of a pair of frames),
- saves a before-snapshot of every array slice it changed to
  ``<workspace>/edits/<id>.npz`` and appends one line to ``edits.jsonl``
  referencing it, so ``undo`` restores the slices and marks the edit undone
  in a new log entry (redo is out of scope).

Older workspaces store only the candidates' centerlines and scores; a pick
there rebuilds the pose from the centerline (latent re-encoded, the current
row's widths and crop carried over).  Edits take the workspace lock the
stages hold (``pipeline.workspace_lock``) so a stage and an edit never write
the same arrays at once; an edit does not wait for a running stage, it raises
``pipeline.WorkspaceBusy`` after ``LOCK_TIMEOUT`` seconds (the browser's
request would otherwise hang for the stage's duration and then apply the
user's intention to arrays the stage rewrote).
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
    read_summary,
    workspace_arrays,
    workspace_image_shape,
    workspace_lock,
    workspace_setup,
)


EDIT_KINDS = ("pick_hypothesis", "flip_orientation", "accept_path", "set_pose", "undo")
PICK_ALGORITHM = "manual:pick"
FLIP_ALGORITHM = "manual:flip"
EDITS_DIR = "edits"
# The pose fields ``pose_from_hypothesis`` returns and ``set_pose`` accepts.
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
# Segments


def _stretches(workspace: Any, state: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    """The propagation stretches of a workspace (``stretches_of`` with its summary)."""

    return stretches_of(state, read_summary(workspace))


def stretches_of(state: dict[str, np.ndarray], summary: dict[str, Any]) -> list[tuple[int, int]]:
    """The propagation stretches: from the summary when it has them, else the runs of non-zero ``source`` in the state."""

    stored = (summary.get("propagation") or {}).get("stretches") or []
    stretches = [(int(a), int(b)) for a, b in stored]
    if stretches:
        return stretches
    source = np.asarray(state.get("source", np.zeros(len(state["frame_index"]), dtype=np.int8))) != 0
    return _runs(source)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive true entries as inclusive ``(first, last)`` pairs."""

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for row, on in enumerate(np.asarray(flags, dtype=bool).tolist() + [False]):
        if on and start is None:
            start = row
        elif not on and start is not None:
            runs.append((start, row - 1))
            start = None
    return runs


def segment_info(workspace: Any, row: int, state: dict[str, np.ndarray] | None = None, stretches: Sequence[tuple[int, int]] | None = None) -> dict[str, Any]:
    """The segment of ``row`` as the API reports it: frames, rows and whether it is a propagation stretch.

    ``state`` (the workspace's arrays, or any dictionary holding
    ``frame_index``, ``fitted`` and ``source``) and ``stretches`` spare the
    disk when the caller already has them loaded (the app answers this for
    every frame shown).
    """

    if state is None:
        state = workspace.load_state()
    n = int(len(state["frame_index"]))
    if not 0 <= int(row) < n:
        raise ValueError(f"row {row} outside 0..{n - 1}")
    if stretches is None:
        stretches = _stretches(workspace, state)
    a, b = _segment_of(int(row), n, state, stretches)
    in_stretch = any(s <= row <= e for s, e in stretches)
    frames = np.asarray(state["frame_index"], dtype=np.int64)
    return {"frames": [int(frames[a]), int(frames[b])], "rows": [a, b], "in_stretch": in_stretch}


def segment_of(workspace: Any, row: int) -> tuple[int, int]:
    """The propagation stretch containing ``row``, else the run of consecutive fitted rows around it that lies outside every stretch.

    Rows are inclusive.  An unfitted row outside every stretch is its own segment.
    """

    rows = segment_info(workspace, row)["rows"]
    return int(rows[0]), int(rows[1])


def _segment_of(row: int, n: int, state: dict[str, np.ndarray], stretches: Sequence[tuple[int, int]]) -> tuple[int, int]:
    for a, b in stretches:
        if a <= row <= b:
            return a, b
    fitted = np.asarray(state.get("fitted", np.ones(n, dtype=bool)), dtype=bool).copy()
    for a, b in stretches:
        fitted[max(a, 0) : min(b, n - 1) + 1] = False
    if not fitted[row]:
        return row, row
    a = b = row
    while a > 0 and fitted[a - 1]:
        a -= 1
    while b < n - 1 and fitted[b + 1]:
        b += 1
    return a, b


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
    arrays["latent"][row] = pose["latent"]
    arrays["width_px"][row] = pose["width_px"]
    arrays["width_shape"][row] = pose["width_shape"]
    arrays["width_profile"][row] = pose["width_profile"]
    arrays["centerline_xy"][row] = pose["centerline_xy"]
    arrays["taper_asymmetry"][row] = float(pose.get("taper_asymmetry", taper_asymmetry(pose["width_profile"])))
    arrays["iou"][row] = pose["iou"]
    arrays["energy"][row] = pose["energy"]
    arrays["total_energy"][row] = pose["total_energy"]
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


Install = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def _pick_rows(
    workspace: Any,
    choices: Sequence[tuple[int, int, bool]],
    *,
    kind: str,
    algorithm: str,
    job: str | None,
    note: str,
    extra: dict[str, Any] | None = None,
    install: Install | None = None,
) -> EditResult:
    """The shared body of ``pick_hypothesis`` and ``accept_path``: write hypotheses into rows in one edit.

    ``install`` (a region accept) rewrites the hypotheses arrays, given the
    loaded dictionary and returning the one to use (it may grow the slot
    count), AFTER the before-snapshot is taken and before the choices are
    read from it: the rows' previous candidates land in the snapshot, so an
    undo brings them back, and a failure leaves the files untouched.
    """

    with workspace_lock(workspace, timeout=LOCK_TIMEOUT):
        loaded = _load(workspace)
        if install is None and "hypotheses_centerline_xy" not in loaded.hypotheses:
            raise ValueError("the workspace has no hypotheses")
        rows = _check_rows([row for row, _, _ in choices], loaded.n)
        if len(rows) != len(choices):
            raise ValueError("a row appears more than once in the choices")
        window = _window(rows, loaded.n)
        before = _capture(loaded, window)
        summary_before = {row: _row_summary(loaded, row) for row in rows}
        if install is not None:
            loaded.hypotheses = install(loaded.hypotheses)
            if "hypotheses_centerline_xy" not in loaded.hypotheses:
                raise ValueError("the installed hypotheses have no centerlines")
            # Arrays the install created (a workspace without hypotheses so far) were blank before it:
            # the snapshot holds blank slices, so an undo empties the rows again.
            index = np.asarray(window, dtype=np.int64)
            for key in _row_arrays(loaded.hypotheses, loaded.n):
                before.setdefault(f"hypotheses:{key}", _blank_like(loaded.hypotheses[key][index], key))
        poses = {int(row): pose_from_hypothesis(loaded.state, loaded.hypotheses, int(row), int(index), bool(mirrored), image_shape=loaded.image_shape) for row, index, mirrored in choices}
        for row, index, mirrored in choices:
            _write_pose(loaded.state, int(row), poses[int(row)])
            loaded.hypotheses["path_index"][int(row)] = int(index)
            loaded.hypotheses["path_mirrored"][int(row)] = bool(mirrored)
        _refresh_ambiguity(loaded, rows)
        edit_id = _next_edit_id(workspace)
        _apply_provenance(workspace, loaded, rows, algorithm, job or f"edit:{edit_id}")
        payload = {
            "choices": [{"row": int(r), "index": int(i), "mirrored": bool(m)} for r, i, m in choices],
            "algorithm": algorithm, "job": job or f"edit:{edit_id}", "note": str(note or ""), **(extra or {}),
        }
        return _commit(workspace, loaded, before, window, rows, kind, payload, summary_before, edit_id)


# ---------------------------------------------------------------------------
# Public edits


def pick_hypothesis(workspace: Any, row: int, index: int, *, mirrored: bool = False, note: str = "") -> EditResult:
    """Make hypothesis ``index`` of ``row`` (mirrored on request) the row's pose; provenance ``manual:pick``."""

    return _pick_rows(workspace, [(int(row), int(index), bool(mirrored))], kind="pick_hypothesis", algorithm=PICK_ALGORITHM, job=None, note=note)


def accept_path(
    workspace: Any,
    choices: Sequence[tuple[int, int, bool]],
    *,
    algorithm: str,
    job: str,
    note: str = "",
    install: Install | None = None,
    extra: dict[str, Any] | None = None,
) -> EditResult:
    """Pick a hypothesis for every row in ``choices`` (row, index, mirrored) as one edit, attributed to ``algorithm`` and ``job``.

    ``install`` puts the candidates the choices index into the hypotheses
    arrays inside the edit (see ``_pick_rows``); ``extra`` adds fields to the
    log entry (a region accept records its ``candidate_set`` so ``undo`` can
    un-mark the set).
    """

    if not choices:
        raise ValueError("no choices given")
    return _pick_rows(
        workspace, [(int(r), int(i), bool(m)) for r, i, m in choices], kind="accept_path", algorithm=str(algorithm), job=str(job), note=note,
        extra=extra, install=install,
    )


def set_pose(workspace: Any, row: int, pose: dict[str, Any], *, algorithm: str, job: str, note: str = "") -> EditResult:
    """Write an explicit pose (the ``pose_from_hypothesis`` fields) into ``row``; it is not one of the stored hypotheses."""

    with workspace_lock(workspace, timeout=LOCK_TIMEOUT):
        loaded = _load(workspace)
        rows = _check_rows([row], loaded.n)
        window = _window(rows, loaded.n)
        before = _capture(loaded, window)
        summary_before = {rows[0]: _row_summary(loaded, rows[0])}
        _write_pose(loaded.state, rows[0], pose)
        if loaded.hypotheses:
            loaded.hypotheses["path_index"][rows[0]] = -1
            loaded.hypotheses["path_mirrored"][rows[0]] = False
        _refresh_ambiguity(loaded, rows)
        _apply_provenance(workspace, loaded, rows, str(algorithm), str(job))
        payload = {"algorithm": str(algorithm), "job": str(job), "note": str(note or "")}
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


def flip_frame(workspace: Any, row: int, *, note: str = "") -> EditResult:
    """``flip_orientation`` of one row."""

    return flip_orientation(workspace, [int(row)], note=note or "flip frame")


def flip_segment(workspace: Any, row: int, *, note: str = "") -> EditResult:
    """``flip_orientation`` of the segment (``segment_of``) around ``row``."""

    a, b = segment_of(workspace, int(row))
    return flip_orientation(workspace, list(range(a, b + 1)), note=note or f"flip segment rows {a}-{b}")


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
        _refresh_ambiguity(loaded, rows)
        record = {"undoes": target["id"], "undone_kind": target.get("kind"), "note": f"undo {target['id']}"}
        if payload.get("candidate_set"):
            record["candidate_set"] = str(payload["candidate_set"])
        result = _commit(workspace, loaded, before, window, rows, "undo", record, summary_before)
        result.undone = str(target["id"])
    if target.get("kind") == "accept_path" and payload.get("candidate_set"):
        # The accepted candidate set (Phase 3) is no longer in the state: un-mark it.
        # (Imported here: ``algorithms`` builds on this module.)
        from .algorithms import unaccept_candidates

        unaccept_candidates(workspace, str(payload["candidate_set"]), rows=rows, edit=result.edit_id, undoes=str(target["id"]))
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
