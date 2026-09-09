"""The algorithm registry: the pipeline's methods as plugins that run on a region of a workspace.

Phase 3 of ``docs/APP_PLAN.md`` (sections 2 and 6).  A region is a run of
rows ``first..last`` of a workspace between two anchors, rows outside it
whose stored poses are trusted.  An algorithm takes the region's masks, the
workspace's fit configuration and prior, and the anchors, and produces
*candidate poses* per row; ``propagation.select_path`` then chooses one path
through them (the anchors fix its ends and its orientation), exactly as the
propagate stage does for a stretch.  Nothing here writes the state: the
candidates are saved under ``<workspace>/candidates/<id>.npz`` (+ ``.json``)
and become the frame's poses only when the user accepts them
(``accept_candidates`` -> ``edits.accept_path``), so a region rerun that
comes back worse than the current track can be discarded.  An accept puts
the set's candidates into the rows' hypotheses inside the same edit, so an
undo restores the previous candidates too and un-marks the set
(``unaccept_candidates``); a set accepted on part of its path stays open for
the rest (``CandidateSet.accepted_rows``).

Every algorithm wraps existing code and adds no fitting of its own:

- ``independent_multistart``: every row fit from the standard starts of its
  mask (both orientations when a prior exists), every start a candidate.
- ``chain_forward`` / ``chain_backward``: one chain from an anchor through the
  region with prediction, temporal prior and beam (``propagation.propagate``
  with one direction).
- ``beam_path``: the pipeline's second pass on the region, both chains plus
  the refit independent poses and anchor diversity.
- ``slow_refit``: the current poses refit under a longer schedule with the
  anchors' length prior.
- ``mirror``: the current poses and their reversals, no fitting: an
  orientation fix over a region.

Anchors need not be adjacent to the region: the algorithms run on a *local*
copy of the arrays in which the anchors sit right next to the region, so the
chains and the path connect the region to the anchors the user chose.

Every region run is one line in ``<workspaces root>/algorithm_outcomes.jsonl``
with the region's metrics before and after (``region_metrics``), so the
question of which defaults make manual work rare can be answered from the log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from .ambiguity import pose_jump_px
from .batch_fit import PRESETS, BatchFitConfig, fit_masks
from .latent import encode_centerline
from .mask_fit import CropWindow, Initialization, MaskFitResult, reverse_initialization, standard_initializations
from .pipeline import (
    SegmentParams,
    SOURCE_CODES,
    empty_hypotheses,
    load_segmentation_model,
    read_summary,
    segment_frames,
    workspace_arrays,
    workspace_frames,
    workspace_image_shape,
    workspace_lock,
    workspace_masks_of,
    workspace_setup,
)
from .propagation import (
    Candidate,
    PropagationConfig,
    comparable_energy,
    prior_penalty,
    propagate,
    select_path,
    slow_schedule,
    warm_initialization,
)
from . import edits
from .workspace import utc_now


MaskArray = NDArray[np.bool_]
Progress = Callable[[float, str], None]

CANDIDATES_DIR = "candidates"
OUTCOMES_FILE = "algorithm_outcomes.jsonl"
PARAMETER_TYPES = ("int", "float", "bool", "str", "choice")
# The sources a path can attribute a chosen candidate to and the provenance
# algorithm ids they map to when the set is accepted.
METRIC_NAMES = ("median_iou", "p10_iou", "frames_below_0_9", "pose_jumps_over_width", "length_jumps_over_3pct", "orientation_flips", "seconds")
SOURCE_DTYPE = "<U24"
START_DTYPE = "<U48"
# How far from a region ``propose_region`` looks for an anchor.
ANCHOR_SEARCH_ROWS = 200


# ---------------------------------------------------------------------------
# Parameters


@dataclass
class Parameter:
    """One algorithm parameter: what the form shows and how a value is checked."""

    name: str
    type: str
    default: Any
    help: str
    choices: list | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.type not in PARAMETER_TYPES:
            raise ValueError(f"parameter {self.name!r}: unknown type {self.type!r}")
        if self.type == "choice" and not self.choices:
            raise ValueError(f"parameter {self.name!r}: a choice needs choices")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def coerce(self, value: Any) -> Any:
        """``value`` as this parameter's type (the default for ``None``), checked against choices and bounds."""

        if value is None:
            return self.default
        if self.type == "int":
            out: Any = int(value)
        elif self.type == "float":
            out = float(value)
        elif self.type == "bool":
            out = value.strip().lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)
        elif self.type == "choice":
            out = str(value)
            if out not in (self.choices or []):
                raise ValueError(f"parameter {self.name!r}: {out!r} is not one of {self.choices}")
        else:
            out = str(value)
        if self.type in ("int", "float"):
            if self.minimum is not None and out < self.minimum:
                raise ValueError(f"parameter {self.name!r}: {out} is below the minimum {self.minimum}")
            if self.maximum is not None and out > self.maximum:
                raise ValueError(f"parameter {self.name!r}: {out} is above the maximum {self.maximum}")
        return out


def resolve_params(parameters: Sequence[Parameter], params: dict[str, Any] | None) -> dict[str, Any]:
    """Every declared parameter with the given value coerced or its default; unknown keys are ignored (as the stages do)."""

    given = dict(params or {})
    return {p.name: p.coerce(given.get(p.name)) for p in parameters}


_PRESET = Parameter("preset", "choice", "fast", "fitting schedule: the fit's own (fast), or its steps scaled to the balanced or reference preset", choices=["fast", "balanced", "reference"])
_BEAM = Parameter("beam", "int", 3, "distinct chain states kept per direction", minimum=1, maximum=8)
_DAMPING = Parameter("prediction_damping", "float", 0.6, "damping of the first-order pose prediction inside chains (0 = copy the neighbour)", minimum=0.0, maximum=1.0)
_TEMPORAL_WEIGHT = Parameter("temporal_prior_weight", "float", 0.01, "weight of the pull toward the predicted pose (0 = off)", minimum=0.0)
_TEMPORAL_SIGMA = Parameter("temporal_prior_sigma_widths", "float", 0.5, "sigma of that pull, in body widths", minimum=0.01)
_CHAIN_SIGMA = Parameter("chain_length_sigma", "float", 0.02, "log-sigma of the length prior inside chains, centred on the anchor length (0 = the fit's own)", minimum=0.0)
_PATH_PARAMS = (
    Parameter("path_temperature", "float", 0.01, "energy scale of the path's node cost", minimum=1e-6),
    Parameter("path_distance_weight", "float", 1.0, "weight of the squared pose distance between consecutive frames", minimum=0.0),
    Parameter("path_inview_weight", "float", 2.0, "weight of the change of the in-view fraction", minimum=0.0),
    Parameter("path_length_weight", "float", 1.0, "weight of the squared log length change", minimum=0.0),
)


# ---------------------------------------------------------------------------
# Context, candidates, candidate sets


@dataclass
class RegionContext:
    """What an algorithm gets: the region, its anchors, the masks, and the workspace's fit setup.

    ``first..last`` are inclusive rows; ``anchor_before`` / ``anchor_after``
    are rows outside the region whose stored pose anchors chains and the
    path (``None`` = no anchor on that side).  ``masks`` holds the cleaned
    masks of the region rows and the anchors (rows without a usable mask are
    absent).  ``state`` is the workspace's state at build time.
    """

    workspace: Any
    first: int
    last: int
    anchor_before: int | None
    anchor_after: int | None
    masks: dict[int, MaskArray]
    config: BatchFitConfig
    prior: Any
    width_template: np.ndarray
    device: torch.device
    state: dict[str, np.ndarray] = field(default_factory=dict)
    image_shape: tuple[int, int] | None = None

    @property
    def rows(self) -> list[int]:
        return list(range(self.first, self.last + 1))

    @property
    def n(self) -> int:
        return int(len(self.state["frame_index"]))

    @property
    def anchors(self) -> list[int]:
        return [r for r in (self.anchor_before, self.anchor_after) if r is not None]

    def anchor_length(self) -> float | None:
        """Geometric mean of the anchors' fitted lengths; ``None`` without anchors (or without a length prior)."""

        lengths = [float(self.state["body_length_px"][r]) for r in self.anchors if bool(self.state["fitted"][r])]
        lengths = [v for v in lengths if math.isfinite(v) and v > 0]
        if not lengths or self.config.length_prior_px is None:
            return None
        return float(np.exp(np.mean(np.log(lengths))))

    def fit_rows(self) -> list[int]:
        """Region rows with a usable mask."""

        return [r for r in self.rows if r in self.masks]


@dataclass
class CandidatePose:
    """One candidate pose of a row: everything ``pipeline.store_result`` writes, plus where it came from.

    ``energy`` is the comparable energy (``propagation.comparable_energy``:
    overlap plus the fit configuration's priors), ``soft_dice`` the overlap
    energy alone.  ``source`` is an algorithm-specific label (``forward``,
    ``backward``, ``independent``, ``current``, ``mirrored``), ``start`` the
    start that won inside the candidate's fit.
    """

    centerline_xy: np.ndarray
    latent: np.ndarray
    width_px: float
    width_shape: np.ndarray
    width_profile: np.ndarray
    body_length_px: float
    points_in_fov: int
    crop: np.ndarray
    energy: float
    soft_dice: float
    iou: float
    source: str
    start: str = ""

    @classmethod
    def from_result(cls, result: MaskFitResult, config: BatchFitConfig, source: str, start: str | None = None, energy: float | None = None) -> "CandidatePose":
        best = result.records[result.best_index]
        return cls(
            centerline_xy=np.asarray(result.centerline_xy, dtype=np.float64),
            latent=np.asarray(result.latent, dtype=np.float64),
            width_px=float(result.width_px),
            width_shape=np.asarray(result.width_shape, dtype=np.float64),
            width_profile=np.asarray(result.width_profile, dtype=np.float64),
            body_length_px=float(result.body_length_px),
            points_in_fov=int(result.points_in_fov),
            crop=np.asarray((result.crop.x0, result.crop.x1, result.crop.y0, result.crop.y1), dtype=np.int64),
            energy=float(comparable_energy(config, result) if energy is None else energy),
            soft_dice=float(best.get("final_soft_dice_energy", float("nan"))),
            iou=float(best.get("final_iou", float("nan"))),
            source=str(source),
            start=str(result.initializations[result.best_index].name if start is None else start),
        )

    @classmethod
    def from_state(cls, arrays: dict[str, np.ndarray], row: int, config: BatchFitConfig, source: str = "current", start: str = "stored") -> "CandidatePose":
        """The stored pose of ``row`` as a candidate (its energy put on the comparable footing)."""

        row = int(row)
        if not bool(arrays["fitted"][row]):
            raise ValueError(f"row {row} has no stored pose")
        dice = float(arrays["energy"][row])
        length, width, shape = float(arrays["body_length_px"][row]), float(arrays["width_px"][row]), np.asarray(arrays["width_shape"][row], dtype=np.float64)
        energy = dice + prior_penalty(config, length, width, shape) if math.isfinite(dice) and length > 0 and width > 0 else float("nan")
        return cls(
            centerline_xy=np.asarray(arrays["centerline_xy"][row], dtype=np.float64).copy(),
            latent=np.asarray(arrays["latent"][row], dtype=np.float64).copy(),
            width_px=width,
            width_shape=shape.copy(),
            width_profile=np.asarray(arrays["width_profile"][row], dtype=np.float64).copy(),
            body_length_px=length,
            points_in_fov=int(arrays["points_in_fov"][row]),
            crop=np.asarray(arrays["crop"][row], dtype=np.int64).copy(),
            energy=energy,
            soft_dice=dice,
            iou=float(arrays["iou"][row]),
            source=source,
            start=start,
        )

    def mirrored(self, coefficients: int, source: str | None = None) -> "CandidatePose":
        """The same body traversed from the other end (``mask_fit.reverse_result`` semantics)."""

        curve = self.centerline_xy[::-1].copy()
        return replace(
            self,
            centerline_xy=curve,
            latent=encode_centerline(curve, coefficients),
            width_shape=self.width_shape[::-1].copy(),
            width_profile=self.width_profile[::-1].copy(),
            source=self.source if source is None else source,
        )

    def to_pose(self) -> dict[str, Any]:
        """The ``edits.set_pose`` / ``pose_from_hypothesis`` field dictionary."""

        return {
            "latent": self.latent, "width_px": self.width_px, "width_shape": self.width_shape, "width_profile": self.width_profile,
            "centerline_xy": self.centerline_xy, "body_length_px": self.body_length_px, "points_in_fov": self.points_in_fov,
            "crop": self.crop, "iou": self.iou, "energy": self.soft_dice, "total_energy": self.energy,
            "source": int(SOURCE_CODES.get(self.source, 0)), "best_start": self.start,
        }


def _as_candidate(pose: CandidatePose, index: int) -> Candidate:
    """A ``propagation.Candidate`` over a minimal ``MaskFitResult``, so ``select_path`` runs on stored candidates too."""

    x0, x1, y0, y1 = (int(v) for v in pose.crop)
    result = MaskFitResult(
        best_index=0,
        initializations=[Initialization(pose.start or pose.source, pose.latent, pose.width_px, pose.width_shape)],
        records=[{"name": pose.start, "final_energy": pose.energy, "final_iou": pose.iou, "final_soft_dice_energy": pose.soft_dice}],
        latent=pose.latent, width_px=pose.width_px, width_profile=pose.width_profile, centerline_xy=pose.centerline_xy,
        crop=CropWindow(x0, x1, y0, y1, 0, 0), rendered_hard_mask=np.zeros((0, 0), dtype=bool), energy_history=np.zeros(0),
        points_in_fov=pose.points_in_fov, body_length_px=pose.body_length_px, width_shape=pose.width_shape,
    )
    return Candidate(pose.source, result, pose.energy, None, pose.start, float("nan"), beam=index)


@dataclass
class CandidateSet:
    """What one region run produced: candidates per row, the path through them, and the region's metrics.

    ``path`` lists ``(row, candidate index, mirrored)`` for the rows the path
    chose a candidate on; ``metrics`` describes the region with the path
    applied, ``metrics_before`` the state at run time (``region_metrics``).
    """

    algorithm: str
    params: dict[str, Any]
    first: int
    last: int
    anchor_before: int | None
    anchor_after: int | None
    rows: list[int]
    candidates: dict[int, list[CandidatePose]]
    path: list[tuple[int, int, bool]]
    metrics: dict[str, Any]
    created_at: str = field(default_factory=utc_now)
    id: str = ""
    job: str = ""
    metrics_before: dict[str, Any] = field(default_factory=dict)
    frames: list[int] = field(default_factory=list)
    # ``accepted`` means every row of the path is in the state; a partial
    # accept lists the rows taken so far in ``accepted_rows`` (an undo takes
    # them out again) and leaves the set open for the rest.
    accepted: bool = False
    accepted_at: str | None = None
    accepted_edit: str | None = None
    accepted_rows: list[int] = field(default_factory=list)
    workspace: str = ""
    recording: str = ""

    @property
    def path_by_row(self) -> dict[int, tuple[int, bool]]:
        return {int(row): (int(index), bool(mirrored)) for row, index, mirrored in self.path}

    def row_accepted(self, row: int) -> bool:
        return bool(self.accepted) or int(row) in {int(r) for r in self.accepted_rows}

    def chosen(self, row: int) -> CandidatePose | None:
        """The path's candidate of ``row`` (mirrored as the path presents it), ``None`` when the path skipped the row."""

        choice = self.path_by_row.get(int(row))
        if choice is None:
            return None
        index, mirrored = choice
        pose = self.candidates[int(row)][index]
        return pose.mirrored(len(pose.latent) - 4) if mirrored else pose

    def summary(self) -> dict[str, Any]:
        """The list entry: id, algorithm, params, rows, frames, anchors, metrics, state."""

        return {
            "id": self.id, "algorithm": self.algorithm, "params": self.params, "rows": [self.first, self.last],
            "frames": list(self.frames) if self.frames else None, "anchor_before": self.anchor_before, "anchor_after": self.anchor_after,
            "metrics": self.metrics, "metrics_before": self.metrics_before, "created_at": self.created_at, "accepted": self.accepted,
            "accepted_at": self.accepted_at, "accepted_edit": self.accepted_edit, "accepted_rows": [int(r) for r in self.accepted_rows], "job": self.job,
            "candidates": int(sum(len(v) for v in self.candidates.values())), "path_rows": len(self.path), "workspace": self.workspace,
        }

    # ----- storage

    def to_npz(self, path: Path | str) -> Path:
        """Save under ``path`` (``.npz``) with the metadata beside it (``.json``); returns the ``.npz`` path."""

        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        rows = [int(r) for r in self.rows]
        counts = [len(self.candidates.get(r, [])) for r in rows]
        width = max(counts, default=0)
        sample = next((c for r in rows for c in self.candidates.get(r, [])), None)
        n_points = int(sample.centerline_xy.shape[0]) if sample is not None else 0
        latent_size = int(sample.latent.shape[0]) if sample is not None else 0
        shape_size = int(sample.width_shape.shape[0]) if sample is not None else 0
        R, C = len(rows), width
        arrays: dict[str, np.ndarray] = {
            "rows": np.asarray(rows, dtype=np.int64),
            "count": np.asarray(counts, dtype=np.int64),
            "centerline_xy": np.full((R, C, n_points, 2), np.nan),
            "latent": np.full((R, C, latent_size), np.nan),
            "width_px": np.full((R, C), np.nan),
            "width_shape": np.full((R, C, shape_size), np.nan),
            "width_profile": np.full((R, C, n_points), np.nan),
            "body_length_px": np.full((R, C), np.nan),
            "points_in_fov": np.zeros((R, C), dtype=np.int64),
            "crop": np.zeros((R, C, 4), dtype=np.int64),
            "energy": np.full((R, C), np.nan),
            "soft_dice": np.full((R, C), np.nan),
            "iou": np.full((R, C), np.nan),
            "source": np.full((R, C), "", dtype=SOURCE_DTYPE),
            "start": np.full((R, C), "", dtype=START_DTYPE),
            "path_index": np.full(R, -1, dtype=np.int64),
            "path_mirrored": np.zeros(R, dtype=bool),
        }
        by_row = self.path_by_row
        for i, row in enumerate(rows):
            for j, pose in enumerate(self.candidates.get(row, [])):
                arrays["centerline_xy"][i, j] = pose.centerline_xy
                arrays["latent"][i, j] = pose.latent
                arrays["width_px"][i, j] = pose.width_px
                if len(pose.width_shape) == shape_size:
                    arrays["width_shape"][i, j] = pose.width_shape
                arrays["width_profile"][i, j] = pose.width_profile
                arrays["body_length_px"][i, j] = pose.body_length_px
                arrays["points_in_fov"][i, j] = pose.points_in_fov
                arrays["crop"][i, j] = pose.crop
                arrays["energy"][i, j] = pose.energy
                arrays["soft_dice"][i, j] = pose.soft_dice
                arrays["iou"][i, j] = pose.iou
                arrays["source"][i, j] = pose.source
                arrays["start"][i, j] = pose.start
            if row in by_row:
                arrays["path_index"][i], arrays["path_mirrored"][i] = by_row[row]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("." + path.name + ".tmp")
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
        meta = {
            "id": self.id, "algorithm": self.algorithm, "params": _json_safe(self.params), "first": self.first, "last": self.last,
            "anchor_before": self.anchor_before, "anchor_after": self.anchor_after, "rows": rows, "frames": list(self.frames),
            "path": [[int(r), int(i), bool(m)] for r, i, m in self.path], "metrics": _json_safe(self.metrics),
            "metrics_before": _json_safe(self.metrics_before), "created_at": self.created_at, "job": self.job,
            "accepted": self.accepted, "accepted_at": self.accepted_at, "accepted_edit": self.accepted_edit,
            "accepted_rows": [int(r) for r in self.accepted_rows],
            "workspace": self.workspace, "recording": self.recording, "candidates": int(sum(counts)),
        }
        _write_json_atomic(path.with_suffix(".json"), meta)
        return path

    @classmethod
    def from_npz(cls, path: Path | str) -> "CandidateSet":
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        rows = [int(r) for r in arrays["rows"].tolist()]
        candidates: dict[int, list[CandidatePose]] = {}
        for i, row in enumerate(rows):
            poses = []
            for j in range(int(arrays["count"][i])):
                poses.append(
                    CandidatePose(
                        centerline_xy=arrays["centerline_xy"][i, j].copy(), latent=arrays["latent"][i, j].copy(),
                        width_px=float(arrays["width_px"][i, j]), width_shape=arrays["width_shape"][i, j].copy(),
                        width_profile=arrays["width_profile"][i, j].copy(), body_length_px=float(arrays["body_length_px"][i, j]),
                        points_in_fov=int(arrays["points_in_fov"][i, j]), crop=arrays["crop"][i, j].copy(),
                        energy=float(arrays["energy"][i, j]), soft_dice=float(arrays["soft_dice"][i, j]), iou=float(arrays["iou"][i, j]),
                        source=str(arrays["source"][i, j]), start=str(arrays["start"][i, j]),
                    )
                )
            candidates[row] = poses
        return cls(
            algorithm=str(meta["algorithm"]), params=dict(meta.get("params") or {}), first=int(meta["first"]), last=int(meta["last"]),
            anchor_before=meta.get("anchor_before"), anchor_after=meta.get("anchor_after"), rows=rows, candidates=candidates,
            path=[(int(r), int(i), bool(m)) for r, i, m in meta.get("path") or []], metrics=dict(meta.get("metrics") or {}),
            created_at=str(meta.get("created_at") or ""), id=str(meta.get("id") or path.stem), job=str(meta.get("job") or ""),
            metrics_before=dict(meta.get("metrics_before") or {}), frames=[int(f) for f in meta.get("frames") or []],
            accepted=bool(meta.get("accepted", False)), accepted_at=meta.get("accepted_at"), accepted_edit=meta.get("accepted_edit"),
            accepted_rows=[int(r) for r in meta.get("accepted_rows") or []],
            workspace=str(meta.get("workspace") or ""), recording=str(meta.get("recording") or ""),
        )


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Candidate set files in a workspace


def candidates_dir(workspace: Any) -> Path:
    return Path(workspace.path) / CANDIDATES_DIR


def _validate_set_id(set_id: str) -> str:
    if not set_id or "/" in set_id or "\\" in set_id or set_id in (".", ".."):
        raise ValueError(f"invalid candidate set id {set_id!r}")
    return set_id


def next_candidate_id(workspace: Any) -> str:
    """``c000001``, ``c000002``, ... from a counter file beside the sets (ids are never reused)."""

    directory = candidates_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    counter = directory / "counter"
    current = 0
    if counter.exists():
        text = counter.read_text().strip()
        if text.isdigit():
            current = int(text)
    current += 1
    tmp = counter.with_name("counter.tmp")
    tmp.write_text(f"{current}\n")
    os.replace(tmp, counter)
    return f"c{current:06d}"


def candidate_set_path(workspace: Any, set_id: str) -> Path:
    return candidates_dir(workspace) / f"{_validate_set_id(str(set_id))}.npz"


def save_candidate_set(workspace: Any, candidate_set: CandidateSet) -> Path:
    if not candidate_set.id:
        candidate_set.id = next_candidate_id(workspace)
    return candidate_set.to_npz(candidate_set_path(workspace, candidate_set.id))


def load_candidate_set(workspace: Any, set_id: str) -> CandidateSet:
    path = candidate_set_path(workspace, set_id)
    if not path.exists() or not path.with_suffix(".json").exists():
        raise FileNotFoundError(f"workspace {workspace.info.name} has no candidate set {set_id!r}")
    return CandidateSet.from_npz(path)


def list_candidate_sets(workspace: Any) -> list[dict[str, Any]]:
    """The ``summary()`` of every stored set, newest first (metadata only: the arrays are not read)."""

    directory = candidates_dir(workspace)
    if not directory.exists():
        return []
    out = []
    for meta_path in sorted(directory.glob("*.json")):
        if not meta_path.with_suffix(".npz").exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        out.append(
            {
                "id": str(meta.get("id") or meta_path.stem), "algorithm": meta.get("algorithm"), "params": meta.get("params") or {},
                "rows": [meta.get("first"), meta.get("last")], "frames": meta.get("frames") or None,
                "anchor_before": meta.get("anchor_before"), "anchor_after": meta.get("anchor_after"),
                "metrics": meta.get("metrics") or {}, "metrics_before": meta.get("metrics_before") or {},
                "created_at": meta.get("created_at"), "accepted": bool(meta.get("accepted", False)), "accepted_at": meta.get("accepted_at"),
                "accepted_edit": meta.get("accepted_edit"), "accepted_rows": [int(r) for r in meta.get("accepted_rows") or []],
                "job": meta.get("job") or "", "candidates": meta.get("candidates"),
                "path_rows": len(meta.get("path") or []), "workspace": meta.get("workspace") or workspace.info.name,
            }
        )
    out.sort(key=lambda e: str(e.get("created_at") or ""), reverse=True)
    return out


def delete_candidate_set(workspace: Any, set_id: str) -> bool:
    """Remove a set's files; whether there was one."""

    path = candidate_set_path(workspace, set_id)
    found = False
    for target in (path, path.with_suffix(".json")):
        if target.exists():
            target.unlink()
            found = True
    return found


def candidate_sets_covering(workspace: Any, row: int, *, include_accepted: bool = False) -> list[CandidateSet]:
    """The stored sets whose region contains ``row``, newest first; sets accepted (as a whole, or on this row) are excluded unless asked."""

    out = []
    for entry in list_candidate_sets(workspace):
        first, last = entry["rows"]
        if first is None or last is None or not int(first) <= int(row) <= int(last):
            continue
        if not include_accepted and (entry["accepted"] or int(row) in {int(r) for r in entry.get("accepted_rows") or []}):
            continue
        try:
            out.append(load_candidate_set(workspace, entry["id"]))
        except (FileNotFoundError, ValueError, KeyError):
            continue
    return out


# ---------------------------------------------------------------------------
# Metrics


def frame_step(frames: np.ndarray) -> int:
    """The workspace's frame step: the smallest positive gap between adjacent frames (1 for a single frame)."""

    gaps = np.diff(np.asarray(frames, dtype=np.int64))
    gaps = gaps[gaps > 0]
    return int(gaps.min()) if len(gaps) else 1


def _consecutive_pairs(arrays: dict[str, np.ndarray], rows: Sequence[int]) -> list[tuple[int, int]]:
    """Fitted adjacent-row pairs touching ``rows`` (a region's boundary pairs with its neighbours included).

    Adjacent rows are consecutive when their frames differ by the workspace's
    step (``frame_step``), so a workspace of every other frame gets its pairs
    too; a gap wider than the step (frames missing from an import) breaks
    the pair.
    """

    fitted = np.asarray(arrays["fitted"], dtype=bool)
    frames = np.asarray(arrays["frame_index"], dtype=np.int64) if "frame_index" in arrays else np.arange(len(fitted))
    step = frame_step(frames)
    n = len(fitted)
    pairs: set[tuple[int, int]] = set()
    for r in rows:
        for a, b in ((r - 1, r), (r, r + 1)):
            if 0 <= a < b < n and fitted[a] and fitted[b] and int(frames[b]) - int(frames[a]) == step:
                pairs.add((a, b))
    return sorted(pairs)


def region_metrics(arrays: dict[str, np.ndarray], rows: Sequence[int], image_shape: tuple[int, int] | None = None, *, length_jump_fraction: float = 0.03) -> dict[str, Any]:
    """How good the CURRENT state is over ``rows``: overlap, jumps into and inside the region, orientation flips.

    A pose jump is a mean in-view point distance between consecutive frames
    above the fitted width (``ambiguity.pose_jump_px``); a length jump a log
    change of more than ``length_jump_fraction``; an orientation flip a pair
    whose ends match better swapped (``pipeline.orientation_consistency``).
    Pairs include the region's boundary with its neighbours, so a region whose
    poses jump relative to the anchors counts as discontinuous.  The result
    of a run of this function on the state before and after a change is
    comparable (``seconds`` is filled in by the run).
    """

    rows = sorted({int(r) for r in rows})
    fitted = np.asarray(arrays["fitted"], dtype=bool)
    n = len(fitted)
    selected = [r for r in rows if 0 <= r < n and fitted[r]]
    iou = np.asarray(arrays["iou"], dtype=np.float64)[selected] if selected else np.zeros(0)
    iou = iou[np.isfinite(iou)]
    curves = np.asarray(arrays["centerline_xy"], dtype=np.float64)
    width = np.asarray(arrays["width_px"], dtype=np.float64)
    length = np.asarray(arrays["body_length_px"], dtype=np.float64)
    pairs = _consecutive_pairs(arrays, rows)
    pose_jumps = length_jumps = flips = 0
    for a, b in pairs:
        jump = pose_jump_px(curves[b], curves[a], image_shape)
        if math.isfinite(jump) and jump > max(float(width[b]), 1.0):
            pose_jumps += 1
        if length[a] > 0 and length[b] > 0 and abs(math.log(length[b] / length[a])) > length_jump_fraction:
            length_jumps += 1
        same = np.linalg.norm(curves[b, 0] - curves[a, 0]) + np.linalg.norm(curves[b, -1] - curves[a, -1])
        swapped = np.linalg.norm(curves[b, 0] - curves[a, -1]) + np.linalg.norm(curves[b, -1] - curves[a, 0])
        if swapped < same:
            flips += 1
    return {
        "frames": len(rows),
        "frames_fitted": len(selected),
        "median_iou": float(np.median(iou)) if len(iou) else None,
        "p10_iou": float(np.percentile(iou, 10)) if len(iou) else None,
        "min_iou": float(iou.min()) if len(iou) else None,
        "frames_below_0_9": int(np.sum(iou < 0.9)),
        "pairs": len(pairs),
        "pose_jumps_over_width": int(pose_jumps),
        "length_jumps_over_3pct": int(length_jumps),
        "orientation_flips": int(flips),
        "seconds": None,
    }


def apply_path(arrays: dict[str, np.ndarray], candidate_set: CandidateSet, rows: Sequence[int] | None = None) -> list[int]:
    """Write the path's poses of ``candidate_set`` into a copy-or-not of ``arrays`` (the metric fields); returns the rows written.

    Only the fields ``region_metrics`` reads are written (centerline, iou,
    width, length, in-view count, fitted): this is how a set's metrics are
    computed without touching the workspace.
    """

    wanted = None if rows is None else {int(r) for r in rows}
    written = []
    for row, index, mirrored in candidate_set.path:
        if wanted is not None and int(row) not in wanted:
            continue
        pose = candidate_set.candidates[int(row)][int(index)]
        if mirrored:
            pose = pose.mirrored(len(pose.latent) - 4)
        arrays["centerline_xy"][row] = pose.centerline_xy
        arrays["iou"][row] = pose.iou
        arrays["width_px"][row] = pose.width_px
        arrays["body_length_px"][row] = pose.body_length_px
        arrays["points_in_fov"][row] = pose.points_in_fov
        arrays["fitted"][row] = True
        written.append(int(row))
    return written


def metrics_with_path(ctx_state: dict[str, np.ndarray], candidate_set: CandidateSet, rows: Sequence[int], image_shape: tuple[int, int] | None) -> dict[str, Any]:
    """``region_metrics`` over ``rows`` after the set's path is applied to a copy of the state."""

    keys = ("fitted", "frame_index", "centerline_xy", "iou", "width_px", "body_length_px", "points_in_fov")
    copy = {k: np.array(ctx_state[k], copy=True) for k in keys if k in ctx_state}
    apply_path(copy, candidate_set)
    return region_metrics(copy, rows, image_shape)


# ---------------------------------------------------------------------------
# The local view: anchors adjacent to the region


@dataclass
class _Local:
    """A slice of the state in which the anchors sit right next to the region (see the module docstring)."""

    arrays: dict[str, np.ndarray]
    masks: dict[int, MaskArray]
    rows: list[int]  # local index -> workspace row
    stretch: tuple[int, int]  # local rows of the region

    def to_local(self, row: int) -> int:
        return self.rows.index(int(row))


def _local_view(ctx: RegionContext) -> _Local:
    n = ctx.n
    before: list[int] = []
    if ctx.anchor_before is not None:
        # Two rows beyond the anchor give the chain its initial velocity and the anchor-diversity refit its start.
        before = [r for r in (ctx.anchor_before - 2, ctx.anchor_before - 1) if 0 <= r < ctx.first] + [ctx.anchor_before]
    after: list[int] = []
    if ctx.anchor_after is not None:
        after = [ctx.anchor_after] + [r for r in (ctx.anchor_after + 1, ctx.anchor_after + 2) if ctx.last < r < n]
    rows = before + ctx.rows + after
    index = np.asarray(rows, dtype=np.int64)
    arrays = {k: v[index] for k, v in ctx.state.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n}
    masks = {i: ctx.masks[r] for i, r in enumerate(rows) if r in ctx.masks}
    a = len(before)
    return _Local(arrays, masks, rows, (a, a + len(ctx.rows) - 1))


def _path_config(params: dict[str, Any], **overrides: Any) -> PropagationConfig:
    """A ``PropagationConfig`` from whichever of its fields ``params`` carries, plus ``overrides``."""

    known = {f for f in PropagationConfig.__dataclass_fields__}
    values = {k: v for k, v in params.items() if k in known}
    if "chain_length_sigma" in values and values["chain_length_sigma"] is not None and float(values["chain_length_sigma"]) <= 0:
        values["chain_length_sigma"] = None
    values.update(overrides)
    return PropagationConfig(**values)


def _choose_path(ctx: RegionContext, candidates: dict[int, list[CandidatePose]], propagation: PropagationConfig) -> list[tuple[int, int, bool]]:
    """``propagation.select_path`` over the candidates with the anchors; ``(row, index, mirrored)`` per chosen row."""

    local = _local_view(ctx)
    local_candidates: dict[int, list[Candidate]] = {}
    for row, poses in candidates.items():
        if poses:
            local_candidates[local.to_local(row)] = [_as_candidate(p, j) for j, p in enumerate(poses)]
    if not local_candidates:
        return []
    chosen = select_path(local_candidates, local.arrays, [local.stretch], ctx.config, propagation, ctx.image_shape)
    path: list[tuple[int, int, bool]] = []
    for local_row, choice in sorted(chosen.items()):
        options = local_candidates[local_row]
        index = next(j for j, c in enumerate(options) if c is choice.candidate)
        path.append((local.rows[local_row], index, bool(choice.mirrored)))
    return path


def _assemble(ctx: RegionContext, algorithm: str, params: dict[str, Any], candidates: dict[int, list[CandidatePose]], propagation: PropagationConfig) -> CandidateSet:
    """The set with its path and metrics, ready to be saved."""

    path = _choose_path(ctx, candidates, propagation)
    frames = np.asarray(ctx.state["frame_index"], dtype=np.int64)
    candidate_set = CandidateSet(
        algorithm=algorithm, params=_json_safe(params), first=ctx.first, last=ctx.last, anchor_before=ctx.anchor_before, anchor_after=ctx.anchor_after,
        rows=ctx.rows, candidates={r: list(candidates.get(r, [])) for r in ctx.rows}, path=path, metrics={},
        frames=[int(frames[ctx.first]), int(frames[ctx.last])], workspace=str(ctx.workspace.info.name), recording=str(ctx.workspace.info.recording),
    )
    candidate_set.metrics = metrics_with_path(ctx.state, candidate_set, ctx.rows, ctx.image_shape)
    return candidate_set


def _preset_schedule(config: BatchFitConfig, preset: str, length_sigma: float | None = None) -> BatchFitConfig:
    """The fit configuration with the steps of ``preset``: ``slow_schedule`` when the rasters match, else the steps scaled.

    ``fast`` keeps the configuration.  A fit configuration whose stages differ
    from the presets' (tests, or a custom schedule) cannot take a preset's
    steps stage by stage, so its own step counts are scaled by the preset's
    total steps over the fast preset's.
    """

    if preset == "fast":
        if length_sigma is not None and config.length_prior_px is not None:
            return replace(config, length_prior_log_sigma=min(config.length_prior_log_sigma, length_sigma))
        return config
    target = PRESETS[preset]
    try:
        return slow_schedule(config, target, length_sigma)
    except ValueError:
        ratio = sum(target.stage_steps) / sum(PRESETS["fast"].stage_steps)
        scaled = replace(config, stage_steps=tuple(max(1, int(round(ratio * s))) for s in config.stage_steps))
        if length_sigma is not None and config.length_prior_px is not None:
            scaled = replace(scaled, length_prior_log_sigma=min(config.length_prior_log_sigma, length_sigma))
        return scaled


def _rank(candidates: dict[int, list[CandidatePose]]) -> dict[int, list[CandidatePose]]:
    """Candidates ordered as the propagate stage stores them: by source (independent, forward, backward), then energy."""

    return {row: sorted(poses, key=lambda p: (SOURCE_CODES.get(p.source, 3), p.energy if math.isfinite(p.energy) else math.inf)) for row, poses in candidates.items()}


def _report(progress: Progress | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(float(min(max(fraction, 0.0), 1.0)), message)


# ---------------------------------------------------------------------------
# The algorithms


class Algorithm(Protocol):
    id: str
    label: str
    scope: str
    description: str
    parameters: list[Parameter]

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet: ...


class _RegionAlgorithm:
    """Shared bookkeeping of the region algorithms: parameter resolution and the registry entry."""

    id = ""
    label = ""
    scope = "region"
    description = ""
    parameters: list[Parameter] = []
    # Which anchors the algorithm cannot run without ("before", "after"); checked
    # when a request is made, before a job is queued and the region segmented.
    needs_anchor: tuple[str, ...] = ()

    def resolve(self, params: dict[str, Any] | None) -> dict[str, Any]:
        return resolve_params(self.parameters, params)

    def check_anchors(self, anchor_before: int | None, anchor_after: int | None) -> None:
        """``ValueError`` when an anchor this algorithm needs is missing."""

        for side, anchor in (("before", anchor_before), ("after", anchor_after)):
            if side in self.needs_anchor and anchor is None:
                raise ValueError(f"{self.id} needs an anchor {side} the region")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "scope": self.scope, "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters], "needs_anchor": list(self.needs_anchor),
        }

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet:  # pragma: no cover - overridden
        raise NotImplementedError


class IndependentMultistart(_RegionAlgorithm):
    id = "independent_multistart"
    label = "Independent multi-start"
    description = (
        "Every frame of the region fit from the standard starts of its mask (skeleton and moment arcs; both orientations when a "
        "recording prior exists), each start kept as a candidate. The path then picks one per frame with the anchors."
    )
    parameters = [_PRESET, *_PATH_PARAMS]

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet:
        params = self.resolve(params)
        config = _preset_schedule(ctx.config, params["preset"])
        rows = ctx.fit_rows()
        shape = np.asarray(ctx.prior.width_shape, dtype=np.float64) if ctx.prior is not None else None
        jobs: list[tuple[int, Initialization]] = []
        _report(progress, 0.02, f"{self.id}: building starts for {len(rows)} frames")
        for row in rows:
            starts = standard_initializations(ctx.masks[row], config=config)
            if shape is not None:
                starts = [replace(s, width_shape=shape) for s in starts]
            if ctx.prior is not None:
                starts = starts + [reverse_initialization(s, config=config) for s in starts]
            jobs.extend((row, s) for s in starts)
        candidates: dict[int, list[CandidatePose]] = {r: [] for r in rows}
        chunk_size = max(1, int(config.max_rows))
        for k in range(0, len(jobs), chunk_size):
            chunk = jobs[k : k + chunk_size]
            results = fit_masks([ctx.masks[r] for r, _ in chunk], [[s] for _, s in chunk], width_template=ctx.width_template, config=config, device=ctx.device)
            for (row, start), result in zip(chunk, results, strict=True):
                candidates[row].append(CandidatePose.from_result(result, ctx.config, "independent", start.name))
            _report(progress, 0.05 + 0.85 * (k + len(chunk)) / max(len(jobs), 1), f"{self.id}: fit {k + len(chunk)}/{len(jobs)} starts")
        _report(progress, 0.92, f"{self.id}: selecting the path")
        return _assemble(ctx, self.id, params, _rank(candidates), _path_config(params))


def _propagate_region(
    ctx: RegionContext, propagation: PropagationConfig, warm_config: BatchFitConfig | None, progress: Progress | None, label: str
) -> tuple[dict[int, list[CandidatePose]], dict[str, Any]]:
    """``propagation.propagate`` over the region as one stretch in the local view; candidates keyed by workspace row."""

    local = _local_view(ctx)
    _report(progress, 0.05, f"{label}: propagating over {len(ctx.rows)} frames")
    raw, info = propagate(
        local.arrays, [local.stretch], local.masks, config=ctx.config, device=ctx.device, width_template=ctx.width_template,
        propagation=propagation, warm_config=warm_config,
    )
    candidates: dict[int, list[CandidatePose]] = {}
    for local_row, options in raw.items():
        row = local.rows[local_row]
        ordered = sorted(options, key=lambda c: (SOURCE_CODES.get(c.source, 3), c.beam))
        candidates[row] = [CandidatePose.from_result(c.result, ctx.config, c.source, c.start_name, c.total_energy) for c in ordered]
    return candidates, info


class _Chain(_RegionAlgorithm):
    direction = "forward"
    parameters = [_DAMPING, _TEMPORAL_WEIGHT, _TEMPORAL_SIGMA, _BEAM, _CHAIN_SIGMA, _PRESET, *_PATH_PARAMS]

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet:
        params = self.resolve(params)
        self.check_anchors(ctx.anchor_before, ctx.anchor_after)
        propagation = _path_config(
            params, forward=self.direction == "forward", backward=self.direction == "backward", refit_independent=False, anchor_diversity=False,
        )
        warm = None if params["preset"] == "fast" else _preset_schedule(ctx.config, params["preset"], propagation.chain_length_sigma)
        candidates, _ = _propagate_region(ctx, propagation, warm, progress, self.id)
        _report(progress, 0.92, f"{self.id}: selecting the path")
        return _assemble(ctx, self.id, params, candidates, propagation)


class ChainForward(_Chain):
    id = "chain_forward"
    label = "Chain forward"
    direction = "forward"
    needs_anchor = ("before",)
    description = "One chain from the anchor before the region through it, frame by frame, warm-started from the neighbour and its prediction, keeping a beam of distinct states."


class ChainBackward(_Chain):
    id = "chain_backward"
    label = "Chain backward"
    direction = "backward"
    needs_anchor = ("after",)
    description = "One chain from the anchor after the region back through it, warm-started from the neighbour and its prediction, keeping a beam of distinct states."


class BeamPath(_RegionAlgorithm):
    id = "beam_path"
    label = "Beam and path (pipeline second pass)"
    description = (
        "The propagate stage on this region: forward and backward chains from the anchors with prediction, temporal prior, beam and anchor "
        "diversity, the stored poses refit under the chain schedule, and one path through all candidates."
    )
    parameters = [
        _BEAM, _DAMPING, _TEMPORAL_WEIGHT, _TEMPORAL_SIGMA, *_PATH_PARAMS,
        Parameter("refit_independent", "bool", True, "refit the stored pose of every frame under the chain schedule as a candidate"),
        Parameter("anchor_diversity", "bool", True, "also start each chain from the anchor refit from the frame beyond it"),
        _CHAIN_SIGMA, _PRESET,
    ]

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet:
        params = self.resolve(params)
        propagation = _path_config(params)
        warm = None if params["preset"] == "fast" else _preset_schedule(ctx.config, params["preset"], propagation.chain_length_sigma)
        candidates, _ = _propagate_region(ctx, propagation, warm, progress, self.id)
        _report(progress, 0.92, f"{self.id}: selecting the path")
        return _assemble(ctx, self.id, params, candidates, propagation)


class SlowRefit(_RegionAlgorithm):
    id = "slow_refit"
    label = "Slow refit"
    description = "The current pose of every frame refit under a longer schedule with the length prior centred on the anchors' length; no temporal prior."
    parameters = [
        Parameter("preset", "choice", "balanced", "fitting schedule (steps of the preset on the fit's rasters)", choices=["fast", "balanced", "reference"]),
        Parameter("length_sigma", "float", 0.02, "log-sigma of the length prior around the anchors' length", minimum=0.001),
        *_PATH_PARAMS,
    ]

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet:
        params = self.resolve(params)
        config = _preset_schedule(ctx.config, params["preset"], params["length_sigma"])
        config = replace(config, temporal_prior_weight=0.0)
        length = ctx.anchor_length()
        if length is not None:
            config = replace(config, length_prior_px=length)
        rows = [r for r in ctx.fit_rows() if bool(ctx.state["fitted"][r])]
        candidates: dict[int, list[CandidatePose]] = {r: [] for r in rows}
        chunk_size = max(1, int(config.max_rows))
        for k in range(0, len(rows), chunk_size):
            chunk = rows[k : k + chunk_size]
            starts = [[warm_initialization(ctx.state["latent"][r], float(ctx.state["width_px"][r]), ctx.state["width_shape"][r], "slow_refit")] for r in chunk]
            results = fit_masks([ctx.masks[r] for r in chunk], starts, width_template=ctx.width_template, config=config, device=ctx.device)
            for row, result in zip(chunk, results, strict=True):
                candidates[row].append(CandidatePose.from_result(result, ctx.config, "independent", "slow_refit"))
            _report(progress, 0.05 + 0.85 * (k + len(chunk)) / max(len(rows), 1), f"{self.id}: refit {k + len(chunk)}/{len(rows)} frames")
        _report(progress, 0.92, f"{self.id}: selecting the path")
        return _assemble(ctx, self.id, params, candidates, _path_config(params))


class Mirror(_RegionAlgorithm):
    id = "mirror"
    label = "Mirror orientation"
    description = "No fitting: every frame's current pose and its reversal are the candidates, and the path follows the anchors' orientation."
    parameters = [*_PATH_PARAMS]

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet:
        params = self.resolve(params)
        candidates: dict[int, list[CandidatePose]] = {}
        for row in ctx.rows:
            if not bool(ctx.state["fitted"][row]):
                continue
            current = CandidatePose.from_state(ctx.state, row, ctx.config, "current", "stored")
            candidates[row] = [current, current.mirrored(ctx.config.coefficients, "mirrored")]
        _report(progress, 0.5, f"{self.id}: selecting the orientation of {len(candidates)} frames")
        # The candidates are each other's mirrors already, so the path needs no mirror nodes of its own.
        return _assemble(ctx, self.id, params, candidates, _path_config(params, path_mirrors=False))


REGISTRY: dict[str, Algorithm] = {
    algorithm.id: algorithm for algorithm in (IndependentMultistart(), ChainForward(), ChainBackward(), BeamPath(), SlowRefit(), Mirror())
}


def get_algorithm(algorithm_id: str) -> Algorithm:
    try:
        return REGISTRY[str(algorithm_id)]
    except KeyError as error:
        raise ValueError(f"unknown algorithm {algorithm_id!r}; expected one of {sorted(REGISTRY)}") from error


def list_algorithms() -> list[dict[str, Any]]:
    """Every registered algorithm as ``{id, label, scope, description, parameters}`` for ``GET /api/algorithms``."""

    return [algorithm.to_dict() for algorithm in REGISTRY.values()]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Building a context


def _resolve_device(device: torch.device | str | None) -> torch.device:
    return torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))


def _segment_params(workspace: Any) -> SegmentParams:
    """The segment stage's settings as the workspace recorded them (checkpoint, threshold, cleanup); the defaults otherwise."""

    summary = read_summary(workspace)
    settings = getattr(workspace.info, "settings", None) or {}
    values: dict[str, Any] = {}
    fingerprint = summary["checkpoint"] if "checkpoint" in summary else settings.get("checkpoint", "unset")
    if fingerprint is None:
        values["checkpoint"] = None
    elif isinstance(fingerprint, dict) and fingerprint.get("path"):
        values["checkpoint"] = str(fingerprint["path"])
    elif isinstance(fingerprint, str):
        values["checkpoint"] = fingerprint
    if "threshold" in summary:
        values["threshold"] = float(summary["threshold"])
    cleanup = summary.get("mask_cleanup") or settings.get("mask_cleanup") or {}
    if "fill_holes" in cleanup:
        values["fill_holes"] = bool(cleanup["fill_holes"])
    if "fill_holes_radius_px" in cleanup:
        values["hole_radius"] = int(cleanup["fill_holes_radius_px"])
    if "largest_component" in cleanup:
        values["largest_only"] = bool(cleanup["largest_component"])
    if "min_worm_pixels" in cleanup:
        values["min_worm_pixels"] = int(cleanup["min_worm_pixels"])
    if "flat_field" in settings:
        values["flat_field"] = bool(settings["flat_field"])
    return SegmentParams.from_dict(values)


def segment_rows(workspace: Any, rows: Sequence[int], device: torch.device, params: SegmentParams | None = None) -> dict[int, MaskArray]:
    """Segment these rows of the workspace's recording the way the segment stage does; masks below ``min_worm_pixels`` are dropped."""

    rows = [int(r) for r in rows]
    if not rows:
        return {}
    params = params or _segment_params(workspace)
    frames = workspace_frames(workspace, params)
    try:
        model = load_segmentation_model(params.checkpoint, device)
        out: dict[int, MaskArray] = {}
        for k in range(0, len(rows), max(1, int(params.slab))):
            chunk = rows[k : k + max(1, int(params.slab))]
            masks, stats, _ = segment_frames(frames, model, [int(workspace.frame_index[r]) for r in chunk], params, device)
            for row, mask, frame_stats in zip(chunk, masks, stats, strict=True):
                if int(frame_stats["worm_pixels"]) >= int(params.min_worm_pixels):
                    out[row] = np.asarray(mask, dtype=bool)
        return out
    finally:
        frames.close()


def store_masks(workspace: Any, masks: dict[int, MaskArray]) -> int:
    """Write freshly segmented ``masks`` (row -> mask) into the workspace's mask store under its lock; returns how many were stored.

    Only rows without a stored mask are written (an override or a stored
    mask always wins), and a store that refuses them (a shape that differs
    from the stored chunks, a read-only directory) is reported, not fatal:
    the masks in memory serve this run either way.
    """

    stored = set(workspace.mask_rows().tolist())
    rows = sorted(int(r) for r in masks if int(r) not in stored)
    if not rows:
        return 0
    try:
        with workspace_lock(workspace):
            workspace.set_masks(rows, [np.asarray(masks[row], dtype=bool) for row in rows])
    except (OSError, ValueError) as error:
        print(f"could not store {len(rows)} segmented masks in {workspace.path}: {error}", flush=True)
        return 0
    return len(rows)


def build_context(
    workspace: Any,
    first: int,
    last: int,
    anchor_before: int | None = None,
    anchor_after: int | None = None,
    device: torch.device | str | None = None,
    *,
    segment_missing: bool = True,
) -> RegionContext:
    """The ``RegionContext`` of rows ``first..last`` with these anchors.

    Masks come from the workspace (overrides first); rows without a stored
    mask are segmented from the recording with the workspace's segment
    settings when ``segment_missing`` is set (an imported run has no masks).
    Anchors must be fitted rows outside the region.
    """

    setup = workspace_setup(workspace)
    with workspace_lock(workspace):
        state = workspace_arrays(workspace, setup.config)
    n = int(len(state["frame_index"]))
    first, last = int(first), int(last)
    if not 0 <= first <= last < n:
        raise ValueError(f"region rows {first}..{last} outside 0..{n - 1}")
    for name, anchor, ok in (("anchor_before", anchor_before, lambda r: r < first), ("anchor_after", anchor_after, lambda r: r > last)):
        if anchor is None:
            continue
        anchor = int(anchor)
        if not 0 <= anchor < n or not ok(anchor):
            raise ValueError(f"{name} row {anchor} is not outside the region {first}..{last} (0..{n - 1})")
        if not bool(state["fitted"][anchor]):
            raise ValueError(f"{name} row {anchor} has no fitted pose")
    anchor_before = None if anchor_before is None else int(anchor_before)
    anchor_after = None if anchor_after is None else int(anchor_after)
    resolved = _resolve_device(device)
    wanted = list(range(first, last + 1)) + [r for r in (anchor_before, anchor_after) if r is not None]
    min_pixels = int((read_summary(workspace).get("mask_cleanup") or {}).get("min_worm_pixels") or 1)
    masks = workspace_masks_of(workspace, max(1, min_pixels))(wanted)
    missing = [r for r in wanted if r not in masks and r not in set(workspace.mask_rows().tolist())]
    if missing and segment_missing:
        fresh = segment_rows(workspace, missing, resolved)
        masks.update(fresh)
        # Keep them: the next region run on these frames (another algorithm to
        # compare, a wider region) reads them instead of loading the segmenter again.
        store_masks(workspace, fresh)
    masks = {r: m for r, m in masks.items() if np.asarray(m).any()}
    return RegionContext(
        workspace=workspace, first=first, last=last, anchor_before=anchor_before, anchor_after=anchor_after, masks=masks,
        config=setup.config, prior=setup.prior, width_template=setup.template, device=resolved, state=state, image_shape=workspace_image_shape(workspace),
    )


# ---------------------------------------------------------------------------
# Proposing a region


def propose_region(workspace: Any, row: int, *, pad: int = 2) -> dict[str, Any]:
    """A region around ``row`` with anchors: the propagation stretch containing it padded by ``pad``, else ten rows either side.

    Anchors are the nearest fitted rows outside the region with ambiguity
    score 0 and overlap at least 0.9 (``None`` when there is none within
    ``ANCHOR_SEARCH_ROWS``).  Rows, not frames.
    """

    state = workspace.load_state()
    n = int(len(state["frame_index"]))
    row = int(row)
    if not 0 <= row < n:
        raise ValueError(f"row {row} outside 0..{n - 1}")
    stretches = edits._stretches(workspace, state)
    stretch = next(((int(a), int(b)) for a, b in stretches if a <= row <= b), None)
    if stretch is not None:
        first, last = max(0, stretch[0] - int(pad)), min(n - 1, stretch[1] + int(pad))
        reason = f"propagation stretch rows {stretch[0]}-{stretch[1]} padded by {int(pad)}"
    else:
        first, last = max(0, row - 10), min(n - 1, row + 10)
        reason = "no propagation stretch contains the frame: ten frames either side"
    anchor_before, anchor_after = propose_anchors(state, first, last)
    return {"first": first, "last": last, "anchor_before": anchor_before, "anchor_after": anchor_after, "reason": reason, "stretch": None if stretch is None else list(stretch)}


def propose_anchors(state: dict[str, np.ndarray], first: int, last: int, *, min_iou: float = 0.9, search: int = ANCHOR_SEARCH_ROWS) -> tuple[int | None, int | None]:
    """The nearest fitted rows outside ``first..last`` with ambiguity score 0 and overlap at least ``min_iou`` (``None`` when none within ``search`` rows)."""

    n = int(len(state["frame_index"]))
    fitted = np.asarray(state.get("fitted", np.zeros(n, dtype=bool)), dtype=bool)
    iou = np.asarray(state.get("iou", np.full(n, np.nan)), dtype=np.float64)
    score = np.asarray(state.get("ambiguity_score", np.zeros(n, dtype=np.int64)))

    def good(r: int) -> bool:
        return bool(fitted[r]) and math.isfinite(float(iou[r])) and float(iou[r]) >= min_iou and int(score[r]) == 0

    before = next((r for r in range(int(first) - 1, max(-1, int(first) - 1 - int(search)), -1) if good(r)), None)
    after = next((r for r in range(int(last) + 1, min(n, int(last) + 1 + int(search))) if good(r)), None)
    return before, after


# ---------------------------------------------------------------------------
# The outcome log


def outcomes_root(workspace: Any) -> Path:
    """The workspaces root a workspace lives in (where the shared outcome log is kept)."""

    return Path(workspace.path).parent


def outcomes_path(root: Path | str) -> Path:
    return Path(root) / OUTCOMES_FILE


def _append_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(_json_safe(record)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_outcome(root: Path | str, record: dict[str, Any]) -> None:
    _append_line(outcomes_path(root), {"kind": "run", **record})


def mark_accepted(
    root: Path | str, candidate_set_id: str, *, workspace: str = "", edit: str | None = None,
    rows: Sequence[int] = (), accepted_rows: Sequence[int] = (), complete: bool = True,
) -> None:
    """Record that (part of) a candidate set was accepted: an ``accepted`` line with the rows this edit took, all rows taken so far, and whether the path is complete."""

    _append_line(
        outcomes_path(root),
        {
            "kind": "accepted", "candidate_set": str(candidate_set_id), "workspace": workspace, "edit": edit, "time": utc_now(),
            "rows": [int(r) for r in rows], "accepted_rows": [int(r) for r in accepted_rows], "complete": bool(complete),
        },
    )


def mark_unaccepted(
    root: Path | str, candidate_set_id: str, *, workspace: str = "", edit: str | None = None, undoes: str | None = None,
    rows: Sequence[int] = (), accepted_rows: Sequence[int] = (),
) -> None:
    """Record that an accept of a candidate set was undone (``undoes`` names the accept's edit); ``accepted_rows`` is what remains accepted."""

    _append_line(
        outcomes_path(root),
        {
            "kind": "unaccepted", "candidate_set": str(candidate_set_id), "workspace": workspace, "edit": edit, "undoes": undoes,
            "time": utc_now(), "rows": [int(r) for r in rows], "accepted_rows": [int(r) for r in accepted_rows],
        },
    )


def outcomes(root: Path | str) -> list[dict[str, Any]]:
    """Every region run in the log, newest first, with the accept state folded in from the ``accepted`` and ``unaccepted`` lines in order.

    ``accepted`` is true only while the whole path is in the state (an undone
    accept clears it); ``accepted_rows`` lists the rows currently accepted,
    ``accepted_at`` and ``accepted_edit`` the latest accept.
    """

    path = outcomes_path(root)
    if not path.exists():
        return []
    runs: list[dict[str, Any]] = []
    marks: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        kind = record.get("kind", "run")
        key = (str(record.get("workspace") or ""), str(record.get("candidate_set") or ""))
        if kind == "accepted":
            # Lines from before partial accepts carry neither rows nor ``complete``: they accepted the whole path.
            marks[key] = {
                "accepted": bool(record.get("complete", True)), "accepted_at": record.get("time"), "accepted_edit": record.get("edit"),
                "accepted_rows": [int(r) for r in record.get("accepted_rows") or record.get("rows") or []],
            }
        elif kind == "unaccepted":
            remaining = [int(r) for r in record.get("accepted_rows") or []]
            marks[key] = {"accepted": False, "accepted_at": None, "accepted_edit": None, "accepted_rows": remaining, "unaccepted_at": record.get("time")}
        elif kind == "run":
            runs.append(record)
    for record in runs:
        key = (str(record.get("workspace") or ""), str(record.get("candidate_set") or ""))
        hit = marks.get(key)
        if hit is not None:
            record.update(hit)
        else:
            record.setdefault("accepted", False)
            record.setdefault("accepted_rows", [])
    runs.reverse()
    return runs


# ---------------------------------------------------------------------------
# Running a region


def run_region(
    workspace: Any,
    algorithm_id: str,
    first: int,
    last: int,
    params: dict[str, Any] | None = None,
    *,
    anchor_before: int | None = None,
    anchor_after: int | None = None,
    device: torch.device | str | None = None,
    progress: Progress | None = None,
    job: str = "",
) -> CandidateSet:
    """Run ``algorithm_id`` on rows ``first..last`` with these anchors; saves the set under ``candidates/`` and logs the outcome.

    The set's id is ``job`` when given (the job runner's id), else the next
    ``c<6 digits>`` of the workspace's counter.  The state is not changed.
    """

    algorithm = get_algorithm(algorithm_id)
    resolved = algorithm.resolve(params)  # type: ignore[attr-defined]
    algorithm.check_anchors(anchor_before, anchor_after)  # type: ignore[attr-defined]
    _report(progress, 0.0, f"{algorithm.id}: loading the region")
    ctx = build_context(workspace, first, last, anchor_before, anchor_after, device)
    before = region_metrics(ctx.state, ctx.rows, ctx.image_shape)
    started = time.perf_counter()
    candidate_set = algorithm.run(ctx, resolved, progress)
    candidate_set.metrics["seconds"] = time.perf_counter() - started
    candidate_set.metrics_before = before
    candidate_set.job = str(job or "")
    candidate_set.id = str(job) if job else next_candidate_id(workspace)
    candidate_set.workspace = str(workspace.info.name)
    candidate_set.recording = str(workspace.info.recording)
    save_candidate_set(workspace, candidate_set)
    frames = np.asarray(ctx.state["frame_index"], dtype=np.int64)
    append_outcome(
        outcomes_root(workspace),
        {
            "time": utc_now(), "workspace": str(workspace.info.name), "recording": str(workspace.info.recording),
            "first": ctx.first, "last": ctx.last, "frames": [int(frames[ctx.first]), int(frames[ctx.last])],
            "anchors": {"before": ctx.anchor_before, "after": ctx.anchor_after}, "algorithm": algorithm.id, "params": resolved,
            "metrics_before": before, "metrics_after": candidate_set.metrics, "candidate_set": candidate_set.id, "job": candidate_set.job,
            "candidates": int(sum(len(v) for v in candidate_set.candidates.values())), "path_rows": len(candidate_set.path), "accepted": False,
        },
    )
    _report(progress, 1.0, f"{algorithm.id}: {len(candidate_set.path)} frames on the path, median IoU {candidate_set.metrics.get('median_iou')}")
    return candidate_set


# ---------------------------------------------------------------------------
# Accepting a set


def _grow_hypotheses(hyps: dict[str, np.ndarray], n: int, slots: int, config: BatchFitConfig) -> dict[str, np.ndarray]:
    """The hypotheses arrays with at least ``slots`` candidates per row (existing candidates kept in place)."""

    current = int(hyps["hypotheses_energy"].shape[1]) if "hypotheses_energy" in hyps else 0
    if current >= slots and "hypotheses_energy" in hyps:
        fresh = empty_hypotheses(n, config, max(1, (current - 1) // 2))
        for key, value in fresh.items():
            hyps.setdefault(key, value)
        return hyps
    beam = max(1, math.ceil((slots - 1) / 2))
    fresh = empty_hypotheses(n, config, beam)
    for key, value in fresh.items():
        old = hyps.get(key)
        if old is None:
            continue
        if old.ndim >= 2 and key.startswith("hypotheses_") and key != "hypotheses_count":
            width = min(old.shape[1], value.shape[1])
            if old.shape[0] == n and old.shape[2:] == value.shape[2:]:
                value[:, :width] = old[:, :width]
        elif old.shape == value.shape:
            value[...] = old
        fresh[key] = value
    return fresh


def install_into(hyps: dict[str, np.ndarray], n: int, candidate_set: CandidateSet, rows: Sequence[int], config: BatchFitConfig) -> dict[str, np.ndarray]:
    """The hypotheses arrays with the set's candidates of ``rows`` in place of those rows' hypotheses (grown when a row has more candidates than slots).

    A stored hypothesis is what ``edits.accept_path`` and a manual pick can
    make a pose, and what the hypotheses table shows, so accepting a set
    first makes its candidates the rows' hypotheses.  This runs inside the
    accept's edit (``accept_path``'s ``install`` hook), after the snapshot of
    the rows' previous candidates is taken, so an undo brings them back.
    """

    wanted = [int(r) for r in rows if candidate_set.candidates.get(int(r))]
    if not wanted:
        return hyps
    slots = max(len(candidate_set.candidates[r]) for r in wanted)
    hyps = _grow_hypotheses(hyps, n, slots, config)
    for row in wanted:
        poses = candidate_set.candidates[row]
        for key, value in hyps.items():
            if key.startswith("hypotheses_") and key != "hypotheses_count":
                value[row] = np.nan if value.dtype.kind == "f" else (False if value.dtype.kind == "b" else (0 if value.dtype.kind in "iu" else ""))
        hyps["hypotheses_beam"][row] = -1
        for j, pose in enumerate(poses):
            hyps["hypotheses_centerline_xy"][row, j] = pose.centerline_xy
            hyps["hypotheses_energy"][row, j] = pose.energy
            hyps["hypotheses_iou"][row, j] = pose.iou
            hyps["hypotheses_source"][row, j] = pose.source
            hyps["hypotheses_start"][row, j] = pose.start
            hyps["hypotheses_beam"][row, j] = j
            hyps["hypotheses_latent"][row, j] = pose.latent
            hyps["hypotheses_width_px"][row, j] = pose.width_px
            if len(pose.width_shape) == hyps["hypotheses_width_shape"].shape[2]:
                hyps["hypotheses_width_shape"][row, j] = pose.width_shape
            hyps["hypotheses_width_profile"][row, j] = pose.width_profile
            hyps["hypotheses_body_length_px"][row, j] = pose.body_length_px
            hyps["hypotheses_points_in_fov"][row, j] = pose.points_in_fov
            hyps["hypotheses_crop"][row, j] = pose.crop
            hyps["hypotheses_soft_dice"][row, j] = pose.soft_dice
        hyps["hypotheses_count"][row] = len(poses)
        hyps["path_index"][row] = -1
        hyps["path_mirrored"][row] = False
        hyps["path_override"][row] = False
        hyps["path_energy_gap"][row] = np.nan
        hyps["path_cost"][row] = np.nan
    return hyps


def accept_candidates(workspace: Any, set_id: str, *, rows: Sequence[int] | None = None, use_path: bool = True, note: str = "") -> edits.EditResult:
    """Make the set's path the poses of ``rows`` (default: every row of the path not yet accepted) as one ``accept_path`` edit.

    The rows' hypotheses become the set's candidates inside the same edit
    (``install_into``), so an undo restores both the poses and the previous
    candidates.  Provenance of the rows becomes the set's algorithm id under
    the job ``candidates:<id>``; the set's metadata records the rows accepted
    and, once the whole path is in, ``accepted``; the outcome log gets an
    ``accepted`` line with the same.  Rows already accepted are refused (an
    accept repeated by a double click makes no second edit).  ``use_path``
    must be true: accepting candidates other than the path's is a manual
    pick per frame.
    """

    if not use_path:
        raise ValueError("only the set's path can be accepted as a whole; pick other candidates per frame")
    candidate_set = load_candidate_set(workspace, set_id)
    if not candidate_set.path:
        raise ValueError(f"candidate set {set_id} has an empty path")
    if candidate_set.accepted:
        raise ValueError(f"candidate set {set_id} is already accepted")
    already = {int(r) for r in candidate_set.accepted_rows}
    wanted = None if rows is None else {int(r) for r in rows}
    on_path = [(int(r), int(i), bool(m)) for r, i, m in candidate_set.path if wanted is None or int(r) in wanted]
    if not on_path:
        raise ValueError("none of the requested rows is on the set's path")
    choices = [c for c in on_path if c[0] not in already]
    if not choices:
        raise ValueError(f"the requested rows of candidate set {set_id} are already accepted")
    chosen_rows = [r for r, _, _ in choices]
    accepted_rows = sorted(already | set(chosen_rows))
    complete = set(accepted_rows) >= {int(r) for r, _, _ in candidate_set.path}
    config = workspace_setup(workspace).config
    n = int(workspace.n)
    job = f"candidates:{candidate_set.id}"
    result = edits.accept_path(
        workspace, choices, algorithm=candidate_set.algorithm, job=job, note=note or f"accept {candidate_set.algorithm} {candidate_set.id}",
        install=lambda hyps: install_into(hyps, n, candidate_set, chosen_rows, config),
        extra={"candidate_set": candidate_set.id, "complete": complete},
    )
    candidate_set.accepted = complete
    candidate_set.accepted_at = utc_now()
    candidate_set.accepted_edit = result.edit_id
    candidate_set.accepted_rows = accepted_rows
    save_candidate_set(workspace, candidate_set)
    mark_accepted(
        outcomes_root(workspace), candidate_set.id, workspace=str(workspace.info.name), edit=result.edit_id, rows=chosen_rows,
        accepted_rows=accepted_rows, complete=complete,
    )
    return result


def unaccept_candidates(workspace: Any, set_id: str, *, rows: Sequence[int], edit: str | None = None, undoes: str | None = None) -> bool:
    """Take ``rows`` out of a set's accepted rows after the accept that put them in was undone (``edits.undo`` calls this); whether the set was found.

    The set is no longer ``accepted``, so the viewer shows it again and it
    can be accepted anew; the outcome log gets an ``unaccepted`` line so the
    run does not count as a success.
    """

    try:
        candidate_set = load_candidate_set(workspace, set_id)
    except FileNotFoundError:
        # The set was discarded after the accept: only the log can record the undo.
        mark_unaccepted(outcomes_root(workspace), set_id, workspace=str(workspace.info.name), edit=edit, undoes=undoes, rows=rows, accepted_rows=[])
        return False
    remaining = sorted({int(r) for r in candidate_set.accepted_rows} - {int(r) for r in rows})
    candidate_set.accepted = False
    candidate_set.accepted_rows = remaining
    candidate_set.accepted_at = None
    candidate_set.accepted_edit = None
    save_candidate_set(workspace, candidate_set)
    mark_unaccepted(outcomes_root(workspace), set_id, workspace=str(workspace.info.name), edit=edit, undoes=undoes, rows=rows, accepted_rows=remaining)
    return True
