"""The algorithm registry: the pipeline's methods as plugins that run on a region of a workspace.

Phase 3 of ``docs/APP_PLAN.md`` (sections 2 and 6), behind the Workspace page's
Refit and Relabel fixes (``docs/APP_SIMPLIFICATION.md``).  A region is a run of
rows ``first..last`` of a workspace between two anchors, rows outside it
whose stored poses are trusted.  An algorithm takes the region's masks, the
workspace's fit configuration and prior, and the anchors, and produces
*candidate poses* per row; ``propagation.select_path`` then chooses one path
through them (the anchors fix its ends and its orientation), exactly as the
propagate stage does for a stretch.  Nothing here writes the state: a run
returns a ``CandidateSet`` (candidates, path, metrics before and after),
which the Refit and Relabel fixes (``worm_pose_gen.fixes``) save as a
preview and install through the edit log only when the user keeps it, so a
refit that comes back worse than the current track can be discarded.

The algorithms share the batched mask fitter and candidate storage:

- ``independent_multistart``: every row fit from the standard starts of its
  mask (both orientations when a prior or the network exists) and the
  network's trace, every start a candidate.
- ``chain_forward`` / ``chain_backward``: one chain from an anchor through the
  region with prediction, temporal prior and beam (``propagation.propagate``
  with one direction).
- ``beam_path``: the pipeline's second pass on the region, both chains plus
  the refit independent poses and anchor diversity.
- ``slow_refit``: the current poses refit under a longer schedule with the
  anchors' length prior.
- ``tracked_head``: a forward fit with acquisition nose landmarks, gentle
  body and strong head priors, and hard head movement and camera bounds.
- ``fixed_body_smoother``: no fitting; the current poses re-expressed as one
  fixed-length, fixed-width body and smoothed jointly under a calibrated
  first-order motion prior (``body_smoother``), untrusted frames bridged.
- ``mirror``: the current poses and their reversals, no fitting: an
  orientation fix over a region.

When the workspace was fit with the body-field network (its summary's
``fit_params.body_net``), every algorithm uses the network as the propagate
stage does: each fit is scored against the region's evidence
(``batch_fit.fit_masks(fields=...)``), the network's trace is one more start
wherever an algorithm builds starts, and every candidate's energy, the
smoother's and the stored poses' too, includes its evidence energy, a
mirrored option paying the mirror's own.  The smoother's joint energy does
not take the evidence: as A-P and end terms it did not help on the sequence
set (``docs/BODY_FIELDS.md``).

Anchors need not be adjacent to the region: the algorithms run on a *local*
copy of the arrays in which the anchors sit right next to the region, so the
chains and the path connect the region to the anchors the user chose.

``run_algorithm`` runs one algorithm on a region (the Refit fix).
``stitch`` serves the Relabel fix: it pins the poses of labeled keyframes
and refits every gap between consecutive keyframes as a propagate stretch
anchored on them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from .ambiguity import pose_jump_px
from .batch_fit import PRESETS, BatchFitConfig, BodyFieldEvidence, field_energies, fit_masks
from .body_smoother import SmoothingProblem, initial_chain, motion_scales, smooth_chains
from .fixed_body import calibrate_body, chain_targets, trusted_rows, whole_body_rows
from .latent import cubic_bspline_basis, decode_centerline, encode_centerline
from .head_fit import HeadConstraint
from .head_tracking import read_head_tracking
from .mask_fit import CropWindow, Initialization, MaskFitResult, crop_window, fill_narrow_holes, hard_iou, render_tube_segments, reverse_initialization, standard_initializations
from .observation import soft_dice_energy
from .pipeline import (
    SegmentParams,
    SOURCE_CODES,
    body_field_inputs,
    load_mask_model,
    read_summary,
    segment_frames,
    workspace_arrays,
    workspace_frames,
    workspace_image_shape,
    workspace_lock,
    workspace_masks_of,
    workspace_predictions,
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

PARAMETER_TYPES = ("int", "float", "bool", "str", "choice")
METRIC_NAMES = ("median_iou", "p10_iou", "frames_below_0_9", "pose_jumps_over_width", "length_jumps_over_3pct", "orientation_flips", "seconds")
SOURCE_DTYPE = "<U24"
START_DTYPE = "<U48"
# How far from a region ``propose_anchors`` looks for an anchor.
ANCHOR_SEARCH_ROWS = 200
# Trusted, fully visible frames nearest the region that calibrate the smoother's fixed body.
CALIBRATION_LIMIT = 100


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
            if not math.isfinite(out):
                raise ValueError(f"parameter {self.name!r}: must be finite")
            if self.minimum is not None and out < self.minimum:
                raise ValueError(f"parameter {self.name!r}: {out} is below the minimum {self.minimum}")
            if self.maximum is not None and out > self.maximum:
                raise ValueError(f"parameter {self.name!r}: {out} is above the maximum {self.maximum}")
        return out


def resolve_params(parameters: Sequence[Parameter], params: dict[str, Any] | None) -> dict[str, Any]:
    """Every declared parameter with the given value coerced or its default; unknown keys are ignored (as the stages do)."""

    given = dict(params or {})
    return {p.name: p.coerce(given.get(p.name)) for p in parameters}


_FILL_HOLES = Parameter("fill_holes", "choice", "workspace", "Use saved workspace masks, fill narrow holes for this refit, or resegment unedited frames without filling holes. Saved masks and manual edits are kept; ignored pixels stay excluded.", choices=["workspace", "on", "off"])

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
# Context and candidates


@dataclass
class RegionContext:
    """What an algorithm gets: the region, its anchors, the masks, and the workspace's fit setup.

    ``first..last`` are inclusive rows; ``anchor_before`` / ``anchor_after``
    are rows outside the region whose stored pose anchors chains and the
    path (``None`` = no anchor on that side).  ``masks`` holds the cleaned
    masks of the region rows and the anchors (rows without a usable mask are
    absent).  ``state`` is the workspace's state at build time.
    ``evidence`` maps a row of ``masks`` to the body-field network's
    evidence and ``network_starts`` a region row to the network's trace start
    (``pipeline.body_field_inputs``); both are empty when the workspace was
    not fit with the network.
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
    mask_revisions: dict[str, str] = field(default_factory=dict)
    evidence: dict[int, BodyFieldEvidence] = field(default_factory=dict)
    network_starts: dict[int, Initialization] = field(default_factory=dict)

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

    def fields_of(self, rows: Sequence[int]) -> list[BodyFieldEvidence | None] | None:
        """The ``fit_masks(fields=...)`` of these rows; ``None`` without a network."""

        return [self.evidence.get(int(r)) for r in rows] if self.evidence else None

    def with_trace(self, row: int, starts: list[Initialization]) -> list[Initialization]:
        """``starts`` plus the network's trace start of ``row`` when there is one."""

        trace = self.network_starts.get(int(row))
        return starts if trace is None else starts + [trace]


@dataclass
class CandidatePose:
    """One candidate pose of a row: everything ``pipeline.store_result`` writes, plus where it came from.

    ``energy`` is the comparable energy (``propagation.comparable_energy``:
    overlap plus the fit configuration's priors and the body-field evidence
    energy), ``soft_dice`` the overlap energy alone.  ``source`` is an
    algorithm-specific label (``forward``, ``backward``, ``independent``,
    ``current``, ``mirrored``), ``start`` the start that won inside the
    candidate's fit.  ``field_energy`` is the evidence part of ``energy`` and
    ``mirror_field_energy`` the evidence energy of the body traversed from
    the other end (``None`` without evidence; ``score_evidence``).
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
    field_energy: float = 0.0
    mirror_field_energy: float | None = None

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
            field_energy=float(best.get("final_field_energy", 0.0)),
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
        """The same body traversed from the other end (``mask_fit.reverse_result`` semantics), paying the mirror's evidence energy."""

        curve = self.centerline_xy[::-1].copy()
        evidence = {} if self.mirror_field_energy is None else {
            "energy": self.energy - self.field_energy + self.mirror_field_energy,
            "field_energy": self.mirror_field_energy, "mirror_field_energy": self.field_energy,
        }
        return replace(
            self,
            centerline_xy=curve,
            latent=encode_centerline(curve, coefficients),
            width_shape=self.width_shape[::-1].copy(),
            width_profile=self.width_profile[::-1].copy(),
            source=self.source if source is None else source,
            **evidence,
        )

    def to_pose(self) -> dict[str, Any]:
        """The ``edits.set_poses`` / ``pose_from_hypothesis`` field dictionary."""

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
        records=[{
            "name": pose.start, "final_energy": pose.energy, "final_iou": pose.iou, "final_soft_dice_energy": pose.soft_dice,
            "final_field_energy": pose.field_energy,
        }],
        latent=pose.latent, width_px=pose.width_px, width_profile=pose.width_profile, centerline_xy=pose.centerline_xy,
        crop=CropWindow(x0, x1, y0, y1, 0, 0), rendered_hard_mask=np.zeros((0, 0), dtype=bool), energy_history=np.zeros(0),
        points_in_fov=pose.points_in_fov, body_length_px=pose.body_length_px, width_shape=pose.width_shape,
    )
    return Candidate(pose.source, result, pose.energy, None, pose.start, float("nan"), beam=index, mirror_field_energy=pose.mirror_field_energy)


def score_evidence(ctx: RegionContext, candidates: dict[int, list[CandidatePose]]) -> None:
    """Score every candidate of a row with evidence, and its mirror, against that evidence (``batch_fit.field_energies``), in place.

    A fit already carries its evidence energy; a stored or placed pose (the
    current poses, the smoother's, a keyframe) gets it here, so every
    candidate of a frame competes on the same footing, and the mirror of each
    pays its own evidence energy in the path (``propagation.Candidate.energy``).
    """

    scored = [(row, pose) for row, poses in candidates.items() if row in ctx.evidence for pose in poses]
    if not scored:
        return
    curves = [c for _, pose in scored for c in (pose.centerline_xy, pose.centerline_xy[::-1])]
    values = field_energies(curves, [ctx.evidence[row] for row, _ in scored for _ in range(2)], ctx.config)
    for k, (_, pose) in enumerate(scored):
        own, mirror = float(values[2 * k]), float(values[2 * k + 1])
        pose.energy = pose.energy - pose.field_energy + own
        pose.field_energy, pose.mirror_field_energy = own, mirror


@dataclass
class CandidateSet:
    """What one region run produced: candidates per row, the path through them, and the region's metrics.

    ``mask_revisions`` records the masks the run fit, so whoever installs the
    path can tell whether the workspace changed underneath it.

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
    metrics_before: dict[str, Any] = field(default_factory=dict)
    frames: list[int] = field(default_factory=list)
    workspace: str = ""
    recording: str = ""
    mask_revisions: dict[str, str] = field(default_factory=dict)

    @property
    def path_by_row(self) -> dict[int, tuple[int, bool]]:
        return {int(row): (int(index), bool(mirrored)) for row, index, mirrored in self.path}

    def chosen(self, row: int) -> CandidatePose | None:
        """The path's candidate of ``row`` (mirrored as the path presents it), ``None`` when the path skipped the row."""

        choice = self.path_by_row.get(int(row))
        if choice is None:
            return None
        index, mirrored = choice
        pose = self.candidates[int(row)][index]
        return pose.mirrored(len(pose.latent) - 4) if mirrored else pose

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
    evidence: dict[int, BodyFieldEvidence]
    network_starts: dict[int, Initialization]

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
    evidence = {i: ctx.evidence[r] for i, r in enumerate(rows) if r in ctx.evidence}
    network_starts = {i: ctx.network_starts[r] for i, r in enumerate(rows) if r in ctx.network_starts}
    a = len(before)
    return _Local(arrays, masks, rows, (a, a + len(ctx.rows) - 1), evidence, network_starts)


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

    score_evidence(ctx, candidates)
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


# Algorithms that never fit a mask: no hole-filling control.
NO_FITTING = ("mirror", "fixed_body_smoother")


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
    fill_holes_default = "workspace"

    def parameter_specs(self) -> list[Parameter]:
        return list(self.parameters) + ([] if self.id in NO_FITTING else [replace(_FILL_HOLES, default=self.fill_holes_default)])

    def resolve(self, params: dict[str, Any] | None) -> dict[str, Any]:
        return resolve_params(self.parameter_specs(), params)

    def check_anchors(self, anchor_before: int | None, anchor_after: int | None) -> None:
        """``ValueError`` when an anchor this algorithm needs is missing."""

        for side, anchor in (("before", anchor_before), ("after", anchor_after)):
            if side in self.needs_anchor and anchor is None:
                raise ValueError(f"{self.id} needs an anchor {side} the region")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "scope": self.scope, "description": self.description,
            "parameters": [p.to_dict() for p in self.parameter_specs()], "needs_anchor": list(self.needs_anchor),
        }

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet:  # pragma: no cover - overridden
        raise NotImplementedError


class IndependentMultistart(_RegionAlgorithm):
    id = "independent_multistart"
    label = "Independent multi-start"
    description = (
        "Every frame of the region fit from the standard starts of its mask (skeleton and moment arcs; both orientations when a "
        "recording prior or the body-field network exists) and the network's trace, each start kept as a candidate. The path then "
        "picks one per frame with the anchors."
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
            # With the network's evidence, as in the fit stage, the evidence decides which end is the head.
            if ctx.prior is not None or ctx.evidence:
                starts = starts + [reverse_initialization(s, config=config) for s in starts]
            jobs.extend((row, s) for s in ctx.with_trace(row, starts))
        candidates: dict[int, list[CandidatePose]] = {r: [] for r in rows}
        chunk_size = max(1, int(config.max_rows))
        for k in range(0, len(jobs), chunk_size):
            chunk = jobs[k : k + chunk_size]
            results = fit_masks(
                [ctx.masks[r] for r, _ in chunk], [[s] for _, s in chunk], width_template=ctx.width_template, config=config, device=ctx.device,
                fields=ctx.fields_of([r for r, _ in chunk]),
            )
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
        propagation=propagation, warm_config=warm_config, evidence=local.evidence, network_starts=local.network_starts,
    )
    candidates: dict[int, list[CandidatePose]] = {}
    for local_row, options in raw.items():
        row = local.rows[local_row]
        ordered = sorted(options, key=lambda c: (SOURCE_CODES.get(c.source, 3), c.beam))
        candidates[row] = [CandidatePose.from_result(c.result, ctx.config, c.source, c.start_name, c.total_energy) for c in ordered]
    _fit_missing(ctx, candidates, ctx.fit_rows())
    return candidates, info


def _fit_missing(ctx: RegionContext, candidates: dict[int, list[CandidatePose]], rows: Sequence[int]) -> None:
    """Fit the ``rows`` no chain reached from their mask's standard starts, so every row with a mask gets a candidate."""

    for row in rows:
        if candidates.get(row):
            continue
        starts = ctx.with_trace(row, standard_initializations(ctx.masks[row], config=ctx.config))
        result = fit_masks([ctx.masks[row]], [starts], width_template=ctx.width_template, config=ctx.config, device=ctx.device, fields=ctx.fields_of([row]))[0]
        candidates[row] = [CandidatePose.from_result(result, ctx.config, "independent", "mask_refit")]


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
        config = replace(config, temporal_prior_weight=0.0, compile_energy=False)
        length = ctx.anchor_length()
        if length is not None:
            config = replace(config, length_prior_px=length)
        rows = ctx.fit_rows()
        candidates: dict[int, list[CandidatePose]] = {r: [] for r in rows}
        chunk_size = max(1, int(config.max_rows))
        for k in range(0, len(rows), chunk_size):
            chunk = rows[k : k + chunk_size]
            starts = [[warm_initialization(ctx.state["latent"][r], float(ctx.state["width_px"][r]), ctx.state["width_shape"][r], "slow_refit")]
                      if bool(ctx.state["fitted"][r]) else ctx.with_trace(r, standard_initializations(ctx.masks[r], config=config)) for r in chunk]
            results = fit_masks(
                [ctx.masks[r] for r in chunk], starts, width_template=ctx.width_template, config=config, device=ctx.device, fields=ctx.fields_of(chunk),
            )
            for row, result in zip(chunk, results, strict=True):
                candidates[row].append(CandidatePose.from_result(result, ctx.config, "independent", "slow_refit"))
            _report(progress, 0.05 + 0.85 * (k + len(chunk)) / max(len(rows), 1), f"{self.id}: refit {k + len(chunk)}/{len(rows)} frames")
        _report(progress, 0.92, f"{self.id}: selecting the path")
        return _assemble(ctx, self.id, params, candidates, _path_config(params))


class TrackedHead(_RegionAlgorithm):
    id = "tracked_head"
    label = "Head-tracked temporal fit"
    fill_holes_default = "off"
    description = (
        "Fits forward using the recording's nose tracking, a gentle pull toward the previous body pose, a strong head prior, "
        "and a hard head movement limit per recorded frame. Starts from the before anchor when selected; "
        "otherwise starts from the tracked nose. Low-confidence tracking falls back to the previous fit. "
        "Requires acquisition tracking in /pos_feature."
    )
    parameters = [
        _PRESET,
        Parameter("tracking_weight", "float", 0.2, "Pull the fitted head toward the acquisition nose landmark; larger values trust tracking more.", minimum=0.0),
        Parameter("previous_pose_weight", "float", 0.005, "Pull the body toward the previous fitted pose. Lower values let the tail follow the mask more freely.", minimum=0.0),
        Parameter("previous_head_weight", "float", 0.5, "Extra pull toward the previous fitted head to resist switching branches at intersections.", minimum=0.0),
        Parameter("head_sigma_px", "float", 6.0, "Distance in pixels used to scale the tracking and previous-head penalties.", minimum=0.1),
        Parameter("max_head_step_px", "float", 8.0, "Maximum head movement in image pixels per recorded frame; multiplied by the source-frame gap for sampled workspaces.", minimum=0.1),
        Parameter("keep_head_in_frame", "bool", True, "Keep the head inside the image. Turn off only for a region where the head leaves the camera.")
    ]

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet:
        params = self.resolve(params)
        rows = ctx.fit_rows()
        if not rows:
            raise ValueError("No usable masks in this region; segment or paint the region first.")
        if ctx.image_shape is None:
            raise ValueError("The head-tracked fit requires the recording's image dimensions.")
        _report(progress, 0.02, "Loading acquisition nose tracking")
        wanted = rows + ctx.anchors
        frames = np.asarray(ctx.state["frame_index"], dtype=np.int64)
        tracking = read_head_tracking(ctx.workspace.recording, frames[wanted], ctx.image_shape)
        heads = {row: tracking.xy[i] for i, row in enumerate(wanted) if tracking.valid[i]}
        if tracking.provenance.get("status") != "available" or (not heads and ctx.anchor_before is None):
            reason = tracking.provenance.get("reason", "no confident in-frame nose observations")
            raise ValueError(f"Head tracking unavailable for this region ({reason}). Choose another fitting method or a region with valid /pos_feature nose tracking.")
        config = _preset_schedule(ctx.config, params["preset"])
        if ctx.anchor_length() is not None:
            config = replace(config, length_prior_px=ctx.anchor_length())

        def oriented_start(start: Initialization, target: np.ndarray) -> Initialization:
            curve = decode_centerline(start.latent, config.coefficients)
            if np.linalg.norm(curve[-1] - target) < np.linalg.norm(curve[0] - target):
                return reverse_initialization(start, config=config)
            return start

        previous: CandidatePose | None = None
        previous_row: int | None = ctx.anchor_before
        # Selected anchors are trusted, fixed poses. Reorienting a local copy
        # would hide a jump at the unchanged workspace boundary.
        for row in ctx.anchors:
            cue = heads.get(row)
            curve = ctx.state["centerline_xy"][row]
            if cue is not None and np.linalg.norm(curve[-1] - cue) + float(ctx.state["width_px"][row]) < np.linalg.norm(curve[0] - cue):
                raise ValueError(f"Anchor at frame {frames[row]} has its head opposite the acquisition nose. Correct its orientation or choose another anchor before refitting.")
        if previous_row is not None:
            previous = CandidatePose.from_state(ctx.state, previous_row, config)
        if previous is None and rows[0] not in heads:
            raise ValueError("The first usable frame has no confident nose tracking. Choose a before anchor or start the region at a frame with valid tracking.")

        candidates: dict[int, list[CandidatePose]] = {}
        observed, fallback, max_motion = 0, 0, 0.0
        for i, row in enumerate(rows):
            cue = heads.get(row)
            reference = None if previous is None else previous.centerline_xy
            target = cue if reference is None else reference[0]
            starts: list[Initialization] = []
            if previous is not None:
                starts.append(warm_initialization(previous.latent, previous.width_px, previous.width_shape, "previous_pose"))
            if bool(ctx.state["fitted"][row]):
                start = warm_initialization(ctx.state["latent"][row], float(ctx.state["width_px"][row]), ctx.state["width_shape"][row], "current_pose")
                starts.append(oriented_start(start, target))
            if previous is None:
                for start in standard_initializations(ctx.masks[row], config=config):
                    start = oriented_start(start, target)
                    if ctx.prior is not None:
                        start = replace(start, width_shape=np.asarray(ctx.prior.width_shape, dtype=np.float64))
                    starts.append(start)
            if row in ctx.network_starts:
                starts.append(oriented_start(ctx.network_starts[row], target))
            gap = 1 if previous_row is None else max(1, int(frames[row] - frames[previous_row]))
            sigma = max(1.0, 0.5 * (previous.width_px if previous is not None else config.default_width_px))
            fit_config = replace(config, temporal_prior_weight=params["previous_pose_weight"], temporal_prior_sigma_px=sigma)
            _report(progress, 0.05 + 0.87 * i / len(rows), f"Fitting frame {frames[row]} ({i + 1}/{len(rows)}) with head movement constraints")
            result = fit_masks(
                [ctx.masks[row]], [starts], width_template=ctx.width_template, config=fit_config, device=ctx.device,
                references=[reference],
                head_constraints=[HeadConstraint(
                    tracking_xy=cue, previous_xy=None if reference is None else reference[0],
                    tracking_weight=params["tracking_weight"], previous_weight=params["previous_head_weight"],
                    sigma_px=params["head_sigma_px"], max_step_px=None if reference is None else params["max_head_step_px"] * gap,
                    keep_in_frame=params["keep_head_in_frame"],
                )],
                fields=ctx.fields_of([row]),
            )[0]
            pose = CandidatePose.from_result(result, ctx.config, "forward", energy=float(result.records[result.best_index]["final_energy"]))
            if reference is not None:
                max_motion = max(max_motion, float(np.linalg.norm(pose.centerline_xy[0] - reference[0])) / gap)
            candidates[row] = [pose]
            previous, previous_row = pose, row
            observed += int(cue is not None)
            fallback += int(cue is None)
            _report(progress, 0.05 + 0.87 * (i + 1) / len(rows), f"Head-tracked fit: {i + 1}/{len(rows)} frames · {'nose tracking' if cue is not None else 'previous-pose fallback'}")
        if ctx.anchor_after is not None:
            gap = max(1, int(frames[ctx.anchor_after] - frames[previous_row]))
            speed = float(np.linalg.norm(previous.centerline_xy[0] - ctx.state["centerline_xy"][ctx.anchor_after, 0])) / gap
            if speed > params["max_head_step_px"] + 1e-4:
                raise ValueError("The head-tracked fit cannot reconnect to the after anchor within the head movement limit. Widen the region, increase the limit, or choose another after anchor.")
            max_motion = max(max_motion, speed)
        # This is a sequential constrained solution. A later orientation search
        # must never reverse it and discard the head guarantee.
        candidate_set = _assemble(ctx, self.id, params, candidates, _path_config({}, path_mirrors=False))
        candidate_set.metrics.update({
            "head_tracking": dict(tracking.provenance), "tracked_frames": observed, "tracking_fallback_frames": fallback,
            "max_head_step_px_per_frame": max_motion,
        })
        return candidate_set


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


def _width_parameters(profile: np.ndarray, template: np.ndarray, config: BatchFitConfig) -> tuple[float, np.ndarray]:
    """The fit's ``(width_px, width_shape)`` that best reproduce a diameter ``profile`` over the width ``template``."""

    log_ratio = np.log(np.asarray(profile, dtype=np.float64) / np.asarray(template, dtype=np.float64))
    width_px = float(np.exp(log_ratio.mean()))
    if not config.width_coefficients:
        return width_px, np.zeros(0)
    basis = cubic_bspline_basis(config.n_points, config.width_coefficients)
    centered = basis - basis.mean(axis=0, keepdims=True)
    shape = np.linalg.lstsq(centered, log_ratio - log_ratio.mean(), rcond=None)[0]
    return width_px, shape.astype(np.float64)


def _overlap(points: np.ndarray, profile: np.ndarray, mask: MaskArray | None, config: BatchFitConfig, device: torch.device, image_shape: tuple[int, int]) -> tuple[np.ndarray, float, float]:
    """``(crop, soft_dice_energy, iou)`` of the tube along ``points`` with diameters ``profile`` against ``mask`` (NaN overlap without a mask)."""

    height, width = image_shape
    if mask is None:
        return np.asarray((0, width, 0, height), dtype=np.int64), float("nan"), float("nan")
    crop = crop_window(mask, config.crop_padding, max(config.stage_downsample))
    local = torch.as_tensor(points - np.array((crop.x0, crop.y0), dtype=np.float64), dtype=torch.float32, device=device)[None]
    rendered = render_tube_segments(local, torch.as_tensor(profile, dtype=torch.float32, device=device)[None], crop.height, crop.width, edge_softness=config.edge_softness)
    target = np.asarray(mask, dtype=bool)[crop.y0 : crop.y1, crop.x0 : crop.x1]
    dice = float(soft_dice_energy(rendered, torch.as_tensor(target, device=device))[0])
    iou = hard_iou((rendered[0] >= config.hard_threshold).cpu().numpy(), target)
    return np.asarray((crop.x0, crop.x1, crop.y0, crop.y1), dtype=np.int64), dice, iou


def _placed_pose(
    ctx: RegionContext, row: int, points: np.ndarray, profile: np.ndarray, width_px: float, width_shape: np.ndarray, source: str, start: str
) -> CandidatePose:
    """A pose that was placed rather than fit (smoothed, or a label's), scored against the row's mask like a fit."""

    assert ctx.image_shape is not None
    crop, dice, iou = _overlap(points, profile, ctx.masks.get(row), ctx.config, ctx.device, ctx.image_shape)
    body_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    energy = dice + prior_penalty(ctx.config, body_length, width_px, width_shape) if math.isfinite(dice) else float("nan")
    height, width = ctx.image_shape
    in_fov = int(np.sum((points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)))
    return CandidatePose(
        centerline_xy=points, latent=encode_centerline(points, ctx.config.coefficients), width_px=width_px, width_shape=width_shape,
        width_profile=np.asarray(profile, dtype=np.float64), body_length_px=body_length, points_in_fov=in_fov, crop=crop,
        energy=float(energy), soft_dice=dice, iou=iou, source=source, start=start,
    )


class FixedBodySmoother(_RegionAlgorithm):
    id = "fixed_body_smoother"
    label = "Fixed-body temporal smoother"
    description = (
        "No mask fitting: the current poses are re-expressed as one fixed-length, fixed-width body and smoothed together "
        "across the region under a first-order motion prior on head position and bending, calibrated from the workspace's "
        "trusted frames. Trusted frames keep their pose; low-overlap or ambiguous frames are bridged from their neighbours. "
        "Head-to-tail orientation follows the anchor before the region, else its first frame; correct flipped anchors first."
    )
    parameters = [
        Parameter("min_iou", "float", 0.9, "Frames with at least this overlap and an ambiguity score below 2 are trusted: they calibrate the body and its motion, and keep their full data weight.", minimum=0.0, maximum=1.0),
        Parameter("untrusted_weight", "float", 0.0, "Data weight of frames that are not trusted, relative to a trusted frame (0 = shaped by the neighbours alone). A hundred points pull hard, so keep this near zero.", minimum=0.0, maximum=1.0),
        Parameter("data_sigma_px", "float", 1.0, "Pixel scale of the pull toward a trusted frame's current pose; larger values let the prior move trusted frames more.", minimum=0.05),
        Parameter("motion_tolerance", "float", 2.0, "Prior sigma as a multiple of the typical frame-to-frame motion of trusted frames; larger values smooth less.", minimum=0.1),
        Parameter("min_calibration_frames", "int", 3, "Trusted, fully visible frames needed to calibrate the body length and width profile.", minimum=1),
    ]

    def run(self, ctx: RegionContext, params: dict[str, Any], progress: Progress | None = None) -> CandidateSet:
        params = self.resolve(params)
        if ctx.image_shape is None:
            raise ValueError("The fixed-body smoother requires the recording's image dimensions.")
        state, config, shape = ctx.state, ctx.config, ctx.image_shape
        frames = np.asarray(state["frame_index"], dtype=np.int64)
        fitted = np.asarray(state["fitted"], dtype=bool)
        curves = np.asarray(state["centerline_xy"], dtype=np.float64)
        rows = [r for r in ctx.rows if fitted[r]]
        if not rows:
            raise ValueError("No fitted poses in this region; run a fitting method first.")
        segments = config.n_points - 1
        _report(progress, 0.05, f"{self.id}: calibrating the body from trusted frames")
        calibration, length, profile = calibrate_body(
            ctx.workspace, state, samples=config.n_points, min_iou=params["min_iou"], min_anchors=params["min_calibration_frames"],
            near=(ctx.first + ctx.last) // 2, limit=CALIBRATION_LIMIT,
        )
        step = length / segments
        basis = cubic_bspline_basis(segments, config.coefficients)
        _report(progress, 0.2, f"{self.id}: calibrating the motion prior")
        whole = np.flatnonzero(trusted_rows(state, params["min_iou"]) & whole_body_rows(state, shape))
        scales = motion_scales([initial_chain(curves[r], length, segments) for r in whole], frames[whole], basis, max_gap=2 * frame_step(frames))
        nodes = ([ctx.anchor_before] if ctx.anchor_before is not None else []) + rows + ([ctx.anchor_after] if ctx.anchor_after is not None else [])
        fixed = np.array([node in ctx.anchors for node in nodes])
        # Orientation follows the chain from its first node; anchors are trusted and never reoriented.
        oriented: dict[int, np.ndarray] = {}
        flips, previous = 0, None
        for node in nodes:
            curve = curves[node]
            if previous is not None:
                same = np.linalg.norm(curve[0] - previous[0]) + np.linalg.norm(curve[-1] - previous[-1])
                swapped = np.linalg.norm(curve[0] - previous[-1]) + np.linalg.norm(curve[-1] - previous[0])
                if swapped < same and node in ctx.anchors:
                    raise ValueError(f"The anchor at frame {frames[node]} is oriented opposite to the chain reaching it. Correct its orientation or choose another anchor.")
                if swapped < same:
                    curve, flips = curve[::-1], flips + 1
            oriented[node] = curve
            previous = curve
        trusted = trusted_rows(state, params["min_iou"])
        count = len(nodes)
        initial = np.stack([initial_chain(oriented[node], length, segments) for node in nodes])
        targets, weights = initial.copy(), np.zeros((count, segments + 1))
        without_targets = 0
        for k, node in enumerate(nodes):
            sampled = chain_targets(oriented[node], length, segments, shape, stop_at_exit=True)
            if "status" in sampled:
                without_targets += int(not fixed[k])
                continue
            supported = int(sampled["count"])
            targets[k, : supported + 1] = sampled["targets"]
            weights[k, : supported + 1] = 1.0 if trusted[node] else params["untrusted_weight"]
        problem = SmoothingProblem(targets, weights, np.diff(frames[nodes]), fixed, initial, step, config.coefficients)
        _report(progress, 0.3, f"{self.id}: smoothing {len(rows)} frames jointly")
        chains, solver = smooth_chains(problem, scales, data_sigma_px=params["data_sigma_px"], tolerance=params["motion_tolerance"])
        width_px, width_shape = _width_parameters(profile, ctx.width_template, config)
        candidates: dict[int, list[CandidatePose]] = {}
        for k, node in enumerate(nodes):
            if fixed[k]:
                continue
            candidates[node] = [_placed_pose(ctx, node, chains[k], profile, width_px, width_shape, "smoothed", "fixed_body")]
            _report(progress, 0.6 + 0.35 * (k + 1) / count, f"{self.id}: overlap of frame {frames[node]} ({k + 1}/{count})")
        head_steps = np.linalg.norm(np.diff(chains[:, 0], axis=0), axis=1) / np.maximum(np.diff(frames[nodes]), 1) if count > 1 else np.zeros(0)
        # The chain fixes the orientation; a later orientation search must not reverse it.
        candidate_set = _assemble(ctx, self.id, params, candidates, _path_config({}, path_mirrors=False))
        candidate_set.metrics.update({
            "fixed_body": {"length_px": length, "segment_length_px": step, "calibration_frames": [int(frames[r]) for r in calibration], "motion_scales": scales.to_dict()},
            "solver": solver, "orientation_flips_applied": flips, "frames_without_targets": without_targets,
            "max_head_step_px_per_frame": float(head_steps.max()) if len(head_steps) else 0.0,
        })
        return candidate_set


REGISTRY: dict[str, Algorithm] = {
    algorithm.id: algorithm for algorithm in (IndependentMultistart(), ChainForward(), ChainBackward(), BeamPath(), SlowRefit(), TrackedHead(), FixedBodySmoother(), Mirror())
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
    """The segment stage's settings as the workspace recorded them (mask source, checkpoint, threshold, cleanup); the defaults otherwise."""

    summary = read_summary(workspace)
    settings = getattr(workspace.info, "settings", None) or {}
    values: dict[str, Any] = {}
    fingerprint = settings.get("checkpoint", summary.get("checkpoint", "unset"))
    if fingerprint is None:
        values["checkpoint"] = None
    elif isinstance(fingerprint, dict) and fingerprint.get("path"):
        values["checkpoint"] = str(fingerprint["path"])
    elif isinstance(fingerprint, str) and fingerprint != "unset":
        values["checkpoint"] = fingerprint
    if (settings.get("mask_source") or summary.get("mask_source")) == "body_net":
        # The summary's checkpoint is then the body-field network's.
        values["mask_source"] = "body_net"
        values["body_net"] = settings.get("body_net") or (summary.get("checkpoint") or {}).get("path")
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
        model = load_mask_model(params, frames, device)
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
    fill_holes: str = "workspace",
    trace_starts: bool = True,
) -> RegionContext:
    """The ``RegionContext`` of rows ``first..last`` with these anchors.

    Masks come from the workspace (overrides first); rows without a stored
    mask are segmented from the recording with the workspace's segment
    settings when ``segment_missing`` is set (a workspace fit before masks were stored has none).
    Explicit hole filling affects this run only. Off resegments unedited rows
    because stored masks may already contain filled pixels; manual masks stay
    authoritative. On fills narrow holes but never includes ignored pixels.
    Anchors must be fitted rows outside the region.  When the workspace was
    fit with the body-field network, every row with a mask is predicted and
    gets its evidence, and with ``trace_starts`` every region row its trace
    start (``pipeline.body_field_inputs``).
    """

    fill_holes = _FILL_HOLES.coerce(fill_holes)
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
    fresh: dict[int, MaskArray] = {}
    if fill_holes == "off":
        # Filled pixels cannot be recovered from stored binary masks. Recreate
        # unedited targets from the recording, keeping manual labels authoritative.
        unedited = [row for row in wanted if workspace.get_override_mask(row) is None]
        fresh = segment_rows(workspace, unedited, resolved, replace(_segment_params(workspace), fill_holes=False))
    elif missing and segment_missing:
        fresh = segment_rows(workspace, missing, resolved)
        masks.update(fresh)
        # Keep them: the next region run on these frames (another algorithm to
        # compare, a wider region) reads them instead of loading the segmenter again.
        if fill_holes == "workspace":
            store_masks(workspace, fresh)
    hole_radius = _segment_params(workspace).hole_radius if fill_holes == "on" else 0
    # Capture target pixels and their fingerprints under the same writer lock.
    # A direct client can edit while missing masks are segmented above.
    with workspace_lock(workspace):
        if hasattr(workspace, "clear_mask_cache"):
            workspace.clear_mask_cache()
        state = workspace_arrays(workspace, setup.config)
        for anchor in (anchor_before, anchor_after):
            if anchor is not None and not bool(state["fitted"][anchor]):
                raise ValueError(f"anchor row {anchor} changed while loading the region; choose anchors again")
        masks = {}
        for row in wanted:
            override = workspace.get_override_mask(row)
            mask = workspace.effective_mask(row)
            if fill_holes == "off" and override is None:
                mask = fresh.get(row)
            elif mask is None:
                mask = fresh.get(row)
            if mask is not None and fill_holes == "on":
                mask, _ = fill_narrow_holes(mask, hole_radius, device=resolved)
                if override is not None:
                    mask = mask & (override != 255)
            if mask is not None and int(np.asarray(mask).sum()) >= max(1, min_pixels):
                masks[row] = mask
        mask_revisions = {str(row): workspace.mask_revision(row) for row in wanted}
    with workspace_predictions(workspace, _segment_params(workspace), resolved) as predictions_of:
        evidence, network_starts = body_field_inputs(predictions_of, masks, setup, range(first, last + 1) if trace_starts else ())
    return RegionContext(
        workspace=workspace, first=first, last=last, anchor_before=anchor_before, anchor_after=anchor_after, masks=masks,
        config=setup.config, prior=setup.prior, width_template=setup.template, device=resolved, state=state, image_shape=workspace_image_shape(workspace),
        mask_revisions=mask_revisions, evidence=evidence, network_starts=network_starts,
    )


# ---------------------------------------------------------------------------
# Anchors


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
# Running a region


def run_algorithm(
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
) -> CandidateSet:
    """Run ``algorithm_id`` on rows ``first..last`` with these anchors and return its candidates, path and metrics; nothing is saved.

    ``metrics_before`` describes the state the algorithm saw and
    ``mask_revisions`` the masks it fit, so whoever installs the result can
    tell whether the workspace changed underneath it.
    """

    algorithm = get_algorithm(algorithm_id)
    resolved = algorithm.resolve(params)  # type: ignore[attr-defined]
    algorithm.check_anchors(anchor_before, anchor_after)  # type: ignore[attr-defined]
    preparation = "resegmenting unedited masks without hole filling" if resolved.get("fill_holes") == "off" else "loading region masks"
    _report(progress, 0.0, f"{algorithm.id}: {preparation}")
    ctx = build_context(
        workspace, first, last, anchor_before, anchor_after, device, fill_holes=resolved.get("fill_holes", "workspace"),
        trace_starts=algorithm.id not in NO_FITTING,
    )
    before = region_metrics(ctx.state, ctx.rows, ctx.image_shape)
    started = time.perf_counter()
    candidate_set = algorithm.run(ctx, resolved, progress)
    candidate_set.metrics["seconds"] = time.perf_counter() - started
    candidate_set.metrics_before = before
    candidate_set.mask_revisions = dict(ctx.mask_revisions)
    candidate_set.workspace = str(workspace.info.name)
    candidate_set.recording = str(workspace.info.recording)
    return candidate_set


# ---------------------------------------------------------------------------
# Stitching keyframes


STITCH = "stitch"


@dataclass
class Keyframe:
    """A pose a person fixed at one row: a head-first centerline in image ``(x, y)`` and the body's diameter at each of its points.

    This is the body fit of a label (``body_fields``: ``centerline_xy`` and
    ``width_profile``); how labels are stored does not matter here.  Any
    number of points is accepted and resampled to the fit's.
    """

    row: int
    centerline_xy: np.ndarray
    width_profile: np.ndarray


def _resample_body(centerline_xy: Any, width_profile: Any, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """The centerline and its diameters at ``n_points`` points evenly spaced in arc length (unchanged when there are ``n_points`` already)."""

    points = np.asarray(centerline_xy, dtype=np.float64)
    profile = np.asarray(width_profile, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("a keyframe's centerline_xy must have shape [N>=2, 2]")
    if profile.shape != (len(points),):
        raise ValueError("a keyframe's width_profile needs one diameter per centerline point")
    if not (np.isfinite(points).all() and np.isfinite(profile).all() and (profile > 0).all()):
        raise ValueError("a keyframe's centerline and widths must be finite and its widths positive")
    if len(points) == n_points:
        return points.copy(), profile.copy()
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if arc[-1] <= 0:
        raise ValueError("a keyframe's centerline has zero length")
    target = np.linspace(0.0, arc[-1], n_points)
    resampled = np.column_stack((np.interp(target, arc, points[:, 0]), np.interp(target, arc, points[:, 1])))
    return resampled, np.interp(target, arc, profile)


def keyframe_pose(ctx: RegionContext, keyframe: Keyframe) -> CandidatePose:
    """The keyframe as a stored pose: resampled to the fit's points, widths as the fit's scale and shape, scored against the row's mask."""

    points, profile = _resample_body(keyframe.centerline_xy, keyframe.width_profile, ctx.config.n_points)
    width_px, width_shape = _width_parameters(profile, ctx.width_template, ctx.config)
    return _placed_pose(ctx, int(keyframe.row), points, profile, width_px, width_shape, "keyframe", "label")


def stitch(
    workspace: Any,
    keyframes: Sequence[Keyframe],
    params: dict[str, Any] | None = None,
    *,
    device: torch.device | str | None = None,
    progress: Progress | None = None,
) -> CandidateSet:
    """Pin the keyframes' poses and refit every gap between consecutive keyframes with those two as fixed anchors; nothing is saved.

    The gaps are refit the way ``beam_path`` refits a region (its parameters
    apply): the gaps are the propagate stage's stretches, every keyframe
    anchors the forward chain of the gap after it and the backward chain of
    the gap before it, all chains run in one lockstep batch, and the path
    through each gap is tied to its two keyframes, which fixes its
    orientation head first.  Only the rows from the first keyframe to the
    last are seen, so frames outside the stretch, which was relabeled
    because it went wrong, give the chains neither velocity nor starts.
    The set covers those rows; a keyframe row has its pinned pose as its only
    candidate, so the path places the keyframes as well.
    """

    params = REGISTRY["beam_path"].resolve(params)  # type: ignore[attr-defined]
    rows = sorted(int(k.row) for k in keyframes)
    if not rows:
        raise ValueError("stitching needs at least one keyframe")
    if len(set(rows)) != len(rows):
        raise ValueError("two keyframes are on the same frame")
    first, last = rows[0], rows[-1]
    _report(progress, 0.0, "stitch: loading masks")
    ctx = build_context(workspace, first, last, None, None, device, fill_holes=params["fill_holes"])
    if ctx.image_shape is None:
        raise ValueError("stitching requires the recording's image dimensions")
    before = region_metrics(ctx.state, ctx.rows, ctx.image_shape)
    started = time.perf_counter()
    pinned = {int(k.row): keyframe_pose(ctx, k) for k in keyframes}
    index = np.arange(first, last + 1)
    local = {k: v[index].copy() for k, v in ctx.state.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == ctx.n}
    for row, pose in pinned.items():
        edits._write_pose(local, row - first, pose.to_pose())
    gaps = [(a + 1 - first, b - 1 - first) for a, b in zip(rows, rows[1:]) if b - a > 1]
    candidates: dict[int, list[CandidatePose]] = {row: [pose] for row, pose in pinned.items()}
    path = [(row, 0, False) for row in rows]
    if gaps:
        propagation = _path_config(params)
        warm = None if params["preset"] == "fast" else _preset_schedule(ctx.config, params["preset"], propagation.chain_length_sigma)
        _report(progress, 0.05, f"stitch: refitting {len(gaps)} gaps between {len(rows)} keyframes")
        raw, _ = propagate(
            local, gaps, {r - first: m for r, m in ctx.masks.items() if first <= r <= last}, config=ctx.config, device=ctx.device,
            width_template=ctx.width_template, propagation=propagation, warm_config=warm,
            evidence={r - first: e for r, e in ctx.evidence.items()}, network_starts={r - first: s for r, s in ctx.network_starts.items()},
            progress=None if progress is None else (lambda p, m: _report(progress, 0.05 + 0.85 * p, f"stitch: {m}")),
        )
        for local_row, options in raw.items():
            ordered = sorted(options, key=lambda c: (SOURCE_CODES.get(c.source, 3), c.beam))
            candidates[local_row + first] = [CandidatePose.from_result(c.result, ctx.config, c.source, c.start_name, c.total_energy) for c in ordered]
        gap_rows = [r for a, b in gaps for r in range(a + first, b + first + 1)]
        _fit_missing(ctx, candidates, [r for r in gap_rows if r in ctx.masks])
        score_evidence(ctx, {r: candidates[r] for r in gap_rows if candidates.get(r)})
        _report(progress, 0.92, "stitch: selecting the path through every gap")
        offered = {r - first: [_as_candidate(p, j) for j, p in enumerate(candidates[r])] for r in gap_rows if candidates.get(r)}
        chosen = select_path(offered, local, gaps, ctx.config, propagation, ctx.image_shape)
        for local_row, choice in chosen.items():
            index_of = next(j for j, c in enumerate(offered[local_row]) if c is choice.candidate)
            path.append((local_row + first, index_of, bool(choice.mirrored)))
    frames = np.asarray(ctx.state["frame_index"], dtype=np.int64)
    candidate_set = CandidateSet(
        algorithm=STITCH, params=_json_safe(params), first=first, last=last, anchor_before=None, anchor_after=None, rows=ctx.rows,
        candidates={r: candidates.get(r, []) for r in ctx.rows}, path=sorted(path), metrics={}, metrics_before=before,
        frames=[int(frames[first]), int(frames[last])], workspace=str(workspace.info.name), recording=str(workspace.info.recording),
        mask_revisions=dict(ctx.mask_revisions),
    )
    candidate_set.metrics = metrics_with_path(ctx.state, candidate_set, ctx.rows, ctx.image_shape)
    candidate_set.metrics["seconds"] = time.perf_counter() - started
    _report(progress, 1.0, f"stitch: {len(candidate_set.path)} frames placed between {len(rows)} keyframes")
    return candidate_set


# ---------------------------------------------------------------------------
# Installing a result


def validate_mask_revisions(workspace: Any, revisions: dict[str, str], rows: Sequence[int],
                            anchors: Sequence[int | None], set_id: str = "") -> None:
    """``ValueError`` when the masks a result was fit to (``revisions``, by row) changed since: it must be rerun, not installed."""
    # Another workspace instance may have rewritten a chunk since it was cached.
    if hasattr(workspace, "clear_mask_cache"):
        workspace.clear_mask_cache()
    if revisions:
        changed = [int(row) for row, revision in revisions.items() if workspace.mask_revision(int(row)) != revision]
        if changed:
            raise ValueError(f"{set_id} is stale: masks changed at rows {changed}; run it again")
    else:
        # Without fingerprints (no stored mask in the region) an edited mask
        # cannot be ruled out, even after a later undo.
        affected = set(rows) | {r for r in anchors if r is not None}
        if affected.intersection(workspace.override_rows()) or any(
            e.get("kind") in ("set_mask", "clear_mask") and affected.intersection(e.get("payload", {}).get("rows", []))
            for e in workspace.edits()
        ):
            raise ValueError(f"{set_id} has unversioned masks that were edited; run it again")
