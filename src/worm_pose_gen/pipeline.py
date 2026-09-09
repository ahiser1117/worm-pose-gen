"""The pose pipeline as stages over a workspace.

``scripts/fit_recording.py`` grew into one long function that segments,
bootstraps a prior, fits, scores, propagates, refits and summarises.  The app
(docs/APP_PLAN.md, section 5) needs each of those steps as a job that can be
rerun on its own over a workspace, so the per-frame logic lives here and the
script composes it.  Every stage reads its inputs from the workspace and
writes its outputs back (``worm_pose_gen.workspace``): masks and their
statistics, the recording prior, the per-frame state arrays in the
``poses.npz`` layout, the hypotheses, provenance for the rows a stage
touched, and a ``summary.json`` in the workspace directory with the same
shape as a run's summary so the viewer's loaders can read a workspace like a
run.

Stages and what they read and write:

- ``segment``: recording frames -> ``masks/`` and the mask statistics in the
  state (``worm_pixels``, ``raw_worm_pixels``, ``pixels_filled``,
  ``components``, ``pixels_outside_largest``, ``mask_on_border``).
- ``prior``: frames spread over the whole recording -> ``recording_prior.json``
  (bootstrap, the cache, or a given file).
- ``fit``: masks + prior -> the independent multi-start fit of every row with
  a mask (algorithm ``independent_fit``).
- ``ambiguity``: state -> the per-frame signals, flags and score.
- ``propagate``: state + masks -> chains through the ambiguous stretches, one
  path per stretch, the hypotheses arrays (``chain_forward``,
  ``chain_backward``, ``independent_refit``).
- ``track``: the track-length pass over clipped and deviating frames
  (``track_length_refit``).
- ``export``: one Parquet row per frame under ``exports/``.

Each stage's parameters are a dataclass whose defaults match the script's
flags; ``from_dict`` ignores unknown keys, so one parameter dict can drive
every stage.  With ``checkpoint=None`` the segmenter is replaced by a
threshold on dark pixels (below 128), which keeps the stages testable on a
synthetic recording without the network.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields as dataclass_fields, replace
import fcntl
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Protocol, Sequence, TypeVar

import h5py
import numpy as np
from numpy.typing import NDArray
import torch

from .ambiguity import FLAG_NAMES, compute_ambiguity, summarize_ambiguity
from .batch_fit import PRESETS, BatchFitConfig, fit_masks
from .label_app import DATASET_PATH, RecordingSource
from .mask_fit import (
    Initialization,
    MaskFitResult,
    default_width_template,
    extend_start_to_length,
    max_bend_widths,
    orient_tail_last,
    orientation_pair,
    reverse_result,
    standard_initializations,
    taper_asymmetry,
)
from .pose_run import clean_mask, flat_fielded, touches_border
from .propagation import (
    PathChoice,
    PropagationConfig,
    ambiguous_stretches,
    continuity_summary,
    jump_seeds,
    propagate,
    select_candidates,
    select_path,
    slow_schedule,
    track_length,
    warm_initialization,
    warm_schedule,
)
from .recording_prior import RecordingPrior, bootstrap_prior_from_masks
from .run_records import checkpoint_fingerprint, timestamp_slug, utc_now
from .segmentation_dataset import DEFAULT_DATASET_ROOT
from .workspace import MASK_CHUNK_ROWS, pack_mask


STAGES = ("segment", "prior", "fit", "ambiguity", "propagate", "track", "export")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "best.ckpt"
EXTERNAL_ROOT = Path(os.environ.get("WORM_POSE_EXTERNAL_ROOT", "/temp_data4/alex/external_artifacts"))
DEFAULT_PRIOR_CACHE = EXTERNAL_ROOT / "recording_priors"
HOLE_FILL_RADIUS_PX = 8
MIN_WORM_PIXELS = 500
TIMING_STAGES = ("read", "flat_field", "network", "cleanup", "init", "fit", "video")
START_SETS: dict[str, tuple[str, ...] | None] = {
    "skeleton": ("skeleton_longest_path",),
    "skeleton+straight": ("skeleton_longest_path", "moments_straight"),
    "skeleton+reversed": ("skeleton_longest_path", "skeleton_longest_path_reversed"),
    "all": None,
}
SOURCE_CODES = {"independent": 0, "forward": 1, "backward": 2}
# Provenance algorithm per ``source`` code of a stored pose.
SOURCE_ALGORITHMS = {0: "independent_fit", 1: "chain_forward", 2: "chain_backward"}
# The candidate a path chose came from one of these; the independent
# candidate of a stretch frame is a refit under the chain schedule.
CANDIDATE_ALGORITHMS = {"independent": "independent_refit", "forward": "chain_forward", "backward": "chain_backward"}
TRACK_ALGORITHM = "track_length_refit"
# Provenance of poses a propagation pass put in place of the independent fit.
PROPAGATION_ALGORITHMS = tuple(CANDIDATE_ALGORITHMS.values())
SUMMARY_FILE = "summary.json"
WORKSPACE_LOCK_FILE = ".lock"
# The per-frame arrays ``store_result`` writes, and the name each is kept
# under so the independent fit survives propagation and can be put back.
INDEPENDENT_COPIES = (
    ("iou_independent", "iou"),
    ("centerline_xy_independent", "centerline_xy"),
    ("width_profile_independent", "width_profile"),
    ("body_length_independent", "body_length_px"),
    ("latent_independent", "latent"),
    ("width_px_independent", "width_px"),
    ("width_shape_independent", "width_shape"),
    ("taper_asymmetry_independent", "taper_asymmetry"),
    ("energy_independent", "energy"),
    ("total_energy_independent", "total_energy"),
    ("points_in_fov_independent", "points_in_fov"),
    ("crop_independent", "crop"),
    ("tube_coverage_independent", "tube_coverage"),
    ("max_bend_widths_independent", "max_bend_widths"),
    ("best_start_independent", "best_start"),
)
BEST_START_DTYPE = "<U48"

Progress = Callable[[float, str], None]
MaskArray = NDArray[np.bool_]
P = TypeVar("P", bound="_Params")


def _help(text: str, **kwargs: Any) -> Any:
    return field(metadata={"help": text}, **kwargs)


@dataclass
class _Params:
    """Base of the stage parameter dataclasses: built from a dict, unknown keys ignored."""

    @classmethod
    def from_dict(cls: type[P], values: dict[str, Any] | None) -> P:
        known = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in (values or {}).items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentParams(_Params):
    checkpoint: str | None = _help("segmenter checkpoint; None thresholds pixels darker than 128 instead (tests)", default=str(DEFAULT_CHECKPOINT))
    threshold: float = _help("probability at or above which a pixel is worm", default=0.5)
    hole_radius: int = _help("largest hole width to fill, in pixels", default=HOLE_FILL_RADIUS_PX)
    fill_holes: bool = _help("fill narrow holes in the mask", default=True)
    largest_only: bool = _help("keep only the largest connected component", default=True)
    min_worm_pixels: int = _help("smaller cleaned masks are not fit", default=MIN_WORM_PIXELS)
    batch_size: int = _help("frames per segmenter forward pass", default=16)
    slab: int = _help("frames read from disk together", default=64)
    dataset_root: str = _help("where the flat field cache lives (<root>/flat_fields)", default=str(DEFAULT_DATASET_ROOT))
    flat_field: bool = _help("flat-field frames with the per-recording correction before segmenting", default=True)


@dataclass
class PriorParams(_Params):
    prior: str = _help("'bootstrap' a recording prior or 'none' to fit with the hard bounds", default="bootstrap")
    prior_file: str | None = _help("use this recording_prior.json instead of bootstrapping", default=None)
    prior_cache: str = _help("directory of cached priors, one per recording and coefficient count", default=str(DEFAULT_PRIOR_CACHE))
    use_cache: bool = _help("read and write the prior cache", default=True)
    rebootstrap: bool = _help("ignore a cached prior and bootstrap again", default=False)
    bootstrap_frames: int = _help("frames spread over the recording for the bootstrap pass", default=64)
    bootstrap_target: int = _help("whole-worm fits wanted; the sample is enlarged up to 4x to reach it", default=12)
    bootstrap_preset: str = _help("fitting schedule of the bootstrap pass", default="balanced")
    # Segmentation of the bootstrap frames (the same switches as the segment stage).
    checkpoint: str | None = _help("segmenter checkpoint for the bootstrap frames", default=str(DEFAULT_CHECKPOINT))
    threshold: float = _help("probability threshold", default=0.5)
    hole_radius: int = _help("largest hole width to fill", default=HOLE_FILL_RADIUS_PX)
    fill_holes: bool = _help("fill narrow holes", default=True)
    largest_only: bool = _help("keep the largest component", default=True)
    min_worm_pixels: int = _help("smaller masks are not fit", default=MIN_WORM_PIXELS)
    batch_size: int = _help("frames per segmenter forward pass", default=16)
    dataset_root: str = _help("flat field cache root", default=str(DEFAULT_DATASET_ROOT))
    flat_field: bool = _help("flat-field frames before segmenting", default=True)


@dataclass
class FitParams(_Params):
    preset: str = _help("fitting schedule: fast, balanced, reference", default="fast")
    fine_stride: int | None = _help("centerline point stride of the finest stage (1 or 2)", default=None)
    padding: int | None = _help("crop padding around the mask in pixels (preset default)", default=None)
    starts: str | None = _help("starting states per frame: skeleton, skeleton+straight, skeleton+reversed, all", default=None)
    compile: bool = _help("render through torch.compile", default=True)
    width_coefficients: int | None = _help("B-spline coefficients of the width correction (0 = symmetric)", default=None)
    width_prior: float | None = _help("prior weight pulling the width correction toward zero", default=None)
    orient: bool = _help("place the thinner (tail) end last when fitting without a prior", default=True)
    prior: str = _help("'bootstrap' (use the workspace prior, bootstrapping it if missing) or 'none'", default="bootstrap")
    prior_shape_weight: float = _help("weight of the width-profile prior once a recording prior is active", default=0.01)
    min_bend_radius: float | None = _help("minimum bend radius in body widths (0 disables the penalty)", default=None)
    row_pixel_budget: int = _help("rows x raster pixels per optimisation group", default=BatchFitConfig.row_pixel_budget)
    init_workers: int = _help("processes for skeleton/moment starts (0 = inline)", default=min(8, os.cpu_count() or 1))
    slab: int = _help("frames fit together", default=64)
    min_worm_pixels: int = _help("rows whose mask is smaller are not fit", default=MIN_WORM_PIXELS)
    overrides: dict[str, Any] = _help("any BatchFitConfig field, applied after the flags above (lists become tuples)", default_factory=dict)


@dataclass
class AmbiguityParams(_Params):
    pass


@dataclass
class PropagateParams(_Params):
    min_score: int = _help("ambiguity score that seeds a stretch", default=2)
    pad: int = _help("frames added on each side of a seed", default=2)
    max_gap: int = _help("seeds closer than this are one stretch", default=3)
    chain_length_sigma: float = _help("log-sigma of the length prior inside chains (0 = the fit's own)", default=0.02)
    prediction_damping: float = _help("damping of the first-order pose prediction inside chains", default=0.6)
    temporal_prior_weight: float = _help("weight of the pull toward the predicted pose", default=0.01)
    temporal_prior_sigma: float = _help("sigma of that pull, in body widths", default=0.5)
    propagate_preset: str = _help("schedule of the stretch refit pass (fast = 70% of the fit's steps)", default="fast")
    jump_seeds: bool = _help("also seed stretches by length, pose and in-view jumps", default=True)
    seed_length_fraction: float = _help("length deviation from the median that seeds a stretch (0 = off)", default=0.0)
    beam: int = _help("distinct chain states kept per direction", default=3)
    anchor_diversity: bool = _help("also start chains from the anchor refit from the frame beyond", default=True)
    refit_independent: bool = _help("refit the stored independent pose under the chain schedule", default=True)
    path: bool = _help("one path per stretch instead of the lowest energy per frame", default=True)
    path_temperature: float = _help("energy scale of the path's node cost", default=0.01)
    path_distance_weight: float = _help("weight of the squared pose distance between consecutive frames", default=1.0)
    path_inview_weight: float = _help("weight of the change of the in-view fraction", default=2.0)
    path_length_weight: float = _help("weight of the squared log length change", default=1.0)


@dataclass
class TrackParams(_Params):
    track_window: int = _help("half-window in frames of the track length median", default=50)
    track_sigma: float = _help("log-sigma of the track length prior in the refit", default=0.02)
    track_tolerance: float = _help("log deviation from the track length that triggers a refit", default=0.02)
    track_refit: str = _help("frames to refit: clipped-deviating, clipped, deviating, all", default="clipped-deviating")


@dataclass
class ExportParams(_Params):
    name: str | None = _help("file stem under exports/ (default: the UTC time)", default=None)


STAGE_PARAMS: dict[str, type[_Params]] = {
    "segment": SegmentParams,
    "prior": PriorParams,
    "fit": FitParams,
    "ambiguity": AmbiguityParams,
    "propagate": PropagateParams,
    "track": TrackParams,
    "export": ExportParams,
}


def _type_name(annotation: Any) -> str:
    text = str(annotation).replace("typing.", "")
    text = text.split(" | None")[0] if text.endswith("| None") else text
    return "dict" if text.startswith("dict") else text


def stage_schema(stage: str) -> list[dict[str, Any]]:
    """Parameter descriptions (name, type, default, help) of one stage, for forms and the API."""

    cls = STAGE_PARAMS[stage]
    defaults = cls()
    return [
        {"name": f.name, "type": _type_name(f.type), "default": getattr(defaults, f.name), "help": f.metadata.get("help", "")}
        for f in dataclass_fields(cls)
    ]


# ---------------------------------------------------------------------------
# Frames and segmentation


class Frames:
    """A recording's frames with its flat field: slab reads, corrected on request."""

    def __init__(
        self, recording: Path, *, dataset_root: Path | str | None = None, flat_field: bool = True, dataset: str = DATASET_PATH
    ) -> None:
        self.path = Path(recording)
        self.dataset_path = dataset
        self._handle: h5py.File | None = None
        self._source = (
            RecordingSource(self.path, Path(dataset_root or DEFAULT_DATASET_ROOT) / "flat_fields", dataset=dataset) if flat_field else None
        )
        self._field: Any = None
        self.field_seconds = 0.0
        with h5py.File(self.path, "r") as handle:
            if dataset not in handle:
                raise KeyError(f"{self.path}: no dataset {dataset}")
            data = handle[dataset]
            self.total = int(data.shape[0])
            self.shape = (int(data.shape[1]), int(data.shape[2]))

    @property
    def dataset(self) -> h5py.Dataset:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle[self.dataset_path]

    def field(self) -> Any:
        """The flat field (fitted and cached on first use); ``None`` when disabled."""

        if self._source is not None and self._field is None:
            t = time.perf_counter()
            self._field = self._source.flat_field()
            self.field_seconds = time.perf_counter() - t
        return self._field

    def read(self, indices: Sequence[int]) -> NDArray[np.uint8]:
        """Raw frames as a [N, H, W] uint8 stack; one slab read when the indices are consecutive."""

        indices = [int(i) for i in indices]
        if indices and indices[-1] - indices[0] + 1 == len(indices):
            return np.asarray(self.dataset[indices[0] : indices[-1] + 1], dtype=np.uint8)
        return np.stack([np.asarray(self.dataset[i], dtype=np.uint8) for i in indices])

    def corrected(self, indices: Sequence[int]) -> tuple[NDArray[np.uint8], float, float]:
        """Flat-fielded frames plus the read and correction seconds."""

        t0 = time.perf_counter()
        raw = self.read(indices)
        t1 = time.perf_counter()
        field = self.field()
        corrected = np.stack([flat_fielded(frame, field) for frame in raw])
        return corrected, t1 - t0, time.perf_counter() - t1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._source is not None:
            self._source.close()


class SegmentationModel(Protocol):
    device: torch.device

    def predict_probability_batch(self, frames: NDArray[np.generic], batch_size: int = 16) -> NDArray[np.float32]: ...


class DarkPixelSegmenter:
    """Stand-in for the network: probability ``1 - value / 255``, so pixels darker than 128 are worm at threshold 0.5.

    Used when a stage runs with ``checkpoint=None`` (tests on synthetic
    recordings, or a dark body on a bright background without a model).
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device if device is not None else "cpu")

    def predict_probability_batch(self, frames: NDArray[np.generic], batch_size: int = 16) -> NDArray[np.float32]:
        return (1.0 - np.asarray(frames, dtype=np.float32) / 255.0).astype(np.float32)


def load_segmentation_model(checkpoint: str | Path | None, device: torch.device | str | None = None) -> SegmentationModel:
    """The promoted segmenter, or the dark-pixel stand-in when no checkpoint is given."""

    if checkpoint is None or str(checkpoint) in ("", "none", "None"):
        return DarkPixelSegmenter(device)
    from .segmenter import load_segmenter

    return load_segmenter(Path(checkpoint), device)


def cleanup_kwargs(params: SegmentParams | PriorParams) -> dict[str, bool]:
    return {"fill_holes": bool(params.fill_holes), "largest_only": bool(params.largest_only)}


def segment_frames(
    frames: Frames,
    model: SegmentationModel,
    indices: Sequence[int],
    params: SegmentParams | PriorParams,
    device: torch.device,
) -> tuple[list[MaskArray], list[dict[str, int]], dict[str, float]]:
    """Read, flat-field, segment and clean these frames; returns masks, statistics and stage seconds.

    The statistics are ``clean_mask``'s plus ``mask_on_border``; an empty
    frame still gets a (false) mask so callers can index by position.
    """

    corrected, read_seconds, field_seconds = frames.corrected(indices)
    t2 = time.perf_counter()
    probability = model.predict_probability_batch(corrected, batch_size=params.batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    masks: list[MaskArray] = []
    stats: list[dict[str, int]] = []
    for prob in probability:
        mask, frame_stats = clean_mask(prob, params.threshold, params.hole_radius, device, **cleanup_kwargs(params))
        frame_stats["mask_on_border"] = int(touches_border(mask, 2)) if frame_stats["worm_pixels"] else 0
        masks.append(np.asarray(mask, dtype=bool))
        stats.append(frame_stats)
    timing = {"read": read_seconds, "flat_field": field_seconds, "network": t3 - t2, "cleanup": time.perf_counter() - t3}
    return masks, stats, timing


MASK_STAT_KEYS = ("worm_pixels", "raw_worm_pixels", "pixels_filled", "components", "pixels_outside_largest")


# ---------------------------------------------------------------------------
# The fit: configuration, starts, per-frame storage


def build_fit_config(params: FitParams) -> BatchFitConfig:
    """The preset with the command-line overrides applied."""

    config = PRESETS[params.preset]
    overrides: dict[str, Any] = {"compile_renderer": bool(params.compile), "row_pixel_budget": int(params.row_pixel_budget)}
    if params.padding is not None:
        overrides["crop_padding"] = int(params.padding)
    if params.fine_stride is not None:
        overrides["stage_point_stride"] = config.stage_point_stride[:-1] + (int(params.fine_stride),)
    if params.width_coefficients is not None:
        overrides["width_coefficients"] = int(params.width_coefficients)
    if params.width_prior is not None:
        overrides["width_shape_prior"] = float(params.width_prior)
    if params.min_bend_radius is not None:
        overrides["min_bend_radius_widths"] = float(params.min_bend_radius)
        if params.min_bend_radius <= 0:
            overrides["bend_weight"] = 0.0
    known = {f.name for f in dataclass_fields(BatchFitConfig)}
    for key, value in (params.overrides or {}).items():
        if key not in known:
            raise ValueError(f"unknown BatchFitConfig field {key!r} in fit overrides")
        overrides[key] = tuple(value) if isinstance(value, list) else value
    return replace(config, **overrides)


def config_from_dict(values: dict[str, Any]) -> BatchFitConfig:
    """A ``BatchFitConfig`` from its ``asdict`` form (lists back to tuples, unknown keys ignored)."""

    known = {f.name for f in dataclass_fields(BatchFitConfig)}
    return BatchFitConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items() if k in known})


@dataclass(frozen=True)
class FitSetup:
    """Everything the independent fit of a frame needs besides its mask."""

    config: BatchFitConfig
    template: np.ndarray
    prior: RecordingPrior | None
    start_set: str
    start_shape: np.ndarray | None
    start_length: float | None
    orient_after_fit: bool

    @property
    def start_names(self) -> tuple[str, ...] | None:
        return START_SETS[self.start_set]


def fit_setup(params: FitParams, prior: RecordingPrior | None) -> FitSetup:
    """Apply the prior to the configuration and choose the start set as the script does."""

    config = build_fit_config(params)
    if prior is not None:
        config = prior.apply(config, shape_weight=params.prior_shape_weight)
    if params.starts is not None:
        start_set = params.starts
    elif prior is not None:
        start_set = "skeleton+reversed"
    else:
        start_set = "all" if params.preset == "reference" else "skeleton+straight"
    return FitSetup(
        config=config,
        template=default_width_template(config.n_points),
        prior=prior,
        start_set=start_set,
        start_shape=np.asarray(prior.width_shape, dtype=np.float64) if prior is not None else None,
        start_length=prior.length_px if prior is not None else None,
        orient_after_fit=prior is None and bool(params.orient),
    )


def initializations_for(
    mask: np.ndarray,
    config: BatchFitConfig,
    names: tuple[str, ...] | None = None,
    width_shape: np.ndarray | None = None,
    target_length_px: float | None = None,
) -> list[Initialization]:
    """Standard starts, restricted to ``names`` when given (falling back to whatever exists).

    ``skeleton_longest_path_reversed`` adds the skeleton start traversed from
    the other end; ``width_shape`` (the prior's profile) is given to every
    start; with ``target_length_px`` a start whose end touches the image
    border is lengthened off camera to that length before fitting.
    """

    starts = standard_initializations(mask, config=config)
    if names is None:
        chosen = starts
    else:
        chosen = [s for s in starts if s.name in names] or starts[:1]
        if target_length_px is not None:
            chosen = [extend_start_to_length(s, mask, target_length_px, config=config) for s in chosen]
        if "skeleton_longest_path_reversed" in names:
            skeleton = next((s for s in chosen if s.name == "skeleton_longest_path"), None)
            if skeleton is not None:
                chosen = chosen + [orientation_pair(skeleton, config=config)[1]]
    if width_shape is not None:
        chosen = [replace(s, width_shape=np.asarray(width_shape, dtype=np.float64)) for s in chosen]
    return chosen


def orientation_gap(result: MaskFitResult) -> float:
    """Energy of the best start of the other orientation minus the winner's; NaN without both orientations."""

    reversed_names = {str(r["name"]) for r in result.records if str(r["name"]).endswith("_reversed")}
    forward = [float(r["final_energy"]) for r in result.records if str(r["name"]) not in reversed_names]
    reverse = [float(r["final_energy"]) for r in result.records if str(r["name"]) in reversed_names]
    if not forward or not reverse:
        return float("nan")
    winner_reversed = str(result.initializations[result.best_index].name) in reversed_names
    best = float(result.records[result.best_index]["final_energy"])
    return (min(forward) if winner_reversed else min(reverse)) - best


def _nan(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float64)


def new_arrays(frame_index: np.ndarray, config: BatchFitConfig) -> dict[str, np.ndarray]:
    """Empty per-frame arrays in the ``poses.npz`` layout for these frames."""

    n = len(frame_index)
    return {
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "fitted": np.zeros(n, dtype=bool),
        "latent": _nan((n, config.coefficients + 4)),
        "width_px": _nan((n,)),
        "centerline_xy": _nan((n, config.n_points, 2)),
        "width_profile": _nan((n, config.n_points)),
        "width_shape": _nan((n, config.width_coefficients)),
        "taper_asymmetry": _nan((n,)),
        "reversed": np.zeros(n, dtype=bool),
        "orientation_gap": _nan((n,)),
        "iou": _nan((n,)),
        "energy": _nan((n,)),
        "total_energy": _nan((n,)),
        "source": np.zeros(n, dtype=np.int8),
        "mask_on_border": np.zeros(n, dtype=bool),
        "points_in_fov": np.zeros(n, dtype=np.int64),
        "body_length_px": _nan((n,)),
        "crop": np.zeros((n, 4), dtype=np.int64),
        "worm_pixels": np.zeros(n, dtype=np.int64),
        "raw_worm_pixels": np.zeros(n, dtype=np.int64),
        "pixels_filled": np.zeros(n, dtype=np.int64),
        "components": np.zeros(n, dtype=np.int64),
        "pixels_outside_largest": np.zeros(n, dtype=np.int64),
        "n_starts": np.zeros(n, dtype=np.int64),
        "tube_coverage": _nan((n,)),
        "max_bend_widths": _nan((n,)),
        "best_start": np.full(n, "", dtype=BEST_START_DTYPE),
        "width_template": default_width_template(config.n_points),
    }


def store_result(arrays: dict[str, np.ndarray], row: int, result: MaskFitResult) -> None:
    """Write one fit into the per-frame arrays."""

    best = result.records[result.best_index]
    arrays["fitted"][row] = True
    arrays["width_shape"][row] = result.width_shape
    arrays["taper_asymmetry"][row] = taper_asymmetry(result.width_profile)
    arrays["latent"][row] = result.latent
    arrays["width_px"][row] = result.width_px
    arrays["centerline_xy"][row] = result.centerline_xy
    arrays["width_profile"][row] = result.width_profile
    arrays["iou"][row] = best["final_iou"]
    arrays["energy"][row] = best["final_soft_dice_energy"]
    arrays["total_energy"][row] = best["final_energy"]
    arrays["points_in_fov"][row] = result.points_in_fov
    arrays["body_length_px"][row] = result.body_length_px
    arrays["crop"][row] = (result.crop.x0, result.crop.x1, result.crop.y0, result.crop.y1)
    if "tube_coverage" in arrays:
        arrays["tube_coverage"][row] = float(best.get("final_coverage", float("nan")))
    if "max_bend_widths" in arrays:
        arrays["max_bend_widths"][row] = max_bend_widths(result.centerline_xy, result.width_px)
    if "best_start" in arrays:
        arrays["best_start"][row] = str(result.initializations[result.best_index].name)


def store_mask_stats(arrays: dict[str, np.ndarray], row: int, stats: dict[str, int]) -> None:
    for key in MASK_STAT_KEYS:
        arrays[key][row] = stats[key]
    arrays["mask_on_border"][row] = bool(stats.get("mask_on_border", 0))


def fit_frames(
    masks: Sequence[MaskArray],
    rows: Sequence[int],
    setup: FitSetup,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    *,
    pool: ProcessPoolExecutor | None = None,
    skipped: dict[str, int] | None = None,
) -> dict[str, float]:
    """Independent multi-start fit of these masks into ``arrays[rows]``; returns init and fit seconds.

    Frames without a start are skipped (``skipped['no_starts']``); when the
    batch fails, the frames are fit one at a time and the ones that still
    fail are counted under ``fit_error``.
    """

    skipped = skipped if skipped is not None else defaultdict(int)
    config, names = setup.config, setup.start_names
    t4 = time.perf_counter()
    if pool is not None:
        starts = list(
            pool.map(
                initializations_for, masks, [config] * len(masks), [names] * len(masks),
                [setup.start_shape] * len(masks), [setup.start_length] * len(masks),
            )
        )
    else:
        starts = [initializations_for(m, config, names, setup.start_shape, setup.start_length) for m in masks]
    keep = [k for k, s in enumerate(starts) if s]
    skipped["no_starts"] += len(starts) - len(keep)
    masks = [masks[k] for k in keep]
    rows = [rows[k] for k in keep]
    starts = [starts[k] for k in keep]
    t5 = time.perf_counter()
    results: list[MaskFitResult | None] = []
    if masks:
        try:
            results = list(fit_masks(masks, starts, width_template=setup.template, config=config, device=device))
        except (ValueError, RuntimeError) as error:
            print(f"batch fit failed ({error}); fitting frames one at a time", flush=True)
            for mask, frame_starts in zip(masks, starts, strict=True):
                try:
                    results.append(fit_masks([mask], [frame_starts], width_template=setup.template, config=config, device=device)[0])
                except (ValueError, RuntimeError):
                    results.append(None)
                    skipped["fit_error"] += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
    t6 = time.perf_counter()
    for row, frame_starts, result in zip(rows, starts, results, strict=True):
        arrays["n_starts"][row] = len(frame_starts)
        if result is None:
            continue
        if setup.orient_after_fit:
            result, flipped = orient_tail_last(result, config=config)
            arrays["reversed"][row] = flipped
        else:
            arrays["orientation_gap"][row] = orientation_gap(result)
        store_result(arrays, row, result)
    return {"init": t5 - t4, "fit": t6 - t5}


def independent_copies(arrays: dict[str, np.ndarray], rows: np.ndarray | None = None) -> None:
    """Keep the independent pose of these rows (all when ``None``) so a viewer can show what propagation replaced.

    The copies hold everything ``store_result`` writes, so a later propagation
    pass can start from the independent fit again (``restore_independent_rows``).
    """

    for name, source in INDEPENDENT_COPIES:
        if source not in arrays:
            continue
        if rows is None or name not in arrays or arrays[name].shape != arrays[source].shape:
            arrays[name] = arrays[source].copy()
        else:
            arrays[name][rows] = arrays[source][rows]


def restore_independent_rows(arrays: dict[str, np.ndarray], algorithm: np.ndarray) -> list[int]:
    """Put the independent fit back on the rows a propagation pass replaced (by provenance); returns those rows.

    Nothing is restored when the workspace predates the full copies
    (``latent_independent`` missing): such a pass is cumulative.
    """

    replaced = np.isin(np.asarray(algorithm).astype(str), PROPAGATION_ALGORITHMS) & np.asarray(arrays["fitted"], dtype=bool)
    rows = np.nonzero(replaced)[0]
    if not len(rows) or "latent_independent" not in arrays:
        return []
    for name, source in INDEPENDENT_COPIES:
        if name in arrays and source in arrays and arrays[name].shape == arrays[source].shape:
            arrays[source][rows] = arrays[name][rows]
    arrays["source"][rows] = 0
    arrays["reversed"][rows] = False
    arrays["orientation_gap"][rows] = np.nan
    return [int(r) for r in rows]


# ---------------------------------------------------------------------------
# Recording prior


def bootstrap_prior(
    frames: Frames,
    model: SegmentationModel,
    params: PriorParams,
    config: BatchFitConfig,
    device: torch.device,
    *,
    progress: Progress | None = None,
) -> tuple[RecordingPrior, dict[str, Any]]:
    """Segment frames spread over the whole recording and estimate its body-size prior.

    Whole worms (mask clear of the border) may be rare, so the sample is
    enlarged up to four times until ``bootstrap_target`` of them are fit.
    """

    total = frames.total
    boot_config = replace(
        PRESETS[params.bootstrap_preset],
        compile_renderer=config.compile_renderer,
        row_pixel_budget=config.row_pixel_budget,
        width_coefficients=config.width_coefficients,
        width_shape_prior=config.width_shape_prior,
    )
    seen: set[int] = set()
    masks: list[MaskArray] = []
    prior = results = used = None
    for factor in (1, 2, 4):
        count = min(params.bootstrap_frames * factor, total)
        indices = [i for i in sorted(set(int(v) for v in np.linspace(0, total - 1, count))) if i not in seen]
        seen.update(indices)
        for chunk_start in range(0, len(indices), params.batch_size):
            chunk = indices[chunk_start : chunk_start + params.batch_size]
            chunk_masks, stats, _ = segment_frames(frames, model, chunk, params, device)
            masks.extend(m for m, s in zip(chunk_masks, stats, strict=True) if s["worm_pixels"] >= params.min_worm_pixels)
            if progress is not None:
                progress(0.5 * (chunk_start + len(chunk)) / max(len(indices), 1), f"bootstrap: segmenting {len(seen)} frames")
        try:
            prior, results, used = bootstrap_prior_from_masks(masks, config=boot_config, device=device, recording=str(frames.path))
        except ValueError as error:
            if factor == 4:
                raise
            print(f"bootstrap: {error}; enlarging the sample", flush=True)
            continue
        if prior.frames_used >= params.bootstrap_target or count >= total:
            break
        print(f"bootstrap: {prior.frames_used} whole worms among {len(used)} fits; enlarging the sample", flush=True)
    assert prior is not None and results is not None and used is not None
    prior = replace(prior, source=f"bootstrap of {len(used)} frames spread over {total}, preset {params.bootstrap_preset}")
    lengths = [r.body_length_px for r in results]
    info = {
        "frames_sampled": len(seen),
        "frames_with_worm": len(masks),
        "frames_fit": len(used),
        "frames_used": prior.frames_used,
        "selection": prior.selection,
        "fit_length_px_p10_p50_p90": [float(v) for v in np.percentile(lengths, [10, 50, 90])],
    }
    return prior, info


def prior_cache_path(params: PriorParams, recording: Path, width_coefficients: int) -> Path | None:
    if not params.use_cache:
        return None
    return Path(params.prior_cache) / f"{Path(recording).stem}_k{width_coefficients}.json"


def resolve_prior(
    frames: Frames,
    params: PriorParams,
    config: BatchFitConfig,
    device: torch.device,
    *,
    model: SegmentationModel | None = None,
    progress: Progress | None = None,
) -> tuple[RecordingPrior | None, str | None, dict[str, Any] | None]:
    """The prior from the given file, the cache, or a bootstrap; returns (prior, source, bootstrap info).

    ``model`` is loaded from ``params.checkpoint`` only when a bootstrap is
    actually needed.  A bootstrapped prior is written to the cache.
    """

    if params.prior_file is not None:
        return RecordingPrior.load(Path(params.prior_file)), str(params.prior_file), None
    if params.prior != "bootstrap":
        return None, None, None
    cache_path = prior_cache_path(params, frames.path, config.width_coefficients)
    if cache_path is not None and cache_path.exists() and not params.rebootstrap:
        return RecordingPrior.load(cache_path), f"cache {cache_path}", None
    t = time.perf_counter()
    if model is None:
        model = load_segmentation_model(params.checkpoint, device)
    prior, info = bootstrap_prior(frames, model, params, config, device, progress=progress)
    info["seconds"] = time.perf_counter() - t
    if cache_path is not None:
        prior.save(cache_path)
    print(
        f"bootstrap: length {prior.length_px:.0f} px (log sigma {prior.log_length_sigma:.3f}), width {prior.width_px:.1f} px,"
        f" {prior.frames_used} of {info['frames_fit']} fits used, {info['seconds']:.0f} s",
        flush=True,
    )
    return prior, "bootstrap", info


# ---------------------------------------------------------------------------
# Propagation and the track pass

MasksOf = Callable[[Sequence[int]], dict[int, MaskArray]]


def propagation_config(params: PropagateParams) -> PropagationConfig:
    return PropagationConfig(
        min_score=params.min_score, pad=params.pad, max_gap=params.max_gap,
        chain_length_sigma=None if params.chain_length_sigma <= 0 else params.chain_length_sigma,
        prediction_damping=params.prediction_damping, temporal_prior_weight=params.temporal_prior_weight,
        temporal_prior_sigma_widths=params.temporal_prior_sigma,
        beam=params.beam, refit_independent=bool(params.refit_independent), path=bool(params.path),
        path_temperature=params.path_temperature, path_distance_weight=params.path_distance_weight,
        path_inview_weight=params.path_inview_weight, path_length_weight=params.path_length_weight,
        anchor_diversity=bool(params.anchor_diversity),
    )


def empty_hypotheses(n: int, config: BatchFitConfig, beam: int) -> dict[str, np.ndarray]:
    """Empty hypotheses arrays for ``n`` rows: independent plus ``beam`` states per direction."""

    hypotheses = 1 + 2 * max(1, beam)
    return {
        "hypotheses_centerline_xy": _nan((n, hypotheses, config.n_points, 2)),
        "hypotheses_energy": _nan((n, hypotheses)),
        "hypotheses_iou": _nan((n, hypotheses)),
        "hypotheses_source": np.full((n, hypotheses), "", dtype="<U12"),
        "hypotheses_start": np.full((n, hypotheses), "", dtype="<U32"),
        "hypotheses_beam": np.full((n, hypotheses), -1, dtype=np.int8),
        "hypotheses_count": np.zeros(n, dtype=np.int64),
        "path_index": np.full(n, -1, dtype=np.int64),
        "path_mirrored": np.zeros(n, dtype=bool),
        "path_override": np.zeros(n, dtype=bool),
        "path_energy_gap": _nan((n,)),
        "path_cost": _nan((n,)),
        "prediction_xy": _nan((n, config.n_points, 2)),
        "prediction_distance_px": _nan((n,)),
    }


@dataclass
class PropagationOutcome:
    """What the propagation pass did: the hypotheses arrays, the rows it replaced by source, and diagnostics."""

    hypotheses: dict[str, np.ndarray]
    chosen_source: dict[int, str]
    stretches: list[tuple[int, int]]
    info: dict[str, Any]


def propagation_pass(
    arrays: dict[str, np.ndarray],
    masks_of: MasksOf,
    setup: FitSetup,
    params: PropagateParams,
    device: torch.device,
    *,
    image_shape: tuple[int, int] | None,
    progress: Progress | None = None,
) -> PropagationOutcome:
    """Plan steps 5, 6a-c on ``arrays`` in place: stretches, chains with prediction and beam, one path per stretch.

    Requires the ambiguity arrays (``ambiguity_score``, ``score_independent``)
    and the independent copies.  The chosen poses replace the stored ones,
    ``source`` records where each came from, and the ambiguity is recomputed.
    """

    config, template = setup.config, setup.template
    prior_dict = None if setup.prior is None else setup.prior.to_dict()
    n = len(arrays["frame_index"])
    t_prop = time.perf_counter()
    propagation = propagation_config(params)
    # The refit pass runs the chosen preset's steps on the fit's own rasters,
    # so every candidate's energy is comparable (step 6c).
    refit_config = None if params.propagate_preset == "fast" else slow_schedule(
        config, PRESETS[params.propagate_preset], propagation.chain_length_sigma
    )
    # Stretches are seeded by the ambiguity score and (step 6b) by the track's
    # jumps: length, pose, or the body entering or leaving the camera while
    # the mask is on the border.
    seeds = jump_seeds(arrays, length_fraction=params.seed_length_fraction) if params.jump_seeds else None
    # The independent fit's score seeds the stretches, so a rerun of the pass
    # finds the same stretches whatever the previous pass changed.
    seed_score = arrays["score_independent"] if "score_independent" in arrays else arrays["ambiguity_score"]
    stretches = ambiguous_stretches(seed_score, arrays["fitted"], propagation, seeds)
    stretch_rows = [row for a, b in stretches for row in range(a, b + 1)]
    # The anchors' masks too, for the chains' second starting state (anchor diversity).
    anchor_rows = [r for a, b in stretches for r in (a - 1, b + 1) if 0 <= r < n and arrays["fitted"][r]]
    if progress is not None:
        progress(0.05, f"propagation: {len(stretches)} stretches, {len(stretch_rows)} frames")
    stretch_masks = masks_of(sorted(set(stretch_rows + anchor_rows)))
    candidates, info = propagate(
        arrays, stretches, stretch_masks, config=config, device=device, width_template=template, propagation=propagation,
        warm_config=refit_config,
    )
    if progress is not None:
        progress(0.8, "propagation: selecting paths")
    t_path = time.perf_counter()
    if propagation.path:
        chosen = select_path(candidates, arrays, stretches, config, propagation, image_shape)
    else:
        chosen = {
            row: PathChoice(candidate, False, False, 0.0, float("nan"))
            for row, candidate in select_candidates(candidates, arrays, config, propagation).items()
        }
    path_seconds = time.perf_counter() - t_path
    before_iou = float(np.nanmedian(arrays["iou"][stretch_rows])) if stretch_rows else None
    # Every candidate of every stretch frame is kept (step 6c), with the
    # path's choice, so the viewer can show what the path chose between and
    # where it overrode the lowest energy.
    hyp_arrays = empty_hypotheses(n, config, propagation.beam)
    hypotheses = hyp_arrays["hypotheses_energy"].shape[1]
    for row, options in candidates.items():
        ranked = sorted(options, key=lambda c: (SOURCE_CODES.get(c.source, 3), c.beam))[:hypotheses]
        hyp_arrays["hypotheses_count"][row] = len(ranked)
        for j, candidate in enumerate(ranked):
            hyp_arrays["hypotheses_centerline_xy"][row, j] = candidate.result.centerline_xy
            hyp_arrays["hypotheses_energy"][row, j] = candidate.total_energy
            hyp_arrays["hypotheses_iou"][row, j] = float(candidate.result.records[candidate.result.best_index]["final_iou"])
            hyp_arrays["hypotheses_source"][row, j] = candidate.source
            hyp_arrays["hypotheses_start"][row, j] = candidate.start_name
            hyp_arrays["hypotheses_beam"][row, j] = candidate.beam
            if row in chosen and chosen[row].candidate is candidate:
                hyp_arrays["path_index"][row] = j
    chosen_source: dict[int, str] = {}
    for row, choice in chosen.items():
        result = reverse_result(choice.candidate.result, config=config) if choice.mirrored else choice.candidate.result
        store_result(arrays, row, result)
        arrays["source"][row] = SOURCE_CODES[choice.candidate.source]
        arrays["orientation_gap"][row] = np.nan
        arrays["reversed"][row] = False
        chosen_source[int(row)] = choice.candidate.source
        hyp_arrays["path_mirrored"][row] = choice.mirrored
        hyp_arrays["path_override"][row] = choice.override
        hyp_arrays["path_energy_gap"][row] = choice.energy_gap
        hyp_arrays["path_cost"][row] = choice.cost
        hyp_arrays["prediction_distance_px"][row] = choice.candidate.distance_to_prediction_px
        if choice.candidate.prediction_xy is not None:
            hyp_arrays["prediction_xy"][row] = choice.candidate.prediction_xy[::-1] if choice.mirrored else choice.candidate.prediction_xy
    arrays.update(hyp_arrays)
    arrays.update(compute_ambiguity(arrays, prior=prior_dict, image_shape=image_shape))
    seconds = time.perf_counter() - t_prop
    info.update(
        {
            "jump_seeds": None if seeds is None else int(np.sum(seeds & arrays["fitted"])),
            "jump_seeds_below_score": None if seeds is None else int(np.sum(seeds & arrays["fitted"] & (arrays["score_independent"] < propagation.min_score))),
            "frames_replaced": len(chosen),
            "replaced_by_source": {name: int(sum(c.candidate.source == name for c in chosen.values())) for name in SOURCE_CODES},
            "refit_preset": params.propagate_preset,
            "path": {
                "enabled": propagation.path,
                "temperature": propagation.path_temperature,
                "distance_weight": propagation.path_distance_weight,
                "inview_weight": propagation.path_inview_weight,
                "length_weight": propagation.path_length_weight,
                "mirrors": propagation.path_mirrors,
                "frames_overriding_lowest_energy": int(sum(c.override for c in chosen.values())),
                "frames_mirrored": int(sum(c.mirrored for c in chosen.values())),
                "energy_gap_p50_p90_max": [float(v) for v in np.percentile([c.energy_gap for c in chosen.values() if c.override], [50, 90, 100])]
                if any(c.override for c in chosen.values()) else None,
                "seconds": path_seconds,
            },
            "stretch_iou_median_before": before_iou,
            "stretch_iou_median_after": float(np.nanmedian(arrays["iou"][stretch_rows])) if stretch_rows else None,
            "stretch_frames_score_at_least_2_before": int(np.sum(arrays["score_independent"][stretch_rows] >= 2)) if stretch_rows else 0,
            "stretch_frames_score_at_least_2_after": int(np.sum(arrays["ambiguity_score"][stretch_rows] >= 2)) if stretch_rows else 0,
            "seconds": seconds,
        }
    )
    after = info["stretch_iou_median_after"]
    print(
        f"propagation: {len(stretches)} stretches, {len(stretch_rows)} frames, {len(chosen)} replaced"
        f" ({info['replaced_by_source']}), stretch median IoU"
        f" {before_iou if before_iou is None else round(before_iou, 3)} -> {after if after is None else round(after, 3)},"
        f" {seconds:.0f} s",
        flush=True,
    )
    return PropagationOutcome(hyp_arrays, chosen_source, [(int(a), int(b)) for a, b in stretches], info)


def track_length_pass(
    arrays: dict[str, np.ndarray],
    masks_of: MasksOf,
    setup: FitSetup,
    params: TrackParams,
    device: torch.device,
    *,
    stretches: Sequence[tuple[int, int]] = (),
    image_shape: tuple[int, int] | None,
) -> tuple[dict[str, Any], list[int]]:
    """Plan step 6b on ``arrays`` in place: refit clipped and length-deviating frames with the track's length prior.

    A clipped body's length is not observable in one frame, so frames outside
    the stretches whose length departs from the track are refit with the
    track's length prior.  Returns the diagnostics and the rows refit.
    """

    config, template, prior = setup.config, setup.template, setup.prior
    n = len(arrays["frame_index"])
    in_stretch_rows = np.zeros(n, dtype=bool)
    for a, b in stretches:
        in_stretch_rows[a : b + 1] = True
    t_track = time.perf_counter()
    track = track_length(arrays, window=params.track_window, min_iou=0.9, fallback_px=None if prior is None else prior.length_px)
    arrays["track_length_px"] = track
    arrays["length_refit"] = np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        deviates = np.abs(np.log(arrays["body_length_px"] / track)) > params.track_tolerance
    clipped = np.asarray(arrays["mask_on_border"], dtype=bool)
    select = {
        "clipped-deviating": clipped & np.nan_to_num(deviates),
        "clipped": clipped,
        "deviating": np.nan_to_num(deviates),
        "all": clipped | np.nan_to_num(deviates),
    }[params.track_refit]
    refit_rows = [int(r) for r in np.nonzero(arrays["fitted"] & select & ~in_stretch_rows)[0] if np.isfinite(track[r])]
    track_masks = masks_of(refit_rows)
    groups: dict[int, list[int]] = defaultdict(list)
    for r in refit_rows:
        if r in track_masks:
            groups[int(round(math.log(track[r]) / 0.02))].append(r)
    warm = warm_schedule(config)
    refit: list[int] = []
    for rows in groups.values():
        group_config = replace(warm, length_prior_px=float(np.exp(np.mean(np.log(track[rows])))), length_prior_log_sigma=params.track_sigma)
        for chunk_start in range(0, len(rows), config.max_rows):
            chunk = rows[chunk_start : chunk_start + config.max_rows]
            starts = [[warm_initialization(arrays["latent"][r], float(arrays["width_px"][r]), arrays["width_shape"][r], "track_length")] for r in chunk]
            results = fit_masks([track_masks[r] for r in chunk], starts, width_template=template, config=group_config, device=device)
            for r, result in zip(chunk, results, strict=True):
                store_result(arrays, r, result)
                arrays["length_refit"][r] = True
                refit.append(int(r))
    seconds = time.perf_counter() - t_track
    info = {
        "window": params.track_window, "sigma": params.track_sigma, "tolerance": params.track_tolerance, "refit": params.track_refit,
        "frames_clipped": int(np.sum(arrays["fitted"] & arrays["mask_on_border"])),
        "frames_deviating": int(np.sum(arrays["fitted"] & np.nan_to_num(deviates))),
        "frames_refit": len(refit), "seconds": seconds,
        "track_length_p10_p50_p90": [float(v) for v in np.nanpercentile(track, [10, 50, 90])] if np.isfinite(track).any() else None,
    }
    if refit:
        prior_dict = None if prior is None else prior.to_dict()
        arrays.update(compute_ambiguity(arrays, prior=prior_dict, image_shape=image_shape))
    print(f"track length: {len(refit)} frames refit with the track prior, {seconds:.0f} s", flush=True)
    return info, refit


# ---------------------------------------------------------------------------
# Summary statistics shared by the script and the workspace


def orientation_consistency(arrays: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """How often consecutive fitted frames agree on which end is the head."""

    fitted = np.nonzero(arrays["fitted"])[0]
    pairs = [(a, b) for a, b in zip(fitted[:-1], fitted[1:], strict=False) if b == a + 1]
    if not pairs:
        return None
    curves = arrays["centerline_xy"]
    agree = 0
    for a, b in pairs:
        same = np.linalg.norm(curves[b, 0] - curves[a, 0]) + np.linalg.norm(curves[b, -1] - curves[a, -1])
        swapped = np.linalg.norm(curves[b, 0] - curves[a, -1]) + np.linalg.norm(curves[b, -1] - curves[a, 0])
        agree += int(same <= swapped)
    return {"consecutive_pairs": len(pairs), "fraction_consistent": agree / len(pairs), "flips": len(pairs) - agree}


def fit_statistics(arrays: dict[str, np.ndarray], config: BatchFitConfig, prior: RecordingPrior | None, *, orient: bool) -> dict[str, Any]:
    """The per-run aggregates of a run's summary that follow from the arrays alone."""

    fitted = np.asarray(arrays["fitted"], dtype=bool)
    iou = arrays["iou"][fitted]
    in_view = arrays["points_in_fov"][fitted] / config.n_points
    best_start = [str(b) for b in arrays.get("best_start", np.asarray([], dtype=str)) if b]
    return {
        "frames_fitted": int(fitted.sum()),
        "iou": None if not len(iou) else {
            "median": float(np.median(iou)), "p10": float(np.percentile(iou, 10)), "min": float(iou.min()),
            "fraction_at_least_0.8": float(np.mean(iou >= 0.8)), "fraction_at_least_0.9": float(np.mean(iou >= 0.9)),
        },
        "body_length_px": None if not fitted.any() else {
            "median": float(np.median(arrays["body_length_px"][fitted])),
            "p10": float(np.percentile(arrays["body_length_px"][fitted], 10)),
            "p90": float(np.percentile(arrays["body_length_px"][fitted], 90)),
            "at_upper_bound": None if config.length_bounds_px is None else int(np.sum(arrays["body_length_px"][fitted] >= 0.99 * config.length_bounds_px[1])),
            "beyond_2_sigma_of_prior": None if prior is None else int(np.sum(
                np.abs(np.log(arrays["body_length_px"][fitted] / prior.length_px)) > 2 * prior.log_length_sigma
            )),
        },
        "orientation": None if prior is None or not fitted.any() else {
            "gap_median": float(np.nanmedian(arrays["orientation_gap"][fitted])),
            "gap_p10": float(np.nanpercentile(arrays["orientation_gap"][fitted], 10)),
            "frames_gap_below_0.002": int(np.sum(arrays["orientation_gap"][fitted] < 0.002)),
            "frames_gap_below_0.01": int(np.sum(arrays["orientation_gap"][fitted] < 0.01)),
            "reversed_start_won": int(sum(b.endswith("_reversed") for b in best_start)),
        },
        "width_px": None if not fitted.any() else {"median": float(np.median(arrays["width_px"][fitted]))},
        "width_model": {
            "coefficients": config.width_coefficients,
            "prior": config.width_shape_prior,
            "tail_placed_last": bool(orient),
            "frames_reversed": int(arrays["reversed"].sum()),
            "taper_asymmetry": None if not fitted.any() else {
                "median": float(np.median(arrays["taper_asymmetry"][fitted])),
                "p10": float(np.percentile(arrays["taper_asymmetry"][fitted], 10)),
                "p90": float(np.percentile(arrays["taper_asymmetry"][fitted], 90)),
                "frames_with_abs_below_0.1": int(np.sum(np.abs(arrays["taper_asymmetry"][fitted]) < 0.1)),
            },
            "orientation_consistency": orientation_consistency(arrays),
        },
        "in_view_fraction": None if not fitted.any() else {"median": float(np.median(in_view)), "frames_below_1": int(np.sum(in_view < 1.0))},
        "continuity": continuity_summary(arrays),
        "ambiguity": summarize_ambiguity(arrays) if fitted.any() and "ambiguity_score" in arrays else None,
        "best_start_counts": {name: int(count) for name, count in zip(*np.unique(best_start, return_counts=True))} if best_start else {},
        "mask": mask_statistics(arrays),
    }


def mask_statistics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "worm_pixels_median": float(np.median(arrays["worm_pixels"])),
        "frames_with_multiple_components": int(np.sum(arrays["components"] > 1)),
        "pixels_outside_largest_median": float(np.median(arrays["pixels_outside_largest"])),
        "frames_with_filling": int(np.sum(arrays["pixels_filled"] > 0)),
    }


# ---------------------------------------------------------------------------
# Stages over a workspace


def _job_id(job: str | None, stage: str) -> str:
    return job or os.environ.get("WORM_POSE_JOB_ID") or f"stage:{stage}"


def _device(device: torch.device | str | None) -> torch.device:
    return torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))


def read_summary(workspace: Any) -> dict[str, Any]:
    """The workspace's synthesised run summary (``summary.json``), falling back to an imported run's."""

    for name in (SUMMARY_FILE, "imported_summary.json"):
        path = Path(workspace.path) / name
        if path.exists():
            return json.loads(path.read_text())
    return {}


def update_summary(workspace: Any, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into the workspace's ``summary.json`` (written atomically)."""

    path = Path(workspace.path) / SUMMARY_FILE
    # Seeded from ``read_summary`` so the first stage on an imported workspace
    # carries the imported run's fit configuration forward.
    summary = dict(read_summary(workspace))
    summary.update(updates)
    summary["finished_at"] = utc_now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=1))
    temporary.replace(path)
    return summary


@contextmanager
def workspace_lock(workspace: Any) -> Iterator[None]:
    """Hold ``<workspace>/.lock`` (an ``flock``) so only one stage writes the workspace at a time, from any process."""

    path = getattr(workspace, "path", None)
    if path is None:
        yield
        return
    with open(Path(path) / WORKSPACE_LOCK_FILE, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"waiting for another stage to finish on {path}", flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def workspace_frames(workspace: Any, params: SegmentParams | PriorParams) -> Frames:
    return Frames(
        Path(workspace.info.recording), dataset_root=params.dataset_root, flat_field=params.flat_field, dataset=workspace_dataset(workspace)
    )


def workspace_dataset(workspace: Any) -> str:
    """The HDF5 dataset holding the frames: the workspace's ``dataset`` setting, else ``/img_nir``."""

    settings = getattr(getattr(workspace, "info", None), "settings", None) or {}
    return str(settings.get("dataset") or DATASET_PATH)


def workspace_prior(workspace: Any) -> RecordingPrior | None:
    path = Path(workspace.path) / "recording_prior.json"
    return RecordingPrior.load(path) if path.exists() else None


def workspace_setup(workspace: Any, params: FitParams | None = None) -> FitSetup:
    """The fit setup of a workspace: its stored fit configuration when a fit ran, else built from ``params``."""

    params = params or FitParams()
    prior = workspace_prior(workspace)
    summary = read_summary(workspace)
    if summary.get("fit_config"):
        config = config_from_dict(summary["fit_config"])
        start_set = str(summary.get("starts") or fit_setup(params, prior).start_set)
        orient = bool((summary.get("width_model") or {}).get("tail_placed_last", params.orient))
        return FitSetup(
            config=config, template=default_width_template(config.n_points), prior=prior, start_set=start_set,
            start_shape=np.asarray(prior.width_shape, dtype=np.float64) if prior is not None else None,
            start_length=prior.length_px if prior is not None else None, orient_after_fit=prior is None and orient,
        )
    return fit_setup(params, prior)


def workspace_arrays(workspace: Any, config: BatchFitConfig) -> dict[str, np.ndarray]:
    """The state arrays with every base array present (missing ones created empty)."""

    arrays = new_arrays(workspace.frame_index, config)
    stored = workspace.load_state()
    for key, value in stored.items():
        if key == "best_start":
            value = np.asarray(value, dtype=BEST_START_DTYPE)
        arrays[key] = value
    return arrays


def workspace_image_shape(workspace: Any) -> tuple[int, int] | None:
    shape = getattr(workspace, "image_shape", None)
    if shape is not None:
        return (int(shape[0]), int(shape[1]))
    try:
        with h5py.File(workspace.info.recording, "r") as handle:
            dataset = handle[workspace_dataset(workspace)]
            return (int(dataset.shape[1]), int(dataset.shape[2]))
    except (OSError, KeyError):
        return None


def workspace_masks_of(workspace: Any, min_worm_pixels: int = 0) -> MasksOf:
    """Masks of workspace rows (overrides first), skipping rows whose mask is missing or too small."""

    def masks_of(rows: Sequence[int]) -> dict[int, MaskArray]:
        out: dict[int, MaskArray] = {}
        for row in rows:
            mask = workspace.effective_mask(int(row))
            if mask is not None and int(np.count_nonzero(mask)) >= min_worm_pixels:
                out[int(row)] = np.asarray(mask, dtype=bool)
        return out

    return masks_of


def _slabs(rows: Sequence[int], size: int) -> list[list[int]]:
    rows = [int(r) for r in rows]
    return [rows[i : i + max(1, size)] for i in range(0, len(rows), max(1, size))]


class _PendingMasks:
    """Packed masks waiting to be written; ``add`` flushes when the chunk changes, ``flush`` at the end."""

    def __init__(self, workspace: Any) -> None:
        self.workspace = workspace
        self.rows: list[int] = []
        self.packed: list[np.ndarray] = []
        self.shape: tuple[int, int] | None = None

    def add(self, row: int, mask: np.ndarray) -> None:
        if self.rows and row // MASK_CHUNK_ROWS != self.rows[0] // MASK_CHUNK_ROWS:
            self.flush()
        self.rows.append(int(row))
        self.packed.append(pack_mask(mask))
        self.shape = (int(mask.shape[0]), int(mask.shape[1]))

    def flush(self) -> None:
        if self.rows and self.shape is not None:
            self.workspace.set_packed_masks(self.rows, self.packed, self.shape)
        self.rows, self.packed = [], []


def run_segment(workspace: Any, params: SegmentParams, *, device: torch.device, progress: Progress | None, job: str) -> dict[str, Any]:
    frames = workspace_frames(workspace, params)
    try:
        model = load_segmentation_model(params.checkpoint, device)
        arrays = workspace.load_state()
        n = workspace.n
        base = new_arrays(workspace.frame_index, BatchFitConfig())
        for key in MASK_STAT_KEYS + ("mask_on_border",):
            arrays.setdefault(key, base[key])
        arrays.setdefault("frame_index", base["frame_index"])
        timing = {stage: 0.0 for stage in TIMING_STAGES}
        segmented = 0
        # Masks are written one chunk file at a time (packed while they wait):
        # rewriting a chunk per slab would re-merge and re-compress it for every slab.
        pending = _PendingMasks(workspace)
        for slab in _slabs(range(n), params.slab):
            masks, stats, slab_timing = segment_frames(frames, model, workspace.frame_index[slab], params, device)
            for row, mask, frame_stats in zip(slab, masks, stats, strict=True):
                store_mask_stats(arrays, row, frame_stats)
                pending.add(row, mask)
            for stage, seconds in slab_timing.items():
                timing[stage] += seconds
            segmented += len(slab)
            if progress is not None:
                progress(segmented / n, f"segmented {segmented}/{n} frames")
        pending.flush()
        workspace.save_state(arrays)
        # Provenance describes the pose of a frame, so segmentation leaves it alone.
        worm = np.asarray(arrays["worm_pixels"])
        result = {
            "frames": n,
            "frames_with_worm": int(np.sum(worm >= params.min_worm_pixels)),
            "frames_empty": int(np.sum(worm == 0)),
            "seconds": timing,
            "flat_field_seconds": frames.field_seconds,
        }
        update_summary(
            workspace,
            {
                "recording": str(frames.path),
                "frames": [int(workspace.frame_index[0]), int(workspace.frame_index[-1])] if n else None,
                "step": int(workspace.info.step),
                "frame_count": n,
                "checkpoint": checkpoint_fingerprint(params.checkpoint),
                "threshold": params.threshold,
                "mask_cleanup": {
                    "fill_holes": params.fill_holes, "fill_holes_radius_px": params.hole_radius,
                    "largest_component": params.largest_only, "min_worm_pixels": params.min_worm_pixels,
                },
                "segment": result,
                "mask": mask_statistics(arrays),
            },
        )
        return result
    finally:
        frames.close()


def run_prior(workspace: Any, params: PriorParams, *, device: torch.device, progress: Progress | None, job: str, fit_params: FitParams | None = None) -> dict[str, Any]:
    frames = workspace_frames(workspace, params)
    try:
        config = build_fit_config(fit_params or FitParams())
        prior, source, info = resolve_prior(frames, params, config, device, progress=progress)
        path = Path(workspace.path) / "recording_prior.json"
        if prior is None:
            if path.exists():
                path.unlink()
        else:
            prior.save(path)
        result = {"prior": None if prior is None else prior.to_dict(), "prior_source": source, "bootstrap": info}
        update_summary(workspace, result)
        return result
    finally:
        frames.close()


def run_fit(workspace: Any, params: FitParams, *, device: torch.device, progress: Progress | None, job: str, all_params: dict[str, Any] | None = None) -> dict[str, Any]:
    if params.prior == "bootstrap" and workspace_prior(workspace) is None:
        run_prior(workspace, PriorParams.from_dict(all_params), device=device, progress=progress, job=job, fit_params=params)
    prior = workspace_prior(workspace) if params.prior == "bootstrap" else None
    setup = fit_setup(params, prior)
    arrays = workspace_arrays(workspace, setup.config)
    if arrays["latent"].shape[1] != setup.config.coefficients + 4 or arrays["width_shape"].shape[1] != setup.config.width_coefficients:
        # A previous fit used another model size: its pose arrays cannot hold this one.
        fresh = new_arrays(workspace.frame_index, setup.config)
        for key in fresh:
            if key not in MASK_STAT_KEYS + ("mask_on_border", "frame_index"):
                arrays[key] = fresh[key]
    mask_rows = [int(r) for r in workspace.mask_rows()]
    override_statistics(workspace, arrays)
    rows = [r for r in mask_rows if int(arrays["worm_pixels"][r]) >= params.min_worm_pixels]
    skipped = {
        "empty_mask": int(np.sum(arrays["worm_pixels"][mask_rows] == 0)),
        "small_mask": int(np.sum((arrays["worm_pixels"][mask_rows] > 0) & (arrays["worm_pixels"][mask_rows] < params.min_worm_pixels))),
        "no_starts": 0, "fit_error": 0,
    }
    # Rows fit again start from nothing: they are independent fits.
    for key in ("fitted", "reversed", "source"):
        arrays[key][rows] = 0
    arrays["orientation_gap"][rows] = np.nan
    if "length_refit" in arrays:
        arrays["length_refit"][rows] = False
    timing = {"init": 0.0, "fit": 0.0}
    pool = ProcessPoolExecutor(max_workers=params.init_workers) if params.init_workers > 0 and rows else None
    done = 0
    try:
        masks_of = workspace_masks_of(workspace)
        for slab in _slabs(rows, params.slab):
            masks_by_row = masks_of(slab)
            slab_rows = [r for r in slab if r in masks_by_row]
            slab_timing = fit_frames([masks_by_row[r] for r in slab_rows], slab_rows, setup, arrays, device, pool=pool, skipped=skipped)
            for stage, seconds in slab_timing.items():
                timing[stage] += seconds
            done += len(slab)
            if progress is not None:
                progress(done / max(len(rows), 1), f"fit {done}/{len(rows)} frames, median iou {np.nanmedian(arrays['iou'][slab]) if slab else float('nan'):.3f}")
    finally:
        if pool is not None:
            pool.shutdown()
    independent_copies(arrays, np.asarray(rows, dtype=np.int64) if "iou_independent" in arrays else None)
    if "ambiguity_score" in arrays:
        # The old scores and candidates described poses this fit replaced.
        refresh_ambiguity(arrays, prior, workspace_image_shape(workspace), rows)
        blank_hypotheses(workspace, rows, setup.config)
    workspace.save_state(arrays)
    fitted_rows = [r for r in rows if arrays["fitted"][r]]
    workspace.set_provenance(fitted_rows, SOURCE_ALGORITHMS[0], job, time.time())
    stats = fit_statistics(arrays, setup.config, prior, orient=params.orient)
    result = {"frames_fit": len(fitted_rows), "frames_skipped": skipped, "seconds": timing, "iou": stats["iou"]}
    update_summary(
        workspace,
        {
            **stats,
            "fit_config": asdict(setup.config),
            "fit_params": params.to_dict(),
            "width_template": "default_width_template",
            "preset": params.preset,
            "starts": setup.start_set,
            "prior": None if prior is None else prior.to_dict(),
            "device": str(device),
            "frames_skipped": skipped,
            "fit": result,
        },
    )
    return result


def override_statistics(workspace: Any, arrays: dict[str, np.ndarray]) -> list[int]:
    """``worm_pixels`` of rows with an override mask counts the override, which is what their fit scores against."""

    rows = [int(r) for r in workspace.override_rows()]
    for row in rows:
        mask = workspace.get_override_mask(row)
        if mask is not None:
            arrays["worm_pixels"][row] = int(np.count_nonzero(mask))
    return rows


def refresh_ambiguity(arrays: dict[str, np.ndarray], prior: RecordingPrior | None, image_shape: tuple[int, int] | None, rows: Sequence[int]) -> None:
    """Recompute the ambiguity arrays after ``rows`` were fit independently (their independent score is the new one)."""

    arrays.update(compute_ambiguity(arrays, prior=None if prior is None else prior.to_dict(), image_shape=image_shape))
    index = np.asarray(rows, dtype=np.int64)
    if "score_independent" in arrays and len(index):
        arrays["score_independent"][index] = arrays["ambiguity_score"][index]


def blank_hypotheses(workspace: Any, rows: Sequence[int], config: BatchFitConfig) -> None:
    """Drop the stored candidates of ``rows``: they were alternatives to poses that no longer exist."""

    stored = workspace.load_hypotheses()
    if "hypotheses_energy" not in stored or not len(rows):
        return
    n, hypotheses = stored["hypotheses_energy"].shape
    empty = empty_hypotheses(n, config, max(1, (hypotheses - 1) // 2))
    index = np.asarray(rows, dtype=np.int64)
    for key, value in stored.items():
        if key in empty and empty[key].shape == value.shape:
            value[index] = empty[key][index]
    workspace.save_hypotheses(stored)


def run_ambiguity(workspace: Any, params: AmbiguityParams, *, device: torch.device, progress: Progress | None, job: str) -> dict[str, Any]:
    setup = workspace_setup(workspace)
    arrays = workspace_arrays(workspace, setup.config)
    image_shape = workspace_image_shape(workspace)
    prior_dict = None if setup.prior is None else setup.prior.to_dict()
    arrays.update(compute_ambiguity(arrays, prior=prior_dict, image_shape=image_shape))
    # The independent fit's score is what seeds the stretches; it is the
    # current score wherever the stored pose is still the independent one
    # (provenance ``independent_fit``, or unknown).
    algorithm = np.asarray(workspace.load_provenance()["algorithm"]).astype(str)
    independent = (np.asarray(arrays["source"]) == 0) & np.isin(algorithm, ("", SOURCE_ALGORITHMS[0]))
    if "score_independent" not in arrays:
        arrays["score_independent"] = arrays["ambiguity_score"].copy()
    else:
        arrays["score_independent"][independent] = arrays["ambiguity_score"][independent]
    independent_copies(arrays, np.nonzero(independent)[0] if "iou_independent" in arrays else None)
    workspace.save_state(arrays)
    summary = summarize_ambiguity(arrays) if arrays["fitted"].any() else None
    update_summary(workspace, {"ambiguity": summary})
    return {"ambiguity": summary}


def run_propagate(workspace: Any, params: PropagateParams, *, device: torch.device, progress: Progress | None, job: str) -> dict[str, Any]:
    setup = workspace_setup(workspace)
    arrays = workspace_arrays(workspace, setup.config)
    image_shape = workspace_image_shape(workspace)
    # The pass starts from the independent fit: rows a previous pass replaced
    # go back to it, and the ambiguity of what is stored is recomputed, so a
    # rerun with other parameters is a fresh pass rather than a second layer.
    restored = restore_independent_rows(arrays, workspace.load_provenance()["algorithm"])
    if restored:
        workspace.save_state(arrays)
        workspace.set_provenance(restored, SOURCE_ALGORITHMS[0], job, time.time())
    run_ambiguity(workspace, AmbiguityParams(), device=device, progress=None, job=job)
    arrays = workspace_arrays(workspace, setup.config)
    if "iou_independent" not in arrays:
        independent_copies(arrays)
    # The pass writes a fresh set of hypotheses for every stretch frame; older ones are replaced.
    outcome = propagation_pass(
        arrays, workspace_masks_of(workspace), setup, params, device, image_shape=image_shape, progress=progress
    )
    state = {k: v for k, v in arrays.items() if k not in outcome.hypotheses}
    workspace.save_state(state)
    workspace.save_hypotheses(outcome.hypotheses)
    by_algorithm: dict[str, list[int]] = defaultdict(list)
    for row, source in outcome.chosen_source.items():
        by_algorithm[CANDIDATE_ALGORITHMS[source]].append(row)
    now = time.time()
    for algorithm, rows in by_algorithm.items():
        workspace.set_provenance(rows, algorithm, job, now)
    summary_updates = {
        "propagation": {**outcome.info, "restored_rows": len(restored)},
        "propagate_params": params.to_dict(),
        "ambiguity": summarize_ambiguity(arrays) if arrays["fitted"].any() else None,
        "continuity": continuity_summary(arrays),
    }
    update_summary(workspace, summary_updates)
    return {
        "stretches": outcome.stretches, "frames_replaced": len(outcome.chosen_source),
        "replaced_by_source": outcome.info.get("replaced_by_source"), "restored_rows": len(restored),
    }


def run_track(workspace: Any, params: TrackParams, *, device: torch.device, progress: Progress | None, job: str) -> dict[str, Any]:
    setup = workspace_setup(workspace)
    image_shape = workspace_image_shape(workspace)
    # The scores must describe the stored poses, whatever ran since the last ambiguity stage.
    run_ambiguity(workspace, AmbiguityParams(), device=device, progress=None, job=job)
    arrays = workspace_arrays(workspace, setup.config)
    stretches = [(int(a), int(b)) for a, b in ((read_summary(workspace).get("propagation") or {}).get("stretches") or [])]
    info, refit = track_length_pass(
        arrays, workspace_masks_of(workspace), setup, params, device, stretches=stretches, image_shape=image_shape
    )
    workspace.save_state(arrays)
    workspace.set_provenance(refit, TRACK_ALGORITHM, job, time.time())
    update_summary(
        workspace,
        {
            "track_length": info,
            "track_params": params.to_dict(),
            "ambiguity": summarize_ambiguity(arrays) if arrays["fitted"].any() else None,
            "continuity": continuity_summary(arrays),
        },
    )
    return info


def _curvature(curve: np.ndarray) -> float:
    """Mean absolute turning angle per unit length along a centerline (1/px)."""

    step = np.diff(curve, axis=0)
    segment = np.linalg.norm(step, axis=1)
    if len(segment) < 2 or not np.isfinite(segment).all():
        return float("nan")
    angle = np.arctan2(step[:, 1], step[:, 0])
    turn = np.abs(np.angle(np.exp(1j * np.diff(angle))))
    return float(np.mean(turn / np.maximum(0.5 * (segment[1:] + segment[:-1]), 1e-6)))


def export_table(arrays: dict[str, np.ndarray], provenance: dict[str, np.ndarray] | None = None) -> "pyarrow.Table":
    """One row per frame: pose, statistics, flags, provenance and kinematics, as a pyarrow table."""

    import pyarrow as pa

    n = len(arrays["frame_index"])
    fitted = np.asarray(arrays["fitted"], dtype=bool)
    curves = np.asarray(arrays["centerline_xy"], dtype=np.float64)
    nan = np.full(n, np.nan)

    def column(name: str, default: np.ndarray) -> np.ndarray:
        return np.asarray(arrays[name]) if name in arrays else default

    def floats(values: np.ndarray) -> "pa.Array":
        # NaN means "no value" in the arrays; Parquet readers expect null for that.
        values = np.asarray(values, dtype=np.float64)
        return pa.array(values, mask=np.isnan(values))

    centroid = np.nanmean(curves, axis=1) if n else np.zeros((0, 2))
    centroid[~fitted] = np.nan
    speed = nan.copy()
    previous: int | None = None
    for row in range(n):
        if not fitted[row]:
            continue
        if previous is not None:
            gap = max(int(arrays["frame_index"][row]) - int(arrays["frame_index"][previous]), 1)
            speed[row] = float(np.linalg.norm(centroid[row] - centroid[previous])) / gap
        previous = row
    curvature = np.asarray([_curvature(curves[row]) if fitted[row] else float("nan") for row in range(n)])
    columns: dict[str, Any] = {
        "frame_index": pa.array(np.asarray(arrays["frame_index"], dtype=np.int64)),
        "fitted": pa.array(fitted),
        "iou": floats(arrays["iou"]),
        "tube_coverage": floats(column("tube_coverage", nan)),
        "body_length_px": floats(arrays["body_length_px"]),
        "width_px": floats(arrays["width_px"]),
        "points_in_fov": pa.array(np.asarray(arrays["points_in_fov"], dtype=np.int64)),
        "source": pa.array(column("source", np.zeros(n, dtype=np.int8)).astype(np.int8)),
        "ambiguity_score": pa.array(column("ambiguity_score", np.zeros(n, dtype=np.int64)).astype(np.int64)),
    }
    for name in FLAG_NAMES:
        columns[f"flag_{name}"] = pa.array(column(f"flag_{name}", np.zeros(n, dtype=bool)).astype(bool))
    provenance = provenance or {}
    columns["provenance_algorithm"] = pa.array([str(v) for v in provenance.get("algorithm", np.full(n, ""))], pa.string())
    columns["provenance_job"] = pa.array([str(v) for v in provenance.get("job", np.full(n, ""))], pa.string())
    columns["provenance_time"] = floats(provenance.get("time", nan))
    columns["centerline_x"] = pa.array([row[:, 0].tolist() if fitted[i] else None for i, row in enumerate(curves)], pa.list_(pa.float64()))
    columns["centerline_y"] = pa.array([row[:, 1].tolist() if fitted[i] else None for i, row in enumerate(curves)], pa.list_(pa.float64()))
    profile = np.asarray(arrays["width_profile"], dtype=np.float64)
    columns["width_profile"] = pa.array([profile[i].tolist() if fitted[i] else None for i in range(n)], pa.list_(pa.float64()))
    columns["centroid_x"] = floats(centroid[:, 0] if n else np.zeros(0))
    columns["centroid_y"] = floats(centroid[:, 1] if n else np.zeros(0))
    columns["speed_px_per_frame"] = floats(speed)
    columns["mean_abs_curvature"] = floats(curvature)
    return pa.table(columns)


def run_export(workspace: Any, params: ExportParams, *, device: torch.device, progress: Progress | None, job: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    setup = workspace_setup(workspace)
    arrays = workspace_arrays(workspace, setup.config)
    table = export_table(arrays, workspace.load_provenance())
    exports = Path(workspace.path) / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    path = exports / f"{params.name or timestamp_slug()}.parquet"
    pq.write_table(table, path)
    result = {"path": str(path), "rows": table.num_rows, "columns": table.column_names}
    update_summary(workspace, {"export": result})
    return result


def run_stage(
    workspace: Any,
    stage: str,
    params: dict[str, Any] | None,
    *,
    device: torch.device | str | None = None,
    progress: Progress | None = None,
    job: str | None = None,
) -> dict[str, Any]:
    """Run one stage over the workspace with ``params`` (unknown keys ignored); returns its summary.

    ``job`` labels the provenance of the rows the stage writes (default: the
    ``WORM_POSE_JOB_ID`` environment variable, else ``stage:<name>``).
    """

    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    params = dict(params or {})
    resolved = _device(device)
    job_id = _job_id(job, stage)
    with workspace_lock(workspace):
        if stage == "segment":
            return run_segment(workspace, SegmentParams.from_dict(params), device=resolved, progress=progress, job=job_id)
        if stage == "prior":
            return run_prior(workspace, PriorParams.from_dict(params), device=resolved, progress=progress, job=job_id, fit_params=FitParams.from_dict(params))
        if stage == "fit":
            return run_fit(workspace, FitParams.from_dict(params), device=resolved, progress=progress, job=job_id, all_params=params)
        if stage == "ambiguity":
            return run_ambiguity(workspace, AmbiguityParams.from_dict(params), device=resolved, progress=progress, job=job_id)
        if stage == "propagate":
            return run_propagate(workspace, PropagateParams.from_dict(params), device=resolved, progress=progress, job=job_id)
        if stage == "track":
            return run_track(workspace, TrackParams.from_dict(params), device=resolved, progress=progress, job=job_id)
        return run_export(workspace, ExportParams.from_dict(params), device=resolved, progress=progress, job=job_id)


def run_all(
    workspace: Any,
    params_by_stage: dict[str, dict[str, Any]] | None = None,
    stages: Sequence[str] = STAGES,
    *,
    device: torch.device | str | None = None,
    progress: Progress | None = None,
    job: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run ``stages`` in pipeline order; ``params_by_stage`` maps a stage to its parameters (a ``'*'`` entry applies to all)."""

    params_by_stage = params_by_stage or {}
    shared = dict(params_by_stage.get("*", {}))
    ordered = [s for s in STAGES if s in stages]
    results: dict[str, dict[str, Any]] = {}
    for k, stage in enumerate(ordered):
        def stage_progress(fraction: float, message: str, k: int = k, stage: str = stage) -> None:
            if progress is not None:
                progress((k + fraction) / len(ordered), f"{stage}: {message}")

        results[stage] = run_stage(workspace, stage, {**shared, **params_by_stage.get(stage, {})}, device=device, progress=stage_progress, job=job)
    return results


def stage_command(workspace_path: Path | str, stage: str, params: dict[str, Any] | None) -> list[str]:
    """The argv that runs one stage as a job (relative to the repo root, where the venv lives)."""

    return [
        ".venv/bin/python", "-m", "worm_pose_gen.pipeline",
        "--workspace", str(workspace_path), "--stage", stage, "--params", json.dumps(params or {}),
    ]


def _report_progress(progress: float, message: str, result: dict[str, Any] | None = None) -> None:
    """Write the job progress file (``jobs.report_progress`` when available, else the same JSON directly)."""

    try:
        from .jobs import report_progress
    except ImportError:
        path = os.environ.get("WORM_POSE_PROGRESS_FILE")
        if not path:
            return
        Path(path).write_text(json.dumps({"progress": float(progress), "message": str(message), "result": result}))
        return
    report_progress(progress, message, result)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one pipeline stage over a workspace.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--params", default="{}", help="JSON object of stage parameters")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    from .workspace import Workspace

    workspace = Workspace.open(args.workspace)
    params = json.loads(args.params)
    _report_progress(0.0, f"{args.stage}: starting")
    result = run_stage(workspace, args.stage, params, device=args.device, progress=_report_progress)
    _report_progress(1.0, f"{args.stage}: done", _json_safe(result))
    print(json.dumps(_json_safe(result), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
