"""Per-recording body-size priors for the tube fitter (plan step 3).

Body length, width scale, and the shape of the width profile vary between
recordings and hardly within one.  A bootstrap pass fits a spread sample of
frames with the hard bounds opened wide, keeps the fits that are fully in
view, clean, and confidently oriented, and takes robust medians.  The
resulting ``RecordingPrior`` replaces the hard length and width bounds with
Gaussian priors and centers the width correction on the recording's profile
(tail last), so every start is tried in both orientations and the energy
decides which end is the tail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .mask_fit import (
    Initialization,
    MaskFitConfig,
    MaskFitResult,
    hard_iou,
    orient_tail_last,
    standard_initializations,
    taper_asymmetry,
)
from .pose_run import touches_border, tube_area_px


# Bounds used while bootstrapping: wide enough never to bind on a real worm,
# present so a runaway start cannot leave the crop with an unbounded length.
BOOTSTRAP_LENGTH_BOUNDS_PX = (100.0, 3000.0)
BOOTSTRAP_WIDTH_BOUNDS_PX = (5.0, 300.0)
# Selection ladder: the first rung with enough frames is used.
# The taper requirement orients profiles before the median; a recording whose
# whole worms show little asymmetry drops it, since near-symmetric profiles
# barely move the median either way.
SELECTION_LADDER: tuple[dict[str, float], ...] = (
    {"min_iou": 0.9, "min_taper": 0.3, "area_ratio": 0.1},
    {"min_iou": 0.9, "min_taper": 0.15, "area_ratio": 0.1},
    {"min_iou": 0.85, "min_taper": 0.1, "area_ratio": 0.15},
    {"min_iou": 0.85, "min_taper": 0.0, "area_ratio": 0.15},
)


@dataclass(frozen=True)
class RecordingPrior:
    """Robust body-size summary of one recording, oriented tail last."""

    length_px: float
    log_length_sigma: float
    width_px: float
    log_width_sigma: float
    width_shape: tuple[float, ...]
    width_shape_sigma: tuple[float, ...]
    frames_used: int
    frames_candidates: int
    selection: dict[str, float]
    recording: str | None = None
    source: str | None = None

    def apply(self, config: MaskFitConfig, *, shape_weight: float = 0.01) -> MaskFitConfig:
        """The configuration with bounds removed and priors centered on this recording.

        ``shape_weight`` replaces the weak pull toward zero of the symmetric
        model.  On one minute of 2024-05-28-02, 0.01 and 0.05 gave the same
        median IoU (0.969 vs 0.967) and the same orientation gap; the lighter
        weight leaves the profile freer on frames whose mask disagrees with
        the recording's median profile.
        """

        if len(self.width_shape) != config.width_coefficients:
            raise ValueError(
                f"prior has {len(self.width_shape)} width coefficients, config expects {config.width_coefficients}"
            )
        return replace(
            config,
            length_bounds_px=None,
            width_bounds_px=None,
            length_prior_px=self.length_px,
            length_prior_log_sigma=self.log_length_sigma,
            width_prior_px=self.width_px,
            width_prior_log_sigma=self.log_width_sigma,
            width_shape_prior_mean=tuple(self.width_shape),
            width_shape_prior=shape_weight,
            default_length_px=self.length_px,
            default_width_px=self.width_px,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordingPrior":
        values = dict(data)
        values["width_shape"] = tuple(float(v) for v in values["width_shape"])
        values["width_shape_sigma"] = tuple(float(v) for v in values["width_shape_sigma"])
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "RecordingPrior":
        return cls.from_dict(json.loads(path.read_text()))


def bootstrap_config(config: MaskFitConfig) -> MaskFitConfig:
    """The configuration for the bootstrap pass: wide bounds, no priors, symmetric center."""

    return replace(
        config,
        length_bounds_px=BOOTSTRAP_LENGTH_BOUNDS_PX,
        width_bounds_px=BOOTSTRAP_WIDTH_BOUNDS_PX,
        length_prior_px=None,
        width_prior_px=None,
        width_shape_prior_mean=None,
    )


def _robust_sigma(values: NDArray[np.float64], floor: float) -> float:
    if len(values) < 2:
        return floor
    mad = float(np.median(np.abs(values - np.median(values))))
    return max(1.4826 * mad, floor)


def select_confident(
    results: Sequence[MaskFitResult],
    masks: Sequence[NDArray[np.generic]],
    *,
    config: MaskFitConfig,
    min_iou: float,
    min_taper: float,
    area_ratio: float,
) -> list[MaskFitResult]:
    """Whole, well-fit, confidently oriented fits, each turned tail last.

    Whole means the mask does not reach the image border: a body cut off by
    the camera edge keeps all its centerline points in view, so the in-view
    count alone would admit it and bias the length prior short.
    """

    chosen = []
    for result, mask in zip(results, masks, strict=True):
        binary = np.asarray(mask, dtype=bool)
        if result.points_in_fov < config.n_points or touches_border(binary, margin=2):
            continue
        crop = result.crop
        if hard_iou(result.rendered_hard_mask, binary[crop.y0 : crop.y1, crop.x0 : crop.x1]) < min_iou:
            continue
        ratio = float(binary.sum()) / max(tube_area_px(result.width_profile, result.body_length_px), 1.0)
        if abs(ratio - 1.0) > area_ratio:
            continue
        if abs(taper_asymmetry(result.width_profile)) < min_taper:
            continue
        oriented, _ = orient_tail_last(result, config=config)
        chosen.append(oriented)
    return chosen


def estimate_recording_prior(
    results: Sequence[MaskFitResult],
    masks: Sequence[NDArray[np.generic]],
    *,
    config: MaskFitConfig,
    min_frames: int = 6,
    log_sigma_floor: float = 0.05,
    shape_sigma_floor: float = 0.05,
    recording: str | None = None,
    source: str | None = None,
) -> RecordingPrior:
    """Robust medians of length, width scale, and width correction over confident fits.

    The log-sigma floor of 5% is the frame-to-frame spread of a whole worm's
    fitted length and width across postures; a tighter prior costs overlap on
    whole worms (measured on 2024-01-31-02: 3% cost 0.012 IoU, 5% cost 0.003).
    """

    if len(results) != len(masks):
        raise ValueError("results and masks must align")
    chosen: list[MaskFitResult] = []
    selection: dict[str, float] = {}
    for rung in SELECTION_LADDER:
        chosen = select_confident(results, masks, config=config, **rung)
        selection = dict(rung)
        if len(chosen) >= min_frames:
            break
    if len(chosen) < min_frames:
        whole = sum(1 for m in masks if not touches_border(np.asarray(m, dtype=bool), margin=2))
        raise ValueError(
            f"only {len(chosen)} of {len(results)} bootstrap fits are usable ({whole} masks clear of the image border);"
            f" {min_frames} are required"
        )
    log_lengths = np.log([r.body_length_px for r in chosen])
    log_widths = np.log([r.width_px for r in chosen])
    shapes = np.stack([r.width_shape for r in chosen]) if config.width_coefficients else np.zeros((len(chosen), 0))
    shape_sigma = tuple(float(_robust_sigma(shapes[:, k], shape_sigma_floor)) for k in range(shapes.shape[1]))
    return RecordingPrior(
        length_px=float(math.exp(np.median(log_lengths))),
        log_length_sigma=_robust_sigma(log_lengths, log_sigma_floor),
        width_px=float(math.exp(np.median(log_widths))),
        log_width_sigma=_robust_sigma(log_widths, log_sigma_floor),
        width_shape=tuple(float(v) for v in np.median(shapes, axis=0)),
        width_shape_sigma=shape_sigma,
        frames_used=len(chosen),
        frames_candidates=len(results),
        selection=selection,
        recording=recording,
        source=source,
    )


def bootstrap_prior_from_masks(
    masks: Sequence[NDArray[np.generic]],
    *,
    config: MaskFitConfig,
    device: Any = None,
    start_names: tuple[str, ...] = ("skeleton_longest_path", "moments_straight"),
    recording: str | None = None,
    source: str | None = None,
    **estimate_kwargs: Any,
) -> tuple[RecordingPrior, list[MaskFitResult], list[NDArray[np.bool_]]]:
    """Fit cleaned masks with the bootstrap configuration and estimate the prior.

    Returns the prior, the bootstrap fits, and the masks that were fit (empty
    masks and masks without a start are dropped), in matching order.
    """

    from .batch_fit import BatchFitConfig, fit_masks

    if not isinstance(config, BatchFitConfig):
        raise TypeError("bootstrap_prior_from_masks needs a BatchFitConfig")
    fit_config = bootstrap_config(config)
    usable: list[NDArray[np.bool_]] = []
    starts: list[list[Initialization]] = []
    for mask in masks:
        binary = np.asarray(mask, dtype=bool)
        if not binary.any():
            continue
        candidates = standard_initializations(binary, config=fit_config)
        chosen = [s for s in candidates if s.name in start_names] or candidates[:1]
        if chosen:
            usable.append(binary)
            starts.append(chosen)
    if not usable:
        raise ValueError("no bootstrap mask produced a start")
    results = fit_masks(usable, starts, config=fit_config, device=device)
    prior = estimate_recording_prior(results, usable, config=config, recording=recording, source=source, **estimate_kwargs)
    return prior, results, usable
