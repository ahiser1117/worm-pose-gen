"""Classical soft-foreground baseline for segmentation experiments.

This is deliberately not a pretrained segmenter.  It turns the existing
robust local dark-ridge score into a bounded soft foreground map, then exposes
both the raw threshold and a minimally cleaned largest connected component.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .classical import ClassicalConfig, _dilate, _erode, _largest_component, robust_dark_ridge


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SoftForegroundConfig:
    """Configuration for :func:`classical_soft_foreground`.

    ``score_midpoint`` is expressed in robust dark-ridge z-score units.
    ``logistic_temperature`` controls softness, while ``probability_threshold``
    independently controls conversion to the raw binary mask.
    """

    local_radius: int = 31
    smooth_radius: int = 2
    score_midpoint: float = 2.6
    logistic_temperature: float = 0.5
    probability_threshold: float = 0.5
    # Disabled by default to preserve the frozen EXP-SMC-001 baseline. When
    # set, low-confidence pixels are admitted only when 8-connected to the
    # high-threshold largest component.
    low_probability_threshold: float | None = None
    close_radius: int = 2
    max_hole_area: int = 64


@dataclass(frozen=True)
class SoftForegroundResult:
    probability_map: FloatArray
    raw_mask: BoolArray
    cleaned_mask: BoolArray
    qc: dict[str, float | int | bool | str]

    @property
    def probability(self) -> FloatArray:
        """Stable short alias used by downstream observation models."""

        return self.probability_map


def fill_small_enclosed_holes(
    mask: NDArray[np.generic], max_hole_area: int
) -> tuple[BoolArray, int, int]:
    """Fill bounded background components no larger than ``max_hole_area``.

    Background connected to any image border is never filled. Connectivity is
    8-neighbor, matching foreground component cleanup and preventing diagonal
    cracks from being mistaken for enclosed holes.
    """

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if max_hole_area < 0:
        raise ValueError("max_hole_area must be non-negative")
    if max_hole_area == 0:
        return values.copy(), 0, 0
    height, width = values.shape
    background = ~values
    visited = np.zeros_like(values)
    output = values.copy()
    filled_count = 0
    filled_area = 0
    for y, x in np.argwhere(background):
        yi, xi = int(y), int(x)
        if visited[yi, xi]:
            continue
        queue = deque([(yi, xi)])
        visited[yi, xi] = True
        component: list[tuple[int, int]] = []
        touches_border = False
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            touches_border |= cy == 0 or cx == 0 or cy == height - 1 or cx == width - 1
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if background[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        if not touches_border and len(component) <= max_hole_area:
            yy, xx = np.asarray(component).T
            output[yy, xx] = True
            filled_count += 1
            filled_area += len(component)
    return output, filled_count, filled_area


def connected_low_threshold_extension(
    high_confidence_component: NDArray[np.generic],
    low_threshold_mask: NDArray[np.generic],
) -> tuple[BoolArray, int, int]:
    """Grow a high-confidence seed through connected low-threshold pixels.

    Returns the extended component, number of newly recovered pixels, and the
    number of low-threshold pixels excluded because they are disconnected from
    the seed. Seed pixels are retained even if morphology placed them just
    outside the low-threshold mask.
    """

    seed = np.asarray(high_confidence_component, dtype=bool)
    eligible = np.asarray(low_threshold_mask, dtype=bool)
    if seed.ndim != 2 or eligible.shape != seed.shape:
        raise ValueError("high-confidence and low-threshold masks must share a 2-D shape")
    output = seed.copy()
    height, width = seed.shape
    queue = deque((int(y), int(x)) for y, x in np.argwhere(seed))
    while queue:
        cy, cx = queue.popleft()
        for ny in range(max(0, cy - 1), min(height, cy + 2)):
            for nx in range(max(0, cx - 1), min(width, cx + 2)):
                if eligible[ny, nx] and not output[ny, nx]:
                    output[ny, nx] = True
                    queue.append((ny, nx))
    recovered = int(np.logical_and(output, ~seed).sum())
    disconnected = int(np.logical_and(eligible, ~output).sum())
    return output, recovered, disconnected


def segment_soft_foreground(
    image: NDArray[np.generic],
    config: SoftForegroundConfig | None = None,
) -> SoftForegroundResult:
    """Compute a classical probability-like foreground baseline.

    The returned values are useful for plumbing and controlled baseline
    experiments, but are not calibrated probabilities from a learned model.
    Cleanup consists only of binary closing and largest-component retention.
    """

    cfg = config or SoftForegroundConfig()
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("image must have shape [height, width]")
    if cfg.local_radius < 0 or cfg.smooth_radius < 0 or cfg.close_radius < 0:
        raise ValueError("radii must be non-negative")
    if cfg.max_hole_area < 0:
        raise ValueError("max_hole_area must be non-negative")
    if not np.isfinite(cfg.logistic_temperature) or cfg.logistic_temperature <= 0:
        raise ValueError("logistic_temperature must be finite and positive")
    if not 0 < cfg.probability_threshold < 1:
        raise ValueError("probability_threshold must lie strictly between zero and one")
    if cfg.low_probability_threshold is not None and not (
        0 < cfg.low_probability_threshold < cfg.probability_threshold
    ):
        raise ValueError(
            "low_probability_threshold must lie strictly between zero and probability_threshold"
        )
    if not np.isfinite(cfg.score_midpoint):
        raise ValueError("score_midpoint must be finite")

    score_config = ClassicalConfig(
        local_radius=cfg.local_radius,
        smooth_radius=cfg.smooth_radius,
    )
    score = robust_dark_ridge(values, score_config)
    logit = np.clip(
        (score - cfg.score_midpoint) / cfg.logistic_temperature,
        -60.0,
        60.0,
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    raw = probability >= cfg.probability_threshold
    closed = _erode(_dilate(raw, cfg.close_radius), cfg.close_radius)
    high_component, high_component_area, component_count = _largest_component(closed)
    component = high_component
    low_area = 0
    hysteresis_recovered_area = 0
    hysteresis_disconnected_low_area = 0
    if cfg.low_probability_threshold is not None:
        low_mask = probability >= cfg.low_probability_threshold
        low_area = int(low_mask.sum())
        component, hysteresis_recovered_area, hysteresis_disconnected_low_area = (
            connected_low_threshold_extension(high_component, low_mask)
        )
    component_area = int(component.sum())
    cleaned, filled_hole_count, filled_hole_area = fill_small_enclosed_holes(
        component, cfg.max_hole_area
    )
    cleaned_area = int(cleaned.sum())
    touches_boundary = bool(
        cleaned_area
        and (
            np.any(cleaned[0])
            or np.any(cleaned[-1])
            or np.any(cleaned[:, 0])
            or np.any(cleaned[:, -1])
        )
    )
    qc: dict[str, float | int | bool | str] = {
        "method": "classical_robust_dark_ridge_logistic_baseline",
        "is_pretrained": False,
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "raw_foreground_area": int(raw.sum()),
        "closed_foreground_area": int(closed.sum()),
        "largest_high_confidence_component_area": int(high_component_area),
        "hysteresis_enabled": cfg.low_probability_threshold is not None,
        "hysteresis_low_probability_threshold": (
            "disabled"
            if cfg.low_probability_threshold is None
            else float(cfg.low_probability_threshold)
        ),
        "low_threshold_foreground_area": low_area,
        "hysteresis_recovered_area": hysteresis_recovered_area,
        "hysteresis_disconnected_low_area": hysteresis_disconnected_low_area,
        "largest_component_area_before_hole_fill": int(component_area),
        "cleaned_foreground_area": int(cleaned_area),
        "filled_hole_count": int(filled_hole_count),
        "filled_hole_area": int(filled_hole_area),
        "closed_component_count": int(component_count),
        "cleaned_touches_boundary": touches_boundary,
    }
    return SoftForegroundResult(
        probability.astype(np.float64, copy=False),
        raw.astype(bool, copy=False),
        cleaned.astype(bool, copy=False),
        qc,
    )


# Backward-compatible descriptive aliases retained for callers that want to
# emphasize that this is a classical baseline rather than a learned segmenter.
ClassicalSoftForegroundConfig = SoftForegroundConfig
ClassicalSoftForegroundResult = SoftForegroundResult
classical_soft_foreground = segment_soft_foreground
