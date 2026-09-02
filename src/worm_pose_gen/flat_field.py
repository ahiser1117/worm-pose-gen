"""Approximate flat-field correction for vignetted NIR recordings.

The illumination field is estimated from multiple frames rather than from the
frame being segmented.  A high temporal quantile suppresses dark, moving
objects and broad spatial smoothing removes the remaining object-scale detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _finite_2d(values: NDArray[np.generic], name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _box_blur(image: FloatArray, radius: int) -> FloatArray:
    if radius < 0:
        raise ValueError("spatial_radius must be non-negative")
    if radius == 0:
        return image.copy()
    padded = np.pad(image, radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    width = 2 * radius + 1
    total = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return total / float(width * width)


def estimate_illumination(
    frames: Iterable[NDArray[np.generic]],
    *,
    temporal_quantile: float = 0.8,
    spatial_radius: int = 31,
    smoothing_passes: int = 2,
) -> FloatArray:
    """Estimate a recording-level illumination field.

    ``frames`` may be any finite iterable of equally shaped 2-D arrays.  For a
    dark worm, an upper temporal quantile approximates the empty background.
    Repeated broad box blurs retain the vignette while suppressing residual
    worm-scale structure.  The exact quantile requires holding the sampled
    calibration frames in memory; callers should pass a representative subset
    for long recordings.
    """

    if not 0.0 <= temporal_quantile <= 1.0:
        raise ValueError("temporal_quantile must be between zero and one")
    if spatial_radius < 0:
        raise ValueError("spatial_radius must be non-negative")
    if smoothing_passes < 0:
        raise ValueError("smoothing_passes must be non-negative")

    collected: list[FloatArray] = []
    shape: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        values = _finite_2d(frame, f"frames[{index}]")
        if shape is None:
            shape = values.shape
        elif values.shape != shape:
            raise ValueError("all frames must have the same shape")
        collected.append(values)
    if not collected:
        raise ValueError("frames must contain at least one frame")

    illumination = np.quantile(
        np.stack(collected, axis=0), temporal_quantile, axis=0
    )
    for _ in range(smoothing_passes):
        illumination = _box_blur(illumination, spatial_radius)
    return np.asarray(illumination, dtype=np.float64)


def _central_values(image: FloatArray, fraction: float) -> FloatArray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("reference_fraction must be in (0, 1]")
    height, width = image.shape
    crop_height = max(1, int(round(height * fraction)))
    crop_width = max(1, int(round(width * fraction)))
    y0 = (height - crop_height) // 2
    x0 = (width - crop_width) // 2
    return image[y0 : y0 + crop_height, x0 : x0 + crop_width]


@dataclass(frozen=True)
class FlatField:
    """A fitted, reusable divisive flat-field correction."""

    illumination: FloatArray
    dark_level: float
    reference_level: float
    gain: FloatArray

    def apply(
        self,
        image: NDArray[np.generic],
        *,
        clip: tuple[float, float] | None = None,
    ) -> FloatArray:
        """Correct one frame, returning float64 without clipping by default."""

        values = _finite_2d(image, "image")
        if values.shape != self.illumination.shape:
            raise ValueError("image and illumination must have the same shape")
        corrected = self.dark_level + (values - self.dark_level) * self.gain
        if clip is not None:
            lower, upper = map(float, clip)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError("clip must be a finite (lower, upper) pair")
            corrected = np.clip(corrected, lower, upper)
        return np.asarray(corrected, dtype=np.float64)


def fit_flat_field(
    frames: Iterable[NDArray[np.generic]],
    *,
    temporal_quantile: float = 0.8,
    spatial_radius: int = 31,
    smoothing_passes: int = 2,
    dark_level: float = 0.0,
    reference_level: float | None = None,
    reference_fraction: float = 0.5,
    minimum_signal: float = 1.0,
    min_gain: float = 0.5,
    max_gain: float = 2.5,
) -> FlatField:
    """Fit an illumination field and its bounded multiplicative gain.

    When ``reference_level`` is omitted, the median illumination in the central
    ``reference_fraction`` of the image is used.  ``minimum_signal`` prevents
    division by pixels at or below the camera dark level.  Explicit gain bounds
    limit noise amplification in poorly illuminated corners.
    """

    for value, name in (
        (dark_level, "dark_level"),
        (minimum_signal, "minimum_signal"),
        (min_gain, "min_gain"),
        (max_gain, "max_gain"),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if minimum_signal <= 0.0:
        raise ValueError("minimum_signal must be positive")
    if min_gain <= 0.0 or max_gain < min_gain:
        raise ValueError("gain bounds must satisfy 0 < min_gain <= max_gain")

    illumination = estimate_illumination(
        frames,
        temporal_quantile=temporal_quantile,
        spatial_radius=spatial_radius,
        smoothing_passes=smoothing_passes,
    )
    signal = illumination - float(dark_level)
    if reference_level is None:
        reference_signal = float(np.median(_central_values(signal, reference_fraction)))
        fitted_reference = float(dark_level) + reference_signal
    else:
        fitted_reference = float(reference_level)
        if not np.isfinite(fitted_reference):
            raise ValueError("reference_level must be finite")
        reference_signal = fitted_reference - float(dark_level)
    if reference_signal <= 0.0:
        raise ValueError("reference_level must be above dark_level")

    denominator = np.maximum(signal, float(minimum_signal))
    gain = np.clip(reference_signal / denominator, min_gain, max_gain)
    if not np.all(np.isfinite(gain)):
        raise ValueError("flat-field gain is not finite")
    return FlatField(
        illumination=illumination,
        dark_level=float(dark_level),
        reference_level=fitted_reference,
        gain=np.asarray(gain, dtype=np.float64),
    )


def estimate_flat_field(
    frames: Iterable[NDArray[np.generic]],
    *,
    temporal_quantile: float = 0.8,
    spatial_radius: int = 31,
    smoothing_passes: int = 2,
    dark_level: float = 0.0,
    reference_level: float | None = None,
    reference_fraction: float = 0.5,
    minimum_signal: float = 1.0,
    min_gain: float = 0.5,
    max_gain: float = 2.5,
) -> FlatField:
    """Public integration alias for :func:`fit_flat_field`.

    The returned object includes both the illumination estimate and correction
    diagnostics, while remaining directly reusable across recording frames.
    """

    return fit_flat_field(
        frames,
        temporal_quantile=temporal_quantile,
        spatial_radius=spatial_radius,
        smoothing_passes=smoothing_passes,
        dark_level=dark_level,
        reference_level=reference_level,
        reference_fraction=reference_fraction,
        minimum_signal=minimum_signal,
        min_gain=min_gain,
        max_gain=max_gain,
    )


def apply_flat_field(
    image: NDArray[np.generic],
    field: FlatField,
    *,
    clip: tuple[float, float] | None = None,
) -> FloatArray:
    """Apply a fitted flat field to one image."""

    if not isinstance(field, FlatField):
        raise TypeError("field must be a FlatField")
    return field.apply(image, clip=clip)
