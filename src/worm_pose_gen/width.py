"""Low-dimensional empirical width-profile models for mask rendering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WidthProfileModel:
    """Mean width and optional orthonormal residual components."""

    mean: FloatArray
    components: FloatArray
    minimum: float
    maximum: float


def fit_width_profile_model(
    profiles: NDArray[np.generic],
    *,
    components: int = 0,
    minimum: float = 1.0,
    maximum: float = 80.0,
) -> WidthProfileModel:
    """Fit a mean profile and up to ``components`` residual PCA directions."""

    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("profiles must have shape [frames>=1, positions>=2]")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("profiles must be finite and positive")
    if components < 0:
        raise ValueError("components must be non-negative")
    if not np.isfinite(minimum) or not np.isfinite(maximum) or not 0 < minimum < maximum:
        raise ValueError("minimum and maximum must be finite with 0 < minimum < maximum")
    mean = values.mean(axis=0)
    effective = min(int(components), max(0, values.shape[0] - 1), values.shape[1])
    if effective:
        _, _, vh = np.linalg.svd(values - mean, full_matrices=False)
        basis = vh[:effective].astype(np.float64, copy=False)
    else:
        basis = np.empty((0, values.shape[1]), dtype=np.float64)
    return WidthProfileModel(mean.astype(np.float64), basis, float(minimum), float(maximum))


def reconstruct_width_profile(
    model: WidthProfileModel,
    *,
    scale: float = 1.0,
    coefficients: NDArray[np.generic] | None = None,
) -> FloatArray:
    """Reconstruct a bounded positive profile from scale and residual terms."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if coefficients is None:
        coefficient = np.zeros(len(model.components), dtype=np.float64)
    else:
        coefficient = np.asarray(coefficients, dtype=np.float64)
        if coefficient.shape != (len(model.components),) or not np.all(np.isfinite(coefficient)):
            raise ValueError("coefficients must match model components and be finite")
    profile = scale * model.mean
    if len(model.components):
        profile = profile + coefficient @ model.components
    return np.clip(profile, model.minimum, model.maximum)


def fit_profile_parameters(
    target: NDArray[np.generic],
    model: WidthProfileModel,
    *,
    fit_scale: bool,
    scale_bounds: tuple[float, float] = (0.8, 1.2),
) -> tuple[FloatArray, float, FloatArray]:
    """Oracle-fit a target profile for controlled width-capacity comparison."""

    values = np.asarray(target, dtype=np.float64)
    if values.shape != model.mean.shape or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("target must be a finite positive profile matching the model")
    lower, upper = map(float, scale_bounds)
    if not 0 < lower <= upper or not np.isfinite(lower + upper):
        raise ValueError("scale_bounds must be finite, positive, and ordered")
    scale = 1.0
    if fit_scale:
        denominator = float(np.dot(model.mean, model.mean))
        scale = float(np.clip(np.dot(values, model.mean) / denominator, lower, upper))
    residual = values - scale * model.mean
    coefficient = model.components @ residual if len(model.components) else np.empty(0)
    profile = reconstruct_width_profile(model, scale=scale, coefficients=coefficient)
    return profile, scale, coefficient.astype(np.float64, copy=False)
