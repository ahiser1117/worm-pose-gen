"""Exact, pixel-preserving camera-window crops of proxy-labelled real images."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.floating]


def support_bitmask(support: BoolArray) -> str:
    """Encode a one-dimensional support mapping without lossy packing metadata."""

    values = np.asarray(support)
    if values.ndim != 1 or values.dtype != np.bool_:
        raise ValueError("support must be a one-dimensional boolean array")
    return "".join("1" if value else "0" for value in values)


def canonical_manifest_sha256(manifest: dict[str, object]) -> str:
    """Hash a manifest using stable compact JSON serialization."""

    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CropRequest:
    """A request to hide one contiguous anatomical end of a centerline."""

    hidden_end: str
    hidden_fraction: float
    output_height: int
    output_width: int


@dataclass(frozen=True)
class RealCrop:
    """A direct source subwindow and its exact translation-only geometry."""

    image: NDArray[np.generic]
    centerline_xy: NDArray[np.float64]
    support: BoolArray
    source_origin_xy: tuple[int, int]
    source_shape: tuple[int, int]

    def source_to_crop(self, points_xy: FloatArray) -> NDArray[np.float64]:
        points = np.asarray(points_xy, dtype=np.float64)
        return points - np.asarray(self.source_origin_xy, dtype=np.float64)

    def crop_to_source(self, points_xy: FloatArray) -> NDArray[np.float64]:
        points = np.asarray(points_xy, dtype=np.float64)
        return points + np.asarray(self.source_origin_xy, dtype=np.float64)


@dataclass(frozen=True)
class CropAttempt:
    """Result of a deterministic crop search."""

    crop: RealCrop | None
    target_support: BoolArray
    rejection_reason: str | None


def target_support(num_points: int, hidden_end: str, hidden_fraction: float) -> BoolArray:
    """Define the requested anatomical support independently of crop geometry.

    The hidden count is ``ceil(num_points * hidden_fraction)``.  Thus the
    declared fraction is a lower bound and the complete complement must remain
    visible; no interior gaps or opposite-end truncation are permitted.
    """

    if num_points < 2:
        raise ValueError("at least two centerline points are required")
    if hidden_end not in {"head", "tail"}:
        raise ValueError("hidden_end must be 'head' or 'tail'")
    if not math.isfinite(hidden_fraction) or not 0 < hidden_fraction < 1:
        raise ValueError("hidden_fraction must lie strictly between zero and one")
    hidden = int(math.ceil(num_points * hidden_fraction - 1e-12))
    if hidden >= num_points:
        raise ValueError("hidden_fraction leaves no visible centerline points")
    support = np.ones(num_points, dtype=bool)
    if hidden_end == "head":
        support[:hidden] = False
    else:
        support[-hidden:] = False
    return support


def half_open_support(
    centerline_xy: FloatArray, origin_xy: tuple[int, int], height: int, width: int
) -> BoolArray:
    """Return exact point membership in an integer, half-open camera window."""

    points = np.asarray(centerline_xy)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centerline_xy must have shape [N, 2]")
    if height <= 0 or width <= 0:
        raise ValueError("crop dimensions must be positive")
    x0, y0 = origin_xy
    x, y = points[:, 0], points[:, 1]
    return (x >= x0) & (x < x0 + width) & (y >= y0) & (y < y0 + height)


def attempt_real_crop(
    image: NDArray[np.generic], centerline_xy: FloatArray, request: CropRequest
) -> CropAttempt:
    """Find a deterministic direct subwindow whose support exactly matches a request."""

    source = np.asarray(image)
    points = np.asarray(centerline_xy, dtype=np.float64)
    desired = target_support(len(points), request.hidden_end, request.hidden_fraction)
    if source.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("centerline_xy must have finite shape [N, 2]")
    height, width = request.output_height, request.output_width
    source_height, source_width = source.shape
    if height <= 0 or width <= 0 or height > source_height or width > source_width:
        raise ValueError("crop dimensions must be positive and fit inside the source")

    visible = points[desired]
    # Integer origins that contain every desired visible point under half-open bounds.
    x_low = max(0, math.floor(float(np.max(visible[:, 0]) - width)) + 1)
    x_high = min(source_width - width, math.floor(float(np.min(visible[:, 0]))))
    y_low = max(0, math.floor(float(np.max(visible[:, 1]) - height)) + 1)
    y_high = min(source_height - height, math.floor(float(np.min(visible[:, 1]))))
    if x_low > x_high or y_low > y_high:
        return CropAttempt(None, desired, "visible_support_does_not_fit")

    visible_center = np.mean(visible, axis=0)
    candidates: list[tuple[float, int, int]] = []
    for y0 in range(y_low, y_high + 1):
        for x0 in range(x_low, x_high + 1):
            if np.array_equal(half_open_support(points, (x0, y0), height, width), desired):
                crop_center = np.asarray((x0 + (width - 1) / 2, y0 + (height - 1) / 2))
                score = float(np.sum((crop_center - visible_center) ** 2))
                candidates.append((score, y0, x0))
    if not candidates:
        return CropAttempt(None, desired, "no_exact_support_window")

    _, y0, x0 = min(candidates)
    pixels = source[y0 : y0 + height, x0 : x0 + width].copy()
    crop_points = points - np.asarray((x0, y0), dtype=np.float64)
    crop = RealCrop(pixels, crop_points, desired.copy(), (x0, y0), source.shape)
    return CropAttempt(crop, desired, None)


def atomic_publish(
    source_path: str | Path, output_path: str | Path, write_partial: Callable[[Path], None]
) -> None:
    """Publish a file atomically while refusing collisions and overwrites."""

    source = Path(source_path).resolve(strict=True)
    output = Path(output_path).resolve(strict=False)
    partial = output.with_suffix(output.suffix + ".partial")
    if output == source or partial == source:
        raise ValueError("output collides with immutable source")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_partial(partial)
        if not partial.is_file():
            raise RuntimeError("writer did not create the requested partial file")
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
