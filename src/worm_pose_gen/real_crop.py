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
import torch
import torch.nn.functional as torch_functional


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


@dataclass(frozen=True)
class ScaledCropRequest:
    """A request for the smallest valid variable-size 4:3 source window."""

    hidden_end: str
    hidden_fraction: float
    output_height: int
    output_width: int
    k_min: int
    k_max: int


@dataclass(frozen=True)
class ScaledRealCrop:
    """A direct source window and its isotropically resized network image."""

    image: NDArray[np.float32]
    source_window: NDArray[np.generic]
    centerline_source_window_xy: NDArray[np.float64]
    centerline_resized_xy: NDArray[np.float64]
    support: BoolArray
    source_origin_xy: tuple[int, int]
    source_window_k: int
    source_shape: tuple[int, int]
    output_shape: tuple[int, int]

    @property
    def scale(self) -> float:
        return self.output_shape[1] / (4 * self.source_window_k)

    @property
    def source_window_shape(self) -> tuple[int, int]:
        return (3 * self.source_window_k, 4 * self.source_window_k)

    def source_to_window(self, points_xy: FloatArray) -> NDArray[np.float64]:
        return np.asarray(points_xy, dtype=np.float64) - np.asarray(
            self.source_origin_xy, dtype=np.float64
        )

    def window_to_source(self, points_xy: FloatArray) -> NDArray[np.float64]:
        return np.asarray(points_xy, dtype=np.float64) + np.asarray(
            self.source_origin_xy, dtype=np.float64
        )

    def source_to_resized(self, points_xy: FloatArray) -> NDArray[np.float64]:
        # Edge-aligned FOV geometry keeps [0, 4k)x[0, 3k) mapped exactly to
        # [0, 256)x[0, 192). Pixel resampling separately follows the frozen
        # align_corners=False center-sampling rule.
        return self.source_to_window(points_xy) * self.scale

    def resized_to_source(self, points_xy: FloatArray) -> NDArray[np.float64]:
        return self.window_to_source(np.asarray(points_xy, dtype=np.float64) / self.scale)


@dataclass(frozen=True)
class ScaledCropAttempt:
    """Result of the deterministic smallest-window search."""

    crop: ScaledRealCrop | None
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


def bilinear_resize_align_corners_false(
    image: NDArray[np.generic], output_height: int, output_width: int
) -> NDArray[np.float32]:
    """Resize one grayscale image with the frozen CPU interpolation contract."""

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output dimensions must be positive")
    tensor = torch.from_numpy(np.ascontiguousarray(source)).to(dtype=torch.float32)
    resized = torch_functional.interpolate(
        tensor[None, None],
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy().copy()


def _closest_unblocked_integer(
    low: int, high: int, ideal: float, blocked: list[tuple[int, int]]
) -> int | None:
    """Return the closest integer to ideal outside inclusive blocked intervals."""

    candidates = {low, high, math.floor(ideal), math.ceil(ideal)}
    for start, stop in blocked:
        candidates.add(start - 1)
        candidates.add(stop + 1)
    allowed = [
        value
        for value in candidates
        if low <= value <= high
        and not any(start <= value <= stop for start, stop in blocked)
    ]
    if not allowed:
        return None
    return min(allowed, key=lambda value: ((value - ideal) ** 2, value))


def _find_exact_window_origin(
    points: NDArray[np.float64], desired: BoolArray, source_shape: tuple[int, int], k: int
) -> tuple[int, int] | None:
    """Find the centered deterministic origin for one fixed 4k by 3k window."""

    source_height, source_width = source_shape
    height, width = 3 * k, 4 * k
    if height > source_height or width > source_width:
        return None
    visible = points[desired]
    hidden = points[~desired]
    x_low = max(0, math.floor(float(np.max(visible[:, 0]) - width)) + 1)
    x_high = min(source_width - width, math.floor(float(np.min(visible[:, 0]))))
    y_low = max(0, math.floor(float(np.max(visible[:, 1]) - height)) + 1)
    y_high = min(source_height - height, math.floor(float(np.min(visible[:, 1]))))
    if x_low > x_high or y_low > y_high:
        return None

    visible_center = np.mean(visible, axis=0)
    x_ideal = float(visible_center[0] - (width - 1) / 2)
    y_ideal = float(visible_center[1] - (height - 1) / 2)
    x_candidates = {x_low, x_high, math.floor(x_ideal), math.ceil(x_ideal)}
    for hidden_x, _ in hidden:
        start = max(x_low, math.floor(float(hidden_x - width)) + 1)
        stop = min(x_high, math.floor(float(hidden_x)))
        if start <= stop:
            x_candidates.update((start - 1, start, stop, stop + 1))
    best: tuple[float, int, int] | None = None
    for x0 in sorted(value for value in x_candidates if x_low <= value <= x_high):
        blocked_y: list[tuple[int, int]] = []
        for hidden_x, hidden_y in hidden:
            if x0 <= hidden_x < x0 + width:
                start = max(y_low, math.floor(float(hidden_y - height)) + 1)
                stop = min(y_high, math.floor(float(hidden_y)))
                if start <= stop:
                    blocked_y.append((start, stop))
        y0 = _closest_unblocked_integer(y_low, y_high, y_ideal, blocked_y)
        if y0 is None:
            continue
        score = (x0 - x_ideal) ** 2 + (y0 - y_ideal) ** 2
        candidate = (score, y0, x0)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return (best[2], best[1])


def attempt_scaled_real_crop(
    image: NDArray[np.generic], centerline_xy: FloatArray, request: ScaledCropRequest
) -> ScaledCropAttempt:
    """Search smallest exact-support 4:3 source window and resize it isotropically."""

    source = np.asarray(image)
    points = np.asarray(centerline_xy, dtype=np.float64)
    desired = target_support(len(points), request.hidden_end, request.hidden_fraction)
    if source.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("centerline_xy must have finite shape [N, 2]")
    if request.output_height <= 0 or request.output_width <= 0:
        raise ValueError("output dimensions must be positive")
    if request.output_width * 3 != request.output_height * 4:
        raise ValueError("output dimensions must have 4:3 aspect ratio")
    if request.k_min <= 0 or request.k_max < request.k_min:
        raise ValueError("k range must be positive and nonempty")
    if request.output_width / 4 != request.output_height / 3:
        raise ValueError("output mapping must be isotropic")

    visible = points[desired]
    max_height, max_width = 3 * request.k_max, 4 * request.k_max
    maximum_can_contain_visible = (
        max_height <= source.shape[0]
        and max_width <= source.shape[1]
        and float(np.ptp(visible[:, 0])) < max_width
        and float(np.ptp(visible[:, 1])) < max_height
    )
    for k in range(request.k_min, request.k_max + 1):
        origin = _find_exact_window_origin(points, desired, source.shape, k)
        if origin is None:
            continue
        x0, y0 = origin
        height, width = 3 * k, 4 * k
        window = source[y0 : y0 + height, x0 : x0 + width].copy()
        output = bilinear_resize_align_corners_false(
            window, request.output_height, request.output_width
        )
        scale = request.output_width / width
        window_points = points - np.asarray((x0, y0), dtype=np.float64)
        resized_points = window_points * scale
        crop = ScaledRealCrop(
            output,
            window,
            window_points,
            resized_points,
            desired.copy(),
            (x0, y0),
            k,
            source.shape,
            (request.output_height, request.output_width),
        )
        return ScaledCropAttempt(crop, desired, None)
    reason = (
        "no_exact_support_window_any_scale"
        if maximum_can_contain_visible
        else "visible_support_does_not_fit_maximum_window"
    )
    return ScaledCropAttempt(None, desired, reason)


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
