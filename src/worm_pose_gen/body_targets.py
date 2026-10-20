"""Per-pixel body-field targets rendered from a fitted, head-first tube.

A hand-labeled mask says which pixels are worm but not which part of the worm
they are.  Fitting the tube model to that mask and orienting it head first
gives the missing labels as deterministic functions of the pose:

``ap``
    normalized arc length (0 at the head, 1 at the tail) of the body segment
    each mask pixel belongs to.  Where two parts of the body touch, the field
    jumps across the contact line, which is what lets a network trained on it
    tell touching limbs apart.
``overlap``
    pixels the tube covers twice with body parts far apart along the body: a
    crossing, where no single arc length is correct.  ``ap`` is undefined
    there and the network predicts the overlap itself instead.
head/tail points
    the two ends of the tube; heatmap targets are Gaussians rendered from
    them (:func:`point_heatmap`).

Coordinates are ``(x, y)`` pixel centers of the full image, as in
:mod:`worm_pose_gen.mask_fit`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

# Two covering segments are separate body parts when they lie more than this
# many body diameters apart along the body.  A bend of the minimum radius the
# fitter allows (0.5 widths) folds the inner side of the tube over itself
# within about one diameter of arc, which is not a crossing.
OVERLAP_ARC_DIAMETERS = 2.0


@dataclass(frozen=True)
class BodyTargets:
    ap: NDArray[np.float32]  # [H,W]; NaN off the mask and on overlaps
    overlap: NDArray[np.bool_]  # [H,W]
    head_xy: FloatArray
    tail_xy: FloatArray
    diameter_px: float  # median tube diameter, the scale of the heatmaps


def _segment_distances(points: FloatArray, centerline: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Distance from each point to each centerline segment, and the segment parameter t."""

    start = centerline[None, :-1, :]
    segment = (centerline[1:] - centerline[:-1])[None, :, :]
    length_sq = np.maximum((segment**2).sum(-1), 1e-12)
    t = np.clip(((points[:, None, :] - start) * segment).sum(-1) / length_sq, 0.0, 1.0)
    closest = start + t[..., None] * segment
    return np.sqrt(((points[:, None, :] - closest) ** 2).sum(-1)), t


def render_body_targets(
    mask: NDArray[np.generic],
    centerline_xy: NDArray[np.generic],
    width_profile: NDArray[np.generic],
    *,
    chunk_pixels: int = 8192,
) -> BodyTargets:
    """A-P field and overlap for every pixel of ``mask`` from a head-first tube.

    Each mask pixel takes the arc position of the segment whose tube surface
    is nearest (largest signed coverage ``radius - distance``), so a pixel
    just outside the fitted tube still gets the part of the body it borders.
    """

    binary = np.asarray(mask, dtype=bool)
    centerline = np.asarray(centerline_xy, dtype=np.float64)
    diameter = np.asarray(width_profile, dtype=np.float64)
    if centerline.ndim != 2 or centerline.shape[1] != 2 or len(centerline) < 2:
        raise ValueError("centerline_xy must have shape [N>=2,2]")
    if diameter.shape != (len(centerline),):
        raise ValueError("width_profile must have one diameter per centerline point")
    step = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(step)))
    total = max(float(arc[-1]), 1e-9)
    median_diameter = float(np.median(diameter))
    separation = OVERLAP_ARC_DIAMETERS * median_diameter

    ap = np.full(binary.shape, np.nan, dtype=np.float32)
    overlap = np.zeros(binary.shape, dtype=bool)
    yy, xx = np.nonzero(binary)
    points = np.stack((xx, yy), axis=1).astype(np.float64)
    for begin in range(0, len(points), chunk_pixels):
        chunk = points[begin : begin + chunk_pixels]
        distance, t = _segment_distances(chunk, centerline)
        radius = 0.5 * ((1.0 - t) * diameter[None, :-1] + t * diameter[None, 1:])
        coverage = radius - distance
        position = (arc[None, :-1] + t * step[None, :]) / total
        nearest = np.argmax(coverage, axis=1)
        rows = np.arange(len(chunk))
        values = position[rows, nearest]
        # A crossing: some covering segment lies far along the body from the nearest one.
        covering = coverage > 0.0
        far = np.abs(position * total - (values * total)[:, None]) > separation
        crossed = (covering & far).any(axis=1) & covering[rows, nearest]
        y, x = yy[begin : begin + len(chunk)], xx[begin : begin + len(chunk)]
        ap[y, x] = np.where(crossed, np.nan, values).astype(np.float32)
        overlap[y, x] = crossed
    return BodyTargets(
        ap=ap,
        overlap=overlap,
        head_xy=centerline[0].copy(),
        tail_xy=centerline[-1].copy(),
        diameter_px=median_diameter,
    )


def point_heatmap(shape: tuple[int, int], xy: NDArray[np.generic], sigma_px: float) -> NDArray[np.float32]:
    """Unit-peak Gaussian at ``xy``; all zeros when the point is not finite."""

    height, width = shape
    point = np.asarray(xy, dtype=np.float64)
    if not np.all(np.isfinite(point)):
        return np.zeros(shape, dtype=np.float32)
    gx = np.exp(-0.5 * ((np.arange(width) - point[0]) / sigma_px) ** 2)
    gy = np.exp(-0.5 * ((np.arange(height) - point[1]) / sigma_px) ** 2)
    return np.outer(gy, gx).astype(np.float32)


def self_contact(centerline_xy: NDArray[np.generic], width_profile: NDArray[np.generic], tolerance: float = 1.1) -> bool:
    """Whether two parts of the body far apart along it touch or cross.

    Points more than :data:`OVERLAP_ARC_DIAMETERS` median diameters apart in
    arc length touch when their distance is below ``tolerance`` times the sum
    of their radii.
    """

    centerline = np.asarray(centerline_xy, dtype=np.float64)
    radius = 0.5 * np.asarray(width_profile, dtype=np.float64)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))))
    far = np.abs(arc[:, None] - arc[None, :]) > OVERLAP_ARC_DIAMETERS * 2.0 * float(np.median(radius))
    distance = np.linalg.norm(centerline[:, None] - centerline[None, :], axis=-1)
    return bool((far & (distance < tolerance * (radius[:, None] + radius[None, :]))).any())
