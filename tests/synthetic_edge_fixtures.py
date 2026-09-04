"""Deterministic curved-tube fixtures for camera-boundary censoring tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SyntheticEdgeCase:
    name: str
    mask: BoolArray
    visible_truth_xy: FloatArray
    expected_sides: frozenset[str]


def _render_tube(points_xy: FloatArray, shape: tuple[int, int], radius: float) -> BoolArray:
    """Rasterize a dense polyline tube, including samples outside ``shape``."""

    yy, xx = np.mgrid[: shape[0], : shape[1]]
    distance_sq = np.full(shape, np.inf, dtype=np.float64)
    # Chunking keeps the temporary [H,W,N] array small enough for unit tests.
    for start in range(0, len(points_xy), 128):
        points = points_xy[start : start + 128]
        candidate = (xx[..., None] - points[:, 0]) ** 2 + (
            yy[..., None] - points[:, 1]
        ) ** 2
        distance_sq = np.minimum(distance_sq, candidate.min(axis=2))
    return distance_sq <= radius**2


def _visible(points_xy: FloatArray, shape: tuple[int, int]) -> FloatArray:
    height, width = shape
    keep = (
        (points_xy[:, 0] >= 0.0)
        & (points_xy[:, 0] <= width - 1.0)
        & (points_xy[:, 1] >= 0.0)
        & (points_xy[:, 1] <= height - 1.0)
    )
    return points_xy[keep]


def make_synthetic_edge_cases(
    *, shape: tuple[int, int] = (96, 96), radius: float = 7.0
) -> tuple[SyntheticEdgeCase, ...]:
    """Return left/right/top/bottom and corner-censored curved worms.

    Each source centerline begins outside the camera and ends well inside it.
    Transforms of one canonical curve make the four single-edge cases exactly
    comparable, while the corner case crosses the top-left corner obliquely.
    """

    height, width = shape
    if height != width:
        raise ValueError("the rotationally symmetric fixture requires a square image")
    coordinate = np.linspace(-28.0, 78.0, 1600)
    canonical = np.column_stack(
        (
            coordinate,
            48.0 + 19.0 * np.sin((coordinate + 18.0) / 30.0),
        )
    )
    last_x = float(width - 1)
    transforms = {
        "left": lambda points: points,
        "right": lambda points: np.column_stack((last_x - points[:, 0], points[:, 1])),
        "top": lambda points: points[:, ::-1],
        "bottom": lambda points: np.column_stack((points[:, 1], last_x - points[:, 0])),
    }
    output: list[SyntheticEdgeCase] = []
    for name, transform in transforms.items():
        points = transform(canonical)
        output.append(
            SyntheticEdgeCase(
                name,
                _render_tube(points, shape, radius),
                _visible(points, shape),
                frozenset((name,)),
            )
        )

    corner_coordinate = np.linspace(-28.0, 68.0, 1800)
    corner = np.column_stack(
        (
            corner_coordinate,
            0.62 * corner_coordinate
            + 1.0
            + 2.2 * np.sin((corner_coordinate + 12.0) / 19.0),
        )
    )
    output.append(
        SyntheticEdgeCase(
            "top_left_corner",
            _render_tube(corner, shape, radius),
            _visible(corner, shape),
            frozenset(("top", "left")),
        )
    )
    return tuple(output)


def nearest_curve_distance(points_xy: FloatArray, curve_xy: FloatArray) -> FloatArray:
    """Nearest-sample distance for densely sampled synthetic ground truth."""

    points = np.asarray(points_xy, dtype=np.float64)
    curve = np.asarray(curve_xy, dtype=np.float64)
    return np.sqrt(((points[:, None] - curve[None, :]) ** 2).sum(axis=2)).min(axis=1)


def edge_region(points_xy: FloatArray, shape: tuple[int, int], band_px: float) -> BoolArray:
    height, width = shape
    points = np.asarray(points_xy, dtype=np.float64)
    clearance = np.minimum.reduce(
        (
            points[:, 0],
            points[:, 1],
            width - 1.0 - points[:, 0],
            height - 1.0 - points[:, 1],
        )
    )
    return clearance < band_px
