"""Exact 8-connected component labels in foreground raster order."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int32]


def label_components(mask: NDArray[np.generic]) -> tuple[IntArray, int]:
    """Label 8-connected foreground components ``1..count``; background is ``0``."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    # SciPy assigns dense labels in first-pixel raster order, preserving the
    # original run-union implementation's IDs and largest-component tie rule.
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=bool), output=np.int32)
    return labels, int(count)


def component_areas(labels: IntArray, count: int) -> NDArray[np.int64]:
    """Pixel count per label, index ``0`` being background."""

    return np.bincount(np.asarray(labels).ravel(), minlength=count + 1).astype(np.int64)


def largest_component(mask: NDArray[np.generic]) -> tuple[BoolArray, int, int]:
    """Largest 8-connected component, its area, and the component count.

    Same contract as ``worm_pose_gen.classical._largest_component``.  Ties go
    to the lowest label, which is the component containing the first
    foreground pixel in row-major order.
    """

    labels, count = label_components(mask)
    if count == 0:
        return np.zeros(labels.shape, dtype=bool), 0, 0
    areas = component_areas(labels, count)
    best = int(np.argmax(areas[1:])) + 1
    return labels == best, int(areas[best]), count
