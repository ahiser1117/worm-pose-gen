"""Exact 8-connected components for binary masks, vectorized in NumPy.

``worm_pose_gen.classical._largest_component`` walks every foreground pixel in
a Python breadth-first search and costs 60--90 ms on a 732 x 968 frame.  This
module labels horizontal runs instead, unions runs that touch between
neighboring rows, and paints labels back with a cumulative sum.  Only the
union-find loop over touching run pairs is Python, and a worm mask has a few
thousand of those rather than tens of thousands of pixels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int32]


def label_components(mask: NDArray[np.generic]) -> tuple[IntArray, int]:
    """Label 8-connected foreground components ``1..count``; background is ``0``."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = binary.shape
    if height == 0 or width == 0 or not binary.any():
        return np.zeros(binary.shape, dtype=np.int32), 0

    # A sentinel background column terminates every run inside its own row.
    stride = width + 1
    padded = np.zeros((height, stride), dtype=bool)
    padded[:, :width] = binary
    flat = padded.ravel()
    delta = np.diff(flat.astype(np.int8), prepend=np.int8(0))
    starts = np.flatnonzero(delta == 1)
    ends = np.flatnonzero(delta == -1)  # exclusive
    rows = starts // stride
    col_start = starts - rows * stride
    col_end = ends - rows * stride
    n_runs = len(starts)

    # Runs of row r get keys in [r*span, r*span + width]; both key arrays are
    # sorted, so the runs of row r+1 touching run i (8-connectivity:
    # col_start_j <= col_end_i and col_end_j >= col_start_i) form the index
    # range [lo_i, hi_i).
    span = width + 2
    key_start = rows * span + col_start
    key_end = rows * span + col_end
    next_row = (rows + 1) * span
    lo = np.searchsorted(key_end, next_row + col_start, side="left")
    hi = np.searchsorted(key_start, next_row + col_end, side="right")
    counts = np.maximum(hi - lo, 0)
    total = int(counts.sum())
    parent = np.arange(n_runs)
    if total:
        first = np.repeat(np.arange(n_runs), counts)
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        second = np.repeat(lo, counts) + offsets
        _union_pairs(parent, first, second)
    roots = _roots(parent)
    _, run_labels = np.unique(roots, return_inverse=True)
    run_labels = run_labels.astype(np.int32).ravel() + 1
    count = int(run_labels.max())

    # +label at each run start and -label at its exclusive end; the running
    # sum is the label inside runs and zero elsewhere.
    steps = np.zeros(height * stride, dtype=np.int32)
    steps[starts] = run_labels
    steps[ends] -= run_labels
    labels = np.cumsum(steps, dtype=np.int32).reshape(height, stride)[:, :width]
    return np.ascontiguousarray(labels), count


def _union_pairs(parent: NDArray[np.int64], first: NDArray[np.int64], second: NDArray[np.int64]) -> None:
    tree = parent.tolist()
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        while tree[a] != a:
            tree[a] = tree[tree[a]]
            a = tree[a]
        while tree[b] != b:
            tree[b] = tree[tree[b]]
            b = tree[b]
        if a != b:
            if a < b:
                tree[b] = a
            else:
                tree[a] = b
    parent[:] = tree


def _roots(parent: NDArray[np.int64]) -> NDArray[np.int64]:
    roots = parent.copy()
    while True:
        grand = roots[roots]
        if np.array_equal(grand, roots):
            return roots
        roots = grand


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
