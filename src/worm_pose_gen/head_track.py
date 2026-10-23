"""The head of a recording chosen by tracking the head heatmap's modes over time.

Each frame's head used to be the head heatmap's maximum
(``body_proposal.FieldPrediction.peak``), and the fit follows it closely
(point 0 lies a median 0.8 px from it on 2023-03-30-01).  Inside a coil the
heatmap is bimodal and its maximum jumps between the modes (frames 4720 and
4725 of 2023-03-30-01: about 95 px to the inner turn and back), and where
the head lies against the body the maximum falls below ``END_THRESHOLD`` for
tens of frames.  Here each frame offers candidates instead (the heatmap's
strongest modes, ``body_proposal.heatmap_modes``, and the acquisition's nose
landmark where it is confident, :mod:`head_tracking`) and a Viterbi decode
over the recording's rows picks one per frame, or none (the head not in
view), at the lowest total of:

- *emission*: ``-log(value)`` for a heatmap mode, ``NOSE_COST`` for the nose,
  ``HIDDEN_COST`` per frame without a head;
- *transition*: ``1.5 * log(1 + (d / s)^2)``, the negative log of a 2-D
  Cauchy density of the head's displacement ``d`` between the two frames
  with scale ``s = STEP_PX`` times their frame distance (the fit's head on
  2023-03-30-01 moves a median 3.9 px per frame, 99th percentile 13.9 px);
- *a gap*: frames without a head between two heads cost ``GAP_COST`` once
  plus ``HIDDEN_COST`` each, and the transition across the gap is scaled by
  its frame distance, up to ``MAX_GAP_ROWS`` rows; a longer gap forgets the
  position and costs ``LOST_COST`` more.

Rows are the recording's fitted frames in order; ``frame_index`` gives their
distance, so missing frames (a step, empty masks) stretch the scale.  An
optional per-frame ``shift_px`` moves the candidates from image to substrate
coordinates before displacements are measured, so a stage move is not a
head jump.

A chosen mode below ``INTERPOLATE_BELOW`` (``body_proposal.END_THRESHOLD``)
only says the head is there about; its position is the straight line, in
frame index and substrate coordinates, between the nearest rows before and
after with a head at least that strong.  A weak head with no such row on
either side before the head leaves view is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Measured on 2023-03-30-01 (docs/BODY_FIELDS.md, *Head track*).
STEP_PX = 8.0
# A row without a head costs what a head of value HIDDEN_VALUE moving STEP_PX
# from the frame before costs: a row with a head pays a transition too.
HIDDEN_VALUE = 0.3
HIDDEN_COST = float(-np.log(HIDDEN_VALUE) + 1.5 * np.log(2.0))
NOSE_COST = float(-np.log(0.5))
GAP_COST = 1.0
LOST_COST = 6.0
MAX_GAP_ROWS = 100
# A head mode this close to a stronger tail peak is the tail (inside a coil the
# head heatmap fires weakly on the tail: 2023-03-30-01 frames 4790-4817).
TAIL_EXCLUSION_PX = 22.0
VALUE_FLOOR = 1e-3
INTERPOLATE_BELOW = 0.3


@dataclass(frozen=True)
class HeadTrack:
    """Per row: the chosen head ``xy`` (NaN without one), its candidate (``-1`` none, ``0..K-1`` a mode, ``K`` the nose), its ``value``, and whether ``xy`` was ``interpolated``."""

    xy: NDArray[np.float64]
    candidate: NDArray[np.int64]
    value: NDArray[np.float64]
    interpolated: NDArray[np.bool_]


def _transition(d: NDArray[np.float64], frames: NDArray[np.float64]) -> NDArray[np.float64]:
    scale = STEP_PX * np.maximum(frames, 1.0)
    cost = 1.5 * np.log1p((d / scale) ** 2)
    return np.where(np.isfinite(cost), cost, np.inf)


def candidates(
    modes: NDArray[np.float64],
    tail: NDArray[np.float64] | None = None,
    nose_xy: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``[n, C, 2]`` positions, ``[n, C]`` values and ``[n, C]`` emission costs (inf: no candidate) of the ``[n, K, 3]`` modes and the nose.

    ``tail`` (``[n, 3]``, the tail heatmap's peaks) removes a mode within
    ``TAIL_EXCLUSION_PX`` of a stronger tail peak; ``nose_xy`` (``[n, 2]``,
    NaN where the acquisition was not confident) adds one candidate per row.
    """

    modes = np.asarray(modes, dtype=np.float64)
    xy, value = modes[..., :2].copy(), modes[..., 2].copy()
    present = np.isfinite(value) & np.isfinite(xy).all(-1)
    if tail is not None:
        tail = np.asarray(tail, dtype=np.float64)
        near = np.linalg.norm(xy - tail[:, None, :2], axis=-1) <= TAIL_EXCLUSION_PX
        present &= ~(near & (tail[:, None, 2] > value))
    cost = np.where(present, -np.log(np.clip(np.where(present, value, 1.0), VALUE_FLOOR, 1.0)), np.inf)
    if nose_xy is not None:
        nose = np.asarray(nose_xy, dtype=np.float64)
        valid = np.isfinite(nose).all(-1)
        xy = np.concatenate((xy, nose[:, None]), 1)
        value = np.concatenate((value, np.where(valid, 1.0, np.nan)[:, None]), 1)
        cost = np.concatenate((cost, np.where(valid, NOSE_COST, np.inf)[:, None]), 1)
    xy[~np.isfinite(cost)] = np.nan
    return xy, value, cost


def track_heads(
    frame_index: NDArray[np.integer],
    modes: NDArray[np.float64],
    *,
    tail: NDArray[np.float64] | None = None,
    nose_xy: NDArray[np.float64] | None = None,
    shift_px: NDArray[np.float64] | None = None,
) -> HeadTrack:
    """The lowest-cost head per row of a recording (see the module's docstring).

    ``frame_index`` (``[n]``, increasing) are the rows' frames, ``modes``
    their head modes ``[n, K, 3]`` (x, y, value; NaN rows past the last
    mode), ``tail`` and ``nose_xy`` as :func:`candidates` takes them, and
    ``shift_px`` (``[n, 2]``) the image-to-substrate offset added to every
    candidate before displacements are measured.
    """

    frames = np.asarray(frame_index, dtype=np.float64)
    n = len(frames)
    xy, value, emission = candidates(modes, tail, nose_xy)
    k = emission.shape[1]
    position = xy if shift_px is None else xy + np.asarray(shift_px, dtype=np.float64)[:, None, :]
    # visible[t, c]: the best total with candidate c the head of row t; it came
    # from (row, candidate) before: row -1 is the start, candidate -1 the lost state.
    visible = np.full((n, k), np.inf)
    from_row = np.full((n, k), -1, dtype=np.int64)
    from_candidate = np.full((n, k), -1, dtype=np.int64)
    # lost[t]: no head in row t after one was seen, the position forgotten; it
    # came from candidate lost_from[t] of row t - 1 (-1: lost there too).
    lost = np.full(n, np.inf)
    lost_from = np.full(n, -1, dtype=np.int64)
    for t in range(n):
        unseen = t * HIDDEN_COST  # no head in any row before
        total = np.full(k, unseen)
        rows = np.full(k, -1, dtype=np.int64)
        cands = np.full(k, -1, dtype=np.int64)
        if t > 0 and lost[t - 1] + GAP_COST + LOST_COST < unseen:
            total[:] = lost[t - 1] + GAP_COST + LOST_COST
            rows[:] = t - 1
        lo = max(0, t - MAX_GAP_ROWS)
        if t > lo:
            gap = t - 1 - np.arange(lo, t)  # rows without a head in between
            d = np.linalg.norm(position[lo:t, :, None, :] - position[t, None, None, :, :], axis=-1)  # [rows, C before, C]
            cost = visible[lo:t, :, None] + _transition(d, (frames[t] - frames[lo:t])[:, None, None])
            cost = (cost + (gap * HIDDEN_COST + (gap > 0) * GAP_COST)[:, None, None]).reshape(-1, k)
            pick = np.argmin(cost, 0)
            chosen = cost[pick, np.arange(k)]
            better = chosen < total
            total = np.where(better, chosen, total)
            rows = np.where(better, lo + pick // k, rows)
            cands = np.where(better, pick % k, cands)
        visible[t], from_row[t], from_candidate[t] = total + emission[t], rows, cands
        if t > 0:
            leave = int(np.argmin(visible[t - 1]))
            if visible[t - 1, leave] <= lost[t - 1]:
                lost[t], lost_from[t] = visible[t - 1, leave] + HIDDEN_COST, leave
            else:
                lost[t] = lost[t - 1] + HIDDEN_COST
    out = HeadTrack(np.full((n, 2), np.nan), np.full(n, -1, dtype=np.int64), np.full(n, np.nan), np.zeros(n, dtype=bool))
    if not n:
        return out
    # Back from the cheapest end: a head in the last row, the head lost, or never seen.
    c = int(np.argmin(visible[-1]))
    ends = (visible[-1, c], lost[-1], n * HIDDEN_COST)
    t, c = (n - 1, c) if ends[0] <= min(ends[1:]) else (n - 1, -1) if ends[1] <= ends[2] else (-1, -1)
    while t >= 0:
        if c < 0:  # lost in row t
            t, c = t - 1, int(lost_from[t])
            continue
        out.xy[t], out.candidate[t], out.value[t] = xy[t, c], c, value[t, c]
        t, c = int(from_row[t, c]), int(from_candidate[t, c])
    _interpolate_weak(out, frames, None if shift_px is None else np.asarray(shift_px, dtype=np.float64))
    return out


def _interpolate_weak(track: HeadTrack, frames: NDArray[np.float64], shift_px: NDArray[np.float64] | None) -> None:
    """Replace each weak head of ``track`` in place (see the module's docstring)."""

    shown = np.isfinite(track.value)
    strong = shown & (track.value >= INTERPOLATE_BELOW)
    weak = np.flatnonzero(shown & ~strong)
    if not len(weak):
        return
    shift = np.zeros_like(track.xy) if shift_px is None else shift_px
    rows = np.arange(len(frames))
    # The nearest strong row before and after each row, and the last and next row without a head.
    before = np.maximum.accumulate(np.where(strong, rows, -1))
    after = np.minimum.accumulate(np.where(strong, rows, len(rows))[::-1])[::-1]
    hidden_before = np.maximum.accumulate(np.where(shown, -1, rows))
    hidden_after = np.minimum.accumulate(np.where(shown, len(rows), rows)[::-1])[::-1]
    for t in weak:
        a, b = before[t], after[t]
        if a < 0 or b >= len(rows) or hidden_before[t] > a or hidden_after[t] < b:
            track.xy[t], track.candidate[t], track.value[t] = np.nan, -1, np.nan
            continue
        w = (frames[t] - frames[a]) / (frames[b] - frames[a])
        substrate = (1 - w) * (track.xy[a] + shift[a]) + w * (track.xy[b] + shift[b])
        track.xy[t] = substrate - shift[t]
        track.interpolated[t] = True
