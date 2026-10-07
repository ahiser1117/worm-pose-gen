"""Neighbouring frames around a labeled frame, and the difference channels made from them.

A network that sees how the image changes around a frame can separate body
parts that touch but move differently, and tell the leading end from the
trailing one.  The context of a frame is every flat-fielded frame within
``max_lag`` of it; a model then uses a sparse subset of symmetric lags, each
contributing one channel ``frame[t + lag] - frame[t - lag]``.

At 20 Hz, lag 16 is 0.8 s.  A frame outside the recording or unreadable is
replaced by the nearest readable frame (keeping the stack rectangular) and
reported invalid; a lag with an invalid end contributes a zero channel rather
than a fabricated difference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .recordings import RecordingSource
from .segmenter import INPUT_STD


MAX_LAG = 16


def read_context(
    source: RecordingSource, frame_index: int, max_lag: int = MAX_LAG
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Flat-fielded frames ``frame_index - max_lag .. frame_index + max_lag`` and their validity.

    The centre frame must be readable.  Any other position that is outside
    the recording or unreadable repeats the nearest readable frame towards
    the centre and is marked invalid.
    """

    offsets = np.arange(-max_lag, max_lag + 1)
    frames: list[NDArray[np.uint8] | None] = [None] * len(offsets)
    valid = np.zeros(len(offsets), dtype=bool)
    for position, offset in enumerate(offsets):
        index = int(frame_index) + int(offset)
        if not 0 <= index < source.frame_count:
            continue
        try:
            _, frames[position] = source.corrected(index)
        except OSError:
            if offset == 0:
                raise
            continue
        valid[position] = True
    centre = max_lag
    if frames[centre] is None:
        raise IndexError(f"frame {frame_index} is outside {source.name}")
    for direction in (-1, 1):
        previous = frames[centre]
        for position in range(centre + direction, centre + direction * (max_lag + 1), direction):
            if frames[position] is None:
                frames[position] = previous
            previous = frames[position]
    return np.stack(frames), valid


def difference_channels(
    context: NDArray[np.floating], valid: NDArray[np.bool_], lags: Sequence[int]
) -> NDArray[np.float32]:
    """``[len(lags),H,W]`` symmetric differences around the centre of a ``[2L+1,H,W]`` stack.

    A lag with either end invalid is all zeros.  Values are on the scale of
    :func:`segmenter.normalize_frame` (intensity / 255 / ``INPUT_STD``), so a
    difference and the centre frame share units.
    """

    stack = np.asarray(context, dtype=np.float32)
    centre = stack.shape[0] // 2
    if len(valid) != stack.shape[0]:
        raise ValueError("valid must have one entry per context frame")
    if lags and (max(lags) > centre or min(lags) < 1):
        raise ValueError(f"lags must lie in 1..{centre}")
    channels = np.zeros((len(lags), *stack.shape[1:]), dtype=np.float32)
    for k, lag in enumerate(lags):
        if valid[centre + lag] and valid[centre - lag]:
            channels[k] = (stack[centre + lag] - stack[centre - lag]) / (255.0 * INPUT_STD)
    return channels
