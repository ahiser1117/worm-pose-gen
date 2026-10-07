"""Read what a new label stores from its recording: the frame, its context frames and the nose landmarks.

A label is self-contained (:mod:`.labels`), so everything it needs from the
recording is read once, when it is first saved, the way the setup says to
read the video (``video.dataset_path``, ``video.flat_field``).  Context
frames come from :func:`temporal_context.read_context`; a frame outside the
recording or unreadable repeats its neighbour and is marked invalid, and its
nose landmark is invalid too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .labels import MAX_LAG


class _RawFrames:
    """A recording read without flat-fielding, in the interface ``read_context`` expects."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self.name = source.name
        self.frame_count = source.frame_count

    def corrected(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        raw = self.source.read(frame_index)
        return raw, raw


def read_label_inputs(
    path: str | Path, frame: int, *, video: dict[str, Any], flat_field_cache: Path, max_lag: int = MAX_LAG,
) -> dict[str, Any]:
    """The keyword arguments of :meth:`.datasets.Dataset.save` that come from the recording."""

    from ..head_tracking import read_head_tracking
    from ..recordings import RecordingSource
    from ..temporal_context import read_context

    path = Path(path).expanduser().resolve()
    dataset_path = str(video.get("dataset_path") or "/img_nir")
    source = RecordingSource(path, Path(flat_field_cache), dataset=dataset_path)
    try:
        frames = source if video.get("flat_field", True) else _RawFrames(source)
        image_raw, image = frames.corrected(int(frame))
        context, valid = read_context(frames, int(frame), max_lag)
        indices = np.arange(-max_lag, max_lag + 1) + int(frame)
        inside = (indices >= 0) & (indices < source.frame_count)
    finally:
        source.close()
    tracking = read_head_tracking(path, np.clip(indices, 0, source.frame_count - 1), image.shape)
    nose_valid = tracking.valid & inside
    return {
        "image": image, "image_raw": image_raw, "context": context, "context_valid": valid,
        "nose_xy": np.where(nose_valid[:, None], tracking.xy, np.nan).astype(np.float64), "nose_valid": nose_valid,
        "source_path": str(path), "dataset_path": dataset_path,
    }
