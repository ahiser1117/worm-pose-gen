"""Find frames worth labeling: spread over each recording, favouring frames the model is least sure of.

This is how a New queue on the Labeling page is filled (the "Find frames"
job).  The frames to find are shared out evenly between the recordings
(the remainder to the longest ones).  Each recording is cut into as many
equal windows as it gets frames, so the picks cover the whole recording
(the worm's behaviour changes over minutes, and a model trained on one
stretch fails on another).  In every window the model looks at a few
evenly spaced candidate frames, and the one it is least sure of is picked.

**Least sure** is measured on the model's worm-probability map ``p``: the
binary entropy ``H(p) = -p log2 p - (1-p) log2 (1-p)`` summed over the
image and divided by the predicted worm area (pixels with ``p >= 0.5``,
at least :data:`MIN_AREA`).  A confident mask has probabilities near 0 or
1 and an entropy near 0; a frame where the model hesitates (a coil, a
blurred or faint body, a body against the edge, debris) has a wide band of
middling probabilities.  Dividing by the area makes a long and a short worm
comparable: the score is the mean uncertainty per worm pixel.  A frame
without a predicted worm scores its entropy over :data:`MIN_AREA`, so a
worm the model barely sees still ranks high.

Without a model (a setup with no default mask model yet) every candidate
scores the same and the window's middle candidate is picked: the frames
are simply spread out.  Frames already labeled in the dataset being saved
to are never candidates.

``python -m worm_pose_gen.frame_search --spec <json>`` runs the search as a
job and reports the picked frames as its result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray


# Candidate frames looked at in each window.
CANDIDATES_PER_WINDOW = 6
# The smallest worm area (pixels) an uncertainty is divided by.
MIN_AREA = 200.0

Predict = Callable[[Sequence[int]], list[NDArray[np.float32]]]


def uncertainty(probability: NDArray[np.floating]) -> float:
    """Mean binary entropy (bits) per predicted worm pixel; see the module docstring."""

    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(entropy.sum() / max(float((p >= 0.5).sum()), MIN_AREA))


def share(total: int, lengths: Sequence[int]) -> list[int]:
    """``total`` frames shared out evenly between recordings of these lengths, the remainder to the longest; never more than a recording has."""

    counts = [0] * len(lengths)
    order = sorted(range(len(lengths)), key=lambda k: -int(lengths[k]))
    remaining = int(total)
    while remaining > 0 and any(counts[k] < lengths[k] for k in order):
        for k in order:
            if remaining and counts[k] < lengths[k]:
                counts[k] += 1
                remaining -= 1
    return counts


def windows(frame_count: int, count: int) -> list[tuple[int, int]]:
    """``count`` consecutive windows ``[first, end)`` covering ``0..frame_count``."""

    edges = np.linspace(0, frame_count, count + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def candidates(first: int, end: int, excluded: set[int], per_window: int = CANDIDATES_PER_WINDOW) -> list[int]:
    """Evenly spaced frames inside ``[first, end)``, centred in their slots, skipping excluded ones (nearest free frame instead)."""

    free = [f for f in range(first, end) if f not in excluded]
    if not free:
        return []
    slots = min(per_window, len(free))
    picks = []
    for k in range(slots):
        target = first + (end - first) * (k + 0.5) / slots
        best = min(free, key=lambda f: (abs(f - target), f))
        if best not in picks:
            picks.append(best)
    return sorted(picks)


def pick(frame_count: int, count: int, predict: Predict | None, excluded: set[int] = frozenset(),
         per_window: int = CANDIDATES_PER_WINDOW, progress: Callable[[float], None] | None = None) -> list[dict[str, Any]]:
    """The frames picked from one recording: one per window, the least sure candidate (``predict=None``: the middle one).

    Returns ``[{"frame", "uncertainty"}]`` in frame order (``uncertainty``
    is ``None`` without a model).
    """

    spans = windows(frame_count, count)
    groups = [candidates(a, b, set(excluded), per_window) for a, b in spans]
    if predict is None:
        return [{"frame": group[len(group) // 2], "uncertainty": None} for group in groups if group]
    picked = []
    for k, group in enumerate(groups):
        if not group:
            continue
        scores = [uncertainty(p) for p in predict(group)]
        best = int(np.argmax(scores))
        picked.append({"frame": group[best], "uncertainty": round(scores[best], 5)})
        if progress is not None:
            progress((k + 1) / len(groups))
    return picked


def find_frames(
    recordings: Sequence[dict[str, Any]], total: int, open_predict: Callable[[dict[str, Any]], tuple[Predict | None, Callable[[], None]]],
    *, per_window: int = CANDIDATES_PER_WINDOW, progress: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Frames from several recordings (``[{"path", "id", "frames", "exclude": [...]}]``), ``total`` in all.

    ``open_predict(recording)`` gives a prediction function over its frame
    indices (or ``None`` without a model) and a function closing what it
    opened.  Returns ``[{"path", "recording", "frame", "uncertainty"}]``,
    recording by recording.
    """

    counts = share(total, [int(r["frames"]) - len(r.get("exclude") or ()) for r in recordings])
    entries: list[dict[str, Any]] = []
    for k, (recording, count) in enumerate(zip(recordings, counts)):
        if not count:
            continue
        name = recording["id"]
        report = None if progress is None else (lambda fraction, k=k, name=name: progress((k + fraction) / len(recordings), f"looking at {name}"))
        if report is not None:
            report(0.0)
        predict, close = open_predict(recording)
        try:
            picked = pick(int(recording["frames"]), count, predict, set(recording.get("exclude") or ()), per_window, report)
        finally:
            close()
        entries.extend({"path": recording["path"], "recording": name, **p} for p in picked)
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    """The job: ``--spec`` is ``{"recordings": [{"path", "id", "frames", "exclude"}], "frames": N, "model": ref | null,
    "libraries": {"lab", "personal"}, "video": {"dataset_path", "flat_field"}, "dataset_root": path}``."""

    from .jobs import report_progress
    from .library import Libraries, load_model
    from .pipeline import Frames

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec)
    model = None
    if spec.get("model"):
        report_progress(0.0, f"loading {spec['model']}")
        roots = spec["libraries"]
        libraries = Libraries(lab=None if roots.get("lab") is None else Path(roots["lab"]), personal=Path(roots["personal"]))
        model = load_model(libraries, spec["model"], args.device)
    video = spec.get("video") or {}

    def open_predict(recording: dict[str, Any]) -> tuple[Predict | None, Callable[[], None]]:
        if model is None:
            return None, lambda: None
        frames = Frames(Path(recording["path"]), dataset_root=spec.get("dataset_root"), flat_field=bool(video.get("flat_field", True)),
                        dataset=str(video.get("dataset_path") or "/img_nir"))
        handle = lambda: frames._handle.close() if frames._handle is not None else None  # noqa: E731
        return (lambda indices: model.recording_probabilities(frames, indices)), handle

    entries = find_frames(spec["recordings"], int(spec["frames"]), open_predict,
                          progress=lambda fraction, message: report_progress(0.02 + 0.97 * fraction, message))
    report_progress(1.0, f"found {len(entries)} frames", result={"entries": entries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
