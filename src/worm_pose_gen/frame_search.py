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

**Types.**  Each candidate is also sorted into the kinds of image it shows
(:func:`frame_types`, from the same prediction), any of:

- ``contact``: the body touches or crosses itself: the model's overlap
  output marks at least :data:`MIN_TYPE_PX` worm pixels (a body-field
  model), or the predicted worm encloses a hole of that size (a loop; the
  only sign a plain segmenter gives);
- ``edge``: the predicted worm reaches the image border (a body partly off
  camera);
- ``pieces``: the predicted worm is in two or more pieces of at least
  :data:`MIN_TYPE_PX` pixels (debris, a second animal, a broken mask);
- ``empty``: less than :data:`MIN_AREA` predicted worm pixels;
- ``clear``: none of the above: one whole body in view, apart from itself.

A search can be limited to some types: then only candidates of any of them
are picked, each window offers :data:`FILTERED_CANDIDATES_PER_WINDOW`
candidates (rarer types need more looks), and a window with no candidate
of those types gives its frame to the least sure leftover matches of the
recording's other windows.  A recording with fewer matches than its share
gives fewer frames.  Every picked frame records its ``types``, so a queue
can be filtered by them afterwards.

Without a model (a setup with no default mask model yet) every candidate
scores the same and the window's middle candidate is picked: the frames
are simply spread out, and there are no types to limit them to.  Frames
already labeled in the dataset being saved to are never candidates.

``python -m worm_pose_gen.frame_search --spec <json>`` runs the search as a
job and reports the picked frames as its result.  With a model, the job
then makes sure each recording has a body-size estimate in the pipeline's
prior cache (:func:`pipeline.resolve_prior`: a bootstrap over frames spread
through the recording, as Analyse's ``prior`` stage does), so the Labeling
page can extend a trace off camera on a recording nobody has labeled or
analysed yet; the result's ``lengths`` gives each recording's (``null``
when no whole body was found).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .library.inference import Outputs


# Candidate frames looked at in each window.
CANDIDATES_PER_WINDOW = 6
# ... when the search is limited to some types.
FILTERED_CANDIDATES_PER_WINDOW = 18
# The smallest worm area (pixels) an uncertainty is divided by; less predicted worm is an ``empty`` frame.
MIN_AREA = 200.0
# The fewest pixels of overlap, of an enclosed hole, or of a piece that make a type.
MIN_TYPE_PX = 30
TYPES = ("contact", "edge", "pieces", "empty", "clear")

Predict = Callable[[Sequence[int]], list[Outputs]]


def uncertainty(probability: NDArray[np.floating]) -> float:
    """Mean binary entropy (bits) per predicted worm pixel; see the module docstring."""

    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(entropy.sum() / max(float((p >= 0.5).sum()), MIN_AREA))


def frame_types(outputs: Outputs) -> list[str]:
    """The kinds of image a prediction shows, in :data:`TYPES` order; see the module docstring."""

    worm = np.asarray(outputs.mask) >= 0.5
    if worm.sum() < MIN_AREA:
        return ["empty"]
    found = []
    overlap = 0 if outputs.overlap is None else int(((np.asarray(outputs.overlap) >= 0.5) & worm).sum())
    hole = int((ndimage.binary_fill_holes(worm) & ~worm).sum())
    if max(overlap, hole) >= MIN_TYPE_PX:
        found.append("contact")
    if worm[0].any() or worm[-1].any() or worm[:, 0].any() or worm[:, -1].any():
        found.append("edge")
    labels, count = ndimage.label(worm, structure=np.ones((3, 3), bool))
    if count > 1 and int((np.bincount(labels.ravel())[1:] >= MIN_TYPE_PX).sum()) > 1:
        found.append("pieces")
    return found or ["clear"]


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
         per_window: int = CANDIDATES_PER_WINDOW, progress: Callable[[float], None] | None = None,
         types: Sequence[str] = ()) -> list[dict[str, Any]]:
    """The frames picked from one recording: one per window, the least sure candidate (``predict=None``: the middle one).

    With ``types``, only candidates of any of them; a window without one
    gives its frame to the least sure leftover matches of the other windows.
    Returns ``[{"frame", "uncertainty", "types"}]`` in frame order
    (``uncertainty`` and ``types`` are ``None`` without a model).
    """

    spans = windows(frame_count, count)
    groups = [candidates(a, b, set(excluded), per_window) for a, b in spans]
    if predict is None:
        return [{"frame": group[len(group) // 2], "uncertainty": None, "types": None} for group in groups if group]
    wanted = set(types)
    picked, leftover = [], []
    for k, group in enumerate(groups):
        if group:
            scored = [{"frame": frame, "uncertainty": round(uncertainty(out.mask), 5), "types": frame_types(out)}
                      for frame, out in zip(group, predict(group))]
            matches = sorted((s for s in scored if not wanted or wanted & set(s["types"])), key=lambda s: -s["uncertainty"])
            picked.extend(matches[:1])
            leftover.extend(matches[1:])
        if progress is not None:
            progress((k + 1) / len(groups))
    leftover.sort(key=lambda s: -s["uncertainty"])
    picked.extend(leftover[: max(0, count - len(picked))])
    return sorted(picked, key=lambda s: s["frame"])


def find_frames(
    recordings: Sequence[dict[str, Any]], total: int, open_predict: Callable[[dict[str, Any]], tuple[Predict | None, Callable[[], None]]],
    *, types: Sequence[str] = (), per_window: int | None = None, progress: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Frames from several recordings (``[{"path", "id", "frames", "exclude": [...]}]``), ``total`` in all, of any of ``types`` (all by default).

    ``open_predict(recording)`` gives a prediction function over its frame
    indices (or ``None`` without a model) and a function closing what it
    opened.  Returns ``[{"path", "recording", "frame", "uncertainty", "types"}]``,
    recording by recording.
    """

    unknown = sorted(set(types) - set(TYPES))
    if unknown:
        raise ValueError(f"unknown frame types {unknown}; choose from {list(TYPES)}")
    if per_window is None:
        per_window = FILTERED_CANDIDATES_PER_WINDOW if types else CANDIDATES_PER_WINDOW

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
            if types and predict is None:
                raise ValueError("limiting the search to frame types needs a model")
            picked = pick(int(recording["frames"]), count, predict, set(recording.get("exclude") or ()), per_window, report, types)
        finally:
            close()
        entries.extend({"path": recording["path"], "recording": name, **p} for p in picked)
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    """The job: ``--spec`` is ``{"recordings": [{"path", "id", "frames", "exclude"}], "frames": N, "types": [type, ...] (optional), "model": ref | null,
    "libraries": {"lab", "personal"}, "video": {"dataset_path", "flat_field"}, "fps": setup fps, "dataset_root": path,
    "prior_cache": path (optional; the pipeline's default cache)}``."""

    from . import library
    from .jobs import report_progress
    from .library import Libraries
    from .library.inference import load_model
    from .pipeline import Frames

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec)
    model = libraries = None
    if spec.get("model"):
        report_progress(0.0, f"loading {spec['model']}")
        roots = spec["libraries"]
        libraries = Libraries(lab=None if roots.get("lab") is None else Path(roots["lab"]), personal=Path(roots["personal"]))
        model = load_model(libraries, spec["model"], device=args.device, fps=spec.get("fps"))
    video = spec.get("video") or {}

    def open_predict(recording: dict[str, Any]) -> tuple[Predict | None, Callable[[], None]]:
        if model is None:
            return None, lambda: None
        frames = Frames(Path(recording["path"]), dataset_root=spec.get("dataset_root"), flat_field=bool(video.get("flat_field", True)),
                        dataset=str(video.get("dataset_path") or "/img_nir"))

        def predict(indices: Sequence[int]) -> list[Outputs]:
            # Each frame with the neighbours its lags need, read as one slab.
            lag = model.max_lag
            out = []
            for index in indices:
                first, last = max(0, index - lag), min(frames.total - 1, index + lag)
                stack, _, _ = frames.corrected(list(range(first, last + 1)))
                out.append(model.predict_sequence(stack, None, [index - first])[0])
            return out

        return predict, frames.close

    entries = find_frames(spec["recordings"], int(spec["frames"]), open_predict, types=spec.get("types") or (),
                          progress=lambda fraction, message: report_progress(0.02 + (0.67 if model else 0.97) * fraction, message))
    lengths = {}
    if model is not None:
        lengths = estimate_lengths(spec, str(library.weights_path(libraries, spec["model"])), args.device,
                                   progress=lambda fraction, message: report_progress(0.7 + 0.29 * fraction, message))
    report_progress(1.0, f"found {len(entries)} frames", result={"entries": entries, "lengths": lengths})
    return 0


def estimate_lengths(
    spec: dict[str, Any], checkpoint: str, device: str | None, *, progress: Callable[[float, str], None] | None = None,
) -> dict[str, float | None]:
    """Each recording's body length from the prior cache, bootstrapping the ones it lacks with the segmenter ``checkpoint``."""

    import torch

    from .pipeline import FitParams, Frames, PriorParams, build_fit_config, cached_prior, resolve_prior

    video = spec.get("video") or {}
    params = PriorParams(checkpoint=checkpoint, dataset_root=str(spec.get("dataset_root")), flat_field=bool(video.get("flat_field", True)))
    if spec.get("prior_cache"):
        params = replace(params, prior_cache=str(spec["prior_cache"]))
    config = build_fit_config(FitParams())
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lengths: dict[str, float | None] = {}
    for k, recording in enumerate(spec["recordings"]):
        if progress is not None:
            progress(k / len(spec["recordings"]), f"estimating the body length of {recording['id']}")
        prior = cached_prior(recording["path"], params.prior_cache)
        if prior is None:
            frames = Frames(Path(recording["path"]), dataset_root=params.dataset_root, flat_field=params.flat_field,
                            dataset=str(video.get("dataset_path") or "/img_nir"))
            try:
                prior, _, _ = resolve_prior(frames, params, config, resolved)
            except ValueError:  # no whole body in the sampled frames
                prior = None
            finally:
                frames.close()
        lengths[recording["id"]] = None if prior is None else prior.length_px
    return lengths


if __name__ == "__main__":
    raise SystemExit(main())
