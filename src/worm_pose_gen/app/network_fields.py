"""Body-field network predictions on workspace frames, for the Workspace page's A-P layer and the developer's crossings.

The network is the body model the workspace was analysed with
(:func:`analysis.workspace_body_net`, loaded once by ``AppState.body_net``);
a workspace analysed without one has no network layers.  A frame is predicted from its flat-fielded recording frame and
lag neighbours exactly as the fit stage reads them (:func:`pipeline.workspace_frames`
with the app's flat-field cache, :class:`body_proposal.RecordingFieldPredictor`);
a neighbour outside the recording gives a zero lag channel.  The response
encodes the A-P field as the library API does (``0`` undefined, else
``1 + round(254 * ap)``), defined where the predicted mask is above 0.5 and
the pixel is not a predicted crossing; the overlap pixels as 0/255; and the
head and tail as the fitter would score them (:func:`body_proposal.field_evidence`
against the workspace's current mask for the row, or the predicted mask when
the row has none): ``None`` when the heatmap peak is below ``END_THRESHOLD``,
farther than ``END_MASK_PX`` from the mask, or within ``END_BORDER_PX`` of the
image edge.  Inference runs one frame at a time; results are kept in a
bounded LRU keyed by workspace, row, mask revision and model.
"""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any

import numpy as np

from .images import data_url
from ..pipeline import SegmentParams, workspace_dataset, workspace_frames

# Cached predictions; each holds two full-frame PNGs (a few hundred kB).
CACHE_ENTRIES = 64
# Recordings kept open for prediction (flat field loaded once each).
OPEN_RECORDINGS = 4


def encode_prediction(prediction: Any, mask: np.ndarray | None) -> dict[str, Any]:
    """The page's layers of one ``FieldPrediction``; ``mask`` is the workspace's current mask of the row, if any."""

    from ..body_proposal import END_THRESHOLD, OVERLAP_THRESHOLD, field_evidence

    body = prediction.mask > 0.5
    crossing = body & (prediction.overlap >= OVERLAP_THRESHOLD)
    defined = body & ~crossing
    ap = np.where(defined, 1 + np.round(254 * np.clip(prediction.ap, 0, 1)), 0).astype(np.uint8)
    reference = mask if mask is not None and np.any(mask) else body
    evidence = field_evidence(prediction, reference)
    point = lambda xy: None if xy is None else [float(xy[0]), float(xy[1])]
    return {
        "width": int(body.shape[1]), "height": int(body.shape[0]),
        "ap": data_url(ap), "overlap": data_url(crossing.astype(np.uint8) * 255),
        "head_xy": point(evidence.head_xy), "tail_xy": point(evidence.tail_xy),
        "head_peak": float(prediction.head.max()), "tail_peak": float(prediction.tail.max()),
        "end_threshold": END_THRESHOLD, "mask_source": "workspace" if reference is mask else "predicted",
    }


class NetworkFields:
    """Predictions of workspace frames: the LRU of encoded results and the recordings they are read from."""

    def __init__(self, app: Any, *, entries: int = CACHE_ENTRIES) -> None:
        self.app = app
        self.entries = entries
        self._cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self._frames: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._lock = threading.Lock()  # the cache and the open recordings
        self._inference = threading.Lock()  # one prediction at a time

    def _cached(self, key: tuple) -> dict[str, Any] | None:
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
            return hit

    def _frames_of(self, workspace: Any) -> Any:
        key = (str(workspace.recording), workspace_dataset(workspace))
        with self._lock:
            frames = self._frames.get(key)
            if frames is not None:
                self._frames.move_to_end(key)
                return frames
        try:
            frames = workspace_frames(workspace, SegmentParams(dataset_root=str(self.app.config.dataset_root)))
        except (OSError, KeyError) as error:
            raise ValueError(f"recording not readable: {error}") from error
        with self._lock:
            self._frames[key] = frames
            while len(self._frames) > OPEN_RECORDINGS:
                self._frames.popitem(last=False)[1].close()
        return frames

    def frame(self, name: str, frame: int) -> dict[str, Any]:
        """The encoded prediction of workspace ``name``'s ``frame`` (``cached`` says whether it was computed before)."""

        from ..body_proposal import RecordingFieldPredictor

        view = self.app.view(name)
        view.refresh()
        workspace = view.workspace
        row = workspace.row_of(frame)
        from .analysis import workspace_body_net

        path = workspace_body_net(workspace)
        if path is None:
            raise ValueError(f"workspace {name} was analysed without a body-field model")
        module = self.app.body_net(path)
        model = str(getattr(module, "checkpoint_path", ""))
        key = (name, str(workspace.recording), row, workspace.mask_revision(row), model)
        hit = self._cached(key)
        if hit is None:
            with self._inference:
                hit = self._cached(key)
                if hit is None:
                    prediction = RecordingFieldPredictor(module, self._frames_of(workspace)).predict([int(frame)])[0]
                    hit = {**encode_prediction(prediction, workspace.effective_mask(row)), "model": model, "lags": list(module.lags)}
                    with self._lock:
                        self._cache[key] = hit
                        while len(self._cache) > self.entries:
                            self._cache.popitem(last=False)
                    return {**hit, "frame": int(frame), "row": row, "cached": False}
        return {**hit, "frame": int(frame), "row": row, "cached": True}

    def close(self) -> None:
        with self._lock:
            for frames in self._frames.values():
                frames.close()
            self._frames.clear()
            self._cache.clear()
