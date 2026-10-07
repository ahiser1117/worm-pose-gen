"""Running library models: load a model card's weights, and ask it for a mask or the body fields.

A card's ``kind`` says how its weights load (:func:`segmenter.load_segmenter`
or :func:`body_net.load_body_net`).  Both kinds give a worm mask: a
segmenter from the frame alone, a body-field net from the frame and the
differences to its neighbours at the card's lags.  :class:`LoadedModel`
hides that difference behind three questions the app and the jobs ask:

- :meth:`LoadedModel.mask_probability`: the mask of one frame of a label
  (its context stack holds the neighbours);
- :meth:`LoadedModel.recording_probabilities`: the masks of frames of a
  recording (:class:`pipeline.Frames` reads the neighbours);
- :meth:`LoadedModel.mask_model`: an object with ``predict_probability_batch``
  over a context stack, which is what chain fits segment context frames
  with (:func:`body_fields.chain_fit`); a body-field net sees each stack
  frame's neighbours within the stack, and zero channels past its ends.

Only a body-field net gives body fields (:meth:`LoadedModel.fields`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .models import ModelCard, get_card, weights_path
from .roots import Libraries


@dataclass
class LoadedModel:
    card: ModelCard
    module: Any
    weights: Path

    @property
    def is_body_net(self) -> bool:
        return self.card.kind == "body_net"

    def fields(self, context: NDArray[np.uint8], valid: NDArray[np.bool_]) -> Any:
        """The body-field prediction (``body_proposal.FieldPrediction``) of a context stack's centre frame."""

        if not self.is_body_net:
            raise ValueError(f"{self.card.ref} is a mask model; it gives no body fields")
        from ..body_proposal import predict_fields

        return predict_fields(self.module, context, valid)

    def mask_probability(self, context: NDArray[np.uint8], valid: NDArray[np.bool_]) -> NDArray[np.float32]:
        """Worm probability of a context stack's centre frame."""

        if self.is_body_net:
            return self.fields(context, valid).mask
        centre = len(context) // 2
        return self.module.predict_probability_batch(context[centre : centre + 1], batch_size=1)[0]

    def recording_probabilities(self, frames: Any, indices: Sequence[int]) -> list[NDArray[np.float32]]:
        """Worm probability of frames of a recording (``frames``: a :class:`pipeline.Frames`)."""

        indices = [int(i) for i in indices]
        if self.is_body_net:
            from ..body_proposal import RecordingFieldPredictor

            return [p.mask for p in RecordingFieldPredictor(self.module, frames).predict(indices)]
        corrected, _, _ = frames.corrected(indices) if indices else (np.zeros((0, *frames.shape), np.uint8), 0, 0)
        return list(self.module.predict_probability_batch(corrected, batch_size=8))

    def mask_model(self) -> Any:
        """Something with ``predict_probability_batch(stack)`` for chain fits."""

        return _StackMasks(self.module) if self.is_body_net else self.module


class _Stack:
    """A context stack in the interface ``RecordingFieldPredictor`` reads frames through."""

    def __init__(self, frames: NDArray[np.uint8]) -> None:
        self.frames = np.asarray(frames)
        self.total = len(self.frames)

    def corrected(self, indices: Sequence[int]) -> tuple[NDArray[np.uint8], float, float]:
        return self.frames[list(indices)], 0.0, 0.0


class _StackMasks:
    def __init__(self, module: Any) -> None:
        self.module = module

    def predict_probability_batch(self, frames: NDArray[np.generic], batch_size: int = 8) -> NDArray[np.float32]:
        from ..body_proposal import RecordingFieldPredictor

        stack = np.asarray(frames, dtype=np.uint8)
        predictions = RecordingFieldPredictor(self.module, _Stack(stack), batch_size=batch_size).predict(range(len(stack)))
        return np.stack([p.mask for p in predictions]).astype(np.float32)


def load_model(libraries: Libraries, ref: str, device: Any = None) -> LoadedModel:
    """A library model's weights on ``device`` (default: the GPU when there is one)."""

    import torch

    card = get_card(libraries, ref)
    path = weights_path(libraries, ref)
    resolved = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if card.kind == "body_net":
        from ..body_net import load_body_net

        module = load_body_net(path, resolved)
    else:
        from ..segmenter import load_segmenter

        module = load_segmenter(path, resolved)
    return LoadedModel(card=card, module=module, weights=path)
