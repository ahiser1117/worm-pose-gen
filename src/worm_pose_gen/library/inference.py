"""Running a library model: load a card by reference and get its outputs for a frame.

Every page that runs a model goes through here (evaluation, training's
target preparation, Labeling's proposals, the Workspace), so a model is
always fed the way it was trained::

    from worm_pose_gen.library import Libraries
    from worm_pose_gen.library.inference import load_model

    model = load_model(libs, "lab:nir-body-lags3", device="cuda", fps=setup.fps)
    label = record.load()
    out = model.predict(label.context, label.context_valid)   # the centre frame of a context stack
    out.mask      # [H,W] worm probability
    out.ap        # [H,W] arc position, 0 head .. 1 tail (None when the model has no such output)
    out.head, out.tail, out.overlap                           # likewise; a segmenter gives only ``mask``

**Inputs.**  A model sees the flat-fielded frame as uint8 (a label's
``image``, the pipeline's corrected frames) and, when its card has lags, one
channel per lag, ``frame[t + lag] - frame[t - lag]``
(:mod:`temporal_context`).  Context is therefore a stack ``[2L+1,H,W]``
centred on the frame, with a validity flag per frame (a label stores ±16
frames); a lag whose end is missing or invalid gives a zero channel, as in
training.  :attr:`LoadedModel.max_lag` says how much context a model needs.

**Frame rate.**  Lags are stored on the card in frames (at the card's
``fps``) and in seconds.  ``load_model(..., fps=...)`` converts the seconds
to the nearest whole frames at another rate, so a model trained at 20 fps
looks 0.8 s back on a 40 fps setup too.

Three ways in:

- :meth:`LoadedModel.predict` -- one frame from its context stack;
- :meth:`LoadedModel.predict_sequence` -- frames of a contiguous sequence
  (a recording stretch), each with its neighbours from the same sequence;
- :meth:`LoadedModel.predict_probability_batch` -- the mask probability of
  every frame of a sequence, the interface body-target building's chain fits
  take from a mask model (:func:`body_fields.chain_fit`).

Loaded models are kept in a small cache keyed by reference, device, frame
rate and the weights file, so pages can call :func:`load_model` per request.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .models import ModelCard, get_card, weights_path
from .roots import Libraries


CACHED_MODELS = 4
OUTPUT_NAMES = ("mask", "ap", "head", "tail", "overlap")


@dataclass(frozen=True)
class Outputs:
    """A model's per-pixel outputs for one frame as probabilities (A-P as a position in [0, 1]); ``None`` where the model has no such output."""

    mask: NDArray[np.float32] | None = None
    ap: NDArray[np.float32] | None = None
    head: NDArray[np.float32] | None = None
    tail: NDArray[np.float32] | None = None
    overlap: NDArray[np.float32] | None = None

    def field_prediction(self) -> Any:
        """As :class:`body_proposal.FieldPrediction`, for the body tools that take one (body-field models only)."""

        from ..body_proposal import FieldPrediction

        if any(getattr(self, name) is None for name in OUTPUT_NAMES):
            raise ValueError("only a model with all five outputs gives a field prediction")
        return FieldPrediction(**{name: getattr(self, name) for name in OUTPUT_NAMES})


def lags_at(card: ModelCard, fps: float | None) -> tuple[int, ...]:
    """The card's lags in frames at ``fps`` (its own lags when ``fps`` is unknown or the card's rate)."""

    frames = tuple(int(lag) for lag in card.inputs.get("lags_frames") or ())
    seconds = card.inputs.get("lags_s")
    if fps is None or seconds is None or card.inputs.get("fps") in (None, fps):
        return frames
    converted = tuple(max(1, int(round(float(s) * float(fps)))) for s in seconds)
    if len(set(converted)) != len(converted):
        raise ValueError(f"{card.ref}: lags {list(seconds)} s collapse at {fps} fps")
    return converted


class LoadedModel:
    """A card with its network on a device; see the module docstring."""

    def __init__(self, card: ModelCard, module: Any, lags: Sequence[int]) -> None:
        self.card = card
        self.module = module
        self.lags = tuple(int(lag) for lag in lags)
        self.outputs = tuple(card.outputs)

    @property
    def ref(self) -> str:
        return self.card.ref

    @property
    def device(self) -> Any:
        return self.module.device

    @property
    def max_lag(self) -> int:
        return max(self.lags, default=0)

    # ------------------------------------------------------------------ running

    def _inputs(self, frames: NDArray[np.generic], valid: NDArray[np.bool_], centre: int) -> NDArray[np.float32]:
        from ..segmenter import INPUT_MEAN, INPUT_STD

        frame = frames[centre].astype(np.float32)
        channels = [(frame / 255.0 - INPUT_MEAN) / INPUT_STD]
        for lag in self.lags:
            later, earlier = centre + lag, centre - lag
            if earlier < 0 or later >= len(frames) or not (valid[later] and valid[earlier]):
                channels.append(np.zeros(frame.shape, np.float32))
            else:
                channels.append((frames[later].astype(np.float32) - frames[earlier].astype(np.float32)) / (255.0 * INPUT_STD))
        return np.stack(channels)

    def _run(self, inputs: NDArray[np.float32]) -> list[Outputs]:
        import torch

        with torch.inference_mode():
            tensor = torch.as_tensor(inputs).to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.module(tensor).float()
            maps = torch.sigmoid(logits).cpu().numpy()
        if self.card.kind == "segmenter":
            return [Outputs(mask=m[0]) for m in maps]
        return [Outputs(**{name: m[k] for k, name in enumerate(OUTPUT_NAMES) if name in self.outputs}) for m in maps]

    def predict_sequence(
        self, frames: NDArray[np.generic], valid: NDArray[np.bool_] | None = None, indices: Sequence[int] | None = None,
        *, batch_size: int = 8,
    ) -> list[Outputs]:
        """Outputs for ``frames[indices]`` (all by default) of a contiguous ``[N,H,W]`` sequence.

        A lag that reaches past the sequence, or to a frame marked invalid,
        gives a zero channel.
        """

        stack = np.asarray(frames)
        if stack.ndim != 3:
            raise ValueError("frames must have shape [N,H,W]")
        ok = np.ones(len(stack), bool) if valid is None else np.asarray(valid, bool)
        chosen = range(len(stack)) if indices is None else [int(i) for i in indices]
        results: list[Outputs] = []
        chosen = list(chosen)
        for start in range(0, len(chosen), max(1, int(batch_size))):
            batch = chosen[start : start + batch_size]
            results.extend(self._run(np.stack([self._inputs(stack, ok, i) for i in batch])))
        return results

    def predict(self, context: NDArray[np.generic], valid: NDArray[np.bool_] | None = None) -> Outputs:
        """Outputs for the centre frame of a ``[2L+1,H,W]`` context stack (a single ``[H,W]`` frame for a model without lags)."""

        stack = np.asarray(context)
        if stack.ndim == 2:
            stack = stack[None]
        if len(stack) % 2 != 1:
            raise ValueError("a context stack has an odd number of frames, centred on the predicted one")
        centre = len(stack) // 2
        if self.max_lag > centre:
            raise ValueError(f"{self.ref} needs ±{self.max_lag} frames of context; the stack has ±{centre}")
        return self.predict_sequence(stack, valid, [centre])[0]

    def predict_probability_batch(self, frames: NDArray[np.generic], batch_size: int = 8) -> NDArray[np.float32]:
        """``[N,H,W]`` worm probability of every frame of a contiguous sequence."""

        if "mask" not in self.outputs:
            raise ValueError(f"{self.ref} has no mask output")
        return np.stack([out.mask for out in self.predict_sequence(frames, batch_size=batch_size)])


# --------------------------------------------------------------------------- loading


def load_module(kind: str, path: Any, device: Any) -> Any:
    """The network of a weights file, for inference, by the card's ``kind``."""

    if kind == "segmenter":
        from ..segmenter import load_segmenter

        return load_segmenter(path, device)
    if kind == "body_net":
        from ..body_net import load_body_net

        return load_body_net(path, device)
    raise ValueError(f"unknown model kind {kind!r}")


_cache: OrderedDict[tuple, LoadedModel] = OrderedDict()
_cache_lock = threading.Lock()


def load_model(libraries: Libraries, ref: str, *, device: Any = None, fps: float | None = None) -> LoadedModel:
    """The model ``ref`` ready to run on ``device`` (default: CUDA when available), with lags for ``fps``."""

    import torch

    card = get_card(libraries, ref)
    path = weights_path(libraries, ref)
    resolved = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    lags = lags_at(card, fps)
    key = (str(libraries.lab), str(libraries.personal), ref, str(resolved), lags, path.stat().st_mtime_ns)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    module = load_module(card.kind, path, resolved)
    trained = tuple(getattr(module, "lags", ()))
    if len(trained) != len(lags):
        raise ValueError(f"{ref}: the card lists {len(lags)} lags but the network takes {len(trained)}")
    model = LoadedModel(card, module, lags)
    with _cache_lock:
        _cache[key] = model
        while len(_cache) > CACHED_MODELS:
            _cache.popitem(last=False)
    return model
