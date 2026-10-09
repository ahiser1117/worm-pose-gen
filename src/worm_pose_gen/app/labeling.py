"""The Labeling page's backend: open a frame, propose and refine its mask, propose or fit its body, save the label.

One frame's label is labeled on one page (``docs/APP_SIMPLIFICATION.md``,
section 3): the mask first, then the body, then Save.  A frame is named by
an *entry* ``{"recording", "frame", "path"?}`` within the current setup;
``path`` is the recording file, needed only while the frame has no label
yet.

**Saving to.** Saves go to the setup's label collection, in the personal
library (:class:`library.Collection`); a recording's first label adds it to
the collection, and every dataset of the setup lists it as not included
until someone chooses its split (the Training page's Datasets tab).

**Where a frame comes from.** A frame the collection has labeled (in
either library) opens from its current label, which holds the image, the context frames
and the nose landmarks, so the recording is not needed.  Any other frame is
captured from its recording (:func:`library.capture.read_label_inputs`);
the captures of the last few frames stay in memory for the proposals and
the save.  The mask the frame opens with is the label's; for a frame of a
Relabel queue the workspace's mask; otherwise the default mask model's
proposal at a threshold of 0.5 (empty without a model).

**Models.** The setup's default ``mask`` model gives the Network proposal
and the ``body`` model (a body-field net) the body proposal: a trace through
its predicted A-P field (:func:`body_proposal.propose_trace`) fit like a hand
trace (:func:`body_fields.trace_fit`).  Library models are loaded once on
the app's device (:mod:`library.inference`); fits run one at a time.

**Saving** writes a new label revision (:meth:`library.Collection.save`) with
the body as decided: a trace (traced by hand or the proposal's), a head end
(``head_xy``: the orientation was flipped or confirmed without a trace),
neither (the automatic orientation), and ``mask_only``.  The origin is the
queue's (``fix`` for Relabel, ``spread`` for a frame search); an edit outside
a queue keeps ``fix`` for a label made by a fix and is ``spread`` otherwise.
Then a job rebuilds the label's body targets (:mod:`library.targets`).
"""

from __future__ import annotations

from collections import OrderedDict
import binascii
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .. import library
from ..jobs import JobSpec
from ..library.targets import recording_length, setup_length, targets_command
from .images import data_url, decode_mask_data_url, mask_to_png_values, probability_to_png
from .routers.jobs import place
from .routers.library import label_row, targets_layers
from .state import NotFound, _integer

# Recording captures kept in memory: each holds the 33 context frames of one frame.
CAPTURES = 4
REFINE_METHODS = ("fill_holes", "largest", "grow", "shrink")
# The widest hole Fill holes closes, as the pipeline's mask cleanup does.
HOLE_FILL_RADIUS_PX = 8
TARGETS_JOB_KIND = "body_targets"


def encode_mask(mask: np.ndarray) -> str:
    return data_url(mask_to_png_values(mask))


def decode_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
    try:
        return decode_mask_data_url(str(value or ""), shape)
    except (OSError, binascii.Error) as error:
        raise ValueError("mask must contain a readable base64 PNG image") from error


def refine_mask(mask: np.ndarray, method: str, device: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """One of the mask tools on labels (0/1/255): Fill holes, Largest (component), Grow or Shrink by a pixel; ignore pixels stay."""

    from ..classical import _dilate, _erode, _largest_component
    from ..mask_fit import fill_narrow_holes
    from ..segmenter import IGNORE_LABEL

    worm = mask == 1
    ignore = mask == IGNORE_LABEL
    info: dict[str, Any] = {}
    if method == "fill_holes":
        worm, added = fill_narrow_holes(worm, HOLE_FILL_RADIUS_PX, device=device)
        info["pixels_added"] = int(added)
    elif method == "largest":
        if worm.any():
            worm, _, count = _largest_component(worm)
            info["components_removed"] = int(count - 1)
    elif method == "grow":
        worm = _dilate(worm, 1)
    elif method == "shrink":
        worm = _erode(worm, 1)
    else:
        raise ValueError(f"unknown refinement {method!r}; expected one of {REFINE_METHODS}")
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[worm] = 1
    out[ignore & ~worm] = IGNORE_LABEL
    return out, info


def encode_ap(ap: np.ndarray) -> str:
    """The A-P field as 0 (undefined) or ``1 + round(254 * ap)``, as the library API sends it."""

    ap = np.asarray(ap, dtype=np.float32)
    return data_url(np.where(np.isfinite(ap), 1 + np.round(254 * np.clip(np.nan_to_num(ap), 0, 1)), 0).astype(np.uint8))


def _point(xy: Any) -> list[float] | None:
    xy = np.asarray(xy, dtype=np.float64)
    return xy.tolist() if xy.shape == (2,) and np.all(np.isfinite(xy)) else None


def _points(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"'{name}' needs at least two [x, y] points")
    return points


def _positive(value: Any, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"'{name}' must be a positive number of pixels")
    return number


def body_layers(mask: np.ndarray, centerline: np.ndarray, width_profile: np.ndarray, fit_iou: float) -> dict[str, Any]:
    """A head-first body as the page draws it: midline, widths, A-P field, ends and how well it fits the mask."""

    from ..body_targets import render_body_targets

    targets = render_body_targets(mask, centerline, width_profile)
    return {
        "centerline_xy": np.asarray(centerline).tolist(), "width_profile": np.asarray(width_profile).tolist(),
        "ap": encode_ap(targets.ap), "head_xy": _point(targets.head_xy), "tail_xy": _point(targets.tail_xy),
        "fit_iou": float(fit_iou),
    }


class Labeling:
    """The services of the Labeling page; one per app (``AppState.labeling``)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._captures: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        # Proposals, fits and target builds share the app's device: one at a time.
        self.fit_lock = threading.Lock()

    @property
    def libraries(self) -> library.Libraries:
        return self.app.libraries

    # ------------------------------------------------------------------ labels

    def existing(self, setup_ref: str, recording: str, frame: int) -> library.LabelRecord | None:
        """The frame's current label in the setup's collection, if it has one."""

        try:
            return library.Collection(self.libraries, setup_ref).get(recording, frame)
        except LookupError:
            return None

    # ------------------------------------------------------------------ frames

    def _entry(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        setup = str(payload.get("setup") or "")
        library.parse_ref(setup)
        entry = payload.get("entry")
        if not isinstance(entry, dict) or not entry.get("recording"):
            raise ValueError("'entry' must name a recording and a frame")
        return setup, {"recording": str(entry["recording"]), "frame": _integer(entry, "frame"), "path": entry.get("path")}

    def capture(self, setup_ref: str, path: str, frame: int) -> dict[str, Any]:
        """What a new label stores from its recording, read once and kept for the next requests about the frame."""

        from ..library.capture import read_label_inputs

        setup = library.get_setup(self.libraries, setup_ref)
        resolved = str(Path(path).expanduser().resolve())
        key = (resolved, int(frame), setup.video.get("dataset_path"), setup.video.get("flat_field"))
        with self._lock:
            if key in self._captures:
                self._captures.move_to_end(key)
                return self._captures[key]
        owner = library.setup_for_recording(self.libraries, resolved)
        if owner != setup_ref:
            raise ValueError(f"{resolved} belongs to {owner or 'no setup'}, not to {setup_ref}")
        try:
            inputs = read_label_inputs(resolved, frame, video=setup.video, flat_field_cache=self.app.config.dataset_root / "flat_fields")
        except OSError as error:
            raise ValueError(f"{Path(resolved).name} cannot be read: {error}") from error
        with self._lock:
            self._captures[key] = inputs
            while len(self._captures) > CAPTURES:
                self._captures.popitem(last=False)
        return inputs

    def inputs(self, setup_ref: str, entry: dict[str, Any], record: library.LabelRecord | None) -> dict[str, Any]:
        """The frame, context and nose of an entry: from its label when it has one, else from its recording."""

        if record is not None:
            label = record.load()
            return {"image": label.image, "image_raw": label.image_raw, "context": label.context, "context_valid": label.context_valid,
                    "nose_xy": label.nose_xy, "nose_valid": label.nose_valid, "source_path": record.source_path,
                    "dataset_path": record.dataset_path}
        if not entry.get("path"):
            raise NotFound(f"{entry['recording']} frame {entry['frame']} has no label; its recording path is needed")
        return self.capture(setup_ref, str(entry["path"]), entry["frame"])

    def _resolve(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any], library.LabelRecord | None, dict[str, Any]]:
        setup, entry = self._entry(payload)
        record = self.existing(setup, entry["recording"], entry["frame"])
        return setup, entry, record, self.inputs(setup, entry, record)

    def model(self, setup_ref: str, ref: str) -> Any:
        """A library model on the app's device at the setup's frame rate (:func:`library.inference.load_model` caches it)."""

        from ..library.inference import load_model

        return load_model(self.libraries, ref, device=self.app.device, fps=library.get_setup(self.libraries, setup_ref).fps)

    def defaults(self, setup_ref: str) -> dict[str, str | None]:
        defaults = library.get_setup(self.libraries, setup_ref).defaults
        body = defaults.get("body")
        if body is not None and library.get_card(self.libraries, body).kind != "body_net":
            body = None
        return {"mask": defaults.get("mask"), "body": body}

    def probability(self, setup_ref: str, inputs: dict[str, Any]) -> np.ndarray | None:
        ref = self.defaults(setup_ref)["mask"]
        if ref is None:
            return None
        with self.fit_lock:
            return self.model(setup_ref, ref).predict(inputs["context"], inputs["context_valid"]).mask

    def open(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Everything the page shows for an entry; see the module docstring for where the mask comes from."""

        setup, entry, record, inputs = self._resolve(payload)
        image = inputs["image"]
        shape = image.shape
        probability = None
        body: dict[str, Any] = {"trace_xy": None, "head_xy": None, "mask_only": False, "trace_extend": False, "trace_length_px": None}
        targets = None
        if record is not None:
            label = record.load()
            mask, source = label.mask, "label"
            body = {"trace_xy": None if label.trace_xy is None else label.trace_xy.tolist(),
                    "head_xy": None if label.head_xy is None else label.head_xy.tolist(), "mask_only": record.mask_only,
                    "trace_extend": bool(label.meta.get("trace_extend")), "trace_length_px": label.meta.get("trace_length_px")}
            targets = targets_layers(self.app, record)
        else:
            mask, source = self._workspace_mask(payload.get("queue"), entry, shape)
            if mask is None:
                probability = self.probability(setup, inputs)
                mask, source = ((probability >= 0.5).astype(np.uint8), "network") if probability is not None else (np.zeros(shape, np.uint8), "empty")
        own_revision = library.Collection(self.libraries, setup).own_revision(entry["recording"], entry["frame"])
        centre = len(inputs["context"]) // 2
        return {
            "setup": setup, "entry": entry, "width": int(shape[1]), "height": int(shape[0]),
            "image": data_url(image), "image_raw": data_url(inputs["image_raw"]),
            "mask": encode_mask(mask), "mask_source": source,
            "probability": None if probability is None else data_url(probability_to_png(probability)),
            "label": None if record is None else label_row(self.app, record), "expected_revision": own_revision,
            "body": body, "targets": targets,
            "nose_xy": _point(inputs["nose_xy"][centre]) if bool(inputs["nose_valid"][centre]) else None,
            "max_lag": centre, "context_valid": np.asarray(inputs["context_valid"]).tolist(), "models": self.defaults(setup),
        }

    def _workspace_mask(self, queue_id: Any, entry: dict[str, Any], shape: tuple[int, ...]) -> tuple[np.ndarray | None, str]:
        """A Relabel frame's mask in its workspace (the override when the user edited it)."""

        if not queue_id:
            return None, ""
        queue = self.app.queues.get(str(queue_id))
        if queue.get("kind") != "relabel":
            return None, ""
        workspace = self.app.workspace(queue["workspace"])
        row = workspace.row_of(entry["frame"])
        labels = workspace.get_override_mask(row)
        if labels is None:
            stored = workspace.get_mask(row)
            labels = None if stored is None else stored.astype(np.uint8)
        if labels is None or labels.shape != tuple(shape):
            return None, ""
        return labels, "workspace"

    def context(self, payload: dict[str, Any]) -> dict[str, Any]:
        *_, inputs = self._resolve(payload)
        return {"max_lag": len(inputs["context"]) // 2, "valid": np.asarray(inputs["context_valid"]).tolist(),
                "frames": [data_url(frame) for frame in inputs["context"]]}

    def network(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The default mask model's worm probability (PNG, 0..255); the page thresholds it itself."""

        setup, *_, inputs = self._resolve(payload)
        probability = self.probability(setup, inputs)
        if probability is None:
            raise ValueError(f"{setup} has no default mask model yet; use Threshold, or choose a model on the Training page")
        return {"probability": data_url(probability_to_png(probability)), "model": self.defaults(setup)["mask"]}

    def refine(self, payload: dict[str, Any]) -> dict[str, Any]:
        method = str(payload.get("method") or "")
        if method not in REFINE_METHODS:
            raise ValueError(f"unknown refinement {method!r}; expected one of {tuple(REFINE_METHODS)}")
        width, height = _integer(payload, "width"), _integer(payload, "height")
        mask = decode_mask(payload.get("mask"), (height, width))
        with self.fit_lock:
            refined, info = refine_mask(mask, method, self.app.device)
        return {"mask": encode_mask(refined), "info": info}

    # ------------------------------------------------------------------ bodies

    def _fit(self, mask: np.ndarray, trace: np.ndarray, extend_to_px: float | None = None) -> dict[str, Any]:
        import torch

        from ..body_fields import fit_config, trace_fit
        from ..mask_fit import default_width_template

        config = fit_config()
        centerline, profile, iou = trace_fit(
            mask, trace, config=config, template=default_width_template(config.n_points),
            device=torch.device(self.app.device), extend_to_px=extend_to_px,
        )
        return body_layers(mask, centerline, profile, iou)

    def body_length(self, setup_ref: str, entry: dict[str, Any], record: library.LabelRecord | None) -> tuple[float | None, str | None]:
        """The recording's typical body length, which a trace extended off camera runs to, and where it came from.

        The first of: ``labels`` (the median whole, well-fit body among the
        recording's labels), ``analysis`` (the body-size prior of its
        analysed workspace), ``estimate`` (the cached prior a Find frames job
        or an earlier analysis bootstrapped over its frames), ``setup`` (the
        median over the setup's recordings with labels: a rough guess).
        """

        from ..pipeline import cached_prior, workspace_prior
        from ..workspace import Workspace

        builder = library.target_builder(self.libraries, setup_ref)
        length = recording_length(self.libraries, setup_ref, entry["recording"], builder)
        if length is not None:
            return length, "labels"
        path = entry.get("path") or (record.source_path if record is not None else None)
        if path:
            name = self.app.workspace_of_recording(Path(path))
            prior = None if name is None else workspace_prior(Workspace.open(self.app.workspace_path(name)))
            if prior is not None:
                return prior.length_px, "analysis"
        prior = cached_prior(path or entry["recording"])
        if prior is not None:
            return prior.length_px, "estimate"
        length = setup_length(self.libraries, setup_ref, builder)
        if length is not None:
            return length, "setup"
        return None, None

    def proposal(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The body model's proposal for the frame's current mask: ``{"status": "ready", "trace_xy", ...body layers}``,
        ``{"status": "no_trace"}`` when the fields give no trace, or ``{"status": "unavailable", "reason"}``."""

        from ..body_proposal import propose_trace

        setup, _, _, inputs = self._resolve(payload)
        ref = self.defaults(setup)["body"]
        if ref is None:
            return {"status": "unavailable", "reason": "this setup has no body-field model"}
        mask = decode_mask(payload.get("mask"), inputs["image"].shape) == 1
        if not mask.any():
            return {"status": "no_trace", "model": ref}
        with self.fit_lock:
            prediction = self.model(setup, ref).predict(inputs["context"], inputs["context_valid"]).field_prediction()
            trace = propose_trace(prediction, mask)
            if trace is None:
                return {"status": "no_trace", "model": ref}
            body = self._fit(mask, trace)
        return {"status": "ready", "model": ref, "trace_xy": trace.tolist(), **body}

    def fit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The body fit along a traced midline (head first) on the frame's current mask.

        ``extend`` continues a trace ending at or past the border off camera
        to the recording's typical body length (:meth:`body_length`,
        :func:`body_fields.trace_fit`); the answer's ``extended_px`` says how
        far it went (0: it did not, as for a trace ending in view or without
        any length), with ``length_px`` and ``length_source``.
        """

        from ..body_fields import extend_trace

        setup, entry, record, inputs = self._resolve(payload)
        trace = _points(payload.get("trace_xy"), "trace_xy")
        mask = decode_mask(payload.get("mask"), inputs["image"].shape) == 1
        if not mask.any():
            raise ValueError("paint the worm before tracing its midline")
        length, source = self.body_length(setup, entry, record) if payload.get("extend") else (None, None)
        path = extend_trace(trace, mask.shape, length)
        extended = float(np.linalg.norm(path[-1] - trace[-1])) if len(path) > len(trace) else 0.0
        with self.fit_lock:
            body = self._fit(mask, trace, length)
        return {"trace_xy": trace.tolist(), "extended_px": extended, "length_px": length, "length_source": source, **body}

    # ------------------------------------------------------------------ saving

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a new label revision, start the job that rebuilds its body targets, and mark the queue entry saved."""

        setup, entry = self._entry(payload)
        queue = self.app.queues.get(str(payload["queue"])) if payload.get("queue") else None
        collection = library.Collection(self.libraries, setup)
        record = self.existing(setup, entry["recording"], entry["frame"])
        inputs = self.inputs(setup, entry, record)
        if queue is not None:
            origin = queue["origin"]
        else:
            origin = "fix" if record is not None and record.origin == "fix" else "spread"
        mask = decode_mask(payload.get("mask"), inputs["image"].shape)
        trace = _points(payload.get("trace_xy"), "trace_xy")
        head = payload.get("head_xy")
        expected = payload.get("expected_revision")
        saved = collection.save(
            recording=entry["recording"], frame=entry["frame"], mask=mask, origin=origin,
            orientation="manual" if trace is not None or head is not None else "auto", head_xy=head, trace_xy=trace,
            trace_extend=bool(payload.get("trace_extend")) and trace is not None,
            trace_length_px=_positive(payload.get("trace_length_px"), "trace_length_px"),
            mask_only=bool(payload.get("mask_only")), expected_revision=None if expected is None else int(expected), **inputs,
        )
        job = None
        if (mask == 1).any():
            spec = JobSpec(kind=TARGETS_JOB_KIND, params={"labels": [saved.identity]}, gpus=1 if self.app.config.gpus else 0,
                           label=f"Body targets of {saved.recording} frame {saved.frame}")
            job = self.app.runner.submit(place(self.app, spec, {}), targets_command(self.libraries, [saved.identity])).to_dict()
        summary = None
        if queue is not None:
            summary = self.app.queues.mark_saved(queue["id"], entry, saved.identity)
        return {"label": label_row(self.app, saved), "job": job, "queue": summary}

