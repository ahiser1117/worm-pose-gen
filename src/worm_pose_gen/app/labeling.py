"""The Labeling page's backend: open a frame, propose and refine its mask, propose or fit its body, save the label.

One frame's label is labeled on one page (``docs/APP_SIMPLIFICATION.md``,
section 3): the mask first, then the body, then Save.  A frame is named by
an *entry* ``{"recording", "frame", "path"?}`` within the current setup;
``path`` is the recording file, needed only while the frame has no label
yet.

**Saving to.** Saves go to the user's personal dataset for the setup: the
one the page asks for (``dataset``), else ``mine:<setup-id>-labels`` when it
exists, else the first personal dataset of the setup.  When there is none,
the first save creates ``mine:<setup-id>-labels``, extending the setup's lab
dataset (the first, when there are several) or standalone when the lab has
none.  Until then the lab dataset is where existing labels are read from.

**Where a frame comes from.** A frame the dataset (or the one it extends)
has labeled opens from the label, which holds the image, the context frames
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
the app's device; fits run one at a time.

**Saving** writes a new label revision (:meth:`library.Dataset.save`) with
the body as decided: a trace (traced by hand or the proposal's), a head end
(``head_xy``: the orientation was flipped or confirmed without a trace),
neither (the automatic orientation), and ``mask_only``.  The origin is the
queue's (``fix`` for Relabel, ``spread`` for a frame search); an edit outside
a queue keeps ``fix`` for a label made by a fix and is ``spread`` otherwise.
Then a job rebuilds the label's body targets (:mod:`library.targets`).
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import binascii
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .. import library
from ..jobs import JobSpec
from ..label_app import Proposer, data_url, decode_mask_data_url, mask_to_png_values, probability_to_png
from ..library.datasets import assign_split
from ..library.targets import recording_length, targets_command
from .routers.jobs import place
from .routers.library import label_row, targets_layers
from .state import NotFound, _integer

# Recording captures kept in memory: each holds the 33 context frames of one frame.
CAPTURES = 4
REFINE_METHODS = {"fill_holes": "fill_holes", "largest": "largest_component", "grow": "dilate", "shrink": "erode"}
TARGETS_JOB_KIND = "body_targets"


def encode_mask(mask: np.ndarray) -> str:
    return data_url(mask_to_png_values(mask))


def decode_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
    try:
        return decode_mask_data_url(str(value or ""), shape)
    except (OSError, binascii.Error) as error:
        raise ValueError("mask must contain a readable base64 PNG image") from error


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
        self._models: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        # Proposals, fits and target builds share the app's device: one at a time.
        self.fit_lock = threading.Lock()

    @property
    def libraries(self) -> library.Libraries:
        return self.app.libraries

    # ------------------------------------------------------------------ datasets

    def saving(self, setup_ref: str, requested: str | None = None) -> dict[str, Any]:
        """Where saves for a setup go: ``{"setup", "dataset", "create", "extends", "choices", "reading"}``.

        ``dataset`` is the existing personal dataset saves go to, or ``None``
        when the first save will create ``create`` (extending ``extends``);
        ``reading`` is the dataset existing labels are read from meanwhile.
        """

        libraries = self.libraries
        setup = library.get_setup(libraries, setup_ref)
        _, setup_id = library.parse_ref(setup.ref)
        datasets = library.list_datasets(libraries, setup.ref)
        mine = [d.ref for d in datasets if d.scope == "mine"]
        lab = [d.ref for d in datasets if d.scope == "lab"]
        default_id = f"{setup_id}-labels"
        if requested:
            if requested not in mine:
                raise ValueError(f"{requested} is not a personal dataset of {setup.ref}")
            chosen = requested
        else:
            chosen = f"mine:{default_id}" if f"mine:{default_id}" in mine else (mine[0] if mine else None)
        extends = lab[0] if lab else None
        return {
            "setup": setup.ref, "dataset": chosen, "create": None if chosen else f"mine:{default_id}",
            "extends": extends, "choices": mine, "reading": chosen or extends,
        }

    def dataset_for_save(self, setup_ref: str, requested: str | None) -> tuple[library.Dataset, bool]:
        """The dataset to save into, created on the first save; and whether it was just created."""

        target = self.saving(setup_ref, requested)
        if target["dataset"]:
            return library.Dataset(self.libraries, target["dataset"]), False
        _, dataset_id = library.parse_ref(target["create"])
        setup = library.get_setup(self.libraries, setup_ref)
        dataset = library.create_dataset(
            self.libraries, dataset_id, setup=setup_ref, extends=target["extends"], name=f"{setup.name}: my labels",
            description="Labels saved from the Labeling page.",
        )
        return dataset, True

    def existing(self, reading: str | None, recording: str, frame: int) -> library.LabelRecord | None:
        if reading is None:
            return None
        try:
            return library.Dataset(self.libraries, reading).get(recording, frame)
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

    def _resolve(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], library.LabelRecord | None, dict[str, Any]]:
        setup, entry = self._entry(payload)
        target = self.saving(setup, payload.get("dataset") or None)
        record = self.existing(target["reading"], entry["recording"], entry["frame"])
        return setup, entry, target, record, self.inputs(setup, entry, record)

    def model(self, ref: str) -> library.LoadedModel:
        """A library model on the app's device, loaded at first use (and again when its weights change)."""

        stamp = library.weights_path(self.libraries, ref).stat().st_mtime
        with self._lock:
            cached = self._models.get(ref)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        loaded = library.load_model(self.libraries, ref, self.app.device)
        with self._lock:
            self._models[ref] = (stamp, loaded)
        return loaded

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
            return self.model(ref).mask_probability(inputs["context"], inputs["context_valid"])

    def open(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Everything the page shows for an entry; see the module docstring for where the mask comes from."""

        setup, entry, target, record, inputs = self._resolve(payload)
        image = inputs["image"]
        shape = image.shape
        probability = None
        body: dict[str, Any] = {"trace_xy": None, "head_xy": None, "mask_only": False}
        targets = None
        if record is not None:
            label = record.load()
            mask, source = label.mask, "label"
            body = {"trace_xy": None if label.trace_xy is None else label.trace_xy.tolist(),
                    "head_xy": None if label.head_xy is None else label.head_xy.tolist(), "mask_only": record.mask_only}
            targets = targets_layers(self.app, record)
        else:
            mask, source = self._workspace_mask(payload.get("queue"), entry, shape)
            if mask is None:
                probability = self.probability(setup, inputs)
                mask, source = ((probability >= 0.5).astype(np.uint8), "network") if probability is not None else (np.zeros(shape, np.uint8), "empty")
        own_revision = 0
        split = None
        if target["dataset"]:
            dataset = library.Dataset(self.libraries, target["dataset"])
            own = dataset.revisions(entry["recording"], entry["frame"])
            own_revision = own[-1].revision if own else 0
            split = dataset.splits().get(entry["recording"])
        elif target["extends"]:
            split = library.Dataset(self.libraries, target["extends"]).splits().get(entry["recording"])
        if split is None and target["reading"]:
            counts = Counter(r.split for r in library.Dataset(self.libraries, target["reading"]).labels())
            split_note = f"{assign_split(counts)} (new recording: set by its first label)"
        else:
            split_note = None
        centre = len(inputs["context"]) // 2
        return {
            "setup": setup, "saving": target, "entry": entry, "width": int(shape[1]), "height": int(shape[0]),
            "image": data_url(image), "image_raw": data_url(inputs["image_raw"]),
            "mask": encode_mask(mask), "mask_source": source,
            "probability": None if probability is None else data_url(probability_to_png(probability)),
            "label": None if record is None else label_row(self.app, record), "expected_revision": own_revision,
            "split": split, "split_note": split_note, "body": body, "targets": targets,
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
            refined, info = Proposer.refine(mask, REFINE_METHODS[method], self.app.device)
        return {"mask": encode_mask(refined), "info": info}

    # ------------------------------------------------------------------ bodies

    def _fit(self, mask: np.ndarray, trace: np.ndarray, length_px: float | None) -> dict[str, Any]:
        import torch

        from ..body_fields import fit_config, trace_fit
        from ..mask_fit import default_width_template

        config = fit_config()
        centerline, profile, iou = trace_fit(
            mask, trace, length_px=length_px, config=config, template=default_width_template(config.n_points),
            device=torch.device(self.app.device),
        )
        return body_layers(mask, centerline, profile, iou)

    def _length(self, target: dict[str, Any], recording: str) -> float | None:
        if target["reading"] is None:
            return None
        builder = library.target_builder(self.libraries, target["setup"])
        return recording_length(self.libraries, target["reading"], recording, builder)

    def proposal(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The body model's proposal for the frame's current mask: ``{"status": "ready", "trace_xy", ...body layers}``,
        ``{"status": "no_trace"}`` when the fields give no trace, or ``{"status": "unavailable", "reason"}``."""

        from ..body_proposal import propose_trace

        setup, entry, target, _, inputs = self._resolve(payload)
        ref = self.defaults(setup)["body"]
        if ref is None:
            return {"status": "unavailable", "reason": "this setup has no body-field model"}
        mask = decode_mask(payload.get("mask"), inputs["image"].shape) == 1
        if not mask.any():
            return {"status": "no_trace", "model": ref}
        with self.fit_lock:
            prediction = self.model(ref).fields(inputs["context"], inputs["context_valid"])
            trace = propose_trace(prediction, mask)
            if trace is None:
                return {"status": "no_trace", "model": ref}
            body = self._fit(mask, trace, self._length(target, entry["recording"]))
        return {"status": "ready", "model": ref, "trace_xy": trace.tolist(), **body}

    def fit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The body fit along a traced midline (head first) on the frame's current mask."""

        setup, entry, target, _, inputs = self._resolve(payload)
        trace = _points(payload.get("trace_xy"), "trace_xy")
        mask = decode_mask(payload.get("mask"), inputs["image"].shape) == 1
        if not mask.any():
            raise ValueError("paint the worm before tracing its midline")
        with self.fit_lock:
            return {"trace_xy": trace.tolist(), **self._fit(mask, trace, self._length(target, entry["recording"]))}

    # ------------------------------------------------------------------ saving

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a new label revision, start the job that rebuilds its body targets, and mark the queue entry saved."""

        setup, entry = self._entry(payload)
        queue = self.app.queues.get(str(payload["queue"])) if payload.get("queue") else None
        dataset, created = self.dataset_for_save(setup, payload.get("dataset") or None)
        record = self.existing(dataset.ref, entry["recording"], entry["frame"])
        inputs = self.inputs(setup, entry, record)
        if queue is not None:
            origin = queue["origin"]
        else:
            origin = "fix" if record is not None and record.origin == "fix" else "spread"
        mask = decode_mask(payload.get("mask"), inputs["image"].shape)
        trace = _points(payload.get("trace_xy"), "trace_xy")
        head = payload.get("head_xy")
        expected = payload.get("expected_revision")
        saved = dataset.save(
            recording=entry["recording"], frame=entry["frame"], mask=mask, origin=origin,
            orientation="manual" if trace is not None or head is not None else "auto", head_xy=head, trace_xy=trace,
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
        return {"label": label_row(self.app, saved), "dataset": dataset.ref, "created": created, "job": job, "queue": summary}

