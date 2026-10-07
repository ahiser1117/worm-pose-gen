"""Body-field targets of a label: a rebuildable cache in the personal library, keyed by label revision and builder.

The targets (the head-first tube fitted to the hand mask, the rendered A-P
field, overlap, head and tail, ``fit_iou``, self-contact) are computed from
the label alone by :func:`body_fields.fit_targets`, the builder the old
``body_fields/`` records came from: a traced label is refit along its trace,
a manual orientation puts the head at the end nearest the chosen point, and
otherwise the acquisition nose or the taper decides.  A tangled frame is
also chain-fit through its context frames, which a mask model segments: the
*builder*, the setup's default ``mask`` model (:func:`target_builder`).

So targets depend on two things, the label revision (its ``sha256``, which
never changes) and the builder, and both name the cache entry::

    <personal>/cache/body_targets/<builder>/<sha256>.npz    the arrays
    <personal>/cache/body_targets/<builder>/<sha256>.json   the meta, which listings read without the arrays

``<builder>`` is the model reference as a file name (``lab.nir-hand284``,
:func:`roots.ref_filename`), or ``none`` for a setup without a mask model.
Changing the setup's default model therefore makes every label's targets
missing until they are rebuilt with the new model; the old entries stay
until someone deletes the directory.  Every user builds targets in their own
library, lab labels included, since the lab library is read-only.

The readers take ``builder``; left at :data:`SETUP_DEFAULT` they look up
the setup's current default for the label's dataset.  A listing of many
labels of one setup passes the builder it looked up once.

Building needs the fitter (a GPU makes it seconds instead of tens of
seconds).  ``python -m worm_pose_gen.library.targets`` builds one label's
targets as a job (the Labeling page starts one after every save).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .datasets import Dataset
from .labels import LabelRecord
from .roots import Libraries, read_json, ref_filename, write_json
from ..workspace import utc_now


CACHE_DIR = Path("cache") / "body_targets"
# ``builder`` left at this value means: the setup's current default mask model.
SETUP_DEFAULT = "setup-default"
NO_BUILDER = "none"


def target_builder(libraries: Libraries, setup_ref: str) -> str | None:
    """The model that builds a setup's targets: its default ``mask`` model, or ``None`` when it has none."""

    from .setups import get_setup

    return get_setup(libraries, setup_ref).defaults.get("mask")


def _builder(libraries: Libraries, record: LabelRecord, builder: str | None) -> str | None:
    if builder != SETUP_DEFAULT:
        return builder
    return target_builder(libraries, Dataset(libraries, record.dataset).setup)


def cache_paths(libraries: Libraries, sha256: str, builder: str | None) -> tuple[Path, Path]:
    directory = libraries.personal / CACHE_DIR / (NO_BUILDER if builder is None else ref_filename(builder))
    return directory / f"{sha256}.npz", directory / f"{sha256}.json"


def cached_meta(libraries: Libraries, record: LabelRecord, builder: str | None = SETUP_DEFAULT) -> dict[str, Any] | None:
    """The meta of a label's built targets, or ``None`` when they are not built yet (with this builder)."""

    return read_json(cache_paths(libraries, record.sha256, _builder(libraries, record, builder))[1])


def load_targets(
    libraries: Libraries, record: LabelRecord, builder: str | None = SETUP_DEFAULT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    """The meta and arrays of a label's targets (``centerline_xy``, ``width_profile``, ``ap``, ``overlap``,
    ``head_xy``, ``tail_xy``, ``diameter_px``; none for a label without a worm), or ``None`` when not built."""

    arrays_path, meta_path = cache_paths(libraries, record.sha256, _builder(libraries, record, builder))
    meta = read_json(meta_path)
    if meta is None:
        return None
    with np.load(arrays_path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return meta, arrays


def write_targets(libraries: Libraries, sha256: str, builder: str | None, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """Store targets; the arrays first, so a meta file always has its arrays."""

    arrays_path, meta_path = cache_paths(libraries, sha256, builder)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = arrays_path.with_name(f".{arrays_path.name}.partial")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(arrays_path)
    write_json(meta_path, meta)


def describe_body(meta: dict[str, Any], centerline: np.ndarray, width_profile: np.ndarray, shape: tuple[int, int]) -> None:
    """Add what listings and the length reference need: self-contact, body length, and whether it is all in view."""

    from ..body_targets import self_contact

    height, width = shape
    meta["self_contact"] = self_contact(centerline, width_profile)
    meta["body_length_px"] = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    meta["in_view"] = bool(np.all((centerline[:, 0] >= 0) & (centerline[:, 0] <= width - 1)
                                  & (centerline[:, 1] >= 0) & (centerline[:, 1] <= height - 1)))


def recording_length(
    libraries: Libraries, dataset_ref: str, recording: str, builder: str | None, *, exclude: str | None = None,
) -> float | None:
    """Median length of the whole, well-fit, untraced bodies among a dataset's built labels of one recording.

    It sets the length of a body that leaves the camera
    (:func:`body_fields.mark_exits`, :func:`body_fields.extend_trace`), as
    :func:`body_fields.recording_length` does for the old store.  ``exclude``
    is the ``sha256`` of the label being built.
    """

    from ..body_fields import LENGTH_REFERENCE_IOU, TRACE_METHODS

    lengths = []
    for other in Dataset(libraries, dataset_ref).labels():
        if other.recording != recording or other.sha256 == exclude:
            continue
        meta = cached_meta(libraries, other, builder)
        if (meta and meta.get("has_body") and meta.get("in_view") and meta.get("fit_iou", 0.0) >= LENGTH_REFERENCE_IOU
                and meta.get("fit_method") not in TRACE_METHODS):
            lengths.append(float(meta["body_length_px"]))
    return float(np.median(lengths)) if lengths else None


def build_targets(
    libraries: Libraries, record: LabelRecord, *, builder: str | None = SETUP_DEFAULT, segmenter: Any = None,
    device: Any = None, force: bool = False,
) -> dict[str, Any]:
    """Fit and store a label's targets unless they are already built (with this builder); returns their meta.

    ``segmenter`` is the builder's mask model when the caller has it loaded
    (anything with ``predict_probability_batch``,
    :meth:`.runtime.LoadedModel.mask_model`); otherwise it is loaded here.
    """

    builder = _builder(libraries, record, builder)
    if not force:
        meta = cached_meta(libraries, record, builder)
        if meta is not None:
            return meta
    import torch

    from .. import body_fields
    from ..head_tracking import HeadTracking
    from ..mask_fit import default_width_template

    label = record.load()
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if segmenter is None and builder is not None:
        from .runtime import load_model

        segmenter = load_model(libraries, builder, device).mask_model()
    config = body_fields.fit_config()
    count = len(label.context)
    tracking = HeadTracking(
        xy=label.nose_xy.astype(np.float32), valid=label.nose_valid, confidence=np.full(count, np.nan, np.float32),
        frame_indices=np.arange(count) - count // 2 + record.frame, source_frame_ids=np.full(count, -1),
        source_timestamps=np.full(count, -1), provenance={"source": "label"},
    )
    mask = label.mask == 1
    built, arrays, targets = body_fields.fit_targets(
        mask, label.context, label.context_valid, tracking, config=config,
        template=default_width_template(config.n_points), device=device,
        length_px=lambda: recording_length(libraries, record.dataset, record.recording, builder, exclude=record.sha256),
        trace=None if label.trace_xy is None else (label.trace_xy, False),
        head_xy=label.head_xy, segmenter=segmenter,
    )
    meta = {
        **built, "label": record.identity, "builder": builder, "built_at": utc_now(), "fit_preset": body_fields.FIT_PRESET,
        "max_lag": label.max_lag,
    }
    if targets is not None:
        describe_body(meta, arrays["centerline_xy"], arrays["width_profile"], mask.shape)
    write_targets(libraries, record.sha256, builder, meta, arrays)
    return meta


# --------------------------------------------------------------------------- job


def targets_command(libraries: Libraries, identities: Sequence[dict[str, Any]]) -> list[str]:
    """The argv of a job that builds the targets of label revisions (``LabelRecord.identity`` dicts)."""

    import json
    import sys

    command = [sys.executable, "-m", "worm_pose_gen.library.targets", "--personal", str(libraries.personal),
               "--labels", json.dumps([{k: i[k] for k in ("dataset", "recording", "frame", "revision")} for i in identities])]
    if libraries.lab is not None:
        command += ["--lab", str(libraries.lab)]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Build the targets of the labels named on the command line with each label's setup's builder."""

    import json

    from ..jobs import report_progress

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--personal", type=Path, required=True)
    parser.add_argument("--lab", type=Path, default=None)
    parser.add_argument("--labels", required=True, help="JSON list of {dataset, recording, frame, revision}")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    libraries = Libraries(lab=args.lab, personal=args.personal)
    wanted = json.loads(args.labels)
    models: dict[str | None, Any] = {}
    built = []
    for k, identity in enumerate(wanted):
        record = Dataset(libraries, identity["dataset"]).get(identity["recording"], int(identity["frame"]), int(identity["revision"]))
        builder = _builder(libraries, record, SETUP_DEFAULT)
        report_progress(k / len(wanted), f"building body targets of {record.key}")
        if builder is not None and builder not in models and (args.force or cached_meta(libraries, record, builder) is None):
            from .runtime import load_model

            models[builder] = load_model(libraries, builder, args.device).mask_model()
        meta = build_targets(libraries, record, builder=builder, segmenter=models.get(builder), device=args.device, force=args.force)
        built.append({**record.identity, "builder": builder, "fit_iou": meta.get("fit_iou"), "has_body": meta.get("has_body")})
    report_progress(1.0, f"built the body targets of {len(built)} label(s)", result={"targets": built})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
