"""Model cards, weights, training records and evaluations.

``models/<id>/`` holds::

    model.json     the card (below)
    weights.ckpt   the one checkpoint with the lowest validation loss
    training/      what the run wrote: hyperparameters, metrics.csv, the exact label revisions used
    evaluations/<scope>.<benchmark>.json   scores on each benchmark

The card::

    {"name": "nir-hand284", "kind": "segmenter" | "body_net", "setup": "lab:nir-flv",
     "inputs": {"preprocessing": "flat_field", "lags_frames": [1, 4, 16], "lags_s": [0.05, 0.2, 0.8],
                "fps": 20.0, "pixel_size_um": 1.0},
     "outputs": ["mask", "ap", "head", "tail", "overlap"],
     "trained_on": [{"dataset": ref, "fingerprint": ..., "counts": {...}, "recordings": n}],
     "parent": ref | null, "hparams": {...}, "author": ..., "created_at": ..., "notes": ...}

``kind`` says how to load the weights (:func:`segmenter.load_segmenter` or
:func:`body_net.load_body_net`).  Lags are stored in frames and in seconds,
so a setup with another frame rate converts them.

Evaluation files are named by the benchmark's reference
(``lab.nir-v1.json``), since a lab and a personal benchmark may share an
id.  The app writes every evaluation into the personal library: a personal
model's go in its own directory, and a lab model's (on any benchmark) in
``<personal>/evaluations/lab.<model>/``, because the lab library is
read-only.  Publishing a model copies its evaluations along.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import getpass
from pathlib import Path
import shutil
from typing import Any, Sequence

from .roots import Libraries, check_id, make_ref, parse_ref, read_json, ref_filename, write_json
from ..workspace import utc_now


MODELS_DIR = "models"
EVALUATIONS_DIR = "evaluations"
KINDS = ("segmenter", "body_net")
OUTPUTS = ("mask", "ap", "head", "tail", "overlap")
WEIGHTS = "weights.ckpt"


@dataclass(frozen=True)
class ModelCard:
    ref: str
    name: str
    kind: str
    setup: str
    inputs: dict[str, Any]
    outputs: tuple[str, ...]
    trained_on: tuple[dict[str, Any], ...] = ()
    parent: str | None = None
    hparams: dict[str, Any] = field(default_factory=dict)
    author: str = ""
    created_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "outputs": list(self.outputs), "trained_on": list(self.trained_on)}


def model_dir(root: Path, model_id: str) -> Path:
    return Path(root) / MODELS_DIR / check_id(model_id)


def lags_in_seconds(lags_frames: Sequence[int], fps: float | None) -> list[float] | None:
    return None if fps is None else [round(int(lag) / float(fps), 6) for lag in lags_frames]


def make_inputs(lags_frames: Sequence[int], *, fps: float | None, pixel_size_um: float | None, preprocessing: str = "flat_field") -> dict[str, Any]:
    lags = [int(lag) for lag in lags_frames]
    return {"preprocessing": preprocessing, "lags_frames": lags, "lags_s": lags_in_seconds(lags, fps), "fps": fps, "pixel_size_um": pixel_size_um}


def validate_card(card: dict[str, Any]) -> None:
    if card.get("kind") not in KINDS:
        raise ValueError(f"unknown model kind {card.get('kind')!r}; expected one of {KINDS}")
    outputs = list(card.get("outputs") or ())
    if not outputs or any(name not in OUTPUTS for name in outputs):
        raise ValueError(f"model outputs must be a non-empty subset of {OUTPUTS}")
    parse_ref(str(card.get("setup")))
    if card.get("parent") is not None:
        parse_ref(card["parent"])
    for key in ("name", "inputs"):
        if not card.get(key):
            raise ValueError(f"a model card needs {key!r}")


def write_model(
    root: Path, model_id: str, card: dict[str, Any], weights: Path, *, training_files: dict[str, Path] | None = None,
    training_records: dict[str, Any] | None = None,
) -> None:
    """Write a model into a library root: its card, a copy of ``weights``, and its training files.

    ``training_files`` maps names under ``training/`` to files to copy;
    ``training_records`` maps names to JSON values to write there.  An
    existing model is never overwritten.
    """

    card = {"author": getpass.getuser(), "created_at": utc_now(), "trained_on": [], "parent": None, "hparams": {}, "notes": "", **card}
    validate_card(card)
    directory = model_dir(root, model_id)
    if directory.exists():
        raise FileExistsError(f"model {model_id} already exists in {root}")
    staging = directory.with_name(f".{model_id}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "training").mkdir(parents=True)
    shutil.copy2(weights, staging / WEIGHTS)
    for name, source in (training_files or {}).items():
        shutil.copy2(source, staging / "training" / name)
    for name, value in (training_records or {}).items():
        write_json(staging / "training" / name, value)
    write_json(staging / "model.json", card)
    staging.rename(directory)


def create_model(libraries: Libraries, model_id: str, card: dict[str, Any], weights: Path, **files: Any) -> ModelCard:
    """A new personal model (the trainer's last step)."""

    write_model(libraries.personal, model_id, card, weights, **files)
    return get_card(libraries, make_ref("mine", model_id))


def get_card(libraries: Libraries, ref: str) -> ModelCard:
    scope, model_id = parse_ref(ref)
    data = read_json(model_dir(libraries.root(scope), model_id) / "model.json")
    if data is None:
        raise LookupError(f"unknown model {ref}")
    return ModelCard(
        ref=ref, name=str(data["name"]), kind=str(data["kind"]), setup=str(data["setup"]), inputs=dict(data["inputs"]),
        outputs=tuple(data["outputs"]), trained_on=tuple(data.get("trained_on") or ()), parent=data.get("parent"),
        hparams=dict(data.get("hparams") or {}), author=str(data.get("author") or ""),
        created_at=str(data.get("created_at") or ""), notes=str(data.get("notes") or ""),
    )


def weights_path(libraries: Libraries, ref: str) -> Path:
    scope, model_id = parse_ref(ref)
    path = model_dir(libraries.root(scope), model_id) / WEIGHTS
    if not path.is_file():
        raise LookupError(f"model {ref} has no weights")
    return path


def training_dir(libraries: Libraries, ref: str) -> Path:
    scope, model_id = parse_ref(ref)
    return model_dir(libraries.root(scope), model_id) / "training"


def list_models(libraries: Libraries, setup: str | None = None) -> list[ModelCard]:
    found = []
    for scope in libraries.scopes():
        directory = libraries.root(scope) / MODELS_DIR
        for path in sorted(directory.glob("*/model.json")) if directory.is_dir() else ():
            card = get_card(libraries, make_ref(scope, path.parent.name))
            if setup is None or card.setup == setup:
                found.append(card)
    return found


# --------------------------------------------------------------------------- evaluations


def evaluation_dirs(libraries: Libraries, model_ref: str) -> list[Path]:
    """Where a model's evaluations are read from; the last is where the app writes new ones."""

    scope, model_id = parse_ref(model_ref)
    if scope == "mine":
        return [model_dir(libraries.personal, model_id) / EVALUATIONS_DIR]
    return [model_dir(libraries.root("lab"), model_id) / EVALUATIONS_DIR, libraries.personal / EVALUATIONS_DIR / ref_filename(model_ref)]


def evaluation_path(libraries: Libraries, model_ref: str, benchmark_ref: str) -> Path:
    """Where the app stores a model's evaluation on a benchmark (always in the personal library)."""

    return evaluation_dirs(libraries, model_ref)[-1] / f"{ref_filename(benchmark_ref)}.json"


def save_evaluation(libraries: Libraries, model_ref: str, benchmark_ref: str, result: dict[str, Any]) -> Path:
    """Store an evaluation (metrics such as ``iou_mean``, ``iou_worst5``, ``head_tail_correct``, ``ap_error``)."""

    get_card(libraries, model_ref)
    path = evaluation_path(libraries, model_ref, benchmark_ref)
    write_json(path, {"model": model_ref, "benchmark": benchmark_ref, "evaluated_at": utc_now(), **result})
    return path


def evaluations(libraries: Libraries, model_ref: str) -> dict[str, dict[str, Any]]:
    """Benchmark ref -> evaluation, a personal evaluation of a lab model replacing the lab's."""

    result: dict[str, dict[str, Any]] = {}
    for directory in evaluation_dirs(libraries, model_ref):
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
            data = read_json(path)
            result[str(data.get("benchmark") or path.stem.replace(".", ":", 1))] = data
    return result
