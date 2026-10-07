"""Training a library model: prepare body targets, train, evaluate on every benchmark, write the model card.

One training run is one job (the Train tab, ``scripts/train.py``; see
docs/APP_SIMPLIFICATION.md, section 4):

1. **Labels.**  The run reads the chosen datasets' ``train`` and ``val``
   labels resolved together (:func:`library.labels`: a dataset brings the
   labels of the dataset it extends, and its own label of a frame overrides
   the inherited one).  Splits are per recording, so validation is on
   recordings the model never trains on.
2. **Prepare targets.**  A body-field model needs every label's body
   targets; missing ones are built first (:func:`model_eval.prepare_targets`).
   A label trains the body outputs only where :func:`model_eval.body_used`
   says so (not mask-only, and settled by a person or fit at
   ``min_fit_iou``); otherwise it trains the mask alone, and its heatmaps
   say nothing either way.
3. **Train** with Lightning: augmentation, the loss, early stopping on the
   validation loss and the plateau schedule are the modules' own
   (:class:`segmenter.SegmentationModule`, :class:`body_net.BodyFieldModule`).
   The model is the one checkpoint with the lowest validation loss; no
   last-epoch checkpoint is kept.
4. **Evaluate** the model on every benchmark of the setup
   (:func:`model_eval.evaluate`).
5. **Write the card** into the personal library (:func:`library.create_model`):
   inputs with the lags in frames and seconds at the setup's frame rate,
   outputs, ``trained_on`` (the exact label revisions by dataset, with
   counts), the parent, every hyperparameter and the notes; ``training/``
   holds ``metrics.csv`` (the loss curves), ``hparams.yaml``,
   ``labels.json`` (the revisions used, by split) and ``run.json``.

**What is trained.**  Starting from a body-field model or from scratch
trains a body-field model (all five outputs; from scratch it starts from
ImageNet weights with no lags or the short lags ±1, 4, 16 frames).
Starting from a mask-only segmenter fine-tunes the segmenter: it stays
until a body-field model matches its mask IoU (section 4, decision 1).  A
fine-tune keeps its parent's lags.

Progress goes to the job file (:func:`jobs.report_progress`): the phase,
the epoch, and the loss curve so far, which the Models tab draws live.
``python -m worm_pose_gen.model_training`` is the command a ``train`` job
runs, and ``scripts/train.py`` is the same command line.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

from . import library
from .jobs import report_progress
from .library import Libraries, LabelRecord
from .library.roots import check_id, parse_ref
from .model_eval import add_library_arguments, body_used, libraries_from, prepare_targets
from .workspace import utc_now


KINDS = ("segmenter", "body_net")
# Temporal context of a model trained from scratch.
CONTEXTS = {"none": (), "short": (1, 4, 16)}
RUNS_DIR = ".training"

# Every hyperparameter the Train tab and the command line take.  ``default``
# is per kind where the kinds differ; ``advanced`` ones are collapsed in the
# form; ``kinds`` limits a parameter to the kinds it applies to.
PARAMETERS: tuple[dict[str, Any], ...] = (
    {"name": "max_epochs", "label": "Max epochs", "type": "int", "default": 300, "min": 1, "max": 10000, "advanced": False,
     "help": "a cap: early stopping normally ends the run first"},
    {"name": "learning_rate", "label": "Learning rate", "type": "float", "default": 3e-4, "min": 1e-8, "max": 1.0, "advanced": False},
    {"name": "batch_size", "label": "Batch size", "type": "int", "default": 4, "min": 1, "max": 128, "advanced": False},
    {"name": "crop_size", "label": "Crop size (px)", "type": "int", "default": 512, "min": 64, "max": 4096, "advanced": True},
    {"name": "patience", "label": "Patience (epochs)", "type": "int", "default": {"segmenter": 5, "body_net": 15}, "min": 1, "max": 1000,
     "advanced": True, "help": "stop after this many epochs without a lower validation loss"},
    {"name": "plateau_patience", "label": "Plateau patience (epochs)", "type": "int", "default": {"segmenter": 2, "body_net": 4},
     "min": 1, "max": 1000, "advanced": True, "help": "halve the learning rate after this many stalled epochs"},
    {"name": "encoder_lr_scale", "label": "Encoder LR scale", "type": "float", "default": 0.25, "min": 0.0, "max": 10.0, "advanced": True},
    {"name": "ap_weight", "label": "A-P loss weight", "type": "float", "default": 1.0, "min": 0.0, "max": 100.0, "advanced": True,
     "kinds": ["body_net"]},
    {"name": "heatmap_weight", "label": "Head/tail loss weight", "type": "float", "default": 0.1, "min": 0.0, "max": 100.0,
     "advanced": True, "kinds": ["body_net"]},
    {"name": "overlap_weight", "label": "Overlap loss weight", "type": "float", "default": 1.0, "min": 0.0, "max": 100.0,
     "advanced": True, "kinds": ["body_net"]},
    {"name": "min_fit_iou", "label": "Minimum body fit IoU", "type": "float", "default": 0.9, "min": 0.0, "max": 1.0, "advanced": True,
     "kinds": ["body_net"], "help": "unreviewed labels whose body fit is worse train the mask only"},
    {"name": "seed", "label": "Seed", "type": "int", "default": 0, "min": 0, "max": 2**31 - 1, "advanced": True},
)

Progress = Callable[..., None]


def default_value(parameter: dict[str, Any], kind: str) -> Any:
    value = parameter["default"]
    return value[kind] if isinstance(value, dict) else value


def parameters_for(kind: str) -> list[dict[str, Any]]:
    return [p for p in PARAMETERS if kind in p.get("kinds", KINDS)]


def validate_params(kind: str, given: dict[str, Any]) -> dict[str, Any]:
    """The kind's parameters with defaults filled in; unknown or out-of-range values are refused."""

    known = {p["name"] for p in PARAMETERS}
    unknown = sorted(set(given) - known)
    if unknown:
        raise ValueError(f"unknown training parameters: {unknown}")
    result = {}
    for p in parameters_for(kind):
        value = given.get(p["name"])
        if value is None or value == "":
            value = default_value(p, kind)
        try:
            number = float(value)
            if isinstance(value, bool) or not math.isfinite(number) or (p["type"] == "int" and not number.is_integer()):
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"{p['label']} must be a{'n integer' if p['type'] == 'int' else ' number'}") from error
        value = int(number) if p["type"] == "int" else number
        if not p["min"] <= value <= p["max"]:
            raise ValueError(f"{p['label']} must lie in {p['min']}..{p['max']}")
        result[p["name"]] = value
    return result


# --------------------------------------------------------------------------- the request and its plan


@dataclass
class TrainRequest:
    """What the Train tab (or the command line) asks for."""

    setup: str
    datasets: list[str]
    name: str = ""
    start_from: str | None = None
    # From scratch only: "none" or "short" (CONTEXTS).
    context: str = "none"
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainRequest":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Plan:
    """A checked request: what kind of model it trains, on which labels, with which settings."""

    request: TrainRequest
    kind: str
    lags: tuple[int, ...]
    params: dict[str, Any]
    train: list[LabelRecord]
    val: list[LabelRecord]
    parent_kind: str | None

    def summary(self) -> dict[str, Any]:
        records = self.train + self.val
        return {
            "kind": self.kind, "lags": list(self.lags), "params": self.params, "name": self.request.name,
            "train": len(self.train), "val": len(self.val), "recordings": len({r.recording for r in records}),
            "mask_only": sum(r.status == "mask_only" for r in records),
        }


def own_dataset_id(libraries: Libraries, datasets: Sequence[str]) -> str:
    """The dataset a generated name starts with: the first personal one, else the first."""

    for ref in datasets:
        if parse_ref(ref)[0] == "mine":
            return parse_ref(ref)[1]
    return parse_ref(datasets[0])[1]


def model_exists(libraries: Libraries, model_id: str) -> bool:
    return (libraries.personal / library.models.MODELS_DIR / model_id).exists()


def suggest_name(libraries: Libraries, datasets: Sequence[str], train_labels: int, taken: Sequence[str] = ()) -> str:
    """``<dataset>-<training labels>``, with ``-2``, ``-3``... when that model exists or a queued run claims it."""

    base = f"{own_dataset_id(libraries, datasets)}-{train_labels}"
    name, n = base, 1
    while model_exists(libraries, name) or name in taken:
        n += 1
        name = f"{base}-{n}"
    return name


def plan(libraries: Libraries, request: TrainRequest, *, taken: Sequence[str] = ()) -> Plan:
    """Check a request against the libraries; ``taken`` are names queued runs will write."""

    setup = library.get_setup(libraries, request.setup)
    if not request.datasets:
        raise ValueError("choose at least one dataset to train on")
    for ref in request.datasets:
        dataset = library.Dataset(libraries, ref)
        if dataset.setup != setup.ref:
            raise ValueError(f"{ref} is a dataset of {dataset.setup}, not of {setup.ref}")
    parent_kind = None
    if request.start_from:
        parent = library.get_card(libraries, request.start_from)
        parent_kind = parent.kind
        kind = parent.kind
        from .library.inference import lags_at

        lags = lags_at(parent, setup.fps)
    else:
        if request.context not in CONTEXTS:
            raise ValueError(f"unknown temporal context {request.context!r}; expected one of {tuple(CONTEXTS)}")
        kind, lags = "body_net", CONTEXTS[request.context]
    params = validate_params(kind, request.params)
    train = library.labels(libraries, request.datasets, "train")
    val = library.labels(libraries, request.datasets, "val")
    if not train:
        raise ValueError("the chosen datasets have no training labels")
    if not val:
        raise ValueError("the chosen datasets have no validation labels (no recording is in the val split)")
    request.name = request.name.strip() if request.name else suggest_name(libraries, request.datasets, len(train), taken)
    check_id(request.name)
    if model_exists(libraries, request.name) or request.name in taken:
        raise ValueError(f"a model named {request.name} already exists; choose another name")
    return Plan(request=request, kind=kind, lags=tuple(lags), params=params, train=train, val=val, parent_kind=parent_kind)


# --------------------------------------------------------------------------- data


class LabelDataset:
    """Library labels as network inputs and targets, for either kind (a torch ``Dataset``).

    A segmenter item is ``{"image": [1,H,W], "mask", "valid": [H,W]}``; a
    body-field item is ``{"image": [1+lags,H,W], "targets": [8,H,W]}`` in
    the row layout :class:`body_net.BodyFieldModule` reads.  Both carry the
    label's ``key``.  Augmentation (crops biased toward the worm, flips,
    rotation, gain/offset, noise) is the one both trainers have always used.
    """

    def __init__(
        self, libraries: Libraries, records: Sequence[LabelRecord], *, kind: str, lags: Sequence[int] = (), augment: bool = False,
        crop_size: int | None = None, seed: int = 0, min_fit_iou: float = 0.9,
    ) -> None:
        self.libraries = libraries
        self.records = list(records)
        self.kind = kind
        self.lags = tuple(int(lag) for lag in lags)
        self.augment = augment
        self.crop_size = crop_size
        self.seed = seed
        self.min_fit_iou = min_fit_iou

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        from .body_net import TARGET_ROWS, _augment, body_target_rows
        from .segmenter import IGNORE_LABEL, INPUT_MEAN, INPUT_STD
        from .temporal_context import difference_channels

        record = self.records[index]
        label = record.load()
        mask = label.mask
        centre = label.max_lag
        if self.lags and max(self.lags) > centre:
            raise ValueError(f"{record.key} stores ±{centre} frames of context; the model needs ±{max(self.lags)}")
        if self.kind == "segmenter":
            needed = [centre]
            targets = np.stack(((mask == 1), (mask != IGNORE_LABEL))).astype(np.float32)
        else:
            needed = sorted({centre} | {centre + lag for lag in self.lags} | {centre - lag for lag in self.lags})
            built = library.load_targets(self.libraries, record)
            if built is None:
                raise FileNotFoundError(f"{record.key}: body targets are not built")
            arrays = built[1] if body_used(record, built[0], self.min_fit_iou) else None
            targets = body_target_rows(mask, arrays)
        frames = label.context[needed].astype(np.float32)
        if self.augment:
            rng = np.random.default_rng((self.seed, index, int(torch.initial_seed()) % (1 << 31)))
            frames, targets = _augment(frames, targets, rng, self.crop_size)
        position = {frame: k for k, frame in enumerate(needed)}
        image = ((frames[position[centre]] / 255.0 - INPUT_MEAN) / INPUT_STD)[None]
        item: dict[str, Any] = {"key": record.key}
        if self.kind == "segmenter":
            item.update(image=torch.as_tensor(image), mask=torch.as_tensor(targets[0]), valid=torch.as_tensor(targets[1]))
            return item
        stack = np.zeros((2 * centre + 1, *frames.shape[1:]), dtype=np.float32)
        for frame, k in position.items():
            stack[frame] = frames[k]
        inputs = np.concatenate((image, difference_channels(stack, label.context_valid, self.lags)), axis=0)
        assert targets.shape[0] == TARGET_ROWS
        item.update(image=torch.as_tensor(inputs), targets=torch.as_tensor(targets))
        return item


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad every array to a common multiple of 32 (at least 64, so the deepest batch norm has pixels); padding is invalid for every loss."""

    import torch
    import torch.nn.functional as F

    height = max(64, max(s["image"].shape[-2] for s in samples))
    width = max(64, max(s["image"].shape[-1] for s in samples))
    height, width = ((height + 31) // 32) * 32, ((width + 31) // 32) * 32
    batch: dict[str, Any] = {"key": [s["key"] for s in samples]}
    for name in samples[0]:
        if name != "key":
            batch[name] = torch.stack([F.pad(s[name], (0, width - s[name].shape[-1], 0, height - s[name].shape[-2])) for s in samples])
    return batch


# --------------------------------------------------------------------------- training


def build_module(libraries: Libraries, plan_: Plan) -> Any:
    """The network to train: the parent's weights with this run's settings, or ImageNet weights."""

    p = plan_.params
    common = dict(learning_rate=p["learning_rate"], encoder_learning_rate_scale=p["encoder_lr_scale"], plateau_patience=p["plateau_patience"])
    if plan_.kind == "segmenter":
        from .segmenter import SegmentationModule

        if plan_.request.start_from:
            return SegmentationModule.load_from_checkpoint(
                str(library.weights_path(libraries, plan_.request.start_from)), map_location="cpu", pretrained=False, **common)
        return SegmentationModule(pretrained=True, **common)
    from .body_net import BodyFieldModule

    body = dict(ap_weight=p["ap_weight"], heatmap_weight=p["heatmap_weight"], overlap_weight=p["overlap_weight"], **common)
    if plan_.request.start_from:
        return BodyFieldModule.load_from_checkpoint(
            str(library.weights_path(libraries, plan_.request.start_from)), map_location="cpu", pretrained=False,
            lags=plan_.lags, **body)
    return BodyFieldModule(lags=plan_.lags, pretrained=True, **body)


def read_curve(metrics_csv: Path) -> list[dict[str, Any]]:
    """Per-epoch ``train_loss`` and ``val_loss`` from Lightning's ``metrics.csv`` (the Details view's curves)."""

    epochs: dict[int, dict[str, Any]] = {}
    try:
        with open(metrics_csv, newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("epoch"):
                    continue
                entry = epochs.setdefault(int(float(row["epoch"])), {"epoch": int(float(row["epoch"])) + 1, "train_loss": None, "val_loss": None})
                for column, name in (("train_loss_epoch", "train_loss"), ("val_loss", "val_loss")):
                    if row.get(column):
                        entry[name] = float(row[column])
    except FileNotFoundError:
        return []
    return [epochs[k] for k in sorted(epochs)]


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _quiet(fraction: float, message: str, result: dict[str, Any] | None = None) -> None:
    pass


def train(
    libraries: Libraries, request: TrainRequest, *, device: str | None = None, num_workers: int | None = None,
    progress: Progress = _quiet,
) -> library.ModelCard:
    """Run the whole job (see the module docstring) and return the new model's card."""

    import lightning as L
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    import torch
    from torch.utils.data import DataLoader

    from .model_eval import evaluate

    started = utc_now()
    plan_ = plan(libraries, request)
    p = plan_.params
    name = plan_.request.name
    setup = library.get_setup(libraries, request.setup)
    use_cuda = device != "cpu" and torch.cuda.is_available()
    torch_device = torch.device(device) if device else torch.device("cuda" if use_cuda else "cpu")
    state: dict[str, Any] = {"phase": "prepare", "model_id": name, "kind": plan_.kind, "epoch": 0, "max_epochs": p["max_epochs"], "curve": []}

    def report(fraction: float, message: str) -> None:
        progress(fraction, message, dict(state))

    prepared = 0
    if plan_.kind == "body_net":
        prepared = prepare_targets(libraries, plan_.train + plan_.val, setup=request.setup, device=torch_device,
                                   progress=lambda f, m: report(0.1 * f, m))
    run_dir = libraries.personal / RUNS_DIR / name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    try:
        L.seed_everything(p["seed"], workers=True)
        torch.set_float32_matmul_precision("high")
        workers = (min(4, os.cpu_count() or 1) if use_cuda else 0) if num_workers is None else int(num_workers)

        def loader(records: list[LabelRecord], training: bool) -> DataLoader:
            dataset = LabelDataset(libraries, records, kind=plan_.kind, lags=plan_.lags, augment=training,
                                   crop_size=p["crop_size"] if training else None, seed=p["seed"],
                                   min_fit_iou=p.get("min_fit_iou", 0.9))
            return DataLoader(dataset, batch_size=p["batch_size"] if training else max(1, p["batch_size"] // 2), shuffle=training,
                              num_workers=workers, persistent_workers=workers > 0, pin_memory=use_cuda, collate_fn=collate)

        module = build_module(libraries, plan_)

        class Progress(Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                steps = max(1, int(trainer.num_training_batches))
                done = (trainer.current_epoch + (batch_idx + 1) / steps) / p["max_epochs"]
                report(0.1 + 0.8 * min(done, 1.0), f"epoch {trainer.current_epoch + 1}, batch {batch_idx + 1}/{steps}")

            def on_train_epoch_end(self, trainer, pl_module):
                metrics = trainer.callback_metrics
                value = lambda key: float(metrics[key]) if key in metrics else None
                state["epoch"] = trainer.current_epoch + 1
                state["curve"].append({"epoch": state["epoch"], "train_loss": value("train_loss_epoch"), "val_loss": value("val_loss")})
                losses = [c["val_loss"] for c in state["curve"] if c["val_loss"] is not None]
                state["best_val_loss"] = min(losses) if losses else None
                report(0.1 + 0.8 * min(state["epoch"] / p["max_epochs"], 1.0), f"epoch {state['epoch']} done")

        best = ModelCheckpoint(dirpath=run_dir, filename="best", monitor="val_loss", mode="min", save_top_k=1,
                               save_last=False, enable_version_counter=False)
        state["phase"] = "train"
        report(0.1, "training")
        trainer = L.Trainer(
            max_epochs=p["max_epochs"], accelerator="gpu" if use_cuda else "cpu", devices=1,
            precision="16-mixed" if use_cuda else "32-true",
            callbacks=[best, EarlyStopping(monitor="val_loss", mode="min", patience=p["patience"]), Progress()],
            logger=CSVLogger(str(run_dir), name="", version=""), default_root_dir=str(run_dir),
            log_every_n_steps=1, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloaders=loader(plan_.train, True), val_dataloaders=loader(plan_.val, False))
        if not best.best_model_path:
            raise RuntimeError("training produced no validated checkpoint")
        checkpoint = Path(best.best_model_path)
        best_epoch = int(torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"]) + 1
        records = plan_.train + plan_.val
        used = sum(
            body_used(r, library.cached_meta(libraries, r), p.get("min_fit_iou", 0.9)) for r in records
        ) if plan_.kind == "body_net" else 0
        preprocessing = "flat_field" if setup.video.get("flat_field", True) else "none"
        card = {
            "name": name, "kind": plan_.kind, "setup": setup.ref,
            "inputs": library.make_inputs(plan_.lags, fps=setup.fps, pixel_size_um=setup.pixel_size_um, preprocessing=preprocessing),
            "outputs": ["mask"] if plan_.kind == "segmenter" else list(library.models.OUTPUTS),
            "trained_on": library.trained_on(records), "parent": request.start_from or None,
            "hparams": {**p, "lags": list(plan_.lags), "context": None if request.start_from else request.context},
            "notes": request.notes,
        }
        run = {
            "request": asdict(plan_.request), "kind": plan_.kind, "started_at": started, "finished_at": utc_now(),
            "epochs_run": state["epoch"], "best_epoch": best_epoch,
            "best_val_loss": None if best.best_model_score is None else float(best.best_model_score),
            "device": str(torch_device), "git": _git_commit(), "targets_built": prepared,
            "labels": {"train": len(plan_.train), "val": len(plan_.val), "body_trained": used},
        }
        files = {n: run_dir / n for n in ("metrics.csv", "hparams.yaml") if (run_dir / n).exists()}
        created = library.create_model(
            libraries, name, card, checkpoint, training_files=files,
            training_records={"labels.json": {"train": [r.identity for r in plan_.train], "val": [r.identity for r in plan_.val]},
                              "run.json": run},
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    state.update(phase="evaluate", model=created.ref)
    benchmarks = library.list_benchmarks(libraries, setup.ref)
    for number, benchmark in enumerate(benchmarks):
        evaluate(libraries, created.ref, benchmark.ref, device=torch_device,
                 progress=lambda f, m: report(0.9 + 0.1 * (number + f) / len(benchmarks), f"{benchmark.ref}: {m}"))
    state["phase"] = "done"
    progress(1.0, f"{created.ref} is ready", dict(state))
    return created


# --------------------------------------------------------------------------- command line


def command(libraries: Libraries, request: TrainRequest, plan_: Plan) -> list[str]:
    """The process a ``train`` job runs: this module's command line for a checked request."""

    argv = [sys.executable, "-m", "worm_pose_gen.model_training", "--library", str(libraries.personal),
            "--setup", request.setup, "--name", plan_.request.name]
    if libraries.lab is not None:
        argv += ["--lab-library", str(libraries.lab)]
    for ref in request.datasets:
        argv += ["--dataset", ref]
    if request.start_from:
        argv += ["--start-from", request.start_from]
    else:
        argv += ["--context", request.context]
    for key, value in plan_.params.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    if request.notes:
        argv += ["--notes", request.notes]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_library_arguments(parser)
    parser.add_argument("--setup", required=True, help="setup reference, e.g. lab:nir-flv")
    parser.add_argument("--dataset", action="append", required=True, help="dataset reference to train on (repeatable)")
    parser.add_argument("--name", default="", help="model id (default: <dataset>-<training labels>)")
    parser.add_argument("--start-from", default=None, help="model to fine-tune (default: from scratch, a body-field model)")
    parser.add_argument("--context", choices=tuple(CONTEXTS), default="none", help="temporal context from scratch: none or short (±1, 4, 16 frames)")
    parser.add_argument("--notes", default="")
    parser.add_argument("--num-workers", type=int, default=None, help="data loader workers (default: 4 on a GPU, 0 on the CPU)")
    for p in PARAMETERS:
        parser.add_argument(f"--{p['name'].replace('_', '-')}", dest=p["name"], type=int if p["type"] == "int" else float,
                            default=None, help=f"{p['label']} (default {p['default']})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    request = TrainRequest(
        setup=args.setup, datasets=list(args.dataset), name=args.name, start_from=args.start_from, context=args.context,
        params={p["name"]: getattr(args, p["name"]) for p in PARAMETERS if getattr(args, p["name"]) is not None}, notes=args.notes,
    )
    started = time.monotonic()

    def progress(fraction: float, message: str, result: dict[str, Any] | None = None) -> None:
        report_progress(fraction, message, result)

    card = train(libraries_from(args), request, device=args.device, num_workers=args.num_workers, progress=progress)
    print(json.dumps({"model": card.ref, "minutes": round((time.monotonic() - started) / 60, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
