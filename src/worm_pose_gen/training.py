"""Reproducible app fine-tuning jobs; initialization is always explicit worm weights.

Submission freezes corpus bytes and initial checkpoint before queueing. The
worker reads that snapshot only and never promotes or overwrites a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import uuid
from typing import Any

from .corpus import CorpusStore
from .jobs import JobSpec, report_progress
from .workspace import _write_json_atomic, utc_now

PARAMETERS = [
    {"name": "epochs", "type": "int", "default": 30, "min": 1, "max": 10000},
    {"name": "batch_size", "type": "int", "default": 4, "min": 1, "max": 128},
    {"name": "crop_size", "type": "int", "default": 512, "min": 64, "max": 4096},
    {"name": "learning_rate", "type": "float", "default": 0.0003, "min": 0.00000001, "max": 1.0},
    {"name": "patience", "type": "int", "default": 5, "min": 1, "max": 1000},
    {"name": "num_workers", "type": "int", "default": 0, "min": 0, "max": 32},
    {"name": "seed", "type": "int", "default": 0, "min": 0, "max": 2147483647},
    {"name": "device", "type": "choice", "default": "auto", "choices": ["auto", "cpu", "cuda"]},
]


def training_schema() -> dict[str, Any]:
    return {"params": PARAMETERS, "defaults": {p["name"]: p["default"] for p in PARAMETERS},
            "initialization": "configured worm checkpoint", "promotion": "explicit selection only",
            "requires": ["train", "val"]}


def validate_params(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = set(payload) - {p["name"] for p in PARAMETERS}
    if unknown:
        raise ValueError(f"unknown fine-tuning parameters: {sorted(unknown)}")
    result = {}
    for p in PARAMETERS:
        name, value = p["name"], payload.get(p["name"], p["default"])
        if p["type"] == "choice":
            if value not in p["choices"]:
                raise ValueError(f"{name} must be one of {p['choices']}")
        else:
            try:
                numeric = float(value)
                if isinstance(value, bool) or not math.isfinite(numeric):
                    raise ValueError()
                if p["type"] == "int" and not numeric.is_integer():
                    raise ValueError()
                value = int(numeric) if p["type"] == "int" else numeric
                if not p["min"] <= value <= p["max"]:
                    raise ValueError()
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be {p['type']} in {p['min']}..{p['max']}") from exc
        result[name] = value
    return result


def list_checkpoints(config) -> list[dict[str, Any]]:
    rows = []
    if config.checkpoint is not None:
        base = Path(config.checkpoint).expanduser().resolve()
        rows.append({"id": "base", "path": str(base), "label": "Configured base checkpoint", "base": True,
                     "exists": base.is_file()})
    for file in sorted(Path(config.checkpoints_root).glob("runs/*/run.json"), reverse=True):
        try:
            run = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        if run.get("state") != "completed":
            continue
        for name in ("best", "last"):
            path = file.parent / f"{name}.ckpt"
            if path.is_file():
                rows.append({"id": f"{file.parent.name}:{name}", "path": str(path.resolve()),
                             "label": f"{run.get('label', file.parent.name)} / {name}", "base": False,
                             "run_id": file.parent.name, "finished_at": run.get("finished_at"),
                             "best_val_loss": run.get("best_val_loss"), "exists": True})
    return rows


def resolve_checkpoint(config, value: str | None = None) -> Path:
    value = value or "base"
    if value == "base" and config.checkpoint is None:
        raise ValueError("configure a worm segmenter checkpoint before fine-tuning")
    for row in list_checkpoints(config):
        if value == row["id"]:
            value = row["path"]
            break
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {value}")
    return path


def fine_tune_job(app, payload: dict[str, Any]) -> tuple[JobSpec, list[str]]:
    params = validate_params(dict(payload.get("params") or {}))
    initial = resolve_checkpoint(app.config, payload.get("checkpoint"))
    run_id = utc_now().replace(":", "-") + "_" + uuid.uuid4().hex[:10]
    run_dir = Path(app.config.checkpoints_root).resolve() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        manifest = CorpusStore(app.config.corpus_root).snapshot(run_dir / "dataset")
        shutil.copy2(initial, run_dir / "init.ckpt")
        record = {"id": run_id, "state": "queued", "label": str(payload.get("label") or "Fine-tune segmenter"),
                  "created_at": utc_now(), "params": params, "dataset": manifest,
                  "init_checkpoint": {"path": str(initial), "sha256": hashlib.sha256((run_dir / "init.ckpt").read_bytes()).hexdigest()},
                  "promotion": False}
        _write_json_atomic(run_dir / "run.json", record)
    except Exception:
        shutil.rmtree(run_dir)
        raise
    spec = JobSpec(kind="fine_tune", params={"run_dir": str(run_dir), **params},
                   gpus=0 if params["device"] == "cpu" else 1, label=record["label"])
    return spec, [sys.executable, "-m", "worm_pose_gen.training", "--run-dir", str(run_dir)]


def run_training(run_dir: Path) -> dict[str, Any]:
    import lightning as L
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    import torch
    from .segmenter import SegmentationModule
    from .segmentation_dataset import SegmentationDataModule

    run_dir = Path(run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    if record["state"] != "queued":
        raise ValueError("a fine-tuning run can only start once")
    params = validate_params(record["params"])
    record.update(state="running", started_at=utc_now(), job_id=os.environ.get("WORM_POSE_JOB_ID"))
    _write_json_atomic(run_dir / "run.json", record)
    try:
        for sample in record["dataset"]["samples"]:
            path = run_dir / "dataset" / "samples" / (sample["sample_id"] + ".npz")
            if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
                raise ValueError(f"training snapshot changed: {sample['sample_id']}")
        if hashlib.sha256((run_dir / "init.ckpt").read_bytes()).hexdigest() != record["init_checkpoint"]["sha256"]:
            raise ValueError("initial checkpoint snapshot changed")
        L.seed_everything(params["seed"], workers=True)
        data = SegmentationDataModule(run_dir / "dataset", batch_size=params["batch_size"], crop_size=params["crop_size"],
                                      num_workers=params["num_workers"], seed=params["seed"], pad_batches=True)
        module = SegmentationModule.load_from_checkpoint(str(run_dir / "init.ckpt"), map_location="cpu",
                                                         pretrained=False, learning_rate=params["learning_rate"])
        class Progress(Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                steps = max(1, trainer.num_training_batches)
                report_progress(min(.99, (trainer.current_epoch + (batch_idx + 1) / steps) / params["epochs"]),
                                f"epoch {trainer.current_epoch + 1}/{params['epochs']}, batch {batch_idx + 1}/{int(steps)}")
        best = ModelCheckpoint(dirpath=run_dir, filename="best", monitor="val_loss", mode="min", save_top_k=1,
                               enable_version_counter=False)
        accelerator = {"cuda": "gpu", "cpu": "cpu", "auto": "auto"}[params["device"]]
        trainer = L.Trainer(max_epochs=params["epochs"], accelerator=accelerator, devices=1,
                            precision="32-true", logger=CSVLogger(str(run_dir), name="", version=""),
                            callbacks=[best, EarlyStopping(monitor="val_loss", patience=params["patience"]), Progress()],
                            default_root_dir=str(run_dir), enable_progress_bar=False, log_every_n_steps=1,
                            num_sanity_val_steps=0)
        trainer.fit(module, datamodule=data)
        trainer.save_checkpoint(run_dir / "last.ckpt")
        if not best.best_model_path:
            raise RuntimeError("training produced no validated checkpoint")
        if record["dataset"]["counts"]["test"]:
            record["test"] = trainer.test(module, datamodule=data, ckpt_path=best.best_model_path, verbose=False)
        record.update(state="completed", finished_at=utc_now(), epochs_run=trainer.current_epoch,
                      best_val_loss=None if best.best_model_score is None else float(best.best_model_score),
                      checkpoints={name: str((run_dir / f"{name}.ckpt").resolve()) for name in ("best", "last")})
        _write_json_atomic(run_dir / "run.json", record)
        report_progress(1, "fine-tuning complete; checkpoint ready to select", result={"run_dir": str(run_dir), "checkpoints": record["checkpoints"]})
        return record
    except BaseException as error:
        record.update(state="cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
        _write_json_atomic(run_dir / "run.json", record)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    def cancelled(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, cancelled)
    run_training(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
