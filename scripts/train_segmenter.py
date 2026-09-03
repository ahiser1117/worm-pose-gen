#!/usr/bin/env python3
"""Fine-tune the worm segmenter on the segmentation store with Lightning.

Checkpoints go to a git-ignored local directory (``checkpoints/segmenter`` by
default): ``best.ckpt`` tracks the highest validation IoU and ``last.ckpt``
the final epoch.  Pass ``--init`` with an earlier checkpoint to continue
fine-tuning after more labels are added.

Every run also leaves ``runs/<start time>.json`` in the checkpoint directory:
the arguments, git revision, the exact train/val/test membership, the
Lightning CSV log directory, the best checkpoint's fingerprint, and the
final metrics.  ``train_summary.json`` is a copy of the latest run's record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

from worm_pose_gen.run_records import checkpoint_fingerprint, git_revision, split_manifest, timestamp_slug, utc_now
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationDataModule
from worm_pose_gen.segmenter import SegmentationModule


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "segmenter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--init", type=Path, default=None, help="checkpoint to continue from")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    data = SegmentationDataModule(
        args.dataset_root, batch_size=args.batch_size, crop_size=args.crop_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    counts = data.store.counts()
    if counts["train"] == 0 or counts["val"] == 0:
        raise SystemExit(f"need train and val samples; store has {counts}")
    if args.init is not None:
        module = SegmentationModule.load_from_checkpoint(
            str(args.init), pretrained=False, learning_rate=args.learning_rate,
            encoder_learning_rate_scale=args.encoder_lr_scale,
        )
    else:
        module = SegmentationModule(
            pretrained=True, learning_rate=args.learning_rate,
            encoder_learning_rate_scale=args.encoder_lr_scale,
        )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = ModelCheckpoint(
        dirpath=args.checkpoint_dir, filename="best", monitor="val_iou", mode="max",
        save_top_k=1, save_last=True, enable_version_counter=False,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[best, EarlyStopping(monitor="val_iou", mode="max", patience=args.patience)],
        logger=CSVLogger(str(args.checkpoint_dir), name="logs"),
        default_root_dir=str(args.checkpoint_dir),
        log_every_n_steps=5,
        enable_progress_bar=True,
    )
    manifest = split_manifest(data.store)
    trainer.fit(module, datamodule=data)
    results = {
        "started_at": started_at,
        "finished_at": None,
        "git": git_revision(PROJECT_ROOT),
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "init_checkpoint": checkpoint_fingerprint(args.init),
        "dataset_root": str(args.dataset_root),
        "counts": counts,
        "splits": manifest,
        "log_dir": str(trainer.logger.log_dir) if trainer.logger is not None else None,
        "best_checkpoint": best.best_model_path,
        "best_val_iou": None if best.best_model_score is None else float(best.best_model_score),
        "last_checkpoint": best.last_model_path,
        "epochs_run": trainer.current_epoch,
        "stopped_early": trainer.current_epoch < args.epochs,
    }
    if counts["test"]:
        test = trainer.test(module, datamodule=data, ckpt_path=best.best_model_path or None, verbose=False)
        results["test"] = test[0] if test else None
    results["finished_at"] = utc_now()
    results["best_checkpoint_fingerprint"] = checkpoint_fingerprint(best.best_model_path or None)
    runs_dir = args.checkpoint_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    record_path = runs_dir / f"{timestamp_slug(started_at)}.json"
    record_path.write_text(json.dumps(results, indent=1))
    (args.checkpoint_dir / "train_summary.json").write_text(json.dumps(results, indent=1))
    printable = {key: value for key, value in results.items() if key != "splits"}
    printable["record"] = str(record_path)
    print(json.dumps(printable, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
