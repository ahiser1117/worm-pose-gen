#!/usr/bin/env python3
"""Fine-tune the worm segmenter on the segmentation store with Lightning.

Each run gets its own directory under ``checkpoints/segmenter/runs/``, named
by start time and ``--name``, holding ``best.ckpt`` (highest validation
IoU), ``last.ckpt`` (the final epoch, which early stopping places
``--patience`` epochs after the best), ``metrics.csv`` (per-epoch curves), and
``run.json``: the arguments, git revision, the exact train/val/test
membership, the checkpoint fingerprints, and the final metrics.  The
directory is git-ignored.

``--train-labels`` restricts the training split to ``bootstrap`` or
``manual`` labels (validation and test always use every label they hold).
Without ``--init`` the model starts from ImageNet weights with no worm
exposure; with it the weights of an earlier checkpoint are the starting
point.

After training, the run's best checkpoint is scored against the currently
promoted ``checkpoints/segmenter/best.ckpt`` on the validation split (mean
IoU over hand-refined labels) and replaces it when it scores higher, so the
labeling app always proposes from the best validated model.  ``--promote``
forces the copy; ``--no-promote`` skips the comparison.  Every decision is
appended to ``checkpoints/segmenter/promotions.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

from worm_pose_gen.run_records import checkpoint_fingerprint, git_revision, split_manifest, timestamp_slug, utc_now
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, LABEL_FILTERS, SegmentationDataModule
from worm_pose_gen.segmenter import SegmentationModule
from worm_pose_gen.segmenter_eval import compare_on_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "segmenter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--name", default="run", help="run name, appended to the timestamped run directory")
    parser.add_argument("--train-labels", choices=LABEL_FILTERS, default="all", help="which training labels to use")
    parser.add_argument("--init", type=Path, default=None, help="checkpoint whose weights start the run")
    parser.add_argument("--promote", action="store_true", help="promote this run's best.ckpt without comparing")
    parser.add_argument("--no-promote", action="store_true", help="never touch <checkpoint dir>/best.ckpt")
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
        num_workers=args.num_workers, seed=args.seed, train_label_filter=args.train_labels,
    )
    counts = data.store.counts()
    train_records = data.train_records()
    counts["train_used"] = len(train_records)
    if not train_records or counts["val"] == 0:
        raise SystemExit(f"need train and val samples; store has {counts} with train labels {args.train_labels!r}")
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
    run_dir = args.checkpoint_dir / "runs" / f"{timestamp_slug(started_at)}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    best = ModelCheckpoint(
        dirpath=run_dir, filename="best", monitor="val_iou", mode="max",
        save_top_k=1, save_last=False, enable_version_counter=False,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[best, EarlyStopping(monitor="val_iou", mode="max", patience=args.patience)],
        logger=CSVLogger(str(run_dir), name="", version=""),
        default_root_dir=str(run_dir),
        log_every_n_steps=5,
        enable_progress_bar=True,
    )
    manifest = split_manifest(data.store)
    manifest["train"] = [row for row in manifest["train"] if row["sample_id"] in {r.sample_id for r in train_records}]
    trainer.fit(module, datamodule=data)
    # Lightning's save_last only writes when a top-k checkpoint is written, so
    # the final-epoch weights are saved explicitly here, before the test pass
    # reloads the best checkpoint into the module.
    trainer.save_checkpoint(run_dir / "last.ckpt")
    results = {
        "name": args.name,
        "run_dir": str(run_dir),
        "started_at": started_at,
        "finished_at": None,
        "git": git_revision(PROJECT_ROOT),
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "train_labels": args.train_labels,
        "init_checkpoint": checkpoint_fingerprint(args.init),
        "dataset_root": str(args.dataset_root),
        "counts": counts,
        "splits": manifest,
        "best_val_iou": None if best.best_model_score is None else float(best.best_model_score),
        "epochs_run": trainer.current_epoch,
        "stopped_early": trainer.current_epoch < args.epochs,
    }
    if counts["test"]:
        test = trainer.test(module, datamodule=data, ckpt_path=best.best_model_path or None, verbose=False)
        results["test"] = test[0] if test else None
    results["finished_at"] = utc_now()
    results["checkpoints"] = {
        "best": checkpoint_fingerprint(run_dir / "best.ckpt"),
        "last": checkpoint_fingerprint(run_dir / "last.ckpt"),
    }
    promoted_path = args.checkpoint_dir / "best.ckpt"
    if args.no_promote:
        results["promotion"] = {"promote": False, "reason": "--no-promote"}
    elif args.promote:
        results["promotion"] = {"promote": True, "reason": "--promote"}
    else:
        results["promotion"] = compare_on_split(run_dir / "best.ckpt", promoted_path, data.store, "val")
    if results["promotion"]["promote"]:
        shutil.copy2(run_dir / "best.ckpt", promoted_path)
        results["promotion"]["promoted_to"] = str(promoted_path)
    results["promotion"]["run_dir"] = str(run_dir)
    results["promotion"]["decided_at"] = utc_now()
    with open(args.checkpoint_dir / "promotions.jsonl", "a") as handle:
        handle.write(json.dumps({k: v for k, v in results["promotion"].items() if k != "per_sample"}) + "\n")
    print(f"promotion: {'yes' if results['promotion']['promote'] else 'no'} ({results['promotion']['reason']})")
    (run_dir / "run.json").write_text(json.dumps(results, indent=1))
    (args.checkpoint_dir / "train_summary.json").write_text(json.dumps(results, indent=1))
    printable = {key: value for key, value in results.items() if key != "splits"}
    print(json.dumps(printable, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
