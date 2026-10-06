#!/usr/bin/env python3
"""Train the body-field network (mask, A-P field, head/tail, overlap) on one store.

``--lags`` picks the temporal difference channels as comma-separated frame
lags (``""`` for the single-frame baseline); every lag must be within the
``max_lag`` the store's body fields were built with.  Each run gets its own
directory under ``checkpoints/body_net/runs/``, named by start time and
``--name``, holding ``best.ckpt`` (lowest validation loss), ``last.ckpt``,
``metrics.csv``, and ``run.json`` (arguments, git revision, split membership,
checkpoint fingerprints, best epoch, and test metrics of the best
checkpoint).  Stopping and the learning-rate schedule follow
``scripts/train_segmenter.py``.  Nothing is promoted: compare runs with
``scripts/evaluate_body_net.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

from worm_pose_gen.body_net import BodyFieldDataModule, BodyFieldModule
from worm_pose_gen.run_records import checkpoint_fingerprint, git_revision, split_manifest, timestamp_slug, utc_now
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "body_net"


def parse_lags(text: str) -> tuple[int, ...]:
    return tuple(sorted({int(part) for part in text.split(",") if part.strip()}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--name", default="run")
    parser.add_argument("--lags", type=parse_lags, default=parse_lags("1,2,4,8,16"), help="comma-separated lags; empty for none")
    parser.add_argument("--epochs", type=int, default=300, help="cap; early stopping normally ends the run first")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.25)
    parser.add_argument("--ap-weight", type=float, default=1.0)
    parser.add_argument("--heatmap-weight", type=float, default=0.1)
    parser.add_argument("--overlap-weight", type=float, default=1.0)
    parser.add_argument("--min-fit-iou", type=float, default=0.9, help="samples whose tube fit is worse train the mask only")
    parser.add_argument("--patience", type=int, default=15, help="stop after this many epochs without a lower validation loss")
    parser.add_argument("--plateau-patience", type=int, default=4, help="halve the learning rate after this many stalled epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    data = BodyFieldDataModule(
        args.dataset_root, args.lags, batch_size=args.batch_size, crop_size=args.crop_size,
        num_workers=args.num_workers, seed=args.seed, min_fit_iou=args.min_fit_iou,
    )
    module = BodyFieldModule(
        lags=args.lags, pretrained=True, learning_rate=args.learning_rate,
        encoder_learning_rate_scale=args.encoder_lr_scale, ap_weight=args.ap_weight,
        heatmap_weight=args.heatmap_weight, overlap_weight=args.overlap_weight,
        plateau_patience=args.plateau_patience,
    )
    run_dir = args.checkpoint_dir / "runs" / f"{timestamp_slug(started_at)}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    best = ModelCheckpoint(
        dirpath=run_dir, filename="best", monitor="val_loss", mode="min",
        save_top_k=1, save_last=False, enable_version_counter=False,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[best, EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)],
        logger=CSVLogger(str(run_dir), name="", version=""),
        default_root_dir=str(run_dir),
        log_every_n_steps=5,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=data)
    trainer.save_checkpoint(run_dir / "last.ckpt")
    results = {
        "name": args.name,
        "run_dir": str(run_dir),
        "started_at": started_at,
        "git": git_revision(PROJECT_ROOT),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "lags": list(args.lags),
        "counts": data.counts(),
        "splits": split_manifest(data.store),
        "best_val_loss": None if best.best_model_score is None else float(best.best_model_score),
        "epochs_run": trainer.current_epoch,
    }
    test = trainer.test(module, datamodule=data, ckpt_path=best.best_model_path or None, verbose=False)
    results["test"] = test[0] if test else None
    results["best_epoch"] = int(torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)["epoch"])
    results["checkpoints"] = {
        "best": checkpoint_fingerprint(run_dir / "best.ckpt"),
        "last": checkpoint_fingerprint(run_dir / "last.ckpt"),
    }
    results["finished_at"] = utc_now()
    (run_dir / "run.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "splits"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
