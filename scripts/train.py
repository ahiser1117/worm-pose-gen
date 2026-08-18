#!/usr/bin/env python3
"""Train one bounded EXP-0004 variant/fold run."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import subprocess
import sys

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from torch.utils.data import DataLoader
import yaml

from worm_pose_gen.model import WormProposalModule
from worm_pose_gen.training_data import make_datasets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("coordinate", "intrinsic"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("preflight.py")), "--require-cuda"],
        check=True,
    )
    config = yaml.safe_load(args.config.read_text())
    if config["experiment"] != "EXP-0004" or config["input"].get("audited_holdout_allowed", False):
        raise RuntimeError("refusing a non-EXP-0004 or audited-holdout-enabled config")
    seed = args.seed if args.seed is not None else int(config["seed"])
    L.seed_everything(seed, workers=True)
    maximum_steps = args.max_steps if args.max_steps is not None else int(config["training"]["maximum_steps"])
    if maximum_steps < 1 or maximum_steps > int(config["training"]["maximum_steps"]):
        raise ValueError("max-steps override must be within the configured bounded budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, proxy_validation, tier_c_validation = make_datasets(
        config["input"]["proxy_hdf5"],
        config["input"]["proxy_sha256"],
        fold=args.fold,
        seed=seed,
        synthetic_train_count=int(config["training"]["synthetic_samples_per_epoch"]),
        synthetic_validation_count=int(config["training"]["synthetic_validation_samples"]),
    )
    workers = int(config["training"]["num_workers"])
    train_loader = DataLoader(
        train,
        batch_size=int(config["training"]["train_batch_size"]),
        shuffle=True,
        num_workers=workers,
    )
    proxy_validation_loader = DataLoader(
        proxy_validation,
        batch_size=int(config["training"]["evaluation_batch_size"]),
        shuffle=False,
        num_workers=workers,
    )
    tier_c_validation_loader = DataLoader(
        tier_c_validation,
        batch_size=int(config["training"]["evaluation_batch_size"]),
        shuffle=False,
        num_workers=workers,
    )
    module = WormProposalModule(
        args.variant,
        image_height=int(config["input"]["height"]),
        image_width=int(config["input"]["width"]),
        num_points=int(config["input"]["body_points"]),
        intrinsic_coefficients=int(config["model"]["intrinsic_coefficients"]),
        anchor_index=int(config["model"]["body_anchor_index"]),
    )
    if sum(parameter.numel() for parameter in module.parameters()) > int(config["model"]["maximum_parameters"]):
        raise RuntimeError("model exceeds configured parameter ceiling")
    latest_checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{args.variant}-fold{args.fold}-step{{step}}",
        monitor=None,
        save_top_k=1,
        save_last=True,
        every_n_train_steps=int(config["training"]["checkpoint_every_steps"]),
        save_on_train_epoch_end=False,
    )
    best_checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{args.variant}-fold{args.fold}-best",
        monitor="val_tier_c_angle_mae_degrees",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )
    last = args.output_dir / "last.ckpt"
    timer = Timer(duration=timedelta(minutes=float(config["resources"]["maximum_minutes_per_run"])))
    trainer = L.Trainer(
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        precision=config["training"]["precision"],
        max_steps=maximum_steps,
        max_epochs=int(config["training"]["maximum_epochs"]),
        callbacks=[latest_checkpoint, best_checkpoint, timer],
        default_root_dir=args.output_dir,
        deterministic=True,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(
        module,
        train_loader,
        [proxy_validation_loader, tier_c_validation_loader],
        ckpt_path=str(last) if last.exists() else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
