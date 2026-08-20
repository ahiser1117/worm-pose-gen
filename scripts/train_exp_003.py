#!/usr/bin/env python3
"""Train one leakage-safe, budget-matched EXP-003 architecture/seed run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader
import yaml

from worm_pose_gen.model import WormProposalModule
from worm_pose_gen.spatial_model import SpatialPoseModule
from worm_pose_gen.training_data import (
    MaterializedPoseDataset,
    ProxyDataset,
    load_proxy_frame_exclusions,
    make_datasets,
    materialized_dataset_sha256,
    sha256_file,
)
try:
    from scripts.train import StatefulFixedBatchSampler
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from train import StatefulFixedBatchSampler


VARIANTS = (
    "global_intrinsic_budget_matched",
    "dense_centerline_field",
    "anchored_intrinsic_grid",
)


def _source_hashes(root: Path) -> dict[str, str]:
    names = (
        "scripts/train_exp_003.py",
        "src/worm_pose_gen/model.py",
        "src/worm_pose_gen/spatial_model.py",
        "src/worm_pose_gen/losses.py",
        "src/worm_pose_gen/training_data.py",
    )
    return {name: sha256_file(root / name) for name in names}


def validate_config(config: dict[str, Any], config_path: Path) -> None:
    if config.get("experiment") != "EXP-003":
        raise RuntimeError("train_exp_003 requires the EXP-003 config")
    if config["evidence_boundary"].get("audited_holdout_allowed") is not False:
        raise RuntimeError("EXP-003 must keep the audited holdout closed")
    if config["training"].get("primary_Tier_A_used_for_gradients") is not False:
        raise RuntimeError("Tier-A annotations must remain evaluation-only")
    if int(config["training"]["fold"]) != 2:
        raise RuntimeError("initial EXP-003 comparison is frozen to fold 2")
    configured = tuple(value["name"] for value in config["paired_architectures"])
    if configured != VARIANTS:
        raise RuntimeError("EXP-003 architecture order/identity changed")
    if tuple(config["training"]["model_seeds"]) != (20260819, 20260820, 20260821):
        raise RuntimeError("EXP-003 model seeds changed")
    if (int(config["input"]["initial_height"]), int(config["input"]["initial_width"])) != (192, 256):
        raise RuntimeError("initial EXP-003 resolution changed")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)


def _model(
    variant: str,
    *,
    seed: int,
    data_seed: int,
    order_sha256: str,
    exclusion_sha256: str,
) -> L.LightningModule:
    common = {
        "learning_rate": 3e-4,
        "model_seed": seed,
        "data_seed": data_seed,
        "training_order_sha256": order_sha256,
    }
    if variant == "global_intrinsic_budget_matched":
        return WormProposalModule(
            "intrinsic", encoder_pool_output=(4, 4), **common
        )
    return SpatialPoseModule(
        variant,
        exclusion_manifest_sha256=exclusion_sha256,
        **common,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite EXP-003 run: {args.output_dir}")
    config = yaml.safe_load(args.config.read_text())
    validate_config(config, args.config)
    if args.seed not in config["training"]["model_seeds"]:
        raise ValueError("seed is outside the frozen EXP-003 seed set")
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 100:
        raise ValueError("smoke steps must lie in [1,100]")
    maximum_steps = (
        args.smoke_steps
        if args.smoke_steps is not None
        else int(config["training"]["maximum_steps"])
    )
    root = Path(__file__).resolve().parents[1]
    pipeline = config["training"]["data_pipeline"]
    if (
        pipeline.get("materialize_before_training") is not True
        or pipeline.get("materialized_tensor_hash_required") is not True
        or int(pipeline["dataloader_workers"]) != 0
    ):
        raise RuntimeError("EXP-003 requires the frozen materialized input pipeline")
    torch.set_num_threads(int(pipeline["cpu_threads"]))
    torch.set_float32_matmul_precision("high")
    exclusion = config["evidence_boundary"]["exclude_from_training"]
    exclusions = load_proxy_frame_exclusions(
        exclusion["manifest"], expected_sha256=exclusion["manifest_sha256"]
    )
    data_seed = int(config["training"]["data_seed"])
    fold = int(config["training"]["fold"])
    raw_train, raw_proxy_validation, raw_tier_c_validation = make_datasets(
        config["input"]["proxy_hdf5"],
        config["input"]["proxy_sha256"],
        fold=fold,
        seed=data_seed,
        synthetic_train_count=512,
        synthetic_validation_count=128,
        excluded_frame_indices=exclusions,
    )
    proxy_train_dataset = raw_train.datasets[0]
    if not isinstance(proxy_train_dataset, ProxyDataset):
        raise RuntimeError("unexpected EXP-003 training dataset topology")
    excluded_proxy_train_rows = proxy_train_dataset.excluded_row_count
    excluded_proxy_validation_rows = raw_proxy_validation.excluded_row_count
    proxy_training_samples = len(proxy_train_dataset)
    materialization_started = time.perf_counter()
    train = MaterializedPoseDataset(raw_train)
    proxy_validation = MaterializedPoseDataset(raw_proxy_validation)
    tier_c_validation = MaterializedPoseDataset(raw_tier_c_validation)
    materialization_seconds = time.perf_counter() - materialization_started
    tensor_hashes = {
        "train": materialized_dataset_sha256(train),
        "proxy_validation": materialized_dataset_sha256(proxy_validation),
        "tier_c_validation": materialized_dataset_sha256(tier_c_validation),
    }
    proxy_train_dataset.close()
    raw_proxy_validation.close()
    sampler = StatefulFixedBatchSampler(
        len(train), int(config["training"]["train_batch_size"]), args.seed
    )
    L.seed_everything(args.seed, workers=True)
    model = _model(
        args.variant,
        seed=args.seed,
        data_seed=data_seed,
        order_sha256=sampler.order_sha256,
        exclusion_sha256=exclusion["manifest_sha256"],
    )
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count > int(config["architecture_common_constraints"]["maximum_parameters"]):
        raise RuntimeError(f"model has {parameter_count} parameters, above the frozen ceiling")
    args.output_dir.mkdir(parents=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step{step:04d}",
        every_n_train_steps=int(config["training"]["checkpoint_every_steps"]),
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    timer = Timer(duration={"minutes": int(config["resources"]["maximum_minutes_per_run"])})
    pin_memory = bool(pipeline["pin_memory"])
    train_loader = DataLoader(
        train, batch_sampler=sampler, num_workers=0, pin_memory=pin_memory
    )
    validation_loaders = [
        DataLoader(
            proxy_validation,
            batch_size=int(config["training"]["evaluation_batch_size"]),
            num_workers=0,
            pin_memory=pin_memory,
        ),
        DataLoader(
            tier_c_validation,
            batch_size=int(config["training"]["evaluation_batch_size"]),
            num_workers=0,
            pin_memory=pin_memory,
        ),
    ]
    logger = CSVLogger(save_dir=args.output_dir, name="logs")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=config["training"]["precision"],
        deterministic=True,
        max_steps=maximum_steps,
        max_epochs=int(config["training"]["maximum_epochs"]),
        callbacks=[checkpoint, timer],
        logger=logger,
        enable_progress_bar=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
    )
    started = time.perf_counter()
    trainer.fit(model, train_loader, validation_loaders)
    elapsed = time.perf_counter() - started
    if trainer.global_step != maximum_steps:
        raise RuntimeError(
            f"training stopped at step {trainer.global_step}, expected {maximum_steps}"
        )
    final_checkpoint = checkpoint_dir / f"final-step{trainer.global_step}.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    if (
        excluded_proxy_train_rows
        + excluded_proxy_validation_rows
        != int(exclusion["accepted_proxy_rows_excluded"])
    ):
        raise RuntimeError("proxy exclusion count changed")
    gpu = torch.cuda.get_device_properties(0)
    metrics = {
        "schema_version": 1,
        "experiment": "EXP-003",
        "status": "SMOKE_ONLY" if args.smoke_steps is not None else "TRAINED_PENDING_EVALUATION",
        "variant": args.variant,
        "model_seed": args.seed,
        "data_seed": data_seed,
        "fold": fold,
        "global_step": trainer.global_step,
        "elapsed_train_and_validation_seconds": elapsed,
        "materialization_seconds": materialization_seconds,
        "materialized_dataset_sha256": tensor_hashes,
        "parameters": parameter_count,
        "training_samples": len(train),
        "proxy_training_samples": proxy_training_samples,
        "proxy_validation_samples": len(proxy_validation),
        "tier_c_validation_samples": len(tier_c_validation),
        "excluded_proxy_train_rows": excluded_proxy_train_rows,
        "excluded_proxy_validation_rows": excluded_proxy_validation_rows,
        "training_order_sha256": sampler.order_sha256,
        "throughput": {
            "optimizer_steps_per_second": trainer.global_step / elapsed,
            "nominal_training_samples_per_second": (
                trainer.global_step * int(config["training"]["train_batch_size"]) / elapsed
            ),
        },
        "config_path": str(args.config.resolve(strict=True)),
        "config_sha256": sha256_file(args.config),
        "exclusion_manifest_sha256": exclusion["manifest_sha256"],
        "proxy_sha256": config["input"]["proxy_sha256"],
        "checkpoint_path": str(final_checkpoint.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "source_sha256": _source_hashes(root),
        "callback_metrics": {
            name: float(value.detach().cpu())
            for name, value in trainer.callback_metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        },
        "gpu": {
            "name": gpu.name,
            "total_memory_bytes": gpu.total_memory,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_runtime": torch.version.cuda,
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "evidence_boundary": {
            "protected_holdout_opened": False,
            "primary_Tier_A_used_for_gradients": False,
        },
    }
    (args.output_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
