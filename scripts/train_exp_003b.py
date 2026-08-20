#!/usr/bin/env python3
"""Train the Tier-C-gated topology-safe EXP-003B rescue model."""

from __future__ import annotations

import argparse
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

try:
    from scripts.train import StatefulFixedBatchSampler
except ModuleNotFoundError:
    from train import StatefulFixedBatchSampler
from worm_pose_gen.topology_rescue_model import SoftAnchoredIntrinsicModule
from worm_pose_gen.training_data import (
    MaterializedPoseDataset,
    ProxyDataset,
    load_proxy_frame_exclusions,
    make_datasets,
    materialized_dataset_sha256,
    sha256_file,
)


def _source_hashes(root: Path) -> dict[str, str]:
    names = (
        "scripts/train_exp_003b.py",
        "src/worm_pose_gen/topology_rescue_model.py",
        "src/worm_pose_gen/spatial_model.py",
        "src/worm_pose_gen/model.py",
        "src/worm_pose_gen/losses.py",
        "src/worm_pose_gen/training_data.py",
    )
    return {name: sha256_file(root / name) for name in names}


def _validate(config: dict[str, Any]) -> None:
    if (
        config.get("experiment") != "EXP-003B"
        or config["evidence_boundary"].get("audited_holdout_allowed") is not False
        or config["evidence_boundary"].get("primary_Tier_A_allowed_before_controlled_gate") is not False
        or config["architecture"].get("name") != "soft_anchored_intrinsic"
        or int(config["training"]["fold"]) != 2
    ):
        raise RuntimeError("invalid EXP-003B preregistration identity")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite EXP-003B run: {args.output_dir}")
    config = yaml.safe_load(args.config.read_text())
    _validate(config)
    allowed_seeds = (
        int(config["training"]["primary_model_seed"]),
        *map(int, config["training"]["repeat_model_seeds"]),
    )
    if args.seed not in allowed_seeds:
        raise ValueError("seed is outside the preregistered EXP-003B set")
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 100:
        raise ValueError("smoke steps must lie in [1,100]")
    maximum_steps = args.smoke_steps or int(config["training"]["maximum_steps"])
    torch.set_num_threads(int(config["training"]["cpu_threads"]))
    torch.set_float32_matmul_precision("high")
    root = Path(__file__).resolve().parents[1]
    exclusion = config["evidence_boundary"]
    exclusions = load_proxy_frame_exclusions(
        exclusion["proxy_exclusion_manifest"],
        expected_sha256=exclusion["proxy_exclusion_manifest_sha256"],
    )
    training = config["training"]
    raw_train, raw_proxy_validation, raw_tier_c_validation = make_datasets(
        config["input"]["proxy_hdf5"],
        config["input"]["proxy_sha256"],
        fold=int(training["fold"]),
        seed=int(training["data_seed"]),
        synthetic_train_count=int(training["synthetic_training_samples"]),
        synthetic_validation_count=int(training["synthetic_validation_samples"]),
        excluded_frame_indices=exclusions,
    )
    proxy_train = raw_train.datasets[0]
    if not isinstance(proxy_train, ProxyDataset):
        raise RuntimeError("unexpected EXP-003B dataset topology")
    excluded_rows = proxy_train.excluded_row_count + raw_proxy_validation.excluded_row_count
    materialization_started = time.perf_counter()
    train = MaterializedPoseDataset(raw_train)
    proxy_validation = MaterializedPoseDataset(raw_proxy_validation)
    tier_c_validation = MaterializedPoseDataset(raw_tier_c_validation)
    materialization_seconds = time.perf_counter() - materialization_started
    dataset_hashes = {
        "train": materialized_dataset_sha256(train),
        "proxy_validation": materialized_dataset_sha256(proxy_validation),
        "tier_c_validation": materialized_dataset_sha256(tier_c_validation),
    }
    proxy_train.close()
    raw_proxy_validation.close()
    if excluded_rows != int(exclusion["accepted_proxy_rows_excluded"]):
        raise RuntimeError("EXP-003B proxy exclusion count changed")
    sampler = StatefulFixedBatchSampler(len(train), int(training["train_batch_size"]), args.seed)
    L.seed_everything(args.seed, workers=True)
    model = SoftAnchoredIntrinsicModule(
        learning_rate=float(training["learning_rate"]),
        model_seed=args.seed,
        data_seed=int(training["data_seed"]),
        training_order_sha256=sampler.order_sha256,
        exclusion_manifest_sha256=exclusion["proxy_exclusion_manifest_sha256"],
    )
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count > int(config["architecture"]["maximum_parameters"]):
        raise RuntimeError("EXP-003B model exceeds its parameter ceiling")
    args.output_dir.mkdir(parents=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step{step:04d}",
        every_n_train_steps=int(training["checkpoint_every_steps"]),
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    timer = Timer(duration={"minutes": int(config["resources"]["maximum_minutes_primary_run"])})
    pin_memory = bool(training["pin_memory"])
    train_loader = DataLoader(
        train, batch_sampler=sampler, num_workers=0, pin_memory=pin_memory
    )
    validation_loaders = [
        DataLoader(proxy_validation, batch_size=int(training["evaluation_batch_size"]), num_workers=0, pin_memory=pin_memory),
        DataLoader(tier_c_validation, batch_size=int(training["evaluation_batch_size"]), num_workers=0, pin_memory=pin_memory),
    ]
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=training["precision"],
        deterministic=True,
        max_steps=maximum_steps,
        max_epochs=int(training["maximum_epochs"]),
        callbacks=[checkpoint, timer],
        logger=CSVLogger(save_dir=args.output_dir, name="logs"),
        enable_progress_bar=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
    )
    started = time.perf_counter()
    trainer.fit(model, train_loader, validation_loaders)
    elapsed = time.perf_counter() - started
    if trainer.global_step != maximum_steps:
        raise RuntimeError("EXP-003B training stopped before its fixed step budget")
    final_checkpoint = checkpoint_dir / f"final-step{trainer.global_step}.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    gpu = torch.cuda.get_device_properties(0)
    metrics = {
        "schema_version": 1,
        "experiment": "EXP-003B",
        "status": "SMOKE_ONLY" if args.smoke_steps is not None else "TRAINED_PENDING_TIER_C_GATE",
        "variant": "soft_anchored_intrinsic",
        "model_seed": args.seed,
        "data_seed": int(training["data_seed"]),
        "fold": int(training["fold"]),
        "global_step": trainer.global_step,
        "elapsed_train_and_validation_seconds": elapsed,
        "materialization_seconds": materialization_seconds,
        "materialized_dataset_sha256": dataset_hashes,
        "parameters": parameter_count,
        "training_samples": len(train),
        "proxy_validation_samples": len(proxy_validation),
        "tier_c_validation_samples": len(tier_c_validation),
        "excluded_proxy_rows": excluded_rows,
        "training_order_sha256": sampler.order_sha256,
        "throughput": {
            "optimizer_steps_per_second": trainer.global_step / elapsed,
            "nominal_training_samples_per_second": trainer.global_step * int(training["train_batch_size"]) / elapsed,
        },
        "config_path": str(args.config.resolve(strict=True)),
        "config_sha256": sha256_file(args.config),
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
            "Tier_A_evaluated": False,
            "repeat_annotations_used": False,
        },
    }
    (args.output_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
