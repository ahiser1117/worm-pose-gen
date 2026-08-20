#!/usr/bin/env python3
"""Train the preregistered EXP-004 5k analytic scale control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers import CSVLogger
import h5py
import torch
from torch.utils.data import DataLoader, Subset
import yaml

try:
    from scripts.preflight import query_physical_gpu_zero
    from scripts.train import StatefulFixedBatchSampler
except ModuleNotFoundError:
    from preflight import query_physical_gpu_zero
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


SHARED_PARENT_SOURCES = (
    "src/worm_pose_gen/topology_rescue_model.py",
    "src/worm_pose_gen/spatial_model.py",
    "src/worm_pose_gen/model.py",
    "src/worm_pose_gen/losses.py",
    "src/worm_pose_gen/training_data.py",
)


def _source_hashes(root: Path) -> dict[str, str]:
    names = ("scripts/train_exp_004.py", *SHARED_PARENT_SOURCES)
    return {name: sha256_file(root / name) for name in names}


def _validate(config: dict[str, Any]) -> None:
    """Fail closed if the analytic-control preregistration has drifted."""

    boundary = config.get("evidence_boundary", {})
    training = config.get("training", {})
    gate = config.get("controlled_gate", {})
    resources = config.get("resources", {})
    if (
        config.get("experiment") != "EXP-004"
        or config.get("phase") != "analytic_5k_scale_control"
        or config.get("parent") != "EXP-003B"
        or boundary.get("audited_holdout_allowed") is not False
        or boundary.get("primary_Tier_A_allowed_before_all_seed_gate") is not False
        or boundary.get("delayed_repeat_annotations_allowed") is not False
        or config.get("architecture", {}).get("name") != "soft_anchored_intrinsic"
        or int(training.get("fold", -1)) != 2
        or int(training.get("data_seed", -1)) != 20260819
        or int(training.get("synthetic_training_samples", -1)) != 5000
        or int(training.get("synthetic_validation_samples", -1)) != 128
        or int(training.get("maximum_steps", -1)) != 10800
        or int(training.get("maximum_epochs", -1)) != 40
        or training.get("materialize_before_training") is not True
        or gate.get("dataset") != "frozen_held_out_Tier_C_128"
        or gate.get("primary_stratum") != "fully_visible_43"
        or float(gate.get("median_full_latent_point_distance_px_at_most", -1)) != 16
        or float(gate.get("median_mean_tangent_error_degrees_at_most", -1)) != 15
        or float(gate.get("median_body_length_error_fraction_at_most", -1)) != 0.15
        or gate.get("Tier_A_tuning_allowed") is not False
        or gate.get("delayed_repeat_annotations_allowed") is not False
        or gate.get("protected_holdout_allowed") is not False
        or int(resources.get("physical_device_index", -1)) != 0
        or int(resources.get("visible_logical_device_index", -1)) != 0
        or int(resources.get("devices", -1)) != 1
    ):
        raise RuntimeError("invalid EXP-004 analytic-control preregistration")


def _validate_parent(config: dict[str, Any], root: Path) -> dict[str, Any]:
    provenance = config["provenance"]
    paths = {
        "parent_config": root / provenance["parent_config"],
        "parent_primary_run_metrics": Path(provenance["parent_primary_run_metrics"]),
        "parent_primary_gate": root / provenance["parent_primary_gate"],
    }
    for name, path in paths.items():
        expected = provenance[f"{name}_sha256"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"EXP-004 parent provenance changed: {name}")
    parent_run = json.loads(paths["parent_primary_run_metrics"].read_text())
    parent_gate = json.loads(paths["parent_primary_gate"].read_text())
    if (
        parent_run.get("experiment") != "EXP-003B"
        or parent_run.get("model_seed") != 20260819
        or parent_run.get("evidence_boundary", {}).get("Tier_A_evaluated") is not False
        or parent_gate.get("controlled_gate", {}).get("passed") is not False
        or parent_gate.get("materialized_tier_c_validation_sha256")
        != provenance["expected_tier_c_validation_materialized_sha256"]
    ):
        raise RuntimeError("EXP-004 parent artifacts have an invalid evidence identity")
    current = _source_hashes(root)
    for name in SHARED_PARENT_SOURCES:
        if current[name] != parent_run.get("source_sha256", {}).get(name):
            raise RuntimeError(f"topology-safe parent source changed: {name}")
    return {
        "validated_artifact_sha256": {
            name: provenance[f"{name}_sha256"] for name in paths
        },
        "shared_parent_source_sha256": {name: current[name] for name in SHARED_PARENT_SOURCES},
    }


def _validate_split_semantics(config: dict[str, Any], root: Path) -> dict[str, Any]:
    provenance = config["provenance"]
    split_path = root / provenance["current_split_manifest"]
    if sha256_file(split_path) != provenance["current_split_manifest_sha256"]:
        raise RuntimeError("current split manifest bytes changed")
    split = json.loads(split_path.read_text())
    expected_records = ["2023-09-19-01", "2023-09-27-01", "2023-10-11-01"]
    folds = {int(value["fold"]): value for value in split.get("development_folds", [])}
    if (
        split.get("development_records") != expected_records
        or 2 not in folds
        or folds[2].get("train") != expected_records[:2]
        or folds[2].get("validation") != expected_records[2:]
    ):
        raise RuntimeError("current split manifest changed EXP-004 fold-2 semantics")
    with h5py.File(Path(config["input"]["proxy_hdf5"]), "r") as handle:
        embedded = str(handle.attrs.get("split_manifest_sha256", ""))
    if embedded != provenance["proxy_embedded_split_manifest_sha256"]:
        raise RuntimeError("proxy embedded split-manifest identity changed")
    return {
        "current_split_manifest_sha256": provenance["current_split_manifest_sha256"],
        "proxy_embedded_split_manifest_sha256": embedded,
        "development_records": expected_records,
        "fold_2_train": expected_records[:2],
        "fold_2_validation": expected_records[2:],
        "byte_hashes_differ_semantics_validated": embedded
        != provenance["current_split_manifest_sha256"],
    }


def _validate_repeat_authorization(
    path: Path | None, *, config_sha256: str, seed: int, primary_seed: int
) -> dict[str, str] | None:
    if seed == primary_seed:
        if path is not None:
            raise ValueError("the primary EXP-004 seed cannot consume repeat authorization")
        return None
    if path is None:
        raise RuntimeError("repeat seed is closed until a passing primary gate authorizes it")
    payload = json.loads(path.read_text())
    if (
        payload.get("experiment") != "EXP-004"
        or payload.get("phase") != "analytic_5k_scale_control"
        or payload.get("model_seed") != primary_seed
        or payload.get("config_sha256") != config_sha256
        or payload.get("controlled_gate", {}).get("passed") is not True
        or payload.get("controlled_gate", {}).get("decision") != "AUTHORIZE_REPEAT_SEEDS_ONLY"
        or payload.get("evidence_boundary", {}).get("Tier_A_evaluated") is not False
        or payload.get("evidence_boundary", {}).get("protected_holdout_opened") is not False
        or payload.get("evidence_boundary", {}).get("repeat_annotations_used") is not False
    ):
        raise RuntimeError("invalid EXP-004 repeat-seed authorization")
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def _validate_gpu(config: dict[str, Any]) -> dict[str, Any]:
    resources = config["resources"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("EXP-004 requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("EXP-004 requires exactly one visible CUDA device")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("physical GPU 0 must appear as logical cuda:0")
    physical = query_physical_gpu_zero()
    if (
        physical.get("physical_index") != int(resources["physical_device_index"])
        or physical.get("uuid") != resources["expected_uuid"]
        or str(physical.get("pci_bus_id", "")).lower()
        != str(resources["expected_pci_bus_id"]).lower()
    ):
        raise RuntimeError("physical GPU 0 identity does not match the preregistration")
    logical = torch.cuda.get_device_properties(0)
    return {
        "physical_device": physical,
        "mapping": {"physical_index": 0, "visible_logical_index": 0},
        "logical_device": {
            "name": logical.name,
            "total_memory_bytes": logical.total_memory,
            "cuda_runtime": torch.version.cuda,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-gate-authorization", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite EXP-004 run: {args.output_dir}")
    config = yaml.safe_load(args.config.read_text())
    _validate(config)
    config_sha256 = sha256_file(args.config)
    training = config["training"]
    primary_seed = int(training["primary_model_seed"])
    allowed_seeds = (primary_seed, *map(int, training["repeat_model_seeds"]))
    if args.seed not in allowed_seeds:
        raise ValueError("seed is outside the preregistered EXP-004 set")
    repeat_authorization = _validate_repeat_authorization(
        args.primary_gate_authorization,
        config_sha256=config_sha256,
        seed=args.seed,
        primary_seed=primary_seed,
    )

    root = Path(__file__).resolve().parents[1]
    parent_provenance = _validate_parent(config, root)
    split_provenance = _validate_split_semantics(config, root)
    gpu = _validate_gpu(config)
    torch.set_num_threads(int(training["cpu_threads"]))
    torch.set_float32_matmul_precision("high")
    boundary = config["evidence_boundary"]
    exclusions = load_proxy_frame_exclusions(
        root / boundary["proxy_exclusion_manifest"],
        expected_sha256=boundary["proxy_exclusion_manifest_sha256"],
    )
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
        raise RuntimeError("unexpected EXP-004 dataset topology")
    excluded_by_split = {
        "train": proxy_train.excluded_row_count,
        "proxy_validation": raw_proxy_validation.excluded_row_count,
    }
    if sum(excluded_by_split.values()) != int(boundary["accepted_proxy_rows_excluded"]):
        raise RuntimeError("EXP-004 proxy exclusion count changed")

    materialization_started = time.perf_counter()
    train = MaterializedPoseDataset(raw_train)
    proxy_validation = MaterializedPoseDataset(raw_proxy_validation)
    tier_c_validation = MaterializedPoseDataset(raw_tier_c_validation)
    materialization_seconds = time.perf_counter() - materialization_started
    dataset_hashes = {
        "train": materialized_dataset_sha256(train),
        "parent_training_prefix": materialized_dataset_sha256(
            Subset(train, range(565))
        ),
        "proxy_validation": materialized_dataset_sha256(proxy_validation),
        "tier_c_validation": materialized_dataset_sha256(tier_c_validation),
    }
    proxy_train.close()
    raw_proxy_validation.close()
    provenance = config["provenance"]
    if (
        len(train) != 5053
        or len(proxy_validation) != 28
        or len(tier_c_validation) != 128
        or dataset_hashes["parent_training_prefix"]
        != provenance["expected_parent_training_prefix_materialized_sha256"]
        or dataset_hashes["proxy_validation"]
        != provenance["expected_proxy_validation_materialized_sha256"]
        or dataset_hashes["tier_c_validation"]
        != provenance["expected_tier_c_validation_materialized_sha256"]
    ):
        raise RuntimeError("EXP-004 materialized dataset provenance changed")

    args.output_dir.mkdir(parents=True)
    materialization_provenance_path = args.output_dir / "materialization_provenance.json"
    materialization_provenance = {
        "schema_version": 1,
        "experiment": "EXP-004",
        "phase": "analytic_5k_scale_control",
        "status": "FROZEN_BEFORE_OPTIMIZATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.resolve(strict=True)),
        "config_sha256": config_sha256,
        "model_seed": args.seed,
        "data_seed": int(training["data_seed"]),
        "fold": int(training["fold"]),
        "materialized_dataset_sha256": dataset_hashes,
        "counts": {
            "training_total": len(train),
            "training_analytic": int(training["synthetic_training_samples"]),
            "training_proxy": len(train) - int(training["synthetic_training_samples"]),
            "proxy_validation": len(proxy_validation),
            "tier_c_validation": len(tier_c_validation),
            "excluded_proxy_rows_by_split": excluded_by_split,
        },
        "parent_provenance": parent_provenance,
        "split_provenance": split_provenance,
        "source_sha256": _source_hashes(root),
        "gpu": gpu,
        "evidence_boundary": {
            "protected_holdout_opened": False,
            "Tier_A_evaluated": False,
            "repeat_annotations_used": False,
        },
    }
    materialization_provenance_path.write_text(
        json.dumps(materialization_provenance, indent=2) + "\n"
    )

    sampler = StatefulFixedBatchSampler(
        len(train), int(training["train_batch_size"]), args.seed
    )
    L.seed_everything(args.seed, workers=True)
    model = SoftAnchoredIntrinsicModule(
        learning_rate=float(training["learning_rate"]),
        model_seed=args.seed,
        data_seed=int(training["data_seed"]),
        training_order_sha256=sampler.order_sha256,
        exclusion_manifest_sha256=boundary["proxy_exclusion_manifest_sha256"],
    )
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count > int(config["architecture"]["maximum_parameters"]):
        raise RuntimeError("EXP-004 model exceeds its parameter ceiling")

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step{step:05d}",
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
        DataLoader(
            proxy_validation,
            batch_size=int(training["evaluation_batch_size"]),
            num_workers=0,
            pin_memory=pin_memory,
        ),
        DataLoader(
            tier_c_validation,
            batch_size=int(training["evaluation_batch_size"]),
            num_workers=0,
            pin_memory=pin_memory,
        ),
    ]
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=training["precision"],
        deterministic=True,
        max_steps=int(training["maximum_steps"]),
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
    maximum_steps = int(training["maximum_steps"])
    if trainer.global_step != maximum_steps:
        raise RuntimeError("EXP-004 training stopped before its fixed step budget")
    final_checkpoint = checkpoint_dir / f"final-step{trainer.global_step}.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    gpu["logical_device"]["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    metrics = {
        "schema_version": 1,
        "experiment": "EXP-004",
        "phase": "analytic_5k_scale_control",
        "status": "TRAINED_PENDING_TIER_C_GATE",
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
        "analytic_training_samples": int(training["synthetic_training_samples"]),
        "proxy_training_samples": len(train) - int(training["synthetic_training_samples"]),
        "proxy_validation_samples": len(proxy_validation),
        "tier_c_validation_samples": len(tier_c_validation),
        "excluded_proxy_rows": sum(excluded_by_split.values()),
        "excluded_proxy_rows_by_split": excluded_by_split,
        "training_order_sha256": sampler.order_sha256,
        "throughput": {
            "optimizer_steps_per_second": trainer.global_step / elapsed,
            "nominal_training_samples_per_second": (
                trainer.global_step * int(training["train_batch_size"]) / elapsed
            ),
        },
        "config_path": str(args.config.resolve(strict=True)),
        "config_sha256": config_sha256,
        "checkpoint_path": str(final_checkpoint.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "source_sha256": _source_hashes(root),
        "parent_provenance": parent_provenance,
        "split_provenance": split_provenance,
        "materialization_provenance_path": str(
            materialization_provenance_path.resolve(strict=True)
        ),
        "materialization_provenance_sha256": sha256_file(materialization_provenance_path),
        "repeat_authorization": repeat_authorization,
        "callback_metrics": {
            name: float(value.detach().cpu())
            for name, value in trainer.callback_metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        },
        "gpu": gpu,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
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
