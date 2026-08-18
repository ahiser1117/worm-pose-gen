#!/usr/bin/env python3
"""Train one bounded EXP-0004 or EXP-0007 variant/fold run."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, Timer
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader
import yaml

from worm_pose_gen.model import WormProposalModule
from worm_pose_gen.training_data import make_datasets, sha256_file
try:  # Supports both ``python scripts/train.py`` and test/package imports.
    from scripts.exp_0007_baseline import validate_exp_0007_config, verify_baseline
except ModuleNotFoundError:  # pragma: no cover - direct-script import path
    from exp_0007_baseline import validate_exp_0007_config, verify_baseline


class ImmutableStepCheckpoint(Callback):
    """Publish one non-rotating checkpoint for external elimination evaluation."""

    def __init__(self, path: Path, step: int) -> None:
        self.path = path
        self.step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step == self.step:
            if self.path.exists():
                raise FileExistsError(f"immutable checkpoint already exists: {self.path}")
            trainer.save_checkpoint(self.path)


class FullyVisibleCountContract(Callback):
    """Fail training if the frozen Tier C fully-visible identity count changes."""

    def __init__(self, expected_count: int) -> None:
        self.expected_count = expected_count

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        name = "val_tier_c_fully_visible_count"
        value = trainer.callback_metrics.get(name)
        if value is None or int(value.item()) != self.expected_count:
            observed = None if value is None else int(value.item())
            raise RuntimeError(
                f"{name}={observed}; expected frozen count {self.expected_count}"
            )


def resolve_protocol(config: dict, requested_variant: str | None) -> tuple[str, tuple[int, int]]:
    experiment = config.get("experiment")
    if experiment == "EXP-0004":
        if requested_variant is None or requested_variant not in config["model"]["variants"]:
            raise ValueError("EXP-0004 requires --variant from its configured variants")
        return requested_variant, (2, 2)
    if experiment == "EXP-0007":
        validate_exp_0007_config(config)
        variant = config["model"]["variant"]
        if requested_variant is not None and requested_variant != variant:
            raise ValueError("EXP-0007 is intrinsic-only")
        if int(config["training"]["maximum_steps"]) != 1200 or int(config["training"]["maximum_epochs"]) != 34:
            raise ValueError("EXP-0007 requires 1200 steps and 34 epochs")
        return variant, tuple(int(value) for value in config["model"]["encoder_pool_output"])
    raise ValueError("train.py supports only EXP-0004 and EXP-0007")


def resolve_resume_checkpoint(
    output_dir: Path, *, resume_last: bool, resume_from: Path | None
) -> Path | None:
    last = output_dir / "last.ckpt"
    if resume_last and resume_from is not None:
        raise ValueError("choose only one of --resume-last and --resume-from")
    if resume_last:
        if not last.is_file():
            raise FileNotFoundError(f"resume requested but checkpoint is missing: {last}")
        return last
    if resume_from is not None:
        return resume_from.resolve(strict=True)
    if last.exists():
        raise FileExistsError("last.ckpt exists; choose --resume-last or a new output directory")
    return None


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_decision_authorization(
    path: Path, config_path: Path, *, purpose: str
) -> dict:
    document = json.loads(path.read_text())
    if (
        document.get("schema_version") != 1
        or document.get("experiment") != "EXP-0007"
        or document.get("config_sha256") != sha256_file(config_path)
        or document.get("code_git_commit") != current_git_commit()
    ):
        raise RuntimeError(f"{purpose} authorization identity mismatch")
    repository_root = Path(__file__).resolve().parents[1]
    sources = document.get("source_sha256")
    if not isinstance(sources, dict) or not sources or any(
        sha256_file(repository_root / name) != digest for name, digest in sources.items()
    ):
        raise RuntimeError(f"{purpose} authorization source hash mismatch")
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError(f"{purpose} authorization has no input provenance")
    for records in inputs.values():
        if not isinstance(records, list):
            raise RuntimeError(f"{purpose} authorization input provenance is malformed")
        for record in records:
            input_path = Path(record["path"])
            if sha256_file(input_path) != record.get("sha256"):
                raise RuntimeError(f"{purpose} authorization input hash mismatch")
    return document


def checkpoint_training_elapsed_seconds(checkpoint: dict) -> float:
    timer_states = [
        value
        for key, value in checkpoint.get("callbacks", {}).items()
        if key == "Timer" or str(key).startswith("Timer{")
    ]
    if len(timer_states) != 1:
        raise RuntimeError("resume checkpoint must contain exactly one Timer state")
    elapsed = float(timer_states[0].get("time_elapsed", {}).get("train", -1))
    if elapsed < 0:
        raise RuntimeError("resume checkpoint has no accumulated training time")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("coordinate", "intrinsic"))
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline-metadata", type=Path)
    parser.add_argument("--baseline-comparison", type=Path)
    parser.add_argument("--expansion-authorization", type=Path)
    parser.add_argument("--repeat-authorization", type=Path)
    parser.add_argument("--resume-last", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["input"].get("audited_holdout_allowed", False):
        raise RuntimeError("refusing an audited-holdout-enabled config")
    variant, pool_output = resolve_protocol(config, args.variant)
    configured_model_seed = int(config.get("model_seed", config.get("seed", 20260818)))
    seed = args.seed if args.seed is not None else configured_model_seed
    data_seed = int(config.get("data_seed", config.get("seed", 20260818)))
    verified_baseline = None
    if config["experiment"] == "EXP-0007":
        repeat_seeds = tuple(int(value) for value in config["decision"]["repeat_model_seeds"])
        if seed not in (configured_model_seed, *repeat_seeds):
            raise ValueError("EXP-0007 model seed is outside the frozen primary/repeat set")
        if seed != configured_model_seed:
            if args.repeat_authorization is None:
                raise RuntimeError("repeat model seeds require a primary near-gate authorization")
            repeat = verify_decision_authorization(
                args.repeat_authorization, args.config, purpose="repeat"
            )
            if (
                not repeat.get("authorize_repeat_seeds", False)
                or seed not in repeat.get("authorized_repeat_model_seeds", [])
            ):
                raise RuntimeError("decision artifact does not authorize this repeat seed")
        elif args.repeat_authorization is not None:
            raise ValueError("primary model seed must not use --repeat-authorization")
        if args.fold != 2:
            if args.expansion_authorization is None:
                raise RuntimeError("folds 0/1 require a passing fold-2 authorization")
            expansion = verify_decision_authorization(
                args.expansion_authorization, args.config, purpose="fold expansion"
            )
            if (
                not expansion.get("authorize_additional_folds", False)
                or int(expansion.get("additional_folds_model_seed", -1)) != seed
            ):
                raise RuntimeError("decision artifact does not authorize this fold/model seed")
        elif args.expansion_authorization is not None:
            raise ValueError("fold 2 must not use --expansion-authorization")
        if args.baseline_metadata is None:
            raise ValueError("EXP-0007 requires --baseline-metadata built before training")
        verified_baseline = verify_baseline(
            args.baseline_metadata,
            args.config,
            fold=args.fold,
            data_seed=data_seed,
        )
    elif args.baseline_metadata is not None:
        raise ValueError("--baseline-metadata applies only to EXP-0007")
    elif args.expansion_authorization is not None or args.repeat_authorization is not None:
        raise ValueError("EXP-0007 decision authorizations do not apply to EXP-0004")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("preflight.py")), "--require-cuda"],
        check=True,
    )
    L.seed_everything(seed, workers=True)
    maximum_steps = args.max_steps if args.max_steps is not None else int(config["training"]["maximum_steps"])
    if maximum_steps < 1 or maximum_steps > int(config["training"]["maximum_steps"]):
        raise ValueError("max-steps override must be within the configured bounded budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, proxy_validation, tier_c_validation = make_datasets(
        config["input"]["proxy_hdf5"],
        config["input"]["proxy_sha256"],
        fold=args.fold,
        seed=data_seed,
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
        variant,
        image_height=int(config["input"]["height"]),
        image_width=int(config["input"]["width"]),
        num_points=int(config["input"]["body_points"]),
        intrinsic_coefficients=int(config["model"]["intrinsic_coefficients"]),
        anchor_index=int(config["model"]["body_anchor_index"]),
        encoder_pool_output=pool_output,
        model_seed=seed,
        data_seed=data_seed,
    )
    if sum(parameter.numel() for parameter in module.parameters()) > int(config["model"]["maximum_parameters"]):
        raise RuntimeError("model exceeds configured parameter ceiling")
    latest_checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{variant}-fold{args.fold}-step{{step}}",
        monitor=None,
        save_top_k=1,
        save_last=True,
        every_n_train_steps=int(config["training"]["checkpoint_every_steps"]),
        save_on_train_epoch_end=False,
    )
    best_checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{variant}-fold{args.fold}-best",
        monitor=(
            "val_tier_c_fully_visible_angle_mae_degrees"
            if config["experiment"] == "EXP-0007"
            else "val_tier_c_angle_mae_degrees"
        ),
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )
    resume_checkpoint = resolve_resume_checkpoint(
        args.output_dir, resume_last=args.resume_last, resume_from=args.resume_from
    )
    resumed = None
    resumed_elapsed_seconds = 0.0
    if resume_checkpoint is not None:
        resumed = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        resumed_hparams = resumed.get("hyper_parameters", {})
        resumed_pool = tuple(resumed_hparams.get("encoder_pool_output", (2, 2)))
        if resumed_hparams.get("variant") != variant or resumed_pool != pool_output:
            raise RuntimeError("resume checkpoint variant/pool identity mismatch")
        if config["experiment"] == "EXP-0007" and (
            int(resumed_hparams.get("model_seed", -1)) != seed
            or int(resumed_hparams.get("data_seed", -1)) != data_seed
        ):
            raise RuntimeError("resume checkpoint model/data seed identity mismatch")
        if (
            config["experiment"] == "EXP-0007"
            and int(resumed.get("global_step", -1)) >= int(config["early_elimination"]["checkpoint_step"])
            and not (args.output_dir / "step300.ckpt").is_file()
        ):
            raise RuntimeError("resume past step 300 requires the immutable step300.ckpt")
        resumed_elapsed_seconds = checkpoint_training_elapsed_seconds(resumed)
        maximum_duration_seconds = 60 * float(config["resources"]["maximum_minutes_per_run"])
        if resumed_elapsed_seconds >= maximum_duration_seconds:
            raise RuntimeError("resume checkpoint has exhausted the cumulative wall-time budget")
    if config["experiment"] == "EXP-0007":
        checkpoint_step = int(config["early_elimination"]["checkpoint_step"])
        if resume_checkpoint is None and maximum_steps > checkpoint_step:
            raise RuntimeError("fresh EXP-0007 training must stop at step 300 for external comparison")
        if resume_checkpoint is not None and maximum_steps > checkpoint_step:
            if args.baseline_comparison is None:
                raise RuntimeError("resume beyond step 300 requires --baseline-comparison")
            comparison = json.loads(args.baseline_comparison.read_text())
            immutable_path = args.output_dir / "step300.ckpt"
            if (
                comparison.get("decision") != "CONTINUE_TO_1200"
                or int(comparison.get("fold", -1)) != args.fold
                or int(comparison.get("model_seed", -1)) != seed
                or int(comparison.get("data_seed", -1)) != data_seed
                or comparison.get("config_sha256") != sha256_file(args.config)
                or comparison.get("baseline_metadata_sha256")
                != sha256_file(args.baseline_metadata)
                or comparison.get("baseline_tensor_sha256")
                != verified_baseline["tensor_sha256"]
                or comparison.get("checkpoint_sha256") != sha256_file(immutable_path)
                or Path(comparison.get("checkpoint_path", "")).resolve()
                != immutable_path.resolve()
                or resume_checkpoint.resolve() != immutable_path.resolve()
                or int(resumed.get("global_step", -1)) != checkpoint_step
            ):
                raise RuntimeError("baseline comparison does not authorize this resume")
            if args.resume_last:
                raise RuntimeError("EXP-0007 continuation must use --resume-from step300.ckpt")
        elif args.baseline_comparison is not None:
            raise ValueError("--baseline-comparison applies only when resuming beyond step 300")
    elif args.baseline_comparison is not None:
        raise ValueError("--baseline-comparison applies only to EXP-0007")
    timer = Timer(duration=timedelta(minutes=float(config["resources"]["maximum_minutes_per_run"])))
    if resumed_elapsed_seconds:
        print(
            json.dumps(
                {
                    "cumulative_train_seconds_before_resume": resumed_elapsed_seconds,
                    "maximum_cumulative_seconds": 60
                    * float(config["resources"]["maximum_minutes_per_run"]),
                }
            )
        )
    callbacks: list[Callback] = [latest_checkpoint, best_checkpoint, timer]
    if config["experiment"] == "EXP-0007":
        callbacks.append(
            ImmutableStepCheckpoint(
                args.output_dir / "step300.ckpt",
                int(config["early_elimination"]["checkpoint_step"]),
            )
        )
        callbacks.append(
            FullyVisibleCountContract(
                int(config["training"]["expected_fully_visible_validation_cases"])
            )
        )
    logger = (
        CSVLogger(save_dir=str(args.output_dir), name="csv", version="")
        if config["experiment"] == "EXP-0007"
        else False
    )
    trainer = L.Trainer(
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        precision=config["training"]["precision"],
        max_steps=maximum_steps,
        max_epochs=int(config["training"]["maximum_epochs"]),
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        deterministic=True,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(
        module,
        train_loader,
        [proxy_validation_loader, tier_c_validation_loader],
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
