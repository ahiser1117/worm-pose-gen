#!/usr/bin/env python3
"""Build and apply the frozen EXP-0007 mean-centerline elimination baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import torch
import yaml

try:  # Supports both direct CLI execution and package-style test imports.
    from scripts.evaluate import evaluate
except ModuleNotFoundError:  # pragma: no cover - direct-script import path
    from evaluate import evaluate
from worm_pose_gen.model import WormProposalModule
from worm_pose_gen.training_data import (
    SyntheticTierCDataset,
    sha256_file,
)


SCHEMA_VERSION = 1
TENSOR_FILENAME = "mean_centerline.npy"
METADATA_FILENAME = "baseline.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_SOURCES = (
    "scripts/exp_0007_baseline.py",
    "scripts/evaluate.py",
    "src/worm_pose_gen/geometry.py",
    "src/worm_pose_gen/model.py",
    "src/worm_pose_gen/renderer.py",
    "src/worm_pose_gen/synthetic.py",
    "src/worm_pose_gen/training_data.py",
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash canonical little-endian float32 raw C-order bytes."""

    value = tensor.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def repository_provenance(*, require_clean: bool) -> dict[str, Any]:
    """Bind generated identities to the exact clean implementation tree."""

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if require_clean and status:
        raise RuntimeError("EXP-0007 provenance requires a clean committed repository")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "git_tree": tree,
        "repository_clean": not bool(status),
        "source_sha256": {
            name: sha256_file(REPOSITORY_ROOT / name) for name in PROVENANCE_SOURCES
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "yaml": yaml.__version__,
        },
    }


def validate_exp_0007_config(config: dict[str, Any]) -> None:
    if config.get("experiment") != "EXP-0007":
        raise ValueError("baseline builder requires an EXP-0007 config")
    if config["input"].get("audited_holdout_allowed", False):
        raise RuntimeError("audited holdout must remain forbidden")
    if config["model"].get("variant") != "intrinsic":
        raise ValueError("EXP-0007 baseline requires the intrinsic-only protocol")
    if list(config["model"].get("encoder_pool_output", ())) != [4, 4]:
        raise ValueError("EXP-0007 encoder_pool_output must be [4, 4]")
    if int(config.get("data_seed", -1)) != 20260818:
        raise ValueError("EXP-0007 data_seed must be fixed at 20260818")
    training = config["training"]
    if int(training["synthetic_samples_per_epoch"]) != 512:
        raise ValueError("EXP-0007 baseline requires exactly 512 training geometries")
    if int(training["synthetic_validation_samples"]) != 128:
        raise ValueError("EXP-0007 baseline requires exactly 128 validation geometries")
    early = config["early_elimination"]
    expected = {
        "checkpoint_step": 300,
        "baseline_definition": "pointwise_arithmetic_mean_in_generator_order",
        "baseline_training_stratum": "exact_512_returned_development_tier_c_centerline_xy_targets_including_deterministic_crops",
        "validation_stratum": "fully_visible_cases_from_exact_128_held_out_tier_c_cases",
        "orientation_alignment": "symmetric_forward_reverse_only_during_error_evaluation",
        "metric": "median_per_point_euclidean_error_original_image_pixels_968x732",
    }
    for key, value in expected.items():
        if early.get(key) != value:
            raise ValueError(f"EXP-0007 early_elimination.{key} must be {value!r}")
    exact = {
        "tensor_dtype": "little_endian_float32",
        "tensor_shape": [100, 2],
        "tensor_hash_serialization": "raw_c_order_bytes",
        "artifact_serialization": "numpy_npy_allow_pickle_false",
        "require_checkpoint_global_step_exact": 300,
    }
    for key, value in exact.items():
        if early.get(key) != value:
            raise ValueError(f"EXP-0007 early_elimination.{key} must be {value!r}")
    if int(training.get("expected_fully_visible_validation_cases", -1)) != 43:
        raise ValueError("EXP-0007 requires exactly 43 fully-visible validation cases")


def symmetric_original_pixel_errors(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-point error after one forward/reverse choice per case."""

    scale = target.new_tensor((968 / 256, 732 / 192))
    forward = torch.linalg.vector_norm((prediction - target) * scale, dim=-1)
    reverse = torch.linalg.vector_norm((prediction - target.flip(-2)) * scale, dim=-1)
    choose_reverse = reverse.mean(-1) < forward.mean(-1)
    return torch.where(choose_reverse[..., None], reverse, forward)


def construct_baseline(config: dict[str, Any], *, fold: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct the frozen tensor from exact dataset returns, retaining no images."""

    validate_exp_0007_config(config)
    if fold not in tuple(int(value) for value in config["training"]["folds"]):
        raise ValueError("fold is outside the configured EXP-0007 folds")
    data_seed = int(config.get("data_seed", 20260818))
    if data_seed != 20260818:
        raise ValueError("EXP-0007 data_seed is frozen at 20260818")
    training_seed = data_seed + fold * 100_000
    validation_seed = data_seed + 5_000_000 + fold * 100_000
    training_count = int(config["training"]["synthetic_samples_per_epoch"])
    validation_count = int(config["training"]["synthetic_validation_samples"])
    training_dataset = SyntheticTierCDataset(
        training_count,
        seed=training_seed,
        profile=config["input"]["synthetic_train_profile"],
    )
    training_centerlines: list[torch.Tensor] = []
    ordered_training_seeds: list[int] = []
    for index in range(training_count):
        sample = training_dataset[index]
        training_centerlines.append(sample["centerline_xy"])
        ordered_training_seeds.append(int(sample["sample_seed"]))
    training_tensor = torch.stack(training_centerlines).to(torch.float32)
    mean_centerline = training_tensor.mean(0)
    validation_targets: list[torch.Tensor] = []
    validation_case_ids: list[str] = []
    validation_dataset = SyntheticTierCDataset(
        validation_count,
        seed=validation_seed,
        profile=config["input"]["synthetic_validation_profile"],
    )
    ordered_validation_seeds: list[int] = []
    for index in range(validation_count):
        sample = validation_dataset[index]
        ordered_validation_seeds.append(int(sample["sample_seed"]))
        if bool(sample["image_support_target"].all()):
            validation_targets.append(sample["centerline_xy"])
            validation_case_ids.append(f"seed:{int(sample['sample_seed'])}")
    targets = torch.stack(validation_targets).to(torch.float32)
    if len(validation_case_ids) != int(config["training"]["expected_fully_visible_validation_cases"]):
        raise RuntimeError("fully-visible validation identity count changed")
    prediction = mean_centerline.expand(len(targets), -1, -1)
    point_error = symmetric_original_pixel_errors(prediction, targets)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "EXP-0007",
        "fold": fold,
        "data_seed": data_seed,
        "construction": {
            "definition": "pointwise_arithmetic_mean_in_generator_order",
            "dtype": "float32",
            "byte_order": "little_endian",
            "raw_hash_layout": "C_order",
            "shape": [100, 2],
            "coordinate_units": "render_pixels_256x192",
            "training_profile": config["input"]["synthetic_train_profile"],
            "training_seed_base": training_seed,
            "training_count": training_count,
            "ordered_training_sample_seeds": ordered_training_seeds,
            "training_case_ids": [f"seed:{value}" for value in ordered_training_seeds],
        },
        "validation": {
            "profile": config["input"]["synthetic_validation_profile"],
            "validation_seed_base": validation_seed,
            "total_candidate_count": validation_count,
            "fully_visible_count": len(validation_case_ids),
            "fully_visible_case_ids": validation_case_ids,
            "ordered_validation_sample_seeds": ordered_validation_seeds,
            "orientation_alignment": "symmetric_forward_reverse_per_case",
            "metric": "median_per_point_euclidean_error_original_image_pixels_968x732",
            "median_point_error_px": float(point_error.median()),
            "fully_visible_target_tensor_shape": list(targets.shape),
            "fully_visible_target_tensor_sha256": tensor_sha256(targets),
        },
        "audited_holdout_opened": False,
        "source_recordings_opened": False,
    }
    return mean_centerline, metadata


def write_baseline(config_path: Path, *, fold: int, output_dir: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    provenance = repository_provenance(require_clean=True)
    tensor, metadata = construct_baseline(config, fold=fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / TENSOR_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    if tensor_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite an existing EXP-0007 baseline")
    canonical = tensor.detach().cpu().numpy().astype("<f4", copy=False)
    with tensor_path.open("xb") as handle:
        np.save(handle, canonical, allow_pickle=False)
    metadata.update(
        {
            "config_path": str(config_path.resolve(strict=True)),
            "config_sha256": sha256_file(config_path),
            "tensor_file": TENSOR_FILENAME,
            "tensor_sha256": tensor_sha256(tensor),
            "tensor_file_sha256": sha256_file(tensor_path),
            "code_provenance": provenance,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path


def verify_baseline(metadata_path: Path, config_path: Path, *, fold: int, data_seed: int) -> dict:
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("experiment") != "EXP-0007":
        raise RuntimeError("unsupported baseline metadata")
    if int(metadata.get("fold", -1)) != fold or int(metadata.get("data_seed", -1)) != data_seed:
        raise RuntimeError("baseline fold/data-seed identity mismatch")
    if metadata.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("baseline config hash mismatch")
    tensor_path = metadata_path.parent / metadata["tensor_file"]
    if sha256_file(tensor_path) != metadata["tensor_file_sha256"]:
        raise RuntimeError("baseline tensor-file hash mismatch")
    with tensor_path.open("rb") as handle:
        array = np.load(handle, allow_pickle=False)
    if array.dtype != np.dtype("<f4") or tuple(array.shape) != (100, 2):
        raise RuntimeError("baseline NumPy artifact contract mismatch")
    tensor = torch.from_numpy(array.astype(np.float32, copy=False))
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != (100, 2):
        raise RuntimeError("baseline tensor contract mismatch")
    if tensor_sha256(tensor) != metadata["tensor_sha256"]:
        raise RuntimeError("baseline raw tensor hash mismatch")
    if metadata.get("code_provenance") != repository_provenance(require_clean=True):
        raise RuntimeError("baseline implementation/environment provenance mismatch")
    _, reconstructed = construct_baseline(yaml.safe_load(config_path.read_text()), fold=fold)
    expected_validation = reconstructed["validation"]
    for name in (
        "fully_visible_case_ids",
        "fully_visible_target_tensor_shape",
        "fully_visible_target_tensor_sha256",
        "median_point_error_px",
    ):
        if metadata["validation"].get(name) != expected_validation.get(name):
            raise RuntimeError(f"baseline reconstructed validation field changed: {name}")
    return metadata


def compare_checkpoint(
    config_path: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    *,
    fold: int,
    output_path: Path,
    device: str = "cpu",
    model_seed: int | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    validate_exp_0007_config(config)
    data_seed = int(config.get("data_seed", 20260818))
    metadata = verify_baseline(metadata_path, config_path, fold=fold, data_seed=data_seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_step = int(config["early_elimination"]["checkpoint_step"])
    if int(checkpoint.get("global_step", -1)) != expected_step:
        raise RuntimeError(f"comparison requires the exact step-{expected_step} checkpoint")
    module = WormProposalModule.load_from_checkpoint(checkpoint_path, map_location=device).to(device).eval()
    if module.variant != "intrinsic" or tuple(module.encoder.pool_output) != (4, 4):
        raise RuntimeError("checkpoint is not the EXP-0007 intrinsic 4x4 model")
    checkpoint_model_seed = int(module.hparams.get("model_seed", -1))
    expected_model_seed = (
        int(model_seed)
        if model_seed is not None
        else int(config.get("model_seed", 20260818))
    )
    if checkpoint_model_seed != expected_model_seed or int(module.hparams.get("data_seed", -1)) != data_seed:
        raise RuntimeError("checkpoint model/data seed identity mismatch")
    validation = SyntheticTierCDataset(
        int(config["training"]["synthetic_validation_samples"]),
        seed=data_seed + 5_000_000 + fold * 100_000,
        profile=config["input"]["synthetic_validation_profile"],
    )
    cases, _ = evaluate(
        module,
        validation,
        int(config["training"]["evaluation_batch_size"]),
        torch.device(device),
    )
    fully_visible = [case for case in cases if bool(case["aligned_support"].all())]
    case_ids = [case["case_id"] for case in fully_visible]
    if case_ids != metadata["validation"]["fully_visible_case_ids"]:
        raise RuntimeError("checkpoint comparison case identities differ from frozen baseline")
    model_median = float(torch.cat([case["point"] for case in fully_visible]).median())
    baseline_median = float(metadata["validation"]["median_point_error_px"])
    eliminated = model_median >= baseline_median
    result = {
        "experiment": "EXP-0007",
        "fold": fold,
        "model_seed": expected_model_seed,
        "data_seed": data_seed,
        "checkpoint_path": str(checkpoint_path.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_global_step": expected_step,
        "baseline_metadata_path": str(metadata_path.resolve(strict=True)),
        "baseline_metadata_sha256": sha256_file(metadata_path),
        "baseline_tensor_sha256": metadata["tensor_sha256"],
        "config_sha256": sha256_file(config_path),
        "code_provenance": metadata["code_provenance"],
        "fully_visible_case_ids": case_ids,
        "baseline_median_point_error_px": baseline_median,
        "model_median_point_error_px": model_median,
        "eliminate": eliminated,
        "decision": "ELIMINATE" if eliminated else "CONTINUE_TO_1200",
        "comparison_operator": "eliminate_if_model_greater_than_or_equal_to_baseline",
        "audited_holdout_opened": False,
        "source_recordings_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite a baseline comparison")
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--baseline-metadata", type=Path, required=True)
    compare.add_argument("--checkpoint", type=Path, required=True)
    compare.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    compare.add_argument("--model-seed", type=int)
    args = parser.parse_args()
    if args.command == "build":
        path = write_baseline(args.config, fold=args.fold, output_dir=args.output_dir)
        print(path.read_text(), end="")
    else:
        result = compare_checkpoint(
            args.config,
            args.baseline_metadata,
            args.checkpoint,
            fold=args.fold,
            output_path=args.output,
            device=args.device,
            model_seed=args.model_seed,
        )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
