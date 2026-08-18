#!/usr/bin/env python3
"""Synchronized CUDA latency, throughput, and memory benchmark."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import lightning
import numpy as np
import torch

from preflight import query_physical_gpu_zero
from worm_pose_gen.model import WormProposalModule
from worm_pose_gen.training_data import normalize_image, sha256_file


WARMUP_ITERATIONS = 10


def summary(milliseconds: list[float], batch_size: int) -> dict[str, float]:
    values = np.asarray(milliseconds)
    return {
        "batch_size": batch_size,
        "p50_milliseconds": float(np.quantile(values, .50)),
        "p95_milliseconds": float(np.quantile(values, .95)),
        "samples_per_second_from_total": float(batch_size * len(values) / (values.sum() / 1000)),
    }


@torch.inference_mode()
def measure_forward(model, batch_size: int, iterations: int) -> dict[str, float]:
    device = next(model.parameters()).device
    image = torch.rand(batch_size, 1, 192, 256, device=device)
    for _ in range(WARMUP_ITERATIONS): model(image)
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(iterations):
        torch.cuda.synchronize(device); started = time.perf_counter(); model(image); torch.cuda.synchronize(device)
        timings.append(1000 * (time.perf_counter() - started))
    return {**summary(timings, batch_size), "peak_memory_bytes": torch.cuda.max_memory_allocated(device)}


@torch.inference_mode()
def measure_pipeline(model, batch_size: int, iterations: int) -> tuple[dict, dict]:
    raw = np.zeros((batch_size, 732, 968), dtype=np.uint8)
    preprocessing, end_to_end = [], []
    for _ in range(iterations):
        started = time.perf_counter()
        batch = torch.stack([normalize_image(image) for image in raw])
        preprocessing.append(1000 * (time.perf_counter() - started))
        batch = batch.cuda(non_blocking=False); model(batch); torch.cuda.synchronize()
        end_to_end.append(1000 * (time.perf_counter() - started))
    return summary(preprocessing, batch_size), summary(end_to_end, batch_size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output: {args.output}")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("preflight.py")), "--require-cuda"],
        check=True,
    )
    if not torch.cuda.is_available(): raise RuntimeError("benchmark requires CUDA")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WormProposalModule.load_from_checkpoint(args.checkpoint, map_location="cuda").cuda().eval()
    properties = torch.cuda.get_device_properties(0)
    physical_gpu = query_physical_gpu_zero()
    preprocess1, end1 = measure_pipeline(model, 1, args.iterations)
    preprocess_batch, end_batch = measure_pipeline(model, args.batch_size, args.iterations)
    result = {
        "gpu": {
            "logical_device": 0,
            "physical_device": physical_gpu,
            "mapping": {"physical_index": physical_gpu["physical_index"], "visible_logical_index": 0},
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "lightning": lightning.__version__,
        },
        "checkpoint": {"path": str(args.checkpoint.resolve(strict=True)), "sha256": sha256_file(args.checkpoint)},
        "variant": model.variant,
        "encoder_pool_output": list(model.encoder.pool_output),
        "model_seed": model.hparams.get("model_seed"),
        "data_seed": model.hparams.get("data_seed"),
        "fold": args.fold,
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "protocol": {
            "warmup_iterations": WARMUP_ITERATIONS,
            "measured_iterations": args.iterations,
            "precision": "float32",
            "input": "grayscale uint8 732x968 for preprocessing/end-to-end; float32 [B,1,192,256] in [0,1] for forward-only",
            "cuda_synchronization": "before and after every measured forward/end-to-end iteration",
        },
        "forward_batch1": measure_forward(model, 1, args.iterations),
        "forward_batched": measure_forward(model, args.batch_size, args.iterations),
        "preprocessing_batch1": preprocess1,
        "end_to_end_batch1": end1,
        "preprocessing_batched": preprocess_batch,
        "end_to_end_batched": end_batch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
