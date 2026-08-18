from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260818


class PhaseZeroSmokeModule(L.LightningModule):
    """Minimal Lightning path used only to validate data-to-GPU plumbing."""

    def __init__(self, n_body: int = 16) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, n_body * 2),
        )

    def forward(self, frames: Tensor) -> Tensor:
        return self.network(frames).reshape(frames.shape[0], -1, 2)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        frames, target = batch
        return torch.nn.functional.smooth_l1_loss(self(frames), target)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("nir_videos/2023-09-19-01.h5"),
    )
    parser.add_argument("--dataset", default="img_nir")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    L.seed_everything(SEED, workers=True, verbose=False)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke test requires exactly one visible CUDA device")

    with h5py.File(args.input, "r") as handle:
        dataset = handle[args.dataset]
        frame = np.asarray(dataset[args.frame_index], dtype=np.float32)

    # A bounded strided view is sufficient for validating the pipeline and
    # avoids treating this infrastructure smoke run as a scientific result.
    frame = np.ascontiguousarray(frame[::8, ::8])
    frame -= float(frame.min())
    scale = max(float(frame.max()), 1.0)
    frame /= scale
    frames = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).repeat(8, 1, 1, 1)
    target = torch.zeros((8, 16, 2), dtype=torch.float32)
    loader = DataLoader(TensorDataset(frames, target), batch_size=4, shuffle=False)

    model = PhaseZeroSmokeModule()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=args.steps,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    start = time.perf_counter()
    trainer.fit(model, train_dataloaders=loader)
    elapsed = time.perf_counter() - start

    with torch.inference_mode():
        prediction = model(frames[:1].to(model.device)).detach().cpu()
    report = {
        "status": "PASS",
        "input": str(args.input.resolve()),
        "dataset": args.dataset,
        "frame_index": args.frame_index,
        "input_shape": list(frame.shape),
        "prediction_shape": list(prediction.shape),
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
