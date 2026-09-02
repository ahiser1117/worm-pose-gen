#!/usr/bin/env python3
"""Evaluate a segmenter checkpoint on the validation and test splits.

Reports per-sample IoU, Dice, precision, and recall over non-ignored pixels,
plus how the network compares with the classical threshold on the same
frames, and writes overlay sheets for the worst samples.  Bootstrapped
labels are not truth, so read the numbers as agreement with the current
labels; the hand-refined subset is the part that matters.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import torch

from worm_pose_gen.classical import ClassicalConfig, segment_dark_ridge
from worm_pose_gen.mask_fit import fill_narrow_holes
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore
from worm_pose_gen.segmenter import IGNORE_LABEL, load_segmenter, masked_binary_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "best.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <checkpoint dir>/evaluation")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--worst", type=int, default=6)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None}
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(len(array)), "median": float(np.median(array)), "mean": float(array.mean()), "min": float(array.min())}


def plot_sample(image: np.ndarray, mask: np.ndarray, probability: np.ndarray, classical: np.ndarray, row: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    lower, upper = np.percentile(image, [1, 99])
    for ax in axes:
        ax.imshow(image, cmap="gray", vmin=lower, vmax=upper)
        ax.set_axis_off()
    label_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    label_rgba[mask == 1] = (1.0, 0.31, 0.64, 0.45)
    label_rgba[mask == IGNORE_LABEL] = (1.0, 0.82, 0.24, 0.35)
    axes[0].imshow(label_rgba, interpolation="nearest")
    axes[0].set_title(f"{row['sample_id']} label (ignore in yellow)", fontsize=9)
    axes[1].imshow(probability, cmap="magma", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title(f"network probability, IoU {row['network']['iou']:.3f}", fontsize=9)
    residual = np.zeros((*mask.shape, 4), dtype=np.float32)
    valid = mask != IGNORE_LABEL
    prediction = probability >= 0.5
    residual[valid & (mask == 1) & ~prediction] = (1.0, 0.31, 0.64, 0.7)
    residual[valid & (mask != 1) & prediction] = (0.34, 0.84, 0.55, 0.7)
    axes[2].imshow(residual, interpolation="nearest")
    axes[2].set_title("network errors: magenta missed, green extra", fontsize=9)
    residual_c = np.zeros((*mask.shape, 4), dtype=np.float32)
    residual_c[valid & (mask == 1) & ~classical] = (1.0, 0.31, 0.64, 0.7)
    residual_c[valid & (mask != 1) & classical] = (0.34, 0.84, 0.55, 0.7)
    axes[3].imshow(residual_c, interpolation="nearest")
    axes[3].set_title(f"classical errors, IoU {row['classical']['iou']:.3f}", fontsize=9)
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.checkpoint.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    store = SegmentationStore(args.dataset_root)
    module = load_segmenter(args.checkpoint, args.device)
    cfg = ClassicalConfig()
    report: dict[str, object] = {"checkpoint": str(args.checkpoint), "dataset_root": str(store.root), "splits": {}}
    for split in args.splits:
        rows = []
        cache = {}
        for record in store.records(split):
            image, mask, _ = store.load(record.sample_id)
            probability = module.predict_probability(image)
            classical, _ = fill_narrow_holes(segment_dark_ridge(image, cfg).component, 8, device=module.device)
            valid = torch.as_tensor(mask != IGNORE_LABEL)[None]
            target = torch.as_tensor((mask == 1).astype(np.float32))[None]
            network = {k: float(v[0]) for k, v in masked_binary_metrics(torch.as_tensor(probability)[None], target, valid).items()}
            baseline = {k: float(v[0]) for k, v in masked_binary_metrics(torch.as_tensor(classical.astype(np.float32))[None], target, valid).items()}
            row = {
                "sample_id": record.sample_id, "recording": record.recording, "frame_index": record.frame_index,
                "label_source": record.label_source, "revision": record.revision,
                "network": network, "classical": baseline,
            }
            rows.append(row)
            cache[record.sample_id] = (image, mask, probability, classical)
        manual = [r for r in rows if "manual" in r["label_source"]]
        summary = {
            "samples": len(rows),
            "network": {k: summarize([r["network"][k] for r in rows]) for k in ("iou", "dice", "precision", "recall")},
            "classical": {k: summarize([r["classical"][k] for r in rows]) for k in ("iou", "dice", "precision", "recall")},
            "hand_refined_samples": len(manual),
            "hand_refined_network_iou": summarize([r["network"]["iou"] for r in manual]),
            "hand_refined_classical_iou": summarize([r["classical"]["iou"] for r in manual]),
            "network_beats_classical": int(sum(r["network"]["iou"] > r["classical"]["iou"] for r in rows)),
        }
        worst = sorted(rows, key=lambda r: r["network"]["iou"])[: args.worst]
        for row in worst:
            image, mask, probability, classical = cache[row["sample_id"]]
            plot_sample(image, mask, probability, classical, row, output_dir / f"{split}_{row['sample_id']}.png")
        report["splits"][split] = {"summary": summary, "per_sample": rows}
        print(json.dumps({split: summary}, indent=1))
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
