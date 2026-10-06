#!/usr/bin/env python3
"""Compare body-field network runs per sample, split by self-contact, and time inference.

For every run directory given, the run's ``best.ckpt`` predicts each sample of
the evaluated splits (validation and test by default; both are held-out
recordings and each is small).  Per sample it records mask IoU, mean A-P
error over mask pixels with a defined target, head and tail peak errors in
pixels, false-positive pixels (what an empty frame is judged by), and
whether the predicted head peak is nearer the true head than
the true tail.  Samples are grouped by whether their fitted body touches or
crosses itself (:func:`body_targets.self_contact`), which is where the
temporal channels are meant to help.

Throughput is measured on full 732 x 968 frames at ``--batch-size`` with
the inputs already on the GPU (the network alone; context reads are the
caller's), so runs with different lag sets are directly comparable.

Writes ``<out>/per_sample.csv`` and ``<out>/summary.json`` and prints the
summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from worm_pose_gen import body_fields
from worm_pose_gen.body_net import BodyFieldDataset, load_body_net
from worm_pose_gen.body_targets import self_contact
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore
from worm_pose_gen.segmenter import masked_binary_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="run directories holding best.ckpt and run.json")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "checkpoints" / "body_net" / "evaluations" / "latest")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def peak(values: np.ndarray) -> np.ndarray:
    y, x = np.unravel_index(int(np.argmax(values)), values.shape)
    return np.array([x, y], dtype=np.float64)


@torch.inference_mode()
def throughput(module, batch_size: int, device: torch.device) -> float:
    inputs = torch.randn(batch_size, module.network.in_channels, 732, 968, device=device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        for _ in range(3):
            module(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        repeats = 10
        for _ in range(repeats):
            module(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return repeats * batch_size / (time.perf_counter() - start)


@torch.inference_mode()
def evaluate_run(run_dir: Path, store: SegmentationStore, splits: list[str], device: torch.device) -> list[dict]:
    module = load_body_net(run_dir / "best.ckpt", device)
    rows = []
    for split in splits:
        dataset = BodyFieldDataset(store, split, module.lags)
        for index in range(len(dataset)):
            item = dataset[index]
            sample_id = item["sample_id"]
            fields, meta = body_fields.load(body_fields.field_path(store.root, sample_id))
            image = item["image"][None].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = module(image)[0].float()
            targets = item["targets"].to(device)
            probability = torch.sigmoid(logits[0])
            iou = float(masked_binary_metrics(probability[None], targets[0][None], targets[1][None])["iou"][0])
            false_positive_px = int(((probability >= 0.5) & (targets[0] < 0.5) & (targets[1] > 0)).sum())
            row = {
                "run": run_dir.name, "split": split, "sample_id": sample_id, "iou": iou, "false_positive_px": false_positive_px,
                "has_body": bool(meta.get("has_body")), "fit_iou": meta.get("fit_iou"),
                "contact": bool(meta.get("has_body")) and self_contact(fields["centerline_xy"], fields["width_profile"]),
                "ap_mae": np.nan, "head_error_px": np.nan, "tail_error_px": np.nan, "orientation_correct": np.nan,
            }
            ap_valid = targets[3] > 0
            if ap_valid.any():
                row["ap_mae"] = float((torch.sigmoid(logits[1]) - targets[2]).abs()[ap_valid].mean())
                head_truth, tail_truth = fields["head_xy"], fields["tail_xy"]
                head_map = targets[4].cpu().numpy()
                tail_map = targets[5].cpu().numpy()
                head_pred, tail_pred = peak(logits[2].cpu().numpy()), peak(logits[3].cpu().numpy())
                if head_map.max() >= 0.95:
                    row["head_error_px"] = float(np.linalg.norm(head_pred - head_truth))
                if tail_map.max() >= 0.95:
                    row["tail_error_px"] = float(np.linalg.norm(tail_pred - tail_truth))
                if head_map.max() >= 0.95 and tail_map.max() >= 0.95:
                    row["orientation_correct"] = float(
                        np.linalg.norm(head_pred - head_truth) < np.linalg.norm(head_pred - tail_truth)
                    )
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    def stats(subset: list[dict]) -> dict:
        def mean(key):
            values = np.array([r[key] for r in subset], dtype=np.float64)
            values = values[np.isfinite(values)]
            return None if not len(values) else float(values.mean())

        def median(key):
            values = np.array([r[key] for r in subset], dtype=np.float64)
            values = values[np.isfinite(values)]
            return None if not len(values) else float(np.median(values))

        return {
            "n": len(subset), "iou_mean": mean("iou"), "iou_median": median("iou"), "ap_mae": mean("ap_mae"),
            "head_error_median_px": median("head_error_px"), "tail_error_median_px": median("tail_error_px"),
            "head_error_mean_px": mean("head_error_px"), "orientation_accuracy": mean("orientation_correct"),
        }

    empty = [r for r in rows if not r["has_body"]]
    return {
        "empty": {"n": len(empty), "false_positive_px_median": float(np.median([r["false_positive_px"] for r in empty])) if empty else None},
        "all": stats(rows),
        "contact": stats([r for r in rows if r["contact"]]),
        "clear": stats([r for r in rows if r["has_body"] and not r["contact"]]),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    store = SegmentationStore(args.dataset_root)
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, summary = [], {}
    for run_dir in args.runs:
        rows = evaluate_run(run_dir, store, args.splits, device)
        all_rows += rows
        module = load_body_net(run_dir / "best.ckpt", device)
        summary[run_dir.name] = {
            "lags": list(module.lags),
            "frames_per_second": throughput(module, args.batch_size, device),
            **summarize(rows),
        }
    with open(args.out / "per_sample.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out / "summary.json").write_text(json.dumps({"splits": args.splits, "runs": summary}, indent=1))
    header = f"{'run':40s} {'lags':16s} {'fps':>6s} {'group':8s} {'n':>3s} {'IoUmed':>6s} {'AP':>6s} {'head':>6s} {'tail':>6s} {'orient':>6s}"
    print(header)
    for name, entry in summary.items():
        print(f"{name[-40:]:40s} empty frames: {entry['empty']['n']}, median false-positive px {entry['empty']['false_positive_px_median']}")
        for group in ("all", "contact", "clear"):
            s = entry[group]
            def f(value, fmt):
                return format(value, fmt) if value is not None else "-"
            print(
                f"{name[-40:]:40s} {','.join(map(str, entry['lags'])) or 'none':16s} {entry['frames_per_second']:6.0f} "
                f"{group:8s} {s['n']:3d} {f(s['iou_median'], '6.3f')} {f(s['ap_mae'], '6.3f')} "
                f"{f(s['head_error_median_px'], '6.1f')} {f(s['tail_error_median_px'], '6.1f')} {f(s['orientation_accuracy'], '6.2f')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
