#!/usr/bin/env python3
"""Evaluate every saved segmenter checkpoint on the validation and test splits.

By default this finds ``best.ckpt`` and ``last.ckpt`` in every run directory
under ``<checkpoint dir>/runs/`` (``--checkpoint`` evaluates specific files
instead).  For each checkpoint it reports per-sample IoU, Dice, precision,
and recall over non-ignored pixels, compares with the classical threshold on
the same frames, and writes overlay sheets for the worst samples.
Bootstrapped labels are not truth, so read the numbers as agreement with the
current labels; the hand-refined subset is the part that matters.

One invocation is one evaluation session, kept under
``<checkpoint dir>/evaluations/<session time>/<run>__<best|last>/``.  Each
``evaluation.json`` holds the time, git revision, checkpoint fingerprint, the
run record the checkpoint came from, a fingerprint of the labels used, the
exact membership of each split (sample id, label source, revision, save
time), the summary, and per-sample metrics.  One summary line per checkpoint
is appended to ``<checkpoint dir>/evaluations/history.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import torch

from worm_pose_gen.classical import ClassicalConfig, segment_dark_ridge
from worm_pose_gen.mask_fit import fill_narrow_holes
from worm_pose_gen.run_records import checkpoint_fingerprint, git_revision, split_manifest, timestamp_slug, utc_now
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore, is_hand_labeled
from worm_pose_gen.segmenter import IGNORE_LABEL, load_segmenter, masked_binary_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "segmenter"
METRIC_NAMES = ("iou", "dice", "precision", "recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint", type=Path, action="append", default=None, help="evaluate only these files")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <checkpoint dir>/evaluations")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--worst", type=int, default=6)
    parser.add_argument("--note", default="", help="free text stored with every record of the session")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None}
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(len(array)), "median": float(np.median(array)), "mean": float(array.mean()), "min": float(array.min())}


def summarize_rows(rows: list[dict]) -> dict:
    manual = [r for r in rows if is_hand_labeled(r["label_source"])]
    empty = [r for r in rows if r["label_pixels"] == 0]
    return {
        "samples": len(rows),
        "empty_label_samples": len(empty),
        "empty_label_network_false_positive_pixels": summarize([float(r["network_pixels"]) for r in empty]),
        "empty_label_classical_false_positive_pixels": summarize([float(r["classical_pixels"]) for r in empty]),
        "network": {k: summarize([r["network"][k] for r in rows]) for k in METRIC_NAMES},
        "classical": {k: summarize([r["classical"][k] for r in rows]) for k in METRIC_NAMES},
        "hand_refined_samples": len(manual),
        "hand_refined_network_iou": summarize([r["network"]["iou"] for r in manual]),
        "hand_refined_classical_iou": summarize([r["classical"]["iou"] for r in manual]),
        "network_beats_classical": int(sum(r["network"]["iou"] > r["classical"]["iou"] for r in rows)),
        "hand_refined_network_beats_classical": int(sum(r["network"]["iou"] > r["classical"]["iou"] for r in manual)),
    }


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


def find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    found = []
    for run_dir in sorted((checkpoint_dir / "runs").glob("*/")):
        for kind in ("best", "last"):
            path = run_dir / f"{kind}.ckpt"
            if path.exists():
                found.append(path)
    return found


def checkpoint_label(path: Path) -> str:
    """``<run dir>__<best|last>`` for run checkpoints, else the file stem."""

    run_record = path.parent / "run.json"
    return f"{path.parent.name}__{path.stem}" if run_record.exists() or path.parent.name.startswith("20") else path.stem


def run_summary(path: Path) -> dict | None:
    record_path = path.parent / "run.json"
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    return {key: record.get(key) for key in ("name", "run_dir", "started_at", "train_labels", "counts", "init_checkpoint", "best_val_iou", "best_val_loss", "best_epoch", "epochs_run")}


def labels_fingerprint(store: SegmentationStore, splits: list[str]) -> str:
    rows = [(r.sample_id, r.revision, r.split) for split in splits for r in store.records(split)]
    return hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()


def main() -> int:
    args = parse_args()
    session_at = utc_now()
    evaluations_dir = args.output_dir or args.checkpoint_dir / "evaluations"
    session_dir = evaluations_dir / timestamp_slug(session_at)
    checkpoints = args.checkpoint or find_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise SystemExit(f"no checkpoints under {args.checkpoint_dir / 'runs'}")
    store = SegmentationStore(args.dataset_root)
    cfg = ClassicalConfig()
    git = git_revision(PROJECT_ROOT)
    membership = split_manifest(store, tuple(args.splits))
    fingerprint = labels_fingerprint(store, args.splits)

    # Labels and the classical baseline are shared by every checkpoint.
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples: dict[str, list[tuple]] = {}
    for split in args.splits:
        rows = []
        for record in store.records(split):
            image, mask, _ = store.load(record.sample_id)
            classical, _ = fill_narrow_holes(segment_dark_ridge(image, cfg).component, 8, device=device)
            valid_np = mask != IGNORE_LABEL
            valid = torch.as_tensor(valid_np)[None]
            target = torch.as_tensor((mask == 1).astype(np.float32))[None]
            baseline = {k: float(v[0]) for k, v in masked_binary_metrics(torch.as_tensor(classical.astype(np.float32))[None], target, valid).items()}
            rows.append((record, image, mask, classical, valid, target, baseline))
        samples[split] = rows

    for checkpoint in checkpoints:
        label = checkpoint_label(checkpoint)
        output_dir = session_dir / label
        output_dir.mkdir(parents=True, exist_ok=True)
        module = load_segmenter(checkpoint, args.device)
        report: dict[str, object] = {
            "evaluated_at": utc_now(),
            "session": session_at,
            "checkpoint_label": label,
            "note": args.note,
            "git": git,
            "checkpoint": checkpoint_fingerprint(checkpoint),
            "run": run_summary(checkpoint),
            "dataset_root": str(store.root),
            "labels_fingerprint": fingerprint,
            "store_counts": store.counts(),
            "split_membership": membership,
            "splits": {},
        }
        for split in args.splits:
            rows = []
            cache = {}
            for record, image, mask, classical, valid, target, baseline in samples[split]:
                probability = module.predict_probability(image)
                valid_np = mask != IGNORE_LABEL
                network = {k: float(v[0]) for k, v in masked_binary_metrics(torch.as_tensor(probability)[None], target, valid).items()}
                row = {
                    "sample_id": record.sample_id, "recording": record.recording, "frame_index": record.frame_index,
                    "label_source": record.label_source, "revision": record.revision, "saved_at": record.saved_at,
                    "label_pixels": int(((mask == 1) & valid_np).sum()),
                    "network_pixels": int(((probability >= 0.5) & valid_np).sum()),
                    "classical_pixels": int((classical & valid_np).sum()),
                    "network": network, "classical": baseline,
                }
                rows.append(row)
                cache[record.sample_id] = (image, mask, probability, classical)
            summary = summarize_rows(rows)
            for row in sorted(rows, key=lambda r: r["network"]["iou"])[: args.worst]:
                image, mask, probability, classical = cache[row["sample_id"]]
                plot_sample(image, mask, probability, classical, row, output_dir / f"{split}_{row['sample_id']}.png")
            report["splits"][split] = {"summary": summary, "per_sample": rows}
            hand = summary["hand_refined_network_iou"]
            print(
                f"{label:48s} {split:4s} n={summary['samples']:3d} network median {summary['network']['iou']['median']:.3f} "
                f"min {summary['network']['iou']['min']:.3f} | hand-refined n={hand['n']} median "
                f"{hand['median'] if hand['median'] is None else round(hand['median'], 3)} | classical median "
                f"{summary['classical']['iou']['median']:.3f}"
            )
        (output_dir / "evaluation.json").write_text(json.dumps(report, indent=1))
        history_line = {
            "evaluated_at": report["evaluated_at"],
            "session": session_at,
            "checkpoint_label": label,
            "record": str(output_dir / "evaluation.json"),
            "checkpoint_sha256": (report["checkpoint"] or {}).get("sha256"),
            "checkpoint_modified_at": (report["checkpoint"] or {}).get("modified_at"),
            "labels_fingerprint": fingerprint,
            "git_commit": git["commit"],
            "note": args.note,
            "splits": {split: report["splits"][split]["summary"] for split in args.splits},
        }
        with open(evaluations_dir / "history.jsonl", "a") as handle:
            handle.write(json.dumps(history_line) + "\n")
        del module
    print(f"session: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
