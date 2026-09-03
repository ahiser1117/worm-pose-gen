#!/usr/bin/env python3
"""Plot the training curves and the evaluation history of the segmenter.

Reads what the training and evaluation scripts leave behind in the checkpoint
directory (``logs/version_*/metrics.csv``, ``runs/*.json``,
``evaluations/*/evaluation.json``) plus the dataset index, and writes four
figures to ``<checkpoint dir>/plots/``:

``training_curves.png``      loss and validation IoU per epoch, one line per run
``evaluation_history.png``   validation and test IoU of every saved evaluation
``latest_evaluation.png``    per-sample IoU of the newest evaluation
``dataset_growth.png``       labels in the store over time, by source
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worm-pose-gen-matplotlib")
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "segmenter"

# Categorical slots in fixed order (validated palette); series keep their slot.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NETWORK, CLASSICAL, HAND = SERIES[0], SERIES[1], SERIES[2]
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "axes.labelcolor": INK_SOFT,
        "text.color": INK, "axes.titlecolor": INK, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <checkpoint dir>/plots")
    return parser.parse_args()


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


# ----------------------------------------------------------------------------- inputs

def read_metrics_csv(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Columns of a Lightning CSV log as ``name -> [(x, value), ...]``.

    Step-level columns are keyed by step; epoch-level columns by epoch.
    """

    series: dict[str, list[tuple[float, float]]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            for name, text in row.items():
                if name in ("epoch", "step") or text in ("", None):
                    continue
                x = float(row["step"]) if name.endswith("_step") else float(row["epoch"])
                series.setdefault(name, []).append((x, float(text)))
    return series


def load_runs(checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Training runs: one per Lightning log version, joined to its run record if any."""

    records = {}
    for path in sorted((checkpoint_dir / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("log_dir"):
            records[Path(record["log_dir"]).resolve()] = record
    runs = []
    for log_dir in sorted((checkpoint_dir / "logs").glob("version_*"), key=lambda p: int(p.name.split("_")[-1])):
        metrics = log_dir / "metrics.csv"
        if not metrics.exists():
            continue
        record = records.get(log_dir.resolve())
        if record is not None:
            label = f"{_iso(record['started_at']).strftime('%Y-%m-%d %H:%M')}  train {record['counts']['train']}"
            if record.get("init_checkpoint"):
                label += "  (from checkpoint)"
        else:
            label = log_dir.name
        runs.append({"label": label, "metrics": read_metrics_csv(metrics), "record": record})
    return runs


def load_evaluations(checkpoint_dir: Path) -> list[dict[str, Any]]:
    reports = [json.loads(p.read_text()) for p in (checkpoint_dir / "evaluations").glob("*/evaluation.json")]
    return sorted(reports, key=lambda r: r["evaluated_at"])


# ----------------------------------------------------------------------------- figures

def plot_training_curves(runs: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    panels = (("train_loss_epoch", "training loss"), ("val_loss", "validation loss"), ("val_iou", "validation IoU"))
    shown = runs[-len(SERIES):]
    if len(runs) > len(shown):
        fig.suptitle(f"last {len(shown)} of {len(runs)} runs", color=INK_SOFT)
    for ax, (name, title) in zip(axes, panels):
        for color, run in zip(SERIES, shown):
            points = run["metrics"].get(name, [])
            if not points:
                continue
            x, y = zip(*points)
            ax.plot(x, y, color=color, label=run["label"], marker="o", markersize=3.5)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        if name == "val_iou":
            ax.set_ylim(min(0.7, ax.get_ylim()[0]), 1.0)
    if shown:
        axes[0].legend(loc="upper right", fontsize=8)
    if not any(run["metrics"] for run in shown):
        axes[1].text(0.5, 0.5, "no training logs yet", ha="center", va="center", color=INK_SOFT, transform=axes[1].transAxes)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_evaluation_history(evaluations: list[dict[str, Any]], path: Path) -> None:
    splits = [s for s in ("val", "test") if any(s in e["splits"] for e in evaluations)] or ["val", "test"]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 3.8), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    times = [_iso(e["evaluated_at"]) for e in evaluations]
    for ax, split in zip(axes, splits):
        rows = [(t, e["splits"][split]["summary"]) for t, e in zip(times, evaluations) if split in e["splits"]]
        if rows:
            t = [r[0] for r in rows]
            def series(key: str, metric: str = "median") -> list[float]:
                return [np.nan if r[1][key][metric] is None else r[1][key][metric] for r in rows]
            ax.plot(t, series("hand_refined_network_iou"), color=NETWORK, marker="o", label="network, hand-refined labels")
            ax.plot(t, series("hand_refined_classical_iou"), color=CLASSICAL, marker="o", label="classical, hand-refined labels")
            ax.plot(t, [r[1]["network"]["iou"]["median"] for r in rows], color=NETWORK, linestyle=":", marker="o",
                    markerfacecolor="white", label="network, all labels")
            ax.plot(t, series("hand_refined_network_iou", "min"), color=NETWORK, linestyle="none", marker="_",
                    markersize=10, label="network, hand-refined minimum")
            for x, r in rows:
                ax.annotate(f"n={r['hand_refined_samples']}", (x, 0.0), xycoords=("data", "axes fraction"),
                            textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7, color=INK_SOFT)
            span = max((t[-1] - t[0]) * 0.08, timedelta(hours=1))
            ax.set_xlim(t[0] - span, t[-1] + span)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_ha("right")
        else:
            ax.text(0.5, 0.5, "no evaluations yet", ha="center", va="center", color=INK_SOFT, transform=ax.transAxes)
        ax.set_title(f"{split} split: median IoU per evaluation")
        lowest = min([line.get_ydata().min() for line in ax.get_lines() if len(line.get_ydata())] + [0.5])
        ax.set_ylim(min(0.5, float(np.nan_to_num(lowest, nan=0.5)) - 0.02), 1.02)
    axes[0].set_ylabel("IoU")
    if evaluations:
        axes[-1].legend(loc="lower left", fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_latest_evaluation(evaluation: dict[str, Any] | None, path: Path) -> None:
    splits = list(evaluation["splits"]) if evaluation else ["val", "test"]
    fig, axes = plt.subplots(len(splits), 1, figsize=(11, 3.2 * len(splits)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    if evaluation is None:
        axes[0].text(0.5, 0.5, "no evaluations yet", ha="center", va="center", color=INK_SOFT, transform=axes[0].transAxes)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return
    checkpoint = evaluation.get("checkpoint") or {}
    fig.suptitle(
        f"evaluated {_iso(evaluation['evaluated_at']).strftime('%Y-%m-%d %H:%M')} UTC, "
        f"checkpoint {str(checkpoint.get('sha256', ''))[:10]} ({checkpoint.get('modified_at', '?')})",
        color=INK_SOFT, fontsize=9,
    )
    for ax, split in zip(axes, splits):
        rows = sorted(evaluation["splits"][split]["per_sample"], key=lambda r: r["network"]["iou"])
        x = np.arange(len(rows))
        hand = np.array(["manual" in r["label_source"] for r in rows])
        network = np.array([r["network"]["iou"] for r in rows])
        classical = np.array([r["classical"]["iou"] for r in rows])
        ax.vlines(x, np.minimum(network, classical), np.maximum(network, classical), color=GRID, linewidth=1.5)
        ax.plot(x, classical, linestyle="none", marker="o", markerfacecolor="white", color=CLASSICAL, label="classical")
        ax.plot(x[~hand], network[~hand], linestyle="none", marker="o", color=NETWORK, label="network, bootstrap label")
        ax.plot(x[hand], network[hand], linestyle="none", marker="D", color=HAND, label="network, hand-refined label")
        for xi, r in zip(x, rows):
            if r.get("label_pixels") == 0:
                ax.annotate("empty label", (xi, network[xi]), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=6, color=INK_SOFT)
        ax.set_xticks(x)
        ax.set_xticklabels([r["sample_id"] for r in rows], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("IoU")
        ax.set_ylim(min(0.5, float(np.nanmin(np.concatenate([network, classical]))) - 0.02) if len(rows) else 0.5, 1.02)
        summary = evaluation["splits"][split]["summary"]
        hand_median = summary["hand_refined_network_iou"]["median"]
        ax.set_title(
            f"{split}: {len(rows)} samples, network median {summary['network']['iou']['median']:.3f}"
            + (f", hand-refined median {hand_median:.3f} over {summary['hand_refined_samples']}" if hand_median is not None else "")
        )
    axes[0].legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_dataset_growth(store: SegmentationStore, path: Path) -> None:
    records = sorted(store.records(), key=lambda r: r.saved_at)
    fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    if records:
        times = [_iso(r.saved_at) for r in records]
        hand = np.array(["manual" in r.label_source for r in records])
        ax.step(times, np.arange(1, len(records) + 1), where="post", color=NETWORK, label="all labels")
        ax.step(times, np.cumsum(hand), where="post", color=HAND, label="hand-refined labels")
        eval_split = np.array([r.split != "train" for r in records]) & hand
        ax.step(times, np.cumsum(eval_split), where="post", color=CLASSICAL, label="hand-refined in val or test")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "store is empty", ha="center", va="center", color=INK_SOFT, transform=ax.transAxes)
    ax.set_title("labels in the store by last save time")
    ax.set_ylabel("samples")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_plots(checkpoint_dir: Path, dataset_root: Path, output_dir: Path) -> list[Path]:
    style()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(checkpoint_dir)
    evaluations = load_evaluations(checkpoint_dir)
    outputs = [
        output_dir / "training_curves.png",
        output_dir / "evaluation_history.png",
        output_dir / "latest_evaluation.png",
        output_dir / "dataset_growth.png",
    ]
    plot_training_curves(runs, outputs[0])
    plot_evaluation_history(evaluations, outputs[1])
    plot_latest_evaluation(evaluations[-1] if evaluations else None, outputs[2])
    plot_dataset_growth(SegmentationStore(dataset_root), outputs[3])
    return outputs


def main() -> int:
    args = parse_args()
    outputs = make_plots(args.checkpoint_dir, args.dataset_root, args.output_dir or args.checkpoint_dir / "plots")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
