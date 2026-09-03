#!/usr/bin/env python3
"""Plot the training curves and the evaluation history of the segmenter.

Reads what the training and evaluation scripts leave behind in the checkpoint
directory (``runs/*/run.json`` with ``metrics.csv``, and
``evaluations/<session>/<checkpoint>/evaluation.json``) plus the dataset
index, and writes five figures to ``<checkpoint dir>/plots/``:

``training_curves.png``        loss and validation IoU per epoch, one line per run
``checkpoint_comparison.png``  hand-refined IoU of every checkpoint in the newest session
``evaluation_history.png``     hand-refined median IoU per checkpoint across sessions
``latest_evaluation.png``      per-sample IoU of every checkpoint in the newest session
``dataset_growth.png``         labels in the store over time, by source
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
    """Training runs from ``runs/*/run.json``, with their per-epoch metrics."""

    runs = []
    for record_path in sorted((checkpoint_dir / "runs").glob("*/run.json")):
        record = json.loads(record_path.read_text())
        metrics_path = record_path.parent / "metrics.csv"
        counts = record.get("counts", {})
        label = f"{record.get('name', record_path.parent.name)}  train {counts.get('train_used', counts.get('train', '?'))}"
        if record.get("init_checkpoint"):
            label += "  (from checkpoint)"
        runs.append({
            "label": label,
            "metrics": read_metrics_csv(metrics_path) if metrics_path.exists() else {},
            "record": record,
        })
    return runs


def short_label(checkpoint_label: str) -> str:
    """``2026-09-03T21-00-00Z_all_labels__best`` -> ``all_labels/best``."""

    run, _, kind = checkpoint_label.partition("__")
    parts = run.split("_", 1)
    name = parts[1] if len(parts) == 2 and parts[0].endswith("Z") else run
    return f"{name}/{kind}" if kind else name


def load_evaluations(checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Every evaluation record, oldest session first."""

    reports = [json.loads(p.read_text()) for p in (checkpoint_dir / "evaluations").glob("*/*/evaluation.json")]
    for report in reports:
        report.setdefault("session", report["evaluated_at"])
        report.setdefault("checkpoint_label", "checkpoint")
    return sorted(reports, key=lambda r: (r["session"], r["checkpoint_label"]))


def latest_session(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not evaluations:
        return []
    session = evaluations[-1]["session"]
    return [e for e in evaluations if e["session"] == session]


def nonempty_minimum(evaluation: dict[str, Any], split: str) -> float:
    """Lowest hand-refined network IoU among frames whose label has worm pixels."""

    rows = evaluation["splits"].get(split, {}).get("per_sample", [])
    values = [r["network"]["iou"] for r in rows if "manual" in r["label_source"] and r.get("label_pixels", 1) > 0]
    return float(min(values)) if values else np.nan


def hand_refined(summary: dict[str, Any], metric: str = "median") -> float:
    value = summary["hand_refined_network_iou"][metric]
    return np.nan if value is None else float(value)


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


def _split_axes(evaluations: list[dict[str, Any]], width: float, height: float):
    splits = [s for s in ("val", "test") if any(s in e["splits"] for e in evaluations)] or ["val", "test"]
    fig, axes = plt.subplots(1, len(splits), figsize=(width * len(splits), height), constrained_layout=True, sharey=False)
    return splits, fig, np.atleast_1d(axes)


def _empty(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", color=INK_SOFT, transform=ax.transAxes)


def plot_checkpoint_comparison(session: list[dict[str, Any]], path: Path) -> None:
    """Hand-refined IoU of every checkpoint in the newest session, one row each."""

    splits, fig, axes = _split_axes(session, 6.0, max(2.6, 0.5 * len(session) + 1.4))
    if session:
        order = sorted(session, key=lambda e: hand_refined(e["splits"][splits[0]]["summary"]) if splits[0] in e["splits"] else -1)
        labels = [short_label(e["checkpoint_label"]) for e in order]
        fig.suptitle(f"evaluation session {_iso(order[0]['session']).strftime('%Y-%m-%d %H:%M')} UTC", color=INK_SOFT, fontsize=9)
    for ax, split in zip(axes, splits):
        if not session:
            _empty(ax, "no evaluations yet")
        else:
            y = np.arange(len(order))
            rows = [e["splits"].get(split, {}).get("summary") for e in order]
            median = np.array([hand_refined(r) if r else np.nan for r in rows])
            low = np.array([nonempty_minimum(e, split) for e in order])
            classical = [r["hand_refined_classical_iou"]["median"] for r in rows if r and r["hand_refined_classical_iou"]["median"] is not None]
            n = next((r["hand_refined_samples"] for r in rows if r), 0)
            ax.hlines(y, low, median, color=NETWORK, linewidth=1.5, alpha=0.5)
            ax.plot(median, y, linestyle="none", marker="o", color=NETWORK, label="network median (line to minimum over non-empty frames)")
            if classical:
                ax.axvline(classical[0], color=CLASSICAL, linestyle="--", linewidth=1.5, label="classical median")
            for yi, value in zip(y, median):
                if not np.isnan(value):
                    ax.annotate(f"{value:.3f}", (value, yi), textcoords="offset points", xytext=(6, 5), fontsize=7, color=INK_SOFT)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlim(min(0.5, float(np.nanmin(low)) - 0.02) if np.isfinite(np.nanmin(low)) else 0.5, 1.02)
            ax.set_title(f"{split}: hand-refined IoU over {n} samples")
            ax.set_xlabel("IoU")
    if session:
        axes[-1].legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_evaluation_history(evaluations: list[dict[str, Any]], path: Path) -> None:
    """Hand-refined median IoU of each checkpoint across evaluation sessions."""

    splits, fig, axes = _split_axes(evaluations, 6.2, 3.8)
    labels: list[str] = []
    for e in evaluations:
        if e["checkpoint_label"] not in labels:
            labels.append(e["checkpoint_label"])
    shown = labels[-len(SERIES):]
    for ax, split in zip(axes, splits):
        if not evaluations:
            _empty(ax, "no evaluations yet")
        else:
            classical_points = []
            for color, label in zip(SERIES, shown):
                rows = [e for e in evaluations if e["checkpoint_label"] == label and split in e["splits"]]
                if not rows:
                    continue
                t = [_iso(e["session"]) for e in rows]
                ax.plot(t, [hand_refined(e["splits"][split]["summary"]) for e in rows], color=color, marker="o", label=short_label(label))
                classical_points += [(x, e["splits"][split]["summary"]["hand_refined_classical_iou"]["median"]) for x, e in zip(t, rows)]
            classical_points = sorted({(x, v) for x, v in classical_points if v is not None})
            if classical_points:
                ax.plot([p[0] for p in classical_points], [p[1] for p in classical_points], color=CLASSICAL, linestyle="--",
                        marker="o", markerfacecolor="white", label="classical")
            times = sorted({_iso(e["session"]) for e in evaluations})
            span = max((times[-1] - times[0]) * 0.08, timedelta(hours=1))
            ax.set_xlim(times[0] - span, times[-1] + span)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_ha("right")
            lowest = min([np.nanmin(line.get_ydata()) for line in ax.get_lines() if len(line.get_ydata())] + [0.5])
            ax.set_ylim(min(0.5, float(np.nan_to_num(lowest, nan=0.5)) - 0.02), 1.02)
        ax.set_title(f"{split}: hand-refined median IoU per session")
    axes[0].set_ylabel("IoU")
    if evaluations:
        axes[-1].legend(loc="lower left", fontsize=8)
        if len(labels) > len(shown):
            fig.suptitle(f"last {len(shown)} of {len(labels)} checkpoints", color=INK_SOFT)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_latest_evaluation(session: list[dict[str, Any]], path: Path) -> None:
    """Per-sample IoU of every checkpoint in the newest session."""

    splits = [s for s in ("val", "test") if any(s in e["splits"] for e in session)] or ["val", "test"]
    fig, axes = plt.subplots(len(splits), 1, figsize=(12, 3.4 * len(splits)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    if not session:
        _empty(axes[0], "no evaluations yet")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return
    shown = session[-len(SERIES):]
    fig.suptitle(f"evaluation session {_iso(session[0]['session']).strftime('%Y-%m-%d %H:%M')} UTC", color=INK_SOFT, fontsize=9)
    for ax, split in zip(axes, splits):
        reference = next((e for e in shown if split in e["splits"]), None)
        if reference is None:
            _empty(ax, f"no {split} evaluation")
            continue
        base = sorted(reference["splits"][split]["per_sample"], key=lambda r: r["classical"]["iou"])
        ids = [r["sample_id"] for r in base]
        x = np.arange(len(ids))
        ax.plot(x, [r["classical"]["iou"] for r in base], linestyle="none", marker="o", markerfacecolor="white",
                color=CLASSICAL, label="classical")
        for color, e in zip(SERIES, shown):
            if split not in e["splits"]:
                continue
            by_id = {r["sample_id"]: r for r in e["splits"][split]["per_sample"]}
            ax.plot(x, [by_id[i]["network"]["iou"] if i in by_id else np.nan for i in ids], linestyle="none", marker="o",
                    markersize=4.5, color=color, alpha=0.85, label=short_label(e["checkpoint_label"]))
        for n, (xi, r) in enumerate(zip(x, base)):
            if r.get("label_pixels") == 0:
                ax.annotate("empty", (xi, 1.0), textcoords="offset points", xytext=(0, 6 + 8 * (n % 2)), ha="center", fontsize=6, color=INK_SOFT)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("IoU")
        lowest = min([np.nanmin(line.get_ydata()) for line in ax.get_lines() if len(line.get_ydata())] + [0.5])
        ax.set_ylim(min(0.5, float(np.nan_to_num(lowest, nan=0.5)) - 0.02), 1.06)
        ax.set_title(f"{split}: {len(ids)} samples, sorted by classical IoU")
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)
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
    session = latest_session(evaluations)
    outputs = [
        output_dir / "training_curves.png",
        output_dir / "checkpoint_comparison.png",
        output_dir / "evaluation_history.png",
        output_dir / "latest_evaluation.png",
        output_dir / "dataset_growth.png",
    ]
    plot_training_curves(runs, outputs[0])
    plot_checkpoint_comparison(session, outputs[1])
    plot_evaluation_history(evaluations, outputs[2])
    plot_latest_evaluation(session, outputs[3])
    plot_dataset_growth(SegmentationStore(dataset_root), outputs[4])
    return outputs


def main() -> int:
    args = parse_args()
    outputs = make_plots(args.checkpoint_dir, args.dataset_root, args.output_dir or args.checkpoint_dir / "plots")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
