#!/usr/bin/env python3
"""Plot the training curves and the evaluation history of the segmenter.

Reads what the training and evaluation scripts leave behind in the checkpoint
directory (``runs/*/run.json`` with ``metrics.csv``, and
``evaluations/<session>/<checkpoint>/evaluation.json``) plus the dataset
index, and writes seven figures to ``<checkpoint dir>/plots/``:

``training_curves.png``        loss and validation IoU per epoch, one line per run, best epoch marked
``checkpoint_comparison.png``  median, interquartile range, and minimum IoU of every best checkpoint in the newest session
``evaluation_history.png``     median IoU per model across evaluation sessions
``latest_evaluation.png``      per-sample IoU of the headline models, grouped by recording
``model_delta.png``            per-sample IoU of the newest model minus its reference model
``iou_ecdf.png``               cumulative distribution of per-sample IoU per headline model, val and test pooled
``dataset_growth.png``         labels in the store over time, and per recording by split

Models are named by ``docs/segmenter_model_names.json`` (``--names``): a
short display name per run directory and a ``headline`` list of the models
drawn in colour in every figure, oldest first.  Runs outside the list are
drawn in grey; the last headline model is the newest and the one before it
is its reference (``--model`` and ``--reference`` override).
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

from worm_pose_gen.run_records import checkpoint_fingerprint
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SPLITS, SegmentationStore, is_hand_labeled


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "segmenter"
DEFAULT_NAMES = PROJECT_ROOT / "docs" / "segmenter_model_names.json"

# Categorical slots in fixed order (validated palette); a headline model keeps its slot in every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
CLASSICAL = "#7a7873"
OTHER = "#c4c2bc"
SPLIT_COLORS = {"train": "#9fb9de", "val": "#2a78d6", "test": "#eb6834"}
INK, INK_SOFT, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#f1f0ec"
DPI = 130


def style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "axes.labelcolor": INK_SOFT,
        "text.color": INK, "axes.titlecolor": INK, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 5.5,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES, help="model display names and the headline list")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <checkpoint dir>/plots")
    parser.add_argument("--model", default=None, help="newest model (display name); default: last headline model evaluated")
    parser.add_argument("--reference", default=None, help="reference model (display name); default: the headline model before --model")
    parser.add_argument("--include-last", action="store_true", help="also show last-epoch checkpoints in the comparison")
    return parser.parse_args()


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


# ----------------------------------------------------------------------------- names

class ModelNames:
    """Display names of runs and the ordered headline list, with a colour and marker per headline model."""

    def __init__(self, names: dict[str, str], headline: list[str]) -> None:
        self.names = dict(names)
        self.headline = list(headline)

    @classmethod
    def load(cls, path: Path | None) -> "ModelNames":
        if path is None or not path.exists():
            return cls({}, [])
        data = json.loads(path.read_text())
        return cls(data.get("names", {}), data.get("headline", []))

    def run_name(self, run_dir: str, fallback: str | None = None) -> str:
        """Display name of a run directory (``2026-..Z_<name>``), else its ``--name``."""

        if run_dir in self.names:
            return self.names[run_dir]
        if fallback:
            return fallback
        parts = run_dir.split("_", 1)
        return parts[1] if len(parts) == 2 and parts[0].endswith("Z") else run_dir

    def checkpoint_name(self, checkpoint_label: str) -> str:
        """``<run dir>__best`` -> model name; ``__last`` keeps a ``/last`` suffix."""

        run, _, kind = checkpoint_label.partition("__")
        name = self.run_name(run)
        return name if kind in ("", "best") else f"{name}/{kind}"

    def is_headline(self, name: str) -> bool:
        return name in self.headline

    def color(self, name: str) -> str:
        return SERIES[self.headline.index(name) % len(SERIES)] if name in self.headline else OTHER

    def marker(self, name: str) -> str:
        return MARKERS[self.headline.index(name) % len(MARKERS)] if name in self.headline else "o"


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


def load_runs(checkpoint_dir: Path, names: ModelNames) -> list[dict[str, Any]]:
    """Training runs from ``runs/*/run.json``, with their per-epoch metrics, oldest first."""

    runs = []
    for record_path in sorted((checkpoint_dir / "runs").glob("*/run.json")):
        record = json.loads(record_path.read_text())
        metrics_path = record_path.parent / "metrics.csv"
        counts = record.get("counts", {})
        name = names.run_name(record_path.parent.name, record.get("name"))
        label = f"{name}  ({counts.get('train_used', counts.get('train', '?'))} train labels"
        label += ", warm start)" if record.get("init_checkpoint") else ")"
        runs.append({
            "name": name,
            "label": label,
            "metrics": read_metrics_csv(metrics_path) if metrics_path.exists() else {},
            "record": record,
        })
    return runs


def load_evaluations(checkpoint_dir: Path, names: ModelNames) -> list[dict[str, Any]]:
    """Every evaluation record, oldest session first, with ``name`` and ``kind`` (best/last) attached."""

    reports = [json.loads(p.read_text()) for p in (checkpoint_dir / "evaluations").glob("*/*/evaluation.json")]
    for report in reports:
        report.setdefault("session", report["evaluated_at"])
        report.setdefault("checkpoint_label", "checkpoint")
        report["name"] = names.checkpoint_name(report["checkpoint_label"])
        report["kind"] = report["checkpoint_label"].partition("__")[2] or "best"
    return sorted(reports, key=lambda r: (r["session"], r["checkpoint_label"]))


def latest_session(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not evaluations:
        return []
    session = evaluations[-1]["session"]
    return [e for e in evaluations if e["session"] == session]


def promoted_sha(checkpoint_dir: Path) -> str | None:
    fingerprint = checkpoint_fingerprint(checkpoint_dir / "best.ckpt")
    return None if not fingerprint or not fingerprint.get("exists", True) else fingerprint.get("sha256")


def hand_rows(evaluation: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [r for r in evaluation["splits"].get(split, {}).get("per_sample", []) if is_hand_labeled(r["label_source"])]


def iou_values(evaluation: dict[str, Any], split: str, *, nonempty: bool = False) -> np.ndarray:
    rows = hand_rows(evaluation, split)
    if nonempty:
        rows = [r for r in rows if r.get("label_pixels", 1) > 0]
    return np.asarray([r["network"]["iou"] for r in rows], dtype=np.float64)


def classical_values(evaluation: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray([r["classical"]["iou"] for r in hand_rows(evaluation, split)], dtype=np.float64)


def choose_models(session: list[dict[str, Any]], names: ModelNames, model: str | None, reference: str | None) -> tuple[str | None, str | None]:
    """The newest headline model with an evaluation in the session, and the headline model before it."""

    present = [n for n in names.headline if any(e["name"] == n for e in session)]
    newest = model or (present[-1] if present else (session[-1]["name"] if session else None))
    if reference is None and newest in present and present.index(newest) > 0:
        reference = present[present.index(newest) - 1]
    return newest, reference


def by_name(session: list[dict[str, Any]], name: str | None) -> dict[str, Any] | None:
    return next((e for e in session if e["name"] == name), None) if name else None


def _splits_present(evaluations: list[dict[str, Any]]) -> list[str]:
    return [s for s in ("val", "test") if any(s in e["splits"] for e in evaluations)] or ["val", "test"]


def _empty(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", color=INK_SOFT, transform=ax.transAxes)


def _date_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")


# ----------------------------------------------------------------------------- figures

def plot_training_curves(runs: list[dict[str, Any]], names: ModelNames, path: Path) -> None:
    """Loss and validation IoU per epoch; headline runs in colour with their best epoch starred, others grey."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), constrained_layout=True)
    panels = (("train_loss_epoch", "training loss (log)"), ("val_loss", "validation loss (log)"), ("val_iou", "validation IoU"))
    ordered = [r for r in runs if not names.is_headline(r["name"])] + [r for r in runs if names.is_headline(r["name"])]
    for ax, (metric, title) in zip(axes, panels):
        for run in ordered:
            points = run["metrics"].get(metric, [])
            if not points:
                continue
            x, y = map(np.asarray, zip(*points))
            headline = names.is_headline(run["name"])
            color = names.color(run["name"])
            ax.plot(x, y, color=color, linewidth=1.9 if headline else 1.0, alpha=1.0 if headline else 0.8,
                    label=run["label"] if headline else None, zorder=3 if headline else 2)
            best_epoch = run["record"].get("best_epoch")
            if headline and best_epoch is not None and metric != "train_loss_epoch":
                hit = np.nonzero(x == best_epoch)[0]
                if len(hit):
                    ax.plot(x[hit[0]], y[hit[0]], marker="*", markersize=13, color=color, markeredgecolor="white", zorder=4)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        if metric == "val_iou":
            ax.set_ylim(0.85, 1.0)
        else:
            ax.set_yscale("log")
    others = [r["name"] for r in runs if not names.is_headline(r["name"])]
    handles, labels = axes[0].get_legend_handles_labels()
    if others:
        handles.append(matplotlib.lines.Line2D([], [], color=OTHER, linewidth=1.0))
        labels.append(f"{len(others)} other runs (grey)")
    if handles:
        axes[1].legend(handles, labels, loc="upper right", fontsize=7.5, title="star = selected epoch", title_fontsize=7.5)
    if not any(run["metrics"] for run in runs):
        _empty(axes[1], "no training logs yet")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_checkpoint_comparison(session: list[dict[str, Any]], names: ModelNames, promoted: str | None, path: Path, include_last: bool) -> None:
    """Median (dot), interquartile range (bar), and minimum over non-empty frames (line) per checkpoint."""

    shown = [e for e in session if include_last or e["kind"] == "best"]
    splits = _splits_present(shown)
    fig, axes = plt.subplots(1, len(splits), figsize=(6.4 * len(splits), max(2.8, 0.42 * len(shown) + 1.6)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    if not shown:
        for ax in axes:
            _empty(ax, "no evaluations yet")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        return
    order = sorted(shown, key=lambda e: np.median(iou_values(e, splits[0])) if len(iou_values(e, splits[0])) else -1)
    labels = [e["name"] + ("   promoted" if promoted and (e.get("checkpoint") or {}).get("sha256") == promoted else "") for e in order]
    y = np.arange(len(order))
    for ax, split in zip(axes, splits):
        classical = None
        lowest = 1.0
        for yi, e in zip(y, order):
            values = iou_values(e, split)
            if not len(values):
                continue
            base = names.color(e["name"]) if names.is_headline(e["name"].split("/")[0]) else OTHER
            color = base if e["kind"] == "best" else OTHER
            low = float(np.min(iou_values(e, split, nonempty=True))) if len(iou_values(e, split, nonempty=True)) else np.nan
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            lowest = min(lowest, low if np.isfinite(low) else 1.0)
            ax.hlines(yi, low, q1, color=color, linewidth=1.2, alpha=0.6)
            ax.hlines(yi, q1, q3, color=color, linewidth=6, alpha=0.85)
            ax.plot(median, yi, marker="o", markersize=7, color=color, markeredgecolor="white", zorder=4)
            ax.annotate(f"{median:.3f}", (q3, yi), textcoords="offset points", xytext=(8, -3), fontsize=7.5, color=INK_SOFT)
            if classical is None and len(classical_values(e, split)):
                classical = float(np.median(classical_values(e, split)))
        if classical is not None:
            ax.axvline(classical, color=CLASSICAL, linestyle="--", linewidth=1.3, label=f"classical threshold, median {classical:.3f}")
        n = len(iou_values(order[-1], split))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        for tick, e in zip(ax.get_yticklabels(), order):
            tick.set_color(INK if names.is_headline(e["name"]) else INK_SOFT)
            tick.set_fontweight("bold" if names.is_headline(e["name"]) else "normal")
        ax.set_xlim(max(0.0, lowest - 0.03), 1.02)
        ax.set_title(f"{split}: IoU over {n} hand-labeled frames", fontsize=9.5)
        ax.set_xlabel("IoU")
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(
        f"evaluation session {_iso(order[0]['session']).strftime('%Y-%m-%d %H:%M')} UTC; dot = median, bar = interquartile range, "
        "line = down to the lowest non-empty frame; bold = headline model", color=INK_SOFT, fontsize=9,
    )
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_evaluation_history(evaluations: list[dict[str, Any]], names: ModelNames, path: Path) -> None:
    """Median IoU of each best checkpoint across evaluation sessions; headline models in colour."""

    splits = _splits_present(evaluations)
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 3.9), constrained_layout=True)
    axes = np.atleast_1d(axes)
    best = [e for e in evaluations if e["kind"] == "best"]
    model_names: list[str] = []
    for e in best:
        if e["name"] not in model_names:
            model_names.append(e["name"])
    for ax, split in zip(axes, splits):
        if not best:
            _empty(ax, "no evaluations yet")
            continue
        classical_points = []
        for name in sorted(model_names, key=lambda n: (names.is_headline(n), n)):
            rows = [e for e in best if e["name"] == name and split in e["splits"]]
            if not rows:
                continue
            t = [_iso(e["session"]) for e in rows]
            headline = names.is_headline(name)
            ax.plot(t, [float(np.median(iou_values(e, split))) for e in rows], color=names.color(name), marker=names.marker(name),
                    linewidth=1.8 if headline else 0.9, markersize=6 if headline else 3.5, label=name if headline else None,
                    zorder=3 if headline else 2)
            classical_points += [(x, float(np.median(classical_values(e, split)))) for x, e in zip(t, rows) if len(classical_values(e, split))]
        classical_points = sorted(set(classical_points))
        if classical_points:
            ax.plot([p[0] for p in classical_points], [p[1] for p in classical_points], color=CLASSICAL, linestyle="--", marker="o",
                    markerfacecolor="white", markersize=4, linewidth=1.2, label="classical threshold")
        times = sorted({_iso(e["session"]) for e in best})
        span = max((times[-1] - times[0]) * 0.08, timedelta(hours=1))
        ax.set_xlim(times[0] - span, times[-1] + span)
        _date_axis(ax)
        ax.set_ylim(0.85, 1.0)
        ax.set_title(f"{split}: median IoU per evaluation session (labels grow between sessions)")
    axes[0].set_ylabel("IoU")
    if best:
        axes[-1].legend(loc="lower right", fontsize=7.5, ncol=2)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _grouped_samples(reference: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Hand-labeled samples of ``split`` ordered by recording, then by the reference model's IoU."""

    return sorted(hand_rows(reference, split), key=lambda r: (r["recording"], r["network"]["iou"]))


def _shade_recordings(ax, rows: list[dict[str, Any]], y_text: float) -> None:
    """Alternate a light band per recording along x and print the recording name at ``y_text``."""

    start, band = 0, 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["recording"] != rows[start]["recording"]:
            if band % 2 == 1:
                ax.axvspan(start - 0.5, i - 0.5, color=BAND, zorder=0)
            ax.text((start + i - 1) / 2, y_text, rows[start]["recording"], ha="center", va="bottom", fontsize=7.5, color=INK_SOFT)
            start, band = i, band + 1


def plot_latest_evaluation(session: list[dict[str, Any]], names: ModelNames, newest: str | None, path: Path) -> None:
    """Per-sample IoU of the headline models, samples grouped by recording and sorted by the newest model."""

    splits = _splits_present(session)
    fig, axes = plt.subplots(len(splits), 1, figsize=(13, 3.8 * len(splits) + 0.4), layout="constrained")
    axes = np.atleast_1d(axes)
    reference = by_name(session, newest)
    if reference is None:
        _empty(axes[0], "no evaluations yet")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        return
    shown = [e for e in session if e["kind"] == "best" and names.is_headline(e["name"])] or [reference]
    legend_title = f"evaluation session {_iso(session[0]['session']).strftime('%Y-%m-%d %H:%M')} UTC; frames sorted by recording, then by {newest}"
    for ax, split in zip(axes, splits):
        if split not in reference["splits"]:
            _empty(ax, f"no {split} evaluation")
            continue
        rows = _grouped_samples(reference, split)
        ids = [r["sample_id"] for r in rows]
        x = np.arange(len(ids))
        lowest = 1.0
        ax.plot(x, [r["classical"]["iou"] for r in rows], linestyle="none", marker="o", markerfacecolor="white", markersize=4.5,
                color=CLASSICAL, label="classical threshold", zorder=2)
        nonempty = [r["sample_id"] for r in rows if r.get("label_pixels", 1) > 0]
        series = []
        for e in shown:
            if split not in e["splits"]:
                continue
            lookup = {r["sample_id"]: r["network"]["iou"] for r in e["splits"][split]["per_sample"]}
            values = np.asarray([lookup.get(i, np.nan) for i in ids])
            series.append((e, values))
            over_nonempty = [lookup[i] for i in nonempty if i in lookup]
            lowest = min(lowest, float(min(over_nonempty)) if over_nonempty else 1.0)
        # The axis floor follows the worst non-empty frame; an empty frame the network painted scores 0 and is drawn clipped.
        floor = max(0.0, min(0.85, lowest - 0.03))
        for e, values in series:
            is_newest = e["name"] == newest
            clipped = values < floor
            ax.plot(x, np.where(clipped, floor + 0.006, values), linestyle="none", marker=names.marker(e["name"]),
                    markersize=7 if is_newest else 5, color=names.color(e["name"]), alpha=1.0 if is_newest else 0.75,
                    label=e["name"], zorder=4 if is_newest else 3, markeredgecolor="white" if is_newest else "none")
            for xi, value in zip(x[clipped], values[clipped]):
                ax.annotate(f"{value:.2f}", (xi, floor + 0.006), textcoords="offset points", xytext=(0, -9), ha="center", fontsize=6,
                            color=names.color(e["name"]))
        for xi, r in zip(x, rows):
            if r.get("label_pixels") == 0:
                ax.annotate("empty", (xi, 1.0), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6, color=INK_SOFT)
        _shade_recordings(ax, rows, floor + 0.02)
        ax.set_xlim(-0.6, len(ids) - 0.4)
        ax.set_ylim(floor, 1.04)
        ax.set_xticks(x)
        ax.set_xticklabels([i.split("_f")[-1] for i in ids], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("IoU")
        ax.set_title(f"{split}: {len(ids)} hand-labeled frames (tick labels are frame indices; an empty frame scores 1 when left blank, 0 otherwise)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", fontsize=7.5, ncol=min(7, len(labels)), title=legend_title, title_fontsize=8)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_model_delta(session: list[dict[str, Any]], newest: str | None, reference_name: str | None, path: Path) -> None:
    """Per-sample IoU of the newest model minus the reference model, by split."""

    new, ref = by_name(session, newest), by_name(session, reference_name)
    splits = _splits_present([e for e in (new, ref) if e])
    fig, axes = plt.subplots(1, len(splits), figsize=(6.6 * len(splits), 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    if new is None or ref is None:
        for ax in axes:
            _empty(ax, "need two headline models in the newest session")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        return
    fig.suptitle(f"{newest} minus {reference_name}: per-frame IoU change (positive = the newer model is better)", color=INK_SOFT, fontsize=9)
    for ax, split in zip(axes, splits):
        rows = _grouped_samples(new, split)
        ref_lookup = {r["sample_id"]: r["network"]["iou"] for r in ref["splits"].get(split, {}).get("per_sample", [])}
        rows = [r for r in rows if r["sample_id"] in ref_lookup]
        delta = np.asarray([r["network"]["iou"] - ref_lookup[r["sample_id"]] for r in rows])
        x = np.arange(len(rows))
        colors = [SERIES[0] if d >= 0 else SERIES[1] for d in delta]
        # Ordinary frames move by hundredths; an empty frame flips between 0 and 1, so the axis is clipped and such bars labeled.
        inliers = np.abs(delta) <= 0.1
        limit = max(0.02, float(np.max(np.abs(delta[inliers]))) * 1.3) if inliers.any() else 0.02
        ax.bar(x, np.clip(delta, -limit, limit), color=colors, width=0.8)
        ax.axhline(0, color=INK_SOFT, linewidth=0.8)
        _shade_recordings(ax, rows, -limit * 0.97)
        for xi, r, d in zip(x, rows, delta):
            clipped = abs(d) > limit
            if abs(d) >= 0.01 or clipped:
                text = r["sample_id"].split("_f")[-1] + (" (empty)" if r.get("label_pixels") == 0 else "")
                if clipped:
                    text += f"  {d:+.2f}"
                    ax.annotate(text, (xi, 0.0), textcoords="offset points", xytext=(4, 4 if d >= 0 else -10), ha="left", fontsize=6, color=INK)
                    continue
                ax.annotate(text, (xi, d), textcoords="offset points", xytext=(0, 4 if d >= 0 else -10), ha="center", fontsize=6, color=INK_SOFT)
        ax.set_ylim(-limit, limit)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_xticks([])
        better, worse = int(np.sum(delta > 0.002)), int(np.sum(delta < -0.002))
        ax.set_title(f"{split}: {better} frames better, {worse} worse, {len(rows) - better - worse} within 0.002; mean {np.mean(delta):+.4f}, median {np.median(delta):+.4f}", fontsize=9)
        ax.set_ylabel("IoU change")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_iou_ecdf(session: list[dict[str, Any]], names: ModelNames, newest: str | None, path: Path) -> None:
    """Cumulative distribution of per-frame IoU for the headline models, val and test pooled."""

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    shown = [e for e in session if e["kind"] == "best" and names.is_headline(e["name"])]
    if not shown:
        _empty(ax, "no headline model in the newest session")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        return
    floor = 0.85
    splits = _splits_present(shown)

    def pooled(e: dict[str, Any], classical: bool = False) -> np.ndarray:
        parts = [classical_values(e, s) if classical else iou_values(e, s) for s in splits]
        return np.concatenate([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.array([])

    def step(values: np.ndarray, **kwargs) -> None:
        values = np.sort(values)
        cumulative = np.arange(1, len(values) + 1) / len(values)
        below = int(np.sum(values < floor))
        label = kwargs.pop("label") + f"  (median {np.median(values):.3f}, {below} below {floor:.2f})"
        ax.step(np.concatenate([[floor], values[values >= floor], [1.0]]), np.concatenate([[below / len(values)], cumulative[values >= floor], [1.0]]),
                where="post", label=label, **kwargs)

    step(pooled(shown[0], classical=True), color=CLASSICAL, linestyle="--", linewidth=1.3, label="classical threshold")
    for e in shown:
        is_newest = e["name"] == newest
        step(pooled(e), color=names.color(e["name"]), linewidth=2.4 if is_newest else 1.5, alpha=1.0 if is_newest else 0.85, label=e["name"])
    n = len(pooled(shown[0]))
    ax.set_xlim(floor, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("IoU")
    ax.set_ylabel("fraction of frames at or below")
    ax.set_title(f"per-frame IoU distribution, {' + '.join(splits)} pooled ({n} hand-labeled frames); lower-right is better")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_dataset_growth(store: SegmentationStore, path: Path) -> None:
    """Labels over time, and labels per recording by split."""

    records = sorted(store.records(), key=lambda r: r.saved_at)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1]})
    if records:
        times = [_iso(r.saved_at) for r in records]
        hand = np.array([is_hand_labeled(r.label_source) for r in records])
        left.step(times, np.arange(1, len(records) + 1), where="post", color=SERIES[0], label="all labels in the store")
        left.step(times, np.cumsum(hand), where="post", color=SERIES[2], label="hand-labeled")
        held_out = np.array([r.split != "train" for r in records]) & hand
        left.step(times, np.cumsum(held_out), where="post", color=SERIES[1], label="hand-labeled, val or test")
        _date_axis(left)
        left.legend(loc="upper left", fontsize=8)
        recordings = sorted({r.recording for r in records})
        bottom = np.zeros(len(recordings))
        for split in SPLITS:
            counts = np.array([sum(1 for r in records if r.recording == rec and r.split == split) for rec in recordings])
            right.bar(recordings, counts, bottom=bottom, color=SPLIT_COLORS[split], label=split, width=0.7)
            bottom += counts
        for x, total in enumerate(bottom):
            right.annotate(str(int(total)), (x, total), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, color=INK_SOFT)
        right.set_xticks(range(len(recordings)))
        right.set_xticklabels(recordings, rotation=35, ha="right", fontsize=7.5)
        right.legend(loc="upper right", fontsize=8)
        right.set_title("labels per recording by split (current store)")
    else:
        _empty(left, "store is empty")
    left.set_title("labels in the store by last save time (bootstrap labels retired 2026-09-05)")
    left.set_ylabel("samples")
    right.set_ylabel("samples")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def make_plots(checkpoint_dir: Path, dataset_root: Path, output_dir: Path, names: ModelNames | None = None, *, model: str | None = None,
               reference: str | None = None, include_last: bool = False) -> list[Path]:
    names = names or ModelNames({}, [])
    style()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(checkpoint_dir, names)
    evaluations = load_evaluations(checkpoint_dir, names)
    session = latest_session(evaluations)
    newest, reference = choose_models(session, names, model, reference)
    outputs = {
        "training_curves": output_dir / "training_curves.png",
        "checkpoint_comparison": output_dir / "checkpoint_comparison.png",
        "evaluation_history": output_dir / "evaluation_history.png",
        "latest_evaluation": output_dir / "latest_evaluation.png",
        "model_delta": output_dir / "model_delta.png",
        "iou_ecdf": output_dir / "iou_ecdf.png",
        "dataset_growth": output_dir / "dataset_growth.png",
    }
    plot_training_curves(runs, names, outputs["training_curves"])
    plot_checkpoint_comparison(session, names, promoted_sha(checkpoint_dir), outputs["checkpoint_comparison"], include_last)
    plot_evaluation_history(evaluations, names, outputs["evaluation_history"])
    plot_latest_evaluation(session, names, newest, outputs["latest_evaluation"])
    plot_model_delta(session, newest, reference, outputs["model_delta"])
    plot_iou_ecdf(session, names, newest, outputs["iou_ecdf"])
    plot_dataset_growth(SegmentationStore(dataset_root), outputs["dataset_growth"])
    return list(outputs.values())


def main() -> int:
    args = parse_args()
    names = ModelNames.load(args.names)
    outputs = make_plots(
        args.checkpoint_dir, args.dataset_root, args.output_dir or args.checkpoint_dir / "plots", names,
        model=args.model, reference=args.reference, include_last=args.include_last,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
