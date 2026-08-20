#!/usr/bin/env python3
"""Bounded EXP-SMC-008A strict-anchor bracket scan around row 22."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.anchors import AnchorConfig, extract_mask_anchor
from worm_pose_gen.segmentation import SoftForegroundConfig, segment_soft_foreground


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "nir_videos/2023-10-11-01.h5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def runs(values: np.ndarray, target: bool) -> list[dict[str, int]]:
    output: list[dict[str, int]] = []
    start: int | None = None
    for position, value in enumerate(values):
        if bool(value) == target and start is None:
            start = position
        if bool(value) != target and start is not None:
            output.append({"start_position": start, "stop_position_inclusive": position - 1, "length": position - start})
            start = None
    if start is not None:
        output.append({"start_position": start, "stop_position_inclusive": len(values) - 1, "length": len(values) - start})
    return output


def main() -> int:
    experiment = json.loads((HERE / "config.json").read_text())
    final_path = ROOT / experiment["final_revision_config"]
    final = json.loads(final_path.read_text())
    segment_config = SoftForegroundConfig(**final["soft_foreground_config"])
    anchor_config = AnchorConfig(**final["anchor_config"])
    if experiment["recording"].startswith("2025-") or "2025-03-06" in str(SOURCE.resolve(strict=True)):
        raise RuntimeError("protected holdout path encountered")
    start = int(experiment["start_frame"])
    stop = int(experiment["stop_frame_inclusive"])
    hard = int(experiment["hard_frame"])
    indices = np.arange(start, stop + 1, dtype=np.int64)
    if len(indices) != 201 or indices[100] != hard:
        raise RuntimeError("authorized centered 201-frame window changed")
    resolved = SOURCE.resolve(strict=True)
    stat = resolved.stat()
    output = Path(experiment["external_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")

    accepted = np.zeros(len(indices), dtype=bool)
    quality = np.zeros(len(indices), dtype=np.float32)
    iou = np.full(len(indices), np.nan, dtype=np.float64)
    boundary_ratio = np.full(len(indices), np.nan, dtype=np.float64)
    rejection_text: list[str] = []
    seg_qc_text: list[str] = []
    anchor_qc_text: list[str] = []
    string_dtype = h5py.string_dtype("utf-8")
    started = time.perf_counter()
    with h5py.File(partial, "x") as destination:
        destination.attrs.update(
            schema_version=1,
            experiment=experiment["experiment"],
            complete=False,
            configured_source_path=str(SOURCE.relative_to(ROOT)),
            resolved_source_path=str(resolved),
            source_size_bytes=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            source_dataset_path=experiment["source_dataset_path"],
            final_revision_config_sha256=sha256(final_path),
            protected_2025_holdout_opened=False,
        )
        destination.create_dataset("frame_index", data=indices)
        masks = destination.create_dataset(
            "cleaned_mask", shape=(len(indices), 732, 968), dtype=bool,
            chunks=(1, 732, 968), compression="gzip", compression_opts=4, shuffle=True,
        )
        centerlines = destination.create_dataset(
            "centerline_xy", data=np.full((len(indices), anchor_config.n_points, 2), np.nan, dtype=np.float32)
        )
        widths = destination.create_dataset(
            "estimated_width", data=np.full((len(indices), anchor_config.n_points), np.nan, dtype=np.float32)
        )
        with h5py.File(SOURCE, "r") as source:
            images = source[experiment["source_dataset_path"]]
            if stop >= len(images):
                raise RuntimeError("authorized window exceeds source bounds")
            timestamps = source["img_metadata/img_timestamp"][:]
            save_selector = source["img_metadata/q_iter_save"][:] == 1
            nir_timestamps = timestamps[save_selector]
            destination.create_dataset("timestamp_raw", data=nir_timestamps[indices])
            for position, frame_index in enumerate(indices):
                frame = images[int(frame_index)]
                segmentation = segment_soft_foreground(frame, segment_config)
                anchor = extract_mask_anchor(
                    segmentation.cleaned_mask,
                    probability=segmentation.probability_map,
                    config=anchor_config,
                )
                masks[position] = segmentation.cleaned_mask
                accepted[position] = anchor.accepted
                quality[position] = anchor.quality_score
                rejection_text.append(";".join(anchor.rejection_reasons))
                seg_qc_text.append(json.dumps({key: scalar(value) for key, value in segmentation.qc.items()}, sort_keys=True))
                anchor_qc_text.append(json.dumps({key: scalar(value) for key, value in anchor.qc.items()}, sort_keys=True))
                iou[position] = float(anchor.qc.get("mask_render_iou", np.nan))
                boundary_ratio[position] = float(anchor.qc.get("boundary_clearance_width_ratio", np.nan))
                if anchor.accepted:
                    centerlines[position] = np.asarray(anchor.centerline_xy, dtype=np.float32)
                    widths[position] = np.asarray(anchor.estimated_width, dtype=np.float32)
                if (position + 1) % 10 == 0 or position == len(indices) - 1:
                    print(f"{position + 1}/{len(indices)}", flush=True)
        destination.create_dataset("accepted", data=accepted)
        destination.create_dataset("quality_score", data=quality)
        destination.create_dataset("rejection_reasons", data=np.asarray(rejection_text, dtype=object), dtype=string_dtype)
        destination.create_dataset("segmentation_qc_json", data=np.asarray(seg_qc_text, dtype=object), dtype=string_dtype)
        destination.create_dataset("anchor_qc_json", data=np.asarray(anchor_qc_text, dtype=object), dtype=string_dtype)
        destination.attrs["complete"] = True
        destination.flush()
    os.replace(partial, output)
    runtime = time.perf_counter() - started

    before_candidates = indices[(indices < hard) & accepted]
    after_candidates = indices[(indices > hard) & accepted]
    before = int(before_candidates[-1]) if len(before_candidates) else None
    after = int(after_candidates[0]) if len(after_candidates) else None
    intervening = after - before - 1 if before is not None and after is not None else None
    short_bout = intervening is not None and intervening <= 20
    accepted_runs = runs(accepted, True)
    rejected_gaps = runs(accepted, False)
    rejection_counts = Counter(reason for text in rejection_text for reason in text.split(";") if reason)
    metrics = {
        "schema_version": 1,
        "experiment": experiment["experiment"],
        "status": "COMPLETED",
        "evidence_role": "development-only natural hard-case bracket diagnostic; no pose inference or accuracy claim",
        "recording": experiment["recording"],
        "configured_source_path": str(SOURCE.relative_to(ROOT)),
        "resolved_source_path": str(resolved),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "frame_window_inclusive": [start, stop],
        "hard_frame": hard,
        "frames_scanned": len(indices),
        "final_revision_config": experiment["final_revision_config"],
        "final_revision_config_sha256": sha256(final_path),
        "soft_foreground_config": asdict(segment_config),
        "anchor_config": asdict(anchor_config),
        "strict_accepted_count": int(accepted.sum()),
        "strict_accepted_density": float(accepted.mean()),
        "hard_frame_strict_accepted": bool(accepted[hard - start]),
        "nearest_strict_anchor_before": before,
        "nearest_strict_anchor_after": after,
        "intervening_frames_between_bracket_anchors": intervening,
        "two_anchor_natural_bout_at_most_20_frames": short_bout,
        "two_anchor_bout_definition": experiment["two_anchor_bout_definition"],
        "accepted_run_count": len(accepted_runs),
        "accepted_runs": [
            {**run, "start_frame": int(indices[run["start_position"]]), "stop_frame_inclusive": int(indices[run["stop_position_inclusive"]])}
            for run in accepted_runs
        ],
        "rejected_gap_count": len(rejected_gaps),
        "rejected_gaps": [
            {**gap, "start_frame": int(indices[gap["start_position"]]), "stop_frame_inclusive": int(indices[gap["stop_position_inclusive"]])}
            for gap in rejected_gaps
        ],
        "longest_accepted_run": max((run["length"] for run in accepted_runs), default=0),
        "longest_rejected_gap": max((gap["length"] for gap in rejected_gaps), default=0),
        "rejection_reason_counts_nonexclusive": dict(sorted(rejection_counts.items())),
        "runtime_seconds": runtime,
        "frames_per_second": len(indices) / runtime,
        "external_output": str(output),
        "external_output_sha256": sha256(output),
        "protected_2025_holdout_opened": False,
        "inferred_smc_poses": False,
    }
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    offsets = indices - hard
    figure, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].step(offsets, accepted.astype(int), where="mid", color="#0969da")
    axes[0].axvline(0, color="#cf222e", linestyle="--", label="hard frame 13785")
    if before is not None:
        axes[0].axvline(before - hard, color="#1a7f37", linestyle=":")
    if after is not None:
        axes[0].axvline(after - hard, color="#1a7f37", linestyle=":")
    axes[0].set_yticks([0, 1], ["reject", "accept"])
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend()
    axes[0].set_title("EXP-SMC-008A strict-anchor timeline")
    axes[1].plot(offsets, iou, label="mask/render IoU", color="#1a7f37")
    axes[1].plot(offsets, boundary_ratio, label="boundary clearance / width", color="#bf8700")
    axes[1].axhline(anchor_config.min_render_iou, color="#1a7f37", linestyle="--", alpha=0.5)
    axes[1].axhline(anchor_config.min_boundary_clearance_widths, color="#bf8700", linestyle="--", alpha=0.5)
    axes[1].axvline(0, color="#cf222e", linestyle="--")
    axes[1].set_xlabel("frame offset from expert-adjudicated hard frame")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(HERE / "timeline.png", dpi=160)
    plt.close(figure)

    def plot_positions(frame_indices: list[int], path: Path, columns: int) -> None:
        figure, axes = plt.subplots(1, columns, figsize=(4 * columns, 3.4), squeeze=False)
        with h5py.File(SOURCE, "r") as source, h5py.File(output, "r") as cached:
            for ax, frame_index in zip(axes.flat, frame_indices):
                position = frame_index - start
                image = source[experiment["source_dataset_path"]][frame_index]
                p1, p99 = np.percentile(image, [1, 99])
                ax.imshow(image, cmap="gray", vmin=p1, vmax=p99)
                ax.contour(cached["cleaned_mask"][position], levels=[0.5], colors=["#ffb000"], linewidths=0.7)
                line = cached["centerline_xy"][position]
                if np.isfinite(line).all():
                    ax.plot(line[:, 0], line[:, 1], color="#00ffff", linewidth=1.0)
                label = "accepted" if accepted[position] else "rejected"
                ax.set_title(f"f{frame_index} {label}")
                ax.axis("off")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

    plot_positions([hard - 20, hard - 10, hard, hard + 10, hard + 20], HERE / "local_overlays.png", 5)
    bracket_frames = [value for value in (before, hard, after) if value is not None]
    plot_positions(bracket_frames, HERE / "bracket_overlays.png", len(bracket_frames))
    print(json.dumps({"accepted": int(accepted.sum()), "before": before, "after": after, "short_bout": short_bout, "runtime_seconds": runtime}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
