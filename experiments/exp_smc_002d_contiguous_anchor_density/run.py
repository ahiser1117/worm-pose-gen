#!/usr/bin/env python3
"""Run the bounded development-only EXP-SMC-002D density scan."""

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
FINAL_CONFIG = ROOT / "configs/smc_exp_001b_002b_revision.json"
SOURCES = {
    "2023-09-19-01": ROOT / "nir_videos/2023-09-19-01.h5",
    "2023-09-27-01": ROOT / "nir_videos/2023-09-27-01.h5",
    "2023-10-11-01": ROOT / "nir_videos/2023-10-11-01.h5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runs(values: np.ndarray, target: bool) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    start: int | None = None
    for position, value in enumerate(values):
        if bool(value) == target and start is None:
            start = position
        if bool(value) != target and start is not None:
            result.append({"start_position": start, "stop_position_inclusive": position - 1, "length": position - start})
            start = None
    if start is not None:
        result.append({"start_position": start, "stop_position_inclusive": len(values) - 1, "length": len(values) - start})
    return result


def scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def main() -> int:
    experiment = json.loads((HERE / "config.json").read_text())
    final = json.loads(FINAL_CONFIG.read_text())
    segment_config = SoftForegroundConfig(**final["soft_foreground_config"])
    anchor_config = AnchorConfig(**final["anchor_config"])
    output = Path(experiment["external_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")

    all_metrics: dict[str, object] = {
        "schema_version": 1,
        "experiment": experiment["experiment"],
        "status": "COMPLETED",
        "evidence_role": "development-only feasibility diagnostic; no accuracy or outcome claim",
        "final_revision_config": str(FINAL_CONFIG.relative_to(ROOT)),
        "final_revision_config_sha256": sha256(FINAL_CONFIG),
        "soft_foreground_config": asdict(segment_config),
        "anchor_config": asdict(anchor_config),
        "protected_2025_holdout_opened": False,
        "sessions": {},
    }
    plot_rows: dict[str, dict[str, np.ndarray]] = {}
    total_started = time.perf_counter()
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(partial, "x") as destination:
        destination.attrs.update(
            schema_version=1,
            experiment=experiment["experiment"],
            complete=False,
            source_dataset_path=experiment["source_dataset_path"],
            final_revision_config_sha256=sha256(FINAL_CONFIG),
            protected_2025_holdout_opened=False,
        )
        for window in experiment["windows"]:
            recording = window["recording"]
            if recording not in SOURCES or recording.startswith("2025-"):
                raise RuntimeError(f"unauthorized recording: {recording}")
            source_path = SOURCES[recording]
            resolved = source_path.resolve(strict=True)
            if "2025-03-06" in str(resolved):
                raise RuntimeError("protected holdout path encountered")
            indices = np.arange(window["start_frame"], window["stop_frame_exclusive"], dtype=np.int64)
            if len(indices) != 101 or int(indices[50]) != int(window["center_frame"]):
                raise RuntimeError(f"invalid 101-frame centered window: {window}")
            accepted: list[bool] = []
            qualities: list[float] = []
            masks: list[np.ndarray] = []
            centerlines = np.full((101, anchor_config.n_points, 2), np.nan, dtype=np.float32)
            widths = np.full((101, anchor_config.n_points), np.nan, dtype=np.float32)
            rejection_text: list[str] = []
            seg_qc_text: list[str] = []
            anchor_qc_text: list[str] = []
            ious: list[float] = []
            boundary_ratios: list[float] = []
            session_started = time.perf_counter()
            stat = resolved.stat()
            with h5py.File(source_path, "r") as source:
                images = source[experiment["source_dataset_path"]]
                for position, frame_index in enumerate(indices):
                    frame = images[int(frame_index)]
                    segmentation = segment_soft_foreground(frame, segment_config)
                    anchor = extract_mask_anchor(
                        segmentation.cleaned_mask,
                        probability=segmentation.probability_map,
                        config=anchor_config,
                    )
                    accepted.append(bool(anchor.accepted))
                    qualities.append(float(anchor.quality_score))
                    masks.append(segmentation.cleaned_mask)
                    rejection_text.append(";".join(anchor.rejection_reasons))
                    seg_qc_text.append(json.dumps({key: scalar(value) for key, value in segmentation.qc.items()}, sort_keys=True))
                    anchor_qc_text.append(json.dumps({key: scalar(value) for key, value in anchor.qc.items()}, sort_keys=True))
                    ious.append(float(anchor.qc.get("mask_render_iou", np.nan)))
                    boundary_ratios.append(float(anchor.qc.get("boundary_clearance_width_ratio", np.nan)))
                    if anchor.accepted:
                        centerlines[position] = anchor.centerline_xy
                        widths[position] = anchor.estimated_width
                    if (position + 1) % 10 == 0 or position == 100:
                        print(f"{recording}: {position + 1}/101", flush=True)
            session_seconds = time.perf_counter() - session_started
            accepted_array = np.asarray(accepted, dtype=bool)
            accepted_runs = runs(accepted_array, True)
            rejected_gaps = runs(accepted_array, False)
            rejection_counts = Counter(
                reason
                for text in rejection_text
                for reason in text.split(";")
                if reason
            )
            session_metrics = {
                "recording": recording,
                "configured_source_path": str(source_path.relative_to(ROOT)),
                "resolved_source_path": str(resolved),
                "source_size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "frame_window_inclusive": [int(indices[0]), int(indices[-1])],
                "center_frame": int(indices[50]),
                "frames_scanned": 101,
                "strict_accepted_count": int(accepted_array.sum()),
                "strict_accepted_density": float(accepted_array.mean()),
                "has_strict_anchor": bool(accepted_array.any()),
                "accepted_frame_indices": indices[accepted_array].tolist(),
                "accepted_runs": [
                    {**run, "start_frame": int(indices[run["start_position"]]), "stop_frame_inclusive": int(indices[run["stop_position_inclusive"]])}
                    for run in accepted_runs
                ],
                "accepted_run_count": len(accepted_runs),
                "adjacent_accepted_pairs": sum(run["length"] - 1 for run in accepted_runs),
                "rejected_gaps": [
                    {**gap, "start_frame": int(indices[gap["start_position"]]), "stop_frame_inclusive": int(indices[gap["stop_position_inclusive"]])}
                    for gap in rejected_gaps
                ],
                "longest_accepted_run": max((run["length"] for run in accepted_runs), default=0),
                "longest_rejected_gap": max((gap["length"] for gap in rejected_gaps), default=0),
                "rejection_reason_counts_nonexclusive": dict(sorted(rejection_counts.items())),
                "runtime_seconds": session_seconds,
                "frames_per_second": 101 / session_seconds,
            }
            all_metrics["sessions"][recording] = session_metrics
            group = destination.create_group(recording)
            group.attrs.update(
                configured_source_path=str(source_path.relative_to(ROOT)),
                resolved_source_path=str(resolved),
                source_size_bytes=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns,
            )
            group.create_dataset("frame_index", data=indices)
            group.create_dataset("accepted", data=accepted_array)
            group.create_dataset("quality_score", data=np.asarray(qualities, dtype=np.float32))
            group.create_dataset("cleaned_mask", data=np.stack(masks), chunks=(1, 732, 968), compression="gzip", compression_opts=4, shuffle=True)
            group.create_dataset("centerline_xy", data=centerlines)
            group.create_dataset("estimated_width", data=widths)
            group.create_dataset("rejection_reasons", data=np.asarray(rejection_text, dtype=object), dtype=string_dtype)
            group.create_dataset("segmentation_qc_json", data=np.asarray(seg_qc_text, dtype=object), dtype=string_dtype)
            group.create_dataset("anchor_qc_json", data=np.asarray(anchor_qc_text, dtype=object), dtype=string_dtype)
            plot_rows[recording] = {
                "indices": indices,
                "accepted": accepted_array,
                "quality": np.asarray(qualities),
                "iou": np.asarray(ious),
                "boundary_ratio": np.asarray(boundary_ratios),
            }
        destination.attrs["complete"] = True
        destination.flush()
    os.replace(partial, output)

    total_seconds = time.perf_counter() - total_started
    total_accepted = sum(int(value["strict_accepted_count"]) for value in all_metrics["sessions"].values())
    total_adjacent_pairs = sum(int(value["adjacent_accepted_pairs"]) for value in all_metrics["sessions"].values())
    sessions_with_adjacent_pairs = sum(int(value["adjacent_accepted_pairs"] > 0) for value in all_metrics["sessions"].values())
    all_metrics.update(
        total_frames=303,
        total_strict_accepted=total_accepted,
        overall_strict_accepted_density=total_accepted / 303,
        every_session_has_strict_anchor=all(bool(value["has_strict_anchor"]) for value in all_metrics["sessions"].values()),
        total_adjacent_accepted_pairs=total_adjacent_pairs,
        sessions_with_adjacent_accepted_pairs=sessions_with_adjacent_pairs,
        every_session_has_adjacent_accepted_pair=sessions_with_adjacent_pairs == len(all_metrics["sessions"]),
        posture_sample_extraction_feasible=True,
        session_general_empirical_dynamics_fitting_feasible=False,
        feasibility_interpretation="All sessions contain static strict anchors, but one session has no adjacent accepted pair and accepted runs are highly fragmented; these windows do not support session-general contiguous dynamics fitting.",
        runtime_seconds=total_seconds,
        frames_per_second=303 / total_seconds,
        external_output=str(output),
        external_output_sha256=sha256(output),
    )
    (HERE / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")

    figure, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col")
    for row, (recording, values) in enumerate(plot_rows.items()):
        x = values["indices"] - values["indices"][50]
        axes[row, 0].step(x, values["accepted"].astype(int), where="mid", color="#0969da")
        axes[row, 0].set_ylim(-0.1, 1.1)
        axes[row, 0].set_yticks([0, 1], ["reject", "accept"])
        axes[row, 0].set_title(recording)
        axes[row, 1].plot(x, values["iou"], label="mask/render IoU", color="#1a7f37")
        axes[row, 1].plot(x, values["boundary_ratio"], label="boundary clearance / width", color="#bf8700")
        axes[row, 1].axhline(anchor_config.min_render_iou, color="#1a7f37", linestyle="--", alpha=0.5)
        axes[row, 1].axhline(anchor_config.min_boundary_clearance_widths, color="#bf8700", linestyle="--", alpha=0.5)
        axes[row, 1].set_title("strict-anchor diagnostics")
    axes[-1, 0].set_xlabel("frame offset from authorized center")
    axes[-1, 1].set_xlabel("frame offset from authorized center")
    axes[0, 1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(HERE / "timeline.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(12, 9), squeeze=False)
    with h5py.File(output, "r") as cached:
        for row, (recording, values) in enumerate(plot_rows.items()):
            accepted_positions = np.flatnonzero(values["accepted"])
            rejected_positions = np.flatnonzero(~values["accepted"])
            selections = [
                int(accepted_positions[0]) if len(accepted_positions) else int(rejected_positions[0]),
                int(accepted_positions[len(accepted_positions) // 2]) if len(accepted_positions) else int(rejected_positions[len(rejected_positions) // 2]),
                int(rejected_positions[len(rejected_positions) // 2]) if len(rejected_positions) else int(accepted_positions[-1]),
            ]
            with h5py.File(SOURCES[recording], "r") as source:
                for column, position in enumerate(selections):
                    frame_index = int(values["indices"][position])
                    image = source[experiment["source_dataset_path"]][frame_index]
                    ax = axes[row, column]
                    p1, p99 = np.percentile(image, [1, 99])
                    ax.imshow(image, cmap="gray", vmin=p1, vmax=p99)
                    mask = cached[recording]["cleaned_mask"][position]
                    ax.contour(mask, levels=[0.5], colors=["#ffb000"], linewidths=0.6)
                    centerline = cached[recording]["centerline_xy"][position]
                    if np.isfinite(centerline).all():
                        ax.plot(centerline[:, 0], centerline[:, 1], color="#00ffff", linewidth=1.0)
                    label = "accepted" if values["accepted"][position] else "rejected"
                    ax.set_title(f"{recording} f{frame_index} {label}", fontsize=8)
                    ax.axis("off")
    figure.tight_layout()
    figure.savefig(HERE / "selected_overlays.png", dpi=160)
    plt.close(figure)
    print(json.dumps({"metrics": str(HERE / "metrics.json"), "runtime_seconds": total_seconds, "accepted": total_accepted}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
