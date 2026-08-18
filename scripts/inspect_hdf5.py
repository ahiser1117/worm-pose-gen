#!/usr/bin/env python3
"""Bounded, read-only forensic audit of the supplied NIR HDF5 recordings.

The script deliberately opens one file at a time and reads at most
``--max-samples`` image frames from each recording.  It never writes to an
input path.  The foreground/body measurements are screening heuristics, not
annotations or ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")


def dataset_description(obj: h5py.Dataset) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    try:
        creation = obj.id.get_create_plist()
        for index in range(creation.get_nfilters()):
            filter_id, flags, parameters, name = creation.get_filter(index)
            filters.append({
                "id": int(filter_id),
                "flags": int(flags),
                "parameters": [int(value) for value in parameters],
                "name": json_value(name),
            })
    except Exception as exc:
        filters.append({"error": error_text(exc)})
    result: dict[str, Any] = {
        "kind": "dataset",
        "shape": list(obj.shape),
        "dtype": str(obj.dtype),
        "chunks": list(obj.chunks) if obj.chunks else None,
        "compression": obj.compression,
        "filters": filters,
    }
    if obj.attrs:
        result["attrs"] = {str(k): json_value(v) for k, v in obj.attrs.items()}
    return result


def schema_inventory(handle: h5py.File) -> dict[str, Any]:
    schema: dict[str, Any] = {}

    def descend(group: h5py.Group, prefix: str = "") -> None:
        try:
            names = list(group.keys())
        except Exception as exc:
            schema[prefix or "/"] = {"kind": "unreadable", "error": error_text(exc)}
            return
        for child_name in names:
            name = f"{prefix}/{child_name}" if prefix else child_name
            try:
                obj = group[child_name]
                if isinstance(obj, h5py.Dataset):
                    schema[name] = dataset_description(obj)
                else:
                    entry: dict[str, Any] = {"kind": "group"}
                    if obj.attrs:
                        entry["attrs"] = {
                            str(k): json_value(v) for k, v in obj.attrs.items()
                        }
                    schema[name] = entry
                    descend(obj, name)
            except Exception as exc:
                schema[name] = {"kind": "unreadable", "error": error_text(exc)}

    descend(handle)
    return schema


def choose_indices(frame_count: int, limit: int) -> np.ndarray:
    """Uniform coverage plus an 8-frame mid-recording locomotion sequence."""
    count = min(frame_count, limit)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= 8:
        return np.unique(np.linspace(0, frame_count - 1, count, dtype=np.int64))
    uniform_count = count - 8
    uniform = np.linspace(0, frame_count - 1, uniform_count, dtype=np.int64)
    start = max(0, min(frame_count - 8, frame_count // 2 - 4))
    sequence = np.arange(start, min(start + 8, frame_count), dtype=np.int64)
    indices = np.unique(np.concatenate([uniform, sequence]))
    # Duplicates are possible in very short recordings; fill deterministically.
    if len(indices) < count:
        candidates = np.linspace(0, frame_count - 1, count * 3, dtype=np.int64)
        indices = np.unique(np.concatenate([indices, candidates]))[:count]
    return np.sort(indices[:count])


def largest_component(mask: np.ndarray) -> dict[str, float] | None:
    """Return an 8-connected component summary on a small downsampled mask."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for y, x in np.argwhere(mask):
        y = int(y)
        x = int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue = deque([(y, x)])
        component: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        if len(component) > len(best):
            best = component
    if not best:
        return None
    points = np.asarray(best, dtype=float)
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    centered = points - points.mean(axis=0)
    eigvals = np.linalg.eigvalsh(centered.T @ centered / max(len(points), 1))
    major = 4.0 * math.sqrt(max(float(eigvals[-1]), 0.0))
    area = float(len(points))
    width_proxy = area / max(2.0 * major, 1.0)
    return {
        "area_downsampled_px": area,
        "bbox_y0": float(y0),
        "bbox_x0": float(x0),
        "bbox_y1": float(y1),
        "bbox_x1": float(x1),
        "major_sigma_span_downsampled_px": major,
        "width_proxy_downsampled_px": width_proxy,
        "touches_boundary": bool(x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1),
    }


def foreground_metrics(frames: np.ndarray) -> list[dict[str, Any]]:
    """Estimate moving foreground by robust temporal-background subtraction."""
    small = frames[:, ::4, ::4].astype(np.float32)
    background = np.median(small, axis=0)
    residuals = np.abs(small - background)
    metrics: list[dict[str, Any]] = []
    for residual in residuals:
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        threshold = max(8.0, median + 6.0 * max(mad, 1.0))
        component = largest_component(residual > threshold)
        metrics.append({"threshold_intensity": threshold, "component": component})
    return metrics


def timestamp_metrics(handle: h5py.File, frame_count: int) -> dict[str, Any]:
    path = "img_metadata/img_timestamp"
    if path not in handle:
        return {"available": False}
    try:
        timestamp = np.asarray(handle[path][...], dtype=np.float64)
        selected = timestamp
        mapping = "all timestamps"
        if "img_metadata/q_iter_save" in handle:
            q_iter = np.asarray(handle["img_metadata/q_iter_save"][...]).astype(bool)
            if q_iter.shape == timestamp.shape and int(q_iter.sum()) == frame_count:
                selected = timestamp[q_iter]
                mapping = "q_iter_save == 1 (count matches img_nir)"
        elif len(timestamp) == 2 * frame_count:
            selected = timestamp[::2]
            mapping = "every other timestamp (length is 2x img_nir; unverified)"
        differences = np.diff(selected)
        positive = differences[differences > 0]
        if not len(positive):
            return {"available": True, "mapping": mapping, "count": len(selected)}
        median_raw = float(np.median(positive))
        # Camera timestamps in these files are empirically nanoseconds. Preserve
        # raw values and label this conversion as an inference.
        inferred_scale = 1e9 if median_raw > 1e5 else 1.0
        median_seconds = median_raw / inferred_scale
        dropped = int(np.sum(positive > 1.5 * median_raw))
        duplicate_or_nonmonotonic = int(np.sum(differences <= 0))
        return {
            "available": True,
            "path": path,
            "count": int(len(selected)),
            "mapping": mapping,
            "unit_interpretation": "nanoseconds inferred from magnitude" if inferred_scale == 1e9 else "seconds assumed",
            "median_interval_raw": median_raw,
            "median_interval_seconds": median_seconds,
            "observed_fps": 1.0 / median_seconds if median_seconds > 0 else None,
            "large_gap_count_gt_1_5x_median": dropped,
            "duplicate_or_nonmonotonic_count": duplicate_or_nonmonotonic,
        }
    except Exception as exc:  # corrupt metadata is itself an audit finding
        return {"available": False, "error": error_text(exc)}


def scalar_metadata(handle: h5py.File) -> dict[str, Any]:
    result: dict[str, Any] = {}
    candidates = ["recording_start"]
    if "metadata" in handle:
        candidates.extend(f"metadata/{name}" for name in handle["metadata"].keys())
    for path in candidates:
        try:
            obj = handle[path]
            if isinstance(obj, h5py.Dataset) and obj.shape == ():
                result[path] = json_value(obj[()])
        except Exception as exc:
            result[path] = {"error": error_text(exc)}
    return result


def inspect_recording(path: Path, max_samples: int) -> tuple[dict[str, Any], np.ndarray | None]:
    resolved = path.resolve(strict=False)
    link_stat = path.lstat()
    record: dict[str, Any] = {
        "name": path.name,
        "input_path": str(path),
        "symlink_target": os.readlink(path) if path.is_symlink() else None,
        "resolved_path": str(resolved),
        "symlink_size_bytes": int(link_stat.st_size),
    }
    try:
        stat = path.stat()
        record.update(size_bytes=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))
    except OSError as exc:
        record["status"] = "unopenable"
        record["error"] = error_text(exc)
        return record, None

    try:
        with h5py.File(path, "r") as handle:
            record["file_attrs"] = {
                str(k): json_value(v) for k, v in handle.attrs.items()
            }
            record["top_level_keys"] = list(handle.keys())
            record["schema"] = schema_inventory(handle)
            record["metadata"] = scalar_metadata(handle)
            image_path = "img_nir" if "img_nir" in handle else None
            record["image_dataset_path"] = image_path
            if image_path is None:
                record["status"] = "no_image_dataset"
                return record, None
            dataset = handle[image_path]
            record["image"] = dataset_description(dataset)
            record["axis_interpretation"] = "[frame, y, x] grayscale" if dataset.ndim == 3 else "unknown"
            record["frame_count"] = int(dataset.shape[0]) if dataset.ndim else 0
            record["timestamp"] = timestamp_metrics(handle, record["frame_count"])
            indices = choose_indices(record["frame_count"], max_samples)
            frames = []
            try:
                # Explicit individual reads avoid h5py's sorted-fancy-index rules
                # and make the bounded access pattern obvious.
                for index in indices:
                    frames.append(np.asarray(dataset[int(index)]))
            except Exception as exc:
                record["status"] = "image_read_error"
                record["error"] = error_text(exc)
                record["sample_indices_attempted"] = [int(i) for i in indices]
                return record, None
            frame_array = np.stack(frames)
            record["status"] = "usable"
            record["sample_indices"] = [int(i) for i in indices]
            record["sample_count"] = int(len(frame_array))
            percentiles = np.percentile(frame_array, [0, 1, 5, 25, 50, 75, 95, 99, 100])
            record["intensity"] = {
                "percentile_levels": [0, 1, 5, 25, 50, 75, 95, 99, 100],
                "percentile_values": [float(v) for v in percentiles],
                "mean": float(frame_array.mean()),
                "std": float(frame_array.std()),
            }
            adjacent_changes = []
            for left, right, a, b in zip(indices[:-1], indices[1:], frame_array[:-1], frame_array[1:]):
                if right == left + 1:
                    adjacent_changes.append({
                        "from_index": int(left),
                        "to_index": int(right),
                        "mean_absolute_difference": float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32)))),
                        "identical": bool(np.array_equal(a, b)),
                    })
            record["adjacent_frame_change"] = adjacent_changes
            fg = foreground_metrics(frame_array)
            record["foreground_heuristic"] = fg
            components = [item["component"] for item in fg if item["component"]]
            if components:
                record["foreground_summary"] = {
                    "detected_fraction": len(components) / len(fg),
                    "boundary_touch_fraction": float(np.mean([c["touches_boundary"] for c in components])),
                    "median_major_sigma_span_px": 4.0 * float(np.median([c["major_sigma_span_downsampled_px"] for c in components])),
                    "median_width_proxy_px": 4.0 * float(np.median([c["width_proxy_downsampled_px"] for c in components])),
                    "note": "moving-foreground proxy at 4x downsampling; not a worm segmentation",
                }
            return record, frame_array
    except Exception as exc:
        record["status"] = "hdf5_open_or_schema_error"
        record["error"] = error_text(exc)
        return record, None


def save_figures(records: list[dict[str, Any]], frames_by_name: dict[str, np.ndarray], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    columns = 4
    figure, axes = plt.subplots(len(records), columns, figsize=(12, 2.25 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        frames = frames_by_name.get(record["name"])
        for col in range(columns):
            ax = axes[row, col]
            ax.axis("off")
            if frames is not None:
                position = round(col * (len(frames) - 1) / max(columns - 1, 1))
                ax.imshow(frames[position], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{record['name']}\nframe {record['sample_indices'][position]}", fontsize=7)
            elif col == 0:
                ax.text(0.02, 0.5, f"{record['name']}\n{record['status']}\n{record.get('error', '')[:95]}", fontsize=7, va="center", wrap=True)
    figure.suptitle("Uniformly distributed NIR samples (failed reads shown as findings)")
    figure.tight_layout()
    figure.savefig(output / "data_audit_recording_montage.png", dpi=140)
    plt.close(figure)

    usable = [record for record in records if record["status"] == "usable"]
    figure, (hist_ax, change_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    for record in usable:
        frames = frames_by_name[record["name"]]
        values, edges = np.histogram(frames[:, ::4, ::4], bins=np.arange(257), density=True)
        hist_ax.plot(edges[:-1], values, label=record["name"], alpha=0.85)
        changes = record["adjacent_frame_change"]
        if changes:
            change_ax.plot([x["from_index"] for x in changes], [x["mean_absolute_difference"] for x in changes], marker="o", label=record["name"])
    hist_ax.set(xlabel="uint8 intensity", ylabel="density", title="Sampled-frame intensity distributions")
    change_ax.set(xlabel="frame index", ylabel="mean absolute difference", title="Adjacent sampled-frame change")
    hist_ax.legend(fontsize=7)
    change_ax.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "data_audit_intensity_and_change.png", dpi=160)
    plt.close(figure)

    if usable:
        record = usable[0]
        frames = frames_by_name[record["name"]]
        indices = np.asarray(record["sample_indices"])
        consecutive = np.flatnonzero(np.diff(indices) == 1)
        start = int(consecutive[0]) if len(consecutive) else 0
        selected = list(range(start, min(start + 8, len(frames))))
        figure, axes = plt.subplots(2, 4, figsize=(12, 5), squeeze=False)
        for ax, pos in zip(axes.flat, selected):
            ax.imshow(frames[pos], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"frame {indices[pos]}")
            ax.axis("off")
        for ax in axes.flat[len(selected):]:
            ax.axis("off")
        figure.suptitle(f"Bounded typical locomotion sequence: {record['name']}")
        figure.tight_layout()
        figure.savefig(output / "data_audit_locomotion_sequence.png", dpi=160)
        plt.close(figure)

        # Screening examples are selected from the same bounded sample, with
        # labels deliberately stating what the simple heuristic can establish.
        candidates: list[tuple[str, int, int, float, float, bool]] = []
        for item in usable:
            item_frames = frames_by_name[item["name"]]
            fg = item.get("foreground_heuristic", [])
            for pos, frame in enumerate(item_frames):
                component = fg[pos].get("component") if pos < len(fg) else None
                boundary = bool(component and component["touches_boundary"])
                elongation = (
                    component["width_proxy_downsampled_px"] / max(component["major_sigma_span_downsampled_px"], 1e-6)
                    if component else 0.0
                )
                candidates.append((item["name"], pos, item["sample_indices"][pos], float(frame.std()), elongation, boundary))
        selections = [
            ("fully in-frame candidate", min((x for x in candidates if not x[5]), key=lambda x: x[4], default=candidates[0])),
            ("boundary-contact candidate\n(anatomical endpoint unresolved)", next((x for x in candidates if x[5]), candidates[0])),
            ("possible partial-FOV candidate\n(requires manual annotation)", max(candidates, key=lambda x: (x[5], x[4]))),
            ("tight-bend/overlap candidate\n(compactness proxy)", max(candidates, key=lambda x: x[4])),
            ("lowest sampled contrast", min(candidates, key=lambda x: x[3])),
            ("highest sampled contrast", max(candidates, key=lambda x: x[3])),
        ]
        figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), squeeze=False)
        for ax, (label, candidate) in zip(axes.flat, selections):
            name, pos, index, _, _, _ = candidate
            ax.imshow(frames_by_name[name][pos], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{label}\n{name}, frame {index}", fontsize=9)
            ax.axis("off")
        figure.suptitle("Heuristic screening examples — candidates, not annotations")
        figure.tight_layout()
        figure.savefig(output / "data_audit_screening_examples.png", dpi=160)
        plt.close(figure)


def split_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    sessions: dict[str, list[str]] = {}
    for record in records:
        target = record.get("symlink_target") or record["resolved_path"]
        parts = Path(target).parts
        project = next((part.removeprefix("prj_") for part in parts if part.startswith("prj_")), "unknown")
        date = record["name"][:10]
        groups.setdefault(project, []).append(record["name"])
        sessions.setdefault(f"{project}:{date}", []).append(record["name"])
    folds = []
    group_names = sorted(groups)
    for test_group in group_names:
        folds.append({
            "test_background_group": test_group,
            "test_recordings": sorted(groups[test_group]),
            "development_background_groups": [g for g in group_names if g != test_group],
        })
    return {
        "status": "provisional; audit readability prevents a final train/validation/test allocation",
        "background_group_key": "project family inferred from read-only symlink target",
        "session_group_key": "project family + acquisition date",
        "guard_interval_frames": None,
        "guard_interval_rationale": "whole recordings/sessions are grouped, so temporal guard intervals are unnecessary",
        "background_groups": {k: sorted(v) for k, v in groups.items()},
        "session_groups": {k: sorted(v) for k, v in sessions.items()},
        "recommended_protocol": "leave-one-background-family-out grouped cross-validation; never split frames within a recording",
        "folds": folds,
        "caveat": "all currently readable recordings belong to starvation; no leakage-safe multi-background evaluation can yet be executed",
    }


def write_outputs(records: list[dict[str, Any]], output: Path, max_samples: int) -> None:
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
    payload = {
        "audit_schema_version": 1,
        "method": {
            "read_mode": "h5py read-only; serial one-file-at-a-time",
            "max_sampled_frames_per_recording": max_samples,
            "sampling": "uniform coverage plus up to 8 consecutive central frames",
            "source_mutation": "none",
        },
        "inventory": {
            "recording_count": len(records),
            "total_size_bytes": sum(int(r.get("size_bytes", 0)) for r in records),
            "status_counts": status_counts,
        },
        "split_proposal": split_manifest(records),
        "recordings": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "data_audit_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    fields = ["name", "status", "size_bytes", "resolved_path", "frame_count", "image_shape", "dtype", "observed_fps", "sample_count", "intensity_p01", "intensity_p50", "intensity_p99", "boundary_touch_fraction", "error"]
    with (output / "data_audit_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            intensity = record.get("intensity", {}).get("percentile_values", [])
            writer.writerow({
                "name": record["name"], "status": record["status"], "size_bytes": record.get("size_bytes"),
                "resolved_path": record.get("resolved_path"), "frame_count": record.get("frame_count"),
                "image_shape": "x".join(map(str, record.get("image", {}).get("shape", []))),
                "dtype": record.get("image", {}).get("dtype"), "observed_fps": record.get("timestamp", {}).get("observed_fps"),
                "sample_count": record.get("sample_count"), "intensity_p01": intensity[1] if len(intensity) > 1 else None,
                "intensity_p50": intensity[4] if len(intensity) > 4 else None, "intensity_p99": intensity[7] if len(intensity) > 7 else None,
                "boundary_touch_fraction": record.get("foreground_summary", {}).get("boundary_touch_fraction"), "error": record.get("error"),
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("nir_videos"))
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--summary-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--figure-dir", type=Path, default=Path("artifacts/final_figures"))
    args = parser.parse_args()
    if not 1 <= args.max_samples <= 32:
        parser.error("--max-samples must be between 1 and 32")
    return args


def main() -> int:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.h5"))
    if not paths:
        raise SystemExit(f"no .h5 files found under {args.input_dir}")
    records: list[dict[str, Any]] = []
    frames_by_name: dict[str, np.ndarray] = {}
    for path in paths:  # serial by design: at most one active HDF5 reader
        print(f"inspecting {path}", flush=True)
        record, frames = inspect_recording(path, args.max_samples)
        records.append(record)
        if frames is not None:
            frames_by_name[record["name"]] = frames
        print(f"  {record['status']}", flush=True)
    write_outputs(records, args.summary_dir, args.max_samples)
    save_figures(records, frames_by_name, args.figure_dir)
    print(f"wrote {args.summary_dir / 'data_audit_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
