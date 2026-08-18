#!/usr/bin/env python3
"""Build the preregistered EXP-0006 balanced real-texture crop artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.real_crop import (
    ScaledCropRequest,
    atomic_publish,
    attempt_scaled_real_crop,
    canonical_manifest_sha256,
    half_open_support,
    support_bitmask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp_0006_balanced_real_crop"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(values.shape, separators=(",", ":")).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def selection_identity(entry: dict[str, Any]) -> tuple[str, int, str, float]:
    return (
        str(entry["source_group"]),
        int(entry["source_frame_index"]),
        str(entry["hidden_end"]),
        float(entry["hidden_fraction"]),
    )


def selection_digest(seed: int, entry: dict[str, Any]) -> str:
    recording, frame, end, fraction = selection_identity(entry)
    payload = f"{seed}|{recording}|{frame}|{end}|{fraction:.2f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_balanced_cases(
    entries: Iterable[dict[str, Any]],
    *,
    seed: int,
    recordings: tuple[str, ...],
    ends: tuple[str, ...],
    fractions: tuple[float, ...],
    per_cell: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """SHA-rank valid cases and take an exact quota from every frozen cell."""

    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    cells: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, str, float]] = set()
    for original in entries:
        if original.get("rejection_reason") is not None:
            continue
        identity = selection_identity(original)
        if identity in seen:
            raise ValueError(f"duplicate valid condition identity: {identity}")
        seen.add(identity)
        recording, _, end, fraction = identity
        if recording not in recordings or end not in ends or fraction not in fractions:
            raise ValueError(f"unexpected valid case cell: {identity}")
        entry = dict(original)
        entry["selection_sha256"] = selection_digest(seed, entry)
        cells[(recording, end, fraction)].append(entry)

    selected: list[dict[str, Any]] = []
    pool_counts: dict[str, int] = {}
    for recording in recordings:
        for fraction in fractions:
            for end in ends:
                key = (recording, end, fraction)
                candidates = sorted(
                    cells.get(key, ()),
                    key=lambda entry: (entry["selection_sha256"], selection_identity(entry)),
                )
                label = f"{recording}|{end}|{fraction:.2f}"
                pool_counts[label] = len(candidates)
                if len(candidates) < per_cell:
                    raise RuntimeError(
                        f"cell {label} has {len(candidates)} valid cases; requires {per_cell}"
                    )
                selected.extend(candidates[:per_cell])
    identities = [selection_identity(entry) for entry in selected]
    if len(identities) != len(set(identities)):
        raise RuntimeError("balanced selection contains duplicate condition identities")
    return selected, pool_counts


def validate_output_targets(output: Path, forbidden: Iterable[Path]) -> None:
    """Refuse output or partial paths that alias any immutable input."""

    resolved_output = output.resolve(strict=False)
    partial = output.with_suffix(output.suffix + ".partial").resolve(strict=False)
    forbidden_resolved = {path.resolve(strict=True) for path in forbidden}
    if resolved_output in forbidden_resolved or partial in forbidden_resolved:
        raise ValueError("output or partial path collides with an immutable input")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    requested_partial = output.with_suffix(output.suffix + ".partial")
    if requested_partial.exists() or requested_partial.is_symlink():
        raise FileExistsError(f"refusing existing partial: {requested_partial}")


def _string_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([record[key] for record in records], dtype=object)


def write_balanced_hdf5(
    path: Path, records: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    """Write a complete balanced artifact to an exclusively created partial path."""

    string_dtype = h5py.string_dtype("utf-8")
    count = len(records)
    with h5py.File(path, "x") as output:
        output.attrs["schema_version"] = 1
        output.attrs["complete"] = False
        for key, value in metadata.items():
            output.attrs[key] = value
        output.create_dataset(
            "resized_image",
            data=np.stack([record["resized_image"] for record in records]),
            chunks=(1, 192, 256), compression="gzip", compression_opts=4, shuffle=True,
        )
        output.create_dataset(
            "source_centerline_xy",
            data=np.stack([record["source_centerline_xy"] for record in records]),
        )
        output.create_dataset(
            "transformed_centerline_xy",
            data=np.stack([record["transformed_centerline_xy"] for record in records]),
        )
        output.create_dataset("support", data=np.stack([record["support"] for record in records]))
        output.create_dataset("source_group", data=_string_array(records, "source_group"),
                              dtype=string_dtype)
        output.create_dataset("source_frame_index", data=[record["source_frame_index"] for record in records])
        output.create_dataset("accepted_image_index", data=[record["accepted_image_index"] for record in records])
        output.create_dataset("sample_position", data=[record["sample_position"] for record in records])
        output.create_dataset("hidden_end", data=_string_array(records, "hidden_end"),
                              dtype=string_dtype)
        output.create_dataset("hidden_fraction", data=[record["hidden_fraction"] for record in records])
        output.create_dataset("source_window_k", data=[record["source_window_k"] for record in records])
        output.create_dataset("source_window_bounds_xyxy_half_open", data=np.stack([
            record["source_window_bounds_xyxy_half_open"] for record in records
        ]))
        for key in (
            "source_to_window_transform", "window_to_source_transform",
            "source_to_resized_transform", "resized_to_source_transform",
        ):
            output.create_dataset(key, data=np.stack([record[key] for record in records]))
        for key in (
            "selection_sha256", "accepted_image_sha256", "source_window_sha256",
            "resized_image_sha256", "support_bitmask",
        ):
            output.create_dataset(key, data=_string_array(records, key), dtype=string_dtype)
        output.attrs["case_count"] = count
        output.attrs["complete"] = True
        output.flush()


def validate_balanced_hdf5(
    path: Path, records: list[dict[str, Any]], *, maximum_bytes: int
) -> dict[str, Any]:
    """Exhaustively validate the published artifact against regenerated records."""

    size = path.stat().st_size
    if size > maximum_bytes:
        raise RuntimeError(f"artifact size {size} exceeds limit {maximum_bytes}")
    with h5py.File(path, "r") as artifact:
        if int(artifact.attrs.get("schema_version", -1)) != 1:
            raise RuntimeError("unexpected artifact schema")
        if not bool(artifact.attrs.get("complete")):
            raise RuntimeError("artifact completion marker is false")
        if int(artifact.attrs.get("case_count", -1)) != len(records):
            raise RuntimeError("artifact case count mismatch")
        required = {
            "resized_image", "source_centerline_xy", "transformed_centerline_xy",
            "support", "source_group", "source_frame_index", "hidden_end",
            "accepted_image_index", "sample_position", "hidden_fraction",
            "source_window_k", "source_window_bounds_xyxy_half_open",
            "source_to_window_transform", "window_to_source_transform",
            "source_to_resized_transform", "resized_to_source_transform",
            "selection_sha256", "accepted_image_sha256", "source_window_sha256",
            "resized_image_sha256", "support_bitmask",
        }
        if not required.issubset(artifact.keys()):
            raise RuntimeError(f"artifact datasets missing: {sorted(required - set(artifact.keys()))}")
        for index, record in enumerate(records):
            image = artifact["resized_image"][index]
            if not np.array_equal(image, record["resized_image"]):
                raise RuntimeError(f"stored image mismatch at case {index}")
            if array_sha256(image) != record["resized_image_sha256"]:
                raise RuntimeError(f"stored image hash mismatch at case {index}")
            for key in ("source_centerline_xy", "transformed_centerline_xy", "support"):
                if not np.array_equal(artifact[key][index], record[key]):
                    raise RuntimeError(f"stored {key} mismatch at case {index}")
            for key in (
                "source_window_bounds_xyxy_half_open", "source_to_window_transform",
                "window_to_source_transform", "source_to_resized_transform",
                "resized_to_source_transform",
            ):
                if not np.array_equal(artifact[key][index], record[key]):
                    raise RuntimeError(f"stored {key} mismatch at case {index}")
            for key in (
                "source_group", "hidden_end", "selection_sha256", "accepted_image_sha256",
                "source_window_sha256", "resized_image_sha256", "support_bitmask",
            ):
                if artifact[key].asstr()[index] != str(record[key]):
                    raise RuntimeError(f"stored {key} mismatch at case {index}")
            for key in (
                "source_frame_index", "accepted_image_index", "sample_position",
                "hidden_fraction", "source_window_k",
            ):
                if artifact[key][index] != record[key]:
                    raise RuntimeError(f"stored {key} mismatch at case {index}")
    return {"size_bytes": size, "sha256": file_sha256(path)}


def plot_overlay(ax: Any, record: dict[str, Any]) -> None:
    image = record["resized_image"]
    low, high = np.percentile(image, (1, 99))
    ax.imshow(image, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    points, support = record["transformed_centerline_xy"], record["support"]
    ax.plot(points[:, 0], points[:, 1], color="#f4f4f5", linewidth=0.5, alpha=0.6)
    ax.scatter(points[support, 0], points[support, 1], c="#19d3ae", s=4)
    ax.scatter(points[~support, 0], points[~support, 1], c="#ff4d6d", s=4)
    ax.set_xlim(-0.5, 255.5)
    ax.set_ylim(191.5, -0.5)
    ax.set_title(record["title"], fontsize=6)
    ax.axis("off")


def plot_evidence(records: list[dict[str, Any]], seed: int, path: Path) -> None:
    rng = np.random.default_rng(seed)
    random_cases = [records[index] for index in rng.choice(len(records), 6, replace=False)]
    ordered = sorted(records, key=lambda record: (record["source_window_k"], record["selection_sha256"]))
    scale_cases = ordered[:3] + ordered[-3:]
    figure, axes = plt.subplots(2, 6, figsize=(16, 5.8), squeeze=False)
    for row, (label, cases) in enumerate((
        ("seeded random", random_cases), ("max scale (left) / min scale (right)", scale_cases)
    )):
        axes[row, 0].text(-0.08, 0.5, label, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontweight="bold")
        for ax, case in zip(axes[row], cases):
            plot_overlay(ax, case)
    figure.suptitle("EXP-0006 balanced crop evidence (green=visible, pink=hidden)")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_all_40_percent(records: list[dict[str, Any]], path: Path) -> None:
    cases = [record for record in records if math.isclose(record["hidden_fraction"], 0.40)]
    figure, axes = plt.subplots(6, 10, figsize=(18, 9), squeeze=False)
    for ax, case in zip(axes.flat, cases):
        plot_overlay(ax, case)
    figure.suptitle("All 60 balanced 40%-hidden cases")
    figure.tight_layout(rect=(0, 0, 1, 0.97), pad=0.35)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_balance(records: list[dict[str, Any]], recordings: tuple[str, ...], path: Path) -> None:
    cell_counts = Counter((record["source_group"], record["hidden_end"],
                           record["hidden_fraction"]) for record in records)
    columns = [(end, fraction) for fraction in (.05, .10, .20, .30, .40)
               for end in ("head", "tail")]
    values = np.asarray([[cell_counts[(recording, end, fraction)]
                          for end, fraction in columns] for recording in recordings])
    figure, ax = plt.subplots(figsize=(12, 3.8))
    image = ax.imshow(values, vmin=0, vmax=max(10, int(values.max())), cmap="Blues")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, str(values[row, column]), ha="center", va="center")
    ax.set_xticks(range(len(columns)), [f"{end}\n{fraction:.0%}" for end, fraction in columns])
    ax.set_yticks(range(len(recordings)), recordings)
    ax.set_title("Selected cases per recording/end/fraction cell")
    figure.colorbar(image, ax=ax, label="cases")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_source_reuse(records: list[dict[str, Any]], path: Path) -> None:
    reuse = Counter((record["source_group"], record["source_frame_index"]) for record in records)
    per_recording = defaultdict(list)
    for (recording, _), count in reuse.items():
        per_recording[recording].append(count)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.arange(0.5, max(reuse.values()) + 1.5)
    axes[0].hist(list(reuse.values()), bins=bins, rwidth=0.85)
    axes[0].set_xlabel("selected conditions per reused source frame")
    axes[0].set_ylabel("unique source frames")
    axes[0].set_title("Source-frame reuse distribution")
    axes[1].bar(per_recording.keys(), [len(values) for values in per_recording.values()])
    axes[1].set_ylabel("unique selected source frames")
    axes[1].set_title("Unique source coverage by recording")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    args = parser.parse_args()
    config = json.loads((args.experiment_dir / "config.json").read_text())
    metrics_path = PROJECT_ROOT / config["exp_0005_metrics_path"]
    proxy_path = Path(config["proxy_path"])
    output_path = Path(config["output_path"])
    if file_sha256(metrics_path) != config["exp_0005_metrics_sha256"]:
        raise RuntimeError("frozen EXP-0005 metrics SHA-256 mismatch")
    exp5 = json.loads(metrics_path.read_text())
    if canonical_manifest_sha256(exp5["case_manifest"]) != config["exp_0005_case_manifest_sha256"]:
        raise RuntimeError("frozen EXP-0005 case-manifest SHA-256 mismatch")
    if exp5["case_manifest_sha256"] != config["exp_0005_case_manifest_sha256"]:
        raise RuntimeError("EXP-0005 recorded case-manifest identity mismatch")
    if file_sha256(proxy_path) != config["proxy_sha256"]:
        raise RuntimeError("immutable proxy SHA-256 mismatch")
    validate_output_targets(output_path, (metrics_path, proxy_path))

    recordings = tuple(config["expected_recordings"])
    ends = tuple(config["hidden_ends"])
    fractions = tuple(float(value) for value in config["hidden_fractions"])
    selected, pool_counts = select_balanced_cases(
        exp5["case_manifest"]["entries"], seed=int(config["seed"]),
        recordings=recordings, ends=ends, fractions=fractions,
        per_cell=int(config["cases_per_recording_condition"]),
    )
    if len(selected) != int(config["expected_cases"]):
        raise RuntimeError("balanced selection does not contain exactly 300 cases")

    records: list[dict[str, Any]] = []
    provenance_checks = interpolation_checks = support_checks = transform_checks = 0
    maximum_roundtrip_error = 0.0
    # Exactly one read-only proxy handle. Every selected accepted image is one bounded read.
    with h5py.File(proxy_path, "r") as proxy:
        for entry in selected:
            group = proxy[entry["source_group"]]
            accepted_index = int(entry["accepted_image_index"])
            sample_position = int(entry["sample_position"])
            image = group["accepted_image"][accepted_index]
            centerline = group["centerline_xy"][sample_position].astype(np.float64)
            if array_sha256(image) != entry["accepted_image_sha256"]:
                raise RuntimeError("selected accepted-image hash mismatch")
            request = ScaledCropRequest(
                entry["hidden_end"], float(entry["hidden_fraction"]), 192, 256, 96, 240
            )
            attempt = attempt_scaled_real_crop(image, centerline, request)
            if attempt.crop is None:
                raise RuntimeError("frozen valid EXP-0005 case no longer regenerates")
            crop = attempt.crop
            x0, y0 = crop.source_origin_xy
            height, width = crop.source_window_shape
            bounds = [x0, y0, x0 + width, y0 + height]
            if crop.source_window_k != entry["source_window_k"] or bounds != entry[
                "source_window_bounds_xyxy_half_open"
            ]:
                raise RuntimeError("regenerated window differs from EXP-0005")
            if array_sha256(crop.source_window) != entry["source_window_sha256"]:
                raise RuntimeError("regenerated direct-window hash differs from EXP-0005")
            provenance_checks += 1
            resized_hash = array_sha256(crop.image)
            if resized_hash != entry["resized_image_sha256"]:
                raise RuntimeError("regenerated interpolation differs from EXP-0005")
            interpolation_checks += 1
            source_support = half_open_support(centerline, (x0, y0), height, width)
            resized_support = half_open_support(crop.centerline_resized_xy, (0, 0), 192, 256)
            if not (np.array_equal(source_support, crop.support)
                    and np.array_equal(resized_support, crop.support)
                    and support_bitmask(crop.support) == entry["requested_support_bitmask"]):
                raise RuntimeError("regenerated support mapping mismatch")
            support_checks += 1
            restored = crop.resized_to_source(crop.centerline_resized_xy)
            error = float(np.max(np.abs(restored - centerline)))
            maximum_roundtrip_error = max(maximum_roundtrip_error, error)
            if error > float(config["maximum_roundtrip_error_px"]):
                raise RuntimeError("regenerated transform exceeds round-trip limit")
            scale = crop.scale
            source_to_window = np.asarray([
                [1.0, 0.0, -float(x0)], [0.0, 1.0, -float(y0)], [0.0, 0.0, 1.0]
            ])
            window_to_source = np.asarray([
                [1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]
            ])
            source_to_resized = np.asarray(entry["source_to_resized_transform"])
            resized_to_source = np.asarray(entry["resized_to_source_transform"])
            if not (np.array_equal(source_to_resized, np.asarray([
                [scale, 0.0, -scale * x0], [0.0, scale, -scale * y0], [0.0, 0.0, 1.0]
            ])) and np.allclose(source_to_resized @ resized_to_source, np.eye(3),
                                atol=1e-12, rtol=0)):
                raise RuntimeError("frozen transform mapping mismatch")
            transform_checks += 1
            records.append({
                "resized_image": crop.image,
                "source_centerline_xy": centerline,
                "transformed_centerline_xy": crop.centerline_resized_xy,
                "support": crop.support,
                "source_group": entry["source_group"],
                "source_frame_index": int(entry["source_frame_index"]),
                "accepted_image_index": accepted_index,
                "sample_position": sample_position,
                "hidden_end": entry["hidden_end"],
                "hidden_fraction": float(entry["hidden_fraction"]),
                "source_window_k": crop.source_window_k,
                "source_window_bounds_xyxy_half_open": np.asarray(bounds, dtype=np.int32),
                "source_to_window_transform": source_to_window,
                "window_to_source_transform": window_to_source,
                "source_to_resized_transform": source_to_resized,
                "resized_to_source_transform": resized_to_source,
                "selection_sha256": entry["selection_sha256"],
                "accepted_image_sha256": entry["accepted_image_sha256"],
                "source_window_sha256": entry["source_window_sha256"],
                "resized_image_sha256": resized_hash,
                "support_bitmask": support_bitmask(crop.support),
                "title": (
                    f"{entry['source_group']} f{entry['source_frame_index']} "
                    f"{entry['hidden_end']} {entry['hidden_fraction']:.0%}, k={crop.source_window_k}"
                ),
            })

    selection_manifest = {
        "schema_version": 1,
        "selection": config["selection"],
        "seed": int(config["seed"]),
        "exp_0005_case_manifest_sha256": config["exp_0005_case_manifest_sha256"],
        "identities": [{
            "source_group": record["source_group"],
            "source_frame_index": record["source_frame_index"],
            "hidden_end": record["hidden_end"],
            "hidden_fraction": record["hidden_fraction"],
            "selection_sha256": record["selection_sha256"],
        } for record in records],
    }
    selection_manifest_hash = canonical_manifest_sha256(selection_manifest)
    metadata = {
        "experiment": config["experiment"],
        "evidence_label": config["evidence_label"],
        "proxy_path": str(proxy_path),
        "proxy_sha256": config["proxy_sha256"],
        "exp_0005_metrics_sha256": config["exp_0005_metrics_sha256"],
        "exp_0005_case_manifest_sha256": config["exp_0005_case_manifest_sha256"],
        "selection_manifest_sha256": selection_manifest_hash,
        "selection_rule": config["selection"],
        "resize_interpolation": "bilinear_align_corners_false",
        "geometry_convention": "xy; edge-aligned isotropic scale; half-open FOV",
    }
    atomic_publish(
        proxy_path, output_path,
        lambda partial: write_balanced_hdf5(partial, records, metadata),
    )
    artifact = validate_balanced_hdf5(
        output_path, records, maximum_bytes=int(config["maximum_output_bytes"])
    )

    figures = args.experiment_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_balance(records, recordings, figures / "balance.png")
    plot_source_reuse(records, figures / "source_reuse.png")
    plot_evidence(records, int(config["seed"]), figures / "random_and_scale_evidence.png")
    plot_all_40_percent(records, figures / "all_40_percent_cases.png")

    cell_counts = Counter(
        f"{record['source_group']}|{record['hidden_end']}|{record['hidden_fraction']:.2f}"
        for record in records
    )
    condition_identities = [
        (record["source_group"], record["source_frame_index"], record["hidden_end"],
         record["hidden_fraction"]) for record in records
    ]
    reuse = Counter((record["source_group"], record["source_frame_index"]) for record in records)
    runtime = time.perf_counter() - started
    success = (
        len(records) == int(config["expected_cases"])
        and set(cell_counts.values()) == {int(config["cases_per_recording_condition"])}
        and len(condition_identities) == len(set(condition_identities))
        and provenance_checks == interpolation_checks == support_checks == transform_checks == len(records)
        and maximum_roundtrip_error <= float(config["maximum_roundtrip_error_px"])
        and artifact["size_bytes"] <= int(config["maximum_output_bytes"])
    )
    metrics = {
        "experiment": config["experiment"],
        "status": "ACCEPT" if success else "REJECT",
        "evidence_label": config["evidence_label"],
        "exp_0005_metrics_sha256": config["exp_0005_metrics_sha256"],
        "exp_0005_case_manifest_sha256": config["exp_0005_case_manifest_sha256"],
        "proxy_sha256": config["proxy_sha256"],
        "selection_manifest_sha256": selection_manifest_hash,
        "selected_cases": len(records),
        "expected_cases": int(config["expected_cases"]),
        "cell_counts": dict(sorted(cell_counts.items())),
        "valid_pool_counts": pool_counts,
        "duplicate_condition_identities": len(condition_identities) - len(set(condition_identities)),
        "direct_window_hash_checks_passed": provenance_checks,
        "interpolation_hash_checks_passed": interpolation_checks,
        "support_checks_passed": support_checks,
        "transform_checks_passed": transform_checks,
        "maximum_roundtrip_error_px": maximum_roundtrip_error,
        "roundtrip_limit_px": float(config["maximum_roundtrip_error_px"]),
        "unique_source_frames": len(reuse),
        "maximum_conditions_per_source_frame": max(reuse.values()),
        "source_reuse_count_distribution": dict(sorted(Counter(reuse.values()).items())),
        "source_window_k": {
            "minimum": min(record["source_window_k"] for record in records),
            "maximum": max(record["source_window_k"] for record in records),
            "median": float(np.median([record["source_window_k"] for record in records])),
        },
        "output_path": str(output_path),
        "output_size_bytes": artifact["size_bytes"],
        "maximum_output_bytes": int(config["maximum_output_bytes"]),
        "output_sha256": artifact["sha256"],
        "complete_marker": True,
        "runtime_seconds": runtime,
        "runtime_limit_seconds": 60 * float(config["maximum_wall_time_minutes"]),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "selection_manifest": selection_manifest,
    }
    (args.experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    summary = {key: value for key, value in metrics.items() if key != "selection_manifest"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
