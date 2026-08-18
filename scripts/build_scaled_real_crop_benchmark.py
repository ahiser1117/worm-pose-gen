#!/usr/bin/env python3
"""Run preregistered EXP-0005 on immutable stored proxy images only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.real_crop import (
    ScaledCropRequest,
    attempt_scaled_real_crop,
    bilinear_resize_align_corners_false,
    canonical_manifest_sha256,
    half_open_support,
    support_bitmask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp_0005_scaled_real_crop"
PROXY_PATH = Path(
    "/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/proxy_v1/proxy_labels.h5"
)
RECORDINGS = ("2023-09-19-01", "2023-09-27-01", "2023-10-11-01")


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


def plot_case(ax: Any, case: dict[str, Any]) -> None:
    image = case["image"]
    low, high = np.percentile(image, (1, 99))
    ax.imshow(image, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    points, support = case["points"], case["support"]
    ax.plot(points[:, 0], points[:, 1], color="#f4f4f5", linewidth=0.6, alpha=0.6)
    ax.scatter(points[support, 0], points[support, 1], c="#19d3ae", s=5)
    ax.scatter(points[~support, 0], points[~support, 1], c="#ff4d6d", s=5)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_title(case["title"], fontsize=7)
    ax.axis("off")


def evidence_montage(
    random_cases: list[dict[str, Any]],
    maximum_hidden: list[dict[str, Any]],
    scale_extremes: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    path: Path,
) -> None:
    rows = (
        ("seeded random valid", random_cases),
        ("40% hidden valid", maximum_hidden),
        ("max scale (left) / min scale (right)", scale_extremes),
        ("rejected request (stored frame)", rejected),
    )
    figure, axes = plt.subplots(4, 6, figsize=(16, 11), squeeze=False)
    for row_index, (label, cases) in enumerate(rows):
        axes[row_index, 0].text(
            -0.08, 0.5, label, transform=axes[row_index, 0].transAxes,
            rotation=90, va="center", ha="right", fontsize=9, fontweight="bold",
        )
        for ax in axes[row_index]:
            ax.axis("off")
        for ax, case in zip(axes[row_index], cases):
            plot_case(ax, case)
    figure.suptitle(
        "EXP-0005 scaled real-texture evidence (green=visible, pink=hidden)",
        fontsize=11,
        y=0.995,
    )
    figure.tight_layout(pad=0.8, rect=(0, 0, 1, 0.975))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def yield_plot(
    per_recording: dict[str, dict[str, int]], accepted: dict[str, int], path: Path
) -> None:
    keys = [f"{end}_{fraction:.2f}" for fraction in (.05, .10, .20, .30, .40)
            for end in ("head", "tail")]
    labels = [f"{end}\n{fraction:.0%}" for fraction in (.05, .10, .20, .30, .40)
              for end in ("head", "tail")]
    figure, ax = plt.subplots(figsize=(12, 5.5))
    x, width = np.arange(len(keys)), 0.24
    for offset, (recording, values) in enumerate(per_recording.items()):
        counts = [values.get(key, 0) for key in keys]
        heights = [100 * count / accepted[recording] for count in counts]
        positions = x + (offset - 1) * width
        ax.bar(positions, heights, width, label=recording)
        for xpos, height, count in zip(positions, heights, counts):
            ax.text(xpos, height + 1, str(count), ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("valid scaled crops (% accepted frames); labels are counts")
    ax.set_xlabel("requested hidden end and fraction")
    ax.set_title("Exact-support scaled camera-window yield")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def scale_plot(valid_cases: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    k_values = np.asarray([case["k"] for case in valid_cases])
    axes[0].hist(k_values, bins=np.arange(k_values.min() - 0.5, k_values.max() + 1.5),
                 color="#3b82f6")
    axes[0].set_xlabel("smallest valid source-window k (window=4k x 3k)")
    axes[0].set_ylabel("valid requests")
    axes[0].set_title("Source-window scale distribution")
    fractions = (.05, .10, .20, .30, .40)
    groups = [[case["k"] for case in valid_cases if case["fraction"] == fraction]
              for fraction in fractions]
    axes[1].boxplot(groups, tick_labels=[f"{value:.0%}" for value in fractions])
    axes[1].set_xlabel("hidden fraction (head and tail combined)")
    axes[1].set_ylabel("smallest valid k")
    axes[1].set_title("Required source-window size by condition")
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
    proxy_path = Path(config["proxy_path"])
    if proxy_path != PROXY_PATH:
        raise RuntimeError("EXP-0005 proxy path differs from its frozen declaration")
    observed_proxy_hash = file_sha256(proxy_path)
    if observed_proxy_hash != config["proxy_sha256"]:
        raise RuntimeError("immutable proxy artifact SHA-256 mismatch")
    if config["resize_interpolation"] != "bilinear_align_corners_false":
        raise RuntimeError("unexpected resize interpolation declaration")

    figures_dir = args.experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fractions = tuple(float(value) for value in config["hidden_fractions"])
    ends = tuple(config["hidden_ends"])
    accepted_by_recording: dict[str, int] = {}
    complete_by_recording: dict[str, int] = {}
    per_recording_condition: dict[str, dict[str, int]] = {
        recording: defaultdict(int) for recording in RECORDINGS
    }
    rejections: Counter[str] = Counter()
    valid_cases: list[dict[str, Any]] = []
    rejected_evidence: dict[tuple[str, str, str], dict[str, Any]] = {}
    manifest_entries: list[dict[str, Any]] = []
    source_errors: list[float] = []
    resized_errors: list[float] = []
    provenance_checks = support_checks = deterministic_checks = 0

    # Exactly one read-only proxy handle; accepted images are individual bounded reads.
    with h5py.File(proxy_path, "r") as proxy:
        if not bool(proxy.attrs.get("complete")) or int(proxy.attrs.get("schema_version", -1)) != 1:
            raise RuntimeError("proxy artifact is incomplete or has an unexpected schema")
        if tuple(proxy.keys()) != RECORDINGS:
            raise RuntimeError(f"unexpected proxy recordings: {tuple(proxy.keys())}")
        for recording in RECORDINGS:
            group = proxy[recording]
            positions = group["accepted_sample_position"][:]
            accepted_by_recording[recording] = len(positions)
            complete_count = 0
            for accepted_index, sample_position in enumerate(positions):
                image = group["accepted_image"][accepted_index]
                image_hash = array_sha256(image)
                centerline = group["centerline_xy"][int(sample_position)].astype(np.float64)
                frame_index = int(group["sample_frame_index"][int(sample_position)])
                frame_complete = True
                for fraction in fractions:
                    for end in ends:
                        request = ScaledCropRequest(
                            end, fraction,
                            int(config["output_height"]), int(config["output_width"]),
                            int(config["source_window_k_min"]),
                            int(config["source_window_k_max"]),
                        )
                        attempt = attempt_scaled_real_crop(image, centerline, request)
                        base_entry: dict[str, Any] = {
                            "source_group": recording,
                            "source_frame_index": frame_index,
                            "accepted_image_index": accepted_index,
                            "sample_position": int(sample_position),
                            "accepted_image_sha256": image_hash,
                            "hidden_end": end,
                            "hidden_fraction": fraction,
                            "requested_support_bitmask": support_bitmask(attempt.target_support),
                        }
                        condition = f"{end}_{fraction:.2f}"
                        if attempt.crop is None:
                            frame_complete = False
                            reason = str(attempt.rejection_reason)
                            rejections[reason] += 1
                            base_entry.update({
                                "source_window_k": None,
                                "source_window_bounds_xyxy_half_open": None,
                                "scale_xy": None,
                                "source_to_window_transform": None,
                                "source_to_resized_transform": None,
                                "resized_to_source_transform": None,
                                "actual_source_support_bitmask": None,
                                "actual_resized_support_bitmask": None,
                                "source_window_sha256": None,
                                "resized_image_sha256": None,
                                "rejection_reason": reason,
                            })
                            manifest_entries.append(base_entry)
                            evidence_key = (recording, reason, end)
                            if evidence_key not in rejected_evidence:
                                rejected_evidence[evidence_key] = {
                                    "image": image.copy(), "points": centerline.copy(),
                                    "support": attempt.target_support.copy(),
                                    "title": f"{recording} f{frame_index}\n{end} {fraction:.0%}: {reason}",
                                }
                            continue

                        crop = attempt.crop
                        x0, y0 = crop.source_origin_xy
                        height, width = crop.source_window_shape
                        k, scale = crop.source_window_k, crop.scale
                        direct = image[y0 : y0 + height, x0 : x0 + width]
                        if not np.array_equal(crop.source_window, direct):
                            raise RuntimeError("direct source-window provenance failed")
                        expected_output = bilinear_resize_align_corners_false(
                            direct, request.output_height, request.output_width
                        )
                        if not np.array_equal(crop.image, expected_output):
                            raise RuntimeError("frozen interpolation provenance failed")
                        provenance_checks += 1
                        before = half_open_support(centerline, (x0, y0), height, width)
                        after = half_open_support(
                            crop.centerline_resized_xy, (0, 0),
                            request.output_height, request.output_width,
                        )
                        if not (np.array_equal(before, crop.support)
                                and np.array_equal(after, crop.support)):
                            raise RuntimeError("independent support recomputation failed")
                        support_checks += 1
                        source_error = float(np.max(np.abs(
                            crop.window_to_source(crop.source_to_window(centerline)) - centerline
                        )))
                        resized_error = float(np.max(np.abs(
                            crop.resized_to_source(crop.source_to_resized(centerline)) - centerline
                        )))
                        source_errors.append(source_error)
                        resized_errors.append(resized_error)
                        repeat = attempt_scaled_real_crop(image, centerline, request)
                        if repeat.crop is None:
                            raise RuntimeError("deterministic repeat rejected a valid crop")
                        if (repeat.crop.source_window_k != k
                                or repeat.crop.source_origin_xy != crop.source_origin_xy
                                or not np.array_equal(repeat.crop.image, crop.image)):
                            raise RuntimeError("scaled crop search/interpolation is nondeterministic")
                        deterministic_checks += 1
                        per_recording_condition[recording][condition] += 1
                        source_to_resized = [
                            [scale, 0.0, -scale * x0],
                            [0.0, scale, -scale * y0],
                            [0.0, 0.0, 1.0],
                        ]
                        base_entry.update({
                            "source_window_k": k,
                            "source_window_bounds_xyxy_half_open": [
                                x0, y0, x0 + width, y0 + height
                            ],
                            "scale_xy": [scale, scale],
                            "source_to_window_transform": [
                                [1.0, 0.0, -float(x0)],
                                [0.0, 1.0, -float(y0)],
                                [0.0, 0.0, 1.0],
                            ],
                            "source_to_resized_transform": source_to_resized,
                            "resized_to_source_transform": [
                                [1 / scale, 0.0, float(x0)],
                                [0.0, 1 / scale, float(y0)],
                                [0.0, 0.0, 1.0],
                            ],
                            "actual_source_support_bitmask": support_bitmask(before),
                            "actual_resized_support_bitmask": support_bitmask(after),
                            "source_window_sha256": array_sha256(crop.source_window),
                            "resized_image_sha256": array_sha256(crop.image),
                            "rejection_reason": None,
                        })
                        manifest_entries.append(base_entry)
                        valid_cases.append({
                            "image": crop.image, "points": crop.centerline_resized_xy,
                            "support": crop.support, "recording": recording,
                            "frame_index": frame_index, "fraction": fraction,
                            "end": end, "k": k, "scale": scale,
                            "title": f"{recording} f{frame_index}\n{end} {fraction:.0%}, k={k}, s={scale:.3f}",
                        })
                complete_count += int(frame_complete)
            complete_by_recording[recording] = complete_count

    max_source_error = max(source_errors, default=0.0)
    max_resized_error = max(resized_errors, default=0.0)
    if max_source_error > float(config["maximum_source_roundtrip_error_px"]):
        raise RuntimeError("source-window round-trip limit exceeded")
    if max_resized_error > float(config["maximum_resized_roundtrip_error_px"]):
        raise RuntimeError("resized round-trip limit exceeded")
    accepted_total = sum(accepted_by_recording.values())
    complete_total = sum(complete_by_recording.values())
    attempts = accepted_total * len(fractions) * len(ends)
    if len(manifest_entries) != attempts:
        raise RuntimeError("case manifest is incomplete")

    rng = np.random.default_rng(int(config["seed"]))
    random_cases = [valid_cases[index] for index in rng.choice(
        len(valid_cases), size=min(6, len(valid_cases)), replace=False
    )]
    maximum_hidden = [case for case in valid_cases if case["fraction"] == max(fractions)][:6]
    ordered = sorted(valid_cases, key=lambda case: (case["k"], case["recording"], case["frame_index"]))
    scale_extremes = ordered[:3] + ordered[-3:]
    rejected = list(rejected_evidence.values())[:6]
    evidence_montage(
        random_cases, maximum_hidden, scale_extremes, rejected,
        figures_dir / "scaled_real_crop_evidence.png",
    )
    yield_plot(
        {key: dict(value) for key, value in per_recording_condition.items()},
        accepted_by_recording, figures_dir / "contract_yield.png",
    )
    scale_plot(valid_cases, figures_dir / "source_window_scales.png")

    per_condition: dict[str, Any] = {}
    for fraction in fractions:
        for end in ends:
            key = f"{end}_{fraction:.2f}"
            counts = {recording: per_recording_condition[recording].get(key, 0)
                      for recording in RECORDINGS}
            per_condition[key] = {
                "valid": sum(counts.values()), "attempted": accepted_total,
                "by_recording": counts,
            }
    k_values = [case["k"] for case in valid_cases]
    k_counts = Counter(k_values)
    case_manifest: dict[str, Any] = {
        "schema_version": 1,
        "proxy_sha256": observed_proxy_hash,
        "evidence_label": config["evidence_label"],
        "coordinate_convention": (
            "xy source pixel centers; integer half-open source/output FOV; "
            "edge-aligned isotropic point transform; pixels bilinear align_corners=False"
        ),
        "array_hash_convention": "sha256(dtype ASCII + compact JSON shape + C-order bytes)",
        "entries": manifest_entries,
    }
    manifest_hash = canonical_manifest_sha256(case_manifest)
    runtime = time.perf_counter() - started
    success = complete_total >= int(config["minimum_complete_frames"])
    metrics: dict[str, Any] = {
        "experiment": config["experiment"],
        "status": "ACCEPT" if success else "REJECT",
        "evidence_label": config["evidence_label"],
        "proxy_path": str(proxy_path),
        "proxy_sha256": observed_proxy_hash,
        "accepted_proxy_frames": accepted_total,
        "accepted_by_recording": accepted_by_recording,
        "attempted_conditions": attempts,
        "valid_crops": len(valid_cases),
        "valid_crop_yield": len(valid_cases) / attempts,
        "complete_frames_all_ten_conditions": complete_total,
        "complete_by_recording": complete_by_recording,
        "minimum_complete_frames": int(config["minimum_complete_frames"]),
        "per_condition": per_condition,
        "rejection_reason_counts": dict(sorted(rejections.items())),
        "source_window_k": {
            "minimum": min(k_values) if k_values else None,
            "maximum": max(k_values) if k_values else None,
            "median": float(np.median(k_values)) if k_values else None,
            "counts": {str(key): value for key, value in sorted(k_counts.items())},
        },
        "maximum_source_roundtrip_error_px": max_source_error,
        "source_roundtrip_limit_px": float(config["maximum_source_roundtrip_error_px"]),
        "maximum_resized_roundtrip_error_px": max_resized_error,
        "resized_roundtrip_limit_px": float(config["maximum_resized_roundtrip_error_px"]),
        "direct_window_and_interpolation_checks_passed": provenance_checks,
        "independent_support_checks_passed": support_checks,
        "deterministic_repeat_checks_passed": deterministic_checks,
        "external_output_retained": False,
        "case_manifest_sha256": manifest_hash,
        "case_manifest": case_manifest,
        "runtime_seconds": runtime,
        "runtime_limit_seconds": 60 * float(config["maximum_wall_time_minutes"]),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
    }
    (args.experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    summary = {key: value for key, value in metrics.items() if key != "case_manifest"}
    summary["case_manifest_entries"] = len(manifest_entries)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
