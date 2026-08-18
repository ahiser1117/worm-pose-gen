#!/usr/bin/env python3
"""Run EXP-0003 using only the immutable proxy-label HDF5 artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from worm_pose_gen.real_crop import (
    CropRequest,
    attempt_real_crop,
    canonical_manifest_sha256,
    half_open_support,
    support_bitmask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp_0003_real_texture_crop"
EXPECTED_RECORDINGS = ("2023-09-19-01", "2023-09-27-01", "2023-10-11-01")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_case(ax: Any, case: dict[str, Any]) -> None:
    image = case["image"]
    low, high = np.percentile(image, (1, 99))
    ax.imshow(image, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    points = case["points"]
    support = case["support"]
    ax.plot(points[:, 0], points[:, 1], color="#f4f4f5", linewidth=0.6, alpha=0.6)
    ax.scatter(points[support, 0], points[support, 1], c="#19d3ae", s=5, label="visible")
    ax.scatter(points[~support, 0], points[~support, 1], c="#ff4d6d", s=5, label="hidden")
    ax.set_title(case["title"], fontsize=7)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.axis("off")


def make_montage(
    random_cases: list[dict[str, Any]],
    maximum_cases: list[dict[str, Any]],
    rejected_cases: list[dict[str, Any]],
    path: Path,
) -> None:
    columns = 6
    rows = (("deterministic random valid", random_cases), ("40% hidden valid", maximum_cases),
            ("rejected request (full stored frame)", rejected_cases))
    figure, axes = plt.subplots(3, columns, figsize=(16, 9), squeeze=False)
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
        "EXP-0003 real-texture crop evidence (green=declared visible, pink=declared hidden)",
        fontsize=11,
    )
    figure.tight_layout(pad=0.8)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def make_yield_plot(
    per_recording_condition: dict[str, dict[str, int]], totals: dict[str, int], path: Path
) -> None:
    conditions = [f"{end}\n{int(fraction * 100)}%" for fraction in (.05, .10, .20, .30, .40)
                  for end in ("head", "tail")]
    keys = [f"{end}_{fraction:.2f}" for fraction in (.05, .10, .20, .30, .40)
            for end in ("head", "tail")]
    figure, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(keys))
    width = 0.24
    colors = ("#3b82f6", "#f59e0b", "#10b981")
    for offset, (recording, values) in enumerate(per_recording_condition.items()):
        denominator = totals[recording]
        heights = [100 * values.get(key, 0) / denominator for key in keys]
        ax.bar(x + (offset - 1) * width, heights, width, label=recording, color=colors[offset])
        for xpos, height, key in zip(x + (offset - 1) * width, heights, keys):
            ax.text(xpos, height + 1, str(values.get(key, 0)), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, conditions)
    ax.set_ylim(0, 105)
    ax.set_ylabel("valid exact-support crops (% accepted frames); labels are counts")
    ax.set_xlabel("requested hidden end and fraction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Strict camera-window contract yield by recording and condition")
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
    if proxy_path != Path("/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/proxy_v1/proxy_labels.h5"):
        raise RuntimeError("EXP-0003 proxy path differs from its frozen declaration")
    observed_hash = sha256(proxy_path)
    if observed_hash != config["proxy_sha256"]:
        raise RuntimeError("immutable proxy artifact SHA-256 mismatch")

    figures_dir = args.experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fractions = tuple(float(value) for value in config["hidden_fractions"])
    ends = tuple(config["hidden_ends"])
    per_recording_condition: dict[str, dict[str, int]] = {
        recording: defaultdict(int) for recording in EXPECTED_RECORDINGS
    }
    accepted_by_recording: dict[str, int] = {}
    complete_by_recording: dict[str, int] = {}
    rejection_reasons: Counter[str] = Counter()
    valid_cases: list[dict[str, Any]] = []
    rejected_evidence: dict[tuple[str, str, str], dict[str, Any]] = {}
    roundtrip_errors: list[float] = []
    provenance_checks = 0
    deterministic_checks = 0
    attempts = 0
    case_manifest_entries: list[dict[str, Any]] = []

    # Exactly one HDF5 handle is used. Images are read one accepted frame at a time.
    with h5py.File(proxy_path, "r") as proxy:
        if not bool(proxy.attrs.get("complete")) or int(proxy.attrs.get("schema_version", -1)) != 1:
            raise RuntimeError("proxy artifact is incomplete or has an unexpected schema")
        if tuple(proxy.keys()) != EXPECTED_RECORDINGS:
            raise RuntimeError(f"unexpected proxy recordings: {tuple(proxy.keys())}")
        for recording in EXPECTED_RECORDINGS:
            group = proxy[recording]
            accepted_positions = group["accepted_sample_position"][:]
            accepted_by_recording[recording] = len(accepted_positions)
            complete = 0
            for accepted_index, sample_position in enumerate(accepted_positions):
                image = group["accepted_image"][accepted_index]
                centerline = group["centerline_xy"][int(sample_position)].astype(np.float64)
                frame_index = int(group["sample_frame_index"][int(sample_position)])
                frame_complete = True
                for fraction in fractions:
                    for end in ends:
                        attempts += 1
                        request = CropRequest(
                            end, fraction, int(config["output_height"]), int(config["output_width"])
                        )
                        attempt = attempt_real_crop(image, centerline, request)
                        condition_key = f"{end}_{fraction:.2f}"
                        if attempt.crop is None:
                            case_manifest_entries.append({
                                "source_group": recording,
                                "source_frame_index": frame_index,
                                "accepted_image_index": accepted_index,
                                "sample_position": int(sample_position),
                                "hidden_end": end,
                                "hidden_fraction": fraction,
                                "crop_bounds_xyxy_half_open": None,
                                "source_to_crop_transform": None,
                                "scale_xy": None,
                                "requested_support_bitmask": support_bitmask(
                                    attempt.target_support
                                ),
                                "actual_support_bitmask": None,
                                "rejection_reason": attempt.rejection_reason,
                            })
                            frame_complete = False
                            rejection_reasons[str(attempt.rejection_reason)] += 1
                            evidence_key = (recording, str(attempt.rejection_reason), end)
                            if evidence_key not in rejected_evidence:
                                rejected_evidence[evidence_key] = {
                                    "image": image.copy(), "points": centerline.copy(),
                                    "support": attempt.target_support.copy(),
                                    "title": f"{recording} f{frame_index}\n{end} {fraction:.0%}: {attempt.rejection_reason}",
                                }
                            continue
                        crop = attempt.crop
                        per_recording_condition[recording][condition_key] += 1
                        x0, y0 = crop.source_origin_xy
                        case_manifest_entries.append({
                            "source_group": recording,
                            "source_frame_index": frame_index,
                            "accepted_image_index": accepted_index,
                            "sample_position": int(sample_position),
                            "hidden_end": end,
                            "hidden_fraction": fraction,
                            "crop_bounds_xyxy_half_open": [
                                x0, y0, x0 + request.output_width, y0 + request.output_height
                            ],
                            "source_to_crop_transform": [
                                [1.0, 0.0, -float(x0)],
                                [0.0, 1.0, -float(y0)],
                                [0.0, 0.0, 1.0],
                            ],
                            "scale_xy": [1.0, 1.0],
                            "requested_support_bitmask": support_bitmask(attempt.target_support),
                            "actual_support_bitmask": support_bitmask(crop.support),
                            "rejection_reason": None,
                        })
                        expected_pixels = image[
                            y0 : y0 + request.output_height, x0 : x0 + request.output_width
                        ]
                        if not np.array_equal(crop.image, expected_pixels):
                            raise RuntimeError("crop pixel provenance check failed")
                        provenance_checks += 1
                        recomputed = half_open_support(
                            centerline, crop.source_origin_xy,
                            request.output_height, request.output_width,
                        )
                        if not np.array_equal(crop.support, recomputed):
                            raise RuntimeError("stored and recomputed half-open support differ")
                        restored = crop.crop_to_source(crop.centerline_xy)
                        error = float(np.max(np.abs(restored - centerline)))
                        roundtrip_errors.append(error)
                        repeat = attempt_real_crop(image, centerline, request)
                        if repeat.crop is None or repeat.crop.source_origin_xy != crop.source_origin_xy:
                            raise RuntimeError("crop search is not deterministic")
                        if not np.array_equal(repeat.crop.image, crop.image):
                            raise RuntimeError("repeated crop pixels differ")
                        deterministic_checks += 1
                        valid_cases.append({
                            "image": crop.image, "points": crop.centerline_xy,
                            "support": crop.support,
                            "recording": recording, "frame_index": frame_index,
                            "fraction": fraction, "end": end,
                            "title": f"{recording} f{frame_index}\n{end} {fraction:.0%} @ ({x0},{y0})",
                        })
                complete += int(frame_complete)
            complete_by_recording[recording] = complete

    maximum_error = max(roundtrip_errors, default=math.nan)
    if roundtrip_errors and maximum_error > float(config["maximum_roundtrip_error_px"]):
        raise RuntimeError("round-trip tolerance exceeded")
    accepted_total = sum(accepted_by_recording.values())
    complete_total = sum(complete_by_recording.values())
    emitted_total = len(valid_cases)
    success = complete_total >= int(config["minimum_complete_frames"])

    rng = np.random.default_rng(int(config["seed"]))
    random_selection = [valid_cases[index] for index in
                        rng.choice(len(valid_cases), size=min(6, len(valid_cases)), replace=False)]
    maximum_pool = [case for case in valid_cases if case["fraction"] == max(fractions)]
    maximum_selection = maximum_pool[:6]
    # Favor both rejection modes and multiple recordings in the evidence row.
    rejected_selection: list[dict[str, Any]] = []
    rejected_cases = list(rejected_evidence.values())
    seen: set[tuple[str, str]] = set()
    selected_ids: set[int] = set()
    for case in rejected_cases:
        key = (case["title"].split()[0], case["title"].split(":")[-1])
        if key not in seen:
            rejected_selection.append(case)
            selected_ids.add(id(case))
            seen.add(key)
        if len(rejected_selection) == 6:
            break
    for case in rejected_cases:
        if len(rejected_selection) == 6:
            break
        if id(case) not in selected_ids:
            rejected_selection.append(case)
            selected_ids.add(id(case))
    make_montage(
        random_selection, maximum_selection, rejected_selection,
        figures_dir / "real_texture_crop_evidence.png",
    )
    make_yield_plot(
        {key: dict(value) for key, value in per_recording_condition.items()},
        accepted_by_recording,
        figures_dir / "contract_yield.png",
    )

    per_condition = {}
    for fraction in fractions:
        for end in ends:
            key = f"{end}_{fraction:.2f}"
            counts = {recording: per_recording_condition[recording].get(key, 0)
                      for recording in EXPECTED_RECORDINGS}
            per_condition[key] = {"valid": sum(counts.values()), "attempted": accepted_total,
                                  "by_recording": counts}
    elapsed = time.perf_counter() - started
    case_manifest: dict[str, Any] = {
        "schema_version": 1,
        "proxy_sha256": observed_hash,
        "evidence_label": (
            "candidate proxy-referenced crop geometry; not anatomical accuracy evidence"
        ),
        "coordinate_convention": (
            "xy pixel centers; integer half-open bounds; source_to_crop is translation only"
        ),
        "entries": case_manifest_entries,
    }
    manifest_sha256 = canonical_manifest_sha256(case_manifest)
    metrics = {
        "experiment": config["experiment"],
        "status": "ACCEPT" if success else "REJECT",
        "evidence_tier": (
            "Candidate proxy-referenced real-texture crops; not anatomical accuracy evidence"
        ),
        "proxy_path": str(proxy_path),
        "proxy_sha256": observed_hash,
        "accepted_proxy_frames": accepted_total,
        "accepted_by_recording": accepted_by_recording,
        "attempted_conditions": attempts,
        "valid_crops": emitted_total,
        "valid_crop_yield": emitted_total / attempts,
        "complete_frames_all_ten_conditions": complete_total,
        "complete_by_recording": complete_by_recording,
        "minimum_complete_frames": int(config["minimum_complete_frames"]),
        "per_condition": per_condition,
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "maximum_roundtrip_error_px": maximum_error,
        "roundtrip_limit_px": float(config["maximum_roundtrip_error_px"]),
        "pixel_provenance_checks_passed": provenance_checks,
        "deterministic_repeat_checks_passed": deterministic_checks,
        "external_output_retained": False,
        "case_manifest_sha256": manifest_sha256,
        "case_manifest": case_manifest,
        "runtime_seconds": elapsed,
        "runtime_limit_seconds": 60 * float(config["maximum_wall_time_minutes"]),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
    }
    (args.experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    stdout_summary = {key: value for key, value in metrics.items() if key != "case_manifest"}
    stdout_summary["case_manifest_entries"] = len(case_manifest_entries)
    print(json.dumps(stdout_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
