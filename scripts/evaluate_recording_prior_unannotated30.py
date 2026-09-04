"""Recording priors on the 30-frame stress set (plan step 3).

For each of the three recordings a prior is bootstrapped from a spread
sample of frames segmented the classical way (as the stress set is), with
the hard bounds opened wide.  The 27 worm frames are then fit with the
``reference`` schedule under (a) the step 2 model with hard bounds and (b)
the recording prior, which removes the bounds, centers the width correction
on the recording's profile, and tries every start in both orientations.
Per frame the run records IoU, body length, in-view fraction, taper
asymmetry, and the orientation energy gap.

Outputs (default ``docs/pose_pipeline_step3/``): ``prior_sweep.json`` with
the priors and per-frame rows, and ``priors/<recording>.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate_mask_fit_unannotated30 as reference_run  # noqa: E402
from evaluate_final_geometry_unannotated30 import DEFAULT_RECORDINGS, recording_records  # noqa: E402
from worm_pose_gen.batch_fit import PRESETS, BatchFitConfig, fit_masks  # noqa: E402
from worm_pose_gen.classical import ClassicalConfig, segment_dark_ridge  # noqa: E402
from worm_pose_gen.flat_field import apply_flat_field  # noqa: E402
from worm_pose_gen.mask_fit import (  # noqa: E402
    MaskFitResult,
    fill_narrow_holes,
    hard_iou,
    orient_tail_last,
    orientation_pair,
    standard_initializations,
    taper_asymmetry,
)
from worm_pose_gen.recording_prior import RecordingPrior, bootstrap_prior_from_masks  # noqa: E402
from worm_pose_gen.run_records import git_revision, utc_now  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "pose_pipeline_step3"
REFERENCE_DIR = PROJECT_ROOT / "docs" / "mask_fit_unannotated30"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", action="append", type=Path, dest="recordings")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--frozen-dir", type=Path, default=reference_run.DEFAULT_FROZEN_DIR)
    parser.add_argument("--preset", default="reference", choices=tuple(PRESETS))
    parser.add_argument("--bootstrap-frames", type=int, default=48)
    parser.add_argument("--bootstrap-preset", default="balanced", choices=tuple(PRESETS))
    parser.add_argument("--shape-weight", action="append", type=float, dest="shape_weights", help="width-profile prior weights to try (default 0.02 and 0.05)")
    parser.add_argument("--skip-bounds", action="store_true", help="do not refit the hard-bounds baseline")
    parser.add_argument("--device", default=None)
    parser.add_argument("--flat-field-frames", type=int, default=reference_run.FLAT_FIELD_SAMPLE_COUNT)
    return parser.parse_args()


def classical_target(frame: np.ndarray, field, classical: ClassicalConfig, device) -> np.ndarray:
    corrected = frame.astype(np.float64) if field is None else apply_flat_field(frame, field, clip=(0.0, 255.0))
    segmentation = segment_dark_ridge(corrected, classical)
    target, _ = fill_narrow_holes(segmentation.component, reference_run.HOLE_FILL_RADIUS_PX, device=device)
    return target


def summarize(rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    ious = np.array([r["iou"] for r in rows])
    deltas = np.array([r["delta_vs_reference"] for r in rows])
    lengths = np.array([r["body_length_px"] for r in rows])
    gaps = np.array([r["orientation_gap"] for r in rows], dtype=float)
    out = {
        "frames": len(rows),
        "seconds_per_frame": seconds / len(rows),
        "iou_median": float(np.median(ious)),
        "iou_mean": float(ious.mean()),
        "iou_min": float(ious.min()),
        "frames_iou_at_least_0.9": int((ious >= 0.9).sum()),
        "delta_median": float(np.median(deltas)),
        "frames_better_by_0.01": int((deltas > 0.01).sum()),
        "frames_worse_by_0.01": int((deltas < -0.01).sum()),
        "length_px_p10_p50_p90": [float(v) for v in np.percentile(lengths, [10, 50, 90])],
        "frames_at_length_bound_750": int((lengths >= 742).sum()),
        "abs_taper_asymmetry_median": float(np.median(np.abs([r["taper_asymmetry"] for r in rows]))),
    }
    if np.isfinite(gaps).any():
        finite = gaps[np.isfinite(gaps)]
        out["orientation_gap_median"] = float(np.median(finite))
        out["frames_gap_below_0.002"] = int((finite < 0.002).sum())
        out["frames_gap_below_0.01"] = int((finite < 0.01).sum())
        out["reversed_start_won"] = int(sum(str(r["best_start"]).endswith("_reversed") for r in rows))
    return out


def orientation_gap(result: MaskFitResult) -> float:
    reversed_names = {str(r["name"]) for r in result.records if str(r["name"]).endswith("_reversed")}
    forward = [float(r["final_energy"]) for r in result.records if str(r["name"]) not in reversed_names]
    reverse = [float(r["final_energy"]) for r in result.records if str(r["name"]) in reversed_names]
    if not forward or not reverse:
        return float("nan")
    winner_reversed = str(result.initializations[result.best_index].name) in reversed_names
    best = float(result.records[result.best_index]["final_energy"])
    return (min(forward) if winner_reversed else min(reverse)) - best


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "priors").mkdir(exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    recordings = list(args.recordings or DEFAULT_RECORDINGS)
    if len(recordings) != 3:
        raise SystemExit("exactly three --recording arguments are required")
    cases, _ = recording_records(recordings)
    frames, fields = reference_run.read_frames(cases, flat_field_frames=args.flat_field_frames)
    _, frozen_curves = reference_run.load_frozen(args.frozen_dir)
    reference = {int(c["sample_index"]): c for c in json.loads((args.reference_dir / "metrics.json").read_text())["per_case"]}
    template = np.load(args.reference_dir / "predictions.npz")["width_template"]
    classical = ClassicalConfig()
    targets: dict[int, np.ndarray] = {}
    recording_of: dict[int, str] = {}
    for case in cases:
        index = int(case["sample_index"])
        recording_of[index] = str(case["recording"])
        targets[index] = classical_target(frames[index], fields[str(case["recording"])], classical, device)
    worm = [i for i in sorted(targets) if not reference[i]["expected_no_worm"]]
    base: BatchFitConfig = PRESETS[args.preset]

    # One prior per recording from a spread sample of classically segmented frames.
    priors: dict[str, RecordingPrior] = {}
    bootstrap_info: dict[str, Any] = {}
    boot_config = replace(PRESETS[args.bootstrap_preset], width_coefficients=base.width_coefficients)
    by_recording: dict[str, dict[str, Any]] = {}
    for case in cases:
        by_recording.setdefault(str(case["recording"]), case)
    for recording, case in by_recording.items():
        started = time.perf_counter()
        masks = []
        with h5py.File(case["resolved_source_path"], "r") as handle:
            dataset = handle[case["source_dataset_path"]]
            total = int(dataset.shape[0])
            for index in sorted(set(int(v) for v in np.linspace(0, total - 1, min(args.bootstrap_frames, total)))):
                target = classical_target(np.asarray(dataset[index], dtype=np.uint8), fields[recording], classical, device)
                if target.sum() >= 500:
                    masks.append(target)
        prior, results, used = bootstrap_prior_from_masks(masks, config=boot_config, device=device, recording=recording, min_frames=6)
        prior = replace(prior, source=f"bootstrap of {len(used)} classically segmented frames, preset {args.bootstrap_preset}")
        priors[recording] = prior
        prior.save(args.output_dir / "priors" / f"{Path(recording).stem}.json")
        lengths = [r.body_length_px for r in results]
        bootstrap_info[recording] = {
            "frames_fit": len(used), "frames_used": prior.frames_used, "selection": prior.selection,
            "fit_length_px_p10_p50_p90": [float(v) for v in np.percentile(lengths, [10, 50, 90])],
            "seconds": time.perf_counter() - started,
        }
        print(f"{Path(recording).stem}: length {prior.length_px:.0f} px (log sigma {prior.log_length_sigma:.3f}), width {prior.width_px:.1f},"
              f" shape {[round(v, 2) for v in prior.width_shape]}, {prior.frames_used}/{len(used)} used", flush=True)

    variants: dict[str, tuple[BatchFitConfig | None, float | None]] = {}
    if not args.skip_bounds:
        variants["bounds"] = (base, None)
    for weight in args.shape_weights or (0.02, 0.05):
        variants[f"prior_shape{weight:g}"] = (base, weight)

    output: dict[str, Any] = {
        "generated_at": utc_now(), "git": git_revision(PROJECT_ROOT), "preset": args.preset,
        "bootstrap": bootstrap_info, "priors": {k: v.to_dict() for k, v in priors.items()}, "variants": {},
    }
    for name, (config, weight) in variants.items():
        masks = [targets[i] for i in worm]
        configs = []
        starts = []
        for index, mask in zip(worm, masks, strict=True):
            frame_config = config if weight is None else priors[recording_of[index]].apply(config, shape_weight=weight)
            configs.append(frame_config)
            curve = frozen_curves.get(index) if reference[index]["frozen_group"] == "frozen_accepted" else None
            frame_starts = standard_initializations(mask, reference_centerline_xy=curve, config=frame_config)
            if weight is not None:
                shape = np.asarray(priors[recording_of[index]].width_shape)
                frame_starts = [replace(s, width_shape=shape) for s in frame_starts]
                skeleton = next((s for s in frame_starts if s.name == "skeleton_longest_path"), None)
                if skeleton is not None:
                    frame_starts.append(orientation_pair(skeleton, config=frame_config)[1])
            starts.append(frame_starts)
        # Priors differ per recording, so fit recording by recording.
        results: dict[int, MaskFitResult] = {}
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for recording in by_recording:
            members = [k for k, index in enumerate(worm) if recording_of[index] == recording]
            fitted = fit_masks([masks[k] for k in members], [starts[k] for k in members], width_template=template, config=configs[members[0]], device=device)
            for k, result in zip(members, fitted, strict=True):
                results[worm[k]] = result
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        rows = []
        for index, mask in zip(worm, masks, strict=True):
            result = results[index]
            gap = orientation_gap(result)
            if weight is None:
                result, _ = orient_tail_last(result, config=base)
            iou = hard_iou(result.rendered_hard_mask, mask[result.crop.y0 : result.crop.y1, result.crop.x0 : result.crop.x1])
            stored = reference[index]
            rows.append({
                "sample_index": index, "recording": Path(recording_of[index]).stem, "frozen_group": stored["frozen_group"],
                "reference_iou": stored["final_iou_target"], "iou": iou, "delta_vs_reference": iou - stored["final_iou_target"],
                "energy": result.records[result.best_index]["final_soft_dice_energy"],
                "total_energy": result.records[result.best_index]["final_energy"],
                "body_length_px": result.body_length_px, "width_px": result.width_px,
                "width_shape": [float(v) for v in result.width_shape],
                "taper_asymmetry": taper_asymmetry(result.width_profile), "orientation_gap": gap,
                "points_in_fov": result.points_in_fov, "best_start": result.initializations[result.best_index].name,
            })
        summary = summarize(rows, seconds)
        output["variants"][name] = {"shape_weight": weight, "summary": summary, "per_frame": rows}
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True)
    (args.output_dir / "prior_sweep.json").write_text(json.dumps(output, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
