"""Compare width models of the tube fitter on the 30-frame stress set.

Step 2 of ``docs/POSE_PIPELINE_PLAN.md``: the symmetric width template is
multiplied by a smooth log-space correction so the head and tail may taper
differently.  This script fits the 27 worm frames of the annotation-free
30-frame set with the ``reference`` schedule of ``batch_fit`` for several
width-model variants and reports each against the stored single-frame
reference run (``docs/mask_fit_unannotated30/metrics.json``), whose width
template it reuses.  Per frame it records IoU, body length, width scale, the
correction coefficients, and the taper asymmetry that labels the tail.

Outputs (default ``docs/pose_pipeline_step2/``): ``width_model_sweep.json``
and ``width_model_worst_frames.jpg`` (the frames the fixed template hurt most,
symmetric versus asymmetric fits with their width profiles).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any

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
    standard_initializations,
    taper_asymmetry,
)
from worm_pose_gen.run_records import git_revision, utc_now  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "pose_pipeline_step2"
REFERENCE_DIR = PROJECT_ROOT / "docs" / "mask_fit_unannotated30"
# Frames the symmetric template hurt most in the stored reference run.
WORST_SAMPLES = (27, 2, 14)

VARIANTS: dict[str, dict[str, Any]] = {
    "symmetric": {"width_coefficients": 0},
    "asym6_prior0": {"width_coefficients": 6, "width_shape_prior": 0.0},
    "asym6_prior1e-3": {"width_coefficients": 6, "width_shape_prior": 1e-3},
    "asym6_prior1e-2": {"width_coefficients": 6, "width_shape_prior": 1e-2},
    "asym8_prior1e-3": {"width_coefficients": 8, "width_shape_prior": 1e-3},
    "asym10_prior1e-3": {"width_coefficients": 10, "width_shape_prior": 1e-3},
    "asym12_prior1e-3": {"width_coefficients": 12, "width_shape_prior": 1e-3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", action="append", type=Path, dest="recordings")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--frozen-dir", type=Path, default=reference_run.DEFAULT_FROZEN_DIR)
    parser.add_argument(
        "--variant", action="append", choices=tuple(VARIANTS), dest="variants",
        help="default: all; a subset is merged into an existing width_model_sweep.json",
    )
    parser.add_argument("--preset", default="reference", choices=tuple(PRESETS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--flat-field-frames", type=int, default=reference_run.FLAT_FIELD_SAMPLE_COUNT)
    return parser.parse_args()


def build_targets(args: argparse.Namespace) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Filled classical masks, corrected frames, and frozen reference curves by sample index."""

    recordings = list(args.recordings or DEFAULT_RECORDINGS)
    if len(recordings) != 3:
        raise SystemExit("exactly three --recording arguments are required")
    cases, _ = recording_records(recordings)
    frames, fields = reference_run.read_frames(cases, flat_field_frames=args.flat_field_frames)
    _, frozen_curves = reference_run.load_frozen(args.frozen_dir)
    classical = ClassicalConfig()
    targets: dict[int, np.ndarray] = {}
    corrected_frames: dict[int, np.ndarray] = {}
    for case in cases:
        index = int(case["sample_index"])
        field = fields[str(case["recording"])]
        corrected = frames[index].astype(np.float64) if field is None else apply_flat_field(frames[index], field, clip=(0.0, 255.0))
        segmentation = segment_dark_ridge(corrected, classical)
        target, _ = fill_narrow_holes(segmentation.component, reference_run.HOLE_FILL_RADIUS_PX, device=args.device)
        targets[index] = target
        corrected_frames[index] = corrected
    return targets, corrected_frames, frozen_curves


def summarize(rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    ious = np.array([r["iou"] for r in rows])
    deltas = np.array([r["delta_vs_reference"] for r in rows])
    asymmetry = np.array([r["taper_asymmetry"] for r in rows])
    return {
        "frames": len(rows),
        "seconds_per_frame": seconds / len(rows),
        "iou_median": float(np.median(ious)),
        "iou_mean": float(ious.mean()),
        "iou_min": float(ious.min()),
        "frames_iou_at_least_0.8": int((ious >= 0.8).sum()),
        "frames_iou_at_least_0.9": int((ious >= 0.9).sum()),
        "delta_median": float(np.median(deltas)),
        "delta_mean": float(deltas.mean()),
        "delta_min": float(deltas.min()),
        "delta_max": float(deltas.max()),
        "frames_better_by_0.01": int((deltas > 0.01).sum()),
        "frames_worse_by_0.01": int((deltas < -0.01).sum()),
        "abs_taper_asymmetry_median": float(np.median(np.abs(asymmetry))),
        "frames_reversed_to_put_tail_last": int(sum(r["reversed"] for r in rows)),
    }


def plot_worst(
    output: Path,
    samples: tuple[int, ...],
    targets: dict[int, np.ndarray],
    frames: dict[int, np.ndarray],
    fits: dict[str, dict[int, MaskFitResult]],
    template: np.ndarray,
    symmetric: str,
    asymmetric: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4.2 * len(samples)), gridspec_kw={"width_ratios": [1.3, 1]})
    axes = np.atleast_2d(axes)
    for row, sample in enumerate(samples):
        target = targets[sample]
        pad = 40
        yy, xx = np.nonzero(target)
        y0, y1 = max(0, yy.min() - pad), min(target.shape[0], yy.max() + pad)
        x0, x1 = max(0, xx.min() - pad), min(target.shape[1], xx.max() + pad)
        ax = axes[row, 0]
        ax.imshow(frames[sample][y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        ax.contour(target[y0:y1, x0:x1].astype(float), levels=[0.5], colors=["#4aa3ff"], linewidths=1.0)
        for name, color in ((symmetric, "#57d68d"), (asymmetric, "#ff50a5")):
            result = fits[name][sample]
            full = np.zeros(target.shape, dtype=float)
            full[result.crop.y0 : result.crop.y1, result.crop.x0 : result.crop.x1] = result.rendered_hard_mask
            ax.contour(full[y0:y1, x0:x1], levels=[0.5], colors=[color], linewidths=1.2)
            iou = result.records[result.best_index]["final_iou"]
            ax.plot([], [], color=color, label=f"{name}: IoU {iou:.3f}")
        tail = fits[asymmetric][sample].centerline_xy[-1]
        head = fits[asymmetric][sample].centerline_xy[0]
        ax.plot(tail[0] - x0, tail[1] - y0, "o", color="#ffd23c", ms=7, mfc="none", label="tail (asymmetric fit)")
        ax.plot(head[0] - x0, head[1] - y0, "s", color="#ffd23c", ms=7, mfc="none", label="head")
        ax.set_title(f"sample {sample:02d}: mask (blue) with symmetric and asymmetric fits")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_axis_off()
        ax = axes[row, 1]
        position = np.linspace(0.0, 1.0, len(template))
        for name, color in ((symmetric, "#57d68d"), (asymmetric, "#ff50a5")):
            result = fits[name][sample]
            ax.plot(position, result.width_profile, color=color, label=f"{name} (scale {result.width_px:.1f} px)")
        ax.set_xlabel("body position (head 0, tail 1 after orientation)")
        ax.set_ylabel("full width, px")
        ax.set_title(f"taper asymmetry {taper_asymmetry(fits[asymmetric][sample].width_profile):+.2f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=100, pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    reference = {int(c["sample_index"]): c for c in json.loads((args.reference_dir / "metrics.json").read_text())["per_case"]}
    template = np.load(args.reference_dir / "predictions.npz")["width_template"]
    targets, frames, frozen_curves = build_targets(args)
    worm = [index for index in sorted(targets) if not reference[index]["expected_no_worm"]]
    reference_median = float(np.median([reference[i]["final_iou_target"] for i in worm]))
    print(f"{len(worm)} worm frames; stored reference median IoU {reference_median:.4f}", flush=True)

    base: BatchFitConfig = PRESETS[args.preset]
    output: dict[str, Any] = {
        "generated_at": utc_now(),
        "git": git_revision(PROJECT_ROOT),
        "preset": args.preset,
        "reference_run": str(args.reference_dir.relative_to(PROJECT_ROOT)),
        "reference_median_iou": reference_median,
        "variants": {},
    }
    fits: dict[str, dict[int, MaskFitResult]] = {}
    for name in args.variants or list(VARIANTS):
        config = replace(base, **VARIANTS[name])
        masks = [targets[i] for i in worm]
        starts = []
        for index, mask in zip(worm, masks, strict=True):
            curve = frozen_curves.get(index) if reference[index]["frozen_group"] == "frozen_accepted" else None
            starts.append(standard_initializations(mask, reference_centerline_xy=curve, config=config))
        fit_masks(masks[:2], starts[:2], width_template=template, config=config, device=device)  # compile warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        results = fit_masks(masks, starts, width_template=template, config=config, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        rows = []
        fits[name] = {}
        for index, result, mask in zip(worm, results, masks, strict=True):
            oriented, reversed_ = orient_tail_last(result, config=config)
            fits[name][index] = oriented
            iou = hard_iou(oriented.rendered_hard_mask, mask[oriented.crop.y0 : oriented.crop.y1, oriented.crop.x0 : oriented.crop.x1])
            stored = reference[index]
            rows.append(
                {
                    "sample_index": index,
                    "frozen_group": stored["frozen_group"],
                    "reference_iou": stored["final_iou_target"],
                    "iou": iou,
                    "delta_vs_reference": iou - stored["final_iou_target"],
                    "energy": oriented.records[oriented.best_index]["final_soft_dice_energy"],
                    "body_length_px": oriented.body_length_px,
                    "reference_body_length_px": stored["body_length_px"],
                    "width_px": oriented.width_px,
                    "width_shape": [float(v) for v in oriented.width_shape],
                    "taper_asymmetry": taper_asymmetry(oriented.width_profile),
                    "reversed": bool(reversed_),
                    "points_in_fov": oriented.points_in_fov,
                    "best_start": oriented.initializations[oriented.best_index].name,
                    "reference_best_start": stored["best_start"],
                }
            )
        summary = summarize(rows, seconds)
        output["variants"][name] = {"config": {k: v for k, v in asdict(config).items() if k in VARIANTS[name]}, "summary": summary, "per_frame": rows}
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True)
        for row in sorted(rows, key=lambda r: r["delta_vs_reference"])[:3]:
            print(f"   worst delta: sample {row['sample_index']:02d} {row['delta_vs_reference']:+.3f} (iou {row['iou']:.3f})", flush=True)
    sweep_path = args.output_dir / "width_model_sweep.json"
    if sweep_path.exists() and args.variants:
        # A partial run adds its variants to the stored sweep instead of replacing it.
        stored = json.loads(sweep_path.read_text())
        stored["variants"].update(output["variants"])
        stored["generated_at"] = output["generated_at"]
        stored["git"] = output["git"]
        output = stored
    sweep_path.write_text(json.dumps(output, indent=1))
    if "symmetric" in fits:
        asymmetric = next((n for n in ("asym8_prior1e-3", "asym6_prior1e-3", "asym6_prior0", "asym6_prior1e-2") if n in fits), None)
        if asymmetric is not None:
            plot_worst(args.output_dir / "width_model_worst_frames.jpg", WORST_SAMPLES, targets, frames, fits, template, "symmetric", asymmetric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
