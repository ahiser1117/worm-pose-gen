#!/usr/bin/env python3
"""Segment a stretch of a recording and fit the body model to every frame.

Frames are read from the HDF5 recording in slabs, flat-fielded with the
per-recording correction the labeling app uses, pushed through the promoted
segmenter, cleaned (probability at or above ``--threshold``, narrow holes
filled, largest component kept), and then fit in GPU batches with
``worm_pose_gen.batch_fit.fit_masks``.  Per frame the run records the latent,
width scale and profile, centerline, body length, in-view fraction, final
energy and overlap, mask statistics, and per-stage timing.

Outputs land in one directory per run: ``summary.json`` (aggregates and
timing), ``poses.npz`` (per-frame arrays), with ``--video`` an MP4 overlay
of the fitted tube outline and centerline on the flat-fielded frame, and
residual images (mask the tube misses in blue, tube outside the mask in red)
for the ``--residual-frames`` worst frames plus any ``--dump-frames``.

By default a recording prior is bootstrapped first (``--prior bootstrap``):
``--bootstrap-frames`` frames spread over the whole recording are fit with
the bounds opened wide, and robust medians of body length, width scale, and
width profile replace the hard bounds with Gaussian priors
(``recording_prior.json`` in the run directory, cached under
``--prior-cache``).  Under that asymmetric prior every frame is started in
both orientations and the energy gap between them is stored as the
orientation confidence.
``scripts/render_pose_run.py`` produces the same video and residual images
for a stored run without refitting.

Example (one minute at 20 fps of an unseen recording, with video):

    scripts/project_env.sh uv run --no-sync --frozen python scripts/fit_recording.py \\
        --recording /store1/shared/all_data_raw/prj_aversion/2024-05-28/2024-05-28-02.h5 \\
        --start 0 --frames 1200 --video
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import torch

from worm_pose_gen.ambiguity import compute_ambiguity, summarize_ambiguity
from worm_pose_gen.batch_fit import PRESETS, BatchFitConfig, fit_masks
from worm_pose_gen.flat_field import apply_flat_field
from worm_pose_gen.label_app import DATASET_PATH, RecordingSource
from worm_pose_gen.mask_fit import (
    Initialization,
    MaskFitResult,
    default_width_template,
    extend_start_to_length,
    orient_tail_last,
    orientation_pair,
    standard_initializations,
    taper_asymmetry,
)
from worm_pose_gen.recording_prior import RecordingPrior, bootstrap_prior_from_masks
from worm_pose_gen.pose_run import clean_mask, draw_overlay, draw_residual, render_tube, residual_caption, residual_rows
from worm_pose_gen.run_records import checkpoint_fingerprint, git_revision, timestamp_slug, utc_now
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT
from worm_pose_gen.segmenter import load_segmenter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "best.ckpt"
EXTERNAL_ROOT = Path(os.environ.get("WORM_POSE_EXTERNAL_ROOT", "/temp_data4/alex/external_artifacts"))
DEFAULT_OUTPUT_DIR = (EXTERNAL_ROOT / "poses") if EXTERNAL_ROOT.exists() else PROJECT_ROOT / "checkpoints" / "poses"
HOLE_FILL_RADIUS_PX = 8
MIN_WORM_PIXELS = 500
STAGES = ("read", "flat_field", "network", "cleanup", "init", "fit", "video")
START_SETS = {
    "skeleton": ("skeleton_longest_path",),
    "skeleton+straight": ("skeleton_longest_path", "moments_straight"),
    "skeleton+reversed": ("skeleton_longest_path", "skeleton_longest_path_reversed"),
    "all": None,
}
DEFAULT_PRIOR_CACHE = EXTERNAL_ROOT / "recording_priors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--frames", type=int, default=1200, help="number of frames to cover (1200 = one minute at 20 fps)")
    parser.add_argument("--step", type=int, default=1, help="fit every k-th frame of the covered range")
    parser.add_argument("--slab", type=int, default=64, help="frames read from disk and fit together")
    parser.add_argument("--batch-size", type=int, default=16, help="frames per segmenter forward pass")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hole-radius", type=int, default=HOLE_FILL_RADIUS_PX, help="largest hole width to fill, in pixels")
    parser.add_argument("--min-worm-pixels", type=int, default=MIN_WORM_PIXELS, help="smaller cleaned masks are not fit")
    parser.add_argument("--init-workers", type=int, default=min(8, os.cpu_count() or 1), help="processes for skeleton/moment starts (0 = inline)")
    parser.add_argument(
        "--preset", default="fast", choices=tuple(PRESETS),
        help="fitting schedule: fast (0.25 s/frame, -0.008 IoU vs reference), balanced (0.46 s, -0.006), reference (4.9 s, exact)",
    )
    parser.add_argument("--fine-stride", type=int, default=None, choices=(1, 2), help="centerline point stride when rendering the finest stage")
    parser.add_argument("--padding", type=int, default=None, help="crop padding around the mask, in pixels (preset default: 32, reference 64)")
    parser.add_argument(
        "--starts", default=None, choices=tuple(START_SETS),
        help="starting states per frame (default skeleton+straight; the reference preset uses all). The skeleton start wins on nearly every frame",
    )
    parser.add_argument("--no-compile", action="store_true", help="render eagerly instead of through torch.compile")
    parser.add_argument(
        "--width-coefficients", type=int, default=None,
        help="cubic B-spline coefficients of the log-space width correction (0 = symmetric template; preset default 6)",
    )
    parser.add_argument("--width-prior", type=float, default=None, help="Gaussian prior weight pulling the width correction toward zero")
    parser.add_argument("--no-orient", action="store_true", help="keep the fitted orientation instead of placing the thinner (tail) end last")
    parser.add_argument(
        "--prior", default="bootstrap", choices=("bootstrap", "none"),
        help="bootstrap a recording prior (length, width, width profile) and fit under it, or fit with the hard bounds",
    )
    parser.add_argument("--prior-file", type=Path, default=None, help="use this recording_prior.json instead of bootstrapping")
    parser.add_argument("--bootstrap-frames", type=int, default=64, help="frames spread over the recording for the bootstrap pass")
    parser.add_argument("--bootstrap-target", type=int, default=12, help="whole-worm fits wanted; the sample is enlarged up to 4x to reach it")
    parser.add_argument("--bootstrap-preset", default="balanced", choices=tuple(PRESETS))
    parser.add_argument("--prior-cache", type=Path, default=DEFAULT_PRIOR_CACHE, help="directory of cached priors, one per recording and coefficient count")
    parser.add_argument("--no-prior-cache", action="store_true", help="neither read nor write the prior cache")
    parser.add_argument("--rebootstrap", action="store_true", help="ignore a cached prior and bootstrap again")
    parser.add_argument("--prior-shape-weight", type=float, default=0.01, help="weight of the width-profile prior once a recording prior is active")
    parser.add_argument("--row-pixel-budget", type=int, default=BatchFitConfig.row_pixel_budget)
    parser.add_argument("--video", action="store_true", help="write an overlay MP4")
    parser.add_argument("--residual-frames", type=int, default=5, help="write residual images for this many lowest-IoU frames")
    parser.add_argument("--dump-frames", default="", help="comma-separated frame indices that also get residual images")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--scale", type=float, default=1.0, help="resize factor for the video")
    parser.add_argument("--quality", type=int, default=5, help="imageio/ffmpeg quality, 0 worst to 10 best")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="where the flat field cache lives")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default=None, help="run name suffix (default: recording stem and frame range)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BatchFitConfig:
    config = PRESETS[args.preset]
    overrides: dict[str, Any] = {"compile_renderer": not args.no_compile, "row_pixel_budget": args.row_pixel_budget}
    if args.padding is not None:
        overrides["crop_padding"] = args.padding
    if args.fine_stride is not None:
        overrides["stage_point_stride"] = config.stage_point_stride[:-1] + (args.fine_stride,)
    if args.width_coefficients is not None:
        overrides["width_coefficients"] = args.width_coefficients
    if args.width_prior is not None:
        overrides["width_shape_prior"] = args.width_prior
    return replace(config, **overrides)


def initializations_for(
    mask: np.ndarray,
    config: BatchFitConfig,
    names: tuple[str, ...] | None = None,
    width_shape: np.ndarray | None = None,
    target_length_px: float | None = None,
) -> list[Initialization]:
    """Standard starts, restricted to ``names`` when given (falling back to whatever exists).

    ``skeleton_longest_path_reversed`` adds the skeleton start traversed from
    the other end; ``width_shape`` (the prior's profile) is given to every
    start; with ``target_length_px`` a start whose end touches the image
    border is lengthened off camera to that length before fitting.
    """

    starts = standard_initializations(mask, config=config)
    if names is None:
        chosen = starts
    else:
        chosen = [s for s in starts if s.name in names] or starts[:1]
        if target_length_px is not None:
            chosen = [extend_start_to_length(s, mask, target_length_px, config=config) for s in chosen]
        if "skeleton_longest_path_reversed" in names:
            skeleton = next((s for s in chosen if s.name == "skeleton_longest_path"), None)
            if skeleton is not None:
                chosen = chosen + [orientation_pair(skeleton, config=config)[1]]
    if width_shape is not None:
        chosen = [replace(s, width_shape=np.asarray(width_shape, dtype=np.float64)) for s in chosen]
    return chosen


def orientation_gap(result: MaskFitResult) -> float:
    """Energy of the best start of the other orientation minus the winner's; NaN without both orientations."""

    reversed_names = {str(r["name"]) for r in result.records if str(r["name"]).endswith("_reversed")}
    forward = [float(r["final_energy"]) for r in result.records if str(r["name"]) not in reversed_names]
    reverse = [float(r["final_energy"]) for r in result.records if str(r["name"]) in reversed_names]
    if not forward or not reverse:
        return float("nan")
    winner_reversed = str(result.initializations[result.best_index].name) in reversed_names
    best = float(result.records[result.best_index]["final_energy"])
    return (min(forward) if winner_reversed else min(reverse)) - best


def flat_fielded(raw: np.ndarray, field) -> np.ndarray:
    return np.clip(np.rint(apply_flat_field(raw, field, clip=(0.0, 255.0))), 0, 255).astype(np.uint8)


def bootstrap_prior(dataset, total: int, field, module, args: argparse.Namespace, config: BatchFitConfig, device: torch.device) -> tuple[RecordingPrior, dict[str, Any]]:
    """Segment frames spread over the whole recording and estimate its body-size prior."""

    boot_config = replace(
        PRESETS[args.bootstrap_preset],
        compile_renderer=not args.no_compile,
        row_pixel_budget=args.row_pixel_budget,
        width_coefficients=config.width_coefficients,
        width_shape_prior=config.width_shape_prior,
    )
    seen: set[int] = set()
    masks: list[np.ndarray] = []
    prior = results = used = None
    # Whole worms (mask clear of the border) may be rare; enlarge the sample until enough are found.
    for factor in (1, 2, 4):
        count = min(args.bootstrap_frames * factor, total)
        indices = [i for i in sorted(set(int(v) for v in np.linspace(0, total - 1, count))) if i not in seen]
        seen.update(indices)
        for chunk_start in range(0, len(indices), args.batch_size):
            chunk = indices[chunk_start : chunk_start + args.batch_size]
            corrected = np.stack([flat_fielded(np.asarray(dataset[i], dtype=np.uint8), field) for i in chunk])
            probability = module.predict_probability_batch(corrected, batch_size=args.batch_size)
            for prob in probability:
                mask, stats = clean_mask(prob, args.threshold, args.hole_radius, device)
                if stats["worm_pixels"] >= args.min_worm_pixels:
                    masks.append(mask)
        try:
            prior, results, used = bootstrap_prior_from_masks(masks, config=boot_config, device=device, recording=str(args.recording))
        except ValueError as error:
            if factor == 4:
                raise
            print(f"bootstrap: {error}; enlarging the sample", flush=True)
            continue
        if prior.frames_used >= args.bootstrap_target or count >= total:
            break
        print(f"bootstrap: {prior.frames_used} whole worms among {len(used)} fits; enlarging the sample", flush=True)
    assert prior is not None and results is not None and used is not None
    prior = replace(prior, source=f"bootstrap of {len(used)} frames spread over {total}, preset {args.bootstrap_preset}")
    lengths = [r.body_length_px for r in results]
    info = {
        "frames_sampled": len(seen),
        "frames_with_worm": len(masks),
        "frames_fit": len(used),
        "frames_used": prior.frames_used,
        "selection": prior.selection,
        "fit_length_px_p10_p50_p90": [float(v) for v in np.percentile(lengths, [10, 50, 90])],
    }
    return prior, info


def _nan(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float64)


def orientation_consistency(arrays: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """How often consecutive fitted frames agree on which end is the head."""

    fitted = np.nonzero(arrays["fitted"])[0]
    pairs = [(a, b) for a, b in zip(fitted[:-1], fitted[1:], strict=False) if b == a + 1]
    if not pairs:
        return None
    curves = arrays["centerline_xy"]
    agree = 0
    for a, b in pairs:
        same = np.linalg.norm(curves[b, 0] - curves[a, 0]) + np.linalg.norm(curves[b, -1] - curves[a, -1])
        swapped = np.linalg.norm(curves[b, 0] - curves[a, -1]) + np.linalg.norm(curves[b, -1] - curves[a, 0])
        agree += int(same <= swapped)
    return {"consecutive_pairs": len(pairs), "fraction_consistent": agree / len(pairs), "flips": len(pairs) - agree}


def main() -> int:
    args = parse_args()
    if args.step < 1 or args.slab < 1:
        raise SystemExit("--step and --slab must be positive")
    started = utc_now()
    config = build_config(args)
    module = load_segmenter(args.checkpoint, args.device)
    device = module.device
    source = RecordingSource(args.recording, args.dataset_root / "flat_fields")
    t = time.perf_counter()
    field = source.flat_field()
    field_seconds = time.perf_counter() - t
    template = default_width_template(config.n_points)

    with h5py.File(args.recording, "r") as handle:
        dataset = handle[DATASET_PATH]
        total = int(dataset.shape[0])
        start = max(0, args.start)
        stop = min(total, start + args.frames)
        if stop <= start:
            raise SystemExit(f"no frames in [{start}, {stop}) of {total}")
        indices = list(range(start, stop, args.step))
        n = len(indices)
        stem = args.name or f"{args.recording.stem}_f{start:06d}-{stop - 1:06d}" + (f"_s{args.step}" if args.step > 1 else "")
        run_dir = args.output_dir / f"{timestamp_slug(started)}_{stem}"
        run_dir.mkdir(parents=True, exist_ok=False)

        prior: RecordingPrior | None = None
        prior_source = None
        bootstrap_info: dict[str, Any] | None = None
        bootstrap_seconds = 0.0
        if args.prior_file is not None:
            prior, prior_source = RecordingPrior.load(args.prior_file), str(args.prior_file)
        elif args.prior == "bootstrap":
            cache_path = None if args.no_prior_cache else args.prior_cache / f"{args.recording.stem}_k{config.width_coefficients}.json"
            if cache_path is not None and cache_path.exists() and not args.rebootstrap:
                prior, prior_source = RecordingPrior.load(cache_path), f"cache {cache_path}"
            else:
                t = time.perf_counter()
                prior, bootstrap_info = bootstrap_prior(dataset, total, field, module, args, config, device)
                bootstrap_seconds = time.perf_counter() - t
                prior_source = "bootstrap"
                if cache_path is not None:
                    prior.save(cache_path)
                print(
                    f"bootstrap: length {prior.length_px:.0f} px (log sigma {prior.log_length_sigma:.3f}), width {prior.width_px:.1f} px,"
                    f" {prior.frames_used} of {bootstrap_info['frames_fit']} fits used, {bootstrap_seconds:.0f} s",
                    flush=True,
                )
        if prior is not None:
            config = prior.apply(config, shape_weight=args.prior_shape_weight)
            prior.save(run_dir / "recording_prior.json")
        if args.starts is not None:
            start_set = args.starts
        elif prior is not None:
            start_set = "skeleton+reversed"
        else:
            start_set = "all" if args.preset == "reference" else "skeleton+straight"
        start_shape = np.asarray(prior.width_shape, dtype=np.float64) if prior is not None else None
        start_length = prior.length_px if prior is not None else None
        orient_after_fit = prior is None and not args.no_orient

        arrays: dict[str, np.ndarray] = {
            "frame_index": np.asarray(indices, dtype=np.int64),
            "fitted": np.zeros(n, dtype=bool),
            "latent": _nan((n, config.coefficients + 4)),
            "width_px": _nan((n,)),
            "centerline_xy": _nan((n, config.n_points, 2)),
            "width_profile": _nan((n, config.n_points)),
            "width_shape": _nan((n, config.width_coefficients)),
            "taper_asymmetry": _nan((n,)),
            "reversed": np.zeros(n, dtype=bool),
            "orientation_gap": _nan((n,)),
            "iou": _nan((n,)),
            "energy": _nan((n,)),
            "points_in_fov": np.zeros(n, dtype=np.int64),
            "body_length_px": _nan((n,)),
            "crop": np.zeros((n, 4), dtype=np.int64),
            "worm_pixels": np.zeros(n, dtype=np.int64),
            "raw_worm_pixels": np.zeros(n, dtype=np.int64),
            "pixels_filled": np.zeros(n, dtype=np.int64),
            "components": np.zeros(n, dtype=np.int64),
            "pixels_outside_largest": np.zeros(n, dtype=np.int64),
            "n_starts": np.zeros(n, dtype=np.int64),
            "width_template": template,
        }
        best_start = ["" for _ in range(n)]
        skipped: dict[str, int] = {"empty_mask": 0, "small_mask": 0, "no_starts": 0, "fit_error": 0}
        timing = {stage: 0.0 for stage in STAGES}

        writer = None
        if args.video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(run_dir / "overlay.mp4"), fps=args.fps, codec="libx264", quality=args.quality, macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
        pool = ProcessPoolExecutor(max_workers=args.init_workers) if args.init_workers > 0 else None
        try:
            for slab_start in range(0, n, args.slab):
                slab = indices[slab_start : slab_start + args.slab]
                t0 = time.perf_counter()
                if args.step == 1:
                    raw = np.asarray(dataset[slab[0] : slab[-1] + 1], dtype=np.uint8)
                else:
                    raw = np.stack([np.asarray(dataset[i], dtype=np.uint8) for i in slab])
                t1 = time.perf_counter()
                corrected = np.stack([
                    np.clip(np.rint(apply_flat_field(frame, field, clip=(0.0, 255.0))), 0, 255).astype(np.uint8) for frame in raw
                ])
                t2 = time.perf_counter()
                probability = module.predict_probability_batch(corrected, batch_size=args.batch_size)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                masks: list[np.ndarray] = []
                fit_rows: list[int] = []
                for offset, prob in enumerate(probability):
                    row = slab_start + offset
                    mask, stats = clean_mask(prob, args.threshold, args.hole_radius, device)
                    for key, value in stats.items():
                        arrays[key][row] = value
                    if stats["worm_pixels"] == 0:
                        skipped["empty_mask"] += 1
                    elif stats["worm_pixels"] < args.min_worm_pixels:
                        skipped["small_mask"] += 1
                    else:
                        masks.append(mask)
                        fit_rows.append(row)
                t4 = time.perf_counter()
                names = START_SETS[start_set]
                if pool is not None:
                    starts = list(
                        pool.map(
                            initializations_for, masks, [config] * len(masks), [names] * len(masks),
                            [start_shape] * len(masks), [start_length] * len(masks),
                        )
                    )
                else:
                    starts = [initializations_for(m, config, names, start_shape, start_length) for m in masks]
                keep = [k for k, s in enumerate(starts) if s]
                skipped["no_starts"] += len(starts) - len(keep)
                masks = [masks[k] for k in keep]
                fit_rows = [fit_rows[k] for k in keep]
                starts = [starts[k] for k in keep]
                t5 = time.perf_counter()
                results: list[MaskFitResult | None] = []
                if masks:
                    try:
                        results = list(fit_masks(masks, starts, width_template=template, config=config, device=device))
                    except (ValueError, RuntimeError) as error:
                        print(f"batch fit failed ({error}); fitting frames one at a time", flush=True)
                        for mask, frame_starts in zip(masks, starts, strict=True):
                            try:
                                results.append(fit_masks([mask], [frame_starts], width_template=template, config=config, device=device)[0])
                            except (ValueError, RuntimeError):
                                results.append(None)
                                skipped["fit_error"] += 1
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                t6 = time.perf_counter()
                by_row: dict[int, MaskFitResult] = {}
                for row, frame_starts, result in zip(fit_rows, starts, results, strict=True):
                    arrays["n_starts"][row] = len(frame_starts)
                    if result is None:
                        continue
                    if orient_after_fit:
                        result, flipped = orient_tail_last(result, config=config)
                        arrays["reversed"][row] = flipped
                    else:
                        arrays["orientation_gap"][row] = orientation_gap(result)
                    by_row[row] = result
                    arrays["fitted"][row] = True
                    arrays["width_shape"][row] = result.width_shape
                    arrays["taper_asymmetry"][row] = taper_asymmetry(result.width_profile)
                    arrays["latent"][row] = result.latent
                    arrays["width_px"][row] = result.width_px
                    arrays["centerline_xy"][row] = result.centerline_xy
                    arrays["width_profile"][row] = result.width_profile
                    arrays["iou"][row] = result.records[result.best_index]["final_iou"]
                    arrays["energy"][row] = result.records[result.best_index]["final_soft_dice_energy"]
                    arrays["points_in_fov"][row] = result.points_in_fov
                    arrays["body_length_px"][row] = result.body_length_px
                    arrays["crop"][row] = (result.crop.x0, result.crop.x1, result.crop.y0, result.crop.y1)
                    best_start[row] = str(result.initializations[result.best_index].name)
                if writer is not None:
                    for offset, frame in enumerate(corrected):
                        row = slab_start + offset
                        result = by_row.get(row)
                        caption = f"{args.recording.stem} frame {indices[row]}"
                        if result is not None:
                            caption += (
                                f"  iou {arrays['iou'][row]:.3f}  length {arrays['body_length_px'][row]:.0f} px"
                                f"  width {arrays['width_px'][row]:.1f} px  in view {arrays['points_in_fov'][row] / config.n_points:.2f}"
                                f"  taper {arrays['taper_asymmetry'][row]:+.2f}"
                            )
                            if prior is not None:
                                caption += f"  gap {arrays['orientation_gap'][row]:.3f}"
                        else:
                            caption += "  no fit"
                        tube = centerline = None
                        if result is not None:
                            tube = np.zeros(frame.shape, dtype=bool)
                            tube[result.crop.y0 : result.crop.y1, result.crop.x0 : result.crop.x1] = result.rendered_hard_mask
                            centerline = result.centerline_xy
                        writer.append_data(draw_overlay(frame, centerline, tube, caption, args.scale))
                t7 = time.perf_counter()
                for stage, seconds in zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6), strict=True):
                    timing[stage] += seconds
                fitted_here = int(arrays["fitted"][slab_start : slab_start + len(slab)].sum())
                print(
                    f"frames {slab[0]}-{slab[-1]}: fit {1000 * (t6 - t5) / len(slab):.0f} ms/frame"
                    f" (init {1000 * (t5 - t4) / len(slab):.0f}, network {1000 * (t3 - t2) / len(slab):.0f},"
                    f" cleanup {1000 * (t4 - t3) / len(slab):.0f}), fitted {fitted_here}/{len(slab)},"
                    f" median iou {np.nanmedian(arrays['iou'][slab_start : slab_start + len(slab)]) if fitted_here else float('nan'):.3f}",
                    flush=True,
                )
        finally:
            if writer is not None:
                writer.close()
            if pool is not None:
                pool.shutdown()
        # Per-frame ambiguity signals (plan step 4) from the stored arrays.
        arrays["best_start"] = np.asarray(best_start)
        arrays.update(
            compute_ambiguity(
                arrays, prior=None if prior is None else prior.to_dict(), image_shape=(int(dataset.shape[1]), int(dataset.shape[2]))
            )
        )
        # Residual images for the worst frames and any requested ones: the
        # frames are read and segmented again, which is cheap for a handful.
        requested = [int(v) for v in args.dump_frames.split(",") if v.strip()]
        residual_files: list[str] = []
        for row in residual_rows(arrays, args.residual_frames, requested):
            frame_index = int(arrays["frame_index"][row])
            raw_frame = np.asarray(dataset[frame_index], dtype=np.uint8)
            frame = np.clip(np.rint(apply_flat_field(raw_frame, field, clip=(0.0, 255.0))), 0, 255).astype(np.uint8)
            probability = module.predict_probability_batch(frame[None], batch_size=1)[0]
            mask, _ = clean_mask(probability, args.threshold, args.hole_radius, device)
            tube = render_tube(
                arrays["centerline_xy"][row], arrays["width_profile"][row], *frame.shape, window=tuple(arrays["crop"][row]), device=device
            )
            image = draw_residual(frame, mask, tube, arrays["centerline_xy"][row], residual_caption(frame_index, arrays, row, mask))
            path = run_dir / f"frame_{frame_index:06d}_iou{float(arrays['iou'][row]):.3f}.png"
            image.save(path)
            residual_files.append(path.name)
    source.close()

    np.savez_compressed(run_dir / "poses.npz", **arrays)
    fitted = arrays["fitted"]
    iou = arrays["iou"][fitted]
    in_view = arrays["points_in_fov"][fitted] / config.n_points
    total_seconds = sum(timing.values())
    summary: dict[str, Any] = {
        "started_at": started,
        "finished_at": utc_now(),
        "recording": str(args.recording),
        "frames": [start, stop - 1],
        "step": args.step,
        "frame_count": n,
        "checkpoint": checkpoint_fingerprint(args.checkpoint),
        "git": git_revision(PROJECT_ROOT),
        "device": str(device),
        "threshold": args.threshold,
        "mask_cleanup": {"fill_holes_radius_px": args.hole_radius, "largest_component": True, "min_worm_pixels": args.min_worm_pixels},
        "fit_config": asdict(config),
        "width_template": "default_width_template",
        "preset": args.preset,
        "starts": start_set,
        "prior": None if prior is None else prior.to_dict(),
        "prior_source": prior_source,
        "bootstrap": None if bootstrap_info is None else {**bootstrap_info, "seconds": bootstrap_seconds},
        "init_workers": args.init_workers,
        "flat_field_seconds": field_seconds,
        "seconds": timing,
        "ms_per_frame": {stage: 1000.0 * seconds / n for stage, seconds in timing.items()},
        "total_ms_per_frame": 1000.0 * total_seconds / n,
        "frames_fitted": int(fitted.sum()),
        "frames_skipped": skipped,
        "iou": None if not len(iou) else {
            "median": float(np.median(iou)), "p10": float(np.percentile(iou, 10)), "min": float(iou.min()),
            "fraction_at_least_0.8": float(np.mean(iou >= 0.8)), "fraction_at_least_0.9": float(np.mean(iou >= 0.9)),
        },
        "body_length_px": None if not fitted.any() else {
            "median": float(np.median(arrays["body_length_px"][fitted])),
            "p10": float(np.percentile(arrays["body_length_px"][fitted], 10)),
            "p90": float(np.percentile(arrays["body_length_px"][fitted], 90)),
            "at_upper_bound": None if config.length_bounds_px is None else int(np.sum(arrays["body_length_px"][fitted] >= 0.99 * config.length_bounds_px[1])),
            "beyond_2_sigma_of_prior": None if prior is None else int(np.sum(
                np.abs(np.log(arrays["body_length_px"][fitted] / prior.length_px)) > 2 * prior.log_length_sigma
            )),
        },
        "orientation": None if prior is None or not fitted.any() else {
            "gap_median": float(np.nanmedian(arrays["orientation_gap"][fitted])),
            "gap_p10": float(np.nanpercentile(arrays["orientation_gap"][fitted], 10)),
            "frames_gap_below_0.002": int(np.sum(arrays["orientation_gap"][fitted] < 0.002)),
            "frames_gap_below_0.01": int(np.sum(arrays["orientation_gap"][fitted] < 0.01)),
            "reversed_start_won": int(np.sum([b.endswith("_reversed") for b in best_start if b])),
        },
        "width_px": None if not fitted.any() else {"median": float(np.median(arrays["width_px"][fitted]))},
        "width_model": {
            "coefficients": config.width_coefficients,
            "prior": config.width_shape_prior,
            "tail_placed_last": not args.no_orient,
            "frames_reversed": int(arrays["reversed"].sum()),
            "taper_asymmetry": None if not fitted.any() else {
                "median": float(np.median(arrays["taper_asymmetry"][fitted])),
                "p10": float(np.percentile(arrays["taper_asymmetry"][fitted], 10)),
                "p90": float(np.percentile(arrays["taper_asymmetry"][fitted], 90)),
                "frames_with_abs_below_0.1": int(np.sum(np.abs(arrays["taper_asymmetry"][fitted]) < 0.1)),
            },
            "orientation_consistency": orientation_consistency(arrays),
        },
        "in_view_fraction": None if not fitted.any() else {"median": float(np.median(in_view)), "frames_below_1": int(np.sum(in_view < 1.0))},
        "ambiguity": summarize_ambiguity(arrays) if fitted.any() else None,
        "best_start_counts": {name: int(count) for name, count in zip(*np.unique([b for b in best_start if b], return_counts=True))},
        "mask": {
            "worm_pixels_median": float(np.median(arrays["worm_pixels"])),
            "frames_with_multiple_components": int(np.sum(arrays["components"] > 1)),
            "pixels_outside_largest_median": float(np.median(arrays["pixels_outside_largest"])),
            "frames_with_filling": int(np.sum(arrays["pixels_filled"] > 0)),
        },
        "outputs": {
            "poses": str(run_dir / "poses.npz"),
            "video": str(run_dir / "overlay.mp4") if args.video else None,
            "residual_frames": residual_files,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "fit_config"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
