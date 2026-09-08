#!/usr/bin/env python3
"""Segment a stretch of a recording and fit the body model to every frame.

Frames are read from the HDF5 recording in slabs, flat-fielded with the
per-recording correction the labeling app uses, pushed through the promoted
segmenter, cleaned (probability at or above ``--threshold``; then, unless
``--no-fill-holes`` / ``--no-largest-component`` / ``--raw-mask`` say
otherwise, narrow holes filled and the largest component kept), and then
fit in GPU batches with
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

After the independent fits, stretches of frames whose ambiguity score
reaches ``--propagate-min-score`` are refit by temporal propagation (plan
step 5): the good pose before the stretch is carried forward through it and
the good pose after it backward, each frame warm-started from its
neighbour, all stretches in lockstep; per frame the lowest total energy
among independent, forward and backward wins (``source`` in ``poses.npz``;
``--no-propagate`` skips this).
``scripts/render_pose_run.py`` produces the same video and residual images
for a stored run without refitting.  The per-frame logic (segmentation,
bootstrap, fit, propagation, track pass, statistics) lives in
``worm_pose_gen.pipeline``; this script composes it over a run directory the
way the app's stages compose it over a workspace.

Example (one minute at 20 fps of an unseen recording, with video):

    scripts/project_env.sh uv run --no-sync --frozen python scripts/fit_recording.py \\
        --recording /store1/shared/all_data_raw/prj_aversion/2024-05-28/2024-05-28-02.h5 \\
        --start 0 --frames 1200 --video
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from worm_pose_gen.ambiguity import compute_ambiguity
from worm_pose_gen.batch_fit import PRESETS, BatchFitConfig
from worm_pose_gen.pipeline import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PRIOR_CACHE,
    EXTERNAL_ROOT,
    PROJECT_ROOT,
    START_SETS,
    TIMING_STAGES as STAGES,
    FitParams,
    Frames,
    PriorParams,
    PropagateParams,
    SegmentParams,
    TrackParams,
    fit_frames,
    fit_setup,
    fit_statistics,
    independent_copies,
    load_segmentation_model,
    new_arrays,
    propagation_pass,
    resolve_prior,
    segment_frames,
    store_mask_stats,
    track_length_pass,
)
from worm_pose_gen.pose_run import (
    clean_mask,
    draw_residual,
    render_tube,
    residual_caption,
    residual_rows,
    write_overlay_video,
)
from worm_pose_gen.run_records import checkpoint_fingerprint, git_revision, timestamp_slug, utc_now
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT


DEFAULT_OUTPUT_DIR = (EXTERNAL_ROOT / "poses") if EXTERNAL_ROOT.exists() else PROJECT_ROOT / "checkpoints" / "poses"
HOLE_FILL_RADIUS_PX = 8
MIN_WORM_PIXELS = 500


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
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True, help="fill narrow holes in the mask before fitting")
    parser.add_argument(
        "--largest-component", action=argparse.BooleanOptionalAction, default=True, help="keep only the largest connected component of the mask"
    )
    parser.add_argument("--raw-mask", action="store_true", help="shorthand for --no-fill-holes --no-largest-component")
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
    parser.add_argument("--no-propagate", action="store_true", help="skip temporal propagation across ambiguous stretches")
    parser.add_argument("--propagate-min-score", type=int, default=2, help="ambiguity score that seeds a stretch")
    parser.add_argument("--propagate-pad", type=int, default=2, help="frames added on each side of a seed")
    parser.add_argument("--propagate-max-gap", type=int, default=3, help="seeds closer than this are one stretch")
    parser.add_argument("--chain-length-sigma", type=float, default=0.02, help="log-sigma of the length prior inside propagation chains (0 = the fit's own)")
    parser.add_argument("--prediction-damping", type=float, default=0.6, help="step 6a: damping of the first-order pose prediction inside chains (0 = copy the neighbour, as before)")
    parser.add_argument("--temporal-prior-weight", type=float, default=0.01, help="step 6a: weight of the pull toward the predicted pose inside chains (0 = off)")
    parser.add_argument("--temporal-prior-sigma", type=float, default=0.5, help="step 6a: sigma of that pull, in body widths")
    parser.add_argument("--propagate-preset", default="fast", choices=tuple(PRESETS), help="step 6c: schedule of the stretch refit pass ('fast' = 70%% of the fit's steps; 'balanced' runs that preset's steps on the fit's rasters, which on the raw spiral let the chains wander: 4 failures against 0)")
    parser.add_argument("--no-track-length", action="store_true", help="step 6b: skip the track length pass (refit of clipped and length-deviating frames with the track's length prior)")
    parser.add_argument("--track-window", type=int, default=50, help="step 6b: half-window in frames of the track length median")
    parser.add_argument("--track-sigma", type=float, default=0.02, help="step 6b: log-sigma of the track length prior in the refit")
    parser.add_argument("--track-tolerance", type=float, default=0.02, help="step 6b: log deviation from the track length that triggers a refit")
    parser.add_argument("--no-jump-seeds", action="store_true", help="step 6b: seed stretches by ambiguity score only, not by length, pose or in-view jumps")
    parser.add_argument("--seed-length-fraction", type=float, default=0.0, help="step 6b: length deviation from the recording's median length that seeds a stretch (0 = do not seed by length; pose and in-view jumps always seed)")
    parser.add_argument(
        "--track-refit", default="clipped-deviating", choices=("clipped-deviating", "clipped", "deviating", "all"),
        help="step 6b: which frames the track pass refits: clipped frames off the track (default), every clipped frame, every frame off the track, or both",
    )
    parser.add_argument("--min-bend-radius", type=float, default=None, help="minimum bend radius in body widths (preset default 0.5; 0 disables the bend penalty)")
    parser.add_argument("--path-length-weight", type=float, default=1.0, help="step 6c: weight of the squared log length change (in 2%% units) between consecutive frames")
    parser.add_argument("--beam", type=int, default=3, help="step 6c: distinct chain states kept per direction")
    parser.add_argument("--no-anchor-diversity", action="store_true", help="step 6b: start each chain from the anchor's pose only, not also from the anchor refit from the frame beyond")
    parser.add_argument("--no-refit-independent", action="store_true", help="step 6c: do not refit the stored independent pose under the chain schedule")
    parser.add_argument("--no-path", action="store_true", help="step 6c: pick the lowest energy per frame instead of one path per stretch")
    parser.add_argument("--path-temperature", type=float, default=0.01, help="step 6c: energy scale of the path's node cost")
    parser.add_argument("--path-distance-weight", type=float, default=1.0, help="step 6c: weight of the squared pose distance (in widths) between consecutive frames")
    parser.add_argument("--path-inview-weight", type=float, default=2.0, help="step 6c: weight of the change of the in-view fraction between consecutive frames")
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


def segment_params(args: argparse.Namespace) -> SegmentParams:
    """Segmentation switches from the command line (``--raw-mask`` turns both cleanup steps off)."""

    return SegmentParams(
        checkpoint=str(args.checkpoint), threshold=args.threshold, hole_radius=args.hole_radius,
        fill_holes=bool(args.fill_holes) and not args.raw_mask, largest_only=bool(args.largest_component) and not args.raw_mask,
        min_worm_pixels=args.min_worm_pixels, batch_size=args.batch_size, slab=args.slab, dataset_root=str(args.dataset_root),
    )


def prior_params(args: argparse.Namespace, segmentation: SegmentParams) -> PriorParams:
    return PriorParams(
        prior=args.prior, prior_file=None if args.prior_file is None else str(args.prior_file), prior_cache=str(args.prior_cache),
        use_cache=not args.no_prior_cache, rebootstrap=args.rebootstrap, bootstrap_frames=args.bootstrap_frames,
        bootstrap_target=args.bootstrap_target, bootstrap_preset=args.bootstrap_preset,
        checkpoint=segmentation.checkpoint, threshold=segmentation.threshold, hole_radius=segmentation.hole_radius,
        fill_holes=segmentation.fill_holes, largest_only=segmentation.largest_only, min_worm_pixels=segmentation.min_worm_pixels,
        batch_size=segmentation.batch_size, dataset_root=segmentation.dataset_root,
    )


def fit_params(args: argparse.Namespace) -> FitParams:
    return FitParams(
        preset=args.preset, fine_stride=args.fine_stride, padding=args.padding, starts=args.starts, compile=not args.no_compile,
        width_coefficients=args.width_coefficients, width_prior=args.width_prior, orient=not args.no_orient, prior=args.prior,
        prior_shape_weight=args.prior_shape_weight, min_bend_radius=args.min_bend_radius, row_pixel_budget=args.row_pixel_budget,
        init_workers=args.init_workers, slab=args.slab, min_worm_pixels=args.min_worm_pixels,
    )


def propagate_params(args: argparse.Namespace) -> PropagateParams:
    return PropagateParams(
        min_score=args.propagate_min_score, pad=args.propagate_pad, max_gap=args.propagate_max_gap,
        chain_length_sigma=args.chain_length_sigma, prediction_damping=args.prediction_damping,
        temporal_prior_weight=args.temporal_prior_weight, temporal_prior_sigma=args.temporal_prior_sigma,
        propagate_preset=args.propagate_preset, jump_seeds=not args.no_jump_seeds, seed_length_fraction=args.seed_length_fraction,
        beam=args.beam, anchor_diversity=not args.no_anchor_diversity, refit_independent=not args.no_refit_independent,
        path=not args.no_path, path_temperature=args.path_temperature, path_distance_weight=args.path_distance_weight,
        path_inview_weight=args.path_inview_weight, path_length_weight=args.path_length_weight,
    )


def track_params(args: argparse.Namespace) -> TrackParams:
    return TrackParams(track_window=args.track_window, track_sigma=args.track_sigma, track_tolerance=args.track_tolerance, track_refit=args.track_refit)


def build_config(args: argparse.Namespace) -> BatchFitConfig:
    """The preset with the command-line overrides applied (kept for callers of this script's helpers)."""

    from worm_pose_gen.pipeline import build_fit_config

    return build_fit_config(fit_params(args))


def main() -> int:
    args = parse_args()
    if args.step < 1 or args.slab < 1:
        raise SystemExit("--step and --slab must be positive")
    started = utc_now()
    segmentation = segment_params(args)
    fitting = fit_params(args)
    model = load_segmentation_model(args.checkpoint, args.device)
    device = model.device
    frames = Frames(args.recording, dataset_root=args.dataset_root)
    frames.field()

    total = frames.total
    start = max(0, args.start)
    stop = min(total, start + args.frames)
    if stop <= start:
        raise SystemExit(f"no frames in [{start}, {stop}) of {total}")
    indices = list(range(start, stop, args.step))
    n = len(indices)
    stem = args.name or f"{args.recording.stem}_f{start:06d}-{stop - 1:06d}" + (f"_s{args.step}" if args.step > 1 else "")
    run_dir = args.output_dir / f"{timestamp_slug(started)}_{stem}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # The recording prior: a given file, the cache, or a bootstrap over the whole recording.
    prior, prior_source, bootstrap_info = resolve_prior(
        frames, prior_params(args, segmentation), fit_setup(fitting, None).config, device, model=model
    )
    if prior is not None:
        prior.save(run_dir / "recording_prior.json")
    setup = fit_setup(fitting, prior)
    config = setup.config
    prior_dict = None if prior is None else prior.to_dict()
    image_shape = frames.shape

    arrays = new_arrays(np.asarray(indices, dtype=np.int64), config)
    skipped: dict[str, int] = {"empty_mask": 0, "small_mask": 0, "no_starts": 0, "fit_error": 0}
    timing = {stage: 0.0 for stage in STAGES}

    # Independent fits, slab by slab: segment, clean, start, fit, store.
    pool = ProcessPoolExecutor(max_workers=args.init_workers) if args.init_workers > 0 else None
    try:
        for slab_start in range(0, n, args.slab):
            slab = indices[slab_start : slab_start + args.slab]
            slab_rows = list(range(slab_start, slab_start + len(slab)))
            t0 = time.perf_counter()
            masks, stats, seg_timing = segment_frames(frames, model, slab, segmentation, device)
            fit_masks_here: list[np.ndarray] = []
            fit_rows: list[int] = []
            for row, mask, frame_stats in zip(slab_rows, masks, stats, strict=True):
                store_mask_stats(arrays, row, frame_stats)
                if frame_stats["worm_pixels"] == 0:
                    skipped["empty_mask"] += 1
                elif frame_stats["worm_pixels"] < args.min_worm_pixels:
                    skipped["small_mask"] += 1
                else:
                    fit_masks_here.append(mask)
                    fit_rows.append(row)
            fit_timing = fit_frames(fit_masks_here, fit_rows, setup, arrays, device, pool=pool, skipped=skipped)
            for stage, seconds in {**seg_timing, **fit_timing}.items():
                timing[stage] += seconds
            fitted_here = int(arrays["fitted"][slab_rows].sum())
            print(
                f"frames {slab[0]}-{slab[-1]}: fit {1000 * fit_timing['fit'] / len(slab):.0f} ms/frame"
                f" (init {1000 * fit_timing['init'] / len(slab):.0f}, network {1000 * seg_timing['network'] / len(slab):.0f},"
                f" cleanup {1000 * seg_timing['cleanup'] / len(slab):.0f}), fitted {fitted_here}/{len(slab)},"
                f" median iou {np.nanmedian(arrays['iou'][slab_rows]) if fitted_here else float('nan'):.3f},"
                f" {time.perf_counter() - t0:.1f} s",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown()

    def segment_rows(rows: list[int]) -> dict[int, np.ndarray]:
        """Re-segment and clean the masks of these rows (masks are not kept from the first pass)."""

        out: dict[int, np.ndarray] = {}
        for chunk_start in range(0, len(rows), args.batch_size):
            chunk = rows[chunk_start : chunk_start + args.batch_size]
            masks, stats, _ = segment_frames(frames, model, [int(arrays["frame_index"][r]) for r in chunk], segmentation, device)
            for r, mask, frame_stats in zip(chunk, masks, stats, strict=True):
                if frame_stats["worm_pixels"] >= args.min_worm_pixels:
                    out[int(r)] = mask
        return out

    # Per-frame ambiguity signals (plan step 4) from the stored arrays; the
    # independent pose itself is kept so a viewer can show what propagation
    # replaced (worm_pose_gen.pose_viewer).
    arrays.update(compute_ambiguity(arrays, prior=prior_dict, image_shape=image_shape))
    arrays["score_independent"] = arrays["ambiguity_score"].copy()
    independent_copies(arrays)
    propagation_info: dict[str, Any] | None = None
    stretches: list[tuple[int, int]] = []
    if not args.no_propagate:
        # Temporal propagation (plan steps 5, 6a-c) across the ambiguous stretches.
        outcome = propagation_pass(arrays, segment_rows, setup, propagate_params(args), device, image_shape=image_shape)
        propagation_info, stretches = outcome.info, outcome.stretches
        timing["propagate"] = propagation_info["seconds"]
    # Track length pass (plan step 6b), after propagation: run before it, it
    # perturbed the stretch anchors (coil_0201: 0 -> 26 frames below 0.9).
    track_info: dict[str, Any] | None = None
    if not args.no_track_length:
        track_info, _ = track_length_pass(arrays, segment_rows, setup, track_params(args), device, stretches=stretches, image_shape=image_shape)
        timing["track"] = track_info["seconds"]

    # The overlay video is rendered from the final arrays, after propagation, so it shows the poses as stored.
    if args.video:
        t_video = time.perf_counter()
        write_overlay_video(
            run_dir / "overlay.mp4", frames.dataset, frames.field(), arrays, caption_prefix=args.recording.stem,
            fps=args.fps, scale=args.scale, quality=args.quality, slab=args.slab, device=device,
        )
        timing["video"] += time.perf_counter() - t_video
    # Residual images for the worst frames and any requested ones: the frames
    # are read and segmented again, which is cheap for a handful.
    requested = [int(v) for v in args.dump_frames.split(",") if v.strip()]
    residual_files: list[str] = []
    for row in residual_rows(arrays, args.residual_frames, requested):
        frame_index = int(arrays["frame_index"][row])
        frame = frames.corrected([frame_index])[0][0]
        probability = model.predict_probability_batch(frame[None], batch_size=1)[0]
        mask, _ = clean_mask(probability, args.threshold, args.hole_radius, device, fill_holes=segmentation.fill_holes, largest_only=segmentation.largest_only)
        tube = render_tube(
            arrays["centerline_xy"][row], arrays["width_profile"][row], *frame.shape, window=tuple(arrays["crop"][row]), device=device
        )
        image = draw_residual(frame, mask, tube, arrays["centerline_xy"][row], residual_caption(frame_index, arrays, row, mask))
        path = run_dir / f"frame_{frame_index:06d}_iou{float(arrays['iou'][row]):.3f}.png"
        image.save(path)
        residual_files.append(path.name)
    frames.close()

    np.savez_compressed(run_dir / "poses.npz", **arrays)
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
        "mask_cleanup": {
            "fill_holes": segmentation.fill_holes, "fill_holes_radius_px": args.hole_radius,
            "largest_component": segmentation.largest_only, "min_worm_pixels": args.min_worm_pixels,
        },
        "fit_config": asdict(config),
        "width_template": "default_width_template",
        "preset": args.preset,
        "starts": setup.start_set,
        "prior": prior_dict,
        "prior_source": prior_source,
        "bootstrap": bootstrap_info,
        "init_workers": args.init_workers,
        "flat_field_seconds": frames.field_seconds,
        "seconds": timing,
        "ms_per_frame": {stage: 1000.0 * seconds / n for stage, seconds in timing.items()},
        "total_ms_per_frame": 1000.0 * total_seconds / n,
        "frames_skipped": skipped,
        **fit_statistics(arrays, config, prior, orient=not args.no_orient),
        "track_length": track_info,
        "propagation": propagation_info,
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
