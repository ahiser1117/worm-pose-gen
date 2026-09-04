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
timing), ``poses.npz`` (per-frame arrays), and with ``--video`` an MP4 overlay
of the fitted tube outline and centerline on the flat-fielded frame.

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
from PIL import Image, ImageDraw
import torch

from worm_pose_gen.batch_fit import PRESETS, BatchFitConfig, fit_masks
from worm_pose_gen.connected_components import largest_component
from worm_pose_gen.flat_field import apply_flat_field
from worm_pose_gen.label_app import DATASET_PATH, RecordingSource
from worm_pose_gen.mask_fit import Initialization, MaskFitResult, default_width_template, fill_narrow_holes, standard_initializations
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
    "all": None,
}
TUBE_RGB = (90, 220, 140)
LINE_RGB = (255, 80, 165)
END_RGB = (255, 210, 60)


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
    parser.add_argument("--row-pixel-budget", type=int, default=BatchFitConfig.row_pixel_budget)
    parser.add_argument("--video", action="store_true", help="write an overlay MP4")
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
    return replace(config, **overrides)


def clean_mask(probability: np.ndarray, threshold: float, hole_radius: int, device: torch.device) -> tuple[np.ndarray, dict[str, int]]:
    raw = probability >= threshold
    stats = {"raw_worm_pixels": int(raw.sum()), "pixels_filled": 0, "components": 0, "pixels_outside_largest": 0, "worm_pixels": 0}
    if not raw.any():
        return raw, stats
    filled, added = fill_narrow_holes(raw, hole_radius, device=device)
    largest, area, count = largest_component(filled)
    stats.update(pixels_filled=int(added), components=int(count), pixels_outside_largest=int(filled.sum()) - int(area), worm_pixels=int(area))
    return largest, stats


def initializations_for(mask: np.ndarray, config: BatchFitConfig, names: tuple[str, ...] | None = None) -> list[Initialization]:
    """Standard starts, restricted to ``names`` when given (falling back to whatever exists)."""

    starts = standard_initializations(mask, config=config)
    if names is None:
        return starts
    chosen = [s for s in starts if s.name in names]
    return chosen if chosen else starts[:1]


def _boundary(mask: np.ndarray) -> np.ndarray:
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def draw_overlay(frame: np.ndarray, result: MaskFitResult | None, caption: str, scale: float) -> np.ndarray:
    rgb = np.repeat(frame[:, :, None], 3, axis=2).astype(np.uint8)
    if result is not None:
        crop = result.crop
        edge = _boundary(result.rendered_hard_mask)
        region = rgb[crop.y0 : crop.y1, crop.x0 : crop.x1]
        region[edge] = TUBE_RGB
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    if result is not None:
        points = [(float(x), float(y)) for x, y in result.centerline_xy]
        draw.line(points, fill=LINE_RGB, width=2)
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=END_RGB, width=2)
    draw.text((8, 8), caption, fill=(255, 255, 255))
    if scale != 1.0:
        image = image.resize((max(2, int(round(image.width * scale)) // 2 * 2), max(2, int(round(image.height * scale)) // 2 * 2)), Image.BILINEAR)
    return np.asarray(image)


def _nan(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float64)


def main() -> int:
    args = parse_args()
    if args.step < 1 or args.slab < 1:
        raise SystemExit("--step and --slab must be positive")
    started = utc_now()
    config = build_config(args)
    start_set = args.starts or ("all" if args.preset == "reference" else "skeleton+straight")
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

        arrays: dict[str, np.ndarray] = {
            "frame_index": np.asarray(indices, dtype=np.int64),
            "fitted": np.zeros(n, dtype=bool),
            "latent": _nan((n, config.coefficients + 4)),
            "width_px": _nan((n,)),
            "centerline_xy": _nan((n, config.n_points, 2)),
            "width_profile": _nan((n, config.n_points)),
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
                    starts = list(pool.map(initializations_for, masks, [config] * len(masks), [names] * len(masks)))
                else:
                    starts = [initializations_for(m, config, names) for m in masks]
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
                    by_row[row] = result
                    arrays["fitted"][row] = True
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
                            )
                        else:
                            caption += "  no fit"
                        writer.append_data(draw_overlay(frame, result, caption, args.scale))
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
    source.close()

    arrays["best_start"] = np.asarray(best_start)
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
            "at_upper_bound": int(np.sum(arrays["body_length_px"][fitted] >= 0.99 * config.length_bounds_px[1])),
        },
        "width_px": None if not fitted.any() else {"median": float(np.median(arrays["width_px"][fitted]))},
        "in_view_fraction": None if not fitted.any() else {"median": float(np.median(in_view)), "frames_below_1": int(np.sum(in_view < 1.0))},
        "best_start_counts": {name: int(count) for name, count in zip(*np.unique([b for b in best_start if b], return_counts=True))},
        "mask": {
            "worm_pixels_median": float(np.median(arrays["worm_pixels"])),
            "frames_with_multiple_components": int(np.sum(arrays["components"] > 1)),
            "pixels_outside_largest_median": float(np.median(arrays["pixels_outside_largest"])),
            "frames_with_filling": int(np.sum(arrays["pixels_filled"] > 0)),
        },
        "outputs": {"poses": str(run_dir / "poses.npz"), "video": str(run_dir / "overlay.mp4") if args.video else None},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "fit_config"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
