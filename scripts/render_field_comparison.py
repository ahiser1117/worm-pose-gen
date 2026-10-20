#!/usr/bin/env python3
"""Side-by-side video of several body-field networks' predictions on one temporal-evaluation clip.

Each panel is the clip's flat-fielded frame with the predicted A-P field
coloured on the mask (head blue through tail red, the turbo map), crossings
hatched white, the mask outline, the head peak as a filled white dot and the
tail peak as a black ring (only where the peak is above the end threshold).
Under each panel a strip marks the frames where that model's A-P field
flipped against the frame before (red), its head jumped (orange) or the head
was missing (grey); the white cursor is the current frame.

Inputs are the ``predictions/<clip>.npz`` of ``scripts/evaluate_temporal.py``
runs::

    scripts/project_env.sh uv run --no-sync --frozen python scripts/render_field_comparison.py --clip contact_0618_12 \\
        --run baseline=<data dir>/runs/temporal/on-food-finetune-329 \\
        --run semi=<data dir>/runs/temporal/arch-semi-1 \\
        --output <data dir>/runs/videos/contact_0618_12.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from worm_pose_gen import temporal_eval
from worm_pose_gen.body_proposal import END_THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run", action="append", required=True, help="label=evaluate_temporal output dir (repeat, in panel order)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--manifest", type=Path, default=temporal_eval.DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=temporal_eval.DEFAULT_CACHE)
    return parser.parse_args()


def turbo(values: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps

    return (colormaps["turbo"](np.clip(values, 0, 1))[..., :3] * 255).astype(np.uint8)


def events(prediction: temporal_eval.ClipPrediction) -> np.ndarray:
    """Per frame: 0 none, 1 head missing, 2 head jump, 3 A-P flip against the previous frame."""

    frames = len(prediction.mask)
    marks = np.zeros(frames, np.uint8)
    present = prediction.head[:, 2] >= END_THRESHOLD
    marks[~present] = 1
    step = np.linalg.norm(np.diff(prediction.head[:, :2], axis=0), axis=1)
    marks[1:][present[1:] & present[:-1] & (step > temporal_eval.BODY_WIDTH_PX)] = 2
    for t in range(1, frames):
        shared = prediction.mask[t - 1] & prediction.mask[t]
        if shared.sum() >= temporal_eval.MIN_SHARED_PX:
            a, b = prediction.ap_float(t - 1)[shared], prediction.ap_float(t)[shared]
            if a.std() > 1e-3 and b.std() > 1e-3 and np.corrcoef(a, b)[0, 1] < 0:
                marks[t] = 3
    return marks


STRIP_COLOURS = {0: (40, 40, 40), 1: (130, 130, 130), 2: (255, 150, 0), 3: (230, 30, 30)}
STRIP_HEIGHT = 10
TITLE_HEIGHT = 18


def panel(frame: np.ndarray, prediction: temporal_eval.ClipPrediction, t: int, label: str, marks: np.ndarray, scale: float) -> np.ndarray:
    rgb = np.repeat(frame[..., None], 3, axis=2).astype(np.float32)
    mask = prediction.mask[t]
    ap = prediction.ap_float(t)
    colour = turbo(np.nan_to_num(ap)).astype(np.float32)
    rgb[mask] = 0.45 * rgb[mask] + 0.55 * colour[mask]
    yy, xx = np.indices(mask.shape)
    hatch = prediction.overlap[t] & (((xx + yy) // 3) % 3 == 0)
    rgb[hatch] = 255
    edge = mask & ~ndimage.binary_erosion(mask)
    rgb[ndimage.binary_dilation(edge)] = (255, 255, 0)
    image = Image.fromarray(rgb.astype(np.uint8))
    width, height = int(round(image.width * scale)), int(round(image.height * scale))
    image = image.resize((width, height), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    for name, fill, outline in (("head", (255, 255, 255), (0, 0, 0)), ("tail", None, (0, 0, 0))):
        x, y, value = getattr(prediction, name)[t]
        if value >= END_THRESHOLD:
            r = 6
            draw.ellipse((x * scale - r, y * scale - r, x * scale + r, y * scale + r), fill=fill, outline=outline, width=3)
    canvas = Image.new("RGB", (width, height + TITLE_HEIGHT + STRIP_HEIGHT), (0, 0, 0))
    canvas.paste(image, (0, TITLE_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), label, fill=(255, 255, 255))
    frames = len(marks)
    for k, mark in enumerate(marks):
        x0 = int(k * width / frames)
        x1 = max(x0 + 1, int((k + 1) * width / frames))
        draw.rectangle((x0, height + TITLE_HEIGHT, x1 - 1, height + TITLE_HEIGHT + STRIP_HEIGHT - 1), fill=STRIP_COLOURS[int(mark)])
    cursor = int((t + 0.5) * width / frames)
    draw.line((cursor, height + TITLE_HEIGHT, cursor, height + TITLE_HEIGHT + STRIP_HEIGHT), fill=(255, 255, 255), width=2)
    return np.asarray(canvas)


def main() -> int:
    import imageio.v2 as imageio

    args = parse_args()
    clip = temporal_eval.clips(args.manifest, [args.clip])[0]
    frames, first = temporal_eval.cached_clip(clip, args.cache_dir)
    runs = []
    for item in args.run:
        label, _, directory = item.partition("=")
        prediction = temporal_eval.load_prediction(Path(directory) / "predictions" / f"{clip.name}.npz")
        runs.append((label, prediction, events(prediction)))
    columns = min(args.columns, len(runs))
    rows = -(-len(runs) // columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", quality=6, macro_block_size=1,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    try:
        for t in range(clip.frames):
            frame = frames[clip.start + t - first]
            panels = [panel(frame, p, t, f"{label}  {clip.name} f{clip.start + t}", marks, args.scale) for label, p, marks in runs]
            blank = np.zeros_like(panels[0])
            panels += [blank] * (rows * columns - len(panels))
            grid = np.concatenate([np.concatenate(panels[r * columns : (r + 1) * columns], axis=1) for r in range(rows)], axis=0)
            pad_h, pad_w = (-grid.shape[0]) % 2, (-grid.shape[1]) % 2
            writer.append_data(np.pad(grid, ((0, pad_h), (0, pad_w), (0, 0))))
    finally:
        writer.close()
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
