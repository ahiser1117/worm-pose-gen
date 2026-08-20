#!/usr/bin/env python3
"""Build a 20 s row-22 diagnostic triptych from immutable experiment evidence.

The raw HDF5 source was found to be corrupted after EXP-SMC-008A completed.  This
builder therefore uses the clean raw frames materialized prospectively by the
hard-bout catalog and joins them to the corresponding cached EXP-SMC-008A masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from worm_pose_gen.classical import _thin


DEFAULT_CACHE = Path(
    "/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/"
    "exp_smc_008a_row22_anchor_bracket/per_frame.h5"
)
DEFAULT_MONTAGE = Path(
    "/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/"
    "exp_smc_000_natural_hard_bout_catalog/coarse_raw/"
    "2023-10-11-01_f013785.png"
)
DEFAULT_OUTPUT = Path(
    "/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/"
    "exp_smc_008a_row22_anchor_bracket/row22_triptych_20s.webp"
)

# The catalog figure contains a 5-column by 4-row grid of clean raw frames.
MONTAGE_SIZE = (1800, 1410)
X_BOUNDS = ((31, 343), (386, 698), (741, 1053), (1095, 1407), (1450, 1762))
Y_BOUNDS = ((75, 311), (344, 580), (613, 849), (881, 1117))
SOURCE_FRAMES = tuple(range(13745, 13822, 4))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return mask & ~interior


def _resize_binary(values: np.ndarray, size: tuple[int, int], thickness: int) -> Image.Image:
    image = Image.fromarray(values.astype(np.uint8) * 255)
    image = image.resize(size, Image.Resampling.NEAREST)
    if thickness > 1:
        image = image.filter(ImageFilter.MaxFilter(thickness))
    return image


def _color_overlay(base: Image.Image, binary: Image.Image, rgb: tuple[int, int, int]) -> None:
    color = Image.new("RGB", base.size, rgb)
    base.paste(color, mask=binary)


def _raw_montage_frames(path: Path, panel_width: int) -> list[Image.Image]:
    montage = Image.open(path).convert("RGB")
    if montage.size != MONTAGE_SIZE:
        raise RuntimeError(f"unexpected hard-bout montage size: {montage.size}")
    frames: list[Image.Image] = []
    for y0, y1 in Y_BOUNDS:
        for x0, x1 in X_BOUNDS:
            panel = montage.crop((x0, y0, x1, y1))
            height = round(panel_width * panel.height / panel.width)
            frames.append(panel.resize((panel_width, height), Image.Resampling.LANCZOS))
    if len(frames) != len(SOURCE_FRAMES):
        raise RuntimeError("montage layout no longer matches the frozen frame list")
    return frames


def _compose(
    original: Image.Image,
    mask: np.ndarray,
    frame_index: int,
    rejection: str,
) -> Image.Image:
    size = original.size
    scale = size[0] / 320.0
    binary_l = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    binary = binary_l.convert("RGB")
    overlay = original.copy()
    overlay_thickness = max(3, int(round(3 * scale)))
    if overlay_thickness % 2 == 0:
        overlay_thickness += 1
    boundary = _resize_binary(_mask_boundary(mask), size, overlay_thickness)
    skeleton = _resize_binary(_thin(mask), size, overlay_thickness)
    _color_overlay(overlay, boundary, (255, 176, 0))
    _color_overlay(overlay, skeleton, (0, 229, 255))

    panel_width, panel_height = size
    header_height = int(round(38 * scale))
    footer_height = int(round(49 * scale))
    canvas = Image.new("RGB", (3 * panel_width, header_height + panel_height + footer_height), "white")
    canvas.paste(original, (0, header_height))
    canvas.paste(binary, (panel_width, header_height))
    canvas.paste(overlay, (2 * panel_width, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(max(16, int(round(16 * scale))), bold=True)
    small_font = _font(max(13, int(round(13 * scale))))
    labels = ("ORIGINAL", "CLEANED BINARY MASK", "OVERLAY (REJECTED)")
    for index, label in enumerate(labels):
        left = index * panel_width
        box = draw.textbbox((0, 0), label, font=title_font)
        draw.text(
            (left + (panel_width - (box[2] - box[0])) / 2, int(round(9 * scale))),
            label,
            fill="black",
            font=title_font,
        )
        if index:
            draw.line(
                (left, 0, left, canvas.height),
                fill=(190, 190, 190),
                width=max(1, int(round(scale))),
            )
    reason = rejection.replace(";", " + ").replace("_", " ")
    status = f"frame {frame_index}  |  REJECTED: {reason}  |  no unique trusted midline"
    draw.text(
        (int(round(10 * scale)), header_height + panel_height + int(round(6 * scale))),
        status,
        fill=(150, 20, 35),
        font=small_font,
    )
    draw.text(
        (int(round(10 * scale)), header_height + panel_height + int(round(27 * scale))),
        "orange = mask boundary; cyan = cyclic/branched skeleton; 1 fps diagnostic slow motion",
        fill=(45, 55, 65),
        font=small_font,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--montage", type=Path, default=DEFAULT_MONTAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--panel-width", type=int, default=320)
    args = parser.parse_args()
    cache_path = args.cache.resolve(strict=True)
    montage_path = args.montage.resolve(strict=True)
    output = args.output.resolve()
    poster = output.with_name("row22_triptych_frame13785.png")
    manifest = output.with_name("row22_triptych_20s.json")
    if output.exists() or poster.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite row-22 clip outputs")
    if args.fps <= 0 or args.panel_width < 160:
        raise ValueError("positive fps and panel width >=160 are required")

    raw_frames = _raw_montage_frames(montage_path, args.panel_width)
    with h5py.File(cache_path, "r") as cache:
        if cache.attrs["experiment"] != "EXP-SMC-008A" or not bool(cache.attrs["complete"]):
            raise RuntimeError("expected the complete EXP-SMC-008A cache")
        if bool(cache.attrs["protected_2025_holdout_opened"]):
            raise RuntimeError("cache reports protected holdout access")
        cached_indices = cache["frame_index"][:]
        lookup = {int(frame): position for position, frame in enumerate(cached_indices)}
        positions = [lookup[frame] for frame in SOURCE_FRAMES]
        if bool(cache["accepted"][positions].any()):
            raise RuntimeError("the frozen row-22 diagnostic unexpectedly contains an accepted anchor")
        masks = cache["cleaned_mask"][positions]
        rejections = cache["rejection_reasons"].asstr()[positions]

    frames = [
        _compose(original, mask, frame_index, str(rejection))
        for original, mask, frame_index, rejection in zip(
            raw_frames, masks, SOURCE_FRAMES, rejections, strict=True
        )
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000 / args.fps))
    frames[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=78,
        method=4,
    )
    hard_position = SOURCE_FRAMES.index(13785)
    frames[hard_position].save(poster, format="PNG", optimize=True)
    record = {
        "schema_version": 1,
        "artifact": "row22_triptych_20s_diagnostic_slow_motion",
        "cache": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "raw_frame_montage": str(montage_path),
        "raw_frame_montage_sha256": sha256(montage_path),
        "source_frames": list(SOURCE_FRAMES),
        "source_frame_stride": 4,
        "frame_count": len(frames),
        "display_fps": args.fps,
        "assumed_acquisition_fps": 20.0,
        "effective_playback_speed": args.fps / (20.0 / 4.0),
        "duration_seconds": len(frames) / args.fps,
        "clip": str(output),
        "clip_sha256": sha256(output),
        "poster": str(poster),
        "poster_sha256": sha256(poster),
        "overlay_interpretation": "orange cleaned-mask boundary; cyan rejected raw skeleton; no inferred SMC pose",
        "protected_2025_holdout_opened": False,
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
