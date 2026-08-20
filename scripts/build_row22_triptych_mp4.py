#!/usr/bin/env python3
"""Encode the intact row-22 evidence triptych as a high-quality H.264 MP4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py

from build_row22_triptych_clip import (
    DEFAULT_CACHE,
    DEFAULT_MONTAGE,
    SOURCE_FRAMES,
    _compose,
    _raw_montage_frames,
    sha256,
)


DEFAULT_OUTPUT = Path(
    "/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/"
    "exp_smc_008a_row22_anchor_bracket/row22_triptych_20s_hq.mp4"
)


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "imageio-ffmpeg is required; run through uv with --with imageio-ffmpeg"
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--montage", type=Path, default=DEFAULT_MONTAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--output-fps", type=int, default=20)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--preset", default="slow")
    args = parser.parse_args()
    cache_path = args.cache.resolve(strict=True)
    montage_path = args.montage.resolve(strict=True)
    output = args.output.resolve()
    manifest = output.with_suffix(".json")
    partial = output.with_suffix(".partial.mp4")
    if output.exists() or manifest.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite HQ MP4 outputs")
    if args.panel_width < 320 or args.output_fps <= 0 or not 0 <= args.crf <= 51:
        raise ValueError("invalid output dimensions, frame rate, or CRF")

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
    width, height = frames[0].size
    if width % 2 or height % 2 or any(frame.size != (width, height) for frame in frames):
        raise RuntimeError("H.264 yuv420p output requires constant even frame dimensions")

    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_executable()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "1",
        "-i",
        "-",
        "-an",
        "-vf",
        f"fps={args.output_fps},format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-profile:v",
        "high",
        "-movflags",
        "+faststart",
        "-y",
        str(partial),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr}")
    os.replace(partial, output)

    record = {
        "schema_version": 1,
        "artifact": "row22_triptych_20s_high_quality_h264",
        "source_integrity_role": "intact prospectively materialized raw-frame catalog; not damaged HDF5/MP4",
        "cache": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "raw_frame_montage": str(montage_path),
        "raw_frame_montage_sha256": sha256(montage_path),
        "source_frames": list(SOURCE_FRAMES),
        "source_frame_stride": 4,
        "duration_seconds": len(frames),
        "output_fps": args.output_fps,
        "width": width,
        "height": height,
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "crf": args.crf,
        "preset": args.preset,
        "mp4": str(output),
        "mp4_size_bytes": output.stat().st_size,
        "mp4_sha256": sha256(output),
        "protected_2025_holdout_opened": False,
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
