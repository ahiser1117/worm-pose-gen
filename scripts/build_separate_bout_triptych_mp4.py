#!/usr/bin/env python3
"""Build an HQ triptych for a separate 2023-09-27 natural hard bout."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from build_row22_triptych_clip import X_BOUNDS, Y_BOUNDS, _compose, sha256
from build_row22_triptych_mp4 import _ffmpeg_executable
from worm_pose_gen.anchors import AnchorConfig, extract_mask_anchor
from worm_pose_gen.segmentation import SoftForegroundConfig, segment_soft_foreground


ROOT = Path(__file__).resolve().parents[1]
MONTAGE = Path(
    "/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/"
    "exp_smc_000_natural_hard_bout_catalog/coarse_raw/"
    "2023-09-27-01_f014554.png"
)
CONFIG = ROOT / "configs/smc_exp_001b_002b_revision.json"
OUTPUT = Path(
    "/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/"
    "diagnostic_videos/2023-09-27-01_f014554_triptych_20s_hq.mp4"
)
SOURCE_FRAMES = tuple(range(14514, 14591, 4))
CROP_MARGIN = 4
PANEL_SIZE = (640, 480)
SEGMENTATION_SIZE = (968, 732)
OUTPUT_FPS = 20
CRF = 14
PRESET = "slow"


def main() -> int:
    montage_path = MONTAGE.resolve(strict=True)
    config_path = CONFIG.resolve(strict=True)
    output = OUTPUT.resolve()
    partial = output.with_suffix(".partial.mp4")
    manifest = output.with_suffix(".json")
    preview = output.with_name(output.stem + "_preview.png")
    if output.exists() or partial.exists() or manifest.exists() or preview.exists():
        raise FileExistsError("refusing to overwrite separate-bout outputs")

    frozen = json.loads(config_path.read_text())
    segmentation_config = SoftForegroundConfig(**frozen["soft_foreground_config"])
    anchor_config = AnchorConfig(**frozen["anchor_config"])
    montage = Image.open(montage_path).convert("RGB")
    if montage.size != (1800, 1410):
        raise RuntimeError(f"unexpected montage dimensions: {montage.size}")

    originals: list[Image.Image] = []
    for y0, y1 in Y_BOUNDS:
        for x0, x1 in X_BOUNDS:
            originals.append(
                montage.crop(
                    (
                        x0 + CROP_MARGIN,
                        y0 + CROP_MARGIN,
                        x1 - CROP_MARGIN,
                        y1 - CROP_MARGIN,
                    )
                ).resize(PANEL_SIZE, Image.Resampling.LANCZOS)
            )
    if len(originals) != len(SOURCE_FRAMES):
        raise RuntimeError("montage layout no longer matches the 20-frame selection")

    composed: list[Image.Image] = []
    per_frame: list[dict[str, object]] = []
    for frame_index, original in zip(SOURCE_FRAMES, originals, strict=True):
        segmentation_input = np.asarray(
            original.convert("L").resize(SEGMENTATION_SIZE, Image.Resampling.LANCZOS)
        )
        segmentation = segment_soft_foreground(segmentation_input, segmentation_config)
        anchor = extract_mask_anchor(
            segmentation.cleaned_mask,
            probability=segmentation.probability_map,
            config=anchor_config,
        )
        rejection = ";".join(anchor.rejection_reasons)
        composed.append(_compose(original, segmentation.cleaned_mask, frame_index, rejection))
        per_frame.append(
            {
                "frame_index": frame_index,
                "accepted": bool(anchor.accepted),
                "rejection_reasons": list(anchor.rejection_reasons),
                "cleaned_foreground_area": int(segmentation.cleaned_mask.sum()),
                "quality_score": float(anchor.quality_score),
            }
        )
    accepted = sum(bool(item["accepted"]) for item in per_frame)
    if accepted:
        raise RuntimeError(
            "this renderer currently labels the diagnostic skeleton as rejected; "
            f"unexpectedly accepted {accepted} frames"
        )

    width, height = composed[0].size
    if width % 2 or height % 2 or any(frame.size != (width, height) for frame in composed):
        raise RuntimeError("H.264 yuv420p output requires constant even dimensions")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_executable(),
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
        f"fps={OUTPUT_FPS},format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        PRESET,
        "-crf",
        str(CRF),
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
        for frame in composed:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr}")
    os.replace(partial, output)
    composed[SOURCE_FRAMES.index(14554)].save(preview, format="PNG", optimize=True)

    record = {
        "schema_version": 1,
        "artifact": "separate_natural_bout_triptych_20s_high_quality_h264",
        "recording": "2023-09-27-01",
        "seed_frame": 14554,
        "distinct_from_prior_video": True,
        "source_integrity_role": "intact prospectively materialized raw-frame catalog; damaged recording files not used",
        "montage": str(montage_path),
        "montage_sha256": sha256(montage_path),
        "source_frames": list(SOURCE_FRAMES),
        "source_frame_stride": 4,
        "crop_margin_materialized_pixels": CROP_MARGIN,
        "segmentation_input_note": "clean catalog panels resized to the original 968x732 image dimensions",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "soft_foreground_config": asdict(segmentation_config),
        "anchor_config": asdict(anchor_config),
        "per_frame": per_frame,
        "accepted_count": accepted,
        "duration_seconds": len(composed),
        "output_fps": OUTPUT_FPS,
        "width": width,
        "height": height,
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "crf": CRF,
        "preset": PRESET,
        "mp4": str(output),
        "mp4_size_bytes": output.stat().st_size,
        "mp4_sha256": sha256(output),
        "preview": str(preview),
        "preview_sha256": sha256(preview),
        "protected_2025_holdout_opened": False,
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "per_frame"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
