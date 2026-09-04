#!/usr/bin/env python3
"""Seed the segmentation store with pipeline-derived labels before any hand editing.

For uniformly spaced frames of each recording, the frame is flat-fielded and
segmented with the frozen local-darkness threshold; narrow holes are filled;
and the label is made *conservative* by marking as ignore (255) every pixel
where the evidence disagrees with itself: a band around the mask boundary,
and any pixel where the raw threshold and the cleaned component differ.
Optionally the mask-fit tube is rendered and its disagreement with the
component is also ignored.  Frames whose component is implausibly small are
skipped, so empty frames are not labeled here.

These labels are a starting point for the first network, not truth.  The
labeling app is where they get corrected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from worm_pose_gen.classical import ClassicalConfig, _dilate, _erode, segment_dark_ridge
from worm_pose_gen.flat_field import apply_flat_field
from worm_pose_gen.label_app import DEFAULT_RECORDINGS, RecordingSource
from worm_pose_gen.mask_fit import MaskFitConfig, fill_narrow_holes, fit_mask, standard_initializations
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore
from worm_pose_gen.segmenter import IGNORE_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", action="append", type=Path, dest="recordings")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--frames-per-recording", type=int, default=40)
    parser.add_argument("--boundary-ignore-px", type=int, default=2)
    parser.add_argument("--min-area", type=int, default=ClassicalConfig().min_area)
    parser.add_argument("--with-mask-fit", action="store_true", help="also ignore where the tube fit disagrees")
    parser.add_argument("--overwrite", action="store_true", help="replace labels that already exist")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def conservative_label(
    image: np.ndarray, *, boundary_ignore_px: int, with_mask_fit: bool, device: str | None
) -> tuple[np.ndarray, dict[str, float | int]] | None:
    cfg = ClassicalConfig()
    segmentation = segment_dark_ridge(image, cfg)
    component = segmentation.component
    if int(component.sum()) < cfg.min_area:
        return None
    filled, added = fill_narrow_holes(component, 8, device=device)
    label = filled.astype(np.uint8)
    ignore = np.zeros_like(filled)
    if boundary_ignore_px > 0:
        ignore |= _dilate(filled, boundary_ignore_px) & ~_erode(filled, boundary_ignore_px)
    # Pixels the raw threshold and the cleaned component disagree on are
    # exactly the debris and thin-end pixels we do not want to assert.
    ignore |= segmentation.high_threshold_mask != filled
    info: dict[str, float | int] = {"holes_filled_px": int(added)}
    if with_mask_fit:
        starts = standard_initializations(filled, config=MaskFitConfig())
        result = fit_mask(filled, starts, config=MaskFitConfig(), device=device)
        tube = np.zeros_like(filled)
        crop = result.crop
        tube[crop.y0 : crop.y1, crop.x0 : crop.x1] = result.rendered_hard_mask
        ignore |= tube != filled
        info["mask_fit_iou"] = float(result.records[result.best_index]["final_iou"])
    label[ignore] = IGNORE_LABEL
    info["ignore_fraction"] = float(ignore.mean())
    return label, info


def main() -> int:
    args = parse_args()
    store = SegmentationStore(args.dataset_root)
    written = 0
    skipped = 0
    for path in list(args.recordings or DEFAULT_RECORDINGS):
        source = RecordingSource(path, store.root / "flat_fields")
        indices = np.linspace(0, source.frame_count - 1, args.frames_per_recording, dtype=np.int64)
        for frame_index in indices:
            frame_index = int(frame_index)
            if not args.overwrite and store.has(source.name, frame_index):
                skipped += 1
                continue
            started = time.perf_counter()
            raw = source.read(frame_index)
            image = np.clip(np.rint(apply_flat_field(raw, source.flat_field(), clip=(0.0, 255.0))), 0, 255).astype(np.uint8)
            outcome = conservative_label(
                image, boundary_ignore_px=args.boundary_ignore_px,
                with_mask_fit=args.with_mask_fit, device=args.device,
            )
            if outcome is None:
                skipped += 1
                print(json.dumps({"recording": source.name, "frame_index": frame_index, "skipped": "no plausible component"}), flush=True)
                continue
            label, info = outcome
            record = store.save(
                source.name, frame_index, image, label, image_raw=raw, source_path=str(source.path),
                label_source="bootstrap_mask_fit" if args.with_mask_fit else "bootstrap_classical",
                flat_fielded=True,
            )
            written += 1
            print(
                json.dumps(
                    {
                        "recording": source.name, "frame_index": frame_index, "split": record.split,
                        "foreground_fraction": round(record.foreground_fraction, 4),
                        "ignore_fraction": round(info["ignore_fraction"], 4),
                        "seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
        source.close()
    print(json.dumps({"written": written, "skipped": skipped, "counts": store.counts(), "root": str(store.root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
