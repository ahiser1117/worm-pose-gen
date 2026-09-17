#!/usr/bin/env python3
"""Import the 2025-2026 body-wall labeling data into its own segmentation store.

Source: ``/storage/fs/homes/katie/omega/model_training/tr_data`` (audited
2026-09-15).  Every HDF5 file there is corrupt; the PNGs beside them carry the
same frames and labels.  Two label kinds are imported:

* ``legacy_manual`` - 276 hand-drawn (Labelbox polygon) full-body masks from
  ``augmented_training_1``.
* ``legacy_reconstructed`` - 653 frames that only exist in the three-channel
  "gradio" format (``-0`` body eroded by ~6 px, ``-2`` a thin ring just outside
  the body).  The full mask is the eroded body grown 8 px (4-connected) but
  never into the ring; on the 138 frames that also have a hand mask this
  reaches IoU 0.996.

PNG frames are the transpose of our ``/img_nir[t]`` layout and their frame
number is 1-based, both verified by exact pixel match against the source
recordings.  A recording name without a suffix (``2025-04-16``) is the same
recording as the suffixed set of that date (``2025-04-16-17``); verified on
every date whose recording still opens and assumed for the rest.

Each sample's ``image`` is flat-fielded with our estimator: from 64 frames of
the source recording when it can still be read, otherwise from the recording's
own labeled frames.  The field is cached under ``<store>/flat_fields/``.

Splits: whole recordings are pledged to validation or test so some animals
never enter training; every other sample takes the store's balanced 80/10/10
assignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage

from worm_pose_gen.flat_field import FlatField, estimate_flat_field
from worm_pose_gen.pose_run import flat_fielded
from worm_pose_gen.segmentation_dataset import SegmentationStore


SOURCE_ROOT = Path("/storage/fs/homes/katie/omega/model_training/tr_data")
LEGACY_DATASET_ROOT = Path("/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/segmentation_legacy_v1")
RAW_DATA = Path("/store1/shared/all_data_raw")
HAND_INPUTS = SOURCE_ROOT / "augmented_training_1/augmented_training_1_inputs"
HAND_LABELS = SOURCE_ROOT / "augmented_training_1/augmented_training_1_labels"
GRADIO_INPUTS = SOURCE_ROOT / "gradio2/gradio2_inputs"
GRADIO_LABELS = SOURCE_ROOT / "gradio2/gradio2_labels"
FLAT_FIELD_SAMPLE_COUNT = 64
GROW_ITERATIONS = 8

# PNG name prefix -> (recording, source recording or None when unreadable/missing)
RECORDINGS: dict[str, tuple[str, str | None]] = {
    "2024-05-16": ("2024-05-16-02", "prj_sexsharedneurons/2024-05-16/2024-05-16-02.h5"),
    "2024-05-16-02": ("2024-05-16-02", "prj_sexsharedneurons/2024-05-16/2024-05-16-02.h5"),
    "2024-05-17": ("2024-05-17-02", "prj_sexsharedneurons/2024-05-17/2024-05-17-02.h5"),
    "2024-05-17-02": ("2024-05-17-02", "prj_sexsharedneurons/2024-05-17/2024-05-17-02.h5"),
    "2024-05-21": ("2024-05-21-01", "prj_sexsharedneurons/2024-05-21/2024-05-21-01.h5"),
    "2024-05-21-01": ("2024-05-21-01", "prj_sexsharedneurons/2024-05-21/2024-05-21-01.h5"),
    "2025-03-13": ("2025-03-13-01", "prj_sexsharedneurons/2025-03-13/2025-03-13-01.h5"),
    "2025-03-13-01": ("2025-03-13-01", "prj_sexsharedneurons/2025-03-13/2025-03-13-01.h5"),
    "2025-03-18-06": ("2025-03-18-06", "prj_sexsharedneurons/2025-03-18/2025-03-18-06.h5"),
    "2025-03-19-01": ("2025-03-19-01", "prj_sexsharedneurons/2025-03-19/2025-03-19-01.h5"),
    "2025-03-25-06": ("2025-03-25-06", "prj_sexsharedneurons/2025-03-25/2025-03-25-06.h5"),
    "2025-03-26": ("2025-03-26-06", "prj_sexsharedneurons/2025-03-26/2025-03-26-06.h5"),
    "2025-03-26-06": ("2025-03-26-06", "prj_sexsharedneurons/2025-03-26/2025-03-26-06.h5"),
    "2025-04-01-01": ("2025-04-01-01", "prj_sexsharedneurons/2025-04-01/2025-04-01-01.h5"),
    "2025-04-02": ("2025-04-02-06", "prj_sexsharedneurons/2025-04-02/2025-04-02-06.h5"),
    "2025-04-02-06": ("2025-04-02-06", "prj_sexsharedneurons/2025-04-02/2025-04-02-06.h5"),
    "2025-04-08": ("2025-04-08-01", "prj_sexsharedneurons/2025-04-08/2025-04-08-01.h5"),
    "2025-04-08-01": ("2025-04-08-01", "prj_sexsharedneurons/2025-04-08/2025-04-08-01.h5"),
    "2025-04-16": ("2025-04-16-17", "prj_sexsharedneurons/2025-04-16/2025-04-16-17.h5"),
    "2025-04-16-17": ("2025-04-16-17", "prj_sexsharedneurons/2025-04-16/2025-04-16-17.h5"),
    "2025-10-21": ("2025-10-21", None),  # -01/-06/-11 all unreadable; which one is unknown
    "2025-10-22": ("2025-10-22", None),  # -01/-07 unreadable
    "2025-11-05": ("2025-11-05-18", "prj_sexsharedneurons/2025-11-05/2025-11-05-18.h5"),
    "2025-11-18": ("2025-11-18-01", "prj_sexsharedneurons/2025-11-18/2025-11-18-01.h5"),
    "2025-11-20": ("2025-11-20-01", None),  # not under all_data_raw
}
# Whole recordings that never enter training (all hand-labeled).
PLEDGED_SPLITS = {
    "2024-05-17-02": "val",
    "2025-03-19-01": "val",
    "2025-03-26-06": "test",
    "2025-03-25-06": "test",
    "2025-04-08-01": "test",
}
NAME = re.compile(r"^(?P<prefix>\d{4}-\d{2}-\d{2}(?:-\d{2})?)_frame_(?P<number>\d{5})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=LEGACY_DATASET_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="import only this many frames (smoke test)")
    return parser.parse_args()


def load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def raw_frame(path: Path) -> np.ndarray:
    """The frame in our ``[732, 968]`` orientation."""

    return np.ascontiguousarray(load_png(path).T)


def hand_mask(stem: str) -> np.ndarray:
    return np.ascontiguousarray((load_png(HAND_LABELS / f"{stem}.png") > 0).T)


def reconstructed_mask(stem: str) -> np.ndarray:
    body = load_png(GRADIO_LABELS / f"{stem}-0.png").astype(bool)
    ring = load_png(GRADIO_LABELS / f"{stem}-2.png").astype(bool)
    grown = ndimage.binary_dilation(body, iterations=GROW_ITERATIONS, mask=~ring)
    return np.ascontiguousarray(grown.T)


def collect_frames() -> list[dict]:
    """Every labeled frame once; a hand mask wins over a reconstructed one."""

    frames: dict[str, dict] = {}
    for label_dir, input_dir, kind in ((GRADIO_LABELS, GRADIO_INPUTS, "legacy_reconstructed"), (HAND_LABELS, HAND_INPUTS, "legacy_manual")):
        for path in sorted(input_dir.glob("*.png")):
            match = NAME.match(path.stem)
            if match is None:
                raise ValueError(f"unexpected frame name {path.name}")
            if kind == "legacy_reconstructed" and not (label_dir / f"{path.stem}-0.png").exists():
                continue
            if kind == "legacy_manual" and not (label_dir / f"{path.stem}.png").exists():
                continue
            recording, source = RECORDINGS[match["prefix"]]
            frames[path.stem] = {
                "stem": path.stem, "input": path, "kind": kind, "recording": recording, "source": source,
                "frame_index": int(match["number"]) - 1,
            }
    return sorted(frames.values(), key=lambda f: (f["recording"] not in PLEDGED_SPLITS, f["stem"]))


def source_frames(source: Path) -> list[np.ndarray]:
    frames = []
    with h5py.File(source, "r") as handle:
        video = handle["/img_nir"]
        for i in np.linspace(0, video.shape[0] - 1, FLAT_FIELD_SAMPLE_COUNT, dtype=np.int64):
            try:
                frames.append(np.asarray(video[int(i)], dtype=np.uint8))
            except OSError:
                continue
    return frames


def recording_flat_field(store_root: Path, recording: str, frames: list[dict]) -> tuple[FlatField, str]:
    cache_dir = store_root / "flat_fields"
    cache = cache_dir / f"{recording}.npz"
    if cache.exists():
        with np.load(cache) as archive:
            field = FlatField(
                illumination=np.asarray(archive["illumination"], dtype=np.float64), dark_level=float(archive["dark_level"]),
                reference_level=float(archive["reference_level"]), gain=np.asarray(archive["gain"], dtype=np.float64),
            )
            return field, f"cache ({archive['origin']})"
    calibration: list[np.ndarray] = []
    origin = "labeled frames"
    source = frames[0]["source"]
    if source is not None and (RAW_DATA / source).exists():
        try:
            calibration = source_frames(RAW_DATA / source)
        except OSError:
            calibration = []
        if len(calibration) >= 8:
            origin = "source recording"
    if len(calibration) < 8:
        calibration = [raw_frame(f["input"]) for f in frames]
    field = estimate_flat_field(
        np.stack(calibration), temporal_quantile=0.8, spatial_radius=31, smoothing_passes=2, min_gain=0.5, max_gain=2.5,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, illumination=field.illumination, dark_level=field.dark_level, reference_level=field.reference_level,
        gain=field.gain, origin=origin,
    )
    return field, origin


def main() -> int:
    args = parse_args()
    store = SegmentationStore(args.dataset_root)
    frames = collect_frames()
    if args.limit is not None:
        frames = frames[: args.limit]
    by_recording: dict[str, list[dict]] = {}
    for frame in frames:
        by_recording.setdefault(frame["recording"], []).append(frame)
    fields = {}
    for recording, group in by_recording.items():
        fields[recording], origin = recording_flat_field(store.root, recording, group)
        print(f"{recording}: flat field from {origin}, {len(group)} frames")
    saved = {"legacy_manual": 0, "legacy_reconstructed": 0}
    for number, frame in enumerate(frames, start=1):
        raw = raw_frame(frame["input"])
        mask = hand_mask(frame["stem"]) if frame["kind"] == "legacy_manual" else reconstructed_mask(frame["stem"])
        image = flat_fielded(raw, fields[frame["recording"]])
        source = str(RAW_DATA / frame["source"]) if frame["source"] else str(frame["input"].parent)
        store.save(
            frame["recording"], frame["frame_index"], image, mask, image_raw=raw, source_path=source,
            label_source=frame["kind"], flat_fielded=True, split=PLEDGED_SPLITS.get(frame["recording"]),
        )
        saved[frame["kind"]] += 1
        if number % 100 == 0:
            print(f"saved {number}/{len(frames)}")
    print("saved", saved, "store counts", store.counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
