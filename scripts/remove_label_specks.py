#!/usr/bin/env python3
"""Remove isolated specks of paint from the hand labels of a segmentation store.

A speck is a connected worm component smaller than ``--max-pixels`` that lies
more than ``--min-distance`` pixels from the label's largest component: brush
leftovers far from the body, not a tail tip at the image edge beside it.
Labels whose largest component is itself a speck-sized sliver are listed but
never changed; decide those by eye.

Dry run by default; ``--apply`` saves each cleaned label through the store,
which archives the previous mask under ``revisions/<sample_id>/`` and raises
the revision (so the sample's body fields become stale).  The label source is
kept: an automated cleanup is not a new hand label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from worm_pose_gen.corpus import CorpusStore
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--max-pixels", type=int, default=30)
    parser.add_argument("--min-distance", type=float, default=10.0)
    parser.add_argument("--apply", action="store_true", help="save the cleaned labels (default: report only)")
    return parser.parse_args()


def specks(worm: np.ndarray, max_pixels: int, min_distance: float) -> tuple[np.ndarray, int]:
    """Mask of the specks to remove, and the largest component's size."""

    components, count = ndimage.label(worm)
    if count < 2:
        return np.zeros_like(worm), int(worm.sum())
    sizes = np.bincount(components.ravel())[1:]
    main = int(np.argmax(sizes)) + 1
    distance = ndimage.distance_transform_edt(components != main)
    remove = np.zeros_like(worm)
    for k in range(1, count + 1):
        part = components == k
        if k != main and sizes[k - 1] < max_pixels and distance[part].min() > min_distance:
            remove |= part
    return remove, int(sizes[main - 1])


def main() -> int:
    args = parse_args()
    store = CorpusStore(args.dataset_root)
    changed, slivers = [], []
    for record in store.records():
        image, label, _ = store.load(record.sample_id)
        worm = label == 1
        if not worm.any():
            continue
        remove, largest = specks(worm, args.max_pixels, args.min_distance)
        if largest < args.max_pixels:
            slivers.append({"sample_id": record.sample_id, "worm_px": int(worm.sum())})
        if not remove.any():
            continue
        cleaned = label.copy()
        cleaned[remove] = 0
        entry = {"sample_id": record.sample_id, "split": record.split, "removed_px": int(remove.sum()),
                 "specks": int(ndimage.label(remove)[1]), "revision": record.revision}
        if args.apply:
            with store.locked():
                saved = store._save(
                    record.recording, record.frame_index, image, cleaned, image_raw=store.load_raw(record.sample_id),
                    source_path=record.source_path, dataset_path=record.dataset_path,
                    label_source=record.label_source, flat_fielded=record.flat_fielded, split=record.split,
                )
            entry["new_revision"] = saved.revision
        changed.append(entry)
    print(json.dumps({"applied": args.apply, "labels": changed, "slivers_left_for_review": slivers}, indent=1))
    print(f"{len(changed)} labels, {sum(e['specks'] for e in changed)} specks, {sum(e['removed_px'] for e in changed)} px"
          f"{'' if args.apply else ' (dry run)'}; {len(slivers)} sliver-only labels to review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
