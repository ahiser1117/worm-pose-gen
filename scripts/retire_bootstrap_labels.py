#!/usr/bin/env python3
"""Retire the bootstrapped labels from the segmentation store.

The classical-threshold bootstrap (``bootstrap_classical``) gave the first
segmenter something to learn from; the hand-refined labels have since
outnumbered them and training on them was shown to cost accuracy
(``docs/SEGMENTATION_LABELING.md``, three-way comparison).  This script moves
every sample whose label was never hand-refined out of the store: the
``.npz`` files and their index rows go to
``<root>/retired/<time>_bootstrap_classical/`` (with an ``index.json`` of the
rows), and the samples are deleted from the store.  Split pledges are kept,
as always, so a retired frame labeled again by hand lands in its old split.

``--dry-run`` lists what would move.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from worm_pose_gen.run_records import timestamp_slug, utc_now
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore, is_hand_labeled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = SegmentationStore(args.dataset_root)
    retiring = [r for r in store.records() if not is_hand_labeled(r.label_source)]
    by_split = {split: sum(r.split == split for r in retiring) for split in ("train", "val", "test")}
    print(f"{len(retiring)} bootstrapped labels to retire {by_split}; {len(store.records()) - len(retiring)} hand-refined labels stay")
    if args.dry_run or not retiring:
        for record in retiring:
            print(f"  {record.sample_id}  {record.split}  {record.label_source}")
        return 0
    archive = store.root / "retired" / f"{timestamp_slug(utc_now())}_bootstrap_classical"
    archive.mkdir(parents=True, exist_ok=False)
    rows = {}
    for record in retiring:
        shutil.copy2(store.sample_path(record.sample_id), archive / f"{record.sample_id}.npz")
        rows[record.sample_id] = asdict(record)
    (archive / "index.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
    for record in retiring:
        store.delete(record.sample_id)
    print(f"archived to {archive}; store now holds {store.counts()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
