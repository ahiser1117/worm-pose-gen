#!/usr/bin/env python3
"""Add temporal context and body-field targets to every sample of a segmentation store.

For each hand-labeled frame this writes ``<root>/body_fields/<sample_id>.npz``
(the record is described in :mod:`worm_pose_gen.body_fields`).  Samples whose
stored record is unchanged (mask revision and ``max_lag``) are skipped unless
``--force``.  ``--review`` also renders one PNG per sample under
``body_fields/review/`` (A-P coloured over the frame, overlap in white, head
green, tail red) for checking the targets by eye.  Coiled and poorly fitting
frames are refit by a chain from neighbouring frames segmented with
``--segmenter`` (the promoted segmenter by default); ``--no-chain`` skips it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worm_pose_gen.body_fields import build, field_path, is_current
from worm_pose_gen.segmenter import load_segmenter
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore
from worm_pose_gen.temporal_context import MAX_LAG


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--max-lag", type=int, default=MAX_LAG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true", help="rebuild samples that are already current")
    parser.add_argument("--review", action="store_true", help="write a review PNG per sample")
    parser.add_argument("--samples", nargs="*", default=None, help="only these sample ids")
    parser.add_argument("--segmenter", type=Path, default=PROJECT_ROOT / "checkpoints" / "segmenter" / "best.ckpt",
                        help="segments the context frames for chain fits")
    parser.add_argument("--no-chain", action="store_true", help="fit every frame on its own")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = SegmentationStore(args.dataset_root)
    records = [r for r in store.records() if args.samples is None or r.sample_id in args.samples]
    todo = [r for r in records if args.force or not is_current(field_path(store.root, r.sample_id), r.revision, args.max_lag)]
    segmenter = None if args.no_chain or not todo else load_segmenter(args.segmenter, args.device)
    summary = build(store, todo, max_lag=args.max_lag, device=args.device, review=args.review, segmenter=segmenter,
                    on_built=lambda meta: print(json.dumps(meta), flush=True))
    print(json.dumps({"built": summary, "skipped_current": len(records) - len(todo)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
