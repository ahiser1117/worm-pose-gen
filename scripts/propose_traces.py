#!/usr/bin/env python3
"""Store network-proposed traces for the body-field records of a store.

For each chosen sample the body-field network predicts its fields from the
record's context, a trace follows from them, and the trace is fit like a hand
trace; the result is stored beside the record's targets as a proposal
(``body_fields.propose``), for accepting or correcting in the app's Body
fields tab.  By default every unreviewed sample with a worm gets one, except
traced samples and those whose proposal is already current (same mask
revision and model); ``--samples`` picks samples, ``--force`` redoes current
proposals.

The network is ``checkpoints/body_net/best.ckpt`` unless ``--checkpoint``
names another.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from worm_pose_gen import body_fields
from worm_pose_gen.body_net import load_body_net
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "body_net" / "best.ckpt")
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def wanted(record, meta, model: str, args: argparse.Namespace) -> bool:
    if args.samples is not None:
        return record.sample_id in args.samples
    if not meta.get("has_body") or body_fields.is_stale(meta, record) or meta.get("fit_method") in body_fields.TRACE_METHODS:
        return False
    if body_fields.review_status(meta) != "unreviewed":
        return False
    proposal = meta.get("proposal") or {}
    current = proposal.get("mask_revision") == record.revision and proposal.get("model") == model
    return args.force or not current


def main() -> int:
    args = parse_args()
    store = SegmentationStore(args.dataset_root)
    module = load_body_net(args.checkpoint, args.device)
    summary: Counter[str] = Counter()
    for record in store.records():
        path = body_fields.field_path(store.root, record.sample_id)
        if not path.exists() or not wanted(record, body_fields.read_meta(path), module.checkpoint_path, args):
            continue
        meta = body_fields.propose(store, record.sample_id, module, device=module.device)
        proposal = meta["proposal"]
        summary[proposal["status"]] += 1
        print(json.dumps({"sample_id": record.sample_id, **{k: proposal.get(k) for k in ("status", "fit_iou", "overlap_px", "points")},
                          "current_fit_iou": meta.get("fit_iou")}), flush=True)
    print(json.dumps({"proposals": dict(summary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
