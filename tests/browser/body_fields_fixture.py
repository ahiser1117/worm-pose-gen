"""Synthetic app for the body-fields browser checks: a corpus of three labels, two with body fields.

``--proposals`` also stores a network proposal on both records: the body traced
the other way round (head at x=90), as :func:`body_fields.propose` would write it.
"""
import argparse
from pathlib import Path

import numpy as np
import uvicorn

from tests.test_body_fields import SHAPE, straight_worm, write_record
from worm_pose_gen import body_fields
from worm_pose_gen.body_targets import render_body_targets
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.corpus import CorpusStore


def add_proposal(root, record, mask):
    centerline, widths, _ = straight_worm()
    reversed_line, reversed_widths = centerline[::-1].copy(), widths[::-1].copy()
    targets = render_body_targets(mask, reversed_line, reversed_widths)
    path = body_fields.field_path(root, record.sample_id)
    arrays, meta = body_fields.load(path)
    arrays.update(proposal_trace_xy=reversed_line[::12].copy(), proposal_centerline_xy=reversed_line,
                  proposal_width_profile=reversed_widths, proposal_ap=targets.ap.astype(np.float16),
                  proposal_overlap=targets.overlap, proposal_head_xy=targets.head_xy, proposal_tail_xy=targets.tail_xy,
                  proposal_diameter_px=np.float64(targets.diameter_px))
    meta["proposal"] = {"status": "ready", "fit_iou": 0.91, "overlap_px": 0, "points": 5, "model": "checkpoints/body_net/stub.ckpt",
                        "lags": [], "mask_revision": record.revision, "created_at": "2026-10-06T00:00:00+00:00"}
    body_fields.save(path, meta, arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--proposals", action="store_true")
    args = parser.parse_args()
    root = args.root
    store = CorpusStore(root / "corpus")
    *_, mask = straight_worm()
    image = np.where(mask, 60, 200).astype(np.uint8)
    for frame, iou in ((1, 0.97), (2, 0.6), (3, None)):
        record = store.save("rec", frame, image, mask.astype(np.uint8), source_path=str(root / "rec.h5"), label_source="manual")
        if iou is not None:
            write_record(store.root, record, fit_iou=iou)
            if args.proposals:
                add_proposal(store.root, record, mask)
    app = create_app(AppConfig(
        workspaces_root=root / "workspaces", recording_roots=(root,), poses_root=root / "runs",
        corpus_root=root / "corpus", checkpoint=None, dataset_root=root / "cache", prior_cache=None,
        notes=root / "notes.json", gpus=(), device="cpu", job_interval=.1, body_net=None,
    ))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
