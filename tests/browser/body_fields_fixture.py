"""Synthetic app for the body-fields browser check: a corpus of three labels, two with body fields."""
import argparse
from pathlib import Path

import numpy as np
import uvicorn

from tests.test_body_fields import SHAPE, straight_worm, write_record
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.corpus import CorpusStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    store = CorpusStore(root / "corpus")
    *_, mask = straight_worm()
    image = np.where(mask, 60, 200).astype(np.uint8)
    for frame, iou in ((1, 0.97), (2, 0.6), (3, None)):
        record = store.save("rec", frame, image, mask.astype(np.uint8), source_path=str(root / "rec.h5"), label_source="manual")
        if iou is not None:
            write_record(store.root, record, fit_iou=iou)
    app = create_app(AppConfig(
        workspaces_root=root / "workspaces", recording_roots=(root,), poses_root=root / "runs",
        corpus_root=root / "corpus", checkpoint=None, dataset_root=root / "cache", prior_cache=None,
        notes=root / "notes.json", gpus=(), device="cpu", job_interval=.1,
    ))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
