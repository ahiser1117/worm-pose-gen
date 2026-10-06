"""Synthetic app for the Paint browser check: a recording, a workspace on it, a manifest, and one saved corpus label."""
import argparse
import json
from pathlib import Path

import h5py
import uvicorn

from tests.test_pose_viewer import _write_recording, _write_run
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.corpus import CorpusStore
from worm_pose_gen.workspace import Workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    recording = root / "recording.h5"
    _write_recording(recording)
    run = root / "runs" / "baseline"
    _write_run(run, recording)
    workspace = Workspace.import_run(root / "workspaces", run, "demo")
    with h5py.File(recording) as handle:
        frames = handle["/img_nir"][:]
    workspace.set_masks(list(range(len(frames))), frames < 140)
    # Frame 1 of the manifest is already labeled, so the queue opens on frame 3.
    CorpusStore(root / "corpus").save_frame(recording, "/img_nir", 1, frames[1], (frames[1] < 140).astype("uint8"), split="val")
    (root / "queue.json").write_text(json.dumps({
        "name": "fixture queue", "recordings": {"rec": {"path": "recording.h5", "split": "val"}},
        "frames": [{"recording": "rec", "frame_index": i, "reasons": ["self_contact"]} for i in (1, 3, 5)]}))
    app = create_app(AppConfig(
        workspaces_root=root / "workspaces", recording_roots=(root,), poses_root=root / "runs",
        corpus_root=root / "corpus", checkpoint=None, dataset_root=root / "cache", prior_cache=None,
        notes=root / "notes.json", gpus=(), device="cpu", job_interval=.1,
    ))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
