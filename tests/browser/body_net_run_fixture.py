"""Synthetic app for the Run panel's body-field network check.

Two workspaces imported from the same run: ``demo`` (fit without the network)
and ``networked`` (its fit summary records ``fit_params.body_net``), and an
app ``--body-net`` checkpoint file that exists.
"""
import argparse
from pathlib import Path

import h5py
import uvicorn

from tests.test_pose_viewer import _write_recording, _write_run
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.pipeline import update_summary
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
    checkpoint = root / "nets" / "best.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"stub")
    with h5py.File(recording) as handle:
        masks = handle["/img_nir"][:] < 140
    for name in ("networked", "demo"):
        workspace = Workspace.import_run(root / "workspaces", run, name)
        workspace.set_masks(list(range(workspace.n)), masks)
    update_summary(Workspace.open(root / "workspaces" / "networked"), {"fit_params": {"body_net": str(checkpoint)}})
    app = create_app(AppConfig(
        workspaces_root=root / "workspaces", recording_roots=(root,), poses_root=root / "runs", corpus_root=root / "corpus",
        checkpoint=None, body_net=checkpoint, dataset_root=root / "cache", prior_cache=None, notes=root / "notes.json",
        gpus=(), device="cpu", job_interval=.1,
    ))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
