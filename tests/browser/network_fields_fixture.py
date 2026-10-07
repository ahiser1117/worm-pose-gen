"""Synthetic app for the network-layers browser check: a fitted workspace and a stub body-field network.

The stub predicts a straight body along y=48 (x 12-116) with A-P rising with x,
the head at (30, 48) and the tail at (100, 48), whatever the frame.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import uvicorn

import worm_pose_gen.body_net
from tests.test_body_proposal import StubModule
from tests.test_pose_viewer import HEIGHT, WIDTH, _write_recording, _write_run
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.body_proposal import FieldPrediction
from worm_pose_gen.body_targets import point_heatmap
from worm_pose_gen.workspace import Workspace


def stub_prediction():
    mask = np.zeros((HEIGHT, WIDTH), np.float32)
    mask[40:57, 12:117] = 1
    ap = np.tile(np.linspace(0, 1, WIDTH, dtype=np.float32), (HEIGHT, 1))
    return FieldPrediction(mask=mask, ap=ap, head=point_heatmap((HEIGHT, WIDTH), np.array([30.0, 48.0]), 3.0),
                           tail=point_heatmap((HEIGHT, WIDTH), np.array([100.0, 48.0]), 3.0), overlap=np.zeros((HEIGHT, WIDTH), np.float32))


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
        workspace.set_masks(list(range(workspace.n)), handle["/img_nir"][:] < 140)
    checkpoint = root / "body_net.ckpt"
    checkpoint.write_bytes(b"stub")
    stub = StubModule(stub_prediction())
    worm_pose_gen.body_net.load_body_net = lambda path, device=None: stub
    app = create_app(AppConfig(
        workspaces_root=root / "workspaces", recording_roots=(root,), poses_root=root / "runs", corpus_root=root / "corpus",
        checkpoint=None, body_net=checkpoint, dataset_root=root / "cache", prior_cache=None, notes=root / "notes.json",
        gpus=(), device="cpu", job_interval=.1,
    ))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
