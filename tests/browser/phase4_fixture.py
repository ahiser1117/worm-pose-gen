"""Synthetic app for the browser workflow; all artifacts live in the supplied temp root."""
import argparse
from dataclasses import asdict
from pathlib import Path

import h5py
import lightning as L
import torch
import uvicorn

from tests.test_pose_viewer import _write_recording, _write_run
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.batch_fit import BatchFitConfig
from worm_pose_gen.pipeline import update_summary
from worm_pose_gen.segmenter import SegmentationModule
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
        masks = handle["/img_nir"][:] < 140
    workspace.set_masks(list(range(len(masks))), masks)
    config = BatchFitConfig(
        stage_downsample=(2,), stage_steps=(2,), stage_lr_scale=(1.,), stage_point_stride=(2,),
        compile_renderer=False, crop_padding=8, crop_multiple=8, length_bounds_px=None,
        width_bounds_px=None, length_prior_px=100., width_prior_px=10.,
        default_length_px=100., default_width_px=10.,
    )
    update_summary(workspace, {"fit_config": asdict(config), "mask_cleanup": {"min_worm_pixels": 1}})
    base = root / "base.ckpt"
    module = SegmentationModule(pretrained=False)
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams),
                "pytorch-lightning_version": L.__version__}, base)
    app = create_app(AppConfig(
        workspaces_root=root / "workspaces", recording_roots=(root,), poses_root=root / "runs",
        checkpoint=base, dataset_root=root / "cache", prior_cache=None,
        notes=root / "notes.json", gpus=(), device="cpu", job_interval=.1,
    ))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
