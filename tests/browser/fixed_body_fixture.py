"""Temporary recording and reviewed poses for the fixed-body browser check."""
import argparse
from pathlib import Path

import uvicorn

from tests.test_fixed_body import make_workspace
from worm_pose_gen.app import AppConfig, create_app

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
make_workspace(args.root)
uvicorn.run(create_app(AppConfig(
    workspaces_root=args.root / "workspaces", recording_roots=(args.root,),
    poses_root=args.root / "runs", dataset_root=args.root / "cache", checkpoint=None,
    notes=args.root / "notes.json", prior_cache=None, gpus=(), device="cpu", job_interval=.1,
)), host="127.0.0.1", port=args.port, log_level="warning")
