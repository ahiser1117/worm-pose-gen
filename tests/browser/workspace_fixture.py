"""Synthetic app for the Workspace page's browser test (tests/browser/workspace.cjs); everything lives in the given temp root.

A lab library holds setup ``nir`` (20 fps, 2.5 µm/px) over ``<root>/recordings``
with a default mask model (an untrained segmenter) and a body model.
The recordings are a worm crawling right through a 320x240 view: the long
one leaves the view at the end and has a dark speck beside the body for a
few frames, so the analysis finds issues; the short one stays unanalysed.
Analyse runs the real pipeline on the CPU with a small fit schedule
(``analysis.analysis_params`` and ``analysis.analysis_command`` are patched,
as are fix jobs), and the body model is a stub whose A-P field rises from
left to right over the dark pixels.  Jobs never get a GPU: the app reports
one so Analyse is enabled, but the runner hands none out.
"""

import argparse
import json
from pathlib import Path
import sys

import h5py
import lightning as L
import numpy as np
import torch
import uvicorn

import worm_pose_gen.body_net
from worm_pose_gen import fixes, library
from worm_pose_gen.app import AppConfig, analysis, create_app
from worm_pose_gen.body_net import OUTPUTS
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import default_width_template, render_tube_segments
from worm_pose_gen.segmenter import SegmentationModule

HEIGHT, WIDTH, LENGTH = 240, 320, 150.0
SRC = str(Path(__file__).resolve().parents[2] / "src")
SMALL = {
    "stage_downsample": [2, 1], "stage_steps": [40, 40], "stage_lr_scale": [1.0, 0.3], "stage_point_stride": [2, 1],
    "crop_padding": 16, "crop_multiple": 8, "length_bounds_px": None, "width_bounds_px": None,
    "length_prior_px": LENGTH, "width_prior_px": 12.0, "default_length_px": LENGTH, "default_width_px": 12.0,
    "width_shape_prior_mean": [0.0] * 6,
}
CPU_PARAMS = {
    "checkpoint": None, "flat_field": False, "min_worm_pixels": 200, "slab": 8, "prior": "none", "compile": False, "init_workers": 0,
    "orient": True, "overrides": SMALL, "beam": 2, "jump_seeds": False,
}


def midline(t: int) -> np.ndarray:
    """Frame ``t``'s head-first midline: a travelling wave, the body moving right by 5 px a frame (the head is on the right)."""

    k = np.arange(16) / 15
    shape = 0.55 * np.sin(2 * np.pi * (1.3 * k) - 0.45 * t)
    curve = decode_centerline(np.concatenate((shape, [np.pi, LENGTH], [105 + 5 * t, HEIGHT / 2])))
    return curve


def write_recording(path: Path, frames: int, speck: range = range(0)) -> None:
    rng = np.random.default_rng(len(path.name) + frames)
    # A blunt head and a thin tail, so the fit can tell them apart.
    taper = np.exp(0.3 * np.linspace(1, -1, len(default_width_template())))
    template = torch.as_tensor(12.0 * default_width_template() * taper, dtype=torch.float32)
    stack = np.empty((frames, HEIGHT, WIDTH), dtype=np.uint8)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    for t in range(frames):
        curve = torch.as_tensor(midline(t), dtype=torch.float32)
        body = render_tube_segments(curve[None], template[None], HEIGHT, WIDTH)[0].numpy() >= 0.5
        image = np.where(body, 60.0, 200.0) + rng.normal(0, 3, (HEIGHT, WIDTH))
        if t in speck:
            image[(yy - 55) ** 2 + (xx - (90 + 5 * t)) ** 2 < 15 ** 2] = 70
        stack[t] = np.clip(image, 0, 255).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=stack)


class StubBodyNet(torch.nn.Module):
    """A body-field network for the synthetic recordings: the body is the dark pixels, A-P rises left to right."""

    def __init__(self) -> None:
        super().__init__()
        self.lags = ()
        self.device = torch.device("cpu")
        self.checkpoint_path = "stub.ckpt"

    def forward(self, images):
        frame = images[:, 0]
        low, high = frame.amin(dim=(1, 2), keepdim=True), frame.amax(dim=(1, 2), keepdim=True)
        dark = (frame < (low + high) / 2).float()
        ap = torch.linspace(1, 0, frame.shape[-1]).expand_as(frame)
        maps = {"mask": dark, "ap": ap, "head": torch.zeros_like(frame), "tail": torch.zeros_like(frame), "overlap": torch.zeros_like(frame)}
        stacked = torch.stack([maps[name] for name in OUTPUTS], dim=1).clamp(1e-4, 1 - 1e-4)
        return torch.log(stacked / (1 - stacked))


def cpu_command(module: str, argv: list[str]) -> list[str]:
    code = f"import sys; sys.path.insert(0, {SRC!r}); from worm_pose_gen.{module} import main; sys.exit(main({argv!r}))"
    return [sys.executable, "-c", code]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    root = args.root
    recordings = root / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)
    write_recording(recordings / "2026-03-14-01.h5", 40, speck=range(12, 16))
    write_recording(recordings / "2026-03-14-02.h5", 12)

    lab = root / "lab"
    library.write_setup(lab, "nir", name="NIR tracker", fps=20.0, pixel_size_um=2.5, recording_roots=[str(recordings)],
                        defaults={"mask": "lab:nir-hand284", "body": "lab:nir-body-lags3"})
    # An untrained segmenter: the frame view runs it for the network layers (analysis uses dark pixels).
    weights = root / "weights.ckpt"
    module = SegmentationModule(pretrained=False)
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams), "pytorch-lightning_version": L.__version__}, weights)
    inputs = library.make_inputs([], fps=20.0, pixel_size_um=2.5)
    library.write_model(lab, "nir-hand284", {"name": "nir-hand284", "kind": "segmenter", "setup": "lab:nir", "outputs": ["mask"], "inputs": inputs}, weights)
    library.write_model(lab, "nir-body-lags3", {"name": "nir-body-lags3", "kind": "body_net", "setup": "lab:nir",
                                                "outputs": ["mask", "ap", "head", "tail", "overlap"], "inputs": inputs}, weights)

    analysis.analysis_params = lambda app, setup, models, mask_source="segmenter": dict(CPU_PARAMS)
    analysis.analysis_command = lambda path, stages, params: cpu_command(
        "pipeline", ["--workspace", str(path), "--stages", ",".join(stages), "--params", json.dumps(params), "--device", "cpu"])
    fixes.fix_command = lambda path, spec: cpu_command("fixes", ["--workspace", str(path), "--run", json.dumps(spec), "--device", "cpu"])
    stub = StubBodyNet()
    worm_pose_gen.body_net.load_body_net = lambda path, device=None: stub

    app = create_app(AppConfig(
        workspaces_root=root / "workspaces",
        dataset_root=root / "cache", gpus=(0,), device="cpu", job_interval=0.2,
        lab_library=lab, library=root / "mine", dev=args.dev,
    ))
    app.state.app_state.runner.gpus = None  # a GPU for the UI's Run on, none for the jobs
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
