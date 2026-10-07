"""Synthetic app for the Labeling page's browser check (tests/browser/labeling.cjs), on the CPU.

- a recording ``2024-05-05-01.h5`` of 60 frames: a dark, bending worm on a
  bright background;
- a lab library with setup ``lab:nir`` (the recording's root), default mask
  model ``lab:nir-mask`` and body model ``lab:nir-body``, and dataset
  ``lab:nir-labels`` with four labels whose body targets are built with
  fit IoUs 0.71, 0.93, 0.88 and (frame 35) not yet;
- an empty personal library, so the first save creates ``mine:nir-labels``;
- a workspace ``ws`` over the recording and a Relabel queue of frames 10, 20
  and 30 (its id is written to ``<root>/fixture.json``).

The models are stand-ins that know each frame's true body: the mask model's
probability is the blurred true mask, the body model predicts the true A-P
field and ends.  Body fits use the fast schedule, and the body-target job
after a save is a no-op.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy import ndimage
import uvicorn

from tests.test_body_proposal import prediction_from
from worm_pose_gen import body_fields, library
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.app import labeling as labeling_service
from worm_pose_gen.app.queues import relabel_queue
from worm_pose_gen.batch_fit import PRESETS
from worm_pose_gen.body_targets import render_body_targets
from worm_pose_gen.library.capture import read_label_inputs
from worm_pose_gen.library.targets import write_targets
from worm_pose_gen.workspace import Workspace

HEIGHT, WIDTH, FRAMES = 240, 320, 60
RECORDING = "2024-05-05-01"


def body(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame k's head-first midline (head at the left) and mask: a sine wave whose phase moves along the body."""

    x = np.linspace(60, 260, 80)
    y = HEIGHT / 2 + 28 * np.sin((x - 60) / 45 + k * 0.15)
    centerline = np.stack((x, y), 1)
    half = np.interp(np.linspace(0, 1, 80), [0, 0.15, 0.85, 1], [4, 8, 7, 2.5])
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    mask = np.zeros((HEIGHT, WIDTH), bool)
    for (cx, cy), r in zip(centerline, half):
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return centerline, mask


class StandIn:
    """A library model that answers from the true bodies, looked up by the frame's pixels."""

    def __init__(self, card, truth):
        self.card = card
        self.truth = truth

    def _true(self, context):
        return self.truth[hashlib.sha1(context[len(context) // 2].tobytes()).hexdigest()]

    def mask_probability(self, context, valid):
        _, mask = self._true(context)
        return np.clip(ndimage.gaussian_filter(mask.astype(np.float32), 1.5) * 1.1, 0, 1)

    def fields(self, context, valid):
        centerline, mask = self._true(context)
        return prediction_from(centerline, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    root = args.root
    recordings = root / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    frames = np.empty((FRAMES, HEIGHT, WIDTH), np.uint8)
    truth = {}
    for k in range(FRAMES):
        centerline, mask = body(k)
        image = np.where(mask, 70.0, 190.0) + rng.normal(0, 4, (HEIGHT, WIDTH)) + np.linspace(-10, 10, WIDTH)[None]
        frames[k] = np.clip(ndimage.gaussian_filter(image, 0.8), 0, 255).astype(np.uint8)
        truth[hashlib.sha1(frames[k].tobytes()).hexdigest()] = (centerline, mask)
    path = recordings / f"{RECORDING}.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=frames)

    lab, personal = root / "lab", root / "mine"
    video = {"flat_field": False}
    library.write_setup(lab, "nir", name="NIR test rig", video=video, fps=20.0, pixel_size_um=1.2, recording_roots=[str(recordings)],
                        defaults={"mask": "lab:nir-mask", "body": "lab:nir-body"})
    weights = root / "stand-in.ckpt"
    weights.write_bytes(b"stand-in")
    inputs = library.make_inputs([], fps=20.0, pixel_size_um=1.2)
    library.write_model(lab, "nir-mask", {"name": "nir-mask", "kind": "segmenter", "setup": "lab:nir", "outputs": ["mask"], "inputs": inputs}, weights)
    library.write_model(lab, "nir-body", {"name": "nir-body", "kind": "body_net", "setup": "lab:nir",
                                          "outputs": ["mask", "ap", "head", "tail", "overlap"], "inputs": inputs}, weights)
    libraries = library.Libraries(lab=lab, personal=personal)
    dataset = library.create_dataset(libraries, "nir-labels", setup="lab:nir", name="NIR labels", scope="lab")
    for frame, iou in ((5, 0.71), (15, 0.93), (25, 0.88), (35, None)):
        captured = read_label_inputs(path, frame, video=video, flat_field_cache=root / "cache")
        centerline, mask = body(frame)
        record = dataset.save(recording=RECORDING, frame=frame, mask=mask.astype(np.uint8), origin="migrated", **captured)
        if iou is not None:
            widths = np.full(len(centerline), 14.0)
            targets = render_body_targets(mask, centerline, widths)
            write_targets(libraries, record.sha256, "lab:nir-mask", {"has_body": True, "fit_iou": iou, "self_contact": frame == 25, "orientation": "nose"}, {
                "centerline_xy": centerline, "width_profile": widths, "ap": targets.ap.astype(np.float16), "overlap": targets.overlap,
                "head_xy": targets.head_xy, "tail_xy": targets.tail_xy, "diameter_px": np.float64(targets.diameter_px),
            })

    body_fields.fit_config = lambda: replace(PRESETS["fast"], length_bounds_px=body_fields.FIT_LENGTH_BOUNDS_PX)
    labeling_service.targets_command = lambda libraries, identities: [sys.executable, "-c", "pass"]
    Workspace.create(root / "workspaces", "ws", path, 0, FRAMES - 1)
    app = create_app(AppConfig(
        workspaces_root=root / "workspaces", recording_roots=(recordings,), poses_root=root / "runs", dataset_root=root / "cache",
        checkpoint=None, prior_cache=None, notes=root / "notes.json", gpus=(), device="cpu", job_interval=0.2, body_net=None,
        lab_library=lab, library=personal, dev=args.dev,
    ))
    state = app.state.app_state
    state.labeling.model = lambda ref: StandIn(library.get_card(libraries, ref), truth)
    queue = relabel_queue(state, "ws", [10, 20, 30])
    (root / "fixture.json").write_text(json.dumps({"relabel": queue["id"]}))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
