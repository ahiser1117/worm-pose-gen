"""Synthetic app for the Training page and the model picker (tests/browser/training.cjs).

Libraries from ``tests/test_model_training.make_library`` (lab setup ``rig``
with dataset ``base``, benchmark ``rig-v1``, models ``body`` and ``seg`` as
the defaults), plus a personal dataset ``copper`` extending ``base`` and a
personal fine-tune ``copper-ft`` with a loss curve and an evaluation.
``lab:seg`` has no evaluation, so the page queues one (a real CPU job).
Two training jobs are already there: one running that reports a growing
loss curve and waits to be cancelled, and one that failed.
"""

import argparse
import csv
from pathlib import Path
import sys

import uvicorn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))

from test_model_training import label_inputs, make_library, store_targets  # noqa: E402
from worm_pose_gen import library, model_eval  # noqa: E402
from worm_pose_gen.app import AppConfig, create_app  # noqa: E402
from worm_pose_gen.jobs import JobSpec  # noqa: E402
from worm_pose_gen.library.roots import write_json  # noqa: E402

RUNNING = """
import time
from worm_pose_gen.jobs import report_progress
curve = []
for epoch in range(1, 7):
    curve.append({"epoch": epoch, "train_loss": 1.2 / epoch ** 0.5, "val_loss": 1.0 / epoch ** 0.4 + 0.05})
    report_progress(0.1 + 0.8 * epoch / 40, f"epoch {epoch} done", {"phase": "train", "model_id": "copper-ft-2", "kind": "body_net",
                    "epoch": epoch, "max_epochs": 40, "curve": curve, "best_val_loss": min(c["val_loss"] for c in curve)})
    time.sleep(0.3)
time.sleep(3600)
"""
FAILED = "import sys; print('Traceback (most recent call last):'); print('RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB'); sys.exit(1)"


def personal_items(libraries, root: Path) -> None:
    copper = library.create_dataset(libraries, "copper", setup="lab:rig", extends="lab:base", name="Copper plates")
    write_json(copper.root / "splits.json", {"copper-1": "train", "copper-2": "test"})
    for recording, frame in (("copper-1", 3), ("copper-1", 9), ("copper-2", 4)):
        centerline, inputs = label_inputs(frame % 3)
        record = copper.save(recording=recording, frame=frame, origin="spread", orientation="manual", head_xy=[10.0, 32.0], **inputs)
        store_targets(libraries, record, centerline)
    metrics = root / "metrics.csv"
    with open(metrics, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "step", "train_loss_epoch", "val_loss"])
        for epoch in range(24):
            writer.writerow([epoch, epoch * 5, "", round(0.9 / (epoch + 1) ** 0.5 + 0.12 + 0.01 * (epoch > 15) * (epoch - 15), 4)])
            writer.writerow([epoch, epoch * 5 + 4, round(1.1 / (epoch + 1) ** 0.6 + 0.05, 4), ""])
    records = library.labels(libraries, ["mine:copper"], "train") + library.labels(libraries, ["mine:copper"], "val")
    library.create_model(libraries, "copper-ft", {
        "name": "copper-ft", "kind": "body_net", "setup": "lab:rig", "outputs": ["mask", "ap", "head", "tail", "overlap"],
        "inputs": library.make_inputs([1, 2], fps=20.0, pixel_size_um=2.0), "trained_on": library.trained_on(records),
        "parent": "lab:body", "hparams": {"max_epochs": 300, "learning_rate": 3e-4, "batch_size": 4, "lags": [1, 2], "min_fit_iou": 0.9},
        "notes": "Fine-tune on the copper plates; heads looked better on copper-1.",
    }, root / "body.ckpt", training_files={"metrics.csv": metrics}, training_records={
        "labels.json": {"train": [r.identity for r in records if r.split == "train"], "val": [r.identity for r in records if r.split == "val"]},
        "run.json": {"epochs_run": 24, "best_epoch": 16},
    })
    for model in ("lab:body", "mine:copper-ft"):
        model_eval.evaluate(libraries, model, "lab:rig-v1", device="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    libraries = make_library(root)
    personal_items(libraries, root)
    app = create_app(AppConfig(
        workspaces_root=root / "workspaces", dataset_root=root / "cache", gpus=(), device="cpu", job_interval=0.2,
        max_concurrent=4, lab_library=libraries.lab, library=libraries.personal,
    ))
    state = app.state.app_state
    # Jobs run on the CPU (the runner has no GPU and CUDA_VISIBLE_DEVICES is
    # empty), but the page must see a place to run them, as on a GPU node.
    state.config.gpus = (0,)
    runner = state.runner
    base = {"kind": "body_net", "setup": "lab:rig", "datasets": ["mine:copper"], "train": 4, "val": 1, "recordings": 3, "context": "none"}
    runner.submit(JobSpec(kind="train", gpus=0, label="Train copper-big", params={**base, "name": "copper-big", "start_from": "lab:body"}),
                  [sys.executable, "-c", FAILED])
    runner.submit(JobSpec(kind="train", gpus=0, label="Train copper-ft-2", params={**base, "name": "copper-ft-2", "start_from": "mine:copper-ft"}),
                  [sys.executable, "-c", RUNNING])
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
