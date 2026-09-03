# Learned segmentation: bootstrap, label, fine-tune, repeat

The mask fit in [`MASK_FIT_EXPERIMENT.md`](MASK_FIT_EXPERIMENT.md) made the
segmentation mask the only evidence for the pose, so mask defects (clipped
thin tails, debris fused to the body, interior texture) now show up directly
as pose errors. This document describes the loop that replaces the
local-darkness threshold with a small fine-tuned network, and the tooling
that keeps labeling throughput high.

The loop is:

`bootstrap labels from the pipeline -> train -> label with the network proposing -> retrain from the last checkpoint -> ...`

Everything lives in this repository. Labels live on flv-c4's local disk.
Checkpoints live in a git-ignored local directory.

## 1. Model

`src/worm_pose_gen/segmenter.py` defines a U-Net whose encoder is
torchvision's ImageNet-pretrained ResNet-18 with the stem collapsed to one
input channel (the RGB stem weights are summed). Skip connections at strides
2 through 32 plus the raw input give full-resolution logits, which thin tails
need. It has `14.3M` parameters and processes one `732 x 968` frame in a few
milliseconds on the project GPU.

`SegmentationModule` is the Lightning wrapper: masked binary cross-entropy
plus soft Dice, IoU/Dice/precision/recall logged over valid pixels, AdamW
with the encoder at a quarter of the decoder's learning rate, and cosine
decay. `predict_probability(frame)` returns a `[H,W]` worm probability map
for any grayscale frame. `load_segmenter(path)` loads a checkpoint for
inference without touching the pretrained-weight download.

Labels use three values: `0` background, `1` worm, `255` ignore. Ignored
pixels contribute to neither the loss nor the metrics, which is what lets
bootstrapped labels be honest about what they do not know.

## 2. Dataset store

`src/worm_pose_gen/segmentation_dataset.py` keeps one `.npz` per labeled
frame under the dataset root (default
`/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/segmentation_v1`
on flv-c4's local ZFS pool) holding the flat-fielded frame the network sees,
the raw frame, and the label. `index.json` records recording, frame index,
split, label source, revision, save time, and label statistics for every
sample, and is rewritten atomically.

A sample's split is assigned when it is first saved, to whichever of train,
validation, and test is furthest below its `80 / 10 / 10` target, so the
proportions hold for a small set. The assignment is pledged in
`splits.json` next to the index, and that file is append-only: re-saving a
refined label bumps the revision and keeps the split, deleting a sample
keeps its pledge, and labeling the same frame again later gets the pledged
split back. A frame that has ever been validation or test therefore never
enters the training set. New pledges are balanced against every pledge ever
made, not just the samples present. All frames come from the same
recordings, so the test split measures generalization across frames, not
across recordings; a held-out recording is the stronger test when more
recordings become readable.

`SegmentationDataModule` serves training crops (`512 px`, half of them
centered on the worm) with flips, right-angle rotations, brightness and
contrast jitter, and noise, and serves validation and test frames whole.

## 3. Bootstrap labels

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/bootstrap_segmentation_labels.py --frames-per-recording 40
```

For uniformly spaced frames of each readable recording, the script
flat-fields the frame, runs the frozen local-darkness threshold, keeps the
largest component, fills narrow holes, and then marks as **ignore** every
pixel the evidence disagrees about: a two-pixel band around the boundary and
every pixel where the raw threshold and the cleaned component differ. That
removes exactly the debris and thin-end pixels the threshold gets wrong.
Frames with no plausible component (the empty ones) are skipped.
`--with-mask-fit` additionally ignores where the rendered tube from the mask
fit disagrees with the component, at about `20 s` per frame.

The first bootstrap wrote 114 samples from the three Section 7 recordings
(91 / 12 / 11), with a median `3.1%` of pixels ignored and 6 empty frames
skipped, in about `0.3 s` per frame.

## 4. Train and evaluate

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/train_segmenter.py
scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate_segmenter.py
```

Training writes `checkpoints/segmenter/best.ckpt` (highest validation IoU),
`last.ckpt`, and Lightning CSV logs under `logs/version_N/`; the directory
is git-ignored. Mixed precision, early stopping on validation IoU, and a
test pass with the best checkpoint are on by default. Without `--init` the
model starts from ImageNet weights with no worm exposure; with `--init
checkpoints/segmenter/best.ckpt` it warm-starts from the last model (the
optimizer and schedule restart either way).

Every run leaves `runs/<start time>.json`: arguments, git revision, the
fingerprint of the checkpoint it started from, the exact train, validation,
and test membership (sample id, label source, revision, save time), the CSV
log directory, the best checkpoint's fingerprint, epochs run, and the final
metrics. `train_summary.json` is a copy of the latest run's record.

Evaluation reports per-sample IoU, Dice, precision, and recall on the
validation and test splits, compares the network with the classical
threshold on the same frames, and separates the hand-refined samples from
the bootstrapped ones. Every evaluation is kept under
`checkpoints/segmenter/evaluations/<evaluation time>/`: `evaluation.json`
holds the time, git revision, checkpoint fingerprint (path, size,
modification time, SHA-256), the split membership at that moment, the
summary, and per-sample metrics with label and predicted pixel counts, and
the worst-sample overlay sheets sit beside it. One summary line per
evaluation is appended to `evaluations/history.jsonl`. `--note` stores a
free-text reason with the record. Bootstrapped labels are not truth, so the
number that matters is the hand-refined subset.

A frame labeled as all background scores 1 when the prediction is also
empty and 0 when the network paints anything, so the summary also reports
the number of false-positive pixels on empty-label frames separately.

### Plots

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/plot_segmenter_history.py
```

writes four figures to `checkpoints/segmenter/plots/`: training and
validation loss and validation IoU per epoch with one line per run,
median IoU of every saved evaluation over time (network and classical on
the hand-refined labels, network on all labels), per-sample IoU of the
newest evaluation with hand-refined and empty-label samples marked, and
the growth of the store by label source.

### First model (bootstrapped labels only)

Trained on the 91 bootstrapped training samples for 26 epochs (early stop,
about two minutes on the project GPU):

| Split | Samples | Network IoU median | Network IoU min | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Validation | 12 | 0.976 | 0.854 | 1.000 | 0.976 |
| Test | 11 | 0.980 | 0.941 | 1.000 | 0.980 |

Two caveats make these numbers a sanity check rather than a result. The
labels were derived from the classical threshold, so the classical baseline
scores IoU `1.000` on them by construction and the comparison is
uninformative until hand-refined labels exist. And the network's only errors
are extra pixels at the tail tips and boundary, exactly where the bootstrap
marked ignore, which is the behavior we want but cannot yet score. The
network is fast enough (a few milliseconds per frame) to propose in the app
without any perceptible delay.

## 5. Label with the app

```bash
scripts/project_env.sh uv run --no-sync --frozen python -m worm_pose_gen.label_app
```

Then open `http://127.0.0.1:8767`. The app opens the three default
recordings read-only (`--recording` overrides), loads
`checkpoints/segmenter/best.ckpt` if it exists, and caches one flat field
per recording under the dataset root.

Each frame arrives with proposals: the network mask (thresholded in the
browser at an adjustable cutoff), the classical component, and the raw
threshold, plus the saved label if the frame was labeled before. The editor
starts from the saved label, else the network, else the classical mask.

Throughput comes from keeping the hands on the keyboard:

| Keys | Action |
|---|---|
| `1` `2` `4` `5` | Replace the mask with the network, classical, threshold, or saved proposal. `Shift` unions, `Alt` intersects, `Ctrl` subtracts. |
| `3` | Replace with the mask-fit tube rendered from the current mask (about 20 s). |
| `H` `L` `D` `S` | Fill narrow holes, keep the largest component, grow, shrink. |
| `B` `E` `I` | Worm, erase, and ignore brushes. `[` `]` change size. |
| `Z` | Undo (40 steps). |
| `Space` or `Enter` | Save and move to the next frame. |
| `N` `P` | Next without saving; previous in history. |
| `F` `O` `0` | Toggle raw/flat-fielded view, cycle overlay opacity, fit view. |
| Wheel; right or `Shift` drag | Zoom; pan. |

The next-frame mode decides what you label. **Network-uncertain** draws a
handful of random unlabeled frames, runs the network on each, and picks the
one with the most pixels between `0.2` and `0.8` probability. That is the
frame the current model needs most. Random and sequential (with a stride)
are also available. A frame with no worm can be saved as all background
(the browser asks for confirmation), which teaches the network that debris
is not worm; the bootstrap skips such frames, so they only enter through
the app.

Saving records `label_source` as `network+manual` or `classical+manual`, so
evaluation can separate hand-refined labels from bootstrapped ones.

## 6. The loop

1. Bootstrap, train, evaluate. The first model learns the threshold's
   behavior with its systematic errors masked out.
2. Label in network-uncertain mode. Correct thin tails, cut fused debris,
   paint ignore over anything ambiguous. Save.
3. Retrain with `--init checkpoints/segmenter/best.ckpt`, evaluate, plot,
   and restart the app so it loads the new checkpoint.
4. Once the hand-refined validation IoU is high and stable, feed the
   network's probability map to the mask fit as its soft target and return
   to evaluating pose estimation.

## What is not established

- No hand-labeled frame exists yet, so no number in this document measures
  agreement with a human. The first model's metrics measure agreement with
  the bootstrapped labels only.
- Self-overlap and coils produce a merged mask however good the segmenter
  is; resolving them is the pose model's job.
- The three recordings share a rig and a year. A recording from another
  setup should be labeled before the network is trusted on it.
