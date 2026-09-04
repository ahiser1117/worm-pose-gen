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
scripts/project_env.sh uv run --no-sync --frozen python scripts/train_segmenter.py --name hand_labels
scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate_segmenter.py
scripts/project_env.sh uv run --no-sync --frozen python scripts/plot_segmenter_history.py
```

Each training run gets its own directory under `checkpoints/segmenter/runs/`,
named by start time and `--name`, holding `best.ckpt` (lowest validation
loss), `last.ckpt` (final epoch), `metrics.csv` (per-epoch curves), and
`run.json`: arguments, git revision, the fingerprint of the checkpoint it
started from, the exact train, validation, and test membership (sample id,
label source, revision, save time), the checkpoint fingerprints, epochs run,
and the final metrics. The directory is git-ignored. Mixed precision and a
test pass with the best checkpoint are on by default.

A run ends when the validation loss (masked BCE plus soft Dice) has not
improved for `--patience 5` epochs, and the learning rate halves whenever
the loss stalls for `--plateau-patience 2` epochs, so the stop means the
model has converged rather than that a schedule tied to an epoch cap ran
out (`--epochs 300` is only a cap). Checkpoint selection uses the same
validation loss, not the thresholded IoU: IoU saturates within ten epochs
while the loss keeps falling, and runs selected on IoU produced models whose
background probability sat near `0.3`. The masks at threshold `0.5` were
fine, but nearly every pixel fell in the app's uncertain band. Selecting on
loss picks a calibrated model with the same or better masks.

`--train-labels manual|bootstrap|all` picks the training labels, hand-refined
by default; validation and test always use every label they hold. Without
`--init` the model starts from ImageNet weights with no worm exposure; with
`--init <checkpoint>` it warm-starts from that model (the optimizer and
schedule restart either way).

After training, the run's best checkpoint is scored against the currently
promoted `checkpoints/segmenter/best.ckpt` on the validation split, and
replaces it when its mean validation loss over the hand-refined validation
labels is lower. That is the quantity training selects and stops on, so a
better-calibrated model wins even when the thresholded masks tie; the mean
IoU of both is recorded alongside. The labeling app loads the promoted
file, so it always proposes from the best validated model.
`--promote` forces the copy, `--no-promote` skips the comparison, and every
decision, with both scores and checkpoint fingerprints, is appended to
`checkpoints/segmenter/promotions.jsonl` and stored in the run record.

Evaluation runs **every** `best.ckpt` and `last.ckpt` under `runs/`
(`--checkpoint` picks files instead). One invocation is a session, kept
under `checkpoints/segmenter/evaluations/<session time>/<run>__<best|last>/`.
Each `evaluation.json` holds the time, git revision, checkpoint fingerprint
(path, size, modification time, SHA-256), the run record it came from, a
fingerprint of the labels used, the split membership at that moment, the
summary, and per-sample IoU, Dice, precision, and recall against both the
network and the classical threshold, with label and predicted pixel counts.
The worst-sample overlay sheets sit beside it, and one summary line per
checkpoint is appended to `evaluations/history.jsonl`. `--note` stores a
free-text reason with every record of the session. Bootstrapped labels are
not truth, so the number that matters is the hand-refined subset; since
every validation and test label has been hand-refined, that is now the
whole held-out set.

A frame labeled as all background scores 1 when the prediction is also
empty and 0 when the network paints anything, and the summary reports the
false-positive pixels on empty-label frames separately.

### Plots

`scripts/plot_segmenter_history.py` writes five figures to
`checkpoints/segmenter/plots/`: training and validation loss and validation
IoU per epoch with one line per run; hand-refined IoU of every checkpoint in
the newest session, with the classical median as a reference; hand-refined
median IoU of each checkpoint across sessions; per-sample IoU of every
checkpoint in the newest session; and the growth of the store by label
source.

### Three-way comparison (September 3, 2026)

Three models were trained from ImageNet weights with identical settings
(seed 0, 40 epochs maximum, patience 12) on different training labels, and
every checkpoint was evaluated against the 21 validation and 20 test labels,
all hand-refined. Two frames in each split are empty. Median IoU over the
split, with the lowest IoU over non-empty frames in brackets; "beats" counts
frames where the network's IoU exceeds the classical threshold's.

| Training labels | Checkpoint | Epoch | Val IoU | Test IoU | Beats classical (val, test) | False-positive px on empty frames (median) |
|---|---|---:|---:|---:|---|---:|
| bootstrap only (91) | best | 7 | 0.918 [0.70] | 0.913 [0.77] | 11/21, 13/20 | 1977 |
| bootstrap only (91) | last | 20 | 0.807 [0.71] | 0.830 [0.76] | 7/21, 6/20 | 3115 |
| hand only (74) | best | 8 | 0.960 [0.86] | 0.967 [0.87] | 18/21, 19/20 | 0 |
| hand only (74) | last | 21 | 0.967 [0.86] | 0.974 [0.86] | 18/21, 18/20 | 538 |
| all (165) | best | 24 | 0.951 [0.90] | 0.953 [0.87] | 20/21, 17/20 | 0 |
| all (165) | last | 37 | 0.950 [0.88] | 0.958 [0.87] | 18/21, 18/20 | 2179 |
| classical threshold | | | 0.901 | 0.883 | | 1163 |

Three things follow. Bootstrap labels hurt: the bootstrap-only model barely
beats the threshold it was distilled from, and keeps getting worse after its
best epoch. Hand labels alone give the highest medians with a fifth fewer
training frames, and mixing the bootstrap labels in costs about one IoU
point at the median while making the worst non-empty frame slightly better.
Early stopping matters: every final-epoch checkpoint paints hundreds to
thousands of pixels on empty frames that its best checkpoint leaves clean,
so `best.ckpt` is the one to use. The hand-only best checkpoint was
promoted to `checkpoints/segmenter/best.ckpt` for the app.

The bootstrap training labels are now the weakest data in the store, and
either revising them in the app or training with `--train-labels manual` is
the better use of them.

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

The **Saved labels** panel lists every sample in the store, filtered by
source (bootstrap or hand-labeled), split, and recording. Clicking a row
opens that frame with its saved label in the editor, and the fourth
next-frame mode, **Browse saved labels**, makes Next and Save + next walk
the filtered list in order. Revising a bootstrap label this way is the
intended fix for the bootstrap's systematic errors: the frame keeps its
split, its revision goes up, and its source becomes `network+manual`, so it
leaves the bootstrap filter and counts as hand-refined in evaluation. The
held-out bootstrap samples are the most valuable to revise first, since
they grow the part of the evaluation that measures agreement with a human.

Saving records `label_source` as `network+manual` or `classical+manual`, so
evaluation can separate hand-refined labels from bootstrapped ones.

## 6. Run on an unseen recording

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/segment_video.py \
  --recording /store1/shared/all_data_raw/prj_aversion/2024-05-28/2024-05-28-02.h5 \
  --start 0 --frames 1200
```

`scripts/segment_video.py` reads a stretch of a recording in slabs,
flat-fields it with the cached per-recording correction, runs the promoted
checkpoint in batches (`--batch-size 16`, about `27 ms` per frame on the
project GPU including padding to full resolution), and writes an MP4 under
`checkpoints/segmenter/videos/` with the mask filled in magenta and
outlined in green, plus a JSON file of per-frame worm pixels, component
counts, and pixels outside the largest component. `--scale 0.5` halves the
output for sharing; `--show-uncertain` tints the `0.2` to `0.8` probability
band yellow.

The first minute of `2024-05-28-02`, a session four months after the newest
labeled recording, segmented cleanly with the hand-only model: every frame
had a worm, the median mask was `28k` pixels, and `211` of `1200` frames had
small extra components (median `0`, at most `3.6k` pixels outside the
largest) from debris. The same run exposed a calibration problem: that
checkpoint, chosen by validation IoU at epoch 8, put background probability
near `0.34`, so `96%` of pixels fell in the uncertain band, while the
all-labels checkpoint trained to epoch 24 put background near `0.09`. IoU at
threshold `0.5` was unaffected, but the app's network-uncertain frame
selection had been close to random and the probability map was useless as
a soft target.

Selecting and stopping on validation loss fixed it. The hand-label run
under that rule with a cosine schedule (`hand_labels_loss`, best at epoch
52 of 60, loss still falling at the cap) reached median IoU `0.973` on 25
validation and `0.975` on 23 test labels against `0.960` / `0.967` for the
IoU-selected model, and put background at `0.10` on the unseen recording.
The run under the plateau schedule with patience 5 (`hand_labels_plateau`,
100 hand labels, best at epoch 99 of 105) took the validation loss from
`0.751` to `0.236` at the same median IoU (`0.972` / `0.974`, worst
non-empty test frame `0.956`), was promoted on that loss, and on the unseen
recording puts background at `0.004` and worm at `0.999`, with `0.07%` of
pixels in the uncertain band and the same masks.

## 7. The loop

1. Bootstrap, train, evaluate. The first model learns the threshold's
   behavior with its systematic errors masked out.
2. Label in network-uncertain mode. Correct thin tails, cut fused debris,
   paint ignore over anything ambiguous. Save.
3. Retrain, evaluate, plot, and restart the app; the new model is promoted
   to the app automatically when its validation loss beats the previous one.
4. Once the hand-refined validation IoU is high and stable, feed the
   network's probability map to the mask fit as its soft target and return
   to evaluating pose estimation.

## What is not established

- The held-out labels are hand-refined by one person; there is no second
  annotator, so inter-annotator agreement is unknown and the IoU ceiling a
  human would reach is not measured.
- Self-overlap and coils produce a merged mask however good the segmenter
  is; resolving them is the pose model's job.
- The three recordings share a rig and a year. A recording from another
  setup should be labeled before the network is trusted on it.
