# Architecture survey: results of the overnight experiments (2026-10-16 → 17)

Five proposals of [`MODEL_ARCHITECTURE_PROPOSALS.md`](MODEL_ARCHITECTURE_PROPOSALS.md)
were tested against the body-field network `mine:on-food-finetune-329`, each by
one agent on its own branch and GPUs, under one protocol
(`/temp_data4/alex/external_artifacts/arch_survey/PROTOCOL.md`); a combination of
the two winners was run afterwards. Everything is merged into
`model-architecture-survey`; every model is in the personal library (`arch-*`).

## Bottom line

- **Best model: `mine:arch-combo-self-fib`** (or **`mine:arch-combo-self-fib-tta4`**
  for the steadiest outputs at four network passes per frame): the baseline
  fine-tuned on hand labels plus pseudo-labels from the pipeline's own
  confident fits (half of every epoch) with fibre distractors drawn into half
  the hand labels. Against the baseline's three seeds it lifts the contact
  benchmark's worst frames from 0.69–0.74 to 0.89–0.90 IoU, cuts the pose
  pipeline's failing frames on the held-out contact clip from 303–332 to
  207–228, halves the tail jumps and lowers A-P jitter by a quarter (half with
  TTA), with A-P error on hand labels unchanged. Two seeds agree.
- **The "contact" failures were not self-contacts.** The worst frames of the b2
  benchmark and the downstream failures on `contact_0618_12` come from a
  straight dark fibre lying on the agar that the network segments as worm and
  joins to the head (still: `arch_survey/videos/summary/contact_0618_12_f14289.png`).
  No benchmark holds a held-out true self-contact, so no change here was shown
  to help (or hurt) real coils. Labeling held-out coil/contact frames is the
  missing measurement.
- **Within the training domain the baseline is already steady frame to frame**
  (0.65 % mask flicker, 2 head jumps over 11 clips); the pipeline's orientation
  flips were 1–3 over 4,500 frames for every model. Temporal improvements show
  up as lower A-P jitter and fewer tail jumps, not as fewer flips.
- **For a new domain the encoder matters most:** a DINOv3 ConvNeXt-T trained
  without the on-food recording (worm004) segments it far better than a
  ResNet-18 trained the same way (IoU 0.935–0.952 vs 0.871–0.912 over three
  runs each, worst frames 0.87–0.89 vs 0.61–0.81). It does not transfer the
  A-P field or the head any better, and in domain it is worse on the fibre clip.

## What was measured

- Hand labels: the three benchmarks (`lab:nir-v1` 13 labels, `mine:nir-flv-labels-b1` 43,
  `-b2` 66), all held out by the training dataset `mine:on-food-finetune`. b1 and b2 hold
  one empty label (2024-06-18-07 f19694); one stray pixel there scores IoU 0, so the
  table leaves it out of the IoUs and reports its stray pixels separately.
- Label-free temporal consistency (`scripts/evaluate_temporal.py`, `docs/temporal_eval_set.json`):
  11 clips (the seven sequence clips, three held-out stretches, one stretch of another
  project's single-worm recording); mask flicker, A-P jitter, head/tail jumps.
- The pose pipeline (fast preset, body-net masks) on the seven sequence clips and on the
  held-out/ood clips (`docs/heldout_sequence_set.json`); the sequence set is nearly saturated
  (3–14 frames below 0.9 for the baseline seeds), `contact_0618_12` is where models differ.
  The pipeline's IoU is against each model's own mask, so read it beside the hand labels.
- Seed noise: two more seeds of the baseline recipe (`arch-base-s1`, `-s2`).

## Results

b2/b1 IoU without the empty label; A-P error on v1/b1/b2; flicker, A-P jitter and jumps over the 11
temporal clips; contact = the pipeline on `contact_0618_12` (600 frames); seq = frames below 0.9 over
the seven sequence clips; flips and pose jumps over all 11 pipeline clips. Regenerate with
`arch_survey/summary/collect.py`.

| model | b2 IoU | b2 worst5 | b1 worst5 | A-P err v1/b1/b2 | empty px | flicker % | A-P jitter | head/tail jumps | contact median | contact <0.9 | seq <0.9 | flips | pose jumps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (seed 0) (`on-food-finetune-329`) | 0.971 | 0.735 | 0.972 | 0.076/0.053/0.061 | 0 | 0.65 | 0.0199 | 2/9 | 0.898 | 303 | 3 | 1 | 2 |
| baseline seed 1 (`arch-base-s1`) | 0.968 | 0.694 | 0.972 | 0.083/0.054/0.065 | 32 | 0.59 | 0.0214 | 5/42 | 0.824 | 332 | 14 | 3 | 6 |
| baseline seed 2 (`arch-base-s2`) | 0.971 | 0.741 | 0.973 | 0.076/0.048/0.058 | 0 | 0.70 | 0.0221 | 3/23 | 0.877 | 331 | 7 | 1 | 3 |
| ensemble: flip TTA (`arch-tta4`) | 0.974 | 0.759 | 0.978 | 0.078/0.053/0.060 | 0 | 0.51 | 0.0131 | 2/4 | 0.903 | 296 | 2 | 1 | 6 |
| ensemble: 3 seeds (`arch-ens3`) | 0.972 | 0.724 | 0.975 | 0.074/0.046/0.057 | 0 | 0.57 | 0.0161 | 2/11 | 0.886 | 314 | 7 | 1 | 2 |
| ensemble: 3 seeds x TTA (`arch-ens3-tta4`) | 0.973 | 0.743 | 0.980 | 0.076/0.048/0.056 | 0 | 0.45 | 0.0112 | 2/4 | 0.910 | 295 | 4 | 1 | 6 |
| semi-supervised (`arch-semi-ft5-s1`) | 0.969 | 0.701 | 0.978 | 0.070/0.043/0.053 | 0 | 0.64 | 0.0133 | 4/24 | 0.822 | 372 | 7 | 1 | 2 |
| semi-supervised seed 0 (`arch-semi-ft5`) | 0.969 | 0.704 | 0.976 | 0.070/0.049/0.059 | 187 | 0.70 | 0.0139 | 4/11 | 0.819 | 366 | 4 | 1 | 2 |
| self-train (`arch-self-ft`) | 0.977 | 0.850 | 0.972 | 0.082/0.062/0.065 | 0 | 0.46 | 0.0137 | 3/8 | 0.951 | 236 | 0 | 1 | 2 |
| self-train seed 1 (`arch-self-ft-s1`) | 0.974 | 0.790 | 0.974 | 0.071/0.051/0.057 | 0 | 0.48 | 0.0149 | 3/11 | 0.931 | 279 | 3 | 1 | 2 |
| self-train from scratch (`arch-self-scratch`) | 0.973 | 0.765 | 0.981 | 0.073/0.049/0.055 | 0 | 0.45 | 0.0114 | 2/32 | 0.941 | 268 | 1 | 1 | 4 |
| synthetic v3 + fibres (`arch-synth-v3-ft30-fib`) | 0.980 | 0.876 | 0.980 | 0.073/0.048/0.056 | 0 | 0.46 | 0.0182 | 3/26 | 0.946 | 252 | 6 | 1 | 2 |
| fibres only (`arch-synth-fibres-ft`) | 0.977 | 0.861 | 0.969 | 0.074/0.057/0.063 | 0 | 0.52 | 0.0193 | 2/6 | 0.950 | 242 | 4 | 1 | 2 |
| hand-only fine-tune control (`arch-synth-ft-control`) | 0.967 | 0.707 | 0.968 | 0.072/0.051/0.057 | 14 | 0.68 | 0.0207 | 2/7 | 0.837 | 354 | 5 | 1 | 2 |
| DINOv3 ConvNeXt-T (`arch-dino-t-c`) | 0.968 | 0.730 | 0.973 | 0.068/0.046/0.054 | 311 | 0.47 | 0.0140 | 2/141 | 0.814 | 341 | 6 | 1 | 2 |
| combo: self-train + fibres (`arch-combo-self-fib`) | 0.981 | 0.890 | 0.979 | 0.076/0.048/0.059 | 0 | 0.46 | 0.0154 | 2/4 | 0.955 | 208 | 7 | 1 | 2 |
| combo: self-train + fibres seed 1 (`arch-combo-self-fib-s1`) | 0.981 | 0.894 | 0.977 | 0.075/0.054/0.060 | 0 | 0.46 | 0.0154 | 3/6 | 0.952 | 228 | 2 | 1 | 2 |
| combo x flip TTA (`arch-combo-self-fib-tta4`) | 0.983 | 0.902 | 0.979 | 0.082/0.049/0.061 | 0 | 0.40 | 0.0106 | 2/2 | 0.954 | 207 | 7 | 1 | 2 |

Network speed (frames/s, batch 8, full 732×968 frames, A5500): ResNet-18 137; flip TTA 34;
3-seed ensemble 45; DINOv3 ConvNeXt-T 80.

## Per proposal

Each agent's write-up, with every run, ablation and reproduce commands, is
`arch_survey/<exp>/RESULTS.md`.

- **Ensembles and flip TTA (P9; `arch_survey/ensemble/`).** TTA is a real gain above seed
  noise (A-P jitter −35 %, b1/b2 IoU up, head/tail correct 1.0 everywhere) for 4× network
  time (≈ +30 % pipeline time); three seeds add little. Neither moves the fibre failures
  (every member makes the same mistake). Where three seeds disagree per pixel predicts the
  pipeline's failing frames on the contact clip before any fit (AUC 0.92–0.98) — a possible
  ambiguity cue, not wired in. Code: `body_ensemble.py`, `scripts/make_body_ensemble.py`.
- **Semi-supervised temporal losses (P1; `arch_survey/semi/`).** Mean teacher with
  mask-sandwich, A-P, head-continuity and flip-equivariance losses on 442 unlabeled
  segments from 38 recordings. A-P is a third steadier (about 20 % beyond an EMA-only
  control), masks/heads/pipeline unchanged, the fibre clip slightly worse. Most of the
  benchmark A-P gain comes from exporting the EMA weights, which ordinary training could do
  for free. Lessons: the teacher turns agar tracks into worm unless its non-largest
  components are negatives; teacher batch-norm statistics from unlabeled crops grow
  background blobs. Code: `unlabeled_clips.py`, `semi_supervised.py`, `--semi`.
- **Self-training from the pipeline (P2; `arch_survey/self-train/`).** 6,976 pseudo-labels
  (4,147 clear, 2,829 bridged by propagation, 652 with self-contact) from 28 workspaces on
  27 unlabeled recordings; tube-rendered targets, tail never trained from them; an audit of
  120 found no visible error. The first change that moved the fibre clip and b2, reproduced
  by a second seed; the hand-only control with the same schedule gains nothing. A second
  round did not help (a better teacher leaves fewer bridged stretches). Code:
  `pseudo_labels.py`, `scripts/build_pseudo_labels.py`, `--pseudo-labels/--pseudo-fraction`.
- **Synthetic coils and fibres (P3; `arch_survey/synthetic/`).** Synthetic worms alone made
  things worse (the model segments more of the fibre); generic fibre distractors on half
  the hand labels fix most of b2 and the fibre clip; with them, v3 synthetic worms add
  better b1 worst frames and lower A-P error. The fibres were added after seeing b2, so b2
  is no longer a blind test of them. Code: `synthetic_clips.py`,
  `--synthetic/--synthetic-fraction/--fiber-share`.
- **DINOv3 ConvNeXt encoder (P5/7; `arch_survey/dinov3/`).** See the bottom line; in domain
  it has the best A-P error but puts 180–970 stray pixels on the empty label and joins the
  fibre (contact 341 below 0.9); 1.7× slower. Weights under Meta's DINOv3 licence (research
  use; stated on the cards). Code: `segmenter.ConvNeXtUNet`, `--encoder`, an Encoder choice
  on the app's Train form, a `weight_decay` training parameter.

## Looking at it

- **In the app**: `scripts/project_env.sh uv run --no-sync --frozen python -m worm_pose_gen.app --workspaces-root /temp_data4/alex/external_artifacts/arch_survey/workspaces`
  (then http://127.0.0.1:8768/). Workspaces, each analysed with the named body model
  giving the masks (the A-P layer comes from the same model):
  - `0618-12_f13000-15999__<model>` — 2024-06-18-12 frames 13000–15999 (held out; the fibre
    stretch is at 14000–14600) for `on-food-finetune-329`, `arch-tta4`, `arch-semi-ft5-s1`,
    `arch-self-ft`, `arch-synth-v3-ft30-fib`, `arch-combo-self-fib`;
  - `0327-21_f8000-10999__<model>` — another project's recording (larger worm), same models;
  - `worm004_f1000-2999__<model>` — the on-food video for `arch-dino-ood-r18` and
    `arch-dino-ood-t` (neither trained on it) and the baseline (trained on it).

  Every `arch-*` model is also in the model picker for your own analyses.
- **Videos** (`/temp_data4/alex/external_artifacts/arch_survey/videos/`): `summary/` has the
  seven-panel comparison (baseline, flip TTA, semi, self-train, synthetic+fibres, DINOv3,
  combo) on `contact_0618_12`, `ood_0327`, `coil_0822`, `spiral_0528`; each panel is the
  frame with the A-P field (head blue to tail red), crossings hatched, the head (white dot)
  and tail (ring), and a strip marking A-P flips (red), head jumps (orange) and missing
  heads (grey). Each experiment's folder has its own videos, and `dinov3/worm004_1700.mp4`
  shows the cross-domain test.

## Code merged into `model-architecture-survey`

Additive throughout; existing checkpoints load unchanged and training without the new
options follows the old path. `run_tests.py --fast`: 169/169 classes.

- `temporal_eval.py`, `scripts/evaluate_temporal.py`, `scripts/render_field_comparison.py`,
  `docs/temporal_eval_set.json`, `docs/heldout_sequence_set.json` (measurement).
- `body_ensemble.py` (ensembles/TTA as one checkpoint, through `load_body_net`).
- `segmenter.ConvNeXtUNet`, `BodyFieldModule(encoder=...)`, `--encoder`, `weight_decay`.
- `pseudo_labels.py`, `synthetic_clips.py`, `unlabeled_clips.py`, `semi_supervised.py`, and
  `model_training.MixedSampler`, which mixes pseudo-labels and synthetic items into the
  hand-label epochs (the two experiments each had their own; one now serves both).
- A pipeline fix found on the way: `batch_fit.field_energies` crashed scoring a pose wholly
  off camera (the mirror of a body completed off camera), seen analysing worm004 with a
  weaker model.

## Suggested next steps

1. Label 30–50 held-out frames of true coils and self-contacts (none exist in any benchmark),
   and a few held-out frames with fibres or tracks, so each failure has its own test.
2. Promote `arch-combo-self-fib` (or its TTA version) as the setup's body model after looking
   at the workspaces; retrain it from the hand labels you add.
3. Export EMA weights in ordinary training (the cheap part of the semi-supervised gain).
4. For new rigs or conditions, try the DINOv3 encoder with the self-train and fibre recipe;
   the A-P transfer failure needs its own fix (scale augmentation is the first candidate).
5. Use seed or TTA disagreement as an ambiguity cue in linking.

## Housekeeping

- `mine:arch-ood-worm004` (dataset and benchmark) was made for the cross-domain test. As a
  benchmark it now scores every model, but every `on-food-finetune` model trained on those
  labels, so it is a ceiling for them; delete
  `/temp_data4/alex/worm-pose-library/benchmarks/arch-ood-worm004.json` to retire it.
- Dead-end models that can be deleted from the picker: `arch-synth-ft30`, `arch-synth-v2-*`,
  and the `arch-semi-*` runs other than `ft5`, `ft5-s1`, `scratch4`, `scratch4-s1` and the
  two EMA controls (`ft-emactl`, `ft-emactl-s1`) (see `arch_survey/semi/RESULTS.md`).
- Large caches: `arch_survey/semi/clips` (20 GB), `arch_survey/synthetic/items_*` (about 18 GB each),
  `arch_survey/self-train/{items,items_r2,ws}`.
- flv-c2 mounts data2 at `/data2`, not `/storage/fs/data2`; `arch_survey/bin/seq_eval.sh` fails
  there on the held-out clips (the ensemble agent's `ensemble/seq_eval_c2.sh` works around it).
