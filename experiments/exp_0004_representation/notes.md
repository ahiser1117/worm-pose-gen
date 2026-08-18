# EXP-0004 — Coordinate versus intrinsic proposal representation

## Hypothesis

A compact intrinsic tangent-angle representation reduces local angle error and
jitter relative to predicting 100 independent centerline coordinates, without
materially worsening point accuracy or throughput.

## Difference from baseline

Both variants use the same small 2-D encoder, normalization, training examples,
augmentation, optimizer budget, and symmetric orientation loss. Only the output
representation/reconstruction changes: direct normalized coordinates versus a
midbody anchor, body length, global orientation, and 16 smooth tangent-basis
coefficients.

## Data/split

Train on accepted candidate proxy frames from the two training recordings of
each frozen development fold plus deterministic development-profile Tier C
samples. Validate separately on the held-out development-recording candidates
and disjoint held-out Tier C geometry/appearance. Only the 24 independently
reviewed overlays form a qualified qualitative Tier B subset; the remaining
candidates are training/engineering evidence. Do not read the audited holdout.
EXP-0003 real-texture crops may be evaluated only as a separate candidate-
proxy-referenced stratum.

## Training/resource budget

- maximum steps/epochs: 1,200 optimizer steps per variant/fold, at most 20 epochs
- wall-time limit: 12 GPU minutes per run; six primary runs
- seed/repeat policy: seed 20260818 for all three folds; if any one-seed metric
  or practical effect lies within 10% of its gate, also run seeds 20260819 and
  20260820 on every fold and require every seed/fold to pass
- checkpoint cadence: every 300 steps; retain latest and best validation only
- expected GPU time: <=1.5 hours including smoke/benchmark
- expected external-storage use: <=3 GiB in checkpoints and experiment outputs
- early termination conditions: non-finite loss/gradient, source HDF5 open,
  split identity mismatch, or median Tier C error worse than a centerline-mean
  predictor after 300 steps

## Success criterion

- primary metric: fold-aggregated held-out Tier C tangent-angle MAE
- numeric practical-effect threshold: ACCEPT intrinsic only if angle MAE is at
  least 5% lower in the pooled three-fold paired comparison, improvement is
  positive on every fold, candidate-proxy median point error is no more than 5%
  worse on every fold, and CUDA throughput is at least 90% of the coordinate
  variant
- proposal reliability gate: selected variant must achieve held-out Tier C
  median point error <=4 px, p95 point error <=10 px, mean tangent-angle error
  <=8 degrees, and p95 per-frame angle MAE <=18 degrees on every fold, with no
  systematic shortcut/topology failure in frozen random/worst overlays before
  refinement or probabilistic phases begin
- variability/confidence rule: pair variants by identical validation seed/case;
  pool case-level errors only after reporting each fold. Use a deterministic
  2,000-resample within-recording paired frame bootstrap as diagnostic. It does
  not replace the all-fold rule. If the 5% effect decision changes across the
  bootstrap, mark INCONCLUSIVE and retain the simpler/faster variant
- pass/fail interpretation: representation preference is controlled Tier C
  plus candidate-proxy engineering evidence, never a manual-label real-accuracy
  claim

## Results

Pending. The first two-step CUDA smoke reached training after strict preflight
and both validation loaders, then failed before its first optimizer update
because `adaptive_avg_pool2d_backward_cuda` has no deterministic implementation
in the installed PyTorch. No metric or checkpoint from that invalid smoke is
used. The fixed-size encoder pool was replaced by an equivalent fixed average
pool before the experiment run; the rerun uses a fresh output directory.

## Figures

Pending random/worst overlays, angle-by-body-position, error distributions,
FOV-proximity error, and accuracy-throughput comparison.

## Runtime

Pending.

## Interpretation

Pending.

## Decision

INCONCLUSIVE

## Next experiment

If a proposal passes the reliability gate, compare centered 1/5/11-frame
context with that representation held fixed.
