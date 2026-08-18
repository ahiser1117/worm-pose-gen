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

Train on accepted Tier B proxy frames from the two training recordings of each
frozen development fold plus deterministic development-profile Tier C samples.
Validate separately on the held-out development recording proxies and disjoint
held-out Tier C geometry/appearance. Do not read the audited holdout. EXP-0003
real-texture crops may be evaluated only as a separate proxy-referenced stratum.

## Training/resource budget

- maximum steps/epochs: 1,200 optimizer steps per variant/fold, at most 20 epochs
- wall-time limit: 12 GPU minutes per run; six primary runs
- seed/repeat policy: seed 20260818 for all three folds; add one confirmation
  seed only if the relative primary effect lies within two percentage points of
  the decision threshold
- checkpoint cadence: every 300 steps; retain latest and best validation only
- expected GPU time: <=1.5 hours including smoke/benchmark
- expected external-storage use: <=3 GiB in checkpoints and experiment outputs
- early termination conditions: non-finite loss/gradient, source HDF5 open,
  split identity mismatch, or median Tier C error worse than a centerline-mean
  predictor after 300 steps

## Success criterion

- primary metric: fold-aggregated held-out Tier C tangent-angle MAE
- numeric practical-effect threshold: ACCEPT intrinsic only if angle MAE is at
  least 5% lower in the pooled three-fold comparison, it wins on at least two
  folds, Tier B proxy median point error is no more than 5% worse, and CUDA
  throughput is at least 90% of the coordinate variant
- proposal reliability gate: selected variant must achieve held-out Tier C
  median point error <=8 px, p95 point error <=20 px, mean tangent-angle error
  <=15 degrees, and show no systematic shortcut/topology failure in the frozen
  random/worst overlays before refinement or probabilistic phases begin
- variability/confidence rule: report per-fold and pooled bootstrap intervals;
  if the 5% effect decision changes under fold bootstrap, mark INCONCLUSIVE and
  retain the simpler/faster variant
- pass/fail interpretation: representation preference is Tier C plus Tier B
  proxy evidence, never a manual-label real-accuracy claim

## Results

Pending.

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
