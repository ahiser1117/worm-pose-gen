# EXP-0003 — Real-texture controlled crop benchmark

## Hypothesis

Accepted, fully visible real proxy frames can be converted into reproducible
camera-window crops that preserve observed NIR texture/background and provide
exact coordinate transforms and independently defined anatomical support.

## Difference from baseline

EXP-0002 validated analytic geometry and simple rendered appearance. This
experiment applies the same evidence contract to the 90 accepted Tier B frames;
their centerlines remain proxy references rather than ground truth.

## Data/split

Read only `accepted_image`, accepted source indices, and corresponding accepted
centerlines from the immutable external `proxy_v1/proxy_labels.h5`. Do not open
any source recording or the audited holdout. Preserve recording identity so
downstream evaluation follows the three frozen whole-session folds. For each
usable frame, attempt head/tail crops at 5%, 10%, 20%, 30%, and 40% hidden.

## Training/resource budget

- maximum steps/epochs: not applicable; deterministic transformation
- wall-time limit: 10 CPU minutes
- seed/repeat policy: base seed 20260818; deterministic source order
- checkpoint cadence: not applicable
- expected GPU time: none
- expected external-storage use: <=2 GiB under `datasets/worm_pose_gen/real_crop_v1`
- early termination conditions: source-recording open, source/proxy overwrite,
  non-invertible transform, inconsistent support, or padded/painted crop pixels

## Success criterion

- primary metric: valid real-texture crop contract yield
- numeric practical-effect threshold: at least 60 accepted proxy frames permit
  all ten end/fraction crop conditions; every emitted crop is a direct
  axis-aligned subwindow of its stored real image, has no generated pixels,
  has max point transform round-trip error <=1e-5 px, and its stored support
  exactly equals independently recomputed half-open FOV membership
- variability/confidence rule: report usable count by recording and condition;
  inspect deterministic random, maximum-hidden, and rejected cases
- pass/fail interpretation: ACCEPT only as Tier B-referenced crop evidence; a
  pass does not establish anatomical accuracy of the proxy centerline

## Results

Pending.

## Figures

Pending real-texture crop montage and contract-yield plot.

## Runtime

Pending.

## Interpretation

Pending.

## Decision

INCONCLUSIVE

## Next experiment

Use accepted crop cases to stratify representation and temporal proposal
evaluation without confusing proxy agreement with manual-label accuracy.
