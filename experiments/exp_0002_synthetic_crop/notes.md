# EXP-0002 — Controlled synthetic geometry and crop benchmark

## Hypothesis

An analytic intrinsic-angle generator and differentiable tube renderer can
provide exact Tier C centerline/support truth and controlled 5–40% anatomical
head/tail censoring without introducing coordinate or boundary artifacts.

## Difference from baseline

Adds controlled synthetic truth; it does not measure real-image accuracy.

## Data/split

Generator seeds 20260818–20261329 are development; 20270000–20270127 are held
out. Development samples use 300–600 px lengths and moderate intrinsic bend
amplitudes; held-out samples use disjoint 250–299 or 601–700 px length bands and
the upper bend-amplitude band declared in the generator configuration. Exact
numeric parameter ranges and nuisance distributions are serialized with the
dataset before generation. No audited-holdout real recording is read. Required
hidden fractions are 5%, 10%, 20%, 30%, and 40% at both head and tail, including
temporally coherent sequences with smoothly moving crop boundaries.

## Training/resource budget

- maximum steps/epochs: generate <=512 development and 128 held-out samples
- wall-time limit: 15 CPU minutes
- seed/repeat policy: exact stored seed per sample
- checkpoint cadence: not applicable
- expected GPU time: none
- expected external-storage use: <=1 GiB under `datasets/worm_pose_gen/synthetic_v1`
- early termination conditions: non-finite geometry, irreversible transforms, or support-mask disagreement

## Success criterion

- primary metric: geometry/render/crop contract validity
- numeric practical-effect threshold: 100% finite 100-point centerlines; exact
  half-open FOV-mask agreement; exact requested hidden point counts; crop
  coordinate round-trip max error <=1e-5 px; renderer gradients finite and
  nonzero; no held-out sample with length outside 250–700 px
- variability/confidence rule: exhaustive contract checks over all generated
  samples plus deterministic random and most-curved montage
- pass/fail interpretation: ACCEPT the benchmark only if every exact invariant
  passes; visual plausibility is reported separately and does not turn synthetic
  evidence into real-image evidence

## Results

Pending corrected-environment rerun.

## Figures

Pending generator and crop-sequence montages.

## Runtime

Pending.

## Interpretation

Pending.

## Decision

INCONCLUSIVE

## Next experiment

Use the frozen Tier B proxy and Tier C generators for representation and
temporal-context comparisons.

Synthetic crops validate geometry and controlled hidden-body behavior. They do
not replace the required proxy-real crop benchmark that preserves real visible
texture/background, which is run only after accepted real proxy centerlines
exist.
