# EXP-0006 — Balanced real-texture crop benchmark

## Hypothesis

The valid EXP-0005 conditions can support a balanced, deterministic benchmark
covering every recording, anatomical end, and 5--40% fraction without changing
crop geometry, silently dropping a condition, or claiming the rejected
same-source-frame series succeeded.

## Difference from baseline

EXP-0005's experimental unit was a source frame with all ten valid conditions;
only 14/90 passed. EXP-0006 prospectively changes the unit to one valid crop
condition. From the frozen 720-case valid pool, it selects exactly 10 cases per
recording for each end/fraction cell: 3 recordings x 2 ends x 5 fractions x 10
= 300 cases. Selection rank is SHA-256 of the frozen seed, recording, frame, end,
and fraction, independent of model performance or proxy quality score.

## Data/split

Use only EXP-0005's immutable 900-case manifest and the immutable `proxy_v1`
accepted-image artifact. Verify both SHA-256 values. Never open source recordings
or the audited holdout. Regenerate only the selected direct source windows and
frozen isotropic resize, retaining exact original/source-window/output mappings.

## Training/resource budget

- maximum steps/epochs: not applicable; deterministic selection/materialization
- wall-time limit: 10 CPU minutes
- seed/repeat policy: seed 20260818; SHA-ranked deterministic selection
- checkpoint cadence: not applicable
- expected GPU time: none
- expected external-storage use: <=250 MiB in `real_crop_balanced_v1`
- early termination conditions: fewer than 10 valid cases in any recording/cell,
  input hash mismatch, output/source collision, generated pixels, transform or
  support mismatch, partial publication, or overwrite attempt

## Success criterion

- primary metric: balanced artifact contract
- numeric practical-effect threshold: exactly 300 completed cases, exactly 10
  per recording/end/fraction cell, zero duplicate condition identities, 100%
  hash/provenance/support/transform/interpolation checks, <=1e-4 px transform
  round trip, complete atomic HDF5 marker, and <=250 MiB output
- variability/confidence rule: contract is exhaustive over all selected cases;
  report source-frame reuse and inspect deterministic random, all 40%-hidden,
  per-recording, and min/max-scale examples
- pass/fail interpretation: ACCEPT establishes a candidate-proxy-referenced
  static real-texture crop benchmark only. It does not restore the rejected
  same-frame claim, establish anatomical accuracy, or supply real temporal truth

## Results

Pending.

## Figures

Pending selection/balance and evidence figures.

## Runtime

Pending.

## Interpretation

Pending.

## Decision

INCONCLUSIVE

## Next experiment

Use this artifact as a frozen candidate-proxy crop stratum for model diagnostics,
while all quantitative hidden-anatomy claims continue to rely on Tier C truth.
