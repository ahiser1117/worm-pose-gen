# EXP-0007 — Intrinsic spatial-bottleneck rescue

## Hypothesis

Increasing only the proposal encoder's retained spatial grid from 2x2 to 4x4
will correct the localization and body-scale regression-to-mean observed in
EXP-0004 while retaining intrinsic topology, reliability, and high throughput.

## Difference from baseline

The frozen baseline is EXP-0004 intrinsic. This experiment changes one
scientific factor: the fixed encoder pool changes from a 12x16-to-2x2 pool to a
12x16-to-4x4 pool. The intrinsic head, 16-coefficient basis, loss weights,
training examples/profiles, optimizer, batch size, folds, seed, normalization,
and evaluation cases remain unchanged. The head grows but must remain below one
million parameters.

Protocol-only corrections make the run auditable: fully-visible Tier C controls
the ordinary-frame checkpoint/gates; every epoch is written to CSV; checkpoint
identity is explicit; and the previously ambiguous early-elimination baseline
is frozen as a hashed pointwise arithmetic mean before training.

## Data/split

Use accepted candidate proxies from the two training recordings in each frozen
development fold plus the same 512 fold-specific development-profile Tier C
samples. Validate candidate proxies separately from the same 128 held-out
Tier C identities, partitioned into fully-visible and artificially cropped
strata. Start with primary fold 2. Do not read source recordings or the audited
2025 holdout.

## Training/resource budget

- maximum steps/epochs: 1,200 optimizer steps and 34 epochs per fold
- wall-time limit: 12 GPU minutes per run
- seed/repeat policy: primary seed 20260818; if any acceptance metric or effect
  lies within 10% of its gate, run seeds 20260819 and 20260820 on every fold and
  require every seed/fold to pass
- checkpoint cadence: every 300 steps; retain periodic latest and best
  fully-visible Tier C angle checkpoint
- expected GPU time: <=45 minutes for primary runs and <=1.5 additional hours
  only if the near-gate repeat rule fires
- expected external-storage use: <=3 GiB
- early termination: non-finite values, identity/preflight failure, >1M
  parameters, wall-time limit, or step-300 fully-visible median point error not
  below the frozen fold-specific mean-centerline baseline

The executable baseline is the pointwise float32 arithmetic mean, in generator
order, of the 512 fold-specific development Tier C centerlines. Its `[100,2]`
tensor SHA-256, construction metadata, and evaluation case identities are
written before model training. On the exact fully-visible subset of the 128
held-out Tier C cases, choose only the better forward/reverse correspondence and
measure per-point Euclidean error after scaling to 968x732 original pixels. At
step 300, eliminate when model median error is greater than or equal to the
frozen baseline median.

## Success criterion

- primary practical effect: primary-fold fully-visible Tier C median point
  error at least 20% below EXP-0004 intrinsic's 116.92 px, while synchronized
  CUDA throughput is at least 90% of its 2,320 samples/s batch-32 result
- reliability: on every development fold, candidate-proxy median/p95 point
  error <=8/20 px and mean/p95-frame angle MAE <=15/30 degrees; fully-visible
  Tier C median/p95 point error <=4/10 px and mean/p95-frame angle MAE <=8/18
  degrees; exact FOV contract, zero failed inference, and no systematic
  topology/shortcut failure in frozen random and worst overlays
- expansion rule: fold 2 must first pass the executable step-300 baseline; run
  to 1,200 steps. Other folds run only if the primary fold passes every
  reliability gate at its best checkpoint
- variability: the all-fold rule is mandatory. The near-gate seed policy above
  applies independently and cannot be replaced by pooled confidence intervals
- interpretation: Tier C supports controlled geometry claims and candidate
  proxies support engineering consistency only; neither is manual truth

## Results

Pending.

## Figures

Pending corrected random/worst overlays, diagnostics, baseline comparison, and
accuracy-throughput evidence.

## Runtime

Pending.

## Interpretation

Pending.

## Decision

PLANNED

## Next experiment

Only if the unchanged proposal reliability gates pass, compare 1/5/11-frame
temporal context. Otherwise reject this rescue and diagnose a different
proposal formulation without opening the holdout.
