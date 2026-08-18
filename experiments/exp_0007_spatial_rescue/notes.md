# EXP-0007 — Intrinsic spatial-bottleneck rescue

## Hypothesis

Increasing only the proposal encoder's retained spatial grid from 2x2 to 4x4
will correct the localization and body-scale regression-to-mean observed in
EXP-0004 while retaining intrinsic topology, reliability, and high throughput.

## Difference from baseline

The frozen baseline is EXP-0004 intrinsic. This experiment changes one
scientific factor: the fixed encoder pool changes from a 12x16-to-2x2 pool to a
12x16-to-4x4 pool. The intrinsic head, 16-coefficient basis, loss weights,
  training examples/profiles, optimizer, batch size, folds, normalization,
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
strata. `data_seed=20260818` fixes all synthetic identities across model-seed
repeats; only parameter initialization/training order changes for model seeds
20260819 and 20260820. Start with primary fold 2. Do not read source recordings
or the audited 2025 holdout.

## Training/resource budget

- maximum steps/epochs: 1,200 optimizer steps and 34 epochs per fold
- wall-time limit: 12 GPU minutes per run
- seed/repeat policy: fixed data seed 20260818 and primary model seed 20260818;
  for positive numeric thresholds only, repeat when
  `abs(value-threshold)/abs(threshold) <= 0.10`, then run model seeds 20260819
  and 20260820 on every fold and require every seed/fold to pass. Exact-contract
  or qualitative failures fail directly and do not trigger repeats
- checkpoint cadence: every 300 steps; retain periodic latest and best
  fully-visible Tier C angle checkpoint
- expected GPU time: <=45 minutes for primary runs and <=1.5 additional hours
  only if the near-gate repeat rule fires
- expected external-storage use: <=3 GiB
- early termination: non-finite values, identity/preflight failure, >1M
  parameters, wall-time limit, or step-300 fully-visible median point error not
  below the frozen fold-specific mean-centerline baseline. Hitting the wall-time
  limit before step 1,200 is `INCONCLUSIVE`, never acceptance

The executable baseline is the pointwise arithmetic mean, in generator order,
of the exact 512 `centerline_xy` targets returned by `SyntheticTierCDataset`,
including its deterministic camera crops. The artifact is a `[100,2]`
little-endian float32 NumPy `.npy` file written with `allow_pickle=False`.
Record both the file SHA-256 and the tensor SHA-256 over raw C-order bytes, plus
the ordered sample-seed manifest, construction metadata, and validation case
identities before model training. On the exact 43 fully-visible cases from the
128 held-out Tier C identities, choose only the better forward/reverse
correspondence and measure per-point Euclidean error after scaling to 968x732
original pixels. Evaluate an immutable checkpoint whose stored `global_step`
is exactly 300; record its digest and eliminate when its median error is greater
than or equal to the frozen baseline median before any resume.

## Success criterion

- primary practical effect: primary-fold fully-visible Tier C median point
  error at least 20% below EXP-0004 intrinsic's 116.92 px, while synchronized
  CUDA throughput is at least 90% of its 2,320 samples/s batch-32 result
- reliability: on every development fold, candidate-proxy median/p95 point
  error <=8/20 px and mean/p95-frame angle MAE <=15/30 degrees; fully-visible
  Tier C median/p95 point error <=4/10 px and mean/p95-frame angle MAE <=8/18
  degrees. Candidate-proxy mean endpoint error must be <=15 px per endpoint,
  median length error <=8%, and support Brier/ECE <=0.12/0.10. Tier C mean
  endpoint error must be <=8 px per endpoint, median length error <=5%, and
  support Brier/ECE <=0.06/0.05. Require exact FOV contract, zero failed
  inference, and no systematic topology/shortcut failure in frozen random and
  worst overlays
- expansion rule: fold 2 must first pass the executable step-300 baseline; run
  to 1,200 steps. Other folds run only if the primary fold passes every
  reliability gate at its best checkpoint
- variability: the all-fold rule is mandatory. The near-gate seed policy above
  applies independently and cannot be replaced by pooled confidence intervals
- interpretation: Tier C supports controlled geometry claims and candidate
  proxies support engineering consistency only; neither is manual truth

This is a geometry-only rescue. Even if every gate above passes, EXP-0007 alone
does not authorize temporal modeling. A subsequent preregistered advancement
experiment must first evaluate the frozen cropped-FOV visible/hidden/boundary
gates, orientation limitations, EXP-0006 candidate-proxy crop evidence, and the
full support contract.

Benchmark the best-checkpoint digest with the identical EXP-0004 harness,
float32 input semantics, physical GPU 0, batch size 32, 100 iterations, and
synchronization. Require at least 90% of the 2,320.38 samples/s intrinsic
reference and >20 fps end-to-end. Report batch-1 p50/p95, forward-only and
end-to-end throughput, preprocessing, peak memory, parameter count, full
environment identity, and checkpoint digest.

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

Only if the unchanged ordinary-frame gates pass, preregister a cropped/support/
orientation advancement gate. Temporal 1/5/11-frame context remains blocked
until that gate passes. Otherwise reject this rescue and diagnose a different
proposal formulation without opening the holdout.
