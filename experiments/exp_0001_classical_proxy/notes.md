# EXP-0001 — Classical high-confidence proxy baseline

## Hypothesis

Robust local-background subtraction followed by conservative component and
centerline quality checks can produce a useful set of easy, fully visible real
proxy labels despite cross-recording intensity shifts.

## Difference from baseline

This is the first scientific pose baseline. It uses no learned parameters and
must not be described as ground truth.

## Data/split

Uniformly sample at most 48 frames from each of the three development recordings
in `configs/split_manifest.json`. Do not open the audited-holdout recording.
Open one HDF5 reader at a time and read frames individually.

## Training/resource budget

- maximum steps/epochs: not applicable; deterministic classical processing
- wall-time limit: 20 CPU minutes
- seed/repeat policy: base seed 20260818; deterministic indices
- checkpoint cadence: not applicable
- expected GPU time: none
- expected external-storage use: <= 250 MiB under `datasets/worm_pose_gen/proxy_v1`
- early termination conditions: source write attempt, >1 reader, or a grossly invalid centerline convention

## Success criterion

- primary metric: conservative accepted-proxy yield per recording
- numeric practical-effect threshold: >=50% of sampled frames accepted in each
  development recording; every accepted pose has 100 uniformly spaced points,
  stays at least 3 px from the image boundary, has plausible 250–750 px length,
  has two stable endpoints, and has >=95% point support inside the foreground
  tube/dark-ridge agreement check
- variability/confidence rule: report per-recording Wilson intervals and all
  rejection reasons; inspect 24 deterministic random accepted cases plus every
  accepted case with the worst quality score
- pass/fail interpretation: ACCEPT only if no more than 2/24 random overlays
  are grossly off the anatomical midline and no worst case reveals a systematic
  topology defect; otherwise REVISE or REJECT

Foreground-tube and dark-ridge checks share preprocessing with the extractor;
they are internal QC, not independent accuracy evidence. The deterministic
human overlay audit is an independent qualitative check only. These proxies may
be training targets and engineering-consistency metrics, but never the sole
arbiter of scientific accuracy; Tier C controlled truth and clearly qualified
qualitative evidence remain separate.

## Results

Pending.

## Figures

Pending random, worst-case, rejection, and angle-profile evidence.

## Runtime

Pending.

## Interpretation

Pending.

## Decision

INCONCLUSIVE

## Next experiment

Use only accepted candidates as Tier B proxy labels; independently validate
geometry with the Tier C synthetic generator before any learned proposal.
