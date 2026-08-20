# EXP-SMC-003 — low-dimensional tangent representation oracle

## Preregistered evidence boundary

This recasts the existing 17-complete-trace representation oracle under the
segmentation-anchored SMC plan without changing the earlier EXP-005 evidence.
It uses only the 17 complete primary development traces from one annotator;
12 truncated and one not-identifiable trace are excluded. The delayed repeat
measurement is unavailable, so no result may be called a human noise floor.
The protected 2025 holdout remains closed.

The test is an oracle projection of known traces. It does not train or evaluate
an image model, segmentation model, dynamics model, or particle filter.

## Frozen hypothesis and comparison

A compact tangent-angle representation plus four external pose values (2-D
translation, global rotation, and body length) can reconstruct complete traces
with negligible loss for later generative inference.

Test fixed cubic-spline and cosine tangent bases at K =
4/6/8/12/16/24/32. Test PCA only by leave-one-recording-out fitting, and mark a
requested K unsupported if it exceeds the rank available in any training fold.
In-sample PCA is deliberately excluded from selection.

## Frozen decision rule

A fixed representation is eligible only if median/p95 per-frame point error is
at most 1.00/1.25 px and median/p95 per-frame mean tangent error is at most
4.00°/4.50°. Select by smallest K, then lowest median tangent error, then
lowest median point error, then family name. PCA is diagnostic only because 17
traces cannot justify a learned posture basis.

A pass establishes oracle representation sufficiency only. It does not
authorize dynamics or SMC because EXP-SMC-001/002 upstream gates failed.

## Status

`SUPPORTED_ORACLE_ONLY`

## Results

Both fixed families first passed all four frozen reconstruction gates at
K=16. The selected cubic tangent spline had median/p95 per-frame point error
0.697/1.074 px and median/p95 per-frame mean tangent error 3.305°/4.307°.
The K=16 cosine basis also passed at 0.666/0.912 px and 3.358°/4.256°; the
preregistered tie-break chose cubic because its median tangent error was lower.

K=12 was insufficient under the tail-sensitive rule. Cubic K=12 reached
0.906/1.721 px and 3.782°/5.051°. Cosine K=12 reached 0.740/1.085 px and
3.613°/4.514°, narrowly exceeding the frozen 4.50° p95 tangent limit. K=24 and
K=32 passed but were not selected because K=16 was already sufficient.

Recording-held-out PCA remained a diagnostic rather than a candidate. Its
median point/tangent errors were 10.150 px/11.916° at K=4, 3.793 px/8.056° at
K=6, and 3.103 px/7.594° at K=8. Requested K=12/16/24/32 were unsupported:
the smallest leave-one-recording-out training-fold rank was nine, so those
dimensions would only relabel a capped lower-rank fit.

## Visual evidence

- [`representation_capacity.png`](figures/representation_capacity.png) compares
  both fixed bases and recording-held-out PCA across supported dimensions.
- [`selected_error_by_body_position.png`](figures/selected_error_by_body_position.png)
  shows the selected cubic K=16 reconstruction error along the body.
- [`metrics.json`](results/metrics.json) contains every per-case fixed-basis
  result, fold identities, unsupported-PCA declarations, hashes, and the
  executable decision.

## Decision and consequence

Choose a **16-coefficient fixed cubic tangent spline**, accompanied by 2-D
translation, global rotation, and body length, for any later generative pose
state. Do not learn a PCA posture basis from these 17 traces.

This result shows only that fixed low-dimensional geometry is not the current
bottleneck on known complete traces. It does not demonstrate image inference,
human-level precision, width-model fidelity, dynamics, or filtering. In
particular it does not override the `NOT_SUPPORTED` EXP-SMC-001/002 upstream
decisions: dynamics and SMC remain unauthorized.

Reproduce into a new experiment directory (the checked-in runner refuses to
overwrite this record):

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/exp_smc_003_representation_oracle.py \
  --experiment-dir /path/to/new/exp_smc_003_directory
```
