# EXP-SMC-004 — empirical width model

## Hypothesis

A recording-level leave-one-frame-out mean width profile plus one bounded
frame scale explains easy-mask geometry nearly as well as one or two residual
PCA width coefficients, while a 10 px centerline error still causes a large
mask-overlap penalty after width-scale refitting.

## Rationale and evidence boundary

The width model must be expressive enough to render natural thickness but must
not absorb pose mistakes. This development oracle uses the manual centerline
only to measure width capacity against EXP-SMC-001B cleaned masks. It excludes
the expert-adjudicated hard cycle `2023-10-11-01-f013785`, uses no manual mask
truth, and cannot establish biological width accuracy. Models are fit
leave-one-frame-out within recording. The protected 2025 holdout remains
closed.

## Frozen comparison and gate

Compare fixed mean, mean times a scale constrained to `[0.8, 1.2]`, one width
PCA coefficient, two width PCA coefficients, and the per-point measured-width
oracle. Because anatomical head/tail identity is not consistently known, every
measured profile is averaged with its reversal before fitting. Prefer the scale
model if its median mask IoU is at least 0.80, PCA-2
improves median IoU by no more than 0.01, fitted scale standard deviation is at
most 0.15, and translating the true centerline by 10 px reduces median IoU by
at least 0.10 even after refitting scale.

## Status

`COMPLETED_SUPPORTED_PROXY_ONLY`.

## Quantitative results

Sixteen complete, non-hard development traces contributed 5/7/4 cases from the
three 2023 recordings. Median cleaned-mask IoU was 0.8676 for the fixed
recording mean, 0.8663 for mean times bounded scale, 0.8674 for width PCA-1,
0.8667 for width PCA-2, and 0.8709 for the per-point measured-width oracle.
PCA-2 improved over the scale model by only 0.00038 IoU. Fitted frame scale was
1.003 at the median with standard deviation 0.0245. After translating the true
centerline by 10 px and refitting scale, median IoU fell by 0.2041. All frozen
capacity and anti-compensation checks passed.

## Visual evidence

[`width_model_summary.png`](figures/width_model_summary.png) shows that every
additional width degree of freedom is nearly indistinguishable in mask IoU,
while the shifted-centerline penalty is well beyond its gate.
[`width_profiles_by_recording.png`](figures/width_profiles_by_recording.png)
shows stable tapered recording-specific profiles; isolated endpoint spikes are
measurement artifacts and are not retained by the mean.

## Failure analysis and decision

There is no width-capacity failure. More importantly, scale and PCA do not earn
their complexity over the fixed recording mean: even the per-point oracle gains
only 0.0033 median IoU. The scripted preregistered gate names the bounded-scale
model as supported, but the overarching least-complexity rule selects the fixed
recording-level mean for initial inference. A tightly constrained scale can be
reintroduced only if later observation-likelihood residuals justify it. Width
cannot hide a 10 px pose translation under this model.

## Consequence

Use the recording mean width profile in EXP-SMC-005 rendering and keep width
out of the first particle state. Proceed to controlled mask-likelihood basin
tests. This remains a single-annotator/manual-centerline oracle against
classical cleaned masks, not manual-mask or biological width truth. The
anti-compensation evidence is specifically a +10 px x-translation with scalar
scale refitting; rotation, shape, length, and PCA refitting require a broader
follow-up and are not claimed here.
