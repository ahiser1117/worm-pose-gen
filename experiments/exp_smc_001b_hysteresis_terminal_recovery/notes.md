# EXP-SMC-001B — connected terminal-recovery revision

## Hypothesis

The binary terminal omissions in EXP-SMC-001 are threshold/cleanup failures:
dim terminal pixels remain connected to the confidently segmented body and can
be recovered without admitting disconnected foreground or destabilizing mask
area.

## Scientific rationale

The parent audit retained a median 98% of the visible trace but only 80% of the
terminal samples. A two-level hysteresis rule directly targets that pattern.
The low threshold is frozen at 0.25, half of the original 0.5 threshold, before
this revision is run. Growth is seeded only from the original high-confidence
largest component and is 8-connected.

## Evidence boundary and frozen method

This is prospective development evidence on the same 30 primary annotations,
not independent validation. All EXP-SMC-001 score, morphology, source, and
sampling settings remain fixed. The protected 2025 holdout remains closed.
The probability map is unchanged and is still an uncalibrated classical score;
its trace mean is reported but deliberately not gated. Rescaling that score
would not demonstrate better foreground structure.

## Frozen structural gate

Require median/p10 visible-trace containment of at least 0.95/0.90 and median
terminal containment of at least 0.90. To reject uncontrolled leakage, require
the recovered area to be no more than 15% of the high-confidence component at
the median and 30% at the 95th percentile, and require the 95th percentile of
adjacent-frame relative area change to be no more than 0.05. No
non-identifiable annotation may provide quantitative evidence. A pass is
capped at `PARTIALLY_SUPPORTED` because there is no pretrained segmenter or
manual mask truth.

## Quantitative results

The revision again retained a median 0.980 of the visible manual trace (10th
percentile 0.960), but median terminal containment remained 0.800 and failed
the required 0.900. Hysteresis growth itself was small: median 0.97% and 95th
percentile 2.20% of the high-confidence component, both far below their 15%
and 30% caps. Adjacent-frame area change remained stable (95th percentile
1.44%). The unchanged soft trace score remained 0.738 and was diagnostic only.
Serial CPU time was 63.78 s for 30 central frames and 189.51 s for 89 unique
central/adjacent frames, excluding I/O and plotting.

## Visual evidence

The [`random montage`](figures/random_cases.png) confirms generally faithful
body occupancy, while the [`worst montage`](figures/worst_cases.png) shows that
the thin anatomical ends still under-reach the cyan manual traces. The low
threshold also made 17/30 cleaned masks contact the FOV and introduced some
diagonally connected fine structure, despite adding little total area. Thus
the recovered pixels were not preferentially the missing anatomical terminals.

## Failure analysis

The parent hypothesis was wrong for this repair: simple connected
lower-threshold growth does not recover the omitted ends. It instead adds small
amounts of low-confidence boundary/noise structure while leaving the median
terminal statistic unchanged. This distinguishes a genuine appearance/model
problem at the thin ends from a correct soft shape hidden just below one binary
threshold.

## Decision

`NOT_SUPPORTED` under the frozen structural gate.

## Consequence

EXP-SMC-001B exhausts the preregistered cleanup-level repair. A learned
foreground segmenter trained or fine-tuned against actual mask/terminal truth
is the minimum prerequisite for reopening the branch. Do not tune more
classical thresholds on these same 30 labels and do not authorize dynamics or
SMC from these masks.
