# EXP-SMC-001 — classical soft-foreground development audit

## Preregistered evidence boundary

This experiment audits a classical soft dark-ridge foreground baseline. It is
not a pretrained segmentation network and there are no manual mask labels.
Consequently its strongest possible decision is `PARTIALLY_SUPPORTED`, even if
all diagnostic gates pass. Trace containment, trace probability, terminal
containment, and any tube/render overlap are centerline- or tube-derived proxy
quantities; they must not be named Dice, IoU against manual masks, or network
segmentation accuracy.

The audit uses exactly the 30 primary development annotations, no repeats, and
no 2025 recording. The source identity and explicit `/img_nir` dataset must
match the frozen records. The non-identifiable case is counted but supplies no
quantitative evidence. Central frames and their immediate neighbors are read
only for the declared stability/timeline diagnostics.

## Frozen gate

Diagnostic sufficiency requires all of: median cleaned-mask/manual-trace
containment at least 0.95; its per-frame 10th percentile at least 0.90; median
terminal containment at least 0.90; median soft probability sampled on the
manual trace at least 0.80; and zero use of non-identifiable evidence.

The run must also report connected components, area, holes, boundary contact,
nearest-mask distance, terminal omission, adjacent-frame area/centroid
stability, serial CPU runtime, and random/worst five-column visual evidence.
The plan-authorized cleanup freezes a 6 px closing radius and
`max_hole_area=1024 px` before the audit. These values were selected during a
pre-result visual plumbing check of the already-disclosed development frame
`2023-09-19-01-f003420`: the original 2 px close left texture holes that made a
visually simple worm topologically branched. This makes the 30-frame audit
development evidence rather than an independent validation set. The
configuration must not be tuned after the audit.

## Method

The frozen classical robust-dark-ridge score was converted to a logistic soft
map, thresholded at 0.5, closed by 6 px, reduced to its largest connected
component, and filled only for enclosed holes no larger than 1,024 px. The
serial CPU audit evaluated the 30 central development frames and the declared
immediate-neighbor stability diagnostics. The protected 2025 holdout remained
closed. Exact input/config digests and all per-case values are in
[`metrics.json`](metrics.json).

## Quantitative results

The cleaned mask contained a median 0.980 of the visible manual trace
(10th percentile 0.958), passing both whole-trace gates. Median terminal
containment was only 0.800, and median soft probability on the trace was
0.738; both failed their 0.900 and 0.800 gates. No non-identifiable annotation
was used quantitatively. Adjacent masks were stable (median relative area
change 0.00343 and median centroid displacement 1.70 px), and every central
cleaned mask had one retained component. Serial CPU segmentation cost was
60.85 s for 30 central frames and 180.54 s for 89 unique central/adjacent
frames, excluding HDF5 reads and plotting.

## Visual evidence

The [`random montage`](figures/random_cases.png) shows that most body material
is retained, while the [`worst-case montage`](figures/worst_cases.png) shows
systematic low confidence and binary-mask under-reach at thin terminals. These
are five-column raw/probability/raw-mask/cleaned-mask/overlay views, not manual
mask comparisons.

## Failure analysis

This is a localized segmentation/calibration failure, not a gross foreground
failure: central trace containment and adjacent-frame stability are strong,
but the last few anatomical samples are often weak or absent. Because the
method is an uncalibrated classical baseline and no manual masks exist, its
soft-probability number must not be interpreted as network calibration or
Dice/IoU. Crucially, terminal omission can make a truly truncated animal look
like a complete interior component to the anchor detector.

## Decision

`NOT_SUPPORTED` under the frozen diagnostic gate (with an evidence ceiling of
`PARTIALLY_SUPPORTED` even on a pass).

## Consequence

Do not authorize dynamics or SMC from this mask configuration. Run one
prospective development revision that grows the high-confidence component only
through connected lower-confidence foreground, targeting dim terminals without
admitting disconnected clutter. Retain the frozen result unchanged and keep
the protected holdout closed.
