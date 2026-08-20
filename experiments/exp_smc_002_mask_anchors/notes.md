# EXP-SMC-002 — mask-native anchor audit

## Preregistered evidence boundary

Anchors are extracted from the frozen cleaned masks produced by EXP-SMC-001.
Complete traces are scored with orientation-symmetric pointwise curve metrics.
Truncated traces receive only one-way visible-trace diagnostics; hidden anatomy
and matched anatomical position are not claimed. Coverage is secondary to the
conditional reliability of accepted anchors.

## Frozen support gate

Among accepted complete anchors, require median per-frame point error at most
8 px, the 95th percentile of per-frame point error at most 20 px, median
per-frame tangent error at most 15 degrees, median endpoint error at most 15 px,
and median length error at most 8%. At least 90% of accepted complete frames
must individually have median point error at most 8 px, and none may have
median point error at or above 20 px. At least 90% of truncated frames must be
rejected. Missing accepted complete anchors cannot yield support.

Report acceptance by recording/selection stratum/trace state, rejection
reasons, skeleton topology QC, width profiles, trace-proxy mask/render overlap,
serial CPU runtime, accepted/rejected and worst-accepted overlays,
threshold-nearest review, and annotated ±1 easy/hard timelines. The
plan-authorized topology cleanup freezes `max_topology_spur_length=16 px`, while
the final curve endpoints must come from the unpruned skeleton. The mask and
anchor cleanup values were selected during the disclosed pre-result plumbing
check described in EXP-SMC-001, so this remains development evidence. The
configuration is frozen before results and must not be tuned afterward.

## Method

The frozen detector used strict assessed topology, retained the unpruned
longest skeleton path for geometry, resampled it to 100 points, measured width
by normal walks, and required the rendered tube to match the cleaned mask.
Every accepted complete trace was scored orientation-symmetrically; truncated
traces were used only for rejection and visible-trace diagnostics. The 30
central frames and declared plus/minus-one timelines were evaluated serially on
CPU. Exact per-case topology, geometry, rejection reasons, and config/input
digests are in [`metrics.json`](metrics.json).

## Quantitative results

The detector accepted 10/30 frames: 7/17 complete, 3/12 truncated, and 0/1
not-identifiable. On the seven accepted complete anchors, median frame point
error was 6.756 px (95th percentile 8.237), median tangent error 4.634 degrees,
median endpoint error 11.661 px, and median length error 3.46%. Six of seven
(85.7%) were individually at most 8 px, below the required 90%; no accepted
complete frame reached 20 px. Only 9/12 truncated frames were rejected (75%,
versus 90% required). Median trace-width-proxy rendered-mask IoU among accepted
frames was 0.955. Runtime was 16.11 s for 30 central masks and 48.21 s for the
90 timeline extractions, excluding segmentation and I/O.

## Visual evidence

The [`accepted montage`](figures/accepted.png) and
[`worst accepted montage`](figures/worst_accepted.png) show close geometric
alignment for accepted complete frames. The [`rejected montage`](figures/rejected.png)
shows that most complex or boundary cases are conservatively withheld, and the
[`timeline`](figures/annotated_windows_timeline.png) shows sparse anchor
availability. The three accepted truncated cases have skeleton boundary
distances 20, 21, and 23 px while their median measured widths are 38, 48.5,
and 46.75 px: the segmentation has dropped the true boundary-contacting
terminal before the fixed 13 px guard can see it.

## Failure analysis

Conditional easy-frame geometry is promising and substantially better than the
prior 12.12 px raw-pixel classical baseline, but anchor selection is not yet
safe. The primary failure is false completeness caused by segmentation terminal
omission. The complete-frame precision proxy also misses its gate by one small
outlier (8.73 px), so this result cannot authorize empirical priors or temporal
inference. Coverage is secondary and was not used to excuse either failure.

## Decision

`NOT_SUPPORTED` under the frozen conditional-reliability gate.

## Consequence

Preserve this audit and prospectively add a width-relative FOV guard: a
putative full-body skeleton must lie at least one median body width from the
image boundary. Re-evaluate the same labeled development set only as revision
evidence, without changing the remaining gates or opening the holdout. If that
revision remains unreliable, do not fit dynamics or implement natural SMC.
