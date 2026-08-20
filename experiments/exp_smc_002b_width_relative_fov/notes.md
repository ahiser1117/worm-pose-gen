# EXP-SMC-002B — width-relative FOV safety revision

## Status and evidence boundary

`COMPLETED_NOT_SUPPORTED`. The method and gates below were preregistered before
the run. This is a prospective development revision on the same 30-frame
primary annotation set used by EXP-SMC-002, not independent confirmation or
final validation. The protected 2025 holdout remained closed.

## Motivation observed before preregistration

EXP-SMC-002 diagnostics exposed false accepted naturally truncated frames whose
centerline boundary distances were approximately 20–23 px while their median
mask-derived widths were approximately 38–49 px. Those masks did not
necessarily touch the image boundary, so the existing fixed 13 px absolute margin
could treat an apparently simple truncated segment as a full-body anchor.

## Single changed factor

Freeze `min_boundary_clearance_widths=1.0`. After the centerline and width
profile are computed, reject the putative full-body anchor when:

```text
centerline_boundary_distance_px < 1.0 * median_width_px
```

The comparison is strict: equality passes this factor. All other mask-anchor
configuration values and every EXP-SMC-002 support threshold remain unchanged.
The detector is evaluated on the prospectively revised EXP-SMC-001B masks, so
this result measures the repaired upstream pipeline rather than isolating the
guard on the frozen parent masks. Coverage remains secondary to conditional
accepted-anchor reliability.

## Frozen decision gate

Among accepted complete anchors, require median per-frame point error ≤8 px,
95th-percentile per-frame point error ≤20 px, median tangent error ≤15°, median
endpoint error ≤15 px, and median length error ≤8%. At least 90% must
individually have median point error ≤8 px, none may have median error ≥20 px,
and at least 90% of naturally truncated frames must be rejected.

The revision must be evaluated without further threshold changes. A favorable
development result would justify later independent validation; it would not by
itself establish final anchor precision.

## Quantitative results

The paired repaired pipeline accepted 3/30 frames, all complete: 3/17 complete,
0/12 truncated, and 0/1 not-identifiable. The width-relative rule therefore
raised truncated rejection from 75% to 100%, passing the safety gate. The three
complete anchors had median point error 6.538 px, 95th-percentile frame error
11.997 px, median tangent error 5.029 degrees, median endpoint error 11.177 px,
and median length error 2.99%. However, only two of three were individually at
most 8 px (66.7% versus the required 90%); the remaining accepted frame was
12.603 px. Coverage fell to 10% overall and zero accepted anchors remained in
the 2023-10-11 recording. Serial CPU cost was 13.98 s for 30 central masks and
42.42 s for the 90 timeline extractions, excluding segmentation and I/O.

## Visual evidence

The [`accepted montage`](figures/accepted.png) shows two close curves and the
remaining 12.6 px accepted mismatch. The [`rejected montage`](figures/rejected.png)
shows that the guard now conservatively excludes visibly truncated cases, but
also removes many otherwise ordinary complete curves near an edge. The
[`timeline`](figures/annotated_windows_timeline.png) confirms very sparse
eligible anchors.

## Failure analysis

The width-relative guard fixes the specific false-completeness symptom but not
its upstream cause. Hysteresis changes the skeleton of the former 8.73 px
complete outlier and worsens it to 12.60 px; the retained sample is too small
and still contaminated for empirical posture, width, or dynamics fitting.
Perfect truncated rejection cannot compensate for conditional complete-anchor
precision below the preregistered threshold.

## Decision

`NOT_SUPPORTED` under the unchanged EXP-SMC-002 reliability gate.

## Consequence

Do not build the anchor-pose dataset, fit dynamics, or implement SMC on this
pipeline. Reopen only after a new segmentation method preserves terminals and
the complete-anchor gate is passed prospectively on development data, followed
by independent validation. The protected 2025 holdout remains closed.
