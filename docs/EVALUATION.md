# Evaluation protocol

This protocol was frozen after the bounded data audit and before model
development. It keeps engineering agreement on real images (Tier B) separate
from controlled synthetic truth (Tier C). There is currently no Tier A manual
ground-truth set; consequently, no result from this project should be described
as manual-ground-truth accuracy. Manual annotation added later will be a new,
separately reported Tier A evaluation with gates declared before its test labels
are examined.

The numerical gates below use the audited 14–25 px rough width range and four
readable whole-session groups. They are frozen before any learned-model result.
They must not be changed in response to validation or final-test performance.
If later evidence makes a gate ill-posed, the project reports it as inapplicable
rather than silently reinterpreting it.

## Geometry and metric conventions

`(0, 0)` is the center of the upper-left pixel, x increases right, and y
increases down. A point is in the FOV exactly when `0 <= x < width` and
`0 <= y < height`. Angles are `atan2(dy, dx)` in radians wrapped to
`[-pi, pi)`. Curvature is `d(theta)/d(arc length)` in radians per
original-image pixel; positive curvature turns clockwise on screen. Exported
poses are uniformly sampled in arc length (nominally 100 samples) and ordered
head to tail. After canonicalization, `head_tail_probability` is the probability
that exported index 0 is the head and must ordinarily lie in `[0.5, 1]`.

Centerline error is the Euclidean point error at matched anatomical samples.
Angle MAE uses the absolute shortest circular difference and is reported in
degrees. Visible and hidden strata come only from the reference observable-
support target, never from model-predicted support. `in_fov_mask` is geometric
point membership and is not substituted for image support. Boundary results use
the five body samples on either side of each support transition. Also report
error versus normalized body coordinate, endpoint error, length error, width-
and length-normalized point error, flip rate, and random/worst-case examples.

## Leakage-safe splits

The executable frozen manifest is `configs/split_manifest.json`. It records each
configured/resolved path, size/mtime identity, explicit half-open frame range,
and grouping key. The three complete readable 2023 sessions form a leave-one-
session-out development cross-validation scheme. Fold 2 (2023-09-19 and
2023-09-27 train; 2023-10-11 validate) is the primary cheap-elimination fold,
but an accepted model-selection result must pass all three frozen folds.

The 2025-03-06 Hamamatsu-condition recording is an **audited holdout**, not a
pristine final test: Phase 1 necessarily inspected 32 declared frames before
the split existed, and its appearance/intensity/rough scale informed the audit.
Those exact indices are listed in the manifest and excluded from post-freeze
evaluation. No other frame from that recording may be read until architecture,
checkpoint, calibration, and thresholds are frozen. The one-time result tests a
known session/camera-condition shift but cannot establish independent-background
generalization. The eight unreadable recordings remain quarantined.

The split manifest records source identity, all grouping keys, allocation,
random seed (base seed 20260818), temporal ranges, guard intervals, and rationale.
No neighboring temporal frames cross folds. Development validation may be
inspected during model selection. The unaudited holdout frames, proxy labels,
aggregate metrics, and examples remain untouched until architecture, checkpoint,
calibration mapping, and thresholds are frozen. The audited holdout is evaluated
once only if development selection succeeds; subsequent changes create a new
study rather than another attempt on it. Development selection ultimately
failed in EXP-0007, so that one-time evaluation was not authorized or run.

## Evidence tiers and numeric gates

Tier B consists only of easy, fully visible real frames for which classical
processing and at least one meaningfully independent check agree. These are
proxy labels, not truth. Each report must disclose shared preprocessing, code,
assumptions, and training labels between the candidate, proxy generator, and
independent check. Tier B agreement supports engineering consistency only.

On the held-out Tier B proxy set, the candidate must meet all of:

| Metric | Tier B gate |
|---|---:|
| Median centerline point error | <= 8.0 px |
| 95th-percentile centerline point error | <= 20.0 px |
| Mean tangent-angle circular MAE | <= 15.0 deg |
| 95th-percentile per-frame angle MAE | <= 30.0 deg |
| Mean endpoint error (head and tail reported separately) | <= 15.0 px each |
| Median absolute body-length error | <= 8.0% |
| Exact recomputed `in_fov_mask` agreement | 100% |
| Hard head/tail flip rate | <= 10.0% |
| Head/tail Brier score | <= 0.15 |
| Image-support Brier score | <= 0.12 |
| Image-support expected calibration error (10 fixed bins) | <= 0.10 |

Tier C uses analytically known centerlines, synthetic worms/warps, and controlled
FOV truncation. It does not establish real-image accuracy. On held-out generator
seeds and parameter combinations absent from training, the candidate must meet:

| Metric | Tier C gate |
|---|---:|
| Median centerline point error, fully visible | <= 4.0 px |
| 95th-percentile centerline point error, fully visible | <= 10.0 px |
| Mean tangent-angle circular MAE, fully visible | <= 8.0 deg |
| 95th-percentile per-frame angle MAE, fully visible | <= 18.0 deg |
| Mean endpoint error | <= 8.0 px each endpoint |
| Median absolute body-length error | <= 5.0% |
| Exact recomputed `in_fov_mask` agreement | 100% |
| Hard head/tail flip rate | <= 3.0% |
| Head/tail Brier score | <= 0.08 |
| Image-support Brier score | <= 0.06 |
| Image-support ECE (10 fixed bins) | <= 0.05 |

For both tiers, every ordinary-frame gate must pass overall and in each audited
visibility/bend stratum containing at least 100 frames. Smaller strata are
reported with intervals but cannot independently pass or fail a gate.

## Required cropped-FOV benchmark

Starting from reference full-body centerlines, create fixed-coordinate artificial
camera windows or evaluator support masks that hide exactly 5%, 10%, 20%, 30%,
and 40% of anatomical samples, separately at the head and tail. Use the same
fractions in temporally coherent sequences with a smoothly moving boundary.
Preserve visible texture/background and avoid artificial edge cues. Save crop
origin, size, original-to-crop transform, source identity, frame index, hidden
end/fraction, and the reference support mask so predictions map exactly back to
original coordinates.

At every required fraction and for both ends, report visible, hidden, and
five-sample boundary-band point/angle errors separately. Compare against the
declared simple single-frame baseline on identical cases. Gates are:

| Metric | Tier B proxy crop gate | Tier C synthetic crop gate |
|---|---:|---:|
| Visible angle MAE increase vs fully visible | <= 5.0 deg | <= 3.0 deg |
| Visible point-error increase vs fully visible | <= 5.0 px | <= 2.0 px |
| Hidden angle MAE at 40% hidden | <= 35.0 deg | <= 25.0 deg |
| Hidden point error at 40% hidden | <= 40.0 px | <= 20.0 px |
| Boundary-band angle MAE at every fraction | <= 25.0 deg | <= 15.0 deg |
| Hidden point-error improvement over baseline | >= 5% at every fraction; >= 10% mean | >= 10% at every fraction; >= 15% mean |
| Hidden angle-MAE improvement over baseline | >= 5% at every fraction; >= 10% mean | >= 10% at every fraction; >= 15% mean |
| Support Brier / ECE | <= 0.12 / 0.10 | <= 0.08 / 0.06 |

Improvement is `(baseline_error - candidate_error) / baseline_error`; the
baseline is frozen before candidate results are viewed. A method cannot hide a
bad fraction behind an average. Sequence experiments additionally report angle
jitter and jerk, but those are diagnostic until the audit supplies a defensible
physical time/pixel scale.

## Uncertainty and orientation calibration

If angle uncertainty is emitted, report empirical coverage at 50%, 80%, and
95%, error versus predicted uncertainty, circular negative log likelihood for
the documented distribution, and calibration stratified by visible fraction,
boundary proximity, and bend magnitude. At each nominal level, absolute
coverage error must be <= 10 percentage points for Tier B and <= 7 points for
Tier C, both overall and for visible/hidden strata with at least 100 samples.
Marginal angular intervals do not justify claims about joint hidden-position
uncertainty.

For `head_tail_probability`, report reliability diagrams, Brier score, ECE,
hard flip rate at probability 0.5 before export canonicalization, and ambiguity
strata. A failure detector, if used instead of meeting the flip gate, must have
at least 95% sensitivity and 90% specificity on Tier B, and 98%/95% on Tier C,
at a threshold frozen on validation data.

## Variability and pass/fail rule

Metrics are computed per frame, but decisions respect the whole-session unit.
Report every one of the three frozen development folds separately and require
every fold to pass an acceptance gate. A deterministic 2,000-resample within-
recording frame bootstrap may describe sampling uncertainty but is explicitly
diagnostic; it is not a substitute for a group interval with only one held-out
session per fold. Exact bounds agreement must have zero failures. All evidence
is labeled limited because there are only three development groups in one
project family.

Use at least three training seeds whenever a one-seed estimate lies within 10%
of a gate or practical-effect threshold. In that case, each seed and fold must
pass and the across-seed mean is reported with its range. One seed on the primary
fold is sufficient only for early elimination; acceptance requires all folds.
Missing values, excluded frames, and failed inference count are always reported
and may not be silently dropped.

## Performance and stopping evidence

End-to-end offline inference must exceed 20 frames/s on physical CUDA device 0
at a documented batch size. Report batch-1 median and p95 latency separately,
aggregate throughput, preprocessing/refinement time, peak GPU memory, parameter
count, and environment identity. A complexity addition is accepted only if it
passes all relevant accuracy gates and improves at least one primary error by
5% (lower 95% interval bound > 0) or cropped hidden-body error by the tier gate,
without reducing throughput by more than 10% or moving below 20 frames/s.
Otherwise prefer the simpler Pareto candidate.

Every decision includes aggregate distributions plus random, worst-case,
endpoint, tight-bend, boundary, cropped, and temporal examples. Tier B and Tier
C tables and claims remain visibly separate in all experiment and final reports.

## Final protocol outcome

EXP-0007 used the permitted one-seed primary-fold early-elimination path. It
passed its step-300 continuation control but failed the unchanged fully-visible
and candidate-proxy reliability gates by large margins and failed the exact
qualitative shortcut gate. The deterministic result is `PRIMARY_FOLD_FAIL`, not
acceptance or a near-gate repeat. Accordingly:

- folds 0/1 and repeat seeds were not authorized;
- cropped-body diagnostics were reported but could not authorize advancement;
- temporal, refinement, and uncertainty phases remained blocked;
- the unaudited portion of the 2025 holdout remained unopened; and
- no final-test metric or accepted final-system throughput exists.

This is the expected fail-closed behavior of the frozen protocol. One primary
fold is sufficient for rejection; every fold would have been required for
acceptance.
