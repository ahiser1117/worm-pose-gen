# Evaluation protocol

This protocol was frozen after the bounded data audit, before model development
and before opening the final test split. It keeps engineering agreement on real images (Tier B) separate
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

The frozen manifest is `configs/split_manifest.json`. The complete 2023-09-19
and 2023-09-27 sessions are training groups; 2023-10-11 is the validation group;
the 2025-03-06 Hamamatsu-condition session is the untouched final-test group.
Whole recordings make temporal guard intervals unnecessary. The eight unreadable
recordings are quarantined, never reassigned as empty/negative data. All readable
recordings share the starvation project family, so this allocation cannot support
cross-project or independent-background generalization claims.

The split manifest records source identity, all grouping keys, allocation,
random seed (base seed 20260818), temporal ranges, guard intervals, and rationale.
No neighboring temporal frames cross splits. Validation and development folds
may be inspected during model selection. Final-test frames, labels, aggregate
metrics, and examples remain untouched until architecture, checkpoint,
calibration mapping, and thresholds are frozen. The final test is evaluated
once; subsequent changes create a new study rather than another attempt on the
same test.

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

Metrics are computed per frame, while resampling respects the highest independent
unit (animal/session/recording group). Report a two-sided 95% group bootstrap
interval with at least 2,000 deterministic resamples. A gate passes only when
the entire 95% interval is on the passing side: the upper bound for maximum-error
or calibration gates, and the lower bound for improvement/minimum-performance
gates. Exact bounds agreement must have zero failures, not merely a favorable
interval. If fewer than five independent groups exist, report grouped fold
results and require every fold to pass; label all resulting evidence limited.

Use at least three training seeds whenever a one-seed estimate or interval lies
within 10% of a gate or practical-effect threshold. In that case, each seed must
pass and the across-seed mean is reported with its range. Otherwise, one seed
plus the group bootstrap is sufficient for early elimination. Missing values,
excluded frames, and failed inference count are always reported and may not be
silently dropped.

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
