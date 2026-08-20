# EXP-008 — Differentiable-refinement capture basin

## Hypothesis

Training-free differentiable rendering can recover local centerline accuracy
from proposals inside a measurable neighborhood of the correct pose.

This is plausible because
[Splender](https://papers.miccai.org/miccai-2025/0860-Paper1793.html) reports
reliable subpixel C. elegans spline refinement, while the earlier repository
result left initialization sensitivity unmeasured.

## Controlled change

Eight held-out-profile analytic Tier-C poses were perturbed independently by:

- translation: 1, 2, 4, 8, 16, 32, 64 original-image pixels;
- rotation: 1, 2, 5, 10, 20, 40 degrees;
- length: plus/minus 2%, 5%, 10%, 20%;
- smooth shape: 2, 5, 10, 20, 40 degrees RMS tangent perturbation;
- three declared combined-severity mixtures.

The same intrinsic optimizer and 0/1/3/5/10 step snapshots were used for raw
robust pixels, pixel plus gradients, and a tube-weighted image likelihood. The
target raster, renderer, pose set, and success arithmetic were identical across
objectives. Pixels outside the image are never instantiated, so the likelihood
is geometrically FOV-censored.

The initial untuned pilot in the sibling
`scientific_exp_008_refinement_capture_basin` directory used overlarge Adam
trust radii and is not scientific evidence. The corrected radii were frozen
before this eight-pose run; the intermediate four-pose tuned run is likewise
superseded by this record.

## Result

The success definition is median point error at most 4 original-image pixels.
That threshold is provisional until EXP-001 measures the human noise floor.

After 10 raw-pixel steps, recovery probability was:

| Perturbation | 100% recovery through | Next tested level |
|---|---:|---:|
| translation | 4 px | 8 px: 87.5% |
| rotation | 2 degrees | 5 degrees: 87.5% |
| length | 2% | 5%: 93.75% |
| shape | 2 degrees RMS | 5 degrees: 87.5% |
| combined | none | mild mixture: 37.5% |

Across all deliberately easy-to-impossible perturbations, objective results
were:

| Objective | Success fraction | Median final point error |
|---|---:|---:|
| raw robust pixel | 48.71% | 4.16 px |
| pixel + gradient | 46.98% | 4.48 px |
| tube-weighted likelihood | 46.12% | 4.59 px |

The raw objective is the leading simple likelihood; neither added objective
earned its complexity on analytic Tier C. The controlled eight-pose CPU sweep
took 729.77 seconds. Raw-pixel 10-step trajectories averaged 0.91 seconds per
case in that resource-contended CPU run; this is not a CUDA throughput result.

The previous rejected proposal lies outside this basin. Among its reported
per-case errors, 0/31 Tier-B proxy frames and 0/43 fully visible Tier-C frames
had median point error at or below even 16 px; their best medians were 33.69 px
and 43.92 px respectively. Applying refinement to that checkpoint would not
test the measured basin and is not authorized as EXP-009.

## Visual evidence

- `refinement_capture_basin.png` directly plots recovery probability against
  each initialization error.
- `refinement_objective_comparison.png` plots error/success against step count.
- `refinement_before_after_overlays.png` shows identical target images before
  and after 10 raw-pixel steps.

## Conclusion

`PARTIALLY SUPPORTED`

A useful Tier-C capture basin exists and is now quantified, but mixed errors
are substantially harder, the sample contains eight analytic poses, no real
Tier-A image has been tested, and the human-scale success threshold is pending.

## Consequence

Keep raw robust pixels as the reference objective. Once a localization-
preserving proposal is trained, report the fraction of its Tier-A and Tier-C
cases inside this measured multidimensional basin before running EXP-009.
Do not refine the existing rejected global proposal. Repeat the perturbation
study on real Tier-A images after EXP-001, and benchmark 1/3/5 steps on CUDA
before judging throughput.
