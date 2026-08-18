# EXP-0002 — Controlled synthetic geometry and crop benchmark

## Hypothesis

An analytic intrinsic-angle generator and differentiable tube renderer can
provide exact Tier C centerline/support truth and controlled 5–40% anatomical
head/tail censoring without introducing coordinate or boundary artifacts.

## Difference from baseline

Adds controlled synthetic truth; it does not measure real-image accuracy.

## Data/split

Generator seeds 20260818–20261329 are development; 20270000–20270127 are held
out. Development samples use 300–600 px lengths and moderate intrinsic bend
amplitudes; held-out samples use disjoint 250–299 or 601–700 px length bands and
the upper bend-amplitude band declared in the generator configuration. Exact
numeric parameter ranges and nuisance distributions are serialized with the
dataset before generation. No audited-holdout real recording is read. Required
hidden fractions are 5%, 10%, 20%, 30%, and 40% at both head and tail, including
temporally coherent sequences with smoothly moving crop boundaries.

## Training/resource budget

- maximum steps/epochs: generate <=512 development and 128 held-out samples
- wall-time limit: 15 CPU minutes
- seed/repeat policy: exact stored seed per sample
- checkpoint cadence: not applicable
- expected GPU time: none
- expected external-storage use: <=1 GiB under `datasets/worm_pose_gen/synthetic_v1`
- early termination conditions: non-finite geometry, irreversible transforms, or support-mask disagreement

## Success criterion

- primary metric: geometry/render/crop contract validity
- numeric practical-effect threshold: 100% finite 100-point centerlines; exact
  half-open FOV-mask agreement; exact requested hidden point counts; crop
  coordinate round-trip max error <=1e-5 px; renderer gradients finite and
  nonzero; no held-out sample with length outside 250–700 px
- variability/confidence rule: exhaustive contract checks over all generated
  samples plus deterministic random and most-curved montage
- pass/fail interpretation: ACCEPT the benchmark only if every exact invariant
  passes; visual plausibility is reported separately and does not turn synthetic
  evidence into real-image evidence

## Results

ACCEPTED on the full predeclared budget: 512 development poses, 128 held-out
poses, 100 body points per pose, and all 6,400 static pose/end/fraction crop
cases. All geometry was finite and fully inside the initial audited 732x968
canvas. Every static crop hid exactly 5/10/20/30/40 points, geometric support
agreed exactly with independently recomputed half-open FOV membership in both
original and 192x256 render coordinates, and the largest camera/source
round-trip residual was 3.41e-13 px (raster mapping: 5.68e-14 px).

Development lengths were 300.349--598.951 px with bend amplitudes
0.2510--0.5497 rad. Held-out lengths were 250.592--699.999 px, with 59 samples
in the 250--299 band and 69 in the 601--700 band; held-out bend amplitudes were
0.6520--0.8955 rad, disjoint from development. Renderer pose/width gradients
were finite and nonzero (combined L1 47.2811).

The 21-frame head and tail crop sequences moved the camera boundary linearly
from 5% to 40% hidden. Hidden counts were monotone, support agreed exactly in
every frame, transform round-trip residuals were at most 1.42e-13 px, and the
largest camera-offset second difference was 2.27e-13 px (floating-point zero).
No synthetic array dataset was persisted; poses regenerate exactly from seed
and profile. No real recording was read.

## Figures

- `figures/generator_montage.png`: deterministic development examples, a
  held-out example, and the most-curved pose (held-out seed 20270090).
- `figures/crop_sequence_montage.png`: frames 0/5/10/15/20 from each smooth
  21-frame head/tail moving-camera sequence. Inputs are rendered directly in
  the camera window and contain no padding or composited crop border.

## Runtime

4.811 seconds for generation, exhaustive static checks, temporal audits,
renderer gradient audit, and figures on CPU with Python 3.13.15 and PyTorch
2.13.0+cu130. Repository output remains below 0.5 MiB and external storage use
is zero.

## Interpretation

The split-specific generator, differentiable renderer, fixed-fraction crop
benchmark, and temporally coherent crop utility satisfy the Tier C geometry,
coordinate, support, distribution-shift, and gradient contracts. The complete
geometry and image-nuisance distributions were serialized to `config.json`
before generation. This is controlled synthetic evidence only: visual
plausibility does not establish accuracy or robustness on real NIR images.

## Decision

ACCEPT

## Next experiment

Use the frozen candidate proxies and Tier C generators for representation and
temporal-context comparisons, retaining the 24 reviewed overlays as a separate
limited qualitative Tier B subset.

Synthetic crops validate geometry and controlled hidden-body behavior. They do
not replace the required proxy-real crop benchmark that preserves real visible
texture/background, which is run only after accepted real proxy centerlines
exist.
