# EXP-SMC-005 — generative mask observation energy

## Hypothesis

A K=16 intrinsic curve rendered with a fixed recording-level width profile
creates a useful local observation basin against the classical probability and
cleaned mask. At least one simple energy should be minimized at or within one
grid step of the known pose for most complete non-hard development traces and
should rise outward under translation, rotation, shape, and length errors.

## Method and evidence boundary

Use the EXP-SMC-003 fixed cubic K=16 reconstruction, EXP-SMC-004 leave-one-
frame-out recording mean width, and EXP-SMC-001B probability/mask. Downsample
isotropically by four to 183×242 for particle-scale rendering. Compare balanced
soft BCE, soft Dice, clipped signed-distance energy, and Dice plus 0.25 signed
distance. Evaluate seven frozen levels for each of four perturbations. Exclude
the expert-adjudicated hard cycle because it is a target for later SMC, not an
easy-pose likelihood oracle. This is development evidence against a classical
segmentation proxy, not manual mask truth. The protected holdout remains
closed.

The perturbation geometry, pivots, downsampling, aggregation, and tie behavior
are frozen in `config.json`. Gradient checks run through all 20 selected latent
pose values (16 cubic shape coefficients, rotation, length, and translation),
not through occupancy pixels. EXP-SMC-004's frozen result selected a bounded
frame scale, but fixed mean width had slightly higher median mask IoU; this
experiment prospectively removes that unnecessary nuisance parameter and keeps
the frozen EXP-SMC-004 metric record unchanged.

## Frozen decision rule

An energy passes if at least 70% of all case/perturbation curves and at least
60% for each perturbation have their minimum at zero or one adjacent grid
level; at least 65% of outward steps are monotonic; median energy at both severe
endpoints exceeds zero-pose energy; and pose gradients are finite and nonzero.
Select the first passing energy in the frozen simplicity order: Dice, balanced
BCE, signed distance, then hybrid.

## Status

`COMPLETED_SUPPORTED_ENERGY_SHAPE_ONLY`.

## Quantitative results

All four candidate energies passed the frozen basin-shape gate. The simplicity
rule selected soft Dice. Across 16 complete non-hard development traces and
four independent perturbations, 64/64 energy curves placed their minimum at
zero or one adjacent grid level. Overall outward-step monotonicity was 0.9714;
the per-perturbation fractions were 0.9792 translation, 1.0000 rotation,
0.9688 shape, and 0.9375 length. Median severe-endpoint minus zero-pose energy
was 0.1100, 0.2507, 0.1135, and 0.1289 for those four perturbations. All
20-value latent gradient groups were finite and nonzero in every case. Median
base-pose soft Dice was 0.8519 (visually audited worst 0.824, median 0.852,
best 0.870).

On the RTX 6000 Ada, render plus soft-Dice evaluation took a median 0.996 ms
for one particle, 2.680 ms for eight (0.335 ms/particle), and 7.914 ms for 32
(0.247 ms/particle) at 183×242.

## Visual evidence

[`energy_perturbation_curves.png`](figures/energy_perturbation_curves.png) shows
the median and interquartile energy basins for all four energies and all four
pose errors. [`base_pose_overlays.png`](figures/base_pose_overlays.png) shows
the worst, median, and best selected-render alignment; endpoint mask defects
remain visible and are part of the observation-proxy limitation.

## Failure analysis, decision, and consequence

No candidate suffered a flat or inverted controlled basin. Soft Dice wins on
simplicity and perfect near-zero minima, not because the alternatives were
invalid. This experiment does not calibrate `exp(-E/T)`, demonstrate global
capture from arbitrary poses, test ambiguous self-intersections, or establish
manual-mask accuracy. Proceed to conservative temporal-prior calibration and
controlled two-anchor SMC/smoothing; calibrate temperature and ESS there before
interpreting particle weights probabilistically.
