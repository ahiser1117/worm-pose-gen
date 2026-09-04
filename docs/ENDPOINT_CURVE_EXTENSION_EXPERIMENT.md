# Curvature-aware extension of the A5 midline endpoints

This step continues the final Experiment A midline through the two rounded end caps of its completed body.

It fixes the specific short-end behavior visible in A5. The skeletonized pose ends inside the orange-plus-green body because thinning does not reach the tips and the frozen pipeline peels skeleton endpoints before selecting its path. We use the recent trajectory of each end to continue a shallow curve until it first reaches the completed-body boundary.

On this frame, both ends reach the intended boundary. The 100-point pose length increases from **692.86 px** to **731.48 px** and remains below the frozen **750 px** length ceiling.

## Evidence boundary

- The input is the rightmost A5 result for frame `3420` from `nir_videos/2023-09-19-01.h5`, dataset `/img_nir`.
- The boundary target is exactly the **orange-plus-green union** in A5: `repair_body_mask`. Green is the original Section 3 component and orange is the smooth-body completion beyond it.
- The original **25,709 Section 3 positives** and the **32,639-pixel completed body** are not changed.
- Only the frozen A5 midline and completed-body mask are used to fit and stop the continuation. Image darkness is not sampled or scored.
- The one-annotator trace is loaded only after both extensions, resampling, and the geometry gates have completed.
- This is still one disclosed development frame. The 7-point context and `0.25 px` integration step are prototype choices, not independently validated parameters.
- The protected 2025 holdout remains unopened.

## Frozen starting point

The A5 downstream pose is produced by thinning the completed body, peeling skeleton endpoints, selecting the longest path, and resampling it to 100 points. Its two endpoints remain inside the rounded caps.

![Frozen A5 endpoint gap](endpoint_curve_extension_experiment/00_frozen_a5_endpoint_gap.png)

Here, **boundary** means the first exit from the completed body. It does not mean the video-frame edge, and it does not mean the boundary of the green Section 3 component alone.

# A6: continue each terminal curve to the completed-body boundary

## A6.1 Fit the trajectory from a terminal neighborhood

For each end independently:

1. Orient the ordered midline so that the end being extended is last.
2. Keep the final 7 stations, covering about `42 px` of existing curve.
3. Compute the heading of every segment and unwrap the angles across `-pi/pi`.
4. Fit heading as a linear function of local arc length.
5. Use the fitted terminal heading as the outward direction and the fitted slope as signed curvature.

This is a constant-curvature local continuation. It is not a straight line drawn from the last segment, and it is not a global refit of the worm.

The two fitted curvatures are small but nonzero:

| End | Context length | Signed curvature | Equivalent radius |
|---|---:|---:|---:|
| Index 0 | 42.05 px | -0.00495 rad/px | 202.21 px |
| Index 99 | 42.06 px | +0.00432 rad/px | 231.37 px |

Over a roughly 20-pixel extension, those values preserve the visible shallow turn without allowing a terminal pixel or one noisy tangent to determine the result.

## A6.2 Advance to the first boundary crossing

Starting at each frozen endpoint, we integrate its fitted circular arc in `0.25 px` steps.

The stopping rule is deliberately local:

1. Continue while the next point belongs to the completed-body mask.
2. Stop at the first foreground-to-background transition.
3. Bisect that final curved step to place the endpoint at the subpixel mask interface.
4. Fail the extension if no boundary is reached within `80 px`.

Stopping on the first exit matters for a bent worm. It prevents a continuation from crossing background and re-entering a nearby arm of the same mask.

![Terminal context and curved continuation](endpoint_curve_extension_experiment/01_curve_context_and_extension.png)

The index-0 end adds **20.13 px** and the index-99 end adds **20.82 px** along their dense continuations. Both hit their first body boundary.

## A6.3 Splice, resample, and rerun the length gate

The two dense continuations are joined to the unchanged A5 samples. The combined polyline is then resampled to the same 100-point representation used everywhere else.

| Measure | Frozen A5 | Curved endpoint extension |
|---|---:|---:|
| Pose points | 100 | 100 |
| Centerline length | 692.86 px | **731.48 px** |
| Length added after resampling | — | **38.63 px** |
| Length gate | pass | **pass** |
| Completed-body area | 32,639 px² | unchanged |
| Maximum full width | 57.97 px | unchanged |
| Original-positive containment | 100% | unchanged |

The unresampled splice adds `40.95 px`. Resampling reduces the measured gain slightly because straight chords between output stations are shorter than the dense curved path.

![A5 and extended poses](endpoint_curve_extension_experiment/02_extended_pose_comparison.png)

This step changes only the midline. It does not rerender or enlarge the completed body, so the A5 area, width, and exact-containment results remain fixed by construction.

## Post-fit centerline audit

Only after the geometry is complete do we compare it with the one-annotator trace.

![Extended pose and post-fit trace](endpoint_curve_extension_experiment/03_manual_trace_postfit_audit.png)

| Pose | Median point error | Mean tangent error | Mean endpoint error | Body-length error |
|---|---:|---:|---:|---:|
| Frozen A5 notch-repair pose | 12.19 px | 5.42 deg | 20.85 px | 59.63 px |
| Curved endpoint extension | **4.38 px** | **3.05 deg** | **3.29 px** | **21.00 px** |

The trace supports the proposed mechanism on this frame, but it did not select the context length, curvature, stopping point, or any other part of the extension.

# Verdict

The additional step does what was intended. It follows the recent curve at both skeleton endpoints, reaches the first boundary of the orange-plus-green completed body, and recovers most of the missing endpoint length without changing the body mask.

This is still geometric self-consistency, not new anatomical evidence. The completed end caps were inferred by the smooth-body model; reaching a wrong cap would produce a correspondingly wrong endpoint. Constant curvature can also be inaccurate if the worm turns sharply just beyond the observed terminal context.

Endpoint retreat is not the only source of length bias. Fixed-count resampling replaces portions of a pixel path with chords, which can shorten a strongly curved path. This experiment isolates and repairs the endpoint contribution only.

Before promotion, freeze these rules and run them unchanged over all 30 development annotations. Report boundary-hit coverage, length-gate failures, endpoint error, full-curve error, and cases where the first-exit guard or `80 px` maximum rejects the continuation.

## Implementation

The reusable implementation is `extend_centerline_to_mask_boundary` in `src/worm_pose_gen/anchors.py`. It retains every input point exactly, fits both terminal curves independently, advances with subpixel circular-arc integration, and bisects the first boundary crossing.

Focused tests cover straight end caps, a curved annular body where a tangent line would drift away, endpoints outside the mask, and the maximum-extension guard.

## Rebuild and inspect

Run from the repository root after rebuilding Experiment A if its inputs changed:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/build_boundary_notch_repair_experiment.py

scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/build_endpoint_curve_extension_experiment.py
```

Machine-readable results:

- [`endpoint_curve_extension_experiment/metrics.json`](endpoint_curve_extension_experiment/metrics.json)
- [`endpoint_curve_extension_experiment/experiment_arrays.npz`](endpoint_curve_extension_experiment/experiment_arrays.npz)

The NPZ preserves the frozen A5 pose and mask, each fitted context and dense extension, the spliced curve, the final 100-point A6 pose, and the post-fit trace.
