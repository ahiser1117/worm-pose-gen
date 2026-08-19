# Failure Modes

This is a living register. Prevalence is reported only where the bounded audit
or an experiment measured it.

| Failure | Detection signal | Planned mitigation | Evidence status |
|---|---|---|---|
| Source HDF5 cannot be opened/read | Explicit per-file audit exception | Exclude unusable files without modifying them; report exact scope | Observed in 8/12 files: six open/schema failures and two image-read failures |
| Head/tail swap | Low orientation confidence, temporal discontinuity | Require manual orientation labels before a learned probability; exploratory output fixes unknown at 0.5 | Not measurable: no Tier A orientation labels and no accepted head/tail head |
| Tail/head leaves FOV | Geometric half-open bounds and support targets | FOV-censored loss plus temporal hidden-body inference, only after proposal acceptance | Exact static/moving Tier C contracts pass; EXP-0007 cropped hidden-body point error reached roughly 156 px at 40% |
| Truncation interpreted as short body | Abrupt length decrease near boundary | Intrinsic length prior and crop training | EXP-0007 cropped Tier C median length error 29.93%; proposal rejected |
| Self-overlap/tight turn | Low support, high residual, implausible topology | Manual difficult-case labels, then uncertainty/alternative hypotheses | Present in audit/proxy visuals but prevalence and model accuracy are not measurable without Tier A |
| Endpoint collapse/spline oversmoothing | Endpoint/body-coordinate error curves | Localization-explicit proposal plus curvature/length QC | EXP-0007 fully-visible endpoint errors 118.01/125.34 px and median length error 37.75% |
| Independent-coordinate topology explosion | Body-length error and random overlays | Prefer intrinsic reconstruction or explicit topology constraints | EXP-0004 coordinate mean length error exceeded 9,300 original px; variant rejected |
| Spatial-bottleneck regression to mean | Centroid/length distributions and fully-visible overlays | A new localization-explicit formulation; do not spend the holdout on this family | EXP-0007 4x4 rescue still 87.54 px median/27.68 deg mean and systematically short/straight/displaced; rescue rejected |
| Proxy endpoint under-reach | Visual endpoint audit and body-length range | Keep proxy labels as limited training candidates; require Tier C/end-position curves | Mild roughly 10--20 px under-reach seen qualitatively in some accepted overlays; no Tier A prevalence |
| Synthetic-to-real appearance gap | Real overlay inspection versus analytic montage | Mix real proxy texture with synthetic geometry; never infer real accuracy from Tier C | Generator is geometrically plausible but substantially simpler/thinner than real NIR worms |
| Training-size physical crop excludes visible anatomy | Exact requested-versus-geometric support contract | Use a larger 4:3 source camera window with recorded isotropic resize | EXP-0003: 0/90 complete frames and 65/900 valid conditions; original design rejected |
| Curved body re-enters an axis-aligned artificial FOV | Exact contiguous-support mismatch | Report rejected complete-series hypothesis; balance only prevalidated conditions across sessions | EXP-0005 rejected the series gate; EXP-0006 accepted 300 condition-level cases with 10/cell and all contracts passing |
| High-frequency temporal jitter | Angle velocity/jerk metric | Short temporal context only after single-frame acceptance | Not evaluated: temporal modeling was correctly blocked |
| Motion blur/low contrast | Support probability and image residual | Tier A difficult-frame labels, robust normalization, calibrated uncertainty | Qualitatively present; no accepted model or prevalence estimate |
| Recording-background leakage | Grouped recording/session split | Hold out entire acquisition groups | Required |
| Refinement local optimum | Image loss worsens or pose moves implausibly | Step cap, proposal prior, accept/reject gate | Not evaluated: proposal was too inaccurate to enter refinement |
| Overconfident failure | Coverage/reliability diagnostics | Do not expose pose confidence until calibrated on Tier A/Tier C errors | Pose uncertainty absent; exploratory output uses explicit pi/0 sentinels; support calibration is not pose calibration |
| Throughput collapse | Batch-1/batched p50/p95 and memory benchmark | Compact model, vectorization, storage-inclusive profiling after accuracy acceptance | Not observed in proposal-only harness (2,461 batch-32 samples/s), but full HDF5 read/write throughput was not benchmarked |
