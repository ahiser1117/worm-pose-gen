# Failure Modes

This is a living register. Prevalence is reported only where the bounded audit
or an experiment measured it.

| Failure | Detection signal | Planned mitigation | Evidence status |
|---|---|---|---|
| Source HDF5 cannot be opened/read | Explicit per-file audit exception | Exclude unusable files without modifying them; report exact scope | Observed in 8/12 files: six open/schema failures and two image-read failures |
| Head/tail swap | Low orientation confidence, temporal discontinuity | Canonicalized orientation hypotheses and calibrated probability | Planned |
| Tail/head leaves FOV | Geometric half-open bounds and support targets | FOV-censored loss plus temporal hidden-body inference | Exact static and moving-camera Tier C crop contracts pass; learned robustness pending |
| Truncation interpreted as short body | Abrupt length decrease near boundary | Intrinsic length prior and crop training | Planned |
| Self-overlap/tight turn | Low support, high residual, implausible topology | Uncertainty and optional alternative hypotheses | Planned |
| Endpoint collapse/spline oversmoothing | Endpoint/body-coordinate error curves | Representation ablation and curvature-aware QC | Proxy extractor rejects endpoint/topology defects; learned ablation pending |
| Proxy endpoint under-reach | Visual endpoint audit and body-length range | Keep proxy labels as limited training candidates; require Tier C/end-position curves | Mild roughly 10--20 px under-reach seen qualitatively in some accepted overlays; no Tier A prevalence |
| Synthetic-to-real appearance gap | Real overlay inspection versus analytic montage | Mix real proxy texture with synthetic geometry; never infer real accuracy from Tier C | Generator is geometrically plausible but substantially simpler/thinner than real NIR worms |
| High-frequency temporal jitter | Angle velocity/jerk metric | Short temporal context if experiment supports it | Planned |
| Motion blur/low contrast | Support probability and image residual | Robust normalization and calibrated uncertainty | Planned |
| Recording-background leakage | Grouped recording/session split | Hold out entire acquisition groups | Required |
| Refinement local optimum | Image loss worsens or pose moves implausibly | Step cap, proposal prior, accept/reject gate | Planned |
| Overconfident failure | Coverage/reliability diagnostics | Post-hoc or learned calibration; quality gate | Planned |
| Throughput collapse | Batch-1/batched p50/p95 and memory benchmark | Compact model, vectorization, adaptive compute | Planned |
