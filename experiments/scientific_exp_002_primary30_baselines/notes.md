# EXP-002 — Primary-30 directional baseline comparison

## Evidence boundary

This is development evidence from one annotator's 30-frame primary pass. It is
not inter-annotator agreement, intra-annotator repeatability, or a human noise
floor. The 10 delayed repeats remain locked. The protected 2025 holdout was not
opened.

Seventeen annotations are complete, 12 are naturally truncated, and one is not
identifiable. Complete traces use orientation-symmetric 100-position geometry.
Truncated traces use only one-way distance from the annotated visible trace to
the nearest point on the predicted curve; this does not score hidden anatomy or
matched anatomical positions.

## Controlled comparison

The unchanged conservative classical method and the unchanged rejected
EXP-0007 4x4 global-regression checkpoint were evaluated on the same 30 raw NIR
frames. The model was invoked only through its fail-closed exploratory API. An
ungated classical curve is reported separately as a diagnostic and is not an
accepted classical output.

## Result

| Method | Candidate output available | Complete traces scored | Median per-frame point error | Median per-frame mean tangent error | Truncated traces scored | Median visible-trace distance |
|---|---:|---:|---:|---:|---:|---:|
| Conservative classical | 14/30 (46.7%) | 11 | 12.12 px | 12.08 deg | 3 | 7.91 px |
| Ungated classical diagnostic | 30/30 | 17 | 12.12 px | 11.63 deg | 12 | 6.75 px |
| Rejected global model | 30/30 | 17 | 80.87 px | 51.04 deg | 12 | 49.32 px |

The conservative classical gate accepted 9/9 proxy-easy frames, 1/6 uniform
frames, 4/6 temporal-window frames, and 0/9 proxy-difficult frames. Its 16
rejections contain 14 boundary-contact, three low-ridge-support, and one
unstable-endpoint flags (cases can have multiple flags). This is appropriate
selective behavior, but its 53.3% failure rate makes it inadequate as an
all-frame solution.

The ungated classical diagnostic follows the annotated anatomy closely in the
overlay audit, including most boundary cases, but is not protected by the old
quality contract. The rejected neural checkpoint instead produces a displaced,
short, low-curvature mean-pose shortcut across recordings. The visual failure
matches its large numerical error and the prior Tier-B/Tier-C negative result.

CPU throughput, excluding source reads, was 4.40 frames/s for serial classical
processing and 18.35 frames/s for batched rejected-model preprocessing and
forward inference. These are engineering diagnostics, not deployment
benchmarks.

## Conclusion

`PARTIAL_BASELINE_COMPLETE`

The new human labels decisively confirm that the rejected global model should
remain rejected. They also show that localization/skeleton evidence in the
images is much stronger than that model captured. The classical method is a
useful geometry reference but not an adequate all-frame method because its
acceptance gate intentionally rejects most difficult and truncated cases.

This does not complete the full EXP-002 suite: WormTracer, a WormPose-style
model, and a localization-preserving anchored model have not yet been evaluated
on the common labels.

## Consequence

Do not run EXP-009 refinement on the rejected global proposal; its 80.87 px
primary-trace error is far outside the measured local capture basin. Prioritize
the localization-preserving EXP-003 architecture and retain the ungated
classical curve only as training/proposal evidence with explicit quality
handling. Do not tune methods closely on these 30 frames; expand Tier A if
candidate methods approach the delayed-repeat scale.
