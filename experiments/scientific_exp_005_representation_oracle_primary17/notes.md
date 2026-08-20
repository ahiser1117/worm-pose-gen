# EXP-005 — Representation oracle on 17 complete primary traces

## Evidence boundary

This is an oracle representation fit, not image-model performance. It uses the
17 complete traces in the single-annotator primary tranche and excludes 12
truncated and one not-identifiable trace. The protected holdout was not opened.
No representation is compared with human precision until the delayed repeat
pass supplies that measurement.

The spline and cosine fits use four separately represented pose values (2D
translation, global rotation, and body length) in addition to the stated shape
coefficient count. Curves are integrated at uniform arc-length spacing and
optimally translated. The resulting 99-angle numerical control has a 0.092 px
median floor because the resampled manual polyline does not have perfectly
equal segment lengths.

## Result

| Representation | Shape coefficients | Median per-frame point error | Median per-frame mean tangent error |
|---|---:|---:|---:|
| Cubic tangent spline | 8 | 2.09 px | 4.85 deg |
| Cubic tangent spline | 16 | 0.70 px | 3.31 deg |
| Cubic tangent spline | 32 | 0.41 px | 2.54 deg |
| Cosine tangent | 8 | 1.10 px | 4.21 deg |
| Cosine tangent | 16 | 0.67 px | 3.36 deg |
| Cosine tangent | 32 | 0.40 px | 2.49 deg |
| PCA tangent, same 17 traces | 8 | 0.83 px | 2.78 deg |
| PCA tangent, same 17 traces | 16 | 0.09 px | approximately 0 deg |
| PCA tangent, leave one recording out | 8 | 3.10 px | 7.59 deg |

The apparent 16-component PCA perfection is an in-sample, rank-limited result
from only 17 traces and is not evidence of generalization. When the complete
traces from an entire recording are held out, eight-component PCA degrades to
3.10 px and 7.59 degrees. Fixed spline/cosine bases do not learn from this small
tranche and therefore provide the cleaner capacity result.

## Conclusion

`SUPPORTED_ORACLE_ONLY`

A 16-coefficient tangent representation has a subpixel coordinate floor on the
available complete traces. Representation compression cannot explain the
12.12 px classical error or 80.87 px rejected-model error. The dominant current
problem is image-to-pose localization/learning, not intrinsic shape capacity.

Whether roughly 3.3 degrees of local tangent smoothing is below human repeat
precision remains unknown. Increasing to 24–32 coefficients reduces the angle
floor modestly at low output cost and is a prudent option for tight shapes.

## Consequence

Use a fixed 16- or 24-coefficient spline/cosine tangent head in the first
localization-preserving model comparison. Do not select a learned PCA basis from
17 complete traces. Revisit PCA after substantially more complete Tier-A/Tier-B
curves are available and validate it by recording-held-out reconstruction.

