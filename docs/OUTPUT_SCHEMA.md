# Worm pose HDF5 output schema

Schema version `1.0.0` stores predictions below `/worm_pose`. Writers create a
same-directory `<destination>.partial`, stream complete frame batches into it,
validate it, set `/worm_pose.attrs["complete"] = true`, flush and close it, and
only then atomically replace the destination path. Resume is not supported:
the presence of a `.partial` file is treated as an interrupted run and must be
investigated or explicitly removed by an operator. A completed destination is
not overwritten by default.

## Coordinate and missing-value convention

Pixel `(0, 0)` is the center of the upper-left pixel; x increases right and y
increases down. A point is in the FOV exactly when `0 <= x < image_width` and
`0 <= y < image_height`. Body index zero is the more probable head and tangents
follow that export direction. Floating pose predictions must be finite: a
missing prediction frame is not representable in the core schema and is
rejected rather than silently filled. Timestamps are either all finite and
strictly increasing or all `NaN` (unavailable). Frame indices are always
present and strictly increasing; gaps explicitly represent source
discontinuities.

## Datasets

All datasets are frame-major, appendable on axis zero, chunked on complete
frames, shuffled, and gzip-compressed. `F` is frame count and `B` is the fixed
number of uniformly spaced body positions. Every dataset has `units`,
`axis_order`, and `missing_value` attributes.

| Path | dtype | shape | units and meaning |
|---|---:|---:|---|
| `/worm_pose/centerline_xy` | float32 | `[F,B,2]` | original-image pixels, final axis `(x,y)` |
| `/worm_pose/tangent_angle` | float32 | `[F,B]` | radians, `atan2(dy,dx)`, wrapped `[-pi,pi)` |
| `/worm_pose/curvature` | float32 | `[F,B]` | radians/original-image pixel; positive is clockwise on display |
| `/worm_pose/in_fov_mask` | bool | `[F,B]` | point membership under the half-open bounds above |
| `/worm_pose/image_support_probability` | float32 | `[F,B]` | probability in `[0,1]` of usable local image evidence |
| `/worm_pose/angle_uncertainty` | float32 | `[F,B]` | non-negative marginal circular angular uncertainty in radians; its exact calibration/interval construction belongs in the model config |
| `/worm_pose/head_tail_probability` | float32 | `[F]` | probability in `[0.5,1]` that exported index zero is the head |
| `/worm_pose/quality_score` | float32 | `[F]` | unitless model-defined score, not inherently a probability |
| `/worm_pose/frame_index` | int64 | `[F]` | source frame index, strictly increasing, gaps permitted |
| `/worm_pose/timestamp` | float64 | `[F]` | seconds, strictly increasing, or uniformly `NaN` when unavailable |

The curvature endpoint rule, quality-score construction, angle-uncertainty
distribution, and any timestamp derivation must be fixed by and retained in
the digested inference configuration. Future minor schema versions may add
`centerline_covariance` or weighted hypothesis datasets; readers must not infer
their presence.

## Attributes and provenance

`/worm_pose` has `schema_version`, `complete`, `body_points`, and, after
successful validation, `frame_count`. `/worm_pose/provenance` records:

- configured and resolved source paths, explicit source dataset path, source
  byte size, and nanosecond mtime identity;
- SHA-256 checkpoint and configuration digests;
- Git commit and a deterministic JSON mapping of package versions;
- the complete geometry convention; and
- image height and width when supplied, enabling deterministic validation of
  `in_fov_mask`.

Source identity is metadata, not permission to mutate the source. Source HDF5
files are opened read-only. Consumers should call `validate_output` or
`open_completed_output`; `.partial`, missing-marker, version-mismatched,
unequal-length, uncompressed, or non-monotonic outputs are rejected.
