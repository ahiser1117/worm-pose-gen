# Annotation Recommendations

No true manual centerline labels are currently supplied. Recommendations will
be finalized after the data audit identifies representative ordinary,
boundary-truncated, tight-turn, self-overlap, motion-blurred, low-contrast, and
head/tail-ambiguous frames.

Each future annotation must record source file identity, source dataset, frame
index and timestamp, original-image `(x, y)` coordinates with pixel centers and
half-open bounds, ordered centerline/body points, head/tail/ambiguous state,
geometric FOV membership, separately judged usable image support, handling of
self-overlap/off-screen anatomy, annotator identity, tool/version, and revision
provenance. A representative subset must be independently double-labeled and
adjudicated; inter-annotator centerline and circular tangent-angle disagreement
will be reported before model error is interpreted as scientific accuracy.
