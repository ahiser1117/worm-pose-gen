# Annotation Recommendations

> **Active single-person protocol (2026-08-19):** Only one person will annotate
> this project. The operational minimum is therefore 30 unique development
> frames plus 10 delayed, prior-trace-blind repeats by that person. This
> measures intra-annotator repeatability and supports directional model triage;
> it does not estimate inter-annotator agreement. The tool and exact workflow
> are documented in `docs/SINGLE_ANNOTATOR_WORKFLOW.md`. The stronger
> multi-person designs below remain reference protocols, not current blockers.

> **Follow-on status (2026-08-19):** The newer scientific plan replaces the
> earlier 256-frame allocation below with a development-only primary tranche,
> preserving every unaudited 2025 holdout frame for frozen final validation.
> The deterministic selection is now materialized at
> `experiments/scientific_exp_001_annotation/selection_manifest.json`: 256
> development frames across all three 2023 sessions, including six complete
> 11-frame windows (66 frames) assigned to two independent annotators. The
> human traces themselves have not been collected. See
> `docs/SCIENTIFIC_EXPERIMENT_STATUS.md` for the active protocol; the older
> allocation below remains as historical documentation of the first study.

No true manual centerline labels were supplied. The current centerlines are
conservative classical candidate proxies and cannot measure real-image
anatomical accuracy. The smallest useful initial tranche is **256 annotated frames**, not
an attempt to label the full 79,704-frame readable inventory.

## Recommended tranche

| Source | Selection | Frames | Purpose |
|---|---|---:|---|
| Three 2023 development sessions | 40 isolated frames/session | 120 | Ordinary and difficult single-frame accuracy |
| Three 2023 development sessions | Two nonoverlapping 11-frame windows/session | 66 | Temporal continuity, flips, and genuine fast changes |
| 2025 audited holdout, only after model freeze | 48 isolated frames | 48 | One-time camera/session-shift evaluation |
| 2025 audited holdout, only after model freeze | Two 11-frame windows | 22 | One-time temporal camera/session-shift evaluation |
| **Total** | | **256** | |

Exclude the 32 disclosed audit indices from the 2025 selections. Freeze the
model, checkpoint, thresholds, calibration, and selection algorithm before
reading or annotating any other 2025 frame. Holdout annotations must never be
fed back into model selection in this study.

Within each source stratum, reserve roughly 20% ordinary full-body frames and
actively enrich the remainder for boundary contact/truncation, tight bends or
self-overlap, low contrast or motion blur, uncertain head/tail appearance, and
large disagreement among the classical proxy, final proposal, and any temporal
prior. These categories may overlap. Select temporal windows before examining
individual model errors inside them so a favorable frame is not cherry-picked.

Independently double-label **64/256 frames (25%)**, stratified across every
session and difficulty category, including at least two complete temporal
windows. Adjudicate only after both initial labels are locked. This is the
minimum recommended evidence; it is not claimed to have been collected here.

## Executable annotation protocol

### Identity and provenance

Each annotation record must store:

- configured and resolved source path, source byte size and nanosecond mtime;
- source HDF5 dataset path, integer frame index, and timestamp when available;
- frozen split role and selection-stratum identifiers;
- annotator pseudonym, tool name/version, annotation-schema version, UTC start
  and completion times, and parent annotation/revision identifier;
- whether the annotator saw a single frame or a declared temporal window;
- model overlays shown, if any. The primary pass should be overlay-blind.

Never modify the source HDF5. Store annotations in a separate versioned file.

### Coordinates and ordering

Coordinates use original-image pixel centers: `(0, 0)` is the upper-left pixel
center, x increases right, y increases down, and half-open bounds are
`0 <= x < width`, `0 <= y < height`. Annotators trace a variable-density
anatomical midline polyline; the export tool arc-length-resamples a complete
body to exactly 100 ordered points without moving its two endpoints.

Ordering is anatomical head to tail only when the annotator selects `head` or
`tail` for both endpoints. If identity is uncertain, store both endpoint
hypotheses and `head_tail_state = ambiguous`; never resolve ambiguity by the
current model or classical proxy. Downstream orientation scoring must either
exclude ambiguous cases or use a symmetric forward/reverse metric.

### Support, overlap, and off-screen anatomy

Each traced vertex receives one status:

- `supported`: centerline is locally visible and usable;
- `occluded_in_fov`: anatomical location is inferred through a self-overlap or
  transient occlusion while its image coordinate remains inside the FOV;
- `outside_fov`: anatomy is known to continue beyond a specific image edge but
  its unobserved coordinate is not annotated;
- `not_identifiable`: evidence is insufficient even to continue the visible
  trace confidently.

Geometric `in_fov_mask` is computed from coordinates and image dimensions; it
is not hand-labeled. `image_support` is the independent categorical judgment
above. For self-overlap, trace the most likely anatomical path, mark the
overlapped span `occluded_in_fov`, and set a confidence flag. For a naturally
truncated worm, terminate the visible polyline at the image boundary, record
which anatomical end is outside and the exit edge/direction, and leave hidden
coordinates missing. Do not extrapolate an invisible tail merely to fill 100
points. Such frames support visible-body, boundary, support, and truncation
metrics but not hidden-body coordinate error. Hidden-body truth remains
available only in controlled crops constructed from a previously full body.

### Quality control and adjudication

The tool must reject out-of-bounds supported points, duplicate consecutive
vertices, zero-length traces, missing endpoint states, and silent orientation
changes between revisions. Reviewers see the raw frame, both locked traces,
and their disagreement map. The adjudicator may accept one trace, create a
third trace, or mark the case unresolved; the original labels remain retained.

Before using model errors as scientific accuracy, report inter-annotator:

- symmetric centerline point error in pixels and normalized by body length;
- circular tangent-angle MAE by normalized body coordinate;
- endpoint disagreement;
- head/tail agreement with `ambiguous` as a separate state;
- support-status agreement and boundary/truncation agreement;
- results stratified by session and difficulty category.

Report the median, p95, and bootstrap 95% interval. Model performance within
the double-label disagreement envelope should not be described as meaningfully
better without more precise annotation. No inter-annotator result is reported
for this repository because no manual annotations were available.
