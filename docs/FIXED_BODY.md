# Fixed body overlay

After processing a recording and repairing its poses, open **Run → Whole
workspace → Detailed stage configuration → Fixed body (optional)** and click
that stage's **Run** button. It is excluded from the default pipeline. To
process an entire recording, use a workspace containing the entire recording;
the pass visits every sampled frame in that workspace.

The result appears in **Layers → Fixed body (dashed = extrapolated)**. The
cyan overlay shows its centerline, outline, equal-length segment markers and
width ticks. Its visibility and opacity can be adjusted independently of the
reviewed pose. It is available during playback as well as on paused frames.

The model uses a single length and head-to-tail diameter profile for the
workspace. By default it calibrates from at least three fully visible poses
with IoU ≥ 0.9, ambiguity score < 2, finite positive widths, and no pending
mask edits. A calibration frame's body must lie inside the image, and its
stored mask, if available, must not touch the border. The median length and
smoothed median width profile are frozen. It uses the current head-to-tail
orientation; correct reversed poses before running this stage.

Expand the stage to change the number of segments (default 99), minimum
calibration IoU, or minimum number of calibration frames. **anchor_frames**
accepts comma-separated source frame IDs of trusted calibration poses. These
replace automatic IoU/ambiguity selection, but must still be fully visible
and current. Insufficient calibration frames produce a job error and leave
any previous fixed-body result intact.

For each usable pose, the head is fixed exactly. Equal-distance targets are
sampled along its ordered midline using absolute distance from the head;
each frame is not rescaled to the model length. Segment angles are refined
together against those targets with a small bend penalty. The lengths of all
segments remain exactly equal, and the width profile never changes per frame.
No temporal smoothing is applied.

After the last complete segment supported inside the field of view, all
remaining segments continue in that segment's direction. Dashed geometry is
extrapolated, including any unsupported remainder before the image boundary.
If a fully visible source is shorter than the model, the unsupported tail is
also dashed. If it is longer, the model stops at its fixed length. Neither
case stretches the body to force an endpoint match.

**Statistics → fixed body** reports the model dimensions, calibration count,
midline RMS error, extrapolated point count, and observed minus fixed length
for fully visible poses. A missing/stale input pose, an offscreen head, or too
little visible body has no fitted overlay. A midline that leaves and re-enters
the image is marked `reentry_unresolved`; this version does not infer a hidden
bridge. Extrapolated geometry is a completion convention, not a measurement of
unseen posture.

The separate `fixed_body.npz` file contains `frame_index`, `centerline_xy`,
one shared `width_profile` (diameters in pixels), `extrapolated`, `in_fov`,
`rms_px`, `length_error_px`, `status`, and JSON `metadata`. Metadata includes
calibration frame IDs, dimensions, job ID, creation time, and input revision.
Unresolved rows contain NaN geometry. The file is written atomically under the
workspace job lock. It does not replace reviewed poses, masks, hypotheses,
provenance, or their existing exports.

Changing input poses or masks marks the saved model outdated and withholds its
overlay. Rerun **Fixed body** to recalibrate and rebuild it from the corrected
workspace. Reloading the app preserves the saved result and its stale status.

## Temporal smoother

The regional algorithm **Fixed-body temporal smoother** (`fixed_body_smoother`
in the Run panel's algorithm dropdown) is the fixed body with a motion prior.
It runs on a region like the other algorithms, produces a candidate set, and
changes nothing until the set is accepted. Unlike the overlay stage it writes
poses: each frame's candidate is the smoothed body encoded as a pose latent,
with the calibrated width profile and its overlap against the frame's mask.

The body is the pipeline's own pose space with the length frozen: 99 equal
links whose tangent angles are the pose latent's 16-coefficient B-spline, so
the candidates decode exactly from their latents. The length and width
profile are calibrated as for the overlay, from up to 100 trusted, fully
visible frames nearest the region (`min_calibration_frames` at least).

The worm is overdamped at its scale, so the prior bounds rates rather than
accelerations: every pair of consecutive frames pays a quadratic penalty on
the change of head position and of each angle coefficient, divided by the
source-frame gap between the rows. The scales of those penalties are
measured, not chosen: the robust spread of the same differences over
consecutive pairs of trusted, fully visible frames across the workspace,
which needs at least ten such pairs. `motion_tolerance` multiplies the
measured sigmas (2 by default), so a typical motion costs little and a jump
of many typical motions costs much.

The data term pulls a frame's body toward equal arc-distance samples of its
current pose, cut at the camera rectangle, with `data_sigma_px` as its pixel
scale. A trusted frame (IoU at least `min_iou` and ambiguity score below 2,
current mask) has full weight; any other frame has `untrusted_weight`, zero
by default, so it is shaped by its neighbours alone. A hundred points pull
hard: a small nonzero weight still lets a badly placed pose win against the
prior, so raise it only deliberately. With a quadratic data term and no
weighting, a temporal prior would smear a wrong frame into its neighbours
instead of fixing it; the weighting is what makes the prior useful.

All frames of the region are solved together with the anchors as fixed
nodes, by Levenberg-Marquardt over a block-tridiagonal normal matrix (18
variables per frame), which converges in a handful of iterations: a
two-thousand-frame region takes a few seconds on CPU. Head-to-tail
orientation follows the chain from the anchor before the region (else its
first frame): a frame whose ends match its predecessor better swapped is
reversed before smoothing, and an anchor after the region oriented against
the chain is an error. The unsupported remainder of a clipped body follows
the prior, so it continues the shape of neighbouring frames rather than a
straight line; it is a completion, not a measurement.

The candidate set's metrics carry the calibration (`fixed_body`: length,
segment length, calibration frames, measured motion scales), the solver's
iterations and convergence, the number of orientation flips applied, the
frames that had no usable targets, and the largest head step per source
frame.
