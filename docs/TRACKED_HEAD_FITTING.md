# Head-tracked temporal fitting

Choose **Head-tracked temporal fit** in the Run panel's regional algorithm
dropdown. Run the selected range, compare the resulting candidate set, and
accept it when satisfied. Running the method does not replace current poses.

The method is intended for coils and self-intersections where overlap alone
lets the head jump onto a different body branch. It fits frames in order,
starting from a selected before anchor or from the acquisition nose. Each
frame is initialized from the previous result and, when available, the
current pose. The first unanchored frame also uses mask-derived starts.
Starts are oriented toward the tracked or previous head before optimization.
The final path cannot reverse those fitted poses.

Tube rendering combines the coverage of overlapping body parts. The nose can
touch or overlap the side of the body without carving a notch into its
silhouette or introducing a separation penalty. This applies to all tube
fitting methods. Rerun an affected region to fit its pose with this behavior;
previously saved poses are not refitted automatically.

| Control | Default | Effect |
| --- | --- | --- |
| Tracking strength | 0.2 | Pulls point 0 toward the acquisition nose. |
| Previous pose strength | 0.005 | Gently pulls in-view body points toward the previous fitted pose while letting the tail follow the mask. |
| Previous head strength | 0.5 | Adds a separate pull toward the previous fitted head. |
| Head distance scale | 6 px | Scales the squared tracking and previous-head penalties. |
| Maximum head movement | 8 px / recorded frame | Hard limit on head displacement, including optimizer starts and every update. |
| Keep head inside frame | On | Constrains the head to the camera rectangle. The rest of the body may leave the camera. |
| Hole filling | Off | Resegment unedited frames without hole filling; workspace masks and narrow-hole filling remain available. |

Hover over a control for its explanation, default, and bounds. Descriptions
also remain available to screen readers without occupying panel space.

The head penalties are squared XY distances divided by the head distance
scale squared, multiplied by their respective weights. The whole-pose prior
uses half the previous body width as its distance scale. Increasing these
weights favors continuity and can reduce overlap when the worm moves quickly;
compare candidate sets before accepting a new setting.

Movement is measured in zero-based image pixels. A workspace sampled every
five source frames allows five times the per-frame displacement. Skipped
frames without usable masks also increase the elapsed source-frame gap.
This constrains the observed endpoints of that interval; it does not infer
the unobserved motion between them.

Selected anchors remain fixed. If confident tracking shows an anchor has its
head and tail reversed, correct that anchor or choose another one. The method
fits forward; if its final head cannot reconnect to an after anchor within
the movement limit, the run reports the conflict without saving a candidate
set. Widen the region, choose a different anchor, or adjust the movement limit.

## Acquisition observations

The reader follows the acquisition writer in
[ConfocalTrackerControl.jl, pinned source](https://github.com/flavell-lab/ConfocalTrackerControl.jl/tree/359832f79da8f037324e3d824f2909e21e176043/src).
HDF5 exposes `/pos_feature` as `[saved frame, coordinate, landmark]` with
shape `[T, 3, 3]`. Coordinates are image x, image y, and confidence;
landmarks are nose, mid, and pharynx. `/pos_feature[:, :2, 0]` gives the
nose, and subtracting one converts its stored Julia image coordinates to
the app's zero-based XY coordinates. These are acquisition predictions,
not manual ground truth. Stage coordinates are not used as image points.

Features and `/img_nir` are saved together. Where camera metadata is present,
the reader checks `/img_metadata/q_iter_save` and `q_recording`, and aligns
source image IDs and timestamps. It samples only requested feature rows
and scans save flags in bounded chunks rather than loading video frames.

Finite, in-frame landmarks with confidence greater than 0.9 are usable.
Low-confidence or invalid observations fall back to the previous fit. A
region with no usable observations can still continue from a trusted before
anchor if the tracking schema is valid. Without a before anchor, the first
usable mask must have valid nose tracking. Missing or inconsistent tracking
metadata produces an actionable error; it is not replaced by an invented
center-of-frame observation.

Candidate metadata records the settings, tracking source, valid/fallback
frame counts, and the largest observed head movement per source frame.
The original mask revisions remain the basis of stale-candidate checks.
Explicit hole-filling choices affect fitting inputs only: saved masks and
manual labels are not rewritten. Off preserves manual foreground exactly;
On fills narrow holes and keeps ignored pixels excluded. Already painted-in
holes cannot be recovered automatically from a manual mask.

## Validation

CPU tests cover orientation selection, distracting tracking points, movement
and image bounds, latent/centerline agreement, batch alignment, sampled
frames and missing masks, lost tracking, fixed anchors, candidate persistence
and acceptance, and temporary hole-filling behavior. Reader tests and a
read-only real-recording check verify the acquisition schema and alignment.
Browser checks cover the registered controls and submitted job parameters.
These checks establish the constraints and workflow; accuracy across real
coiled recordings still needs evaluation with reviewed poses.
