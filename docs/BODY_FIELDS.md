# Body-field targets: build and correct

The body-field network (`src/worm_pose_gen/body_net.py`) predicts, besides
the worm mask, a per-pixel A-P field (normalized arc length, 0 at the head,
1 at the tail), head and tail heatmaps, and an overlap map (pixels covered
twice by body parts far apart along the body), from the flat-fielded frame
plus symmetric difference channels `frame[t+lag] - frame[t-lag]`. A hand
label only says which pixels are worm, so the other targets come from
fitting the tube model to the hand mask and orienting it head first
(`body_targets.py`). A wrong head end or a bad fit silently teaches the
network the wrong thing, so every label's body can be confirmed or corrected
on the pose app's Labeling page. The pose fitter can score its fits against the network's
predictions and start them from its traces
([Body-field evidence in the pose fitter](#body-field-evidence-in-the-pose-fitter)).

## Records

The targets of a library label are built by `body_fields.fit_targets` into
the personal library's target cache, keyed by the label revision and the
model that built them (`worm_pose_gen.library.targets`; a job after every
Labeling save, and before training). The human body fields (orientation, a
traced midline, mask only) are part of the label itself. The records below
are those of the old segmentation store, which `scripts/migrate_to_library.py`
read the human body fields from.

`<store>/body_fields/<sample_id>.npz`, one per sample of a segmentation
store, is described in `src/worm_pose_gen/body_fields.py`: the 33 context
frames `t-16..t+16` and their validity, the fitted head-first tube
(`centerline_xy`, `width_profile`), the rendered `ap`, `overlap`, `head_xy`,
`tail_xy`, `diameter_px`, the acquisition `nose_xy` when there was one, and
a JSON `meta`. The build writes `sample_id`, `mask_revision`, `max_lag`,
`fit_preset`, `has_body`, `orientation` (`nose`, `nose_nearby`, `taper`),
`nose_offset`, `orientation_margin`, `fit_iou`, `overlap_px` and
`fit_method`. A single-frame fit cannot find a coiled or self-touching body,
so a frame that fits below IoU 0.93 or touches itself is also fit by a
chain from the nearest clear context frame on each side (the context frames
are segmented with the promoted segmenter); the better fit is kept, and
`fit_method` is `chain` with `chain_anchor_offset` and
`independent_fit_iou`. Edits in the old Body fields screen added or changed:

| Field | Written by | Meaning |
|---|---|---|
| `orientation: "manual"` | Flip | head and tail swapped by hand; `orientation_margin` changes sign |
| `review` | Accept / Reject / Unreviewed | `unreviewed` (also when absent), `accepted`, `rejected`; a rejected sample trains the mask only |
| `reviewed_at` | Accept / Reject / Unreviewed | UTC ISO time of the last review |
| `fit_method: "traced"` or `"trace_as_drawn"`, `trace_xy`, `auto_fit_iou` | Trace midline | the body refit along clicked points (see below); `review` becomes `accepted` and `orientation` `manual` |

A record is **stale** when the store's mask revision differs from
`meta["mask_revision"]`: the label was edited after the targets were built.
Rebuilding replaces the whole record, so a rebuilt sample is unreviewed
again and its orientation is re-derived from the nose landmark or taper,
unless the record holds a trace: then the body is refit along the stored
trace to the current mask and stays accepted.
Every write goes to a temporary file that replaces the record.

Build or refresh the records of an old store (samples whose record is current
are skipped; `--force` rebuilds them, `--samples` limits the run):

```bash
scripts/project_env.sh uv run --no-sync python scripts/build_body_fields.py --review
```

## Correcting a fit by tracing the midline

Where the mask is right but the fitted body takes the wrong route through a
crossing or contact, the head, tail and A-P targets are all wrong with it.
A trace (Labeling's **Trace midline**; `body_fields.trace_fit`, and
`fit_targets(..., trace=...)` for the stored targets) takes points clicked
along the body from head to tail, through the crossing in the order the
body actually goes, and refits the tube from them: the trace is the start,
every point of the fit is pulled toward the same-numbered point of the
resampled trace (weight 0.02 at half a body width), the head toward the
first click and the tail toward the last (weight 0.2 within 6 px, in view
or off camera), and the tube is otherwise fit to the hand mask, so it
keeps the traced route and the person's ends but finds the body's edges
and width itself. The length
bounds do not apply (the trace sets the length), and a non-interpenetration
term (`MaskFitConfig.separation_weight` 1.0 at full separation) keeps two
parts of the body far apart along it from sitting inside each other: the
rendered tube is a union, so without it a head or tail touching the body
slid onto it at no cost. Pairs the trace itself puts on top of each other,
and their neighbours within a body width along both runs, are a deliberate
crossing and exempt. On 28 traced records it cut the median overlap from
252 to 92 px (mean 391 to 134) for 0.002 median IoU; pinning the width to
the recording's median profile instead made the overlap worse. The trace
is the body from end to end, so a trace that stops at the image border ends
the body there, and its targets treat it as cut off by the camera like any
fit (`mark_exits`: no tail in view, the A-P field scaled to the visible
share of the recording's typical body length); a tail whose place off
camera is known is clicked there. For a tail far off camera, **Extend off
camera** (X, saved with the label as `trace_extend`) continues a trace that
ends within 8 px of the border, or past it, straight off camera to the
recording's typical length (`extend_trace`), and leaves the tail free; on
Alex's labels the extended bodies come out at 0.97–1.04 of that length.
The length is the first of (`Labeling.body_length`, named in the trace
panel): the median whole, well-fit body among the recording's labels; the
body-size prior of its analysed workspace (`recording_prior.json`); the
prior cached for the recording (`pipeline.cached_prior`), which any
analysis writes and every Find frames job now makes for each recording it
searches (`frame_search.estimate_lengths`, about a minute per recording
without one on an RTX A5500); and, as a rough guess, the median over the
setup's recordings with labels (684–827 px, median 765, on nir-flv). On four
recordings the analysis prior was within 2% of the labels' length. The
length used is saved with the label (`trace_length_px`), so rebuilding its
targets extends it the same way.
Before the toggle every such trace was extended, and the tail had no pull of
its own: on Alex's 52 traced
labels of 2026-10-08 the fitted tail missed the last click by a median 7 px
in view (max 18) and 170–230 px at or past the border; with both ends
pulled to their clicks it misses by at most 0.2 px, for a median IoU change
of −0.004 in view, −0.011 at the border (the tail taper now sits at the
edge) and −0.001 off camera. Pulling the off-camera part of a trace harder
(up to 100 times the in-view weight) did not bring off-camera clicks closer
where the visible mask disagrees with them, so off-camera trace points do
not pull.
Keeping the trace itself as the midline with a template width scaled to the
mask (`as_drawn`) fits the mask much worse (median IoU about 0.26 below the
refit on the chain-fit frames); the Labeling page does not offer it, and
only old records that used it keep it.

On the 16 chain-fit frames, simulated traces of 10 clicks with a quarter
body width of click noise recovered the poses to 0.17 body widths (median)
and the A-P field to 0.011; 6 clicks with 0.4 widths of noise lost about
0.09 IoU and left one pose off by more than half a width, so trace about
ten points, more around loops, and check the preview's IoU.

A part of the body that leaves the camera and re-enters cannot be traced
through; its pixels take the A-P value of the nearest traced segment.

## Correct on the Labeling page

The Labeling page's body tools work on the frame's current mask
([`APP_SIMPLIFICATION.md`](APP_SIMPLIFICATION.md), section 3):

- **Proposal.** The setup's body-field model predicts the frame's fields
  from its context frames, a trace follows from them
  (`body_proposal.propose_trace`) and is fit like a hand trace; it is drawn
  as the Proposal layer (dashed midline, dotted outline). When the server
  has a GPU it is computed as the frame opens (0.73 s median on an RTX
  A5500, 0.5 s of it the trace fit, whose steps replay as CUDA graphs; 4.3 s
  before the crossing exemption's reach was fixed per fit, which had kept
  them eager); without one (15–20 s on the CPU) only on **Propose**; a
  spinner on the frame shows while it is computed.
  **Use proposal** (G) takes it. A proposal made for an earlier mask is
  recomputed first.
- **Flip head/tail** (H) swaps the ends of the shown body.
- **Trace midline** (T): click from the head along the body to the tail,
  through crossings in the order the body goes, about 10 points and more
  around loops. The points are numbered from the head. A click adds a
  point at the tail, or at the head after clicking the head point (a ring
  marks the end being extended); a click on the line inserts a point there;
  dragging a point moves it and right-clicking one removes it; Backspace
  removes the end point and Esc cancels. Points may lie outside the frame:
  when the head and tail are in view but part of the body has left it,
  trace that part outside (zoom out to make room) and the fit follows it.
  The fitted head and tail sit on the first and last points; **Extend off
  camera** (X) instead continues a trace that ends at the border to the
  recording's typical body length, for a tail far off camera. **Fit** (Enter) refits along the
  trace and shows the fitted midline, the body outline, its A-P field and
  fit IoU, then **Accept** or **Discard**; moving the points after a fit
  drops it, to fit again. **Edit proposal** starts the trace from the
  proposal's points; accepting the edit makes it the proposal, so editing
  again (or **Use proposal**) starts from the edited version.
- **Mask only**: the body is unclear and the label trains the mask only.

Layers are the Mask, the A-P field (the body's, else the proposal's), the
Midline with head and tail (and the acquisition nose), and the Proposal;
`--dev` adds the overlap, the tube outline and a stored trace. The temporal
context shows the 33 context frames (Frames, with an offset and Play) or
the Difference channel the network sees for a lag of 1 to 16.

Saving writes a label revision with the body as decided (a trace, a head
end, or the automatic orientation) and starts a job that builds its targets.
The same operations are available headless: `POST /api/labeling/proposal`,
`/api/labeling/fit` and `/api/labeling/save` (`app/labeling.py`).

Coverage: `tests/test_body_fields.py` (record staleness, nose choice,
traces through a crossing, camera exits), `tests/test_body_proposal.py`,
`tests/test_library_targets.py`, `tests/test_queues.py` (the Labeling
backend, proposals with a stub network) and `tests/browser/labeling.cjs`.

## Body-field evidence in the pose fitter

To see what the network tells the fitter on a workspace's frames, turn on
the **A-P field** layer of the Workspace page (key 4; with `--dev` the layer
menu also has the network's crossings). The frame is predicted as the fit
stage predicts it (`GET /api/workspaces/{name}/network-fields`,
`app/network_fields.py`: `pipeline.workspace_frames` with the app's flat-field
cache, neighbours outside the recording as zero lag channels), and the ends
are judged by `body_proposal.field_evidence` against the row's current mask,
so an end the layer leaves out is one the fit does not use. The network is
the body model the workspace was analysed with; a workspace analysed without
one has no A-P layer.

The network's predictions also steer the tube fit of a recording. With
`--body-net <checkpoint>` (`scripts/fit_recording.py`) or the fit stage's
`body_net` parameter (`pipeline.FitParams`; in the pose app, Analyse sets it
to the weights of the setup's body model), every fitted frame is predicted
from its flat-fielded neighbours at the checkpoint's lags
(`body_proposal.RecordingFieldPredictor`, slab by slab), and the prediction
enters the fit four ways:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/fit_recording.py \
  --recording /store1/shared/all_data_raw/prj_aversion/2024-05-28/2024-05-28-02.h5 \
  --start 0 --frames 1200 --body-net <library>/models/nir-body-lags3/weights.ckpt
```

- **Evidence terms** (`batch_fit.fit_masks(fields=...)` with the
  `MaskFitConfig.field_*` weights; the evidence comes from
  `body_proposal.field_evidence`). The *A-P term* samples the predicted A-P
  field at every midline point of the front half of the body
  (`batch_fit.AP_ANTERIOR_FRACTION`) and compares it with the point's own
  position `s` along the body: the mean of `((A-P - s) / 0.05)^2` over the
  points where the field is known, times `FIELD_AP_WEIGHT` (0.001). The field is
  known on the cleaned mask, except on predicted crossings (overlap above
  0.5). At a contact the A-P field changes across the contact line, so a pose
  routed onto the wrong limb pays for it even where its overlap with the mask
  is the same, and a reversed pose pays everywhere. An A-P weight of 0.01 or
  more pulled fits off the mask edges. The labels' A-P runs to their own
  tail, which people place inconsistently, so the back half is left out (see
  [Head-anchored body of fixed length](#head-anchored-body-of-fixed-length)).
  The *end term* is the squared distance of the fit's first point from the
  predicted head, in units of 10 px, times `FIELD_END_WEIGHT` (0.01); the
  predicted tail only orients starts (below). An end counts only when
  its heatmap peak is at least 0.3, lies within 20 px of the cleaned mask
  (`END_MASK_PX`) and more than 20 px from the image edge (`END_BORDER_PX`).
  A peak farther off the mask sits on a part the cleanup removed, like the
  re-entering tail tip of tail_reentry_0623, and dragged the fit off the
  body; real tails lie up to 16 px off reviewed masks, which miss the thin
  tip. Where the body leaves the camera the tail heatmap fires (on edge_0528
  on 31 frames, switching on and off), and an end term pinning the tail there
  pulled the whole body into view. The evidence part of a fit's energy is
  stored per frame as `field_energy` (included in `total_energy`;
  `field_energy_independent` keeps the independent fit's). An edit that
  places, sets or flips a pose makes it NaN (not scored).
- **Trace start** (`body_proposal.trace_start`). The trace proposed from the
  A-P bands (`propose_trace`, as for the app's proposals) becomes a
  head-first start with the recording prior's width and width profile (the
  width is not measured across the mask, as it is for a hand trace or a fit
  without a prior). Where the head passes the end rules, the body is laid
  from the head along the bands, without the predicted tail, at exactly the
  prior's length (`mask_fit.lay_centerline`): a trace zigzagging between the
  turns of a spiral is cut there; a shorter one continues past its last band
  point along the mask's midline (a walk on the mask's distance transform
  that does not run back along the body laid so far), straight on past the
  mask's end, and off camera where it ends within 80 px of mask pixels on
  the image border, by the rule of the standard starts. The tail's position
  is an outcome, not an input. Without the head in view (4% of the sequence
  set's frames, nearly all in the tight spiral of spiral_0528), the trace
  runs from its first band to the tail and is lengthened off camera to the
  prior's length (`mask_fit.extend_start_to_length`: an end within 80 px of
  mask pixels on the image border). The 8 px last-point test of hand traces
  is not enough here: the last band point of a clipped body lies a median
  23 px from the edge (90th percentile 60 px, edge_0528), and fits from the
  short traces squeezed the whole body into view.
- **Fixed length.** With a recording prior, every fit of the recording keeps
  the prior's length (`MaskFitConfig.length_fixed`, set by
  `pipeline.fit_setup`): `batch_fit.fit_masks` first lengthens each start
  off camera and lays it from its first point to `length_prior_px`
  (`mask_fit.lay_start_to_length`), and the length is not optimized.
  Propagation's chains and the region algorithms set `length_prior_px` to
  the anchor's length, so a chain keeps its anchor's length (the prior's,
  unless the anchor is a pose placed by hand), and the track pass finds no
  length deviating from its track, so it refits nothing. Without a prior
  (and so in the held-out fits of earlier sections) the length is fit.
- **Orientation from the ends.** Where the evidence holds both ends (each
  passing the end rules above), every standard start is fit once, turned so
  that its first and last points lie nearer the predicted head and tail
  (`mask_fit.head_first`); otherwise it is fit both ways
  (`mask_fit.orientation_pair`). The trace is fit head first, and the
  result is not re-oriented by its taper afterwards: the evidence decides
  which end is the head. A frame started one way only has no
  `orientation_gap` (NaN).

Propagation and the track-length pass use the same evidence when the
workspace's fit used the network (the summary's `fit_params.body_net`, which
`fit_recording.py` writes too). Their rows are predicted 64 at a time (a
prediction is five full-image maps), and every fit of the passes (anchor
refits, stored-pose refits, chain steps, track refits) is scored against the
evidence. A candidate's comparable energy includes its evidence energy, so
the independent and chain candidates of a frame compete on the same footing,
and a mirrored option of the path pays the mirror's own evidence energy
(`propagation.Candidate.energy`), so the path cannot flip a pose against the
evidence for free. Each chain also fits the network's trace once per stretch
frame, as one more start of its best beam state, so the trace competes under
that state's temporal prior. The propagation summary counts
`frames_with_evidence`, `network_starts_offered` and `network_starts_won`.
The region algorithms use it too ([below](#region-algorithms)).

### Held-out poses

`scripts/evaluate_field_fitting.py` fits the hand masks of the reviewed
validation and test records four ways and scores the fits against the
reviewed poses: 32 traced records (hand-traced contacts and coils) and 34
other accepted records (ordinary postures). *Within ½ w* counts fits whose
visible ground-truth points lie a mean of less than half a body width from
the fitted midline; *A-P* is the mean error of the fit's rendered A-P field
over the mask; *head* is the fraction with the head at the true head end;
IoU is the median. On flv-c3
(`docs/pose_pipeline_fields/heldout_field_fitting.json`):

| Fitter | Traced: within ½ w | A-P | head | IoU | Accepted: within ½ w | A-P | head | IoU |
|---|---|---|---|---|---|---|---|---|
| standard starts, oriented by the taper | 15/32 | 0.261 | 0.72 | 0.613 | 34/34 | 0.132 | 0.71 | 0.964 |
| plus the trace start | 32/32 | 0.166 | 0.66 | 0.954 | 34/34 | 0.130 | 0.71 | 0.964 |
| evidence, both orientations, no trace | 5/32 | 0.209 | 1.00 | 0.389 | 34/34 | 0.008 | 1.00 | 0.961 |
| **evidence and trace (the pipeline)** | **32/32** | **0.025** | **1.00** | **0.946** | **34/34** | **0.014** | **1.00** | **0.962** |

Neither part works alone. The trace start finds the route through a contact
(it wins 31 of the 32 traced frames), but the taper still puts the head at
the wrong end in a third of the frames. The evidence fixes every head, but
from the standard starts it cannot move a fit to the right route and only
pulls it off the mask.

### Sequence set

The seven 300-frame clips of `docs/sequence_eval_set.json`
(`scripts/evaluate_sequence_set.py` with
`'--extra=--body-net checkpoints/body_net/best.ckpt'`), fit on flv-c3
A5500s; the result files are in `docs/pose_pipeline_fields/`. Each cell is
frames with IoU below 0.9 / pose jumps over a body width / length jumps over
3% / in-view changes (`propagation.continuity_summary`).

| Clip | Independent: current fitter | Independent: network | Propagation: current fitter | Propagation: network |
|---|---|---|---|---|
| spiral_0131 | 162 / 24 / 24 / 2 | 2 / 1 / 19 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 8 / 0 |
| loop_0131 | 53 / 4 / 9 / 27 | 16 / 2 / 23 / 21 | 0 / 1 / 0 / 11 | 0 / 0 / 4 / 7 |
| omega_0822 | 60 / 3 / 8 / 11 | 5 / 0 / 5 / 5 | 0 / 0 / 0 / 5 | 0 / 0 / 2 / 5 |
| coil_0822 | 128 / 7 / 19 / 14 | 12 / 0 / 13 / 14 | 1 / 0 / 1 / 6 | 0 / 0 / 1 / 2 |
| spiral_0528 | 138 / 15 / 18 / 5 | 48 / 3 / 64 / 5 | 0 / 0 / 0 / 3 | 1 / 0 / 0 / 3 |
| edge_0528 | 50 / 0 / 7 / 47 | 18 / 0 / 10 / 35 | 3 / 1 / 10 / 19 | 2 / 0 / 4 / 19 |
| tail_reentry_0623 | 8 / 1 / 4 / 8 | 6 / 1 / 4 / 6 | 2 / 0 / 3 / 2 | 5 / 1 / 5 / 4 |
| **total** | **599 / 54 / 89 / 114** | **107 / 7 / 138 / 86** | **6 / 2 / 14 / 46** | **8 / 1 / 24 / 40** |

The columns are `sequence_eval_independent_base.json`,
`sequence_eval_independent_fields_fixed.json`,
`sequence_eval_propagate_base.json` and
`sequence_eval_propagate_fields_fixed.json`. The other files are the steps
below, with propagation and the network unless named otherwise:
`sequence_eval_independent_fields.json` and
`sequence_eval_propagate_fields.json` (before the end rules and the
extended trace), `_fix2_on0` (the extended trace), `_fix2b_on0` (plus ends
at the image edge ignored), `_fix3_on2b` (plus ends off the mask ignored),
`_fields_fix1` (the tail dropped from the end term, on the state before the
end rules), and `_fields_fix4_on2b3` and `_base_fix4` (the track-pass
guard).

- **Independent fits.** The network removes most single-frame failures on
  coils and spirals (frames below 0.9: 599 to 107; pose jumps 54 to 7). More
  lengths jump (89 to 138), because a fit takes the length of its trace's
  shape (not of the start: see the prior's length below). In the tight inner turns of spiral_0528 a band's nearest patch can
  lie on the neighbouring turn, so the trace zigzags between turns and comes
  out a median 12% too long, while a clean trace runs through the band
  centres and is about 8% short; frames alternating between the two make
  most of its 64 length jumps. Propagation removes them.
- **With propagation.** The current fitter already gets most frames right
  here; with the network there are fewer pose jumps and in-view changes, 2
  more frames below 0.9 and 10 more length jumps. The network leaves fewer
  frames ambiguous (740 stretch frames against 974), so more independent
  fits reach the output, and their lengths jitter: before the end rules,
  the frame-to-frame change outside the stretches was a median 0.005-0.007
  in log length against 0.002-0.003 without the network, and the tail's end
  term caused it (spiral_0131 independent fits: 0.0062, 0.0029 without any
  end term, 0.0033 with the head's alone). Dropping the tail from the end
  term took the seven clips from 39 / 6 / 46 / 62 to 20 / 0 / 18 / 58, but
  the held-out traced A-P error rose from 0.025 to 0.034 (a 20 px tail
  sigma instead of 10 gave 0.026), so the tail term stayed then; it was
  dropped with the head-anchored body of fixed length
  ([below](#head-anchored-body-of-fixed-length)). Without the head's term
  the head drifts at the camera edge (edge_0528 independent pose jumps 13
  to 41).
- **The end rules and the extended trace.** Before them, the network with
  propagation scored 39 / 6 / 46 / 62 (`sequence_eval_propagate_fields.json`),
  worse than the current fitter on every count. Lengthening the trace off
  camera and ignoring ends at the image edge took edge_0528 from 22 / 2 / 9 /
  31 to 2 / 0 / 4 / 19; ignoring ends off the mask took tail_reentry_0623
  from 14 / 0 / 10 / 4 to 5 / 1 / 5 / 4 (`*_fix2*`, `*_fix3*`).
- **The trace start at the prior's width and length.** Until 2026-10-08 the
  trace start took the width measured across the mask along the trace and
  only the prior's width profile. With the prior's width as well the seven
  clips went from 107 / 7 / 138 / 86 to 104 / 9 / 146 / 84 independently
  and from 8 / 1 / 24 / 40 to 7 / 1 / 23 / 34 with propagation (edge_0528
  in-view changes 19 to 15, tail_reentry_0623 4 to 2;
  `sequence_eval_{independent,propagate}_trace_prior_width.json`, base
  re-run on the same flv-c3 A6000 with identical counts to the `_fixed`
  files). Scaling a trace in view to the prior's length as well (its
  latent length, the traced shape kept) did not take the length jumps
  away: on spiral_0528 the fits from a start at exactly the prior's length
  still came out 0.91-0.95 or 1.06-1.12 of it, the length of the traced
  shape (zigzagging between turns or cutting through them), not of the
  start; it gave 113 / 8 / 148 / 84 and 7 / 1 / 22 / 34
  (`*_trace_scaled.json`), 55 frames below 0.9 on spiral_0528 against 47,
  and was not adopted: the head-anchored body
  ([below](#head-anchored-body-of-fixed-length)) cuts or continues the trace
  at the prior's length instead of scaling it, and holds that length in the
  fit. Neither change moves the held-out table, which fits
  without a prior (the scaling: traced IoU 0.946 either way, accepted 0.962
  to 0.961).
- **Track-pass guard, not adopted.** Refitting a converged clipped pose with
  the track's length prior pulls its off-camera part into view whatever the
  schedule, prior or evidence (before the end rules, 16 of edge_0528's 22
  frames below 0.9 were such refits). Keeping the stored pose when a refit
  loses more than 0.01 IoU took the six clips that reach the image edge from
  8 / 1 / 16 / 40 to 4 / 1 / 26 / 28, and without the network from
  6 / 2 / 14 / 46 to 6 / 2 / 34 / 46: the refits it rejects are the ones
  correcting the length.

### Masks from the network

The network predicts a mask too, from the same temporal context. The
segment stage can take its masks from it instead of the segmenter
(`pipeline.SegmentParams.mask_source`: `segmenter`, the default, or
`body_net`, which reads the `body_net` checkpoint;
`scripts/fit_recording.py --mask-source body_net --body-net <checkpoint>`;
in the pose app, **Masks from** in the Analyse dialog, offered when a body
model is chosen). The same cleanup applies (threshold 0.5, narrow holes
filled, largest component kept), and the prior's bootstrap takes its masks
from the same source (its cache file gets a `_body_net` suffix). With the
network as the mask source an analysis needs no mask model: the workspace
settings record `mask_source`, no mask model and no segmenter checkpoint,
and the summary's `checkpoint` is the network's. The region algorithms
segment rows the same way (`algorithms._segment_params`); the `--dev`
probability layer, which runs the segmenter on one frame, is left out.
`fit_recording.py` fits each slab with the predictions that made its masks
(`pipeline.BodyNetMasks.last`) instead of predicting it again; the
workspace stages predict again in the fit stage.

Against the 13 labels of `nir-v1` (the test recordings;
`mask_source_benchmark_iou.json`, mean / worst label), raw masks
(probability ≥ 0.5): segmenter nir-hand284 0.988 / 0.979, network
nir-body-lags3 0.980 / 0.970; after the cleanup 0.984 / 0.977 against
0.976 / 0.966. On the seven clips the two cleaned masks agree to a median
IoU of 0.978–0.989 per clip.

The sequence set with each mask source, both with the network's evidence
and propagation, the library's nir-hand284 and nir-body-lags3 and no prior
cache, on one flv-c3 A5500 (`sequence_eval_mask_segmenter.json`,
`sequence_eval_mask_body_net.json`; `mask_source_scores.json`). Each cell is
frames with IoU below 0.9 against the run's own masks / pose jumps / length
jumps / in-view changes, then frames below 0.9 when the final poses of both
runs are rendered against the segmenter's cleaned masks:

| Clip | Segmenter masks | Network masks | Segmenter masks, scored on segmenter masks | Network masks, scored on segmenter masks |
|---|---|---|---|---|
| spiral_0131 | 1 / 1 / 3 / 0 | 1 / 1 / 6 / 0 | 1 | 1 |
| loop_0131 | 0 / 0 / 2 / 7 | 3 / 0 / 7 / 9 | 4 | 6 |
| omega_0822 | 0 / 0 / 0 / 5 | 0 / 0 / 3 / 5 | 0 | 1 |
| coil_0822 | 0 / 0 / 0 / 2 | 0 / 0 / 2 / 2 | 2 | 3 |
| spiral_0528 | 1 / 0 / 0 / 3 | 1 / 0 / 3 / 3 | 1 | 1 |
| edge_0528 | 11 / 0 / 10 / 25 | 1 / 0 / 10 / 24 | 38 | 14 |
| tail_reentry_0623 | 2 / 0 / 1 / 4 | 3 / 0 / 12 / 2 | 2 | 3 |
| **total** | **15 / 1 / 16 / 46** | **9 / 1 / 43 / 45** | **48** | **29** |

- **Fewer poor fits, nearly all on edge_0528.** The network's masks take
  the fewest frames below 0.9 there (11 to 1, and 38 to 14 against the
  segmenter's masks); on the other six clips there are more, 8 against 4
  (15 against 10). Independent fits below 0.9 fall from 125 to 81, 65 to 35 of them on
  spiral_0528.
- **Lengths jump more** (16 to 43, 11 of them on tail_reentry_0623). The
  network's masks change more from frame to frame: the median change of the
  mask's log area between frames is 0.0023–0.0045 per clip against
  0.0020–0.0032, and every clip but edge_0528 has more length jumps.
- **Cost.** The segmenter's pass (11 ms per frame) and the fit's prediction
  (23 ms) become one prediction (21 ms); propagation, with 785 stretch
  frames against 722, takes 104 ms per recording frame against 93.

The segmenter stays the default: the network's masks trade the camera-edge
failures for worse fits on the other clips and jittery lengths, and the
segmenter's masks score better on the held-out labels.

### Cost

Mean ms per frame over the seven clips, by stage (other jobs shared flv-c3,
so these are rough):

| Stage | Independent: current | Independent: network | Propagation: current | Propagation: network |
|---|---|---|---|---|
| read, segmenter, cleanup, video | 41 | 40 | 40 | 40 |
| body-field network | – | 18 | – | 17 |
| init (starts) | 17 | 116 | 17 | 116 |
| fit | 216 | 331 | 215 | 328 |
| propagate | – | – | 564 | 571 |
| track | 26 | 25 | 13 | 17 |
| **total** | **301** | **529** | **849** | **1089** |

- The fit costs 1.5 times as much: with the recording prior the standard
  starts are already the skeleton both ways, and the trace is a third start
  for every frame.
- Most of the added init time was the trace start's width measurement
  (`anchors.estimate_width_along_normals` in `init_from_centerline`, about
  70-80 ms per frame); `propose_trace` takes about 8.5 ms and
  `field_evidence` 0.6 ms per frame on the flv-c3 CPU. Since the walk was
  vectorized the measurement takes 1.1 ms per frame (median over 70
  sequence-set poses), and init with the network costs 20 ms per frame
  (re-run on flv-c3, 2026-10-08, with and without the measurement); with a
  prior the trace start now skips it.
- Propagation costs 1620 ms per stretch frame against 1215: the chains fit
  11.2 rows per stretch frame against 6.3, partly the network's trace (1452
  offered, 195 won). With fewer stretch frames the pass costs about the same
  per recording frame.
- The evidence holds a full-image A-P map (2.8 MB at 732×968) for every row
  of a pass for the whole pass, and propagation runs with the network logged
  CUDA caching-allocator retries of about 1.5 GB on the 24 GB A5500s
  without failing.

### One orientation where the ends are clear

Since the evidence put the head right on every held-out frame, a frame whose
head and tail both pass the end rules fits its standard starts head first
only (see *Orientation from the ends* above). The propagation and track
passes refit warm poses in one orientation and are unchanged. Measured on
flv-c3 against the state before, with `nir-body-lags3` (the lab model of
nir-flv; result files in `docs/pose_pipeline_fields/orientation_starts/`):

| | Before | One orientation |
|---|---|---|
| frames started one way | – | 1019 / 2100 (edge_0528: 0) |
| starts per frame | 3.00 | 2.47 |
| independent: frames < 0.9 / pose / length / in-view | 142 / 44 / 180 / 78 | 143 / 44 / 179 / 78 |
| propagation: frames < 0.9 / pose / length / in-view | 21 / 1 / 17 / 44 | 20 / 1 / 15 / 44 |
| held-out traced: within ½ w / A-P / head / IoU | 30/32 / 0.038 / 1.00 / 0.948 | 30/32 / 0.038 / 1.00 / 0.948 |
| held-out accepted: within ½ w / A-P / head / IoU | 34/34 / 0.016 / 1.00 / 0.962 | 34/34 / 0.016 / 1.00 / 0.962 |
| fit ms per frame (independent runs, propagation runs) | 29.8, 39.5 | 26.2, 30.0 |

With the recording prior the standard starts are the skeleton alone, so a
frame drops from three starts to two. Every clip but one differs from
before by at most one frame or jump; edge_0528, where the body leaves the
camera, never has both ends and is fit exactly as before. The held-out run
of `scripts/evaluate_field_fitting.py` builds the starts as the fit stage
does (before, it also fit the trace reversed), and its scores are unchanged
to the third decimal. The fit time is the
batched GPU fit of the fit stage, on an A6000 shared with other jobs, so
the saving (a tenth to a quarter) is rough. These runs used the lab model
rather than `checkpoints/body_net/best.ckpt`, so their numbers differ from
the tables above.

### Region algorithms

The Workspace page's Refit and Relabel fixes (`algorithms.py`) use the
network the way propagation does when the workspace's fit used it
(`fit_params.body_net`). `algorithms.build_context` predicts every row of the
region and its anchors (`pipeline.workspace_predictions`,
`pipeline.body_field_inputs`) into `RegionContext.evidence`, and for the
algorithms that fit masks the trace start of every region row into
`RegionContext.network_starts`. Then:

- every fit is scored against the evidence (`fit_masks(fields=...)`):
  `independent_multistart`, `slow_refit`, `tracked_head`, the frames no
  chain reached, and the chains of `chain_forward`, `chain_backward`,
  `beam_path` and `stitch`, which pass the evidence and the trace starts
  on to `propagation.propagate`;
- the trace is one more start wherever an algorithm builds starts:
  every frame of `independent_multistart` (as its own candidate, and the
  standard starts in both orientations, as in the fit stage), the chains
  (once per chain and frame), the frames no chain reached, the frames
  `slow_refit` has no pose for, and `tracked_head` (oriented toward the
  tracked head like its other starts);
- every candidate, including the stored poses of `mirror`, the smoother's
  placed poses and the keyframes, is scored against its frame's evidence
  and its mirror's (`algorithms.score_evidence`): `CandidatePose.energy`
  includes `field_energy`, and `CandidatePose.mirrored` (and so a mirrored
  node of the path) pays `mirror_field_energy` instead.

The smoother's joint energy does not take the evidence. Its LM solver can:
the end term is a squared distance of the chain's first and last points,
and the A-P term a residual whose Jacobian is the field's bilinear gradient
times the point's, so both linearise like the data term. On the
stretches below, weighted 100 times the fit's weights they changed the
frames below IoU 0.9 from 276 to 271 and the pose jumps from 5 to 6;
1000 to 100000 times made both worse (287–302 frames, 14–30 jumps), and
the A-P term alone (293 and 304 at 100 and 1000) or the ends alone (274 and
287) did no better (`region_algorithms_smoother_sweep.json`), so they were
not kept. The untrusted frames the evidence would place are coils and
contacts, where a fixed-length body bridged from its neighbours misses the
mask whatever pulls its ends.

The fixed_body stage (`fixed_body.run_fixed_body`) fits a chain to each
stored pose, not to a mask, so it has nothing to score.

Measured with `scripts/evaluate_region_algorithms.py` on flv-c2 A5500s:
the seven clips analysed into workspaces with the network
(`nir-body-lags3`, the default stages), then every algorithm run on each of
the 33 stretches the propagate stage refit (735 frames), anchored on the
nearest good frames (`propose_anchors`), with the code before
(`region_algorithms_base.json`) and after (`region_algorithms_evidence.json`).
flv-c2 cannot read `/store1`, so the clips were copied with 16 frames of
margin under each recording's own file name, which reuses the recording's
cached flat field and prior. Each cell sums `region_metrics` over the
stretches with the path applied: frames below IoU 0.9 / pose jumps over a
body width / length jumps over 3% / orientation flips. Before any run the
stretches score 6 / 1 / 5 / 1 (propagation's result).

| Clip | independent_multistart: before | after |
|---|---|---|
| spiral_0131 | 110 / 18 / 7 / 2 | 10 / 9 / 20 / 3 |
| loop_0131 | 47 / 6 / 10 / 1 | 5 / 4 / 14 / 0 |
| omega_0822 | 60 / 3 / 2 / 1 | 11 / 3 / 25 / 0 |
| coil_0822 | 101 / 11 / 14 / 0 | 16 / 0 / 26 / 0 |
| spiral_0528 | 138 / 5 / 3 / 0 | 65 / 29 / 72 / 10 |
| edge_0528 | 2 / 6 / 11 / 0 | 0 / 0 / 7 / 0 |
| tail_reentry_0623 | 5 / 3 / 5 / 0 | 7 / 0 / 3 / 0 |
| **total** | **463 / 52 / 52 / 4** | **114 / 45 / 167 / 13** |

| Algorithm (all clips) | before | after |
|---|---|---|
| beam_path | 3 / 0 / 2 / 0 | 4 / 1 / 3 / 0 |
| slow_refit | 3 / 2 / 2 / 0 | 4 / 1 / 4 / 2 |
| fixed_body_smoother | 276 / 5 / 4 / 0 | 276 / 5 / 4 / 0 |
| mirror | 6 / 1 / 5 / 1 | 6 / 1 / 5 / 1 |

- **Independent multi-start** gains what the independent fit gains: frames
  below 0.9 fall from 463 to 114, and lengths jump more (52 to 167), since a
  fit inherits its trace's length. On spiral_0528 the trace wins 137 of the
  141 frames of the long stretch, 31 of them 840–1050 px long against about
  750 on the clean frames (the zigzag between turns above), and the routes through
  the inner turns make most of its pose jumps and all ten orientation
  flips: their ends swap places relative to the frame before without the
  evidence preferring either orientation much (a mirror costs 0.03–0.4
  more, against 7–23 on the clean frames).
- **Beam and path, slow refit** were already close to the propagate stage's
  result and stay there; they now reproduce it, which on spiral_0131 keeps
  its one frame below 0.9 (a trace start of 776 px chosen by a chain at
  frame 214) that the runs without evidence did not have.
- **Mirror** chooses the same orientation everywhere: the anchors decide it
  on these stretches, and the evidence agrees.
- `tracked_head` was not measured: the clip copies carry no acquisition
  nose tracking.
- Run time without the prediction: 95 s against 79 s for
  `independent_multistart` over the 735 frames, 215 s against 187 s for
  `beam_path`, 19 s against 17 s for `slow_refit`. `build_context` predicts
  the region and its anchors first (64 rows at a time, about 18 ms a row)
  and measures the trace starts (about 0.1 s a row, skipped for `mirror`
  and the smoother); a region's evidence holds a 2.8 MB A-P map per row, so
  a region of a few thousand frames needs several GB of host memory.

### Combined

The trace start's prior width, one orientation where the ends are clear,
the region algorithms' evidence and the body-net mask toggle together,
against the state before any of them, on flv-c3 A5500s with
`nir-body-lags3` and `checkpoints/segmenter/best.ckpt`, each run with a
fresh prior cache (result files in `docs/pose_pipeline_fields/integration/`).
Cells as in the sequence set above; the held-out rows are the pipeline's
`both` variant of `scripts/evaluate_field_fitting.py`. Frames below 0.9 of
the body-net-mask runs are scored against their own masks.

| | Before | Combined | Combined, `--mask-source body_net` |
|---|---|---|---|
| independent: frames < 0.9 / pose / length / in-view | 142 / 44 / 177 / 84 | 148 / 46 / 173 / 86 | 144 / 22 / 237 / 102 |
| propagation: frames < 0.9 / pose / length / in-view | 15 / 1 / 16 / 46 | 14 / 1 / 16 / 48 | 7 / 1 / 45 / 47 |
| held-out traced: within ½ w / A-P / head / IoU | 30/32 / 0.038 / 1.00 / 0.948 | 30/32 / 0.038 / 1.00 / 0.948 | – |
| held-out accepted: within ½ w / A-P / head / IoU | 34/34 / 0.016 / 1.00 / 0.962 | 34/34 / 0.016 / 1.00 / 0.962 | – |
| ms per frame (propagation runs): masks / fields / init / fit / propagate / track | 18.4 / 17.5 / 21.0 / 28.6 / 78.5 / 4.7 | 18.4 / 16.8 / 20.0 / 24.6 / 76.3 / 4.6 | 28.2 / 0 / 19.6 / 25.3 / 77.6 / 4.7 |

The propagated result, the pipeline's output, is unchanged by the combined
fit-stage changes (every clip within one frame or two in-view changes), and
the fit is 14% faster. The six more independent frames below 0.9 come from
the trace start's prior width alone (that branch by itself scores
148 / 46 / 174 / 84, `seq_trace_independent.json`), almost all on
spiral_0528, and propagation absorbs them. The body-net masks repeat their
own section's result: half the frames below 0.9 (edge_0528 11 to 2) and
three times the length jumps (16 to 45; tail_reentry_0623 1 to 12), with
one prediction of 20 ms a frame replacing the segmenter, its cleanup and the
separate fields pass (masks here are the read, flat-field, network and
cleanup times; with body-net masks the cleanup alone is 8 ms of it). The
segmenter stays the default.

### Head-anchored body of fixed length

With the network, fits jumped in length from frame to frame (the combined
state above: 173 length jumps over 3% independently, 237 with body-net
masks; `independent_multistart` 52 to 167). The network's tail is the
unreliable part: people cannot place the tail consistently when they trace
labels (a trace fit pins it to the last click), while the head is reliable.
The tail entered the fit three ways: the trace start ended at the predicted
tail, the end term pulled the fit's last point there, and the A-P field,
normalized by each label's own length, pulled on the fit's length through
the A-P term. Now the body is anchored at the network's head and has the
recording prior's length (*Trace start* and *Fixed length* above), the end
term scores the head only, and the A-P term scores the front half of the
body. Measured as ablations against the combined state, on flv-c3
(A5500s; C's sequence runs and the final held-out run on its A6000),
with `nir-body-lags3`, `checkpoints/segmenter/best.ckpt`, a fresh prior
cache per run and `--ambiguous-frames 0` (result files in
`docs/pose_pipeline_fields/head_anchored/`; the base re-run reproduces the
combined state's counts exactly):

- **A**: the trace start laid from the head at the prior's length (fits
  free in length, end term on both ends, A-P term on the whole body);
- **B**: A, and every fit keeps the prior's length;
- **C**: B, and the end term on the head only;
- **D1**: C, and the A-P term compared up to a per-fit scale (the scale
  that best maps `s` to the field's A-P, so a label's length does not
  matter);
- **D2**: C, and the A-P term on the front half of the body only (adopted;
  "final" is D2 as committed, whose walk computes its distance transform
  only within its reach and whose encoded trace is set to the length
  without a second walk, the rules unchanged).

Cells are frames with IoU below 0.9 / pose jumps over a body width /
length jumps over 3% / in-view changes; held-out cells are the pipeline's
`both` variant of `scripts/evaluate_field_fitting.py` (within ½ w / A-P /
head / IoU), which now lays the trace and fixes the length at each sample's
recording length (`body_fields.recording_length`) as the fit stage does
with a prior.

| | Independent | Propagation | median IoU (propagation, per clip) | Held-out traced | Held-out accepted |
|---|---|---|---|---|---|
| combined state (before) | 148 / 46 / 173 / 86 | 14 / 1 / 16 / 48 | 0.963–0.971 | 30/32 / 0.038 / 1.00 / 0.948 | 34/34 / 0.016 / 1.00 / 0.962 |
| A | 165 / 60 / 162 / 78 | 11 / 0 / 22 / 50 | 0.963–0.971 | 31/32 / 0.038 / 1.00 / 0.946 | 34/34 / 0.017 / 1.00 / 0.962 |
| B | 284 / 83 / 0 / 54 | 52 / 1 / 0 / 42 | 0.952–0.965 | 30/32 / 0.034 / 1.00 / 0.909 | 34/34 / 0.029 / 1.00 / 0.953 |
| C | 151 / 88 / 0 / 54 | 4 / 2 / 0 / 38 | 0.955–0.968 | 30/32 / 0.043 / 1.00 / 0.936 | 34/34 / 0.030 / 1.00 / 0.959 |
| D1 | 149 / 84 / 0 / 54 | 3 / 2 / 0 / 36 | 0.956–0.968 | 31/32 / 0.043 / 1.00 / 0.938 | 34/34 / 0.030 / 1.00 / 0.959 |
| D2 | 159 / 83 / 0 / 54 | 4 / 2 / 0 / 38 | 0.962–0.968 | 31/32 / 0.041 / 1.00 / 0.951 | 34/34 / 0.029 / 1.00 / 0.961 |
| **final** | **151 / 90 / 0 / 52** | **4 / 2 / 0 / 34** | **0.962–0.968** | **30/32 / 0.043 / 1.00 / 0.949** | **34/34 / 0.029 / 1.00 / 0.962** |
| before, `--mask-source body_net` | 144 / 22 / 237 / 102 | 7 / 1 / 45 / 47 | 0.958–0.968 | – | – |
| final, `--mask-source body_net` | 125 / 78 / 0 / 49 | 6 / 2 / 0 / 31 | 0.960–0.968 | – | – |

The three clips where the variants differ most (independent; propagation):

| | spiral_0528 | edge_0528 | tail_reentry_0623 |
|---|---|---|---|
| before | 68 / 31 / 70 / 7; 0 / 0 / 0 / 3 | 28 / 2 / 12 / 37; 11 / 0 / 9 / 25 | 5 / 0 / 2 / 6; 2 / 0 / 1 / 4 |
| A | 84 / 42 / 70 / 5; 1 / 0 / 0 / 3 | 15 / 0 / 12 / 27; 6 / 0 / 12 / 19 | 5 / 0 / 0 / 6; 2 / 0 / 1 / 2 |
| B | 123 / 58 / 0 / 15; 2 / 0 / 0 / 15 | 10 / 0 / 0 / 11; 0 / 0 / 0 / 3 | 33 / 0 / 0 / 4; 34 / 0 / 0 / 2 |
| C | 94 / 62 / 0 / 15; 1 / 0 / 0 / 15 | 10 / 0 / 0 / 11; 0 / 0 / 0 / 3 | 4 / 0 / 0 / 4; 1 / 0 / 0 / 2 |
| D1 | 92 / 57 / 0 / 15; 0 / 0 / 0 / 13 | 10 / 0 / 0 / 11; 0 / 0 / 0 / 3 | 4 / 0 / 0 / 4; 1 / 0 / 0 / 2 |
| D2 | 100 / 56 / 0 / 15; 1 / 0 / 0 / 15 | 10 / 0 / 0 / 9; 0 / 0 / 0 / 3 | 4 / 0 / 0 / 4; 1 / 0 / 0 / 2 |
| final | 96 / 59 / 0 / 13; 1 / 0 / 0 / 11 | 10 / 0 / 0 / 9; 0 / 0 / 0 / 5 | 4 / 0 / 0 / 4; 1 / 0 / 0 / 2 |

- **The propagated result**, the pipeline's output, has no length jumps
  and fewer poor frames: 14 to 4 below 0.9 (edge_0528 11 to 0) and 48 to
  34 in-view changes, for one more pose jump on spiral_0131. On frames whose
  whole body is in view the length's frame-to-frame change had a standard
  deviation of 0.007 in log length (0.029 independently); it is now 0. The
  tail, now an outcome, moves less between frames (median step 6.0 px
  against 7.4, edge_0528 5.7 against 9.0; the head's 4.8 against 4.7).
  The network leaves fewer stretches to propagate (619 frames against
  725), and the track pass, which finds no length off its track, refits
  nothing (304 frames before).
- **Each part needs the next one.** Laying the trace at the prior's length
  (A) alone takes few length jumps away, since the fits are still free
  and the end term still pulls the last point to the predicted tail; with
  the length held (B) that pull fights a body that cannot reach it and
  drags it across the mask (tail_reentry_0623 2 to 34 propagated frames
  below 0.9, spiral_0131 1 to 11); dropping the tail from the end term (C)
  removes it. Before the trace was continued along the mask, the
  continuation went straight along the end tangent, and A, B and C scored
  217 / 64 / 180 / 74, 365 / 83 / 0 / 50 and 184 / 88 / 0 / 52
  independently (`seq_*nowalk_independent.json`): where the A-P field
  flattens toward a label's early tail the bands stop up to 100 px short of
  the body's end, and a straight continuation left the mask.
- **The A-P term.** With a fixed length the whole-body A-P term pulls
  against the body wherever a label's tail was placed early or late: B and
  C lost median IoU on every clip (0.952–0.968 against 0.963–0.971). A
  per-fit scale (D1) gave the same counts with that lower IoU (held-out
  traced 0.938); the front half alone (D2) brings the median IoU back to
  0.962–0.968 and the held-out traced IoU to 0.951, with the same counts.
- **Independent fits** pay for the fixed length: a body that is shorter in
  the mask than the prior's length (a contracted body, a mask missing the
  thin tail tip) has to put the rest somewhere, so the independent fits
  jump more in pose (46 to 90, spiral_0528 31 to 59) and spiral_0528 has
  more poor frames (68 to 96); propagation removes both. The excess also
  goes off camera where it can: on spiral_0528, whose mask is clear of the
  border until its last 40 frames, the tail tip leaves the image on frames
  where the mask does not reach it (around frames 9113–9130; in-view
  changes 3 to 11), and edge_0528 has 19 whole-body frames against 37.
- **Without the head** (heatmap peak below 0.3 or at the image edge) the
  trace runs to the tail as before and the fit keeps the prior's length.
  On the sequence set's frames that is 92 of 2100 (counted on the
  network's own masks, `head_counts.json`): 80 on spiral_0528, 10 on
  spiral_0131, 2 on coil_0822; no head was at the image edge (edge_0528's
  body leaves the camera tail first).
- **Held-out poses.** The traced contacts and coils are fit as well as
  before (30/32 within half a width, IoU 0.949 against 0.948, every head
  right). The A-P error rose (traced 0.038 to 0.043, accepted 0.016 to
  0.029): the error compares A-P normalized by the fit's length, now the
  recording's, with A-P normalized by the label's own traced length.
- **Region algorithms** (`scripts/evaluate_region_algorithms.py` on
  flv-c3, workspaces analysed by each version, so the stretches differ: 33
  stretches of 727 frames before, 27 of 624 after; `region_base.json`,
  `region_final.json`): `independent_multistart` goes from
  116 / 50 / 170 / 17 to 145 / 90 / 0 / 20 (frames below 0.9 / pose jumps /
  length jumps / orientation flips), its poor frames and pose jumps again
  mostly on spiral_0528 (96 / 59); `beam_path` 3 / 1 / 4 / 1 to
  3 / 2 / 0 / 2, `slow_refit` 4 / 1 / 5 / 2 to 5 / 2 / 0 / 2, `mirror`
  4 / 1 / 7 / 1 to 4 / 2 / 0 / 2 (each at its stretches' own starting
  score).
- **Cost** (ms per frame, propagation runs): init 20.4 to 23.4 (the walk
  along the mask computes a distance transform around the trace's end) and
  fit 25.7 to 33.8 (every start is laid to the length first; for the
  standard starts, whose skeleton ends stop short of the body's ends, that
  is a walk too), but propagation 80.7 to 50.5 and the track pass 4.9 to 0:
  165 against 191 in all. Independent runs: 113 against 110.
- **Next: a tight soft length prior.** The frozen length cannot tell a
  real contraction from a mask missing the tail tip, which is what the
  independent fits pay for above. A Gaussian prior on log length centred
  on the recording's length with a sigma of 1–2% (`length_prior_px` with a
  small `length_prior_log_sigma` instead of `length_fixed`) would let a
  contracted body shorten a little instead of folding the excess into a
  coil or off camera. Not measured yet; compare it with the frozen length
  on the sequence set (independent pose jumps, spiral_0528) and hold the
  length only on frames whose head passes the end rules, not per recording.
