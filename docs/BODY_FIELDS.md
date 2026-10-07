# Body-field targets: build, review, correct

The body-field network (`src/worm_pose_gen/body_net.py`) predicts, besides
the worm mask, a per-pixel A-P field (normalized arc length, 0 at the head,
1 at the tail), head and tail heatmaps, and an overlap map (pixels covered
twice by body parts far apart along the body), from the flat-fielded frame
plus symmetric difference channels `frame[t+lag] - frame[t-lag]`. A hand
label only says which pixels are worm, so the other targets come from
fitting the tube model to the hand mask and orienting it head first
(`body_targets.py`). A wrong head end or a bad fit silently teaches the
network the wrong thing, so every record can be inspected and corrected in
the app. The pose fitter can score its fits against the network's
predictions and start them from its traces
([Body-field evidence in the pose fitter](#body-field-evidence-in-the-pose-fitter)).

## Records

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
`independent_fit_iou`. Edits in the app add or change:

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

Build or refresh the records of a store (samples whose record is current
are skipped; `--force` rebuilds them, `--samples` limits the run):

```bash
scripts/project_env.sh uv run --no-sync python scripts/build_body_fields.py --review
```

## Correcting a fit by tracing the midline

Where the mask is right but the fitted body takes the wrong route through a
crossing or contact, the head, tail and A-P targets are all wrong with it.
`body_fields.apply_trace(store, sample_id, points)` takes points clicked
along the body from head to tail, through the crossing in the order the
body actually goes, and refits the tube from them: the trace is the start,
every point of the fit is pulled toward the same-numbered point of the
resampled trace (weight 0.02 at half a body width) and the head toward the
first click, and the tube is otherwise fit to the hand mask, so it keeps
the traced route but finds the body's edges and width itself. The length
bounds do not apply (the trace sets the length), and a non-interpenetration
term (`MaskFitConfig.separation_weight` 1.0 at full separation) keeps two
parts of the body far apart along it from sitting inside each other: the
rendered tube is a union, so without it a head or tail touching the body
slid onto it at no cost. Pairs the trace itself puts on top of each other,
and their neighbours within a body width along both runs, are a deliberate
crossing and exempt. On 28 traced records it cut the median overlap from
252 to 92 px (mean 391 to 134) for 0.002 median IoU; pinning the width to
the recording's median profile instead made the overlap worse. A trace
whose last point is within 8 px of the image border is continued straight
off camera to the recording's typical body length (the median of its whole,
well-fit records from the same recording file), so a body leaving the
camera is not squeezed into view.
`as_drawn=True` keeps the trace itself as the midline with a template width
scaled to the mask; it fits the mask much worse (median IoU about 0.26 below
the refit on the chain-fit frames) and is a fallback only. Without `commit=True` the
result is a preview and nothing is written.

On the 16 chain-fit frames, simulated traces of 10 clicks with a quarter
body width of click noise recovered the poses to 0.17 body widths (median)
and the A-P field to 0.011; 6 clicks with 0.4 widths of noise lost about
0.09 IoU and left one pose off by more than half a width, so trace about
ten points, more around loops, and check the preview's IoU.

A part of the body that leaves the camera and re-enters cannot be traced
through; its pixels take the A-P value of the nearest traced segment.

## Review in the app

The **Body fields** tab of the pose app shows the records of the app's
corpus store, so point `--corpus-root` at the segmentation store:

```bash
scripts/project_env.sh uv run --no-sync python -m worm_pose_gen.app \
  --corpus-root /temp_data4/alex/external_artifacts/datasets/worm_pose_gen/segmentation_v1
```

`--body-net` names the body-field network that proposes traces (default
`checkpoints/body_net/best.ckpt`); it is loaded on the app's device the first
time a sample is proposed.

The left column lists every sample with its status (current, stale, or
missing), split, head source, fit IoU, self-contact and overlap, filtered by
any of these, by review status, by fit method and by whether it has a
proposal; a traced record carries a *traced* (or *trace as drawn*) badge and
its automatic fit's IoU, and a record with a proposal a *proposal* badge and
the proposal's fit IoU (*old proposal* when the mask changed after it was
made). The canvas shows the labeled frame with
toggleable layers: hand mask, A-P field (viridis, purple head to yellow
tail), overlap pixels (white), fitted tube outline and centerline, head
(green) and tail (red) markers, the acquisition nose (yellow cross), and a
traced record's stored trace (dashed orange, numbered from the head), and
the network's proposal: its tube dotted violet, its ends ringed violet and
its trace points as violet dots. **Proposal A-P field** shows the proposal's
A-P field and overlap in place of the record's (off by default).
Hovering reports the label, A-P value and difference value under the
cursor. **Temporal context** scrubs or plays the 33 context frames (a frame
outside the recording repeats the nearest readable one and is marked), and
**Difference** shows the channel the network sees for a lag of 1 to 16,
grey at zero with an adjustable gain; a lag with an invalid end is a zero
channel.

Corrections:

- **Flip head/tail** reverses the tube, swaps the end points and replaces
  `ap` by `1 - ap`.
- **Accept** and **Reject** record the review and, by default, move to the
  next sample in the filtered list.
- **Trace midline** (T) corrects a fit that takes the wrong route through a
  crossing or contact. Click from the head along the body to the tail,
  through crossings in the order the body goes, about 10 points and more
  around loops; each click adds a numbered point (point 1, the head, is
  green), a drag moves a point, a click on the trace inserts one, a
  right-click removes one, Backspace removes the last and Esc leaves trace
  mode. Dragging elsewhere and the wheel still pan and zoom. **Fit** (Enter) refits along the trace on the
  app's device (a few seconds on a GPU, about 10 s on a CPU) and shows the
  preview's A-P field, cyan tube and head/tail markers over the frame next to
  the record's green tube, with *IoU new vs old*; hovering reads the
  preview's A-P values. **Use trace as drawn** previews the trace itself as
  the midline (the fallback above). **Accept** writes the previewed fit
  (accepted, orientation manual, `auto_fit_iou` keeping the replaced fit);
  **Discard** drops the preview and keeps the points for adjusting. A later
  rebuild refits along the stored trace.
- **Proposals.** `scripts/propose_traces.py` precomputes the network's
  traces (by default for unreviewed samples), and **Propose** runs it on the
  open sample (one at a time, on the app's device). The summary reads
  *Proposal: IoU new vs current*. **Accept proposal** (G) makes the
  proposal's fit the targets without a refit (a traced record with
  `trace_source` `network`, accepted) and, like Accept, moves to the next
  sample. **Edit proposal** loads the proposal's trace into Trace midline as
  editable points: drag a point to move it, click on the trace to insert a
  point there (a click elsewhere extends it at the tail), right-click a point
  to remove it, then Fit and Accept as a hand trace. A proposal made for an
  older mask is not shown and cannot be accepted; propose again.
- **Edit mask in Paint** opens the sample in Paint (its **Return to Body
  fields** button comes back). Saving
  it raises the mask revision, which marks the record stale; return to Body
  fields and **Rebuild targets**, which queues a job (shown in Jobs) that
  refits the tube to the current mask, about 10-40 s on a GPU. Edits to a
  sample wait until its rebuild job finishes.

| Keys | Action |
|---|---|
| `N` `P` | Next / previous sample in the filtered list |
| `H` | Flip head/tail |
| `T` | Trace midline on / off; while tracing `Backspace` removes the last point, `Enter` fits, `Esc` cancels (other sample keys pause) |
| `A` `R` | Accept / reject |
| `G` | Accept the proposal (then the next sample) |
| `D` | Frames / difference view |
| `←` `→` | Step the context offset (frames) or the lag (difference) |
| `Space` | Play the context frames |
| `0`, double-click | Fit view |

The same operations are available headless under `/api/body-fields`:
`GET /api/body-fields` (filters `split`, `orientation`, `review`, `status`,
`contact=yes|no`, `method` (fit method), `proposal=yes|no`, `min_iou`,
`max_iou`; each row has `proposal` (`ready`, `no_trace`, `stale` or null) and
`proposal_fit_iou`), `GET /api/body-fields/{id}` (layers
as PNG data URLs, with a `proposal` object of the same layers when a ready
proposal belongs to the current mask), `GET /api/body-fields/{id}/context`,
and `POST` to `/{id}/flip`, `/{id}/review` (`{"status": ...}`),
`/{id}/rebuild`, `/{id}/propose`, `/{id}/accept-proposal` and
`/{id}/trace` (`{"points": [[x, y], ...], "as_drawn": false, "commit": false}`:
the layers of the preview, or with `commit` of the written record; it runs in
the request, one fit at a time). Every edit returns 400 while the sample's
rebuild job is pending; `/propose` also when no network checkpoint is
configured.

Coverage: `tests/test_body_fields.py` (record edits, staleness, atomic
writes, traces, routes, proposals with a stub network),
`tests/test_body_proposal.py`, `tests/browser/body_fields.cjs` (the screen
end to end on a synthetic store, including a trace preview and accept) and
`tests/browser/body_fields_proposal.cjs` (accepting with G, editing a
proposal's points, Propose without a network).

## Body-field evidence in the pose fitter

The network's predictions also steer the tube fit of a recording. With
`--body-net <checkpoint>` (`scripts/fit_recording.py`) or the fit stage's
`body_net` parameter (`pipeline.FitParams`), every fitted frame is predicted
from its flat-fielded neighbours at the checkpoint's lags
(`body_proposal.RecordingFieldPredictor`, slab by slab), and the prediction
enters the fit three ways:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/fit_recording.py \
  --recording /store1/shared/all_data_raw/prj_aversion/2024-05-28/2024-05-28-02.h5 \
  --start 0 --frames 1200 --body-net checkpoints/body_net/best.ckpt
```

- **Evidence terms** (`batch_fit.fit_masks(fields=...)` with the
  `MaskFitConfig.field_*` weights; the evidence comes from
  `body_proposal.field_evidence`). The *A-P term* samples the predicted A-P
  field at every midline point and compares it with the point's own position
  `s` along the body: the mean of `((A-P - s) / 0.05)^2` over the points
  where the field is known, times `FIELD_AP_WEIGHT` (0.001). The field is
  known on the cleaned mask, except on predicted crossings (overlap above
  0.5). At a contact the A-P field changes across the contact line, so a pose
  routed onto the wrong limb pays for it even where its overlap with the mask
  is the same, and a reversed pose pays everywhere. An A-P weight of 0.01 or
  more pulled fits off the mask edges. The *end term* is the squared distance
  of the fit's first and last points from the predicted head and tail, in
  units of 10 px, times `FIELD_END_WEIGHT` (0.01). An end counts only when
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
  head-first start with the recording prior's width profile, lengthened off
  camera to the prior's length by the rule of the standard starts
  (`mask_fit.extend_start_to_length`: an end within 80 px of mask pixels on
  the image border). The 8 px last-point test of hand traces is not enough
  here: the last band point of a clipped body lies a median 23 px from the
  edge (90th percentile 60 px, edge_0528), and fits from the short traces
  squeezed the whole body into view.
- **Both orientations.** Every standard start is fit both ways
  (`mask_fit.orientation_pair`) and the trace head first, and the result is
  not re-oriented by its taper afterwards: the evidence decides which end is
  the head.

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
The region algorithms (`algorithms.py`) do not use the evidence yet.

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
  lengths jump (89 to 138), because a fit inherits the length of its trace
  start. In the tight inner turns of spiral_0528 a band's nearest patch can
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
  sigma instead of 10 gave 0.026), so the tail term stays for now; without
  the head's term the head drifts at the camera edge (edge_0528 independent
  pose jumps 13 to 41).
- **The end rules and the extended trace.** Before them, the network with
  propagation scored 39 / 6 / 46 / 62 (`sequence_eval_propagate_fields.json`),
  worse than the current fitter on every count. Lengthening the trace off
  camera and ignoring ends at the image edge took edge_0528 from 22 / 2 / 9 /
  31 to 2 / 0 / 4 / 19; ignoring ends off the mask took tail_reentry_0623
  from 14 / 0 / 10 / 4 to 5 / 1 / 5 / 4 (`*_fix2*`, `*_fix3*`).
- **Track-pass guard, not adopted.** Refitting a converged clipped pose with
  the track's length prior pulls its off-camera part into view whatever the
  schedule, prior or evidence (before the end rules, 16 of edge_0528's 22
  frames below 0.9 were such refits). Keeping the stored pose when a refit
  loses more than 0.01 IoU took the six clips that reach the image edge from
  8 / 1 / 16 / 40 to 4 / 1 / 26 / 28, and without the network from
  6 / 2 / 14 / 46 to 6 / 2 / 34 / 46: the refits it rejects are the ones
  correcting the length.

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
- Most of the added init time is the trace start's width measurement
  (`anchors.estimate_width_along_normals` in `init_from_centerline`, about
  70-80 ms per frame); `propose_trace` takes about 8.5 ms and
  `field_evidence` 0.6 ms per frame on the flv-c3 CPU.
- Propagation costs 1620 ms per stretch frame against 1215: the chains fit
  11.2 rows per stretch frame against 6.3, partly the network's trace (1452
  offered, 195 won). With fewer stretch frames the pass costs about the same
  per recording frame.
- The evidence holds a full-image A-P map (2.8 MB at 732×968) for every row
  of a pass for the whole pass, and propagation runs with the network logged
  CUDA caching-allocator retries of about 1.5 GB on the 24 GB A5500s
  without failing.
