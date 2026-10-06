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
the app.

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
