# App simplification

Status: implemented on branch `app-simplification` (waves 1–3, 2026-10-07).
This document collects the decisions of the app simplification pass. It
replaces `UI_TASK_TABS_PLAN.md` (deleted) and the UI sections of
`APP_PLAN.md`. Section 5 ends with what was built differently from the text
above it. Section 1 (storage) comes first because the
model picker, Labeling and Training pages are built on it; the page-by-page
reviews follow as sections 2–4.

## Who uses the app

| Role | Works in | Needs |
|---|---|---|
| **Analyst** | Workspace; occasionally Labeling | Import a recording, run with the right model without choosing anything, find and fix bad stretches, export. Never sees datasets, hyperparameters or diagnostics. |
| **Trainer** | Labeling, Training | A new microscope or condition where the lab models fail: label frames (bootstrapped by the closest lab model), fine-tune or train from scratch, compare against the lab models, make the result the default for their setup. |
| **Developer** | Everything, plus diagnostics | Design and test methods, build the first models, publish models and datasets to the lab library. |

There are no role modes in the UI. Analysts never need to open the Training
page, and the research diagnostics appear only when the app is started with
`--dev`.

## 1. Storage: libraries, setups, datasets, models

### Two libraries with the same layout

- **Lab library**: a shared directory on the lab file system,
  `/storage/fs/store1/shared/worm-pose-models` on the flv-c machines (the NFS path; `/store1` is only a symlink on some hosts). It will be copied
  to the Engaging cluster, where the path will differ. The app looks up the
  path by hostname (the convention used elsewhere in the lab's code), and
  `--lab-library` overrides it. It holds the models and datasets the
  developer publishes. The app reads it and never writes to it.
- **Personal library**: a directory for each user (default under their home
  directory). Everything the app creates goes here: new labels, datasets,
  trained models and setups.

The app lists both, tagging each item **Lab** or **Mine**. Items are
referenced as `lab:<id>` or `mine:<id>`. Publishing (copying a personal model
or dataset into the lab library and freezing it) is a developer command-line
step, `worm-pose publish`. It is not an app control.

Workspaces are not library items: they stay personal and record which model
and settings produced them.

```
<library>/
  setups/<setup>.json
  labels/<setup>/                       # the setup's label collection
    index.json
    <recording>/<frame>/<revision>.npz
  datasets/<dataset>/
    dataset.json
    splits.json                         # recording -> split; absent = not included
  benchmarks/<benchmark>.json           # frozen test-label list
  models/<model>/
    model.json                          # the model card
    weights.ckpt
    training/                           # hparams, metrics.csv, exact label revisions used
    evaluations/<benchmark>.json
```

### Setup

A setup is one microscope. Recording conditions (food, plates, stimuli) are
not setups: when lab models do poorly under a condition, the user fine-tunes
a model on their own labels from that condition, within the same setup. A setup holds the facts the pipeline needs to
read and normalize its videos, which today are hard-coded for one rig:

- video source: file type and dataset path (today `/img_nir` is assumed),
  flat-field on or off
- **pixel size (µm/px)** and **frame rate**. Models are trained at one scale
  and one frame rate. The pipeline rescales frames to the model's scale, and
  converts temporal lags from seconds to frames, using these two numbers.
  Without them, a model that is good on one microscope can quietly fail on
  another.
- `default_model`: the model the Workspace uses for this setup. This replaces
  the global `best.ckpt` and `promotions.jsonl`. The lab library sets one
  default per setup. A user can override it in their personal library, for
  example with their condition-specific fine-tune. Changing a default records
  who changed it and why, as the promotion log does today.

A recording is assigned to a setup at import, and the analyst never chooses a
model unless they want to.

### Label

A label is what a person decided about one frame: the frame, its mask, and
optionally the body annotation (head/tail orientation, a traced midline,
accept/reject). It merges the current segmentation sample and the human
fields of the body-field record. Anything computed from the annotation (the
rendered A-P field, heatmaps, tube fit) is a rebuildable cache, not part of
the label.

A label stores its temporal context (today's ±16 frames) inside the file, so
it does not depend on the raw recording. Raw recordings are on stores that
corrupt (32 of 79 on store1), and other users may not have access to them.

Each save is a new revision, as in the current corpus, so a training run can
record exactly which revisions it used.

### Dataset and splits

Every label of a setup goes into the setup's **collection**: the lab
library's part (published by the developer) and the user's personal part,
where the app saves. A frame's label is its newest revision in either part,
so a user's edit of a lab label is a personal revision and nothing is
copied. The collection holds every recording that has a label.

A **dataset** holds no labels. It chooses, for each recording of its
setup's collection, the split its labels go to: train, val, test or **not
included**. Every recording starts not included, in a new dataset and when
it is labeled for the first time, so nothing joins a training split unless
someone puts it there (decided October 9, 2026; this replaces datasets that
held labels, extended a lab dataset and pledged a split by a recording's
first label). Training reads a dataset's labels as they are when it starts
and records the exact revisions and their splits.

**Splits are assigned per recording, not per frame.** The question a user
asks is "will this model work on my next recording", and frames from the same
16-minute recording are not independent. Today the 3 largest of the 12
labeled recordings have frames in train, val *and* test, while the newer
corpus recordings already fall into a single split. With per-recording
splits, a recording's labels all go to the split its dataset chooses, and
labels added to the recording later follow it. A split can be changed;
models already trained keep the record of what they used.

### Benchmark

A benchmark is a frozen list of test-label revisions for one setup, for
example `nir-v1`. Every model is scored on every benchmark for its setup, so
results in the model picker are comparable. When the test set has grown
enough to matter, you freeze a new version (`nir-v2`) and rescore all models.
Benchmarks are never updated in place. A trainer's own test recordings can be
frozen into a personal benchmark in the same way. For a condition fine-tune,
the picker shows both numbers: the personal benchmark shows whether it helps
on their condition, and the lab benchmark shows what it costs elsewhere.

Three metrics, shown in the picker:

- **Mask IoU**: mean, and the worst 5% of frames (the worst frames are what an
  analyst has to fix by hand)
- **Head/tail correct**: percentage of frames (body-field models only)
- **A-P error**: mean normalized arc-length error inside the mask (body-field
  models only)

### Model card (`model.json`)

| Field | Example |
|---|---|
| name | `nir-hand284` (generated from dataset and label count; replaces `segmenter_model_names.json`) |
| setup | `lab:nir-flv` |
| inputs | flat-fielded frame; temporal lags `[1, 4, 16]` frames at 20 fps; 1.0 µm/px |
| outputs | `mask`, `ap`, `head`, `tail`, `overlap` |
| trained on | `lab:nir-labels@3` + `mine:copper-plates` — 196 train / 49 val labels from 9 recordings |
| started from | `lab:nir-hand284` or ImageNet |
| evaluations | per benchmark: IoU mean / worst 5%, head-tail %, A-P error |
| author, created, notes | free text from the training form |

The segmenter and the body-field net become one concept: a model with a set
of outputs. Pipeline stages declare which outputs they need. The picker shows
any stage a model cannot drive (for example, a mask-only model shows that it
gives no head/tail evidence).

### What happens to the current stores

| Today | Becomes |
|---|---|
| `segmentation_v1` + app corpus | lab dataset `nir-labels` (setup `nir-flv`), splits re-assigned per recording |
| `segmentation_legacy_v1` (Katie's labels) | not migrated: annotated by a different method, and many source videos are corrupt. The multi-store training path that existed for it is removed. |
| `<store>/body_fields/*.npz` | human fields move into the label; rendered targets become a cache |
| `checkpoints/segmenter/runs/*`, `checkpoints/body_net/runs/*` | model cards for the models worth keeping; the rest are deleted |
| `best.ckpt`, `promotions.jsonl` | `default_model` on the setup |
| `segmenter_model_names.json` | the `name` field of each card |

### Decisions (2026-10-07)

1. Lab library: `/store1/shared/worm-pose-models` on flv-c, resolved by
   hostname, with a different path to come on Engaging.
2. A setup is a microscope. Conditions are handled by personal fine-tunes.
3. Splits are per recording. The current split is redone, and r2–r4 are
   rescored on the first benchmark.
4. The lab library starts with the best body-field net, rescored on the
   first benchmark. r4-hand284 is used only until the segmenter is removed
   (section 4, decision 1).
5. Katie's legacy data is not included.
6. Publishing is a developer CLI step (`worm-pose publish`).

## 2. Workspace page

Reviewed with Alex 2026-10-07; decisions at the end of the section.

Today the Workspace has 5 task tabs (Run, Inspect, Masks, Compare, Export),
a right panel with 4 tabs and 11 sections, 35 overlay layers, 18 extra
timeline series and about 110 controls. The proposal keeps three regions:

```
┌ Recordings │ Workspace · Labeling · Training ───── 2023-06-23-01 · nir-hand284 · ▓▓▓░ fitting 62% │ Export ┐
├────────────┬──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Issues     │                                                                                              │
│ 12/40 done │                               frame                                                          │
│ ▸ 1204–1260│                         (4 layer toggles)                                                    │
│   head/tail│                                                                                              │
│   uncertain│                                                                                              │
│ [Looks ok] │                                                                                              │
│ Fix:       │                                                                                              │
│  Flip      │                                                                                              │
│  Refit     │                                                                                              │
│  Edit mask │                                                                                              │
│  Relabel   │                                                                                              │
│ Fixes (3) ↶│                                                                                              │
├────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┤
│ ◀ ▶ Play  frame [1231]  speed [1×]      issue track ▬▬ ▬   ▬▬▬                                            │
│                                         curvature kymograph (head → tail × time)                          │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Left:** the review panel, which is the issue list plus fixes.
- **Center:** the frame.
- **Bottom:** transport, the issue track, and the curvature kymograph.

The kymograph is the pipeline's scientific output, and bad stretches show up
in it as breaks and sign flips. With the issue track above it, it replaces
the per-series charts.

Everything marked *dev* below is shown only when the app is started with
`--dev`. Everything marked *remove* is deleted from the code.

### Header and context strip

| Today | Proposal |
|---|---|
| Import, Open buttons | **Merge** into a single **Recordings** home screen (below) |
| Screen tabs: Workspace, Paint, Labels, Body fields, Training | **Redesign:** Workspace · Labeling · Training |
| Side panel toggle | **Remove** (the side panel becomes dev-only) |
| Workspace menu, fit-network badge, active recording | **Merge** into one title: recording · model name |
| Context strip: frame, selection first/last, Use current frame, Use stretch | **Remove.** An issue defines the range a fix acts on. For a range with no issue, right-drag on the timeline (a left drag scrubs). |
| — | **New:** analysis progress (stage + %, failure with Retry) |
| — | **New:** Export button |

### Recordings screen (replaces Import and Open)

Two tables of recordings for the chosen setup: the ones analysed or
analysing (including a failed analysis), most recently opened first, then
the ones not analysed yet; each recording is in one of them. Columns:
recording, length, status (not analysed / analysing / *N* issues to review /
reviewed / exported), last opened. One action per row: **Analyse**, or
**Open** if a workspace already exists. While an analysis runs the tables
poll; a poll rebuilds only the rows that changed, so the scroll position and
the thumbnails stay.

Analyse uses the setup's default model on the whole recording and opens the
new workspace, whose left panel shows one progress bar per stage (from the
analysis job's `stages` in the workspace status). The model is shown with a
**Change** link that opens the model picker.

| Today | Proposal |
|---|---|
| Recording filter, Refresh | Keep the filter; **remove** Refresh (rescan on open) |
| Recording roots note, server data note | **Remove** (an error shows only when a root is unreadable) |
| Add recording… file explorer (path, Go, Up, breadcrumbs, shortcuts, show all files) | **Keep, simplified:** a path box with browse, filtered to video files; the setup is chosen when adding |
| Table columns: frames, size, ok·P, runs, ws | **Redesign** to the columns above |
| Preview thumbnail + frame number | **Keep** the thumbnail, **remove** the frame picker |
| New workspace: name, first, last, step | **Remove.** One workspace per recording, always the whole recording. *dev:* range and step. |
| Existing runs / workspaces (read-only runs, "Create editable copy") | **Remove.** The CLI pipeline writes workspaces too, so read-only runs disappear as a concept. |
| Open screen (filter, catalog list, Resume, Create editable copy) | **Merge** into the table (Open) |

### Run task (Stages, Run all, scope, Job GPU, Regions)

| Today | Proposal |
|---|---|
| Stage checklist + Run all | **Remove.** Analysis runs all default stages on Analyse. Re-analyse (for example with another model) is in the model picker. *dev:* the stage checklist. |
| Operation scope (current / selection / whole) | **Remove.** The scope comes from the issue or the timeline selection. |
| Job GPU | **Remove** (automatic). *dev:* keep. |
| Regions: Use current stretch / Around frame / Anchors / Clear, first/last/anchor before/after, algorithm picker (8 algorithms), parameter form, Run on region | **Redesign** as the **Refit** fix: anchors are automatic, and the algorithm is chosen from the issue type. *dev:* algorithm picker and parameters. |

### Inspect task → Issues panel

| Today | Proposal |
|---|---|
| Segments needing attention, with reasons | **Keep, as the issue list.** Reasons become plain words ("head/tail uncertain", "coiled", "mask fits poorly", "leaves the view"). |
| Minimum segment length + Show shorter segments | **Remove.** One fixed rule (today's 8 frames), with short issues merged into neighbours. |
| Mark reviewed & next, Skip to next unreviewed | **Keep** as **Looks OK** (marks the issue reviewed and moves to the next) and Next/Previous |
| Correct masks / Correct orientation / Try another fit / Label range in Paint | **Keep** as the four fixes below (Label range becomes Relabel) |
| Jump to: flagged ◀▶ + score, low IoU ◀▶ + threshold, stretch ◀▶, jump ▶, edge ▶, provenance ◀▶ + select, Worst frames | **Remove** (the issue list replaces them) |
| Review notes (tags, comment, list, path) | **Remove.** Review state is *Looks OK* or fixed. |

### Fixes (replaces Masks, Compare, flip buttons, History)

Each fix acts on the selected issue's frames:

1. **Flip head/tail**: the whole issue, or this frame only.
2. **Refit**: runs the recommended algorithm with automatic anchors, then
   shows before/after with **Keep** / **Discard**. This replaces candidate
   sets A/B and their comparison.
3. **Edit mask**: an inline brush on this frame (Worm / Background, size,
   Undo, Network proposal). Save refits the affected stretch automatically.
4. **Relabel**: select a stretch (an issue, or a range dragged on the
   timeline). The app proposes a sparse set of keyframes: the stretch's ends
   plus one about every *N* frames, where *N* is a single spacing control.
   Click the timeline to add or remove keyframes. The keyframes open as a
   queue in Labeling, where each gets a full label (mask, head/tail, midline).
   When the queue is done, the workspace **stitches** the stretch: every gap
   between consecutive keyframes is refit with the two keyframes as fixed
   anchors (the existing region algorithms already take an anchor before and
   after), and the result is shown before/after like Refit. The keyframe
   labels also join the user's dataset for that setup, so a relabel improves
   both the workspace and the next model. This replaces "Send to Labeling".

Below the fixes is a **Fixes** list: what each fix changed, with Undo on each.

| Today | Proposal |
|---|---|
| Masks: Save, Save & refit range, Refit saved frame/range | **Merge** into one Save (always refits the affected stretch) |
| Masks: Edit mask toggle, brush ×3 + diameter | **Keep** Worm / Background + size; **remove** Ignore (Labeling only) and the toggle |
| Masks: Proposals ×4, threshold, saved source, combine, preview, Apply | **Keep** only "Network proposal"; the rest move to Labeling |
| Masks: Refine (fill holes, largest, grow, shrink, tube fit) | **Remove** here (Labeling only) |
| Masks: Save destinations (also save training label, corpus split, save to corpus, remove override) | **Remove.** Training labels come from **Relabel**; the split is per recording; Undo removes an override. |
| Masks: Draft (undo, clear, discard) | **Keep** Undo; Discard happens on leaving the frame (the existing dialog) |
| Compare: candidate sets, Refresh, A/B/Current overlays, Accept, Try another method, Frame hypotheses & orientation | **Merge** into the before/after of Refit and Relabel; the candidate-set UI is **removed** |
| Flip frame / Flip segment (in Statistics) | **Move** into Fixes |
| History: Edits list + Undo + Refresh | **Merge** into the Fixes list |
| History: Outcomes table (IoU, <0.9, jumps, flips before→after) | **Remove** |

### Canvas toolbar and layers

| Today | Proposal |
|---|---|
| Raw / flat | *dev* (analysts see the flat-fielded frame) |
| Fit view | **Keep**: the Fit button or 0 (double-click does not fit) |
| Starts | **Remove** |
| Original prediction / Editable mask / Opacity | **Show only while editing a mask** |
| 35 layers, each with an opacity slider | **Redesign** as 4 toggles: Mask, Midline + head/tail, Body outline, A-P field (only if the model outputs one). With a body model, a **Model outputs** menu adds each output channel raw (implemented Oct 2026). *dev:* the full list. |
| Threshold override + Use override | **Remove** |
| Compare run (same recording) | *dev* |

### Right panel

| Today | Proposal |
|---|---|
| Statistics: classification badge, tags, frame stats table | **Redesign** as one line under the frame: frame status (OK / issue reason / fixed by hand) |
| Hypotheses (6c), Candidate sets on this frame | **Remove** |
| Ambiguity flags, Width along the body, Curvature along the body, Mask pipeline, Summary | *dev* drawer ("Frame details") |
| Jobs tab (state filter, this workspace, Refresh, GPU for retries) | **Remove** from Workspace. Progress and failures go in the header. *dev:* the jobs drawer. |
| History tab | **Merge** into Fixes (above) |
| Layers tab | **Move** to the canvas toolbar (4 toggles) |

### Timeline

| Today | Proposal |
|---|---|
| ◀ ▶ Play, frame number + Go | **Keep**; Enter replaces Go |
| FPS number | **Redesign** as a speed choice (0.5×, 1×, 2×, 4×) |
| Loop selection | **Keep**: it loops the selected issue or range |
| Flag / classification / provenance strips | **Merge** into one issue track (issue type, reviewed, fixed) |
| Extra series (18), independent fit, compare run checkboxes | *dev* |
| — | **New:** curvature kymograph (the first main view; it may be changed or extended later) |

### Export task

| Today | Proposal |
|---|---|
| Review checklist, job check, Open Jobs / Open Masks / Inspect segments, Refresh status | **Redesign** as one dialog from the header button: "*N* issues not reviewed — export anyway?" |
| Export / snapshot name | **Remove** (the name is automatic: recording + date) |
| Parquet snapshot with configuration, masks, provenance, review records | **Redesign** as one documented per-recording table, plus a small metadata file (model, setup, app version). Masks and provenance stay in the workspace. |

Export columns: frame, time, midline points, curvature, full-body width
profile, head/tail, velocity, and per-frame status (auto / reviewed / fixed /
unresolved). Further derived features will be added later. Each is a
function of the midline, the width and time, so the exporter keeps a list of
feature functions, and adding a feature means adding one function.

Velocity in µm/s needs the pixel size (from the setup) and the stage position
of each frame, if the stage tracks the worm. The flv-c recordings
(ConfocalTrackerControl.jl) carry both: `/pos_stage` holds one stage reading
per saved frame in units of 0.1 µm (NaN on a failed serial read, 3–20% of
frames, in gaps of mostly 1–4 frames), and `/img_metadata/img_timestamp`
holds the camera clock in nanoseconds for every camera frame (40 Hz; the
saved 20 fps frames are those with `q_iter_save` and `q_recording` set).
The real rate is about 19.5 fps because of dropped frames, so time comes from
the timestamps. The world position is the image position minus the stage
position, with stage x/y along image x/y (checked on fitted centroids), and
the stage is read about half a frame after the exposure. Recordings without
these datasets export velocity in the image frame, and `export.json` says so.

### Decisions (2026-10-07)

1. The four fixes are Flip, Refit, Edit mask and Relabel. Relabel is new: a
   sparse keyframe relabel with stitching (above). It reuses the Body fields
   controls in Labeling.
2. Read-only runs are removed; the CLI pipeline writes workspaces.
3. One workspace per recording, always the whole recording.
4. The curvature kymograph is the main timeline view for now. Its color
   scale clips curvature × body length at ±15 (median about 4 and 90th
   percentile about 10 on a real recording), confirmed by Alex 2026-10-07.
   Issues closer than 8 sampled frames stay merged into one, also confirmed.
5. Export adds velocity and the full-body width profile; more derived
   features will come later.
6. Hypotheses (6c), Outcomes, the candidate-set UI and Starts are deleted.

## 3. Labeling page

Reviewed with Alex 2026-10-07; decisions at the end of the section.

This page merges Paint, Labels and Body fields. Today those three screens
have about 75 controls. They edit the same thing, one frame's label, in two
halves, and moving between the halves goes through "Edit mask in Paint" and
"Rebuild targets" jobs. On one page, a label is labeled once: mask first,
then body, then **Save & next**.

```
┌ Labeling ──────────────── Saving to the setup's labels ─────────────────────────────────────────┐
├──────────────┬─────────────────────────────────────────────────────┬────────────────────────────┤
│ Queue        │                                                     │ 1 Mask                     │
│ relabel      │                                                     │  Worm Bkgd Ignore  size    │
│ 1204–1260    │                    frame                            │  Network  Threshold ─○─    │
│ ▓▓▓▓░░ 4/7   │                                                     │  Fill holes  Largest       │
│ ● f1204 done │                                                     │  Grow  Shrink  Undo Revert │
│ ● f1214 done │                                                     │ 2 Body                     │
│ ○ f1224      │                                                     │  Use proposal  Flip        │
│ ○ ...        │                                                     │  Trace midline             │
│ ◀ Prev Next ▶│  context: ◀──○──▶ offset  Play  Frames/Difference   │  ☐ mask only               │
│              │  layers: Mask  A-P  Midline  Proposal               │ [Save & next ⏎]            │
└──────────────┴─────────────────────────────────────────────────────┴────────────────────────────┘
```

Saves go to the setup's collection, in the user's personal library. The lab
library is read-only, so a user's edit of a lab label saves a new personal
revision, which is then the frame's label. The Labeling page shows no split:
splits are each dataset's choice, made in the Training page's Datasets tab.

### Queues (replaces Paint's chooser and the Labels screen)

A queue is an ordered list of frames to label. There are three ways to get
one:

- **From a workspace**: Relabel keyframes, and returning to the workspace
  stitches the stretch when the queue is done.
- **New queue**: pick recordings and a number of frames, then **Find
  frames**. A background job runs the current model over the recordings and
  picks frames spread over each recording, favouring frames the model is
  least sure of. **Image types** limit the search to frames of any of the
  checked kinds: self-contact, at the edge, in pieces, no worm, clear (from
  the model's prediction; needs the setup's mask model). A progress bar
  shows the search, and the queue opens when it finishes. This is the
  bootstrap path for a new microscope. A queue's frame list can be filtered
  by recording, status (to do / saved) and image type, and sorted by queue
  order, least sure first, or recording and frame; Prev/Next and Save &
  next walk the list as shown.
- **Browse labels**: existing labels, filtered by recording, split and status
  (mask only / complete). They can be sorted by **body fit IoU, lowest
  first**: the IoU between the label's mask and the body fitted to it, so
  labels whose body targets disagree with their mask come up first.

| Today | Proposal |
|---|---|
| Open groups list | **Keep** as the queue list |
| Labeling manifests + Other manifest path + Open manifest | **Remove** from the app. *dev:* manifests become queues from the CLI. |
| Relabel a recording section (recording, name, first, last, step, Open section) | **Remove.** Workspace Relabel and New queue cover it. |
| Labels screen: filter text, Refresh, Source / Split / Recording filters, Browse filtered labels, counts, list | **Merge** into Browse labels; **remove** the Source filter (bootstrap labels are retired) and Refresh |
| Body fields browser filters: Status (current/stale/missing), Split, Head from, Fit method, Proposal, Review, Self-contact, Fit IoU min/max | **Merge** into Browse labels as status (mask only / complete). Self-contact stays as one checkbox, since it finds hard frames. Fit IoU min/max becomes the lowest-first sort, with each label's IoU shown in the list. The rest **remove**: stale and missing disappear because targets rebuild automatically. |
| Position, progress bar, Previous, Next, Next unlabeled | **Keep** Prev / Next and the progress; Next goes to the next unlabeled frame in an unfinished queue |

### Mask tools

| Today | Proposal |
|---|---|
| Brush: Worm, Background, Ignore + diameter | **Keep** Worm, Background and diameter; **remove** Ignore (migrated labels keep the ignore pixels they already have, and training still excludes them from the loss) |
| Proposals: Network, Classical, Threshold, Saved | **Keep** Network and Threshold (the fallback on a new microscope where no network works yet); **remove** Classical; Saved becomes **Revert** |
| Network threshold slider | **Keep** (one slider shared by Network and Threshold) |
| Combine (replace / union / intersect / subtract) | **Remove** (always replace; paint to add or remove) |
| Preview proposal checkbox + Apply | **Merge:** a proposal button shows its preview, and Apply (A) takes it |
| Refine: Fill holes, Largest, Grow, Shrink, Tube fit | **Keep** the first four; **remove** Tube fit |
| Draft: Undo, Clear draft, Discard draft | **Keep** Undo; **merge** Clear and Discard into **Revert** |
| Toolbar: Raw / flat, Fit view, Mask toggle, Opacity | **Keep** (Raw / flat matters when labeling); Mask toggle and opacity become one opacity slider with a key toggle |
| Save label, Save + next | **Merge** into Save & next (Enter); S saves in place |

### Body tools (from Body fields)

When the model outputs body fields, its proposal is computed as the frame
opens and drawn as a layer. Most frames need only **Use proposal** or
**Flip**.

| Today | Proposal |
|---|---|
| Sample summary banner | **Redesign** as one line: recording, frame, split, status |
| Flip head/tail (H) | **Keep** |
| Trace midline (T) + its panel: Fit, Remove last, Cancel, then Accept / Use trace as drawn / Discard | **Keep** Trace with Fit, Remove last, Cancel, and after the fit Accept / Discard; **remove** Use trace as drawn |
| Accept proposal (G), Edit proposal, Propose | **Keep** Use proposal (G); Edit proposal becomes Trace pre-filled with the proposal's points; **remove** Propose (automatic) |
| Review: Accept (A) / Reject (R) / Unreviewed | **Redesign:** saving means accepted; Reject becomes the **mask only** checkbox (the body is unclear and the label trains the mask only); **remove** Unreviewed |
| Next sample after accept or reject | **Remove** (Save & next always advances) |
| Edit mask in Paint, Rebuild targets, Jobs | **Remove.** The mask is on the same page, and targets rebuild in the background on save. |
| Temporal context: Frames / Difference (D), offset slider, Play (Space), fps, lag slider, difference gain | **Keep** Frames / Difference, offset, Play, lag; **remove** fps (fixed 10) and gain (automatic contrast) |
| Layers: hand mask, A-P field, overlap, tube outline, centerline, head/tail, acquisition nose, stored trace, proposal, proposal A-P | **Redesign** as 4 toggles: Mask, A-P field (the proposal's until saved), Midline + head/tail (with the acquisition nose when there is one), Proposal. *dev:* overlap, tube outline, stored trace. |

### Decisions (2026-10-07)

1. New queue picks frames automatically in a background job with a progress
   bar. Picking frames by hand can come later if needed.
2. Relabel keyframes join the dataset by default. Each label records its
   origin (spread sample or fix), and benchmarks are built only from
   spread-sampled test labels.
3. The fit IoU filters are kept, as a lowest-first sort in Browse labels.
   Tube fit, Use trace as drawn, the fit-method filter and difference gain
   are removed.
4. No Ignore brush.

## 4. Training page and the model picker

Reviewed with Alex 2026-10-07; decisions at the end of the section.

Today the Training screen has three numbered steps: label counts, a
fine-tune form (segmenter only; 4 settings plus 4 advanced), and a
checkpoint dropdown with "Use in workspace". The body-field net is trained
only from `scripts/train_body_net.py`. Evaluations come from
`scripts/evaluate_segmenter.py` and `scripts/evaluate_body_net.py` into
separate places. "Promotion" is a CLI flag that overwrites `best.ckpt`.

The proposal has three tabs, all for the current setup: **Models**,
**Datasets** and **Train**.

### The model picker (Models tab, and "Change model" in the Workspace)

One table, one row per model. Lab and personal models are mixed together, and
★ marks the setup's default:

| Model | From | Inputs | Outputs | Trained on | IoU mean | IoU worst 5% | Head/tail | A-P err |
|---|---|---|---|---|---|---|---|---|
| ★ nir-hand284 | Lab | frame | mask | 196 labels · 9 rec | 0.95 | 0.88 | — | — |
| nir-body-lags3 | Lab | frame + motion ±1/4/16 fr (0.05/0.2/0.8 s) | mask · A-P · head · tail · overlap | 196 · 9 rec | 0.94 | 0.86 | 100% | 0.06 |
| copper-ft | Mine | same as parent | same as parent | +40 labels · 3 rec | … | … | … | … |

(Numbers are illustrative.)

- **Benchmark selector:** the metric columns are for one benchmark,
  chosen above the table: the setup's lab benchmark or a personal one.
  **Not evaluated** shows until that evaluation has finished; it runs on its
  own in the background.
- **Missing outputs:** a model missing an output that a pipeline stage uses
  shows what it can't do (for example "no head/tail: orientation from body
  taper only").
- **New microscope:** when the setup is new, the picker also lists lab models
  from other microscopes with their numbers there. This is how a new
  microscope starts: pick the closest one as the default.
- **Row actions:** **Use as default** (asks for a one-line reason; this
  replaces promotion), **Train from this**, and **Details**.
- **Details** shows the training curves (train and validation loss), the
  worst benchmark frames as overlays, the settings, the exact labels used,
  and notes.

### Datasets tab

A dataset picker (Lab and Mine) and **New dataset…** (a name, and either
nothing included or another dataset's splits to start from). The chosen
dataset shows every recording of the setup's collection with its label
count and status, and a split for each: Train, Val, Test or Not included
(read-only for a lab dataset). Totals by split, in labels and recordings,
including Not included, update with every choice. Clicking a recording
opens Labeling's Browse labels filtered to it, with the dataset's splits.

- **Readiness:** the tab warns when a dataset can't train or be evaluated
  honestly: no validation recording, no test recording, or too few labels.
- **Freeze benchmark:** freezes the dataset's spread-sampled test labels as a
  personal benchmark (`mine:<dataset>-b1`). Every model for the setup is then
  evaluated on it in the background. Evaluations of lab models on a personal
  benchmark are stored in the personal library, because the lab library is
  read-only.

### Train tab

| Control | Proposal |
|---|---|
| Start from | A model (default: the setup's default) or **From scratch** |
| Temporal context | From scratch only: none / short (±1, 4, 16 frames). Lags are stored in seconds on the card, so another frame rate converts them. |
| Training data | One dataset of the setup (default: your first): its train and val recordings' labels |
| Max epochs, learning rate, batch size | Shown; early stopping normally ends the run first |
| Advanced (collapsed): crop size, patience, plateau patience, encoder LR scale, loss weights (A-P, heatmap, overlap), minimum body fit IoU, seed | Kept for the developer and experienced trainers |
| Name, notes | The name is generated (dataset + label count) and editable; notes go on the card |
| Start | One job: prepare body targets if needed → train → evaluate on every benchmark for the setup → write the card |

The running job appears at the top of the Models table with epoch, a live
loss curve and Cancel. A failed run shows its error there until it is
dismissed.

| Today | Proposal |
|---|---|
| 1. Training data: counts, readiness, Add training labels | **Move** to the Datasets tab; Add training labels becomes Labeling's New queue |
| 2. Fine-tune: epochs, batch, LR, device; advanced crop, patience, workers, seed | **Keep** epochs, LR, batch, and crop / patience / seed under Advanced; **remove** device and workers (automatic); **add** the body-net settings |
| 3. Use a model: checkpoint select, Refresh, Use in workspace | **Replace** with the picker's Use as default, and the Workspace's Change model |
| View training jobs | **Remove** (the running job is shown in the Models table) |
| `--train-labels` (all / bootstrap / manual) | **Remove** (bootstrap labels are retired) |
| `--promote` / `--no-promote`, `best.ckpt`, `promotions.jsonl` | **Remove** (Use as default, with a reason) |
| `evaluations/`, `evaluations_legacy/`, `history.jsonl` | **Replace** with `models/<model>/evaluations/<benchmark>.json` |
| `last.ckpt` | **Remove**: a model is the one checkpoint with the lowest validation loss |
| Device select | **Replace** with **Run on** (below) |

### Where jobs run

At startup the app checks two things: which GPUs the node it runs on has
(an interactive session on Engaging, or an flv-c machine), and whether SLURM
is available (`sbatch` and `squeue` on the path). The Train form and Analyse
show one **Run on** choice built from what it found:

- **This machine**: the GPUs found (automatic choice; *dev:* a specific GPU)
- **SLURM**: partition and time limit, prefilled from a per-host default
  table, the same hostname lookup as the lab library path

If neither exists, training and analysis are disabled, with the reason
shown.

Behind this, one job record has two executors:

- **Local:** today's `jobs.py` queue.
- **SLURM:** writes a batch script that runs the same job command, submits
  it, and keeps the SLURM job id in the job record. State comes from
  `squeue`/`sacct`, and logs from the job's output file, so a job survives an
  app restart. A SLURM job runs on a compute node, so it can only use the
  libraries and workspace if they are on storage that node can read. The
  app checks that before submitting.

### Decisions (2026-10-07)

1. **The mask-only segmenter is removed.** There is one architecture, the
   body-field net, and the Outputs choice is gone from the Train form.
   **Gate:** on the old per-frame test split (39 labels, the same metric
   function), the segmenter r4-hand284 reached mask IoU 0.983, and the
   body-field net runs reached 0.85–0.87. Rescored on the first benchmark
   `nir-v1` (13 labels from 3 held-out recordings) by the new evaluator, the
   gap is much smaller: segmenter 0.988 mean / 0.979 worst label, body-field
   net 0.980 / 0.970 (head/tail 13 of 13, A-P error 0.094). `nir-v1` is too
   small to call the gate met. The segmenter, its trainer and its evaluator are
   deleted once a body-field net matches its mask IoU on the first benchmark.
   Until then, the pipeline's mask still comes from r4. Body-target building
   also uses the segmenter today (it segments context frames for chain fits),
   so that switches to the setup's default model.
2. A model is the one checkpoint with the lowest validation loss.
3. The split between shown and Advanced hyperparameters is accepted.
4. Jobs run on the local node's GPUs or through SLURM, chosen per job from
   what the app finds at startup (above).

## 5. Implementation

Branch `app-simplification`, stacked on `temporal-segmenter`. Agreed with
Alex 2026-10-07: keep the mask-only segmenter until the body-field net
matches its mask IoU (section 4, decision 1). Everything else in sections 1–4
is built.

The work runs in three waves. Within a wave, parallel agents own disjoint
files. Each wave is merged and tested before the next starts. After each
wave the app still starts and its tests pass.

### Wave 1: backend foundations (parallel)

**A. Library** (`src/worm_pose_gen/library/`, `app/routers/library.py`,
`scripts/migrate_to_library.py`, `scripts/publish.py`, tests)

- `Libraries(lab: Path | None, personal: Path)`. The lab root is looked up
  by hostname (`LAB_LIBRARY_BY_HOST`: flv-c2/c3/c4 →
  `/store1/shared/worm-pose-models`) and is read-only to the app. The
  personal root defaults by host as well (flv-c →
  `/temp_data4/<user>/worm-pose-library`, elsewhere
  `~/worm-pose-library`). `--lab-library` and `--library` override them.
- References are strings: `lab:<id>` and `mine:<id>`.
- **Setup** (`setups/<id>.json`) holds: `name`, `description`, `video`
  (`dataset_path`, default `/img_nir`; `flat_field`), `pixel_size_um`, `fps`,
  `recording_roots`, and `defaults` keyed by role (`{"mask": ref, "body":
  ref}`), with an append-only `defaults_log.jsonl` (who, when, reason). A
  personal `setups/<id>.override.json` can override the defaults of a lab
  setup. A recording belongs to the setup whose root contains it, or is
  registered in the personal `recordings.json`.
- **Collection** (`labels/<setup ref as lab.id>/` in each library): every
  label of the setup. A frame's label is its newest revision in either
  library (by `saved_at`, the lab's on a tie).
- **Dataset** (`datasets/<id>/dataset.json`: `setup`, `name`,
  `description`) has `splits.json`, which maps recording id → split for the
  recordings it includes; any other recording of the collection is not
  included. Personal datasets' splits can be changed at any time.
- **Label**: immutable revisions under `labels/<setup>/<recording>/<frame>/`,
  with an index for fast listing. A revision stores:
  - `image` (flat-fielded uint8), `image_raw`, and `mask` (0/1; 255 = ignore,
    only in migrated labels)
  - the context frames t−16..t+16 with their validity
  - the human body fields: `orientation` (`auto` or `manual`) with the
    flip, an optional `trace_xy`, and `mask_only`
  - provenance: `origin` (`spread`, `fix` or `migrated`), `author`,
    `saved_at`
  
  Derived body targets (tube fit, A-P, heatmaps, overlap, `fit_iou`,
  self-contact) are a cache in the personal library, keyed by label
  revision. A personal revision of a frame is newer than the lab's, so it
  is the frame's label.
- **Benchmark** (`benchmarks/<id>.json`) is a frozen list of (library,
  setup, recording, frame, revision, sha256) entries for one setup, taken
  from one dataset's test recordings. It takes only `spread` and `migrated`
  test labels, and is never updated in place.
- **Model card** (`models/<id>/model.json`) holds:
  - `name` and `kind`: `segmenter` or `body_net`, which says how to load it
  - `setup`
  - `inputs`: preprocessing, `lags_frames`, `lags_s`, `fps`,
    `pixel_size_um`
  - `outputs`: a subset of `mask`, `ap`, `head`, `tail`, `overlap`
  - `trained_on`: dataset refs with label fingerprints and counts
  - `parent`, `hparams`, `author`, `created_at`, `notes`
  - weights in `weights.ckpt` (the single checkpoint with the lowest
    validation loss)
  
  Evaluations go in `models/<id>/evaluations/<benchmark>.json`. Evaluations
  of a lab model on a personal benchmark go in the personal library.
- **Read API** for the pages (`/api/library/...`): setups, recordings by
  setup, datasets, labels (filter by recording, split and status; sort by
  fit IoU), models with their evaluations, and benchmarks. **Writes:** save
  a label revision, freeze a benchmark, set a default.
- **Migration** (`scripts/migrate_to_library.py --out <root>`) takes the
  union of `segmentation_v1` and the app corpus
  (`<workspaces>/corpus`). For the same (recording, frame), the newest
  revision wins. Human body fields come from the `body_fields/*.npz`
  records. Splits are reassigned per recording, with at least one recording
  each in val and test. The script writes setup `nir-flv`, dataset
  `nir-labels`, benchmark `nir-v1`, and model cards for r4-hand284 (`mask`)
  and one body-field net run given on the command line (`body`). The
  legacy store is not migrated. Agents only ever run it into temporary
  directories; the real run into `/store1/shared/worm-pose-models` happens
  after review.
- **`scripts/publish.py`** copies a personal model or dataset into the lab
  library (developer step).

**B. Compute and jobs** (`src/worm_pose_gen/jobs.py`,
`src/worm_pose_gen/compute.py`, `app/routers/jobs.py`, tests)

- `detect_compute()` → the local GPUs (on the node the app runs on), and
  SLURM availability (`sbatch`, `squeue`, `sacct`) with its partitions.
- `SLURM_DEFAULTS_BY_HOST` holds partition, time limit and GPU request. A
  `SlurmBackend` sits next to `LocalGPUBackend` and implements the same
  backend protocol:
  - it writes a batch script that runs the job's command, submits it with
    `sbatch --parsable`, and stores the SLURM job id on the record
  - it polls `squeue`/`sacct`, cancels with `scancel`, and survives a
    restart
  - before submitting, it refuses paths a compute node cannot read (it
    checks that they are not node-local: `/tmp`, `/scratch` or `/dev/shm`)
- `JobSpec` gains `run_on` (`local` or `slurm`) and `slurm` (partition,
  time). `GET /api/compute` reports what `detect_compute()` found.
- Tests use fake `sbatch`/`squeue`/`sacct`/`scancel` scripts on the `PATH`.

**C. Export** (`src/worm_pose_gen/export_table.py`, `app/exporting.py`,
`pipeline.run_export`, tests)

- One table per recording plus `export.json` metadata (model refs, setup,
  pixel size, fps, app git revision).
- Columns: `frame`, `time_s`, midline points, curvature, width profile,
  head/tail xy, centroid, velocity, and `status` (`auto`, `reviewed`,
  `fixed` or `unresolved`).
- Derived features are a list of functions `(midline, width, time, meta) →
  columns`, so adding one is a single function.
- Velocity uses the stage position when the recording has one. Agent C
  finds out whether the HDF5 files carry stage positions; if they do not,
  velocity is in the image frame, and `export.json` says so.
- The export name is automatic.

**D. Fix algorithms** (`src/worm_pose_gen/fixes.py`, `app/inspection.py`,
`app/regions.py`, `algorithms.py` additions, tests)

- **Issues API:** the attention segments with plain-language reasons
  (head/tail uncertain, coiled, mask fits poorly, leaves the view). Short
  issues are merged using the fixed 8-frame rule. Each issue has a reviewed
  or fixed state.
- **Refit:** maps the issue type to an algorithm, chooses anchors
  automatically, and returns a before/after preview to Keep or Discard.
  This replaces the candidate-set accept flow.
- **Keyframe proposal** for Relabel: the stretch ends plus one keyframe
  about every N frames.
- **Stitch:** given keyframe poses (from labels), it refits every gap
  between consecutive keyframes with the two keyframes as fixed anchors,
  and returns the same Keep/Discard preview.
- Fixes go through the existing edit log, so Undo works.

### Wave 2: the three pages (parallel, after wave 1 is merged)

Before wave 2, the integrator restructures `index.html` into a shell: a
header with Workspace · Labeling · Training, three page sections, `--dev`
exposed through `/api/config`, and one JS/CSS file set per page. Each page
agent edits only its own section and files.

- **W. Workspace:** the Recordings screen, one workspace per recording
  (the whole recording), Analyse with the setup's defaults and Run on, the
  Issues panel, the four fixes, the Fixes list with Undo, 4 layer toggles,
  the kymograph and issue track, the Export button, and the dev drawer.
- **L. Labeling:** queues (from Relabel, New queue as a "Find frames" job
  with progress, Browse labels with the fit-IoU sort), mask tools, body
  tools, Save & next into the personal dataset, and the background rebuild
  of body targets.
- **T. Training:** the model picker (also used by the Workspace's Change
  model), the Datasets tab with Freeze benchmark, and the Train tab. Both
  trainers read library datasets. A training job is: prepare targets →
  train → evaluate on the setup's benchmarks → write the card.
  Evaluation jobs fill in missing evaluations.

### Wave 3: removal and integration

- Delete the old screens and code paths:
  - the Paint, Labels and Body fields screens, the Run, Compare and Export
    tasks, the jump buttons and review notes
  - candidate sets, Outcomes, Starts, Hypotheses (6c), read-only runs and
    "Create editable copy"
  - the legacy store and multi-store training, the bootstrap label filter,
    `best.ckpt` promotion, `last.ckpt`, the old evaluation folders, and
    `segmenter_model_names.json`
- Update the README and the browser tests, then run the full suite.

Done. Wave 3 deleted the `pose_viewer_ui` package and the stdlib viewer
(`pose_viewer.py`; its frame layers, statistics and series moved to
`app/frame_view.py`), the standalone labeler (`label_app.py` and
`label_app_ui`; `RecordingSource` moved to `recordings.py`, the PNG helpers
to `app/images.py`, the mask refinements to `app/labeling.py`), the corpus
(`corpus.py` and `/api/corpus`), the old browser tests, and the endpoints
`/api/state`, `/api/run`, `/api/frame`, `/api/pose`, `/api/starts`,
`/api/notes`, `/api/note*`, `/api/outcomes`, `/api/stages`,
`/api/recordings` (list, datasets, register, unregister, preparation,
prepare), `POST /api/workspaces` and `/import`, `GET /api/workspaces`,
`/api/workspaces/{name}/pose`, `/starts`, `/snapshot`, `/edits`,
`/segment`, `/inspection`, `/region`, `/candidates`, and
`DELETE .../mask`. Candidate sets lost their storage, accept flow and
outcome log (`run_algorithm` and the fixes' previews remain), the edits
lost `pick_hypothesis`, `accept_path` and the segment flips, the pipeline
lost `--region-run` and the stage schemas, workspaces lost run import and
snapshots, the app lost `--poses-root`, `--run`, `--recording-root`,
`--corpus-root`, `--checkpoint`, `--body-net` and `--notes` and the
recording registry, and the scripts for the bootstrap labels, the legacy
store, the speck cleanup and the old Body fields proposals went with them.
The `worm-pose-labeler` and `worm-pose-viewer` commands are gone;
`worm-pose-app --queue <manifest>` opens a labeling manifest as a Labeling
queue.

### What was built differently

- **The segmenter stays** until a body-field net matches its mask IoU
  (section 4, decision 1): a setup has a `mask` and a `body` default, and
  the Train form fine-tunes a segmenter from a segmenter. Analyse can take
  the masks from the body model instead (**Masks from**; the workspace
  settings record `mask_source`), which then needs no mask model
  ([`BODY_FIELDS.md`](BODY_FIELDS.md#masks-from-the-network)).
- **Analyse is one job** of kind `analyse`, `pipeline --stages` over the
  default stages in one process, so its progress spans the analysis. The
  dev stage list includes the opt-in `fixed_body` stage.
- **Run on** appears only when there is a choice (local GPUs and SLURM);
  with one place to run, the dialogs say nothing.
- **The Refit override** (dev) is an algorithm menu and a JSON field of
  parameter values, not a generated form; `GET /api/algorithms` documents
  the parameters.
- **New queue** picks, in each window of a recording, the candidate frame
  with the highest mask entropy per worm pixel (`frame_search.py`). Image
  types come from the same prediction: overlap output or an enclosed hole
  (self-contact), mask on the border (edge), two or more pieces, under 200
  worm pixels (no worm), else clear. A type-limited search looks at three
  times the candidates per window and gives empty windows' frames to the
  recording's other matches; a recording short of matches gives fewer.
- **The body-target builder** is the setup's default mask model, part of
  the cache key with the label revision; changing the default makes the
  targets missing until rebuilt.
- **Body proposals on the CPU.** Labeling computes the body proposal when a
  frame opens only when the server's models run on a GPU (`gpu` in
  `/api/config`); without one (15–20 s a frame) it shows **Propose**.
- **The probability layer** (the segmenter run on a rested frame) is
  computed only with `--dev`; analysts see the stored mask.
- **Not built:** the dev "Compare run (same recording)" control, and a
  drawing of the `fixed_body` overlay on the Workspace page (the frame
  payload still carries it).
- **Old data kept readable:** `PLACED_JOB_PREFIXES` still counts the
  `candidates:` provenance of sets accepted before the fixes, and the fixes
  list still names old `pick_hypothesis` and `accept_path` edits. Workspaces
  imported from runs need their `imported_summary.json` renamed to
  `summary.json`. Recordings registered by hand in the old
  `recordings_registry.json` must be added again with **Add recording**.
  `docs/pose_review/notes.json` is kept as data.

### Wave 2 contracts between the pages

- **UI shell** (`src/worm_pose_gen/app_ui/`):
  - `shell.js` routes `#<page>/<params>` to the page module's `mount(section, ctx)` / `show(params)` / `hide()`.
  - `ctx` offers `config`, `dev`, `navigate(page, params)`, `setHeader(page, {context, actions})` and `toast(message, kind)`.
  - Shared helpers: `api.js` (`api`, `post`, `query`, `el`) and `frame_canvas.js` (`FrameCanvas`, `maskCanvas`, `drawMidline`).
  - `style.css` holds the tokens and the shared classes. Each page owns `<page>.js` / `<page>.css` and any `<page>_*.js` modules. Changes to the shared files stay small and additive.
- **Model picker** (Training owns it, the Workspace uses it): `openModelPicker(ctx, {setup, role, current}) → Promise<ref | null>` in `model_picker.js`.
- **Relabel round trip** (Labeling owns the backend):
  1. The Workspace creates the queue with `POST /api/queues {kind: "relabel", workspace, frames}` → `{id}` and calls `ctx.navigate("labeling", "queue/<id>")`.
  2. When the queue is complete, Labeling offers **Back to workspace**, which goes to `#workspace/<workspace>/stitch/<id>`.
  3. The Workspace then calls `POST /api/queues/<id>/stitch`. This writes the keyframe labels' masks into the workspace as mask overrides, then starts the stitch through `fixes.run_stitch`. It answers like `POST /api/workspaces/<ws>/fixes/stitch` (`{job, preview, plan}`), and the Workspace shows the preview with Keep / Discard.
- **Edit mask** (Workspace): after the mask is saved, the refit's preview is kept automatically once its job finishes. It appears in the Fixes list with Undo.
- **Analyse** does not export: `export` leaves the default stages, and exports come only from the Export button, which passes the setup's `pixel_size_um` and `fps`.
- **Body-target cache key:** the label revision plus the model that built the targets, so changing the default model rebuilds them.

## 6. Follow-ups after this branch

Decided with Alex 2026-10-07, outside this branch's scope:

- **SLURM defaults on Engaging:** partition `ou_bcs_normal`, 12:00:00, 1 GPU,
  8 CPUs, 32 GB, as in `compute.SLURM_DEFAULTS_BY_HOST`. `ou_bcs_low` is
  avoided because preempted jobs requeue and restart from scratch.
- **Automatic head/tail patches:** short head/tail swaps between consecutive
  frames (about 12 issues on a real 1200-frame workspace) stay in the issue
  list for now. The next step is to repair them automatically before review,
  so the analyst sees only the ones the automatic patch could not settle.
- **Masks first, before mask-free stitching:** when a stitch gap's masks are
  bad, improve the masks rather than bridging without them. One candidate is
  a hole fill that fills only holes fully enclosed by the worm (full
  containment), leaving openings to the background unfilled.
