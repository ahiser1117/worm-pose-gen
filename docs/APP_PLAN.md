# App plan: from diagnostic viewer to pose pipeline front end

Written 2026-09-08 on branch `pose-app`, after steps 1--6 of
[`POSE_PIPELINE_PLAN.md`](POSE_PIPELINE_PLAN.md) merged. The pose viewer
(`worm_pose_gen.pose_viewer`) becomes the front end for running the
pipeline, auditing results, diagnosing failures and fixing them by hand, and
every capability of the front end is available headless through the same
API for use inside larger pipelines. Each phase ends with something usable;
the "Status" section at the bottom is updated as phases land.

**The browser UI of this plan is superseded** by
[`APP_SIMPLIFICATION.md`](APP_SIMPLIFICATION.md) (implemented 2026-10-07 on
branch `app-simplification`): three pages (Workspace, Labeling, Training)
over a lab and a personal library, with the fixes Flip, Refit, Edit mask and
Relabel in place of hypothesis picks, candidate sets, region runs and the
outcome log. That redesign also removed the read-only runs and their import,
the stdlib viewer, review notes, snapshots, the user corpus and the
standalone labeler. The principles, the workspace, the stages as jobs and
the algorithm registry below still hold; the UI parts of sections 3 and 9
are kept as history.

## 1. Users and setting

- One user at a time. The server runs on the user's own machine (today a
  4-GPU server) and is reached remotely from an interactive session on a
  computing cluster, so the app is a localhost service behind an SSH tunnel:
  no authentication, one writer, simple locking.
- Jobs run on the local GPUs now. Later the same jobs are submitted to Slurm
  to run in parallel on cluster resources while the UI stays an interactive
  remote session, so the job runner has pluggable backends from the start.
- The app will be distributed to other labs. They will not have this
  project's training corpus; they get a packaged segmenter checkpoint and the
  means to build their own labeled corpus from the app and fine-tune the
  packaged model on it.
- Exports go to Parquet.
- The set of algorithms grows. The goal is a set of defaults under which
  manual intervention is rarely needed; the app is the instrument for
  discovering those defaults, so it records what each algorithm did on each
  region and lets two algorithms be compared side by side on the same frames.

## 2. Principles

1. **Everything goes through the API.** The UI calls the same HTTP endpoints
   a script would; a Python client and a command line mirror them. A feature
   without an endpoint does not exist.
2. **Candidates in, one path out.** Every algorithm, automatic or manual,
   produces candidate poses for frames; the path selection of step 6c
   accepts them into the track. Manual work means choosing among candidates,
   drawing a mask, or flipping orientation, never dragging points.
3. **Provenance on every frame.** A workspace records for each frame which
   algorithm with which parameters produced its pose, when, and by which job
   or edit; an append-only edit log makes every intervention reproducible
   and reversible.
4. **Algorithms are plugins.** A registry entry declares its scope (frame,
   region between anchors, recording), its parameters and their defaults,
   and what it needs; the UI renders the form and the API exposes it, so a
   new method is one class.
5. **Long work is a job.** Any computation longer than a frame is queued,
   reports progress, survives a server restart, and can be cancelled.

## 3. Architecture

The modules today (after the simplification; the plan's original map is in
the git history):

```
worm_pose_gen.app            FastAPI application (uvicorn), serves the UI and the API
  routers/analysis           the Recordings screen, Analyse, workspace status, the kymograph
  routers/workspaces         a workspace's payload, frames, network fields, exports
  routers/fixes              issues, Looks OK, Flip, Refit, keyframes, stitch, previews, the fixes list, undo
  routers/masks              Edit mask: reversible mask overrides
  routers/labeling, queues   the Labeling page and its queues
  routers/library, training  the libraries; the model picker, training and evaluation jobs
  routers/jobs               submit, list, status, log, cancel, retry; where jobs can run
  routers/algorithms         the registry (the developer's Refit override)
  routers/recordings         thumbnails and the file explorer
worm_pose_gen.app_ui         the browser UI: one module set per page
worm_pose_gen.workspace      on-disk workspace: arrays, masks, provenance, edit log
worm_pose_gen.edits          the interventions: place poses, flip, edit a mask, undo; before-snapshots
worm_pose_gen.fixes          issues, refit plans, keyframes, previews, keep, the fixes list
worm_pose_gen.jobs           runner with backends: local GPUs, SLURM
worm_pose_gen.algorithms     registry and the existing methods wrapped as region algorithms
worm_pose_gen.library        setups, datasets, labels, benchmarks, model cards, body targets
worm_pose_gen.model_training, model_eval   training and evaluation on library datasets
```

History: the stdlib server of the viewer was replaced by FastAPI; the
endpoints kept their shapes so the UI carried over. Phase 1 had `routers/viewer` (the viewer's endpoints, `/api/state`, `/run`,
`/frame`, `/pose`, `/starts`, notes), `routers/recordings`,
`routers/workspaces`, `routers/jobs` (jobs and the stage schemas) and
`routers/static`. Phase 2 adds `routers/edits` (`GET/POST
/api/workspaces/{name}/edits`, `GET .../segment`) over `worm_pose_gen.edits`;
Phase 3 adds `routers/algorithms` (`GET /api/algorithms`, `GET
/api/workspaces/{name}/region`, `GET/DELETE .../candidates[/{id}]`, `POST
.../candidates/{id}/accept`, `GET /api/outcomes`) over
`worm_pose_gen.algorithms`, with region jobs submitted as `kind: region`
through `POST /api/jobs`. Requests and responses speak frames; rows appear
beside them in responses. Edits and accepts answer 409 while a job writes
the workspace (`pipeline.WorkspaceBusy`). Phase 4 adds `routers/masks` and
`routers/corpus`, with `kind: fine_tune` jobs through the existing queue.
The stdlib viewer (`worm-pose-viewer`) was kept for read-only runs until the
simplification removed it with every endpoint of this paragraph except the
jobs, the masks and `GET /api/algorithms`.

## 4. Workspace

One workspace per recording range (the app keeps one per recording, always
the whole recording), replacing the write-once run directory:

```
<workspaces>/<name>/
  workspace.json         recording path, frame range (first, last, step), settings (setup, models)
  state.npz              current per-frame arrays (the poses.npz layout)
  hypotheses.npz         candidates per frame (hypotheses_*, path_*, prediction_*)
  provenance.npz         per frame: algorithm id, job or edit id, unix time
  masks/chunk_NNNNN.npz  bitpacked cleaned masks, 1024 rows per chunk, a few KB per frame
                         (the segment stage writes them; a refit stores the masks
                         it had to segment on the fly)
  overrides/masks/       sparse full uint8 labels: 0 background, 1 worm, 255 ignore
                        (NNNNNNN.npz per row; effective geometry mask is labels == 1)
  recording_prior.json   the prior the fit stage uses (bootstrapped, cached or given)
  summary.json           a run-shaped summary kept up to date by the stages
  edits.jsonl            append-only log of every intervention with its inputs; an undo
                         is itself an entry naming the edit it undoes
  edits/<id>.npz         the before-snapshot of every array slice edit <id> changed
                         (state, hypotheses and provenance rows), what undo restores
  fixes/<id>.npz + .json a Refit or Relabel preview waiting for Keep or Discard
  human_review.json      the rows marked Looks OK, with their fingerprints
  exports/<recording>_<time>/   the per-recording table and export.json
  .lock                  flock held by the stage process that is writing the workspace;
                         edits take it too and give up after two seconds (WorkspaceBusy)
<workspaces>/jobs/       job records shared by all workspaces: <id>.json, <id>.log,
                         <id>.progress.json and the id counter (ids are never reused)
<workspaces>/queues/     the Labeling queues
<workspaces>/recordings_index.json   the recording catalog's per-file cache
```

Labels, datasets and models live in the libraries
([`APP_SIMPLIFICATION.md`](APP_SIMPLIFICATION.md), section 1).

Rows are positions in `range(first, last + 1, step)`; every per-frame
array has one entry per row. Every file is written through a temporary
file and `os.replace`, so a crash mid-write leaves the previous version
intact. The job records live beside the workspaces rather than inside them
(the plan's `jobs/<id>.json`) so one queue can be listed without opening
every workspace and so a job can outlive the workspace it ran on.

Probability maps are recomputed on demand (700 KB a frame is too much to
keep); masks are kept because refits and mask edits need them.

## 5. Pipeline stages as jobs

Each stage is a job over a frame range with parameters, reading and writing
the workspace; stages can be rerun independently:

1. segment (network probabilities, threshold, cleanup rules) -> masks
2. prior (bootstrap or cached recording prior)
3. fit independent (multi-start, batched) -> state, statistics
4. ambiguity (flags, score, seeds)
5. propagate and path (chains with prediction, beam, anchor diversity, path)
6. track length pass
7. export (Parquet)

`fit_recording.py` becomes the composition of these stages and keeps
working from the command line. Since Phase 3 the propagate stage treats
rows placed by hand or by a kept fix (`manual:*` and the region algorithms'
provenance) as fixed: the stretches are cut around them and the chains
anchor on them, so a rerun of the stage does not undo an intervention.
Analyse runs the default stages in one process (`--stages`). A Refit or a
stitch is a fix job (`python -m worm_pose_gen.fixes --workspace ... --run
'<spec json>'`); it writes a preview, never the state.

## 6. Algorithm registry

```python
class Algorithm(Protocol):
    id: str                      # "chain_forward", "beam_path", "independent_multistart", ...
    label: str
    scope: str                   # "region" for every algorithm so far
    description: str
    parameters: list[Parameter]  # name, type (int/float/bool/str/choice), default, bounds or choices, help
    needs_anchor: tuple[str, ...]   # ("before",) / ("after",) for the chains, () otherwise
    def run(self, ctx: RegionContext, params: dict, progress=None) -> CandidateSet
```

`RegionContext` gives the region's rows and masks, the workspace's fit
configuration and prior, the current state and hypotheses and the anchors
(rows outside the region whose poses are trusted; they need not be adjacent,
the algorithm works on a local copy where they are). A `CandidateSet` holds
per-row candidates (`CandidatePose`: centerline, latent, widths, crop,
energy, IoU, source and start) and the path `propagation.select_path` chose
through them with the anchors fixing its ends and orientation, plus the
region's metrics before and after (`region_metrics`: median and p10 IoU,
frames below 0.9, pose jumps over a body width, length jumps over 3%,
orientation flips, seconds). Registered: `independent_multistart`,
`chain_forward`, `chain_backward`, `beam_path` (the pipeline's second pass),
`slow_refit`, `tracked_head`, `fixed_body_smoother` (the fixed body's joint
temporal smoother, `body_smoother`) and `mirror`; the track-length refit is
still only a stage.
`run_algorithm` returns the set without writing anything; the Refit fix
saves its path as a preview (`fixes/`), and Keep installs it through one
`set_pose` edit with provenance `<algorithm>` / `fix:<preview id>`. The
stored candidate sets, their accept flow and the outcome log of Phase 3 were
removed with the simplification.

## 7. Phases

**Phase 1: foundation.** FastAPI app with the viewer's endpoints; HDF5
browser over configured roots (recordings, frame counts, existing runs and
priors, thumbnails); workspace format with mask persistence and import of
existing runs; job runner with the local-GPU backend, progress and
cancellation; stages 1--7 runnable from the UI on a chosen range.
*Done when* a minute of a chosen recording is fit from the browser, appears
in the viewer as a workspace, and any stage can be rerun from the UI.

**Phase 2: first interventions.** Hypothesis pick per frame, orientation flip
per frame and per segment, accept or reject a path, undo through the edit
log; provenance shown in the viewer (colour by source, filter by algorithm).
*Done when* a coil stretch can be corrected by picking among stored
hypotheses and the correction is recorded and reversible.

**Phase 3: region reruns and comparison.** Anchor selection (proposed, then
adjusted by the user), the registry with the existing algorithms and their
forms, a region run as a job producing candidates, side-by-side comparison
of two algorithms on the same region, the outcome log.
*Done when* a failing region is fixed by choosing anchors and an algorithm
from the UI, and two algorithms can be compared on it.

**Phase 4: segmentation override and the user's corpus.** The labeling app's
brush inside the viewer, a mask override layer, refit of the affected frames
and their stretch, edited masks saved into the user's own segmentation store,
and a fine-tune job from the packaged checkpoint on that store; the labeling
app is retired into this one.
*Done when* a mask is corrected in the viewer, the frames refit, the label
lands in the store, and a fine-tune produces a new checkpoint selectable for
segmentation.

**Phase 5: headless.** Python client and `worm-pose` command line mirroring
the API; Parquet export of per-frame pose, statistics, flags, provenance and
kinematics (speed, bending, curvature); OpenAPI documentation.
*Done when* the Phase 1--4 workflow runs from a script without the UI.

**Phase 6: audit and scale.** Audit queue walking flagged frames by severity
with verified or bad marks feeding the labeling queue and truth sets; Slurm
job backend; whole-recording runs; packaging with the bundled checkpoint.

## 8. Risks and open items

- Mask storage grows with the corpus of workspaces (about 300 MB per hour of
  video); a retention rule (keep masks for the frames in stretches and
  edits, recompute the rest) may be needed.
- The coil clips are bistable (step 6b): region reruns will sometimes return
  a worse path than the current one, so acceptance must stay explicit and
  the comparison view must make the difference visible.
- A Slurm backend needs the GPU worker to run from a job script against a
  shared filesystem; the workspace layout is chosen so that works without
  the server.
- The viewer's UI is a single script; the phases add a lot of surface, and
  splitting it into modules is part of Phase 1.

## 9. Status

| Phase | State | Notes |
|---|---|---|
| 1 | done (2026-09-08) | foundation landed on branch `pose-app`; details below |
| 2 | done (2026-09-09) | picks, flips, undo, provenance strip and jumps landed; open: whole-state rewrite per edit, no redo, no browser test in `tests/` |
| 3 | done (2026-09-09) | registry, region jobs, candidate sets, comparison, accept, outcome log landed; open: beam_path takes minutes on a 240-row coil, no warning when a set equals the current track |
| 4 | done (2026-09-09) | reversible brush overrides, stale-result guards, corpus revisions, frozen-input fine-tune jobs and checkpoint selection; legacy manifest queue retained for compatibility |
| simplification | done (2026-10-07) | the UI and much of Phases 2--4 replaced by [`APP_SIMPLIFICATION.md`](APP_SIMPLIFICATION.md); SLURM backend (Phase 6) landed with it |
| 5 | not started | |
| 6 | not started | |

**Phase 1.** `worm_pose_gen.app` (FastAPI, `worm-pose-app`) serves the
viewer's endpoints and adds `/api/recordings` (cached catalog of the HDF5
roots, thumbnails), `/api/workspaces` (create, import a run, open, frame and
pose payloads, snapshot, edits) and `/api/jobs` and `/api/stages` (submit,
list, status, log, cancel; stage parameter schemas). Behind it:
`workspace.py` (the layout of section 4), `jobs.py` (queue with a local-GPU
backend, one job per workspace, progress files, restart recovery,
cancellation), `pipeline.py` (the seven stages of section 5 as functions and
a `--stage` command line; `scripts/fit_recording.py` composes them and keeps
its flags) and `recordings.py`. The UI is split into
`api/layers/charts/viewer/panels/app.js` and gains Recordings and Pipeline
tabs. The smoke test fit 30- and 80-frame ranges of 2024-01-31-02 from the
browser through every stage; the 80-frame range matched the reference run
(IoU median 0.966, no frame below 0.9). Left open: the "Run all" chain is
driven by the browser tab (sessionStorage) and pauses when the tab closes;
command-kind jobs run any argv (a localhost-only tool); thumbnails are
computed cold per request (about 1 s on a real recording); rerunning `fit`
blanks the refit rows' hypotheses, which `propagate` must rerun to fill; the
`worm-pose-app` console script appears in `.venv/bin` only after `uv sync`
(`python -m worm_pose_gen.app` works without); the Python client and command
line are Phase 5.

**Phase 2.** `edits.py` (`pick_hypothesis`, `flip_frame`, `flip_segment`,
`flip_orientation`, `accept_path`, `set_pose`, `undo`, `list_edits`,
`segment_info`) writes poses the way the stages do, records provenance
(`manual:pick`, `manual:flip`, or the algorithm and job a caller passes),
refreshes the ambiguity signals of the touched rows and their neighbours and
saves a before-snapshot per edit under `edits/<id>.npz`; older workspaces
with centerline-only hypotheses get the pose rebuilt from the centerline.
`routers/edits` exposes `POST /api/workspaces/{name}/edits` with kinds
`pick_hypothesis`, `flip` (frame, segment or a frame list) and `undo`,
answering with the edit, the refreshed frame, a series patch of the touched
rows and the provenance so the browser updates in place; `GET .../edits` is
the log, `GET .../segment?frame=` the stretch around a frame. The UI
(`edits.js`) has "use" per hypothesis, Flip frame / Flip segment, an Edits
section with Undo (Ctrl+Z), a provenance strip over the timeline with a
legend and jumps by algorithm or to edited frames. Verified on the imported
coil run (`2026-09-08T20-02-21Z_6c_coil_0201_6b_anchor`, frames
17500--18699): pick, frame flip, segment flip over 237 rows and three
undos restored the imported poses bit for bit. Left open: every edit
rewrites `state.npz` (about a second on 1200 rows) and recomputes the
series over all rows for the patch, so whole-recording workspaces will feel
it; undo recomputes the ambiguity window rather than restoring hand-set
flags; no redo; the browser checks live outside `tests/` as Playwright
scripts; provenance colours for unknown algorithms follow their position in
the run's algorithm list.

**Phase 3.** `algorithms.py` (section 6) with the six region algorithms,
`RegionContext`, `CandidateSet`, `region_metrics`, `propose_region` and
`propose_anchors` (nearest fitted rows outside the region with ambiguity
score 0 and IoU at least 0.9), `run_region`, `accept_candidates`,
`unaccept_candidates` and the outcome log; `pipeline.py` gains
`--region-run` and `region_command`, and `run_propagate` keeps placed rows
fixed (`placed_rows`, `split_stretches`). `app/regions.py` and
`routers/algorithms` speak frames; region jobs go through `POST /api/jobs`
with `kind: region`. The UI (`regions.js`) has the Regions section (Use
current stretch, Around frame, Anchors, the algorithm form generated from
the registry, Run on region), candidate set cards with before -> after
metrics, Show as layer A or B, a comparison table of the current track and
two sets, Accept (whole or partial, disabled while an accept runs), Discard,
and the Outcomes table. Verified on the coil workspace: `GET /region` at
frame 17975 proposed frames 17765--18005 with anchors 17764 and 18006;
`beam_path` with the defaults ran 5 min 58 s (330 s in the algorithm) for
723 candidates over 241 rows and improved the region from median IoU
0.94238 to 0.94282, min 0.91610 to 0.92099, orientation flips 2 to 0, with
no frame below 0.9 before or after; accepting it made frame 17975's pose
the chosen candidate exactly with provenance `beam_path` /
`candidates:j00000002`, and `mirror` on the same region chose the
unmirrored candidate everywhere. Left open: a `beam_path` run over a whole
coil stretch takes minutes (the chains are quicker for a first look);
accepting a set whose path equals the current track only rewrites
provenance and the UI does not warn; the region's pose jump count stayed
at 1 across algorithms on that coil; the track-length refit is not a region
algorithm.

**Phase 4.** The viewer's `masks.js` brings worm/background/ignore painting,
transformed brush coordinates, draft protection and draft undo into Paint;
original prediction and editable labels are separate layers. `routers/masks`
provides `GET/POST/DELETE /api/workspaces/{name}/mask` in frames with optional
optimistic revisions. `edits.set_mask` logs save/clear operations, persists
before/after labels before writing, and restores the old override, poses,
hypotheses and provenance on undo. Ordinary write failures roll back;
write-ahead snapshots remain available after a process interruption (automatic
multi-file crash recovery is not implemented). Changing a mask marks its pose
stale, clears obsolete measurements/independent baselines/hypotheses and updates
effective-mask statistics. Ignore labels remain in the corpus; fitting uses
only explicit worm pixels. Workspace shape detection honors custom HDF5 datasets.

Refit frame/stretch controls prepare existing region jobs for explicit review
and acceptance. Slow refit and propagation candidates can initialize a row
invalidated by an override. Candidate sets capture mask input fingerprints,
including anchors, with their input arrays; acceptance rechecks under the
workspace lock. Legacy sets with unversioned inputs are rejected after relevant
mask edits. The UI marks stale sets and disables acceptance. Accepted chain
poses remain fixed when propagation reruns, identified by candidate provenance
as well as algorithm name. Corpus updates are independent actions and are not
undone with workspace edits.

`CorpusStore` extends the segmentation store with canonical recording-path /
HDF5-dataset identity, immutable revisions and a process lock. Existing stores
are supported through `--corpus-root`; relabeling and deletion preserve split
pledges. `routers/corpus` exposes label list/save/read/update/delete, training
schemas and checkpoint list/selection. The Labels browser supports structured source/split/recording filtering,
repainting and deleting saved labels; users may deliberately pledge a new label
to train, validation or test. `training.fine_tune_job` freezes label bytes and
initialization before enqueueing, and the worker trains from those inputs,
reports progress and writes unique best/last checkpoints and a run record.
Mixed image sizes use padded batches with padding excluded from the loss.
The app never auto-promotes or overwrites the configured base checkpoint.
Selecting a completed checkpoint updates future segmentation and previews,
retaining historical mask provenance and overrides. Model and frame caches
notice checkpoint replacement. CPU fine-tunes do not reserve a GPU.

The `worm-pose-labeler` launcher now opens the unified app with its legacy flags;
`--queue` opens Paint on the manifest. The old Python
module/helper APIs remain available. Packaging the checkpoint and further audit
integration remain Phase 6 work. The existing research training
script keeps its historical promotion behavior; app jobs use the new worker.

Validation includes focused mask transactions and race checks, imported / fresh /
custom-dataset API round trips, an actual small refit followed by accept and
undo with outside-region poses unchanged, corpus revision/split tests, a real
one-epoch CPU fine-tune, checkpoint-cache checks and browser brush tests.
Before the task-tabs redesign, the combined app, workspace, edits, jobs,
segmentation, corpus, viewer, pipeline and algorithm regression run passed all
132 tests on CPU.
A Chromium app smoke test painted and saved an override, saved/repainted its
corpus copy, saved a validation label, completed a fine-tune job and selected
its checkpoint for both preview and segmentation. It then ran a targeted
refit, accepted its candidate and verified that undo restored the stale pose.
The training smoke uses synthetic labels to validate the workflow; model quality
on new lab data still requires held-out evaluation.

The task-tabs redesign (Inspect → Paint → Rerun → Review, then a Paint
screen independent of workspaces) followed in September and early October
2026; both were replaced by the three pages of
[`APP_SIMPLIFICATION.md`](APP_SIMPLIFICATION.md).
