# App plan: from diagnostic viewer to pose pipeline front end

Written 2026-09-08 on branch `pose-app`, after steps 1--6 of
[`POSE_PIPELINE_PLAN.md`](POSE_PIPELINE_PLAN.md) merged. The pose viewer
(`worm_pose_gen.pose_viewer`) becomes the front end for running the
pipeline, auditing results, diagnosing failures and fixing them by hand, and
every capability of the front end is available headless through the same
API for use inside larger pipelines. Each phase ends with something usable;
the "Status" section at the bottom is updated as phases land.

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

```
worm_pose_gen.app            FastAPI application (uvicorn), serves the UI and the API
  routers/recordings         browse HDF5 roots, recording metadata, thumbnails
  routers/workspaces         create, list, open, snapshot, export
  routers/frames             layers, statistics, hypotheses (the viewer's endpoints today)
  routers/jobs               submit, list, status, log, cancel
  routers/algorithms         registry: scopes, parameter schemas, run on a region
  routers/edits              hypothesis pick, orientation flip, accept path, mask override, notes
  routers/corpus             labels into the user's segmentation store, fine-tune job
worm_pose_gen.workspace      on-disk workspace: arrays, masks, provenance, edit log, jobs
worm_pose_gen.jobs           runner with backends: local GPUs (now), Slurm (later)
worm_pose_gen.algorithms     registry and the existing methods wrapped as plugins
worm_pose_gen.client         Python client over the API; `worm-pose` command line
worm_pose_gen.pose_viewer_ui the browser UI (grows from the viewer)
```

The stdlib server of the viewer is replaced by FastAPI; the endpoints keep
their shapes so the UI carries over. Static assets stay in the package.

## 4. Workspace

One workspace per recording range, replacing the write-once run directory
(existing runs import as read-only baselines):

```
<workspaces>/<name>/
  workspace.json     recording path, frame range, settings, imported runs
  masks/             bitpacked cleaned masks in chunks of 1024 frames (npz), a few KB per frame
  overrides/masks/   sparse: frames whose mask the user edited
  state.npz          current per-frame arrays (the poses.npz layout, plus below)
  provenance.npz     per frame: algorithm id, job or edit id, time
  hypotheses.npz     candidates per frame as stored today (hypotheses_*, path_*)
  edits.jsonl        append-only log of every intervention with its inputs
  jobs/<id>.json     job record and log
  snapshots/<time>/  copies of state and provenance on demand
  exports/           Parquet files
```

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
working from the command line.

## 6. Algorithm registry

```python
class Algorithm(Protocol):
    id: str                      # "chain_forward", "beam_path", "independent_multistart", ...
    scope: Literal["frame", "region", "recording"]
    parameters: dict[str, Parameter]   # name -> type, default, bounds, help
    def run(self, ctx: Context, frames: range, params: dict) -> Candidates
```

`Context` gives masks, the recording prior, the current state and the
anchors; `Candidates` are per-frame lists of poses with energies, ready for
the path selection. Wrapped first: the independent multi-start fit, forward
and backward chains with prediction, the beam-and-path second pass, the
slow-schedule refit, the track-length refit, mirror orientation. Every run
on a region is logged with its outcome (overlap, jumps, frames accepted), so
the defaults question can be answered from the log.

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
| 1 | not started | |
| 2 | not started | |
| 3 | not started | |
| 4 | not started | |
| 5 | not started | |
| 6 | not started | |
