# Worm Pose Geometry

Research code for a conservative 2D *C. elegans* pose pipeline on NIR video.
The repository is organized around one geometric algorithm:

`local-darkness mask -> skeleton pose -> smooth containing body -> narrow-notch repair -> curvature-aware endpoint extension`

and, as of September 2026, its intended replacement: a generative body model
fit directly to the segmentation mask
(`flat field -> local-darkness mask -> render tube model -> optimize pose against the mask`),
documented in [`docs/MASK_FIT_EXPERIMENT.md`](docs/MASK_FIT_EXPERIMENT.md).

The canonical description, evidence boundary, current results, and limitations
are in
[`docs/POSE_ESTIMATION_TO_ENDPOINT_CURVE_EXTENSION.md`](docs/POSE_ESTIMATION_TO_ENDPOINT_CURVE_EXTENSION.md).

This remains research code. The algorithm has been exercised on one annotated
development frame and an annotation-free 30-frame stress set; it is not
deployment-authorized, and the protected holdout remains unopened. The
annotation-matched 30-frame audit is permanently retired: the exact source
frames behind 21 of the 30 manual traces were lost with their corrupted
recordings, and substituting other frames under those traces would be invalid.

## Algorithm stages

The integrated document above is the main entry point. These focused reports
and their adjacent generated assets retain the evidence for each stage:

1. [`docs/POSE_ESTIMATION_EXPLAINER.md`](docs/POSE_ESTIMATION_EXPLAINER.md) —
   conservative classical extraction from a real frame.
2. [`docs/SMOOTH_BODY_PRIOR_EXPERIMENT.md`](docs/SMOOTH_BODY_PRIOR_EXPERIMENT.md) —
   smooth midline and containing-width model.
3. [`docs/BOUNDARY_NOTCH_REPAIR_EXPERIMENT.md`](docs/BOUNDARY_NOTCH_REPAIR_EXPERIMENT.md) —
   geometry-only narrow-notch repair.
4. [`docs/ENDPOINT_CURVE_EXTENSION_EXPERIMENT.md`](docs/ENDPOINT_CURVE_EXTENSION_EXPERIMENT.md) —
   local constant-curvature continuation to the completed-body boundary.
5. [`docs/final_algorithm_unannotated30/FRAME_STEPS.md`](docs/final_algorithm_unannotated30/FRAME_STEPS.md) —
   per-frame diagnostics for the 30-frame operational stress run.
6. Section 8 of the integrated document — three follow-on runs on the same 30
   frames: recording-level flat-fielding with field-of-view completion
   (`docs/final_algorithm_edge_aware_unannotated30/`), visible-only
   edge-censored repair (`docs/final_algorithm_edge_censored_unannotated30/`),
   and the rejected interactively tuned segmentation setting
   (`docs/final_algorithm_tuned_local_darkness_unannotated30/`).
7. [`docs/MASK_FIT_EXPERIMENT.md`](docs/MASK_FIT_EXPERIMENT.md) — the first
   generative-model step: the frame is flat-fielded, segmented, and the
   20-value body model plus a width scale is rendered as a soft tube and fit
   directly to the mask by gradient descent, producing a pose on all 27 worm
   frames of the stress set.
8. [`docs/SEGMENTATION_LABELING.md`](docs/SEGMENTATION_LABELING.md) — the
   learned-segmentation loop: bootstrap labels from the pipeline, fine-tune a
   pretrained ResNet-18 U-Net with Lightning, and refine labels in the
   browser app with the network proposing.

## Setup

Python 3.13 and `uv` are required. The checked-in wrapper keeps the environment
and caches local to the checkout.

```bash
scripts/bootstrap_environment.sh
scripts/project_env.sh uv run --no-sync --frozen python -m unittest \
  tests.test_classical tests.test_anchors tests.test_annotation tests.test_latent
```

The default one-frame builders expect the cached proxy HDF5 and annotation JSON
at the paths declared in `scripts/build_smooth_body_prior_experiment.py`.
Equivalent paths can be supplied through each script's command-line options.

## Rebuild the documented one-frame progression

Run the stages in order because the endpoint builder consumes the notch-repair
arrays:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/build_pose_estimation_explainer.py
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/build_smooth_body_prior_experiment.py
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/build_boundary_notch_repair_experiment.py
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/build_endpoint_curve_extension_experiment.py
```

## Tune the local-darkness segmentation

Launch the local browser app against the cached proxy frames:

```bash
scripts/project_env.sh uv run --no-sync --frozen python -m \
  worm_pose_gen.heuristic_tuner
```

Then open `http://127.0.0.1:8766`. The controls recompute the same local
background, denoising, threshold, closing, and largest-component stages used by
the classical extractor, starting from the frozen `ClassicalConfig` defaults
(`31 px` background radius, `2 px` denoise radius, `2.6 z` cutoff, hysteresis
disabled, `2 px` closing). Optional connected hysteresis admits a lower cutoff
only where it remains connected to the high-confidence worm component; the
downloaded JSON maps directly to `ClassicalConfig`.

Settings that look clean in the tuner must be evaluated on the 30-frame stress
run before promotion. The one setting promoted so far (`61 / 3 / 4.25 / 2.05 /
8`) cut acceptance from 11 to 3 of 30 frames and was reverted; its run is kept
under `docs/final_algorithm_tuned_local_darkness_unannotated30/`.

Use `--proxy-hdf5 /path/to/proxy_labels.h5` or `--port PORT` to override the
defaults. The app binds to localhost and does not modify the source HDF5.

## Segment with a fine-tuned network

Bootstrap labels, train, evaluate, and label interactively:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/bootstrap_segmentation_labels.py --frames-per-recording 40
scripts/project_env.sh uv run --no-sync --frozen python scripts/train_segmenter.py --name hand_labels
scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate_segmenter.py
scripts/project_env.sh uv run --no-sync --frozen python scripts/plot_segmenter_history.py
scripts/project_env.sh uv run --no-sync --frozen python -m worm_pose_gen.app \
  --corpus-root /temp_data4/alex/external_artifacts/datasets/worm_pose_gen/segmentation_v1
```

Labels are stored under
`/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/segmentation_v1`
on flv-c4 with an 80/10/10 train/val/test assignment; checkpoints go to the
git-ignored `checkpoints/segmenter/` directory. The bootstrap step only
matters for a fresh store: the bootstrapped labels of this one were retired
on 2026-09-05 (`scripts/retire_bootstrap_labels.py`), every label is
hand-refined, and the promoted model is `r2-hand165` (see
`docs/segmenter_model_names.json` for the model names the plots use). New
labeling work uses the pose app at `http://127.0.0.1:8768`: paint masks in
Paint, save labels, browse them in Labels, and fine-tune in Training (see
[Segmentation corrections and corpus](#segmentation-corrections-and-corpus)).
The `worm-pose-labeler` command now opens this unified interface while
preserving its old recording, dataset-root, device and port flags.
The launcher's `--queue` opens the manifest in unified Paint navigation.
The module `python -m worm_pose_gen.label_app` retains the legacy interface described in
[`docs/SEGMENTATION_LABELING.md`](docs/SEGMENTATION_LABELING.md).

A targeted round (coils, self-contact, holes, fragments, camera-edge frames
across 13 recordings, with some animals held out for validation or test only)
is queued in `docs/labeling_round_2/manifest.json`, built by
`scripts/build_labeling_manifest.py` from the clip-candidate scans. Open it
with:

```bash
scripts/project_env.sh uv run --no-sync --frozen python -m worm_pose_gen.label_app \
  --queue docs/labeling_round_2/manifest.json
```

The "Queue (manifest)" next mode walks the frames in order; each save
pledges the recording's split.

## Fit poses over a recording

Segment, clean, and fit the body model to every frame of a stretch, in GPU
batches, with an optional overlay video:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/fit_recording.py \
  --recording /store1/shared/all_data_raw/prj_aversion/2024-05-28/2024-05-28-02.h5 \
  --start 0 --frames 1200 --video --scale 0.5
```

Each run writes `summary.json`, `poses.npz`, and `overlay.mp4` to a
timestamped directory under `/temp_data4/alex/external_artifacts/poses/`.
`--preset fast` (default, about 0.25 s/frame) trades 0.008 median IoU against
`--preset reference`, which reproduces the single-frame fitter exactly at
about 5 s/frame. The width of the tube is a scale times a symmetric template
times a smooth log-space correction (`--width-coefficients`, default 6, 0 for
the symmetric model; `--width-prior` pulls it toward zero), so the two ends
may taper differently; every stored pose is oriented with the thinner end,
the tail, last (`--no-orient` keeps the fitted orientation). In the overlay
the head is a square and the tail a circle. Each run also writes residual
images (mask the tube misses in blue, tube outside the mask in red) for its
`--residual-frames` worst frames and any `--dump-frames`;
`scripts/render_pose_run.py` produces the video and residual images for a
stored run without refitting, and `scripts/compare_pose_runs.py` puts
several runs side by side on the same frames.

Runtime improvements and reproducible GPU 3 benchmarks are documented in
[Runtime optimization](docs/RUNTIME_OPTIMIZATION.md). The default optimizations
preserve fitting schedules. `--compile-energy` additionally compiles the whole
independent fitting loss; it adds startup cost and should be benchmarked on
your workload before enabling.

The pose viewer is the interactive way to look at a run:

```bash
scripts/project_env.sh uv run --no-sync --frozen python -m worm_pose_gen.pose_viewer
# every run under /temp_data4/alex/external_artifacts/poses/; --run <dir> adds
# others, --only-runs serves just those; then open http://127.0.0.1:8768/
```

It scrubs through a run's frames (arrow keys, play, click or drag on the
timeline, wheel to zoom the timeline range; the image, centerlines and saved
statistics follow the cursor). During playback and scrubbing, expensive
layers are deferred. Pausing or releasing the scrubber restores the selected
frame's masks, tube outlines and residuals with the existing layer settings.
Workspace candidate checks, mask fingerprints and comparison requests also
wait until motion stops; candidate acceptance always validates its input masks.
Background prefetching only loads cheap previews. The viewer
composites every layer the pipeline produced on the flat-fielded frame,
each with its own toggle and opacity: the segmenter's probability heat map, the thresholded mask, the
pixels a hole fill adds and the pixels the largest-component rule drops,
the mask the fit was scored against, the fitted tube's outline and
centerline with head and tail markers, the residual (mask the tube misses
in blue, tube outside the mask in red), the independent fit that
propagation replaced (runs fit after 2026-09-07 store it), the pose of a
second run of the same recording, the fitter's skeleton and moment starts,
and the crop window. The right panel lists every per-frame statistic the
run tracks, the nine ambiguity flags with the value each tested against its
threshold, a classification of the frame (clean, watch, or ambiguous;
coil, camera edge, fragmented mask, or suspected fit failure; propagated
forward or backward), the width profile along the body against the
symmetric template and the recording prior's shape, the body's curvature,
and the mask statistics recomputed on the spot next to the stored ones.
The timeline shows IoU (with the independent fit's), body length with the
prior's two-sigma band, mask and visible tube area, the ambiguity score, a
selectable extra series (pose jump, self-contact, width, energy, ...), the
flag raster, and a class/source strip, with propagation stretches shaded.
"Jump to" walks flagged or low-IoU frames, stretches, jumps and edge
frames; review notes (tags and a comment per frame) are appended to the
file named by `--notes`, `docs/pose_review/notes.json` by default. The
three panels around the frame resize by dragging their splitters and
collapse from the splitter buttons (double-click resets); the layout is
remembered by the browser.

By default the run first bootstraps a recording prior: frames spread over
the whole recording are fit with the hard bounds opened, whole worms (mask
clear of the image border) are kept, and robust medians of body length,
width scale and width profile become Gaussian priors that replace the hard
bounds (`recording_prior.json` in the run directory, cached under
`/temp_data4/alex/external_artifacts/recording_priors/`). A body that
leaves the camera is started at the prior length, extended off camera
through the point where the mask meets the border, and its off-camera part
is censored; the in-view fraction reports how much was seen. Every frame is
started in both orientations and the energy gap between them is stored.
`--prior none` restores the hard bounds, `--prior-file` reuses a stored
prior, `--rebootstrap` ignores the cache. Every frame also gets an
ambiguity score (`worm_pose_gen.ambiguity`: low overlap, self-overlap or
missed body by area, self-contact, enclosed holes, fragments, length far
from the prior, a jump since the previous frame); a score of 2 or more marks
a frame whose single-frame answer should not be trusted without its
neighbours. `scripts/ambiguity_report.py` recomputes it for stored runs.
Stretches of such frames are then refit by temporal propagation: the good
pose before the stretch is carried forward through it and the good pose
after it backward, each frame warm-started from its neighbour, all
stretches in lockstep, and per frame the lowest total energy among
independent, forward and backward wins (`source` in `poses.npz`;
`--no-propagate` skips it). Inside a chain each frame is also started from
a first-order prediction of the pose (the last change of shape, rotation
and centroid carried on with damping 0.6, `--prediction-damping`) and the
fit is pulled toward that prediction by a temporal prior
(`--temporal-prior-weight`, default 0.01, sigma half a width). The
propagation pass is the second pass over the ambiguous stretches: it
keeps up to three distinct chain states per direction (`--beam`), refits
each stretch frame's independent pose under the chain schedule with the
stretch's anchor length, and chooses one candidate per frame along the
stretch by dynamic programming over all candidates and their mirrors
(energy over `--path-temperature` plus the squared pose distance in
widths times `--path-distance-weight`, the in-view change times
`--path-inview-weight`, and the squared log length change times
`--path-length-weight`; `--no-path` restores the lowest energy per frame).
`--propagate-preset balanced` runs that preset's steps in the pass; it
was slower and worse on the raw spiral, so the default stays `fast`.
Stretches are also seeded by jumps of the track (a pose jump of more
than a width, or the body entering or leaving the camera at the border;
`--no-jump-seeds`, `--seed-length-fraction` adds length jumps). After
propagation a track length pass refits the frames outside the stretches
whose mask reaches the border and whose length departs from the track
(the median length of the whole bodies within `--track-window` frames)
with that length as their prior (`--track-refit`, `--track-sigma`,
`--track-tolerance`, `--no-track-length`); run before propagation it
disturbed the stretch anchors. The fitter's energy penalises bends tighter than a
radius of `--min-bend-radius` body widths (default 0.5; 0 disables). Each
frame also stores the fraction of the tube covered by mask
(`tube_coverage`), which stays high when a low IoU comes from mask the
tube does not claim, such as a plate streak segmented as worm; the viewer
tags such frames "mask has extra body (segmentation)". Every candidate is stored (`hypotheses_*`
arrays, `path_*`, `prediction_xy`) and the viewer draws them, ranked by
energy, with the path's choice and where it overrode the lowest energy.

The sequence evaluation set, seven 300-frame clips with coils, self-contact,
fragments and camera exits, is the manifest `docs/sequence_eval_set.json`;
`scripts/find_sequence_clips.py` proposes such clips from a mask-only scan
and `scripts/evaluate_sequence_set.py` fits and scores the set. The plan
this belongs to, with measurements, is
[`docs/POSE_PIPELINE_PLAN.md`](docs/POSE_PIPELINE_PLAN.md);
`scripts/evaluate_width_model_unannotated30.py` and
`scripts/evaluate_recording_prior_unannotated30.py` compare width models and
priors on the 30-frame set.

## Pose app

The pose app is the front end for running the pipeline, auditing its results
and fixing them by hand: picking among a frame's stored hypotheses, flipping
orientations, rerunning an algorithm on a region between anchors and
accepting its result, every step recorded and reversible; its plan and status
are in [`docs/APP_PLAN.md`](docs/APP_PLAN.md). Start it with

```bash
scripts/project_env.sh uv run --no-sync --frozen python -m worm_pose_gen.app
# or, after `uv sync`, `worm-pose-app`; then open http://127.0.0.1:8768/
```

By default it browses the recordings under
`/store1/shared/all_data_raw/prj_aversion` (`--recording-root`, repeatable),
lists the runs under `/temp_data4/alex/external_artifacts/poses/`
(`--poses-root`, `--run`) and keeps its workspaces and job records under
`/temp_data4/alex/external_artifacts/workspaces` (`--workspaces-root`);
`--gpus 1,2,3` restricts the GPUs jobs run on (one job per GPU at a time,
`--max-concurrent`), `--checkpoint` names the segmenter for on-demand
probability maps and for the segment stage. The app binds to localhost and
is meant to be reached through an SSH tunnel; nothing authenticates.

Choose **Job GPU** in Run for stage and region jobs, or **GPU for retries**
in Jobs before clicking **Retry** on a finished job. The selection is remembered
in the browser; a specific GPU waits for its slot instead of switching devices.
The available devices come from `--gpus` (physical GPU IDs); for example,
`--gpus 3,0 --max-concurrent 1` makes GPU 3 the automatic first choice.
The API accepts `"gpu": 3` on job submissions and
`POST /api/jobs/{id}/retry`; `null` means automatic selection. Omitting the
GPU on a retry preserves the previous job's device and parameters.

The workspace UI follows **Run → Inspect → Paint → Compare → Export**.
The selected frame, range, view and draft survive task changes. Import and Open
are separate central panels. Statistics, Layers, Jobs and History are tabs in
the resizable right panel; Shortcuts remains a toggleable drawer.
The header keeps the current frame, exact selection bounds, and saved/draft
state visible. Frame-specific side panels are hidden outside the workspace;
Training can open Jobs and restores the workspace's side-panel view on return.

- **Import** browses any accessible directory, uses `/img_nir` automatically,
  and prepares the illumination correction before showing the corrected preview.
  Calibration, computation and loading steps are visible. Choose **Entire
  recording** (the default) or **Selected range**, check the inclusive bounds
  and frame count, then **Create workspace & configure pipeline**.
- **Open** selects a workspace or read-only run before an explicit **Resume
  workspace**, **Open run read-only**, or **Create editable copy** action.
- **Run** opens new workspaces with setup guidance, a checkpoint/configuration
  summary, detailed stage settings and **Run pipeline**. It runs checked stages
  (segment, prior, fit, ambiguity, propagate and track) sequentially and opens
  Jobs. Completion offers **Inspect results**. Current-frame and selected-range
  runs retain their anchors, parameters and explicit scope. Scope comes before
  configuration, and segmentation checks that the selected checkpoint is available.
- **Inspect** lists unprocessed and flagged segments with reasons and human-review
  state. **Minimum segment length** defaults to **8 sampled frames** and is
  adjustable and remembered across reloads. The queue and next-segment actions
  honor this filter; shorter regions and their flags are preserved. Selecting a segment seeks
  to it and sets the range for looping, painting, orientation correction and
  another fit. **Mark reviewed & next** persists human review independently of automatic
  flags. Corrections preserve reviews of unchanged segments; changed frames and
  their neighbors need review again. Global input/configuration changes reset review.
- **Paint** keeps its existing brush, proposal and refinement tools visible.
  Save actions stay pinned in the tool pane; narrow layouts place the viewer first.
  **Save & refit selected range** saves the current frame's mask before opening
  Run with the selected bounds. Refit candidates are never accepted automatically.
- **Compare** switches the main viewer between Current/A/B overlays and synchronized
  side-by-side previews of selected options. One acceptance group shows exact bounds, changed mask
  inputs block stale candidates, and another attempt preserves the range.
- Deleting a computed option requires confirmation naming its option and range.
- **Export** uses a central checklist with links to Inspect, Paint and Jobs. It shows saved workspace scope, destination and outstanding review
  status, then writes named Parquet from a matching workspace snapshot. Names
  cannot overwrite prior exports. The download points to the captured result,
  so later corrections cannot change it. Export is separate from Run pipeline.

Labels browses saved segmentation samples and opens frames for labeling;
Training presents label counts and prerequisites before common settings, with
advanced settings in a disclosure and explicit checkpoint selection. These are
optional supporting workflows. Jobs remains available alongside Training and
shows progress, logs, results and cancellation. Job records survive restarts.
Stage jobs use `python -m worm_pose_gen.pipeline --workspace ... --stage ...`.

Recordings are found under the recording roots (`--recording-root`, repeatable;
default `/store1/shared/all_data_raw/prj_aversion`), and can be added from any
accessible directory through Import. The browser provides editable paths,
folder navigation, breadcrumbs and Up. The Import header and Close button
remain visible while its contents scroll. Registration uses
`POST /api/recordings/register {path, dataset}` and preparation uses
`POST /api/recordings/prepare`; `GET /api/recordings/preparation` reports its
current step. The registry lives in `<workspaces root>/recordings_registry.json`.
Scripts can use `GET /api/files?path=` (`&all=1` lists all files) and
`GET /api/recordings/datasets?path=` to inspect files and alternate datasets.

The read-only viewer (`worm-pose-viewer`) serves the same UI on the same
default port but has none of these endpoints; the UI says so in the
Import and Run screens and disables what needs the app, and errors
from the server appear as a toast over the frame as well as in the status
line.

A workspace holds a recording range's masks, poses, hypotheses and
provenance (which algorithm and job or edit produced each frame's pose, and
when), its edit log with a before-snapshot per edit, its candidate sets,
snapshots and Parquet exports (one row per frame with pose, statistics,
flags, provenance and kinematics); the layout is in section 4 of the plan.
Existing runs appear read-only in the viewer's source list and "Import run as
workspace" copies one into a workspace so its stages can be rerun and its
frames edited. The frame panel shows each frame's provenance (algorithm, job
or edit, time) and the source summary counts frames per algorithm; a second
run or workspace of the same recording can be compared on the same frames.

### Interventions

After reviewing and repairing poses, the optional **Fixed body** stage adds a
separate overlay with a single frozen length and width profile across the
workspace. Run it from **Run → Whole workspace → Detailed stage configuration**;
it is excluded from the default pipeline. See [Fixed body overlay](docs/FIXED_BODY.md)
for calibration, extrapolation, and result storage.

On a workspace the viewer edits frames; every edit is one line of
`edits.jsonl` and can be undone:

- **Hypothesis pick.** The frame panel's hypotheses table lists the stored
  candidates of the frame (the independent starts and the forward and
  backward chains the propagate stage kept, with their IoU and whether the
  path chose them); "use" on a row makes that candidate the frame's pose, as
  it is or mirrored. Older workspaces that store only the candidates'
  centerlines get the pose rebuilt from the centerline (latent re-encoded,
  the frame's widths and crop carried over). The frame keeps its ambiguity
  signals up to date: the edit recomputes them for the touched rows and their
  neighbours, since a pose jump belongs to a pair of frames.
- **Flips.** "Flip frame" reverses the frame's orientation; "Flip segment"
  reverses the whole propagation stretch around it (or, outside a stretch,
  the run of fitted frames between the neighbouring stretches), which is how
  a coil that came out backwards end to end is fixed in one step.
- **Undo.** The Undo button in History undoes the
  newest live edit; the Edits section lists the log newest first, strikes
  undone entries through, and clicking an entry jumps to its frames. Undo
  restores the before-snapshot of every array slice the edit touched
  (`edits/<id>.npz`), so a pick, flip or accept comes back bit for bit,
  and it is itself a log entry (redo is not offered).
- **Provenance.** Workspaces get a provenance strip over the timeline,
  one colour per algorithm (independent fit, forward and backward chain,
  track refit, the region algorithms) with manual edits in pink and a tick
  on edited rows; the legend under the charts counts frames per algorithm,
  the "◀ prov. / prov. ▶" buttons jump between frames of the chosen
  algorithm or to the frames manual edits touched, and the frame panel names
  the algorithm, the job or edit and the time for the current frame.

Edits answer 409 while a job is writing the workspace (and raise
`pipeline.WorkspaceBusy` after two seconds when another process holds its
lock), and a rerun of the propagate stage keeps manually placed and accepted
rows fixed and anchors its chains on them rather than overwriting them.

### Regions

The Run tab's selected-range controls reruns an algorithm on part of a
workspace and compares the result with the current track before anything is
changed:

- **Region and anchors.** "Use current stretch" proposes the propagation
  stretch around the current frame padded by two frames, "Around frame" ten
  frames either side, or type the first and last frame; "Anchors" proposes
  anchors, the nearest fitted frames outside the region with ambiguity
  score 0 and IoU at least 0.9 (`GET /api/workspaces/{name}/region?frame=`
  or `?first=&last=`). Anchors need not be adjacent: the algorithms run on a
  local copy in which the anchors sit next to the region, so the chains and
  the path connect the region to the anchors chosen. Everything is in frames.
- **Algorithms.** The registry (`GET /api/algorithms`) lists each
  algorithm with its parameters, defaults and bounds, and the form is
  generated from it. `independent_multistart` fits every frame from the
  standard starts of its mask (both orientations); `chain_forward` and
  `chain_backward` run one chain from the anchor before or after the region
  with prediction (`prediction_damping`), temporal prior
  (`temporal_prior_weight`, `temporal_prior_sigma_widths`), beam width
  (`beam`) and a length prior centred on the anchors (`chain_length_sigma`);
  `beam_path` is the pipeline's second pass on the region (both chains, the
  refit independent poses with `refit_independent`, `anchor_diversity`);
  `slow_refit` refits the current poses under a longer schedule with the
  anchors' length prior (`preset`, `length_sigma`); `mirror` fits nothing and
  offers each frame's pose and its reversal, an orientation fix over a
  region. **Head-tracked temporal fit** (`tracked_head`) uses the recording's
  acquisition nose landmarks with a gentle previous-pose prior and strong head priors,
  an adjustable maximum head movement (default 8 pixels per recorded frame),
  and a head-in-frame constraint. It runs forward and keeps its fitted head
  orientation. See [tracked-head fitting](docs/TRACKED_HEAD_FITTING.md) for
  the tracking schema, controls, and anchor behavior. **Fixed-body temporal
  smoother** (`fixed_body_smoother`) fits nothing: it re-expresses the
  region's current poses as one fixed-length, fixed-width body and smooths
  them jointly under a first-order motion prior on head position and bending
  whose scales are measured on the workspace's trusted frames. Trusted frames
  (IoU at least `min_iou`, ambiguity score below 2) keep their pose; the
  others are bridged from their neighbours (`untrusted_weight`,
  `data_sigma_px`, `motion_tolerance`, `min_calibration_frames`). See
  [fixed body](docs/FIXED_BODY.md#temporal-smoother). Other algorithms share the path-selection weights
  (`path_temperature`, `path_distance_weight`, `path_inview_weight`,
  `path_length_weight`) and, where it fits, the schedule `preset`. The chains
  require their anchor; a request without one is a 400.
- **Hole filling for a refit.** Each fitting method offers **Hole filling**:
  **Use workspace masks** retains existing behavior; **On** fills narrow holes
  for this run; **Off** resegments unedited frames without hole filling, since
  saved binary masks may already have holes filled. Manual masks remain the
  source for edited frames, and ignored pixels stay excluded. These options
  do not replace saved masks. The choice is recorded with the candidate set.
  Head-tracked temporal fit defaults to Off; other fitting methods default to
  Use workspace masks. Parameter explanations appear on hover.
  Mirror and the fixed-body smoother have no hole-filling control because
  they do no fitting.
- **Candidate sets.** "Run on region" submits a job (`POST /api/jobs` with
  `kind: region`, `workspace`, `algorithm`, `first`, `last`,
  `anchor_before`, `anchor_after`, `params`); the job stores per-frame
  candidates and the path chosen among them under `candidates/<job id>.npz`
  and writes nothing to the state. The set's card shows the region's metrics
  before and after (median and p10 IoU, frames below 0.9, pose jumps over a
  body width, length jumps over 3%, orientation flips); "Show" overlays a
  set's chosen candidates on the frame as layer A or B and the comparison
  table sets the current track and up to two sets side by side, and the
  frame panel lists the sets covering the current frame with their
  candidate for it. A region job on an imported workspace segments the
  masks it lacks and stores them, so the next run on those frames is faster.
- **Accept.** "Accept" (`POST .../candidates/{id}/accept`, optionally with
  `rows`, a list of frame numbers, for part of the path) installs the set's
  path as the frames' poses in one `accept_path` edit, puts the set's
  candidates into the frames' hypotheses, and records the algorithm and
  `candidates:<id>` as provenance; undo restores the previous poses and
  hypotheses and un-marks the set. A set accepted on part of its path stays
  open for the rest; accepting rows already accepted is refused; "Discard"
  (`DELETE .../candidates/{id}`) removes a set.
- **Outcomes.** Every region run appends a line to
  `<workspaces root>/algorithm_outcomes.jsonl` with the region, anchors,
  algorithm and parameters, the metrics before and after, and later whether
  it was accepted or the accept undone (`GET /api/outcomes?workspace=&algorithm=`);
  the Outcomes table filters by algorithm and workspace, so which defaults
  make manual work rare can be read off the log.

A region run is a subprocess the job runner starts, and it also works by
hand (rows, not frames, in the spec; the id names the candidate set):

```bash
PYTHONPATH=src .venv/bin/python -m worm_pose_gen.pipeline \
  --workspace /temp_data4/alex/external_artifacts/workspaces/<name> \
  --region-run '{"algorithm": "beam_path", "first": 265, "last": 505,
                 "anchor_before": 264, "anchor_after": 506,
                 "params": {"beam": 3}, "id": "coil_beam"}'
```

Everything the UI does goes through the HTTP API (`/api/recordings`,
`/api/workspaces` with `/{name}/edits`, `/{name}/segment`, `/{name}/region`
and `/{name}/candidates`, `/api/algorithms`, `/api/outcomes`, `/api/jobs`,
`/api/stages`, and the viewer's `/api/state`, `/api/run`, `/api/frame`,
`/api/pose`, `/api/starts`), so a script can drive the same work; `/docs` is
the generated OpenAPI page. The stdlib viewer,
`python -m worm_pose_gen.pose_viewer` (`worm-pose-viewer`), remains for
looking at runs read-only without the job runner; it serves the same UI on
the same default port, so run only one of the two or pass `--port`.

### Segmentation corrections and corpus

In **Paint**, use Worm, Background or Ignore with the brush-size control.
Network, Classical, raw Threshold and Saved proposals have explicit previews;
Apply combines them by replace, union, intersection or subtraction. Fill holes,
largest component, grow, shrink and tube fit are undoable draft operations.
All controls stay expanded, including inapplicable controls with disabled reasons.
Right, middle, Shift or Alt drag pans. The Shortcuts drawer lists the single
binding registry; no chord changes meaning between tasks. Z/Ctrl+Z/Cmd+Z only
undo draft changes. Saved edits are undone explicitly in History.

Next frame supports sequential stride, network uncertainty, random unlabeled,
manifest queue and filtered saved labels. The declared workspace, selection,
recording or manifest pool bounds traversal; P returns to visited targets.
Changing frames or sources with an unsaved draft offers Save, Discard or Stay.
Changing task tabs preserves the draft.

**Save mask** records a reversible workspace edit. **Also save a training label**
is optional; Save + next advances only after all selected destinations succeed.
Partial saves report each result and retry only the incomplete destination.
The full label keeps ignore pixels, while the fitter uses only explicit worm
pixels. Mask changes invalidate the old pose and mark it for refitting.
**Remove override** restores the automatic segmentation and is undoable through
History. Clear draft and Discard draft affect only the unsaved draft. Corpus
labels remain independent: undoing a workspace edit does not undo a corpus save.

**Refit frame…** prepares an independent multi-start region run; **Refit
stretch…** prepares a slow refit on the proposed stretch. Review its bounds,
anchors and parameters, run the job, compare candidates, then explicitly
Accept. Neither button reruns the whole workspace. Candidate sets record the
input masks, including anchors, and cannot be accepted after those masks
change. A segmentation-stage rerun preserves overrides.

**Save label to corpus** copies the current draft and raw/corrected source
images into `--corpus-root` (default `<workspaces-root>/corpus`). Choose Auto
for balanced 80/10/10 assignment or explicitly pledge a new label to train,
validation or test. A frame retains its pledge through edits, deletion and
relabeling. Recording identity includes the full path and HDF5 dataset;
same-named files do not collide. Each app save archives an immutable revision.
Pass an existing segmentation store to `--corpus-root` to continue using it.

In **Labels**, filter by source, split and recording, open, repaint or delete
saved labels, or open a new recording frame. Fine-tuning in **Training** needs
at least one train and one validation label. **Start fine-tune job** snapshots
the exact label revisions and configured worm checkpoint before queueing;
later corpus edits cannot change the training inputs. Progress, logs and
cancellation use the right-panel Jobs tab. Outputs live under
`--checkpoints-root` (default `<workspaces-root>/checkpoints`) in separate run
directories with the input snapshot, run record, metrics and best/last weights.
The app does not replace the base checkpoint or automatically promote a run.

After a job finishes, select its checkpoint and click **Use in workspace**.
This changes future segmentation and on-demand probabilities while retaining
the checkpoint provenance of stored masks. Run segment when ready to replace
the automatic masks; manually saved overrides still take precedence. The
research script `scripts/train_segmenter.py` retains its earlier promotion
behavior; the app uses `worm_pose_gen.training`.

The same operations are available through the API:

| Operation | Endpoint |
|---|---|
| Read/save/clear override | `GET/POST/DELETE /api/workspaces/{name}/mask` (`frame` query or body; POST includes PNG `mask` and optional `revision`) |
| Undo saved override | `POST /api/workspaces/{name}/edits` with `kind: undo` |
| Browse/save labels | `GET /api/corpus`, `POST /api/corpus/labels` with `workspace`, `frame`, optional `split` |
| Read/edit/delete label | `GET/PUT/DELETE /api/corpus/labels/{sample_id}` |
| Label draft/proposals/refinement/traversal | `POST /api/labeling/frame`, `/proposals`, `/refine`, `/save`, `/next` (under `/api/labeling`) |
| Queue manifests | `GET/POST /api/labeling/manifests` |
| Training schema/job | `GET /api/training`, `POST /api/jobs` with `kind: fine_tune` and `params` |
| List/select checkpoint | `GET /api/checkpoints`, `POST /api/workspaces/{name}/checkpoint` with `checkpoint` ID or path |

Browser regressions include `mask_editor.cjs`, `paint_tasks.cjs`,
`task_labels_review.cjs`, `task_shortcuts.cjs`, `workflow_run.cjs`,
`workflow_compare_export.cjs`, `inspection_filter.cjs`, `usability_library.cjs`
and `usability_shell.cjs` under `tests/browser`; the shell test uses a separately
started `phase4_fixture.py` at `UI_BASE_URL` (default `http://127.0.0.1:18770`). The complete workflow
is `tests/browser/phase4_workflow.cjs`, which starts its own synthetic CPU app,
trains for one epoch, accepts a regional refit, checks undo, records human
review and downloads a named snapshot export. It removes its
temporary labels/checkpoints on completion. Run either with `node` from the
repository root; set
`PLAYWRIGHT_MODULE` and `CHROMIUM_EXECUTABLE` if they are not installed in the
default locations. Python coverage includes `test_mask_edits`, `test_mask_api`,
`test_corpus`, `test_corpus_api`, `test_labeling_api`, `test_phase4_integration`
and `test_label_launcher`. The approved design is in
[`docs/UI_TASK_TABS_PLAN.md`](docs/UI_TASK_TABS_PLAN.md).

## Evaluate the frozen pipeline

The annotation-free stress run accepts exactly three `--recording` arguments
when the documented default recordings are unavailable:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/evaluate_final_geometry_unannotated30.py \
  --recording /path/to/first.h5 \
  --recording /path/to/second.h5 \
  --recording /path/to/third.h5 \
  --workers 3
```

The mask fit and the two follow-on comparisons use the same three recordings
and frame positions:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/evaluate_mask_fit_unannotated30.py
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/evaluate_edge_aware_geometry_unannotated30.py --workers 3
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/evaluate_edge_censored_geometry_unannotated30.py --workers 3
```

`scripts/evaluate_final_geometry_primary30.py` is the annotation-matched audit.
It requires readable copies of the exact source frames, which no longer exist,
so it cannot run to completion. It is kept because the unannotated evaluators
import its per-frame fitting code.

## Repository layout

- `src/worm_pose_gen/` contains reusable geometry, classical extraction, the
  mask fitter, the segmenter and its dataset store, the labeling app, and
  supporting research modules.
- `scripts/` contains only environment setup and the builders/evaluators for
  the current geometric pipeline.
- `docs/` contains the current algorithm narrative and its generated evidence.
- `tests/` contains focused geometry tests plus reusable-library coverage.
- `experiments/` retains machine-readable research outputs. The primary audit
  consumes the frozen selection manifest and baseline metrics stored there;
  historical narrative notes and embedded runners have been removed.
- `artifacts/` and `configs/` remain as data from the earlier research program;
  they are not part of the current documentation or script workflow.
