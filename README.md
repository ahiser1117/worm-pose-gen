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
   learned-segmentation loop: hand labels, a pretrained ResNet-18 U-Net
   fine-tuned with Lightning, and how the labels were made and moved into
   the library.
9. [`docs/BODY_FIELDS.md`](docs/BODY_FIELDS.md) — the body-field network's
   targets (A-P field, head/tail, overlap) built from the hand labels, and
   the network's evidence and trace starts in the pose fitter (`--body-net`).

## Setup

Python 3.13 and `uv` are required. The checked-in wrapper keeps the environment
and caches local to the checkout.

```bash
scripts/bootstrap_environment.sh
scripts/project_env.sh uv run --no-sync --frozen python -m unittest \
  tests.test_classical tests.test_anchors tests.test_annotation tests.test_latent
```

[Tests](#tests) has the whole suite and the browser tests.

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

The segmenter and the body-field network are library models (see
[Libraries](#libraries-setups-datasets-models)), trained and evaluated on
library datasets from the pose app's Training page or from the command line
([Training and evaluation](#training-and-evaluation-from-the-command-line)):

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/train.py \
  --setup lab:nir-flv --dataset lab:nir-labels --start-from lab:nir-hand284
scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate_model.py --model lab:nir-hand284
```

The hand labels of the old store
(`/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/segmentation_v1`)
and of the app corpus were migrated into the lab dataset `nir-labels`
(`scripts/migrate_to_library.py`). New labels are made on the pose app's
Labeling page. They go into your personal dataset for the setup.

Targeted labeling rounds are manifests: coils, self-contact, holes,
fragments and camera-edge frames across 13 recordings in
`docs/labeling_round_2/manifest.json` (built by
`scripts/build_labeling_manifest.py` from the clip-candidate scans), and 34
held-out self-contact frames in `docs/labeling_round_3_contact/manifest.json`.
`worm-pose-app --queue <manifest>` opens one as a Labeling queue; the
splits come from the dataset, per recording.

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

`--body-net <weights>` (a body-field model's `models/<id>/weights.ckpt` in a
library) adds the body-field network:
its A-P field and head/tail score every fit (independent, propagation and
track pass), its proposed trace is an extra start, and the evidence rather
than the taper decides the orientation. On the sequence set it cuts the
independent fits' frames below IoU 0.9 from 599 to 107, at about 230 ms
more per frame ([Body-field evidence in the pose fitter](docs/BODY_FIELDS.md#body-field-evidence-in-the-pose-fitter)).

Runtime improvements and reproducible GPU 3 benchmarks are documented in
[Runtime optimization](docs/RUNTIME_OPTIMIZATION.md). The default optimizations
preserve fitting schedules. `--compile-energy` additionally compiles the whole
independent fitting loss; it adds startup cost and should be benchmarked on
your workload before enabling.

To look at the poses of a recording, analyse it in the pose app (the
Workspace page; see [Pose app](#pose-app)). With `--dev` it shows every layer
and per-frame series the pipeline produces.

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
tube does not claim, such as a plate streak segmented as worm; the frame
classification tags such frames "mask has extra body (segmentation)". Every
candidate is stored (`hypotheses_*` arrays, `path_*`, `prediction_xy`), with
the path's choice and where it overrode the lowest energy.

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

The pose app analyses recordings, finds the stretches that need a person,
fixes them, labels frames and trains models. Its design is
[`docs/APP_SIMPLIFICATION.md`](docs/APP_SIMPLIFICATION.md). Start it with

```bash
scripts/project_env.sh uv run --no-sync --frozen python -m worm_pose_gen.app
# or, after `uv sync`, `worm-pose-app`; then open http://127.0.0.1:8768/
```

The app binds to localhost and is meant to be reached through an SSH tunnel;
nothing authenticates. Workspaces, Labeling queues and job records live under
`/temp_data4/alex/external_artifacts/workspaces` (`--workspaces-root`), and the
per-recording flat fields under `--dataset-root`
(`<dataset-root>/flat_fields`). `--gpus 1,2,3` restricts the GPUs jobs run on
(one job per GPU at a time, `--max-concurrent`), and `--device` sets the device
of the server's own models (Labeling's proposals, the Workspace's A-P layer;
default the first job GPU, else the CPU). `--dev` also shows the research
diagnostics described below. `--queue <manifest>` opens a labeling manifest
(`scripts/build_labeling_manifest.py`) as a Labeling queue and prints its
address.

### The three pages

The header switches between three pages. The addresses are
`#workspace`, `#labeling` and `#training`, so they can be bookmarked.

- **Workspace.** The Recordings screen lists the recordings of a setup, with
  their status (not analysed, analysing, *N* issues to review, reviewed,
  exported). **Add recording** registers a file from anywhere to a setup.
  **Analyse** runs the pipeline over the whole recording with the setup's
  default models as one job, and **Change** picks other models first.
  **Open** shows the recording's workspace:
  - on the left, the **Issues** list (stretches with plain reasons such as
    "head/tail uncertain", "coiled", "mask fits poorly" or "leaves the
    view"), **Looks OK** (O) and Prev/Next;
  - the four fixes on the selected issue: **Flip** (the whole issue, or
    this frame only), **Refit** (the algorithm comes from the issue's
    reasons, the anchors are automatic, and a before/after preview offers
    Keep or Discard), **Edit mask** (a brush on this frame; Save refits
    around it and keeps the result) and **Relabel** (sparse keyframes, one
    every *N* frames, labeled in a Labeling queue and then stitched);
  - below the fixes, the **Fixes** list, with Undo on each fix;
  - in the middle, the frame with four layers (keys 1–4): Mask, Midline
    (head a square, tail a circle), Outline, and the A-P field when the
    workspace's body model has one;
  - at the bottom, transport, the issue track and the curvature kymograph;
  - **Export** in the header, which writes one documented table per
    recording (`exports/<recording>_<UTC time>/`: a Parquet table with
    midline, curvature, width profile, head/tail, centroid, velocity and
    per-frame status, plus `export.json` with units and the models;
    see `worm_pose_gen/export_table.py`).
- **Labeling.** One frame's label at a time: the mask first (Worm and
  Background brush, Network and Threshold proposals with one slider, Fill
  holes, Largest, Grow, Shrink, Undo, Revert), then the body (Use proposal,
  Flip, Trace midline, Mask only), then **Save & next** (Enter). Saves go to
  your personal dataset for the setup, created on the first save and
  extending the lab dataset. Frames come from queues: a workspace's Relabel
  keyframes, **New queue** (a job that picks frames spread over the chosen
  recordings, favouring those the model is least sure of), and **Browse
  labels** (existing labels, lowest body fit IoU first). The body-field
  model's proposal is computed when a frame opens if the server has a GPU.
  Without one it takes 15–20 s on the CPU, so it is computed only on
  **Propose** (or Use proposal, G). The context strip shows frames t±16
  as Frames or Difference.
- **Training.** **Models** is the model picker. It has one row per model,
  Lab and Mine together, with ★ for the setup's defaults, the inputs and
  outputs, and the scores on the chosen benchmark (mask IoU mean and worst
  5%, head/tail correct, A-P error). **Not evaluated** rows are scored by a
  background job. The row actions are **Use as default** (with a reason;
  this replaces promotion), **Train from this** and **Details** (loss
  curves, worst benchmark frames, settings, the labels used). **Datasets**
  shows the labels by split and recording, warns when a dataset cannot be
  trained or evaluated honestly, and **Freeze benchmark** freezes its test
  labels. **Train** starts one job: prepare the body targets, train, keep
  the checkpoint with the lowest validation loss, evaluate on every
  benchmark of the setup, and write the model card.

With `--dev` the Workspace also shows the raw frame, every layer of the
frame payload (the segmenter's probability and the cleanup steps behind the
mask, the fitted and independent tubes, the network's crossings), the
**Details** drawer (classification, the ambiguity flags with their values and
thresholds, width and curvature along the body, mask statistics, the
workspace summary), one extra per-frame series on the timeline, a jobs
drawer, the Refit algorithm and parameter override, and the stage list and
range in the Analyse dialog. Only `--dev` runs the segmenter on a rested
frame, because analysts see the stored mask.

### Libraries: setups, datasets, models

Models, labels, setups and benchmarks live in two libraries with the same
layout (`worm_pose_gen.library`):

- the **lab library**, read-only to the app:
  `/storage/fs/store1/shared/worm-pose-models` on flv-c2, flv-c3 and flv-c4
  (`library.LAB_LIBRARY_BY_HOST`, `--lab-library`);
- your **personal library**, where everything the app makes goes:
  `/temp_data4/<user>/worm-pose-library` on the flv machines, else
  `~/worm-pose-library` (`--library`).

A **setup** is one microscope: its video dataset and flat-field setting, its
pixel size and frame rate, its recording roots, and the default model of
each role (`mask`, `body`). A personal override of a lab setup's defaults
is logged with who changed it and why. A **dataset** is a set of labels for
one setup. Its splits are assigned per recording and are append-only, and a
personal dataset can extend a lab one. A **label** revision stores the frame,
its context frames, the mask and the human body fields. A **benchmark** is a
frozen list of test-label revisions. A **model** is a card (`model.json`),
`weights.ckpt`, its training records and its evaluations. Items are named
`lab:<id>` or `mine:<id>`.

The first lab library is built from the old stores (`segmentation_v1`, the
app corpus and their body-field records) and two trained runs:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/migrate_to_library.py \
  --out /path/to/new-lab-library --body-run checkpoints/body_net/runs/<run> [--seed-cache <personal library>]
```

It writes setup `nir-flv`, dataset `nir-labels` (splits per recording),
benchmark `nir-v1`, and the model cards of the segmenter run
(`--segmenter-run`, `nir-hand284`) and the body-field net run. Publishing a
personal setup, dataset or model into the lab library is a developer step:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/publish.py model mine:copper-ft \
  --default body --reason "fixes heads on copper plates"
```

A published item is frozen: an id the lab already has is refused.

### Training and evaluation from the command line

The Training page's jobs run these commands, which also work by hand:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/train.py \
  --setup lab:nir-flv --dataset mine:nir-copper --start-from lab:nir-body-lags3 --max-epochs 50
scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate_model.py \
  --model mine:nir-copper-196 --benchmark lab:nir-v1
```

`scripts/train.py --help` lists the settings (`--context none|short` from
scratch, learning rate, batch size, patience, loss weights and more). A
model keeps the one checkpoint with the lowest validation loss, and the
evaluations go to `models/<id>/evaluations/`. The code is in
`worm_pose_gen.model_training` and `worm_pose_gen.model_eval`.

### Jobs on local GPUs or SLURM

At startup the app finds this node's GPUs and whether SLURM (`sbatch`,
`squeue`, `sacct`, `scancel`) is on the path (`GET /api/compute`). Analyse
and Train then offer one **Run on** choice: **This machine**, or **SLURM**
with partition and time prefilled from the host's defaults
(`compute.SLURM_DEFAULTS_BY_HOST`; on Engaging `ou_bcs_normal`, 12 h, one
GPU). A SLURM job writes a batch script that runs the same command and is
followed by its job id, so it survives an app restart. The app refuses
node-local paths (`/tmp`, `/scratch`, `/dev/shm`) for it. When neither
exists, analysis and training are disabled and the reason is shown. Job
records, progress and logs live under `<workspaces root>/jobs`.
`POST /api/jobs` with `kind: stage` runs one pipeline stage
(`python -m worm_pose_gen.pipeline --workspace ... --stage ...`), and
`kind: command` runs any command. Both take `run_on`, `slurm` and `gpu`.

Everything the pages do goes through the HTTP API (`/api/home`,
`/api/analyse`, `/api/workspaces/{name}/…` with `status`, `frame`,
`issues`, `fixes`, `mask`, `kymograph`, `network-fields` and `export`, then
`/api/labeling`, `/api/queues`, `/api/library`, `/api/training`, `/api/jobs`
and `/api/compute`), so a script can drive the same work. `/docs` is the
generated OpenAPI page.

### Tests

Python tests use `unittest`. List the modules explicitly, because
`unittest discover` does not work with this layout. Run them from the
repository root, on the CPU:

```bash
env PYTHONPATH=$PWD/src:$PWD CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg \
  .venv/bin/python -m unittest tests.test_app tests.test_fixes tests.test_queues
# the whole suite (about 20 minutes on one flv machine):
env PYTHONPATH=$PWD/src:$PWD CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg \
  .venv/bin/python -m unittest $(ls tests/test_*.py | sed 's|/|.|; s|\.py$||')
```

The browser tests drive the three pages with Playwright against synthetic
CPU apps that each test starts itself (`tests/browser/*_fixture.py`):
`tests/browser/workspace.cjs`, `tests/browser/labeling.cjs` and
`tests/browser/training.cjs`. Run them with `node` from the repository root.
Set `PLAYWRIGHT_MODULE` (a `playwright` or `playwright-core` module directory)
and `CHROMIUM_EXECUTABLE` when they are not installed in the default
locations, `LD_LIBRARY_PATH` when Chromium's libraries are not on the system,
and `PYTHON` when the interpreter is not `.venv/bin/python`. `WS_SHOTS`,
`SCREENSHOTS` and `SCREENSHOT_DIR` (Workspace, Labeling and Training) name a
directory for screenshots of the main states. The header of each test lists
what it covers.

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
  mask fitter, the pipeline stages, the segmenter and the body-field network,
  the library (`library/`), the pose app (`app/`, its browser UI in
  `app_ui/`), and supporting research modules.
- `scripts/` contains environment setup, the builders and evaluators of the
  geometric pipeline, the fitting and evaluation scripts, and the library's
  command-line steps (`train.py`, `evaluate_model.py`, `migrate_to_library.py`,
  `publish.py`).
- `docs/` contains the current algorithm narrative and its generated evidence.
- `tests/` contains focused geometry tests plus reusable-library coverage.
- `experiments/` retains machine-readable research outputs. The primary audit
  consumes the frozen selection manifest and baseline metrics stored there;
  historical narrative notes and embedded runners have been removed.
- `artifacts/` and `configs/` remain as data from the earlier research program;
  they are not part of the current documentation or script workflow.
