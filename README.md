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
scripts/project_env.sh uv run --no-sync --frozen python -m worm_pose_gen.label_app
```

Labels are stored under
`/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/segmentation_v1`
on flv-c4 with an 80/10/10 train/val/test assignment; checkpoints go to the
git-ignored `checkpoints/segmenter/` directory. The labeling app runs at
`http://127.0.0.1:8767`, proposes masks from the current checkpoint and the
classical pipeline, refines them with pipeline elements, and saves edited
labels back into the store. Details and keyboard shortcuts are in
[`docs/SEGMENTATION_LABELING.md`](docs/SEGMENTATION_LABELING.md).

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
`--no-propagate` skips it).

The sequence evaluation set, seven 300-frame clips with coils, self-contact,
fragments and camera exits, is the manifest `docs/sequence_eval_set.json`;
`scripts/find_sequence_clips.py` proposes such clips from a mask-only scan
and `scripts/evaluate_sequence_set.py` fits and scores the set. The plan
this belongs to, with measurements, is
[`docs/POSE_PIPELINE_PLAN.md`](docs/POSE_PIPELINE_PLAN.md);
`scripts/evaluate_width_model_unannotated30.py` and
`scripts/evaluate_recording_prior_unannotated30.py` compare width models and
priors on the 30-frame set.

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
