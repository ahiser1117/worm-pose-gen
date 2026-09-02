# Worm Pose Geometry

Research code for a conservative 2D *C. elegans* pose pipeline on NIR video.
The repository is now organized around one geometric algorithm:

`local-darkness mask -> skeleton pose -> smooth containing body -> narrow-notch repair -> curvature-aware endpoint extension`

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

The two follow-on comparisons use the same three recordings and frame
positions:

```bash
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

- `src/worm_pose_gen/` contains reusable geometry, classical extraction, and
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
