# Worm Pose Geometry

Research code for a conservative 2D *C. elegans* pose pipeline on NIR video.
The repository is now organized around one geometric algorithm:

`local-darkness mask -> skeleton pose -> smooth containing body -> narrow-notch repair -> curvature-aware endpoint extension`

The canonical description, evidence boundary, current results, and limitations
are in
[`docs/POSE_ESTIMATION_TO_ENDPOINT_CURVE_EXTENSION.md`](docs/POSE_ESTIMATION_TO_ENDPOINT_CURVE_EXTENSION.md).

This remains research code. The algorithm has been exercised on one annotated
development frame and an annotation-free 30-frame stress set; it is not
deployment-authorized, and the protected holdout remains unopened.

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

`scripts/evaluate_final_geometry_primary30.py` is the annotation-matched audit.
It requires the frozen selection manifest, baseline case list, annotation JSON,
and readable copies of the exact source frames. The integrated document records
why that audit is currently incomplete and why proxy substitution is invalid.

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
