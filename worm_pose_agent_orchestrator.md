# Autonomous Research Orchestrator: Probabilistic 2D *C. elegans* Pose Estimation

## 1. Mission

Build, evaluate, and deliver a high-throughput, high-accuracy system for estimating 2D *C. elegans* body pose from NIR behavior video believed to have been recorded at approximately 20 fps and 10× magnification. Treat frame rate, magnification, pixel scale, and orientation as supplied hypotheses until the HDF5 audit verifies them; retain 20 fps as the initial minimum throughput target unless the audit justifies a stricter target.

The primary scientific output is **body angle as a function of arc length and time**, not generic keypoints. The system must remain useful when part of the worm—especially the tail—leaves the field of view (FOV). It should distinguish image-supported pose from inferred pose and expose uncertainty when the image does not constrain the answer.

You are the **research orchestrator**. You may delegate narrow implementation, evaluation, debugging, visualization, and review tasks to subagents. You own the experimental plan, interfaces, integration, evidence standards, and final conclusions.

This file is the canonical research specification, but it becomes an active agent instruction only when the user explicitly asks the agent to follow it or when active project instructions explicitly reference it. At the start of a new run, confirm that this file is in scope and inspect any higher-priority environment or project instructions before acting. When a requirement is repeated later as a deliverable checklist, interpret the earlier, more specific definition as canonical rather than inventing a second meaning.

Your job is not merely to produce code. Your job is to autonomously:

1. inspect the supplied HDF5 recordings;
2. establish reliable baselines;
3. formulate and test explicit hypotheses;
4. create visual and quantitative evidence for every important decision;
5. reject ideas that do not justify their complexity;
6. converge on a robust final method;
7. package the result as a reusable PyTorch Lightning component suitable for a larger pipeline;
8. leave a clear research record showing what was tried, what worked, what failed, and why.

Do not optimize for novelty. Optimize for a defensible final system.

---

## 2. Non-negotiable constraints

### Runtime and environment

- The node has four CUDA-capable GPUs. Use **physical CUDA device 0 exclusively** for all training, inference, refinement, and GPU benchmarks. Set `CUDA_VISIBLE_DEVICES=0` before launching any Python process so that physical device 0 appears to the process as `cuda:0`; do not schedule work on devices 1–3. Verify the physical-to-logical mapping using the physical GPU UUID or PCI identity in addition to the visible logical index, record that identity in experiment metadata, and assert that PyTorch sees exactly one CUDA device. If device 0 is not visible, stop GPU-dependent work and report the environment problem rather than silently selecting another GPU or treating CPU results as representative.
- Use **Python 3.13** for the project. Pin the minor version with `.python-version` and constrain `requires-python` to `>=3.13,<3.14`; lock dependencies with `uv.lock` and record the exact interpreter patch version in environment and experiment metadata.
- All Python installation, environment management, dependency management, locking, and Python execution must go through **`uv`**. Use `uv python`, `uv sync`, `uv lock`, `uv add`/`uv remove`, and `uv run python ...`; do not invoke bare `python`, `python3`, `pip`, `venv`, Conda, Poetry, or another Python/package manager for project work.
- The final training/inference implementation must use **PyTorch Lightning**.
- Keep dependencies minimal.
- Do not modify the machine's GPU driver, CUDA installation, system Python, or unrelated system packages.
- Do not add experiment-tracking services or cloud dependencies.
- Prefer standard library + PyTorch + Lightning + NumPy + h5py + Matplotlib.
- Add another dependency only when it materially reduces implementation risk or provides a clearly useful baseline. Record the justification in `docs/DEPENDENCIES.md`.
- A dependency used only to generate training labels or research baselines should be a development/research dependency when possible, not a final runtime dependency.

### Permissions and environment preflight

- The canonical external artifact root is **`/temp_data4/alex/external_artifacts`**. On this node, `/temp_data4` may resolve to `/storage/fs/temp_data4`; record both the configured path and `realpath` result, but use the canonical path in user-facing commands and configuration. Do not substitute the sibling path `/temp_data4/alex/worm_pose`.
- Before bootstrap writes or dependency installation, resolve and test the actual permissions for the repository, `/temp_data4/alex/external_artifacts`, the repository-local `uv` cache/environment/Python-install directories, network access, and CUDA device 0.
- A symbolic link inside the repository does **not** extend sandbox permissions to its target. Resolve links with `realpath` or equivalent and verify that `/temp_data4/alex/external_artifacts` and each required child directory are writable. If the resolved target is outside the active writable sandbox, request approval or an explicit writable-root grant; do not redirect large outputs back into the repository to avoid the approval.
- GPU commands may likewise require approved execution even when the host has GPUs. Use the normal approval flow when CUDA is hidden by the sandbox; do not modify drivers, device files, or security settings.
- Dependency downloads may require network approval. Ask for the narrow approval needed by `uv`; do not use alternate package managers or unreviewed download scripts as a workaround.
- Before the first stateful `uv` command, configure writable locations. This project intentionally keeps its environment and disposable caches in ignored repository-local paths while reserving the external root for checkpoints, generated data, profiler traces, and accumulating experiment outputs. Persist the equivalent of the following in a checked-in, non-secret command wrapper and document it in `docs/ENVIRONMENT.md`:

  ```bash
  export PROJECT_ROOT=/absolute/path/to/worm-pose-gen
  export WORM_POSE_EXTERNAL_ROOT=/temp_data4/alex/external_artifacts
  export UV_PROJECT="$PROJECT_ROOT"
  export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
  export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"
  export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.uv-python"
  export UV_PYTHON=3.13
  export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
  export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
  export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
  export MPLBACKEND=Agg
  export CUDA_VISIBLE_DEVICES=0
  ```

  Resolve `PROJECT_ROOT` from the wrapper location rather than assuming the caller's working directory. Create and write-test those directories before use. Then run `uv python install 3.13` if a suitable 3.13 interpreter is not already available, `uv python pin 3.13`, and `uv sync --python 3.13`. Invoke every project `uv` command through `scripts/project_env.sh` so these exports apply across separate agent shell calls. Commit `.python-version`, `pyproject.toml`, and `uv.lock`.
- Configure Lightning checkpoints, profiler output, and local CSV/JSON logs beneath the external artifact root. Do not rely on Lightning's default output directory, and disable any logger that would add a service or undeclared dependency.

### Large-file storage policy

The working repository should remain lightweight apart from the explicitly approved, Git-ignored local `.venv`, `.uv-python`, and `.cache` directories. By default, treat any other generated non-source file of at least 10 MiB, or any other generated collection expected to exceed 100 MiB, as large. A documented exception may be made for a canonical final figure when keeping it in the repository materially improves reviewability.

- **Except for the local `.venv`, `.uv-python`, and disposable `.cache`, any large generated file must be stored somewhere under `/temp_data4/alex/external_artifacts`, not directly in the working directory or elsewhere under `/temp_data4/alex`.**
- At project bootstrap, create a project-specific storage root such as:

  ```text
  /temp_data4/alex/external_artifacts/
  ├── artifacts/
  ├── datasets/
  └── experiments/
  ```

  and place bulky outputs beneath it in organized subdirectories.
- This applies especially to:
  - copied or transformed video/HDF5 data;
  - extracted frame datasets;
  - pseudo-label datasets;
  - synthetic training datasets;
  - checkpoints;
  - optimizer/training-state snapshots;
  - cached embeddings/features;
  - rendered videos;
  - large image sequences;
  - large NumPy/PyTorch arrays;
  - temporary shards;
  - profiler traces;
  - large benchmark outputs;
  - any other artifact that would unnecessarily inflate the working directory.
- **Symbolic links from the repository into `/temp_data4/alex/external_artifacts/...` are allowed and encouraged** when they make paths convenient or preserve the expected repository layout.
- A repository symlink is a path convenience only. It works for reads or writes only when the resolved target is accessible under the active sandbox/approval policy.
- If using one project-root link, prefer `external_artifacts -> /temp_data4/alex/external_artifacts`. Code and metadata must still record the configured and resolved external roots rather than depending on the link's location.
- Small source files, configs, Markdown reports, CSV summaries, JSON metrics, and reasonably sized final figures should remain in the repository.
- Code must not rely on one developer's current working directory. Centralize external storage paths in configuration and provide sensible defaults.
- Record the resolved external artifact root in the experiment metadata so all outputs are traceable.
- Never delete or overwrite unrelated or protected content already present under `/temp_data4/alex/external_artifacts`. Create only documented project subdirectories and remove only files created and positively identified by this project.

### Data handling

- Never assume the HDF5 schema. Inspect it first.
- Never overwrite source HDF5 files.
- Treat supplied HDF5 symlinks as read-only source inputs. Do not copy the recordings merely to make their paths more convenient.
- Never silently rescale, crop, transpose, reinterpret, or normalize data without documenting the transformation.
- Preserve input frame coordinates so predictions can always be mapped back to the original image.
- Avoid loading an entire recording into RAM or GPU unless inspection proves it is small enough and doing so is advantageous.
- Implement chunked/streaming HDF5 access suitable for full recordings.
- Treat the source recordings as potentially high-latency network storage. Bound concurrent readers and total read bandwidth; do not assume all host CPUs or RAM are available to this project.
- In multiprocessing data loaders, open one read-only HDF5 handle per worker after worker creation. Never share a live h5py/HDF5 handle across a fork, and close handles deterministically.

### Scientific rigor

- Do **not** call pseudo-labels ground truth.
- Do **not** report synthetic accuracy as equivalent to real-image accuracy.
- Do **not** randomly split neighboring frames across train/validation/test. Temporal leakage is unacceptable.
- Use the highest leakage-safe grouping supported by the audit. Keep all recordings from the same animal, acquisition session, or effectively identical background/setup in one split. Within that constraint, prefer holding out entire recordings. If the number of independent groups is too small, use grouped cross-validation or, only as a last resort, large contiguous temporal blocks separated by documented guard intervals.
- Keep the final test set untouched until model-selection decisions are complete.
- Never use the same pseudo-label generator as both the main training target and the sole arbiter of model quality. Tier-B proxy-label metrics can test engineering consistency, but claims that one method is scientifically more accurate require independent evidence such as manual labels, controlled synthetic truth, a genuinely independent method, or clearly qualified qualitative review.
- Every major claim in the final report must point to a metric, figure, or controlled experiment.
- Always examine random examples and worst-case examples in addition to aggregate metrics.

### Engineering rigor

- The repository must remain runnable after every accepted experiment.
- Use tests for geometry, angle conventions, coordinate transforms, HDF5 I/O, visibility handling, and rendering/refinement gradients.
- Prefer simple interfaces and plain configuration files over framework-heavy abstractions.
- Do not add a component because it sounds sophisticated. Add it only if an experiment shows that it improves a relevant metric enough to justify its cost.

---

## 3. Definition of the final product

The final result should be a reusable package with a PyTorch Lightning training module and an efficient inference path.

Use one explicit geometry convention everywhere. Pixel coordinate `(0, 0)` is the center of the upper-left pixel; `x` increases right and `y` increases down. A centerline point is geometrically in the FOV exactly when `0 <= x < image_width` and `0 <= y < image_height`. Export points in the model's selected head-to-tail order, with body index 0 chosen as the most probable head. Tangents follow that exported direction.

For every frame, the preferred output schema is:

- `centerline_xy`: shape `[N_body, 2]` in original image pixel coordinates;
- `tangent_angle`: shape `[N_body]` in radians, using image coordinates (`x` right, `y` down), `atan2(dy, dx)`, wrapped to `[-pi, pi)`;
- `curvature`: shape `[N_body]`, defined as `d(tangent_angle)/d(arc_length)` in radians per original-image pixel; with `y` increasing down, positive curvature turns clockwise in the displayed image. Evaluate it at the same uniformly spaced body positions and document endpoint handling;
- `in_fov_mask`: boolean shape `[N_body]`, deterministically computed from `centerline_xy` using the half-open image bounds above; this describes centerline-point membership, not whether the full worm tube is inside the image;
- `image_support_probability`: shape `[N_body]`, estimating whether usable image evidence supports each anatomical location; this is distinct from geometric FOV membership and from pose uncertainty;
- `angle_uncertainty`: shape `[N_body]`, representing calibrated marginal circular angular uncertainty in radians; document the distribution or interval construction and do not imply that these marginals capture joint hidden-body uncertainty;
- `head_tail_probability`: calibrated scalar probability that exported body index 0 is truly the head. Because export order is canonicalized to the more probable orientation, this value should ordinarily be in `[0.5, 1]`; preserve weighted orientation hypotheses when ambiguity must be represented downstream;
- when calibrated hidden-body position or joint shape uncertainty is claimed, either `centerline_covariance` with shape `[N_body, 2, 2]` plus documented correlation limitations, or a compact set of weighted latent/centerline hypotheses. Marginal angle uncertainty alone is insufficient evidence for calibrated hidden-position uncertainty;
- `quality_score`: scalar per frame with a documented construction and interpretation; do not call it a probability unless it is calibrated as one;
- optional latent spline coefficients used internally;
- timestamps/frame indices and sufficient metadata to map results back to the source recording.

Use a default of approximately **100 uniformly spaced body positions** for exported pose unless experiments strongly support another value.

The final inference API should support:

1. one frame;
2. a short temporal window;
3. a streamed full HDF5 recording;
4. batched GPU inference;
5. saving results to a new HDF5 file without modifying the source.

### Output HDF5 contract

Define a versioned output schema before training the final model. At minimum:

- store predictions beneath a documented group such as `/worm_pose` with a `schema_version` attribute;
- specify every dataset path, dtype, shape, units, missing-value convention, chunking, and compression;
- chunk frame-major datasets for streamed append and efficient temporal reads;
- store the configured and resolved source path, source dataset path, source size/mtime identity, frame-index/timestamp mapping, checkpoint digest, config digest, Git commit, package versions, and geometry convention;
- write to a same-filesystem partial output, mark completion explicitly, flush safely, and rename atomically only after validation; either support documented frame-level resume or clearly reject incomplete outputs;
- validate frame count, monotonic indices/timestamps where applicable, finite values, bounds semantics, and metadata before declaring inference complete.

The final package should not require notebooks to run.

---

## 4. Repository layout

Create and maintain approximately this structure:

```text
.
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── baseline.yaml
│   ├── final.yaml
│   └── resource_budget.yaml
├── src/
│   └── worm_pose_gen/
│       ├── __init__.py
│       ├── data.py
│       ├── geometry.py
│       ├── model.py
│       ├── losses.py
│       ├── renderer.py
│       ├── refinement.py
│       ├── inference.py
│       ├── metrics.py
│       ├── io.py
│       └── visualization.py
├── scripts/
│   ├── inspect_hdf5.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer_hdf5.py
│   └── benchmark.py
├── tests/
├── experiments/
│   ├── index.csv
│   └── exp_XXXX_name/
│       ├── config.json
│       ├── metrics.json
│       ├── notes.md
│       ├── stdout.log
│       └── figures/
├── artifacts/
│   ├── checkpoints/
│   ├── example_predictions/
│   └── final_figures/
└── docs/
    ├── ENVIRONMENT.md
    ├── DATA_AUDIT.md
    ├── EVALUATION.md
    ├── DESIGN.md
    ├── EXPERIMENT_LOG.md
    ├── EXPERIMENT_FLOW.md
    ├── DECISIONS.md
    ├── DEPENDENCIES.md
    ├── FAILURE_MODES.md
    ├── ANNOTATION_RECOMMENDATIONS.md
    └── FINAL_REPORT.md
```

Preserve the existing import package name `worm_pose_gen` unless a migration is explicitly justified. You may otherwise modify this layout when there is a clear engineering reason, but preserve the same conceptual separation.

Directories such as `artifacts/checkpoints/` may be symbolic links into the corresponding directory under `/temp_data4/alex/external_artifacts`. Do not put large binary artifacts directly in the repository merely to preserve the example layout.

---

## 5. Orchestrator behavior

### 5.1 Autonomous loop

For each modeling idea, follow this loop:

1. **State a hypothesis.**
   - Example: “Adding 11-frame temporal context will reduce tangent-angle error near the cropped tail without materially degrading throughput.”

2. **State the cheapest decisive experiment.**
   - Define the baseline, changed variable, dataset split, metrics, maximum steps/epochs, wall-time and storage budgets, seeds or repeat policy, and a numeric success/failure criterion before implementation.
   - If no scientifically established effect-size threshold exists, choose and justify a practical threshold relative to the baseline before seeing the result. Include uncertainty or variability across seeds/samples when it could change the decision.

3. **Delegate implementation when useful.**
   - Give a subagent one narrow task with explicit inputs, expected files changed, tests, and deliverables.

4. **Run a smoke test.**
   - Catch shape errors, NaNs, data leakage, coordinate errors, and performance pathologies before full training.

5. **Run the controlled experiment.**
   - Make nontrivial training runs resumable. Save bounded-cadence checkpoints and sufficient state/configuration to resume after interruption without treating a restarted run as independent evidence.

6. **Generate figures automatically.**
   - Scalar metrics alone are insufficient.
   - Produce at least one figure that directly answers the experiment's hypothesis whenever the hypothesis is visually testable.
   - Prefer plots, overlays, sequence panels, and error distributions that make the comparison understandable without reading raw metric files.

7. **Inspect results.**
   - Include random samples, best cases, worst cases, boundary/cropped cases, and temporal sequences.

8. **Record a decision.**
   - `ACCEPT`, `REJECT`, `REVISE`, or `INCONCLUSIVE`.
   - Explain why using evidence.

9. **Update the next experiment.**
   - Do not blindly follow the original plan if the data contradict it.

### 5.2 Complexity discipline

Do not perform large hyperparameter sweeps early.

Use this order:

1. verify data and metrics;
2. establish a simple baseline;
3. test architecture families at small scale;
4. test the most important structural hypotheses;
5. only then tune the leading system.

Use one-factor comparisons, small factorial experiments, or successive halving. Avoid dozens of near-identical runs.

If a new component fails its predeclared practical-effect threshold while adding substantial runtime, dependencies, or failure modes, prefer the simpler system. Record the threshold and tradeoff rather than describing the improvement afterward with undefined terms such as “small” or “meaningful.”

### 5.3 Accuracy–throughput Pareto principle

Maintain a table and plot of model variants versus:

- visible-body angle error;
- visible-body centerline error;
- cropped-body robustness;
- uncertainty quality;
- frames/second;
- GPU memory;
- parameter count;
- preprocessing/refinement cost.

Do not optimize accuracy in isolation. Keep the Pareto frontier visible throughout the project.

At minimum, final end-to-end offline inference must exceed the acquisition rate of **20 fps** on CUDA device 0 at a documented batch size. Also report batch-1 latency and throughput separately. Do not describe an acausal or heavily batched pipeline as real-time solely because its aggregate throughput exceeds 20 fps.

---

## 6. Subagent roles

### Agent runtime and delegation authority

- The primary orchestrator must run **`gpt-5.6-sol` with `xhigh` reasoning effort**.
- Every subagent must run **`gpt-5.6-sol` with `medium` reasoning effort** unless the user explicitly changes this specification.
- Set the subagent model and reasoning effort explicitly at spawn time. Because the subagent reasoning effort differs from the orchestrator's, provide the bounded task context and referenced files needed for the assignment rather than relying on an implicit full-history/model inheritance behavior.
- The orchestrator has standing authority to create, direct, interrupt, reassign, and retire subagents as it sees fit for in-scope project work. It does not need separate user confirmation for each delegation, provided the assignment follows this document's file-ownership, evidence, resource, permission, and safety constraints.
- Delegation does not expand the user's requested scope or authorize otherwise restricted external or destructive actions. The orchestrator remains accountable for every delegated result and must inspect it before relying on it.
- A subagent may create a nested subagent only when the orchestrator explicitly authorizes nested delegation in that assignment. Nested agents use the same `gpt-5.6-sol`/`medium` specification and count against the same active-slot limit.
- If the required model or reasoning effort is unavailable, do not silently substitute another configuration. Report the mismatch and continue without that delegation when useful, or request direction if the missing agent configuration blocks the work.

Create subagents dynamically when useful. Do not keep them alive without a concrete task.

This environment supports at most four active agent slots including the orchestrator, so use at most three concurrent subagents. Subagents share the same working tree: their edits become visible immediately and are not isolated branches waiting to be merged. Assign disjoint file ownership, inspect every returned change, and treat the orchestrator's role as integration and review rather than a later filesystem merge.

Parallelize read-only inspection, CPU analysis, documentation, and clearly disjoint implementation when useful. Serialize training, inference benchmarks, and other substantial GPU work on CUDA device 0 unless a deliberate same-device concurrency test is itself the experiment. Never let independent subagents start competing GPU jobs accidentally.

Useful roles include:

### Data audit subagent

Responsibilities:

- inspect HDF5 groups/datasets, shapes, dtypes, chunking, compression, timestamps, metadata;
- identify the image dataset(s);
- sample frames throughout each recording;
- quantify image size, intensity range, frame rate if available, dropped/duplicate frames if detectable;
- estimate how often the worm touches or exits the FOV;
- identify obvious acquisition changes across recordings;
- write `docs/DATA_AUDIT.md` and inspection figures.

### Baseline subagent

Responsibilities:

- implement the simplest reasonable segmentation/centerline baseline;
- determine where classical image processing succeeds and fails;
- produce high-confidence pseudo-label candidates for easy full-body frames;
- quantify confidence criteria;
- never label uncertain frames as truth.

### Geometry/model subagent

Responsibilities:

- implement intrinsic angle/spline representations;
- define centerline reconstruction;
- validate gradients and angle wrapping;
- implement the Lightning proposal model;
- keep model size modest unless evidence supports scaling.

### Renderer/refinement subagent

Responsibilities:

- implement differentiable image-to-pose refinement;
- make FOV censoring explicit;
- benchmark full-image rendering versus cheaper local/cross-sectional likelihoods;
- ensure refinement is vectorized and GPU-friendly.

### Evaluation subagent

Responsibilities:

- implement leakage-safe splits;
- build synthetic-crop benchmarks from fully visible real frames;
- produce accuracy, calibration, failure, and throughput plots;
- audit whether metrics can be gamed by oversmoothing.

### Integration/reviewer subagent

Responsibilities:

- review the current candidate as if trying to falsify its claims;
- inspect interfaces, tests, dependencies, performance, and failure cases;
- identify missing ablations or unjustified conclusions;
- avoid large rewrites unless a concrete defect requires them.

### Subagent contract

Every delegated task should specify:

- model `gpt-5.6-sol` and reasoning effort `medium`;
- objective;
- hypothesis or issue being addressed;
- files it may edit;
- files/interfaces it must not break;
- command(s) that should pass;
- expected figures/metrics;
- return format: summary, changed files, tests run, unresolved risks.

Prefer parallel subagents only for independent work that respects the shared-worktree and single-GPU constraints. Do not allow multiple subagents to edit the same central interfaces concurrently.

The orchestrator owns all architectural decisions and integration.

---

## 7. Phase 0 — Bootstrap and reproducibility

Before scientific work:

1. inspect the repository and supplied files;
2. resolve `/temp_data4/alex/external_artifacts`, verify the resolved target and required child directories are writable under the active permissions, request approval if required, and record both configured and resolved paths;
3. create repository symlinks to approved external output directories where useful, then verify a small write/read round trip through each link before launching expensive work;
4. use `scripts/project_env.sh` to export `WORM_POSE_EXTERNAL_ROOT`, repository-local `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`, `UV_PYTHON_INSTALL_DIR`, `UV_PYTHON=3.13`, writable library cache paths, and `CUDA_VISIBLE_DEVICES=0` as defined above, then create/write-test the corresponding directories;
5. initialize `uv` project metadata if absent, install or select Python 3.13 with `uv`, pin `.python-version` to 3.13, constrain `requires-python` to `>=3.13,<3.14`, and verify that `scripts/project_env.sh uv run python` reports Python 3.13 from the repository-local `.venv`;
6. establish the minimum dependency set with `uv`, generate `uv.lock`, and verify `uv sync --frozen --python 3.13` from a clean environment;
7. establish source-control provenance before experiment IDs begin: record the initial status, ensure large/generated files are excluded, and create a baseline commit containing the intended source/configuration files. If commit creation is not authorized, record a patch/tree digest and explicitly mark exact Git-based reproducibility as pending;
8. verify physical CUDA device 0 through approved host inspection and `uv run python`/PyTorch: record physical index, UUID or PCI identity, logical index, and assert `torch.cuda.device_count() == 1`;
9. record:
   - exact Python 3.13 patch version and interpreter path;
   - PyTorch version;
   - Lightning version;
   - CUDA availability;
   - GPU physical index, logical index, UUID/PCI identity, and name;
   - GPU memory;
   - relevant HDF5 library version;
   - configured and resolved paths for source data, artifact root, virtual environment, `uv` cache, and uv-managed Python installation;
   - input filesystem/mount type and whether it is local, networked, and read-only;
   - active sandbox/approval limitations relevant to reproducibility;
10. create a deterministic seed policy covering Python/NumPy/PyTorch, data-loader workers, split generation, augmentation, and repeat policy; document any nondeterministic CUDA operations retained for performance;
11. create the experiment directory structure, using symlinks into `/temp_data4/alex/external_artifacts/...` for large artifact locations where appropriate;
12. create a tiny end-to-end smoke command, executed with `uv run python`, that loads frames and runs a dummy Lightning forward pass;
13. measure one small pilot epoch or fixed-step run, then write `configs/resource_budget.yaml` covering per-run wall time, maximum steps/epochs, aggregate GPU time, checkpoint cadence, external-storage budget, maximum CPU/data-loader workers, maximum host and pinned memory, maximum concurrent HDF5 readers, and source-filesystem I/O budget before any full run.

Do not spend significant time on infrastructure beyond what is required for reliable experiments.

**Deliverable:** `docs/ENVIRONMENT.md` or equivalent reproducibility section in `README.md`.

---

## 8. Phase 1 — Data forensics

Do not write the pose model before understanding the recordings.

### Required HDF5 inspection

The current repository inventory contains 12 source HDF5 symlinks totaling approximately 155.66 GiB. Recompute and record this inventory at bootstrap because the supplied files may change.

For every input file, inspect metadata and a bounded, uniformly distributed frame sample and record:

- top-level and nested keys;
- image dataset path;
- shape and axis interpretation;
- dtype;
- chunks/compression;
- number of frames;
- image dimensions;
- nominal/observed frame rate;
- timestamps if present;
- useful metadata;
- whether frames are grayscale or multi-channel;
- whether image orientation differs across files.

Record the total number and byte size of the supplied recordings at bootstrap. A full sequential scan of every frame is not implied by “inspect every file”; perform one only for a question that requires it, with streaming I/O, a stated runtime budget, and a reusable cached summary under the external artifact root.

### Required visualizations

Create:

1. a montage of frames sampled uniformly through each recording;
2. intensity histograms/percentiles by recording;
3. a short sequence showing typical locomotion;
4. examples of:
   - fully visible worm;
   - head near boundary;
   - tail near boundary;
   - body partially outside FOV;
   - tight bend/turn;
   - motion blur if present;
   - low-contrast or unusual frames;
5. a rough estimate of worm size in pixels and body width;
6. if feasible, a plot of frame-to-frame image change to identify discontinuities.

### Questions to answer

- Is there normally one worm or can multiple worms appear?
- Does the FOV move?
- Does image scale change?
- Is head/tail visually distinguishable from static appearance?
- How frequent are truncated frames?
- How often does self-overlap occur?
- Is illumination stable enough for simple intensity normalization?
- Are there recording-specific backgrounds that could create leakage?

**Deliverables:** `docs/DATA_AUDIT.md` and `artifacts/final_figures/data_audit_*`.

Do not proceed until the dataset split strategy is written down.

---

## 9. Phase 2 — Evaluation protocol before model development

Accuracy cannot be improved if it is not measurable.

### 9.1 Split strategy

Preferred order:

1. hold out entire animals/acquisition sessions/background groups, including all linked recordings from each group;
2. grouped recording-level cross-validation if only a few independent groups exist;
3. only if unavoidable, use large non-overlapping temporal blocks separated by guard intervals.

Never place adjacent frames from the same temporal neighborhood in different splits. Write the grouping keys, guard-interval length, and rationale into the split manifest so that the split is reproducible and auditable.

### 9.2 Three evidence tiers

Keep these separate in reports:

#### Tier A — true manual ground truth

Use if supplied. This is the strongest evaluation source.

#### Tier B — high-confidence real-image pseudo-labels

Generate only on easy, fully visible frames where classical processing and/or multiple independent methods agree. Treat these as **proxy labels**, not truth.

Report whether each evaluation method shares code, assumptions, preprocessing, or labels with the pseudo-label generator. Do not use correlated agreement as if it were independent validation.

#### Tier C — controlled synthetic truth

Use known geometric warps, synthetic worms, and especially synthetic FOV truncation where the pre-crop pose is known. This is excellent for controlled ablations but does not replace real-image evaluation.

### 9.3 Cropped-tail benchmark

This is a required benchmark.

From high-confidence fully visible real frames or sequences:

1. establish a reference full-body centerline;
2. create artificial camera windows or support masks that hide controlled fractions of the head or tail while keeping a fixed, documented output coordinate frame;
3. preserve the real worm texture and background wherever visible, provide the loss/evaluator with the true observable-support mask, and avoid synthetic boundary cues that do not occur in production images;
4. evaluate pose error separately on:
   - visible body;
   - hidden body;
   - a boundary band near the FOV edge;
5. test multiple hidden fractions, for example approximately 5%, 10%, 20%, 30%, and 40% of body length;
6. repeat with temporally coherent moving crop boundaries on sequences;
7. save the crop transform/support mask so reference and predicted coordinates can be mapped exactly between cropped and original frames.

This benchmark provides an objective way to test whether temporal inference and visibility modeling actually solve the problem they claim to solve.

### 9.4 Core metrics

Implement at least:

- centerline point error in pixels;
- centerline point error normalized by estimated body width and/or body length;
- tangent-angle MAE in degrees with proper circular wrapping;
- error versus normalized body coordinate `s`;
- endpoint error;
- body length consistency;
- exact `in_fov_mask` agreement with the documented coordinate convention;
- `image_support_probability` classification/calibration against independently defined support targets;
- head/tail flip rate;
- calibration and discrimination of `head_tail_probability`, including ambiguous cases rather than only hard-orientation accuracy;
- temporal angle error where reference sequences exist;
- prediction jitter/jerk as a secondary metric;
- runtime frames/second;
- GPU memory;
- batch-size scaling.

For hidden-body predictions, report them separately from visible-body accuracy. Define visible fraction as the fraction of the reference anatomical centerline inside observable image support; do not derive the evaluation stratum solely from the model's own visibility prediction.

### 9.5 Uncertainty metrics

If a model emits uncertainty, measure whether uncertainty means anything:

- error versus predicted uncertainty;
- empirical coverage of nominal confidence intervals;
- calibration curves;
- calibration stratified by visible fraction;
- calibration near the FOV boundary;
- calibration on tight bends/turns.

A model must not receive credit for “probabilistic” output unless uncertainty correlates with actual error.

**Deliverable:** `docs/EVALUATION.md` or equivalent section in `docs/DESIGN.md`.

---

## 10. Phase 3 — Classical and simple learned baselines

The purpose of baselines is to determine how hard the dataset actually is and to create training/evaluation scaffolding.

### Baseline A: classical segmentation + centerline

Attempt a minimal method appropriate to NIR grayscale data:

- robust intensity normalization;
- foreground/background separation;
- largest plausible connected component if appropriate;
- centerline extraction/skeletonization;
- spline fitting and arc-length resampling;
- temporal head/tail continuity.

If this requires an image-processing dependency, isolate it as a research dependency and document why it was added.

Record exactly which frames are considered high-confidence. Useful criteria may include:

- one connected worm-like component;
- plausible area;
- plausible length/width;
- two endpoints;
- no boundary contact;
- smooth spline fit;
- agreement with neighboring frames.

### Baseline B: direct learned pose

Train a small Lightning model on high-confidence real pseudo-labels plus synthetic perturbations.

Test at least one simple representation such as:

- direct fixed-length centerline coordinates;
- heatmaps followed by centerline extraction;
- intrinsic angle/spline coefficients.

The goal is not exhaustive search. The goal is to identify a stable learned proposal network.

### Required baseline figures

- overlay centerlines on random held-out frames;
- overlay worst-error frames;
- angle-versus-body-coordinate examples;
- error versus body coordinate;
- error versus FOV proximity;
- failure montage;
- throughput benchmark.

**Decision gate:** Do not proceed to sophisticated probabilistic machinery until a neural proposal model is reliable enough to initialize refinement on ordinary fully visible frames.

---

## 11. Preferred pose representation

Start from an **intrinsic body representation** because the scientific target is angle along the body.

A preferred candidate is:

```text
latent pose z_t =
    anchor position (x, y)
    + global orientation phi
    + body length L
    + K spline coefficients describing tangent angle along arc length
    + optional width scale
```

Represent tangent angle approximately as

```text
theta(s) = phi + sum_j c_j B_j(s)
```

and reconstruct the centerline by integrating unit tangent vectors along normalized arc length.

### Experiments

Test a modest number of spline coefficients first, for example roughly 12–24. Do not assume more coefficients improve scientific accuracy.

Compare:

- coordinate-space prediction;
- angle/spline prediction;
- optionally a compact learned basis or PCA/eigenworm-style basis.

Evaluate local angle fidelity, endpoint stability, and throughput.

### Geometry tests

Unit tests must verify:

- straight worm reconstruction;
- constant curvature / circular arc;
- known synthetic sinusoidal angle profile;
- translation/rotation equivariance where expected;
- angle wrapping at `-pi/pi`;
- arc-length resampling;
- gradient propagation through reconstruction.

---

## 12. Phase 4 — Temporal proposal model

Test whether temporal context materially improves pose and head/tail stability.

The primary research mode is offline inference over recorded HDF5 data, so centered/acausal temporal windows are allowed and should be the default comparison when they improve accuracy. Record the number of future frames used, resulting look-ahead latency, padding at sequence boundaries, and behavior at discontinuities. If online use is claimed or required, implement and evaluate a separate causal mode that uses no future frames; never transfer an offline/acausal latency or accuracy claim to the causal mode without measurement.

Start small. A sensible candidate is:

1. lightweight 2D frame encoder;
2. short temporal fusion over approximately 5–11 frames;
3. heads for pose, image-support probability, head/tail confidence, and optional uncertainty; compute geometric `in_fov_mask` from reconstructed coordinates rather than learning it as an unconstrained head.

Prefer temporal convolutions, pooling, or a small attention block over a large video transformer unless the simpler methods fail.

### Required hypothesis

> Short temporal context improves body-angle accuracy and boundary robustness, especially when the tail is partially outside the FOV, without sacrificing acceptable throughput.

### Controlled ablation

Compare at minimum:

- 1 frame;
- short context, e.g. 5 frames;
- longer context, e.g. 11 frames.

Keep the encoder and training budget as similar as practical.

Evaluate:

- full-body frames;
- synthetic crop benchmark;
- real boundary-touching frames;
- head/tail flip rate;
- throughput.

Reject longer context if it fails the numeric practical-effect threshold declared before the ablation.

---

## 13. Phase 5 — Differentiable image-space refinement

Only begin this phase once the proposal model reliably lands near the correct pose.

The goal is to refine the proposal using the actual NIR image, analogous to probabilistic inverse graphics but specialized to a 2D worm.

### 13.1 Initial renderer

Implement a differentiable renderer or image likelihood for a worm-shaped tube around the predicted centerline.

Possible nuisance parameters:

- width scale/profile;
- foreground intensity;
- background intensity;
- local contrast;
- blur;
- small illumination gradient.

Keep nuisance modeling simple initially.

### 13.2 FOV censoring is mandatory

If a predicted body segment lies outside the image, do not penalize it for failing to explain pixels that do not exist.

The image loss must be defined only over observable image support. The model must be able to represent a worm whose latent centerline continues outside the FOV.

Track both geometric `in_fov_mask` and learned `image_support_probability` explicitly. Use the FOV/support mask—not the learned probability—to censor image losses. A point can be geometrically in frame yet weakly supported because of self-overlap, blur, low contrast, or occlusion.

### 13.3 Compare likelihood implementations

Full soft rasterization may be unnecessarily expensive. Test at least two strategies if feasible:

#### Full-image soft rendering

Render a soft worm mask/intensity image and compare against a local ROI.

#### Local cross-sectional likelihood

Sample image intensities around predicted centerline normals or a narrow tube around the centerline. This may preserve subpixel alignment information at much lower computational cost.

The second approach may be preferable for throughput. Let measurements decide.

### 13.4 Losses

Start with robust image terms such as:

- pixel/intensity residual;
- edge/gradient agreement;
- mask/tube overlap where useful.

Only add a learned neural image likelihood if simpler image-space refinement fails because appearance mismatch dominates.

### 13.5 Optimization

Start with a small fixed number of refinement steps, e.g. 1–5.

Optimize only a compact pose latent rather than arbitrary body coordinates.

Vectorize across frames/hypotheses. Benchmark eager and compiled modes if `torch.compile` is stable in the local environment, but do not make compilation a hard dependency.

### Required experiment

Compare:

```text
proposal only
vs.
proposal + 1 step refinement
vs.
proposal + 3 steps
vs.
proposal + 5 steps
```

Report marginal accuracy gain per added millisecond.

**Decision gate:** retain refinement only if it provides clear visible-body accuracy or boundary robustness gains at acceptable cost.

---

## 14. Phase 6 — Probabilistic/uncertainty-aware temporal inference

Do not implement expensive general-purpose MCMC unless simpler approaches fail.

The preferred hierarchy is:

1. amortized neural proposal;
2. deterministic or MAP refinement;
3. uncertainty estimate;
4. multiple hypotheses only when the frame is genuinely ambiguous.

### 14.1 Temporal prior

Begin with a simple latent dynamics prior over spline coefficients and global motion, such as:

- constant velocity for centroid/orientation;
- first- or second-order autoregression for pose coefficients.

Fit or estimate its scale from training sequences.

Only replace it with a learned dynamics model if experiments demonstrate a meaningful improvement.

### 14.2 Ambiguity-aware inference

Potential triggers for additional computation:

- high proposal uncertainty;
- low image-likelihood improvement;
- boundary truncation;
- head/tail ambiguity;
- tight coiling/self-overlap;
- disagreement between proposal and temporal prior.

For such frames, test a small bank of parallel hypotheses/particles rather than full MCMC.

Examples:

- head/tail-flipped hypotheses;
- 4–16 perturbations of spline coefficients;
- multiple candidate hidden-tail continuations.

Propagate promising hypotheses for a few frames when future evidence can disambiguate them.

### 14.3 Adaptive compute hypothesis

> Spending refinement/particle computation only on uncertain frames can approach the accuracy of always-on expensive inference while preserving high average throughput.

Benchmark:

- proposal only;
- fixed refinement on every frame;
- adaptive refinement;
- adaptive particles on the hardest frames.

Report both average and worst-case latency.

---

## 15. Training-data strategy

Because dense manual centerline labels may be unavailable, use multiple data sources while keeping their roles explicit.

### 15.1 High-confidence real pseudo-labels

Use classical processing or agreement among methods to identify easy, fully visible frames. Prefer quality over quantity.

### 15.2 Synthetic geometric augmentation

From real worms with a known/reference centerline, generate training examples by:

- translation;
- rotation;
- moderate scale changes;
- realistic bending/warping in intrinsic angle space;
- intensity/contrast variation;
- blur;
- noise;
- background perturbation;
- motion-consistent sequence augmentation.

### 15.3 Synthetic FOV truncation

Oversample examples where:

- 5–40% of the tail is outside the FOV;
- 5–40% of the head is outside the FOV;
- the body exits at different boundary orientations;
- the truncation changes smoothly over time.

Derive exact `in_fov_mask` targets from the known crop geometry. Train `image_support_probability` only from separately defined support targets; do not relabel geometric membership as generic “visibility.”

### 15.4 Self-training loop

If differentiable refinement reliably improves proposals:

```text
proposal -> refinement -> quality filter -> improved pseudo-label -> retrain proposal
```

Run this loop only if held-out evaluation improves. Do not recursively self-train without a quality gate.

---

## 16. High-priority hypothesis backlog

Test these roughly in order, but adapt based on evidence.

### H1 — Intrinsic angle/spline representation

**Claim:** Predicting intrinsic body shape reduces local jitter and improves angle accuracy relative to independent coordinates.

**Measure:** angle MAE, centerline error, temporal stability, parameter count, throughput.

### H2 — Temporal context

**Claim:** 5–11 frame context improves cropped-tail and head/tail robustness.

**Measure:** synthetic crop benchmark, boundary cases, flip rate, fps.

### H3 — Image-space refinement

**Claim:** a few differentiable refinement steps recover subpixel/local-angle accuracy missed by the proposal network.

**Measure:** visible-body error and angle error before/after refinement, milliseconds per frame.

### H4 — Explicit image support and FOV censoring

**Claim:** censoring off-screen body segments prevents boundary-induced pose distortion.

**Measure:** error as hidden fraction increases; error in the last visible 10–20% of the body.

### H5 — Temporal prior for hidden body

**Claim:** a compact dynamics prior gives better hidden-tail estimates than simply extrapolating the final visible tangent.

**Measure:** hidden-body error in synthetically cropped sequences.

### H6 — Adaptive computation

**Claim:** uncertainty-triggered refinement/particles retains most expensive-model accuracy at much higher average throughput.

**Measure:** accuracy–fps Pareto curve and worst-case latency.

### H7 — Calibrated uncertainty

**Claim:** predicted uncertainty identifies frames/body regions with genuinely higher pose error.

**Measure:** coverage, reliability curves, error-vs-uncertainty correlation.

### H8 — Learned image likelihood

Test only if needed.

**Claim:** a small learned feature-space likelihood is more robust than raw pixel/edge losses to real NIR appearance variation.

**Measure:** real-image refinement reliability versus added training/runtime complexity.

---

## 17. Failure modes to actively search for

Maintain `docs/FAILURE_MODES.md` from the beginning.

At minimum inspect:

- head/tail swaps;
- tail disappearing from FOV;
- head disappearing from FOV;
- partial body truncation interpreted as a shorter worm;
- self-overlap;
- omega/tight turns;
- endpoint collapse;
- spline oversmoothing;
- high-frequency angle jitter;
- motion blur;
- low contrast;
- frame-boundary artifacts;
- model dependence on recording-specific background;
- implausible body length changes;
- uncertainty that remains low on obvious failures;
- temporal smoothing that erases genuine fast posture changes;
- refinement drifting into a wrong local optimum;
- throughput collapse at large batch size or long sequences.

For each accepted final model, create a failure montage with representative examples.

---

## 18. Visualization requirements

Every major experiment must produce figures sufficient for a human to audit the result without opening tensors.

Required recurring visualizations:

1. raw image + predicted centerline overlay;
2. color-coded geometric FOV membership and image-support probability along the centerline;
3. uncertainty displayed along the centerline where available;
4. tangent angle versus normalized body coordinate;
5. angle heatmap over time for representative sequences;
6. error versus normalized body coordinate;
7. error versus visible fraction;
8. predicted uncertainty versus actual error;
9. random held-out examples;
10. worst-case examples;
11. failure montage;
12. accuracy–throughput Pareto plot;
13. before/after refinement overlays;
14. cropped-tail sequence visualization showing observed versus inferred body.

Use consistent coordinate and angle conventions across all plots.

Do not cherry-pick only attractive examples.

---

## 19. Performance engineering

Throughput matters scientifically because this system may become part of a larger behavioral pipeline.

### Benchmarking rules

- set `CUDA_VISIBLE_DEVICES=0` and record the physical-to-logical device mapping;
- ensure no unrelated project process is competing for device 0, or record unavoidable contention;
- exclude one-time model initialization from steady-state fps;
- report warm-up separately;
- benchmark batch 1 separately from batch sizes relevant to full-recording inference;
- report GPU model and software versions;
- report frames/s and milliseconds/frame;
- report latency distribution, including at least median and p95, rather than only an average;
- measure GPU memory;
- benchmark compute-only model execution separately from storage-inclusive end-to-end inference that includes HDF5 reading, preprocessing, refinement, and output serialization;
- because source recordings may be network-mounted, identify the source mount and report whether storage-inclusive results use cold/unprimed or warm filesystem cache; do not present a warm-cache number as guaranteed sustained NFS throughput;
- report both proposal-only and final-system throughput;
- report average latency and difficult-frame latency if using adaptive compute;
- label centered/acausal and causal temporal benchmarks separately and include look-ahead latency for acausal windows;
- use the same documented benchmark harness, frame sample, precision, batch size, and synchronization policy for model comparisons.

### Optimization priorities

Prefer, in order:

1. batched tensor operations;
2. vectorized temporal windows;
3. avoiding CPU↔GPU synchronization;
4. pinned-memory/efficient HDF5 chunk loading where helpful;
5. compact models;
6. local ROI likelihoods instead of full-frame work;
7. adaptive refinement;
8. mixed precision if it does not degrade geometry;
9. `torch.compile` only if stable and actually beneficial.

Do not use custom CUDA kernels unless profiling shows a clear bottleneck that cannot be solved cleanly otherwise.

---

## 20. Experiment record format

Every experiment gets a unique sequential ID.

Example:

```text
experiments/exp_0017_temporal_11f/
```

`notes.md` must contain:

```markdown
# EXP-0017 — 11-frame temporal context

## Hypothesis
...

## Difference from baseline
...

## Data/split
...

## Training/resource budget
- maximum steps/epochs:
- wall-time limit:
- seed/repeat policy:
- checkpoint cadence:
- expected GPU time:
- expected external-storage use:
- early termination conditions:

## Success criterion
- primary metric:
- numeric practical-effect threshold:
- variability/confidence rule:
- pass/fail interpretation:

## Results
...

## Figures
...

## Runtime
...

## Interpretation
...

## Decision
ACCEPT | REJECT | REVISE | INCONCLUSIVE

## Next experiment
...
```

Update `experiments/index.csv` with at least:

- experiment ID;
- date/time;
- git commit, or an explicitly documented tree/patch digest only when committing was not authorized;
- parent/baseline experiment;
- short hypothesis;
- primary metrics;
- fps;
- status/decision;
- path to notes.

Also maintain a concise human-readable `docs/EXPERIMENT_LOG.md` summarizing the sequence of decisions rather than duplicating every raw metric.

### Experiment-flow narrative

In addition to the detailed experiment log, continuously maintain a higher-level scientific narrative in:

```text
docs/EXPERIMENT_FLOW.md
```

This is a **required final deliverable** and should be understandable to a researcher who does not want to read individual experiment folders.

Its purpose is to answer:

> What ideas did we consider, what did we test, what happened, what did we conclude, and how did each result determine what we tried next?

Organize it around the major hypotheses rather than around implementation chronology alone. For each important hypothesis or branch, include:

1. **Hypothesis**
   - a short, high-level statement of the idea being tested and why it might matter;

2. **Test**
   - what comparison or experiment was run;
   - what was held constant;
   - what metric or visual evidence would count as support;

3. **Result**
   - the main quantitative result in a compact form;
   - the most informative graph, overlay, montage, or sequence visualization;
   - uncertainty/error bars where relevant;

4. **Conclusion**
   - `SUPPORTED`, `NOT SUPPORTED`, `PARTIALLY SUPPORTED`, or `INCONCLUSIVE`;
   - a plain-language interpretation of what was learned;

5. **Consequence**
   - how this changed the next experiment or the final architecture.

The experiment flow should be **visual-first**. Do not make it a wall of prose.

For each major hypothesis, prefer a compact pattern like:

```text
Hypothesis
   ↓
What was compared
   ↓
Key graph / visual evidence
   ↓
Result
   ↓
Conclusion
   ↓
What changed next
```

Required visual elements should include, where applicable:

- side-by-side prediction overlays;
- body-angle heatmaps;
- error distributions;
- error-versus-body-position curves;
- crop-robustness curves;
- before/after refinement comparisons;
- uncertainty calibration plots;
- throughput-versus-accuracy plots;
- representative sequence strips;
- failure montages;
- concise annotated plots showing the winning and losing variants.

Create at least one **project-level experiment-flow overview figure** that shows the major branches of reasoning from baseline to final system. It should visually distinguish:

- hypotheses that were supported;
- hypotheses that were rejected;
- hypotheses that were revised;
- the path that led to the final model.

This overview should emphasize scientific decisions rather than every small hyperparameter run.

The final `docs/EXPERIMENT_FLOW.md` must reference the canonical final figures rather than duplicating large binary assets. Whenever possible, use the same figures in both the experiment-flow document and `docs/FINAL_REPORT.md`.

---

## 21. Decision log

`docs/DECISIONS.md` should contain only consequential architectural choices.

For each decision record:

- question;
- options considered;
- evidence;
- decision;
- consequences;
- condition that would justify revisiting it.

Examples:

- coordinate versus angle representation;
- temporal window length;
- whether refinement is retained;
- whether learned likelihood is justified;
- whether particles are necessary;
- final number of body points;
- final dependency set.

---

## 22. Final model selection

Do not select the final model from a single scalar.

Construct a Pareto comparison that includes:

- tangent-angle accuracy on visible body;
- centerline accuracy;
- synthetic FOV-truncation robustness;
- head/tail reliability;
- uncertainty calibration;
- real difficult-frame qualitative performance;
- fps;
- GPU memory;
- model and pipeline complexity.

Prefer the simplest model close to the best Pareto frontier unless a more complex model exceeds the predeclared practical-effect threshold on a scientifically relevant metric without an unacceptable throughput, memory, or reliability cost.

Evaluate the selected configuration once on each applicable untouched final-test evidence tier after selection is complete. Do not use the result to resume model selection. If an implementation defect invalidates the run, document the defect and rerun protocol explicitly rather than silently replacing the result.

---

## 23. Final package requirements

The final accepted implementation must provide:

### Training

A command similar to:

```bash
uv run python scripts/train.py --config configs/final.yaml
```

### Evaluation

```bash
uv run python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/best.ckpt \
  --config configs/final.yaml
```

### HDF5 inference

```bash
uv run python scripts/infer_hdf5.py \
  --input /path/to/recording.h5 \
  --output /path/to/recording_pose.h5 \
  --checkpoint artifacts/checkpoints/best.ckpt
```

Do not require these exact argument names if a cleaner interface emerges, but preserve equivalent functionality.

### PyTorch Lightning integration

Expose a clean Lightning module that can be imported into a larger pipeline without running CLI code.

Avoid embedding HDF5-specific assumptions deeply inside the model class. Keep:

- data access;
- model;
- geometry;
- refinement;
- persistence

as separate components.

### Tests

At minimum test:

- HDF5 frame indexing;
- normalization determinism;
- angle wrap metric;
- centerline reconstruction;
- resampling;
- pixel-center/bounds conventions and deterministic half-open `in_fov_mask` computation;
- head-to-tail export canonicalization and `head_tail_probability` semantics;
- curvature sign and units in image coordinates;
- image-support target semantics and calibration;
- FOV censoring;
- refinement gradient sanity;
- output HDF5 schema version, shapes, dtypes, provenance metadata, partial-output handling, and completion validation;
- checkpoint load + one-batch inference.

---

## 24. Required final deliverables

The project is not complete until all of the following exist.

### A. `README.md`

A concise user-facing explanation of:

- what the tool does;
- installation with `uv`;
- training;
- inference;
- expected input/output;
- example results;
- limitations.

### B. `docs/DATA_AUDIT.md`

What was actually found in the supplied recordings.

### C. `docs/DESIGN.md`

The final architecture, pose representation, temporal model, visibility semantics, refinement, uncertainty model, and HDF5 interface.

### D. `docs/EXPERIMENT_LOG.md`

Chronological experimental record with concise links to the detailed per-experiment evidence.

### E. `docs/EXPERIMENT_FLOW.md`

A high-level, visual-first account of the research process.

It must summarize the major hypotheses, what was tested, the observed results, the conclusions drawn, and how each conclusion changed the next step or final design. It should rely heavily on plots, overlays, heatmaps, montages, and other graphical evidence rather than prose alone.

At minimum it must include:

1. an overview diagram of the major experiment/decision flow;
2. one concise section per major hypothesis;
3. the decisive visual evidence for each accepted or rejected architectural idea;
4. a clear indication of which branches were abandoned and why;
5. the evidence trail that leads from the initial baseline to the selected final model;
6. cross-links to the underlying experiment IDs and detailed logs for auditability.

A reader should be able to understand the project's scientific reasoning from this document without opening every experiment directory.

### F. `docs/DECISIONS.md`

The small set of important architecture decisions and their evidence.

### G. `docs/FAILURE_MODES.md`

Known failures, prevalence if measurable, detection strategy, and mitigation.

### H. `docs/ANNOTATION_RECOMMENDATIONS.md`

If no true manual labels were supplied, identify the smallest set of manual annotations that would most improve confidence in the conclusions. Prioritize active-learning-style cases such as boundary truncation, tight turns, and uncertain head/tail frames.

Also define an annotation protocol that another researcher can execute without guessing: source file/frame identity, original-image coordinate convention, fixed body-point ordering or centerline representation, head/tail labels including an `ambiguous` state, visible/support masks, treatment of self-overlap and off-screen body, annotator/tool/version provenance, and an adjudication rule. Double-label a representative subset and report inter-annotator centerline/angle disagreement so model error can be interpreted relative to label uncertainty.

### I. `docs/FINAL_REPORT.md`

Must include:

1. problem statement;
2. supplied data summary;
3. evaluation protocol;
4. baselines;
5. a high-level experiment-flow summary, with the major hypotheses, tests, visual results, conclusions, and resulting decisions;
6. hypotheses tested;
7. accepted/rejected ideas;
8. final method;
9. quantitative results;
10. accuracy–throughput Pareto analysis;
11. uncertainty/calibration analysis if applicable;
12. cropped-tail results;
13. qualitative successes;
14. failure cases;
15. limits of the evidence, especially pseudo-label versus true-label limitations;
16. final recommendations;
17. next experiments worth doing only if additional accuracy is needed.

### J. Final figures

Place publication-quality versions of the most informative figures in:

```text
artifacts/final_figures/
```

At minimum:

- project-level experiment-flow / decision-path overview;
- representative overlay;
- body-angle heatmap over time;
- error versus body coordinate;
- crop robustness curve;
- uncertainty calibration plot if relevant;
- accuracy–throughput Pareto plot;
- failure montage.

### K. Reusable model artifacts

- final config;
- best checkpoint stored under `/temp_data4/alex/external_artifacts/artifacts/checkpoints/` if it is large, with a repository symlink or documented path;
- inference code;
- output schema documentation;
- benchmark results.

---

## 25. What to do if true labeled data are absent

Do not stop the project merely because dense manual annotations were not supplied.

Proceed using:

- high-confidence pseudo-labels;
- controlled synthetic geometry;
- synthetic FOV truncation of real full-body frames;
- temporal consistency;
- qualitative inspection;
- agreement among independent methods.

However, maintain an explicit distinction between:

```text
measured against true manual labels
measured against high-confidence proxy labels
measured against controlled synthetic truth
qualitative only
```

The final report must never imply more certainty than the evidence supports.

If a small manual annotation set would resolve a major uncertainty, prepare an annotation recommendation, identify the exact frames, and explain how many labels are likely needed and why. Continue the autonomous work that can be done without those labels.

---

## 26. Stopping criteria

During Phase 2, before model development, convert the qualitative terms below into dataset-specific numeric acceptance gates in `docs/EVALUATION.md`. Define thresholds separately for each evidence tier and include the variability/uncertainty rule used to decide whether a gate is met. Do not choose or relax these thresholds after inspecting final-test results.

Stop adding model complexity when all of the following are true:

1. the candidate meets the predeclared visible-body centerline and angle-error gates on ordinary held-out frames;
2. cropped-tail behavior exceeds the simple baseline by the predeclared practical-effect threshold across the required hidden fractions;
3. head/tail flip rate is below its declared maximum or failures are detected with the declared sensitivity/specificity;
4. uncertainty meets the declared calibration/coverage criteria on unobserved and ambiguous body regions, if uncertainty is included;
5. end-to-end offline inference exceeds 20 fps on CUDA device 0 at the documented batch size, with batch-1 and latency-distribution results reported separately;
6. remaining errors are dominated by genuinely ambiguous images or missing ground truth rather than obvious engineering defects;
7. recent complexity additions fail the predeclared Pareto-improvement threshold;
8. the final code is tested, documented, and reproducible.

Do not continue indefinitely in pursuit of tiny metric improvements.

---

## 27. Initial execution order

Unless data inspection gives a reason to change course, begin in this order:

1. bootstrap `uv` environment and repository;
2. inspect all HDF5 files;
3. write data audit and split policy;
4. create evaluation metrics and synthetic crop benchmark;
5. establish classical easy-frame pseudo-label baseline;
6. train a small single-frame learned proposal model;
7. compare coordinate versus intrinsic angle/spline representation;
8. add temporal context and test 1/5/11-frame variants;
9. establish a strong proposal-only baseline and throughput measurement;
10. implement simple differentiable refinement;
11. compare full-render versus local/cross-sectional image likelihood if runtime warrants it;
12. add explicit FOV censoring/visibility modeling;
13. evaluate simple temporal prior for hidden tail;
14. add calibrated uncertainty if it improves failure detection or hidden-body reporting;
15. test adaptive refinement/particles only if ambiguity remains important;
16. run focused tuning on the leading method;
17. freeze model selection;
18. evaluate once on untouched test data;
19. package the final inference path;
20. produce the visual experiment-flow narrative showing hypotheses → tests → results → conclusions → next decisions;
21. produce final documentation, figures, and conclusions.

---

## 28. Default research philosophy

Use the neural network for **amortized proposal**, geometry for **structural correctness**, temporal information for **continuity and missing-body inference**, and image-space refinement for **precision**.

Treat the probabilistic layer as a way to represent real ambiguity and allocate computation intelligently—not as a requirement to run expensive sampling everywhere.

The ideal final system should behave approximately like:

```text
HDF5 video
   ↓
streamed temporal windows
   ↓
fast Lightning proposal model
   ↓
centerline / tangent-angle latent + visibility + confidence
   ↓
optional small differentiable refinement
   ↓
adaptive temporal / multi-hypothesis inference only when needed
   ↓
centerline + body angles + visibility + uncertainty
   ↓
output HDF5 + QC metrics
```

A successful project is one where the final architecture is supported by experiments, not one where every proposed component survives.

---

## 29. Final instruction to the orchestrator

Act as a skeptical research lead, not a task checklist executor.

When an experiment fails, diagnose it and change course. When a simpler method wins, keep the simpler method. When a visual contradicts a scalar metric, investigate the metric. When the evidence is weak, say so. When uncertainty is unavoidable because the worm is outside the FOV, represent that uncertainty rather than inventing certainty.

Leave behind a system another researcher can run, understand, benchmark, and extend without reconstructing your reasoning from terminal history.
