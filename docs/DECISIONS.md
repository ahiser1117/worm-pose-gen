# Decisions

## D-0001 — Reproducible local environment and external storage

- **Question:** Where should environments, caches, and accumulating outputs live?
- **Options:** system environment; repository-only; repository-local runtime plus external outputs.
- **Evidence:** Phase 0 bootstrap and write/read probes passed with the specified paths.
- **Decision:** Keep `.venv`, `.uv-python`, and `.cache` repository-local and ignored; store accumulating outputs beneath `/temp_data4/alex/external_artifacts`.
- **Consequence:** Commands must use `scripts/project_env.sh`; outputs record configured and resolved roots.
- **Revisit if:** the external root becomes unavailable or the repository-local runtime ceases to be writable.

## D-0002 — Source I/O remains bounded and read-only

- **Question:** How should the 155.656 GiB source inventory be accessed?
- **Options:** copy locally; scan eagerly; bounded streaming reads from supplied symlinks.
- **Evidence:** the accepted source mount is read-only NFSv4 and random access is available for a subset of files.
- **Decision:** Use bounded frame-indexed reads with at most two readers; never copy or modify source recordings.
- **Consequence:** Storage-inclusive benchmarks must distinguish cache state and I/O from compute.
- **Revisit if:** an experiment has a predeclared need for a full sequential scan within the resource budget.

## D-0003 — Whole-session split with limited generalization scope

- **Question:** What is the strongest leakage-safe split supported by readable data?
- **Options:** frame split; temporal blocks; whole recordings/sessions; background-family holdout.
- **Evidence:** only four recordings are readable and all share the starvation project family; one 2025 session records a different camera condition.
- **Decision:** use three leave-one-session-out development folds across the readable 2023 sessions. Reserve the unaudited frames of 2025-03-06 for a one-time post-freeze audited-holdout evaluation and exclude the 32 pre-split audit indices.
- **Consequence:** claims are limited to session/camera-shift evidence and cannot establish pristine-test or cross-project/background generalization.
- **Revisit if:** readable independent-background recordings or a separately approved dataset become available; do not alter the frozen audited holdout within this study.

## D-0004 — Mixed proxy/synthetic training evidence

- **Question:** What supervision is defensible without supplied manual centerlines?
- **Options:** classical labels alone; analytic synthetic data alone; explicitly separated candidate-proxy, reviewed Tier B qualitative, and Tier C controlled evidence.
- **Evidence:** EXP-0001 accepted 90/144 conservative real candidates with 0/24 gross visual failures, while EXP-0002 passed all 6,400 exact crop contracts and differentiable-renderer checks.
- **Decision:** train proposals from candidate proxies and synthetic data, but never treat proxy agreement as independent real accuracy or synthetic success as appearance validation. Reserve Tier B language for the 24 independently reviewed qualitative cases.
- **Consequence:** every model result must identify its evidence tier; head/tail supervision stays symmetric/ambiguous until true orientation labels exist.
- **Revisit if:** manually annotated Tier A labels become available.
