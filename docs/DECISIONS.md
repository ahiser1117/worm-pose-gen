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

## D-0005 — Condition-level real-texture crop benchmark

- **Question:** How can real texture be retained after complete same-frame crop series failed their frozen geometry gate?
- **Options:** relax the support contract; discard difficult cells; balance only prevalidated crop conditions.
- **Evidence:** EXP-0005 rejected the same-frame hypothesis with 14/90 complete frames, while EXP-0006 produced exactly 10 valid cases in every one of 30 recording/end/fraction cells and passed all 300 provenance checks.
- **Decision:** freeze the EXP-0006 condition-level artifact as a static candidate-proxy engineering stratum without relaxing geometry or reinterpreting EXP-0005.
- **Consequence:** the benchmark has 300 rows but only 87 unique source frames; it cannot establish anatomical accuracy, biological independence, or temporal performance.
- **Revisit if:** independently annotated real sequences support a complete temporal benchmark.

## D-0006 — Reject the 2x2-bottleneck proposal family

- **Question:** Is either direct-coordinate or intrinsic regression reliable enough to initialize later phases?
- **Options:** accept coordinates; accept intrinsic; reject both and revise one factor.
- **Evidence:** EXP-0004 coordinate predictions formed extreme zigzags; intrinsic predictions were smooth but the best primary-fold fully-visible result was 116.92 px median point error and 23.35 degrees mean angle error against 4 px and 8 degree gates.
- **Decision:** reject both frozen candidates, retaining intrinsic structure only as the controlled baseline for a 4x4 spatial-bottleneck rescue.
- **Consequence:** do not run other folds, temporal context, refinement, or uncertainty until a revised ordinary-frame proposal passes unchanged gates.
- **Revisit if:** EXP-0007 passes its executable early-elimination rule and all reliability gates.

## D-0007 — Stop model expansion after the 4x4 rescue fails

- **Question:** Does the one-factor spatial rescue justify additional folds or later modeling phases?
- **Options:** accept fold 2 and expand; repeat a near-gate result; reject under the frozen rule.
- **Evidence:** EXP-0007 improved the primary-fold median by 25.13% and reached 2,461 batch-32 samples/s, but fully-visible Tier C remained at 87.54/233.24 px median/p95 and 27.68/53.29 degrees mean/p95-frame versus 4/10 px and 8/18 degree gates. Endpoints, length, candidate proxies, and the hash-bound qualitative shortcut gate also failed.
- **Decision:** record `PRIMARY_FOLD_FAIL`. Do not run folds 0/1, repeat seeds, temporal context, refinement, uncertainty, or the audited holdout.
- **Consequence:** no model is accepted or deployable. Preserve the holdout for a future study with a reliable development-fold proposal.
- **Revisit if:** a new, preregistered localization-explicit architecture passes every unchanged ordinary-frame development gate; do not reinterpret EXP-0007 as near-gate evidence.

## D-0008 — Package only fail-closed exploratory inference

- **Question:** How can the tested persistence stack remain reusable without presenting the rejected checkpoint as a final model?
- **Options:** omit inference; expose an apparently normal model CLI; require explicit diagnostic opt-in and semantic sentinels.
- **Evidence:** the geometry/writer/checkpoint integration is testable, but the model lacks validated orientation, pose uncertainty, and quality heads and failed scientific selection.
- **Decision:** retain the checkpoint as `exp_0007_rejected_diagnostic.ckpt`; require `--allow-exploratory`; mark outputs `exploratory_rejected_checkpoint`; reject temporal calls; export `0.5`, `pi`, and `0` as documented unknown/rejected sentinels.
- **Consequence:** researchers can reproduce failures and exercise atomic HDF5 output, but outputs must not be used for biological measurement or described as calibrated.
- **Revisit if:** a future checkpoint passes the full selection, orientation, uncertainty, and storage-inclusive inference gates; that model requires a new validation status and config rather than removing safeguards from this artifact.
