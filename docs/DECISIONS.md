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

## D-0009 — Stop EXP-004 after the analytic 5k control fails

- **Question:** Does analytic diversity at the plan's first 5k scale make the topology-safe model reliable enough to begin real-texture synthesis or real-label evaluation?
- **Options:** advance directly to real texture; run repeat seeds; stop under the frozen controlled gate.
- **Evidence:** With 5,000 analytic development cases and approximately matched dataset exposure, the primary seed improved EXP-003B's fully-visible Tier-C point and length errors by 24.77% and 23.33%, but observed 46.64 px, 25.17 degrees, and 0.261 length error versus frozen limits of 16 px, 15 degrees, and 0.15. Every gate failed. Tensor, exclusion, parent-source, checkpoint, GPU, and split-semantic provenance all validated.
- **Decision:** Record `ANALYTIC_5K_PRIMARY_CONTROLLED_GATE_FAIL`. Do not run repeat seeds, add real texture, or evaluate Tier A. Keep delayed repeats and the protected holdout closed.
- **Consequence:** Data scale alone does not rescue the unchanged topology-safe decoder. The next experiment, if any, must state a new Tier-C-only architecture, loss, or supervision hypothesis rather than tuning against real annotations.
- **Revisit if:** a separately preregistered controlled proposal passes all three frozen Tier-C criteria on every authorized seed.

## D-0010 — Replace the neural-proposal authorization gate for the SMC branch

- **Question:** Does D-0007's failed direct-regression proposal prohibit the segmentation-anchored generative branch defined by the later SMC plan?
- **Options:** require a neural ordinary-frame proposal anyway; reinterpret the old failure; give the new branch its own explicit prerequisite.
- **Evidence:** `worm_pose_smc_experiment_orchestrator.md` explicitly replaces direct neural pose regression with segmentation, conservative classical easy-frame anchors, a low-dimensional generative model, and temporal smoothing. EXP-SMC-001/002 independently test the new branch's actual upstream assumptions; the frozen first audit found accurate conditional complete anchors but unsafe terminal completeness.
- **Decision:** D-0007 remains binding for direct learned pose proposals, their temporal extensions, and the protected holdout. It does not block the distinct SMC branch. Authorization to fit anchor-derived priors or run temporal inference now requires a prospectively validated segmentation/anchor detector under EXP-SMC-001/002 (or an explicitly numbered revision), with no reliance on the rejected neural checkpoint.
- **Consequence:** no dynamics or SMC work is authorized by the current `NOT_SUPPORTED` EXP-SMC-001/002 result. One targeted segmentation/anchor-integrity revision is permitted; direct neural pose regression remains out of scope unless this structured approach is demonstrated to be blocked.
- **Revisit if:** the revised anchor detector passes its reliability gate, or the segmentation/anchor assumption fails after a documented targeted repair.

## D-0011 — Stop the SMC branch at the segmentation/anchor gate

- **Question:** Does the prospective terminal-recovery and width-relative FOV revision make the anchor set safe enough for posture, width, and dynamics fitting?
- **Options:** authorize downstream latent/SMC experiments; tune another cleanup threshold on the same labels; stop at the failed prerequisite.
- **Evidence:** EXP-SMC-001B added only 0.97%/2.20% median/p95 connected area and kept adjacent-area p95 at 1.44%, but median terminal containment remained 0.80 versus 0.90. EXP-SMC-002B rejected all 12 truncated cases but accepted only 3/17 complete frames, only 2/3 within 8 px versus 90%, with zero anchors in one session. Independent review confirmed that the six hard-bout candidates are unadjudicated and their sparse legacy brackets are generally hundreds of frames away.
- **Decision:** record `UPSTREAM_CORE_ASSUMPTION_FAIL_FOR_AVAILABLE_SEGMENTATION`. Do not build the empirical anchor dataset or run EXP-SMC-003 through EXP-SMC-015. Do not reinterpret the low-dimensional oracle as authorization, because the required anchor-derived data and dynamics have not passed their gates.
- **Consequence:** the central natural-bout SMC question remains unanswered rather than negatively answered. Preserve all negative evidence and the hard-bout catalog; keep the 2025 holdout closed.
- **Revisit if:** a new frozen foreground method trained/evaluated with fresh terminal-aware mask truth passes the terminal and full anchor-reliability gates and yields adequate validated contiguous anchors in every development session.

## D-0012 — Expert visual adjudication authorizes development continuation

- **Question:** Does explicit expert review of the complete EXP-SMC-002B visual set change the development stop/continue decision without rewriting the frozen quantitative evidence?
- **Options:** retain D-0011 as an absolute stop; alter the original annotation/metrics; preserve them and record a separate bounded expert adjudication.
- **Evidence:** EXP-SMC-002C binds all 30 one-based visual rows to the immutable EXP-SMC-002B metrics digest. The expert identifies row 2 (`2023-09-19-01-f017959`) as a genuinely mistaken annotation, row 22 (`2023-10-11-01-f013785`) as an expected hard case for SMC rather than a required easy anchor, and judges the remaining rows visually adequate to move forward. The source annotation and frozen `NOT_SUPPORTED` metrics remain unchanged.
- **Decision:** D-0011 remains the historical frozen-gate decision but is superseded as the current development-governance decision. Record `EXPERT_VISUAL_DEVELOPMENT_CONTINUATION_AUTHORIZED`. Authorize development-only anchor-dataset construction and EXP-SMC-003 onward in prerequisite order. Do not describe EXP-SMC-001B/002B as quantitative passes.
- **Consequence:** the branch may test latent representation, width, renderer, dynamics, and controlled/natural SMC on development recordings. Expert review is not a quantitative annotation noise floor, independent validation, holdout evidence, calibration evidence, or deployment authorization. Row 22 remains an untested downstream SMC challenge, not a solved case.
- **Revisit if:** downstream anchor-distribution checks reveal systematic contamination, controlled renderer/dynamics gates fail, or SMC cannot retain the correct mode on adjudicated hard bouts. Keep the protected 2025 holdout closed until a later explicit final-selection decision.

## D-0013 — Retain the minimal geometric and observation components

- **Question:** Which independently tested components are justified for any future generative pose study?
- **Options:** discard the branch wholesale; retain every passing candidate; retain only the least-complex passing components within their evidence boundaries.
- **Evidence:** EXP-SMC-003's selected fixed cubic K=16 state reconstructed 17 complete traces at 0.697/1.074 px median/p95 point error and 3.305/4.307 degrees median/p95 tangent error. In EXP-SMC-004, bounded width scale reached 0.8663 median proxy-mask IoU, PCA-2 added only 0.00038, scale SD was 0.0245, and a 10 px shift cost 0.2041 IoU; the fixed recording mean was slightly better than bounded scale and simpler. In EXP-SMC-005, soft Dice placed all 64 controlled perturbation minima at zero or an adjacent level, achieved 0.9714 outward monotonicity, and had finite nonzero gradients for every 20-value pose group.
- **Decision:** Retain the fixed 16-coefficient cubic tangent representation with translation, rotation, and length; the fixed recording-level mean width profile outside the initial particle state; and soft Dice as the observation energy. Record `COMPONENTS_RETAINED_WITH_ORACLE_PROXY_LIMITS`.
- **Consequence:** These components may seed a future preregistered study, but they do not jointly establish natural pose inference, calibrated likelihoods, biological width truth, or human-level accuracy. Component retention does not override the later dynamics and anchor-bracket failures.
- **Revisit if:** independent mask truth, a broader anti-compensation test, or natural hard-case evidence reverses the component ordering. Keep the protected holdout closed.

## D-0014 — Reject H5 and session-general empirical dynamics

- **Question:** Do the strict-anchor windows support a session-general natural-motion transition that improves on persistence?
- **Options:** pool all accepted transitions; fit per-session dynamics despite missing coverage; reject H5 and restrict any fallback to synthetic controls.
- **Evidence:** EXP-SMC-002D accepted 87/303 frames but produced adjacent-pair counts of 7/45/0 by recording, with 45/52 pairs from one session. EXP-SMC-006 had five-frame prediction counts of 0/16/0 and no 20-frame case. On the 40 shared one-frame forecast keys, persistence was 3.572 px median error and the best non-persistence diagnostic was 4.484 px, 25.6% worse and worse in both contributing recordings.
- **Decision:** Record `H5_NOT_SUPPORTED__SESSION_GENERAL_DYNAMICS_FAILED`. Do not fit or claim an empirical natural-motion model from these anchors. The EXP-SMC-006 zero-drift block-diagonal random walk is a declared synthetic control parameterization only.
- **Consequence:** Natural difficult-bout inference cannot borrow scientific validity from the fitted/velocity diagnostics. New natural-dynamics work requires prospectively adequate contiguous transitions in every development session or a separately preregistered sparse-time hypothesis.
- **Revisit if:** improved anchors or new temporal annotations meet a frozen per-session transition minimum and a leakage-safe model beats persistence. Keep the protected holdout closed.

## D-0015 — Controlled SMC execution does not support H8 or natural SMC

- **Question:** Does controlled recovery justify terminal-anchor smoothing or SMC superiority claims?
- **Options:** treat passing synthetic gates as natural support; claim a generic terminal-anchor benefit; retain only controlled algorithm evidence.
- **Evidence:** EXP-SMC-007's held-out nominal median trajectory errors were 2.19 px forward and 1.95 px terminal-reweighted, truth survival was 1.00, and all frozen synthetic stress gates passed. However, exact two-anchor interpolation had the lowest median trajectory error in all 6/6 scenarios, while terminal reweighting improved forward SMC in only 2/6 and worsened it in four.
- **Decision:** Record `SUPPORTED_CONTROLLED_SYNTHETIC_ONLY`. Controlled execution and truth survival are supported; SMC superiority over interpolation, a general terminal-anchor smoothing benefit, and H8 are `NOT_SUPPORTED`. Natural SMC remains unauthorized.
- **Consequence:** EXP-SMC-007 is an implementation smoke test against renderer-matched synthetic random walks, not evidence for natural motion, self-contact recovery, or a validated empirical prior.
- **Revisit if:** a prospectively valid natural two-anchor bout exists and a frozen comparison shows an SMC benefit without using protected holdout data.

## D-0016 — Stop natural row-22 SMC at the missing-anchor prerequisite

- **Question:** Can the expert-designated hard row 22 serve as a short natural two-anchor SMC test under the unchanged final anchor pipeline?
- **Options:** infer poses despite absent anchors; widen or tune after seeing the result; record the prerequisite failure and stop.
- **Evidence:** EXP-SMC-008A scanned exactly `2023-10-11-01` frames 13685-13885 around frame 13785 and accepted 0/201. There was no strict anchor before or after the hard frame, no accepted run, and one 201-frame rejected gap. All 201 frames triggered `branch_pixels` and `cycle`, 183 triggered abrupt/implausible width checks, and 150 triggered low render IoU. No SMC pose was inferred.
- **Decision:** Record `NATURAL_ROW22_CORE_ANCHOR_ASSUMPTION_FAIL`. The designated hard case has no <=20-frame strict two-anchor bracket, so natural SMC cannot be evaluated under the current design. Do not widen the window, tune the detector, or infer an unanchored trajectory within this experiment.
- **Consequence:** The branch ends with controlled-only algorithm evidence and an unanswered natural reconstruction question. This is a failure of the required nearby-reliable-anchor assumption for the designated natural case, not evidence that SMC reconstructed it badly or that no anchor exists anywhere else in the recording.
- **Revisit if:** a newly preregistered foreground/anchor method yields independently reliable local brackets, or prospectively annotated bouts provide endpoints under a new experiment ID. The protected 2025 holdout remains closed.
