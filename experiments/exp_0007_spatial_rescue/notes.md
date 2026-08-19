# EXP-0007 — Intrinsic spatial-bottleneck rescue

## Hypothesis

Increasing only the proposal encoder's retained spatial grid from 2x2 to 4x4
will correct the localization and body-scale regression-to-mean observed in
EXP-0004 while retaining intrinsic topology, reliability, and high throughput.

## Difference from baseline

The frozen baseline is EXP-0004 intrinsic. This experiment changes one
scientific factor: the fixed encoder pool changes from a 12x16-to-2x2 pool to a
12x16-to-4x4 pool. The intrinsic head, 16-coefficient basis, loss weights,
  training examples/profiles, optimizer, batch size, folds, normalization,
  and evaluation cases remain unchanged. The head grows but must remain below one
million parameters.

Protocol-only corrections make the run auditable: fully-visible Tier C controls
the ordinary-frame checkpoint/gates; every epoch is written to CSV; checkpoint
identity is explicit; and the previously ambiguous early-elimination baseline
is frozen as a hashed pointwise arithmetic mean before training.

## Data/split

Use accepted candidate proxies from the two training recordings in each frozen
development fold plus the same 512 fold-specific development-profile Tier C
samples. Validate candidate proxies separately from the same 128 held-out
Tier C identities, partitioned into fully-visible and artificially cropped
strata. `data_seed=20260818` fixes all synthetic identities across model-seed
repeats; only parameter initialization/training order changes for model seeds
20260819 and 20260820. Start with primary fold 2. Do not read source recordings
or the audited 2025 holdout.

## Training/resource budget

- maximum steps/epochs: 1,200 optimizer steps and 34 epochs per fold
- wall-time limit: 12 GPU minutes per run
- seed/repeat policy: fixed data seed 20260818 and primary model seed 20260818;
  for positive numeric thresholds only, repeat when
  `abs(value-threshold)/abs(threshold) <= 0.10`, then run model seeds 20260819
  and 20260820 on every fold and require every seed/fold to pass. Exact-contract
  or qualitative failures fail directly and do not trigger repeats
- checkpoint cadence: every 300 steps; retain periodic latest and best
  fully-visible Tier C angle checkpoint
- expected GPU time: <=45 minutes for primary runs and <=1.5 additional hours
  only if the near-gate repeat rule fires
- expected external-storage use: <=3 GiB
- early termination: non-finite values, identity/preflight failure, >1M
  parameters, wall-time limit, or step-300 fully-visible median point error not
  below the frozen fold-specific mean-centerline baseline. Hitting the wall-time
  limit before step 1,200 is `INCONCLUSIVE`, never acceptance

The executable baseline is the pointwise arithmetic mean, in generator order,
of the exact 512 `centerline_xy` targets returned by `SyntheticTierCDataset`,
including its deterministic camera crops. The artifact is a `[100,2]`
little-endian float32 NumPy `.npy` file written with `allow_pickle=False`.
Record both the file SHA-256 and the tensor SHA-256 over raw C-order bytes, plus
the ordered sample-seed manifest, construction metadata, and validation case
identities before model training. On the exact 43 fully-visible cases from the
128 held-out Tier C identities, choose only the better forward/reverse
correspondence and measure per-point Euclidean error after scaling to 968x732
original pixels. Evaluate an immutable checkpoint whose stored `global_step`
is exactly 300; record its digest and eliminate when its median error is greater
than or equal to the frozen baseline median before any resume.

## Success criterion

- primary practical effect: primary-fold fully-visible Tier C median point
  error at least 20% below EXP-0004 intrinsic's 116.92 px, while synchronized
  CUDA throughput is at least 90% of its 2,320 samples/s batch-32 result
- reliability: on every development fold, candidate-proxy median/p95 point
  error <=8/20 px and mean/p95-frame angle MAE <=15/30 degrees; fully-visible
  Tier C median/p95 point error <=4/10 px and mean/p95-frame angle MAE <=8/18
  degrees. Candidate-proxy mean endpoint error must be <=15 px per endpoint,
  median length error <=8%, and support Brier/ECE <=0.12/0.10. Tier C mean
  endpoint error must be <=8 px per endpoint, median length error <=5%, and
  support Brier/ECE <=0.06/0.05. Require exact FOV contract, zero failed
  inference, and no systematic topology/shortcut failure in frozen random and
  worst overlays
- expansion rule: fold 2 must first pass the executable step-300 baseline; run
  to 1,200 steps. Other folds run only if the primary fold passes every
  reliability gate at its best checkpoint
- variability: the all-fold rule is mandatory. The near-gate seed policy above
  applies independently and cannot be replaced by pooled confidence intervals
- interpretation: Tier C supports controlled geometry claims and candidate
  proxies support engineering consistency only; neither is manual truth

This is a geometry-only rescue. Even if every gate above passes, EXP-0007 alone
does not authorize temporal modeling. A subsequent preregistered advancement
experiment must first evaluate the frozen cropped-FOV visible/hidden/boundary
gates, orientation limitations, EXP-0006 candidate-proxy crop evidence, and the
full support contract.

Benchmark the best-checkpoint digest with the identical EXP-0004 harness,
float32 input semantics, physical GPU 0, batch size 32, 100 iterations, and
synchronization. Require at least 90% of the 2,320.38 samples/s intrinsic
reference and >20 fps end-to-end. Report batch-1 p50/p95, forward-only and
end-to-end throughput, preprocessing, peak memory, parameter count, full
environment identity, and checkpoint digest.

## Locked command trail

These are the checked-in CLI stages and flags for the final audited fold-2 run.
Run them from clean commit `a886a314ef3409cc52a760dd2b6e845fdeb0752c`.
`<EXTERNAL_ARTIFACT_ROOT>` is a required writable external-storage placeholder,
not a repository path. For the audited run it was
`/temp_data4/alex/external_artifacts`; `EXP7_ROOT`, `BASELINE_DIR`, and `RUN_DIR`
therefore resolve to the preserved final artifact directories shown below.
Several writers refuse overwrite, while evaluation writes into its requested
directory. This block therefore records the locked trail only: use a new
external root for a fresh reproduction and never rerun it over final evidence.

```bash
EXTERNAL_ARTIFACT_ROOT="<EXTERNAL_ARTIFACT_ROOT>"
EXP7_ROOT="$EXTERNAL_ARTIFACT_ROOT/experiments/worm_pose_gen/exp_0007"
BASELINE_DIR="$EXP7_ROOT/baselines_final/fold2"
RUN_DIR="$EXP7_ROOT/intrinsic_fold2_seed20260818_final"
CONFIG="configs/spatial_rescue.yaml"
```

1. Build the frozen fold-2 arithmetic-mean baseline, then verify the artifact,
   raw tensor identity, clean code provenance, and 43-case validation stratum.

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/exp_0007_baseline.py build \
  --config "$CONFIG" --fold 2 --output-dir "$BASELINE_DIR"

sha256sum "$BASELINE_DIR/baseline.json" "$BASELINE_DIR/mean_centerline.npy"
# 2d3e22299e9334dbbf265385a108c94270309bd9a68f68a112b0322cb7d87a6e  baseline.json
# 388c2b3a3143cf67b817c5ff267fc94ba8b2faa60aad67d589d2cbc6d109fa2c  mean_centerline.npy

jq -e '
  .experiment == "EXP-0007" and .fold == 2 and .data_seed == 20260818 and
  .tensor_sha256 == "bc54ccf0b4dc8f545bd999283f7450d61e99e9abac2c5de8678ea1eed561a052" and
  .validation.fully_visible_count == 43 and
  .code_provenance.git_commit == "a886a314ef3409cc52a760dd2b6e845fdeb0752c" and
  .code_provenance.repository_clean == true and
  .audited_holdout_opened == false and .source_recordings_opened == false
' "$BASELINE_DIR/baseline.json"
```

2. Start fresh and stop at exactly 300 optimizer steps. EXP-0007's training
   entry point rejects a fresh request beyond step 300.

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/train.py \
  --config "$CONFIG" --variant intrinsic --fold 2 \
  --output-dir "$RUN_DIR" --max-steps 300 --seed 20260818 \
  --baseline-metadata "$BASELINE_DIR/baseline.json"

sha256sum "$RUN_DIR/step300.ckpt"
# 17c652b5d06b7a2923376bd84a310b58b0780e424fd1b5007a7ff44e3c9cabb3  step300.ckpt
```

3. Compare that immutable checkpoint to the verified baseline. The comparison
   artifact is the only authorization for continuation; assert its exact
   checkpoint step and `CONTINUE_TO_1200` decision before resuming. The compare
   CLI's checked-in default device is CPU, so no unrecorded device flag is used.

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/exp_0007_baseline.py compare \
  --config "$CONFIG" --baseline-metadata "$BASELINE_DIR/baseline.json" \
  --checkpoint "$RUN_DIR/step300.ckpt" --fold 2 \
  --model-seed 20260818 --output "$RUN_DIR/baseline_comparison.json"

jq -e '
  .checkpoint_global_step == 300 and
  .checkpoint_sha256 == "17c652b5d06b7a2923376bd84a310b58b0780e424fd1b5007a7ff44e3c9cabb3" and
  .eliminate == false and .decision == "CONTINUE_TO_1200"
' "$RUN_DIR/baseline_comparison.json"
```

4. Resume only from the immutable step-300 checkpoint, bind the comparison,
   and finish the frozen 1,200-step budget. `--resume-last` is deliberately not
   valid for this continuation.

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/train.py \
  --config "$CONFIG" --variant intrinsic --fold 2 \
  --output-dir "$RUN_DIR" --max-steps 1200 --seed 20260818 \
  --baseline-metadata "$BASELINE_DIR/baseline.json" \
  --baseline-comparison "$RUN_DIR/baseline_comparison.json" \
  --resume-from "$RUN_DIR/step300.ckpt"

sha256sum "$RUN_DIR/intrinsic-fold2-best.ckpt"
# 7606c7fbb990fee59624f6882155dea250fecab8059988d69441254ff7bc13d0  intrinsic-fold2-best.ckpt
```

5. Evaluate and benchmark that exact best-checkpoint digest. Evaluation writes
   `metrics.json` and the five review PNGs beneath `evaluation/`; benchmarking
   uses the frozen synchronized batch-32, 100-iteration CUDA protocol.

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate.py \
  --config "$CONFIG" --checkpoint "$RUN_DIR/intrinsic-fold2-best.ckpt" \
  --fold 2 --output-dir "$RUN_DIR/evaluation" --device cuda \
  --model-seed 20260818 --baseline-metadata "$BASELINE_DIR/baseline.json"

scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/benchmark_model.py \
  --checkpoint "$RUN_DIR/intrinsic-fold2-best.ckpt" \
  --output "$RUN_DIR/benchmark.json" --batch-size 32 --iterations 100 --fold 2
```

6. Qualitative review has no checked-in executable CLI. A reviewer must inspect
   `diagnostics.png`, `tier_B_random.png`, `tier_B_worst.png`,
   `tier_C_random.png`, and `tier_C_worst.png`, then produce the schema-bound
   `qualitative_review.json`. The final audited review is copied beside these
   notes; inventing an automated review command would not reproduce the run.

7. Bind evaluation, benchmark, and completed qualitative review with the
   deterministic fail-closed decision engine.

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/exp_0007_decide.py \
  --config "$CONFIG" --metrics "$RUN_DIR/evaluation/metrics.json" \
  --benchmarks "$RUN_DIR/benchmark.json" \
  --reviews "$RUN_DIR/qualitative_review.json" \
  --output "$RUN_DIR/primary_fold_decision.json"

jq -e '
  .decision == "PRIMARY_FOLD_FAIL" and
  .authorize_additional_folds == false and .authorize_repeat_seeds == false and
  .final_accept_geometry_rescue == false and .authorize_temporal_modeling == false
' "$RUN_DIR/primary_fold_decision.json"
```

## Results

The audited primary-fold run used model/data seed 20260818 on fold 2 from clean
commit `a886a314ef3409cc52a760dd2b6e845fdeb0752c` (tree
`7d9573dd141a7ea637cfa15ac6f70aa65aa57b5b`), config SHA-256
`b498ef82c65c6114725dae4b3bf6b19056b3601235aa9cc6daf5cf7bbff3b616`,
and the 4x4 intrinsic model with 722,137 parameters.

The immutable step-300 checkpoint (`17c652b5...cabb3`, stored global step 300)
passed early elimination: fully-visible Tier C median point error was 102.15 px
versus the frozen arithmetic-mean baseline's 243.28 px, so training continued
to all 1,200 optimizer steps. Final evaluation and benchmarking used the best
fully-visible Tier C checkpoint at global step 720, SHA-256
`7606c7fbb990fee59624f6882155dea250fecab8059988d69441254ff7bc13d0`.

The two primary practical-effect gates passed. Fully-visible Tier C median point
error was 87.54 px, a 25.13% improvement over EXP-0004 intrinsic's 116.92 px
(required >=20%). Synchronized batch-32 end-to-end throughput was 2,461.33
samples/s, 106.07% of the 2,320.38 samples/s reference (required >=90%); forward
throughput was 30,297.19 samples/s. Batch-1 forward p50/p95 was 0.863/0.884 ms,
batch-1 end-to-end p50/p95 was 1.306/1.348 ms, and the end-to-end offline rate
also exceeded the >20 fps gate. The parameter ceiling and benchmark identity,
precision, synchronization, device, and checkpoint-binding contracts passed.

The unchanged reliability gates failed by large margins:

- Candidate proxy (31 cases): median/p95 point error 83.95/239.04 px versus
  <=8/20; mean/p95-frame angle error 61.00/101.29 degrees versus <=15/30;
  endpoint errors 157.93/156.90 px versus <=15 each; median length error 31.66%
  versus <=8%.
- Fully-visible Tier C (43 cases): median/p95 point error 87.54/233.24 px versus
  <=4/10; mean/p95-frame angle error 27.68/53.29 degrees versus <=8/18;
  endpoint errors 118.01/125.34 px versus <=8 each; median length error 37.75%
  versus <=5%.
- Support calibration itself passed on both ordinary-frame strata, exact
  reported/recomputed FOV agreement held, and failed inference count was zero.
  The separately reported 85 cropped Tier C controls had 80.06 px median point
  error, 22.29-degree mean angle error, 0.1055 Brier, and 0.0878 ECE; they were
  not substituted for the ordinary-frame gate.

The completed qualitative review found a systematic topology/shortcut failure:
predictions regress toward short, nearly straight mean poses and frequently
miss position, extent, curvature, and orientation on both candidate-proxy and
analytic Tier C images. This exact/qualitative failure directly blocks the
near-gate repeat policy. No repeat seeds or additional folds were authorized.
The audited holdout and source recordings were not opened.

### Superseded pilots

The external `intrinsic_fold2_seed20260818_smoke` and
`intrinsic_fold2_seed20260818_bound` directories were setup/protocol pilots and
are explicitly superseded by `intrinsic_fold2_seed20260818_final`. Likewise,
the earlier `baselines`, `baselines_bound`, and `baselines_exact` variants are
superseded by `baselines_final`. None of those pilot outputs contributes to the
reported metrics, figures, checkpoint decision, or experiment-index row. Full
external artifacts remain preserved in place.

## Figures

The five final evaluation figures are copied byte-for-byte under `figures/`:
`diagnostics.png`, candidate-proxy `tier_B_random.png` and `tier_B_worst.png`,
and analytic `tier_C_random.png` and `tier_C_worst.png`. Visual reinspection
agrees with the recorded qualitative review: the red predictions are commonly
too short and straight, displaced from the worm, and wrong in orientation or
curvature; worst cases are not isolated outliers. Diagnostics show high error
throughout the body rather than only at FOV boundaries.

Raw small evidence is retained locally as `metrics.json`, `benchmark.json`,
`baseline_comparison.json`, `primary_fold_decision.json`, and
`qualitative_review.json`. The corresponding SHA-256 values remain those bound
in the deterministic decision artifact; copied-file hashes were verified
against the preserved external originals.

## Runtime

Artifact timestamps record approximately 57 seconds for the step-0-to-300
training segment and 3 minutes 53 seconds for the resumed step-300-to-1,200
segment, about 4 minutes 50 seconds of training-segment wall time. The complete
final pipeline from initial run metadata through baseline comparison, resume,
evaluation, synchronized benchmark, qualitative review, and deterministic
decision spanned approximately 8 minutes 46 seconds. Both are below the
12-GPU-minute per-run budget. No new source or GPU execution was performed when
integrating this evidence.

## Interpretation

Retaining a 4x4 encoder grid improves the primary fold's median localization
relative to EXP-0004 and preserves excellent throughput, but it does not solve
the underlying proposal shortcut. Absolute point, endpoint, length, and angle
errors remain far outside every ordinary-frame reliability gate, and the frozen
overlays establish a systematic failure. Low support calibration error is not
evidence of pose accuracy because these ordinary frames are almost entirely
visible. The result supports rejecting this spatial-bottleneck-only rescue, not
relaxing the gates or pooling across unrun folds. It provides no authorization
for cropped-FOV advancement or temporal modeling.

## Decision

PRIMARY_FOLD_FAIL — the primary effect and throughput passed, but ordinary-frame
reliability and the exact qualitative topology gate failed. Additional folds
and repeat seeds are not authorized; geometry rescue acceptance and temporal
modeling remain false.

## Next experiment

Stop this proposal branch. Keep temporal 1/5/11-frame context, refinement, and
the audited holdout blocked. Before another learned proposal, collect and
adjudicate the manual Tier A tranche in `docs/ANNOTATION_RECOMMENDATIONS.md`;
then preregister one localization-explicit architecture against the unchanged
ordinary-frame gates. A cropped/support/orientation advancement experiment is
permitted only after that new proposal passes every required development fold.
