# Experiment flow

The project ended with a scientifically valid negative result: evidence
infrastructure was established, but neither proposal family met the frozen
ordinary-frame reliability gate. No final model was accepted and the audited
holdout was preserved.

![Project-level experiment and decision flow](../artifacts/final_figures/experiment_flow_overview.png)

## H0 — Can bounded real candidates and analytic truth support controlled research?

```text
EXP-0001 conservative real extraction      EXP-0002 analytic geometry
90 / 144 candidate proxies                 6,400 / 6,400 crop checks
0 / 24 reviewed gross failures             <=3.41e-13 px round trip
                 \                          /
                  v                        v
       separate real-texture proxy and exact-geometry strata
```

**Test.** Freeze conservative classical acceptance criteria, inspect random and
worst cases, and independently validate the differentiable renderer and static/
moving FOV transforms. Keep evidence tiers separate.

**Visual evidence.** The
[`random accepted proxies`](../experiments/exp_0001_classical_proxy/figures/random_accepted_overlays.png)
show usable real texture but occasional endpoint under-reach. The
[`controlled crop sequence`](../experiments/exp_0002_synthetic_crop/figures/crop_sequence_montage.png)
shows exact known support under a moving camera.

**Conclusion — PARTIALLY SUPPORTED.** These strata are sufficient for
engineering and controlled-geometry tests, but neither is Tier A manual truth.

**Consequence.** Proceed to small proposal comparisons while prohibiting
real-anatomical accuracy claims.

## H0b — Can a literal training-size window create the real crop benchmark?

```text
EXP-0003: 256 x 192 source window
        65 / 900 valid; 0 / 90 complete
                         |
                         v
EXP-0005: larger 4:3 source window + isotropic resize
        720 / 900 valid; 14 / 90 complete (gate >=60)
                         |
                         v
REJECT complete same-frame series; curved anatomy re-enters the window
```

**Test.** Hide 5/10/20/30/40% of either anatomical end while preserving a
contiguous visible complement, exact transforms, and unchanged real pixels.

**Visual evidence.** EXP-0003's
[`yield curve`](../experiments/exp_0003_real_texture_crop/figures/contract_yield.png)
shows that the physical window is too small. EXP-0005's
[`scaled crop evidence`](../experiments/exp_0005_scaled_real_crop/figures/scaled_real_crop_evidence.png)
shows the residual geometric re-entry failure.

**Conclusion — NOT SUPPORTED.** Scaling raises condition yield but does not
make the preregistered complete-frame benchmark feasible.

**Consequence.** Change the experimental unit prospectively rather than relax
the failed gate.

## H0c — Can valid conditions form a balanced static real-texture artifact?

```text
frozen EXP-0005 valid pool
          |
SHA-ranked 10 cases per recording x end x hidden fraction
          |
300 / 300 exact cases; 87 unique frames; all transforms verified
```

**Test.** Select exactly ten prevalidated conditions in each of 30 cells and
bind every source window, interpolation, support mask, and transform by digest.

**Visual evidence.** The
[`balance plot`](../experiments/exp_0006_balanced_real_crop/figures/balance.png)
shows exact cell coverage; the
[`all 40% cases`](../experiments/exp_0006_balanced_real_crop/figures/all_40_percent_cases.png)
expose the hardest selected stratum.

**Conclusion — SUPPORTED, LIMITED.** EXP-0006 is an accepted static
candidate-proxy engineering benchmark. It is not anatomical truth, a temporal
sequence benchmark, or a rescue of the rejected complete-frame hypothesis.

**Consequence.** Retain it for future gated crop work, but keep model-selection
claims anchored to Tier C until manual labels exist.

## H1 — Does intrinsic regression produce a reliable ordinary-frame proposal?

```text
same 2x2 encoder bottleneck
       /                                 \
direct 100-point coordinates             intrinsic 16-coefficient curve
208.66 px / 46.49 deg                    116.92 px / 23.35 deg
zigzag and >9,300 px length error        smooth but misplaced/scale collapsed
       \                                 /
        v                               v
                 both fail 4 px / 8 deg gates
```

**Test.** Compare only the output representation under the same data, fold,
encoder, budget, and evaluation identities.

**Visual evidence.** Random Tier C overlays show the
[`coordinate topology explosion`](../experiments/exp_0004_representation/results/coordinate_best/tier_C_random.png)
and the smoother but
[`mislocalized intrinsic predictions`](../experiments/exp_0004_representation/results/intrinsic_best/tier_C_random.png).

**Conclusion — NOT SUPPORTED.** Intrinsic geometry is structurally better, but
the shared proposal is nowhere near a reliable initializer.

**Consequence.** Reject coordinates. Preregister one intrinsic-only factor:
increase retained encoder spatial resolution from 2×2 to 4×4 without changing
the data, head, losses, or gates.

## H1b — Does a 4×4 spatial bottleneck rescue intrinsic localization?

```text
step 300: 102.15 px < 243.28 px frozen mean-pose baseline
                         |
                         v
continue exact checkpoint to step 1,200; select step 720
                         |
    primary effect: 87.54 px, 25.13% better than EXP-0004 [PASS]
    throughput: 2,461 samples/s, ratio 1.061              [PASS]
    ordinary reliability + visual shortcut               [FAIL]
                         |
                         v
PRIMARY_FOLD_FAIL: no folds, repeats, temporal branch, or holdout
```

**Test.** EXP-0007 changed only the retained grid, used an executable hashed
step-300 baseline, resumable exact training order, fixed 43-case fully-visible
Tier C gate, candidate proxies, synchronized GPU benchmark, and hash-bound
random/worst qualitative review.

**Result.** Fully-visible Tier C median/p95 point error was 87.54/233.24 px and
mean/p95-frame angle error was 27.68°/53.29° against 4/10 px and 8°/18° gates.
Candidate-proxy errors were worse. Endpoints and body length failed by large
margins. The qualitative review found systematic short, straight, displaced
predictions.

![Representative EXP-0007 failure](../artifacts/final_figures/representative_overlay.png)

![Worst controlled failures](../artifacts/final_figures/failure_montage.png)

**Conclusion — NOT SUPPORTED.** More retained spatial features improved one
localization scalar but did not fix the shortcut or scientific reliability.
This is an exact/qualitative failure, not a near-gate result.

**Consequence.** The deterministic decision forbids all expansion. The 2025
audited holdout remains unopened, and temporal/refinement/uncertainty hypotheses
remain blocked.

## Cross-cutting result — speed was never the limiting factor

![Accuracy-throughput comparison](../artifacts/final_figures/accuracy_throughput_pareto.png)

Every proposal exceeded the 20 fps acquisition floor by two orders of
magnitude in the common in-memory batch-32 harness. None approached the 4 px
ordinary-frame gate. The correct Pareto conclusion is not that EXP-0007 is a
fast final system; it is that accuracy, not compute, is the unresolved problem.
The benchmark excludes HDF5 reading and output serialization.

## Cropped-body and support branch

![Controlled crop robustness](../artifacts/final_figures/crop_robustness.png)

EXP-0007's independent-frame hidden-body error worsens at large crop fractions.
The moving-FOV angle heatmap shows the same failure under exact temporal crop
geometry without implying that a temporal model was tested.

![Controlled moving-FOV angle heatmap](../artifacts/final_figures/body_angle_heatmap.png)

Support probability had measurable calibration, but this is not pose
uncertainty:

![Image-support calibration](../artifacts/final_figures/support_calibration.png)

## Final path

```text
accepted audit / geometry / persistence infrastructure
                         +
rejected diagnostic checkpoint behind explicit opt-in
                         |
                         v
NO DEPLOYMENT-AUTHORIZED MODEL
                         |
                         v
collect 256 manual labels -> test one localization-explicit architecture
```

The full numeric record is in `docs/EXPERIMENT_LOG.md` and each EXP folder.
Consequential architecture choices are in `docs/DECISIONS.md`; limitations and
the annotation protocol are in `docs/FINAL_REPORT.md` and
`docs/ANNOTATION_RECOMMENDATIONS.md`.
