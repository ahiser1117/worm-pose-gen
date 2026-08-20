# EXP-003B — Topology-safe soft-anchor rescue

Status: `PRIMARY_CONTROLLED_GATE_FAIL`.

This bounded follow-up was preregistered after the complete EXP-003 overlays and
Tier-C diagnostics identified two implementation failures: independently
decoded dense points violated curve order, and anchored inference selected a
hard grid cell that did not match its soft training path.

The rescue used a 12×16 soft anchor at both training and inference, spatially
pooled features at that anchor, and one 16-coefficient intrinsic curve with an
explicit length loss. This construction guarantees a single ordered curve and
has no hard-cell selection transition. The primary controlled gate was frozen
in `configs/scientific_exp_003b_topology_rescue.yaml` (SHA-256
`9f3cbd0fcd7fb24243b4380947687473e3435ce9e630554eadf7a9a513ab9a98`)
before training.

The primary seed 20260819 trained for all 1,200 CUDA steps on the same 565
materialized samples and nine proxy-row exclusions as EXP-003. It used 882,242
parameters and ran at 21.20 optimizer steps/s. The frozen held-out Tier-C set
contained the expected 128 cases and 43 fully visible cases with tensor SHA-256
`e7b881c4cc7037f94b64371c1e414906f825f10682793fc0580f225d2adc9df3`.

## Preregistered gate

| Fully visible Tier-C metric | Required | Observed |
|---|---:|---:|
| Median full-latent point distance | ≤16 px | 61.99 px |
| Median mean tangent error | ≤15° | 25.77° |
| Median body-length error fraction | ≤0.15 | 0.34 |

Decision: `REJECT_WITHOUT_TIER_A`. The two repeat seeds were not authorized,
the 30 primary annotations were not evaluated, delayed repeats were not used,
and the protected holdout remains closed.

Guaranteeing curve order and removing hard-cell selection was insufficient.
Combined with the held-out analytic failure, the next justified question is
training diversity/scale and then realistic appearance, as specified by
EXP-004. A new data experiment should start at the plan's 5k analytic control
and remain Tier-C-gated before adding real-texture synthesis or returning to
Tier A.

Gate artifact:
`experiments/scientific_exp_003_localization/results_materialized_v1/exp_003b_primary_tier_c_gate.json`
(SHA-256 `7086bb672e3646ed02c8e3bc12d2f0fb63f04ac7f7dbf3f9294dcb94822c1f77`).
