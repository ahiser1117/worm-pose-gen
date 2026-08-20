# EXP-003 — Localization-preserving architecture comparison

Status: `COMPLETE_REJECT_IMPLEMENTATIONS`.

## Evidence boundary and paired design

The global, dense-field, and anchored-grid models were trained for the frozen
three seeds in `configs/scientific_exp_003_localization.yaml`. All nine runs
used 1,200 optimizer steps, the same 565 materialized training tensors, the same
156 validation tensors, and the same sample order within each seed. The
hash-bound exclusion removed six training and three validation candidate-proxy
rows inside the 636 recording/frame Tier-A neighborhoods. Tier-A labels were
evaluation-only; delayed repeats and the protected holdout were not opened.

The completed matrix shares config SHA-256
`adf0ba5cd9fd9cc28e37e84ccf50132c8db57ee73e862d20976651eb05305851`
and training-tensor SHA-256
`2d40732c5119cc3c3dba7e1d24fb82f16115d903a0e04dc369900fb827cf4c07`.
The aggregate fails closed unless the materialized tensors, training source,
checkpoint hashes, and per-seed training-order hashes agree.

## CUDA correction and compute audit

An initial global run was interrupted at step 389 before publishing run metrics
because the deterministic renderer regenerated synthetic images on CPU every
epoch. It used more than 100 CPU cores while the GPU waited. That incomplete
checkpoint is not evidence and was not resumed or included.

The final matrix materialized and hashed all tensors once per process, capped
PyTorch CPU work at four threads, used pinned-memory transfers, and ran on
physical GPU 0, an NVIDIA RTX 6000 Ada. A 100 ms utilization trace during the
optimizer loop showed sustained 10–13% SM utilization, 2.5 GHz SM clock, about
90 W, and 1,063 MiB resident. The models contain only 0.72–0.84 million
parameters, so low absolute occupancy is expected. All nine complete runs took
459.15 seconds of train/validation time plus 261.25 seconds of one-time tensor
materialization. Median throughput was 29.67 steps/s for global, 21.16 for
dense, and 21.39 for anchored.

## Frozen Tier-A result

The values below are medians across the three fixed seed-level medians. Each
seed evaluated the same 17 complete and 12 truncated primary traces; all models
returned finite curves for all 30 frames.

| Architecture | Complete point error (px) | Complete tangent error (deg) | Truncated visible distance (px) | Seeds passing 30% gate |
|---|---:|---:|---:|---:|
| Global intrinsic | 72.36 | 59.78 | 64.81 | reference |
| Dense centerline field | 83.10 | 81.97 | 11.21 | 0/3 |
| Anchored intrinsic grid | 113.22 | 54.50 | 33.72 | 0/3 |

Dense varied from a 10.7% complete-trace improvement in seed 20260819 to 19.1%
and 52.7% degradations in seeds 20260820 and 20260821. The paired bootstrap
intervals either crossed zero or favored the global reference. Anchored was
worse in all three seeds. No spatial implementation passed the frozen gate.

The overlays explain the superficially strong truncated dense metric: dense
predictions often cover the visible worm but traverse it back and forth. That
lowers the one-way visible distance while producing extreme body length and
tangent error. Anchored predictions frequently choose the wrong hard grid cell.

## Controlled Tier-C diagnosis

The same checkpoints were evaluated on the byte-identical 128-case held-out
Tier-C tensors (43 fully visible, 85 cropped). Median-of-seed median full-latent
point error was 59.64 px for global, 31.33 px for dense, and 65.79 px for hard
anchored inference. Dense's apparent localization gain came with a 7.56 median
body-length error fraction, confirming an ordering/topology failure under exact
synthetic truth. Anchored soft-mixture error was 59.41 px versus 65.79 px for
hard selection, so hard/soft mismatch contributes but does not explain the
overall failure.

A decoder-bound diagnostic confirmed that changing dense soft expectation to
per-channel hard heatmap argmax does not solve topology: the median-of-seed
median point error is 34.05 px and body-length error fraction remains 5.95.
For anchored grids, truth-only oracle candidate selection improves the hard
65.79 px result to 44.72 px (25.06 degrees tangent error and 0.27 length error).
Thus learned cell selection is faulty, but the candidate set itself is also not
accurate enough.

This evidence rejects the hypothesis **for these decoder/training
implementations**. It does not show that spatial features are unhelpful.
Failures occur on controlled truth, so resolution ablation and model-size
scaling are not authorized. The next bounded experiment should first require an
ordered dense decoder (or explicit arc-length/tangent topology constraint) and
hard-aligned per-cell anchored supervision to pass a Tier-C topology gate.
Only then should a new model be evaluated once on Tier A. Real-texture/domain
experiments remain relevant after the controlled decoder failure is fixed.

## Artifacts

- `results_materialized_v1/metrics.json`: paired Tier-A metrics, bootstrap
  intervals, checkpoint identities, and training provenance.
- `results_materialized_v1/architecture_comparison.png`: seed-level comparison.
- `results_materialized_v1/error_by_body_position.png`: complete-trace error.
- `results_materialized_v1/primary_seed_overlays.png`: qualitative failure audit.
- `results_materialized_v1/tier_c_diagnostic.json`: initial exact-truth diagnostic.
- `results_materialized_v1/tier_c_decoder_diagnostic_v2.json`: dense argmax and
  anchored oracle-candidate bounds.
