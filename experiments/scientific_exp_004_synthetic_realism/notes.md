# EXP-004 — Synthetic realism, analytic 5k scale control

Status: `ANALYTIC_5K_PRIMARY_CONTROLLED_GATE_FAIL`.

EXP-003 and EXP-003B are closed negative results. This control does not rerun
or tune either failed implementation. It asks the next data-scale question
with the unchanged topology-safe soft-anchor model before any real-texture
generator is introduced.

## Preregistered intervention

The only scientific intervention is analytic development diversity: 512 to
5,000 deterministic Tier-C samples. The same 53 leakage-safe candidate-proxy
training rows remain, so their mixture fraction falls from about 9.4% to 1.0%.
Optimizer steps rise from 1,200 to 10,800 to hold dataset exposure nearly
constant (33.33 versus 34.18 approximate passes), rather than under-training
the larger dataset. Fold 2, data/model seed 20260819, batch size 16, learning
rate 3e-4, float32 precision, image geometry, losses, and architecture remain
fixed. The model source is byte-identical to EXP-003B and has 882,242
parameters.

The complete preregistration is
`configs/scientific_exp_004_analytic_5k_control.yaml`, SHA-256
`01fb976198ea704896aa9de9e83162fa6a7c830c2e627023904d83ed029d3e28`.
The fixed primary budget is 10,800 optimizer steps, at most 40 epochs, and a
30-minute fail-safe on physical GPU 0 only.

## Frozen evidence and provenance

- candidate-proxy HDF5 SHA-256:
  `1f891a38c4175e5d6e38d776fbb73c96fe2e2a796e4aecfd386e6ab9da34a11c`;
- Tier-A ±11-frame exclusion manifest SHA-256:
  `3eccdb08121a79f5aa4515d9eb67d57aea656a121ea023c653675edc52427854`;
- excluded accepted rows: six training and three validation;
- materialized proxy-validation SHA-256:
  `9764ff4e28b4b8f2587820e393b8f2f61450ac39c9cc22004cd9f3ba26963b67`;
- frozen held-out Tier-C-128 SHA-256:
  `e7b881c4cc7037f94b64371c1e414906f825f10682793fc0580f225d2adc9df3`;
- expected materialized 565-sample EXP-003B prefix SHA-256:
  `2d40732c5119cc3c3dba7e1d24fb82f16115d903a0e04dc369900fb827cf4c07`.

The proxy embeds an older split-manifest byte hash while the current manifest
has evolved. The protocol acknowledges this rather than treating the hashes as
equal, then verifies the unchanged semantic fold: Sep-19 and Sep-27 train;
Oct-11 validates. Before optimization, the runner must write an immutable
materialization artifact containing the new 5,053-sample training tensor hash,
the unchanged validation hashes, source hashes, counts, exclusion counts, and
physical GPU UUID/PCI identity.

## Frozen controlled gate

The exact same 43 fully visible cases within held-out Tier-C-128 are primary.
All three criteria must pass conjunctively:

| Metric | Required |
|---|---:|
| Median full-latent point distance | ≤16 px |
| Median mean tangent error | ≤15° |
| Median body-length error fraction | ≤0.15 |

A primary failure stops the control. A primary pass authorizes only seeds
20260820 and 20260821, each with identical provenance and thresholds. Only a
separate all-three-pass artifact may authorize one primary Tier-A evaluation.
Tier A cannot be used for training, tuning, checkpoint selection, or reranking.
Delayed repeats and the protected holdout remain closed regardless of this
control's result.

## Pre-run verification

Twenty-six focused CPU tests passed, including exact config rejection,
hash/count reconstruction of frozen Tier C, finite conjunctive gate logic,
repeat-seed authorization, topology, exclusions, metrics, and physical-GPU
identity parsing. Parent config, run, gate, shared model sources, split
semantics, and all declared hashes validated before launch. No Tier-A values,
delayed repeats, or protected-holdout frames were read.

The complete repository suite then passed 149/149 tests. CUDA preflight verified
that exactly one device was visible and mapped logical `cuda:0` to physical GPU
0, UUID `GPU-f72d2ba7-8334-183e-e368-2c527e8a39e6`, PCI
`00000000:01:00.0`.

## Primary run and result

The primary seed completed exactly 10,800 steps. Materialization took 188.76 s
and training plus validation took 451.99 s (23.89 optimizer steps/s). The
pre-optimization artifact froze 5,053 training tensors with SHA-256
`3020af947684c08ec9483b554562c18e64b08695db1b4c27d94afc8e60b65f23`.
Its parent prefix, proxy validation, and held-out Tier-C hashes all matched the
preregistration. Peak allocated GPU memory was 458,805,760 bytes.

The fixed final checkpoint SHA-256 is
`140639360e5a509dcd812b91e64b3c2644edca85799e78ad3bb22a730aa78769`.
The immutable run metrics SHA-256 is
`f2ce75bbfd15dd2c01532a023491077a124ec68b93367eecc230bfe8370944ce`;
the pre-optimization materialization record is
`c1ac072aff02ef673c1d213f366b824596a28e9be71b74a582bc28333e60a1ff`.

| Fully visible Tier-C metric | Required | Observed |
|---|---:|---:|
| Median full-latent point distance | ≤16 px | 46.64 px |
| Median mean tangent error | ≤15° | 25.17° |
| Median body-length error fraction | ≤0.15 | 0.261 |

Decision: `PRIMARY_CONTROLLED_GATE_FAIL_KEEP_ALL_PROTECTED_EVIDENCE_CLOSED`.
The gate artifact SHA-256 is
`85659ee8ff3e2ed695e94ada123f6ebc3e4cfd064357faf455f98b60b61141e0`.

Relative to EXP-003B, the 5k control reduced median point error by 24.77% and
length error by 23.33%, but tangent error by only 2.31%. All three absolute
gates still failed. More analytic diversity and exposure help localization and
length but do not make the unchanged decoder reliable even on controlled
analytic imagery. Real-texture synthesis is therefore not authorized yet: an
appearance change cannot explain or repair failure on the analytic truth it
would use as a geometry control.

Seeds 20260820 and 20260821 were not run. The 30 Tier-A annotations were not
evaluated or used for tuning, delayed repeats were not accessed, and the
protected holdout remains closed. A subsequent experiment must be separately
preregistered and restricted to Tier C until it earns a controlled gate.
