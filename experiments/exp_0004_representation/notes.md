# EXP-0004 — Coordinate versus intrinsic proposal representation

## Hypothesis

A compact intrinsic tangent-angle representation reduces local angle error and
jitter relative to predicting 100 independent centerline coordinates, without
materially worsening point accuracy or throughput.

## Difference from baseline

Both variants use the same small 2-D encoder, normalization, training examples,
augmentation, optimizer budget, and symmetric orientation loss. Only the output
representation/reconstruction changes: direct normalized coordinates versus a
midbody anchor, body length, global orientation, and 16 smooth tangent-basis
coefficients.

## Data/split

Train on accepted candidate proxy frames from the two training recordings of
each frozen development fold plus deterministic development-profile Tier C
samples. Validate separately on the held-out development-recording candidates
and disjoint held-out Tier C geometry/appearance. Only the 24 independently
reviewed overlays form a qualified qualitative Tier B subset; the remaining
candidates are training/engineering evidence. Do not read the audited holdout.
EXP-0003 real-texture crops may be evaluated only as a separate candidate-
proxy-referenced stratum.

## Training/resource budget

- maximum steps/epochs: 1,200 optimizer steps per variant/fold, at most 20 epochs
- wall-time limit: 12 GPU minutes per run; six primary runs
- seed/repeat policy: seed 20260818 for all three folds; if any one-seed metric
  or practical effect lies within 10% of its gate, also run seeds 20260819 and
  20260820 on every fold and require every seed/fold to pass
- checkpoint cadence: every 300 steps; retain latest and best validation only
- expected GPU time: <=1.5 hours including smoke/benchmark
- expected external-storage use: <=3 GiB in checkpoints and experiment outputs
- early termination conditions: non-finite loss/gradient, source HDF5 open,
  split identity mismatch, or median Tier C error worse than a centerline-mean
  predictor after 300 steps

## Success criterion

- primary metric: fold-aggregated held-out Tier C tangent-angle MAE
- numeric practical-effect threshold: ACCEPT intrinsic only if angle MAE is at
  least 5% lower in the pooled three-fold paired comparison, improvement is
  positive on every fold, candidate-proxy median point error is no more than 5%
  worse on every fold, and CUDA throughput is at least 90% of the coordinate
  variant
- proposal reliability gate: selected variant must achieve held-out Tier C
  median point error <=4 px, p95 point error <=10 px, mean tangent-angle error
  <=8 degrees, and p95 per-frame angle MAE <=18 degrees on every fold, with no
  systematic shortcut/topology failure in frozen random/worst overlays before
  refinement or probabilistic phases begin
- variability/confidence rule: pair variants by identical validation seed/case;
  pool case-level errors only after reporting each fold. Use a deterministic
  2,000-resample within-recording paired frame bootstrap as diagnostic. It does
  not replace the all-fold rule. If the 5% effect decision changes across the
  bootstrap, mark INCONCLUSIVE and retain the simpler/faster variant
- pass/fail interpretation: representation preference is controlled Tier C
  plus candidate-proxy engineering evidence, never a manual-label real-accuracy
  claim

## Results

The first two-step CUDA smoke reached training after strict preflight and both
validation loaders, then failed before its first optimizer update because
`adaptive_avg_pool2d_backward_cuda` has no deterministic implementation in the
installed PyTorch. No metric or checkpoint from that invalid smoke was used.
The fixed-size encoder pool was replaced by an equivalent fixed average pool,
and the corrected smoke passed before the primary runs.

The original evaluator then exposed a protocol implementation defect: it mixed
85 artificially cropped Tier C cases into the ordinary fully-visible gate,
contrary to the frozen `docs/EVALUATION.md`. The same checkpoints were rerun
after separating 43 fully-visible cases from 85 cropped cases. The invalid
mixed outputs remain in external storage; all results below use the corrected
strata and record checkpoint path, digest, and step.

Primary fold 2 reached the frozen 20-epoch cap. Because a periodic mid-epoch
resume is not exactly resumable, the retained latest periodic checkpoint is
step 600 rather than the nominal 720-step epoch total. The best coordinate
checkpoint is step 576, SHA-256
`7757ded321d16bdbc35e8aa26a119d3bb80fd7757d6982d5054c020f6e596843`;
the best intrinsic checkpoint is step 360, SHA-256
`e780d318b82b4ea529a8b6badfe7c980f1ac9bf992395fcd92fad2b2da75d62b`.

| Corrected primary-fold metric | Coordinate | Intrinsic | frozen gate |
|---|---:|---:|---:|
| fully-visible Tier C median point error | 208.66 px | 116.92 px | <=4 px |
| fully-visible Tier C p95 point error | 382.69 px | 275.84 px | <=10 px |
| fully-visible Tier C mean angle MAE | 46.49 deg | 23.35 deg | <=8 deg |
| fully-visible Tier C p95 frame angle MAE | 69.53 deg | 31.59 deg | <=18 deg |
| candidate-proxy median point error | 142.12 px | 72.00 px | <=8 px |
| candidate-proxy mean angle MAE | 66.06 deg | 55.81 deg | <=15 deg |

The retained step-600 intrinsic state improved fully-visible median point error
to 108.06 px but worsened mean angle to 25.02 degrees; it still failed every
geometry gate. Direct coordinates developed a decisive topology defect: mean
body-length error exceeded 9,300 original-image pixels because independently
predicted points formed a high-frequency zigzag. Intrinsic reconstruction
prevented that defect, but regressed toward mean location/length and remained
far from a reliable proposal. Neither variant advanced to the other two folds.

The preregistered early-elimination phrase "centerline-mean predictor" was not
fully specified or implemented. Post-hoc diagnostic definitions put the
baseline median above 328 px, so both models beat it at step 300; the ambiguous
clause was therefore not used to terminate or accept a run. EXP-0007 freezes an
executable mean artifact and comparison before training.

## Figures

The repository evidence bundles contain corrected metrics, random/worst
candidate-proxy and Tier C overlays, and body-position/error/FOV diagnostics in
`results/coordinate_best/` and `results/intrinsic_best/`. Random and worst
overlays show that coordinate outputs are jagged and poorly localized, while
intrinsic outputs are smooth but frequently centered away from the worm or too
short. These visuals agree with the point, angle, and length metrics.

## Runtime

Each first 300-step CUDA segment took about two minutes; continuation to the
retained step-600 state remained well below the 12-minute per-run limit. On the
specified RTX 6000 Ada physical device 0, synchronized best-checkpoint
benchmarks measured coordinate/intrinsic batch-1 end-to-end p50 latency of
1.01/1.34 ms and batch-32 throughput of 2,418/2,320 samples/s. Both easily clear
the 20 fps runtime target, but speed cannot compensate for failed accuracy.

## Interpretation

The intrinsic representation is structurally preferable to independent points,
but EXP-0004 does not establish an acceptable proposal or a valid all-fold
representation win. The shared 2x2 spatial bottleneck is a plausible cause of
the severe localization/scale underfit. Because the intrinsic head already
removes the coordinate topology failure, the next experiment changes only
spatial bottleneck resolution rather than repeating a two-variant sweep.

## Decision

REJECT — neither frozen variant passes the primary-fold ordinary-frame gate, so
the experiment stops before all-fold comparison, refinement, temporal, or
probabilistic phases.

## Next experiment

EXP-0007: retain intrinsic geometry and change only the encoder pool from 2x2
to 4x4, with an executable frozen mean-centerline early-elimination baseline
and exact epoch logging. Continue to temporal context only if the revised
proposal passes the unchanged gates.
