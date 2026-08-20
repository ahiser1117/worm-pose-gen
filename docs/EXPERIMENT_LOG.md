# Experiment Log

No scientific experiment IDs were issued before the reproducibility baseline
commit `0422168`. Phase 0 validated the environment and a two-step Lightning
data-to-GPU smoke path; it is infrastructure evidence, not a model result.

Scientific entries are added only after their hypothesis, budget, metric, and
numeric success criterion have been written in the corresponding experiment
directory.

## Phase 1–2 — Data forensics and evaluation freeze

A bounded 32-frame-per-recording audit found 4/12 readable recordings and
79,704 usable NIR frames at approximately 20 Hz. The exact findings and figures
are in `docs/DATA_AUDIT.md`. Geometry/metric tests and the output HDF5 contract
were established before model development. `configs/split_manifest.json`
freezes three whole-session development folds. The 2025 recording is an audited
holdout: its 32 pre-split audit frames are disclosed and excluded, and every
other frame remains unread until selection is frozen.

## Phase 3 — Proxy and controlled-geometry baselines

**EXP-0001 (ACCEPT, limited)** generated 90 conservative candidate proxy
centerlines from 144 uniformly sampled development frames (62.5%). Per-session
yield was 68.8%, 54.2%, and 64.6%; the 24-case deterministic visual audit found
no gross midline failures. All 90 are training/QC scaffolding rather than ground
truth; only the 24 independently reviewed overlays form even a limited
qualitative Tier B subset. Anatomical head/tail identity remains unvalidated. See
`experiments/exp_0001_classical_proxy/notes.md` and its accepted/rejected
overlay montages.

**EXP-0002 (ACCEPT)** validated the Tier C intrinsic generator, differentiable
tube renderer, and exact static/temporal FOV crops. All 6,400 crop cases passed;
the maximum coordinate round-trip residual was `3.41e-13 px`, and renderer
gradients were finite and nonzero. The synthetic appearance is deliberately
simple and cannot support a real-image accuracy claim. See
`experiments/exp_0002_synthetic_crop/notes.md`.

Together these experiments permit a small learned proposal comparison while
keeping evidence roles separate: real texture with candidate proxy geometry
versus exact geometry with simplified appearance.

**EXP-0003 (REJECT)** tested the first real-texture crop design: a literal
`256 x 192` source window. Only 65/900 requested conditions were geometrically
valid and 0/90 frames supported all ten required head/tail fractions, far below
the predeclared 60-frame gate. Exact transforms, pixel provenance, and support
all passed for the 65 emitted cases, so the failure is informative: the physical
window was too small to retain the long visible complement. EXP-0005 revises
the camera geometry while retaining the original gate.

**EXP-0005 (REJECT)** corrected physical scale and raised valid yield to
720/900, but only 14/90 frames supported a complete ten-condition series versus
the unchanged >=60 gate. The remaining failure is geometric re-entry: an
axis-aligned camera rectangle cannot always exclude exactly one contiguous
anatomical end of a curved candidate centerline. EXP-0006 changes the benchmark
unit explicitly and prospectively to a balanced crop condition, rather than
pretending the rejected same-frame hypothesis passed.

**EXP-0006 (ACCEPT)** materialized the preregistered condition-level revision:
300 immutable real-texture crops, exactly 10 in every recording/end/fraction
cell. All 300 source-window hashes, interpolation hashes, support mappings, and
transforms passed, with `5.68e-14 px` maximum round-trip error. The 26,206,788
byte atomic artifact has SHA-256
`57f104cc3a77ad0833257fdedadf153a03b73ac731656dca84de593319e0f849`.
Its rows reuse 87 source frames, and its centerlines remain candidate proxies;
acceptance establishes a balanced static engineering benchmark, not anatomical
accuracy, temporal truth, or success of EXP-0005's rejected same-frame claim.

**EXP-0004 (REJECT)** tested direct coordinates against a 16-coefficient
intrinsic proposal on the primary development fold. A corrected rerun separated
the frozen 43 fully-visible Tier C cases from 85 artificial crops after the
initial evaluator incorrectly mixed them. Direct coordinates produced a severe
zigzag/topology failure. Intrinsic geometry removed that failure and was faster
to learn, but its best fully-visible result was still 116.92 px median point
error and 23.35 degrees mean angle error versus 4 px and 8 degree gates. Both
were fast (intrinsic batch-1 end-to-end p50 1.34 ms; batch-32 2,320 samples/s),
but neither was reliable enough to advance beyond the cheap-elimination fold.
The evidence supports intrinsic structure but rejects the shared 2x2-bottleneck
proposal family.

**EXP-0007 (PRIMARY_FOLD_FAIL)** tested the single preregistered rescue factor:
retaining a 4x4 rather than 2x2 encoder grid in the intrinsic model. The
immutable step-300 checkpoint passed the cheap continuation rule (102.15 px
versus the 243.28 px frozen mean-centerline baseline), and the selected step-720
checkpoint improved the EXP-0004 primary-fold median by 25.13% while running at
2,461 batch-32 samples/s. Reliability nevertheless failed decisively:
fully-visible Tier C median/p95 point error was 87.54/233.24 px and
mean/p95-frame angle error was 27.68/53.29 degrees, versus 4/10 px and 8/18
degree gates. Candidate-proxy errors, both endpoints, body length, and the
frozen qualitative review also failed. Random/worst overlays show a systematic
short, straight, displaced mean-pose shortcut. See
`experiments/exp_0007_spatial_rescue/notes.md` and its hash-bound decision.

The exact qualitative failure disables the near-gate repeat rule. The
deterministic artifact authorizes neither additional folds nor repeat seeds and
sets both geometry acceptance and temporal authorization false. The audited
holdout remained unopened. The project therefore stops scientifically with no
accepted/deployable model; the retained checkpoint exists only for diagnostic
reproduction and explicit opt-in testing of the output scaffold.

## Follow-on localization and data controls

**EXP-004 analytic 5k control (PRIMARY CONTROLLED GATE FAIL)** increased the
deterministic analytic development set from 512 to 5,000 samples for the
unchanged topology-safe soft-anchor model. Optimizer steps rose from 1,200 to
10,800 to keep exposure near 34 dataset passes. The materialized 565-sample
parent prefix, 28 proxy-validation tensors, and held-out Tier-C-128 tensors all
reproduced their frozen hashes; six training and three validation proxy rows
were excluded under the existing Tier-A ±11-frame leakage manifest.

The primary seed completed on verified physical GPU 0. On the frozen 43 fully
visible Tier-C cases it reached 46.64 px median full-latent point distance,
25.17 degrees median mean tangent error, and 0.261 median body-length error
fraction versus gates of 16 px, 15 degrees, and 0.15. This improves point and
length error over EXP-003B but fails all three criteria. Repeat seeds and
real-texture synthesis were stopped; Tier A, delayed repeats, and the protected
holdout remained closed.

## Segmentation-anchored generative branch

**EXP-SMC-001 (NOT SUPPORTED)** audited a classical robust-dark-ridge soft
foreground baseline on the 30 primary development annotations plus declared
adjacent-frame diagnostics. Median/p10 visible-trace containment was
0.980/0.958, but median terminal containment was 0.800 versus 0.900 and the
uncalibrated soft trace score was 0.738 versus 0.800. No manual masks exist, so
these are trace-proxy diagnostics rather than Dice/IoU or network accuracy.

**EXP-SMC-002 (NOT SUPPORTED)** accepted 10/30 frames. Its seven accepted
complete anchors were conditionally accurate (6.756 px median point error and
4.634 degrees median tangent error), but only 6/7 were individually within
8 px and three of twelve truncated frames were falsely accepted. Their
20–23 px boundary clearances were smaller than their 38–49 px median widths,
identifying segmentation terminal omission as the false-completeness mechanism.

**EXP-SMC-001B (NOT SUPPORTED)** prospectively grew the high-confidence
component only through connected pixels above a frozen 0.25 low threshold.
Growth was small and stable (median/p95 0.97%/2.20% of seed area; adjacent-area
p95 1.44%), but terminal containment remained 0.800. Seventeen of thirty masks
now contacted the FOV, showing that the extra low-confidence pixels were not
selectively the omitted anatomical terminals.

**EXP-SMC-002B (NOT SUPPORTED)** paired the revised masks with a one-median-
width FOV clearance rule. It rejected all 12 truncated frames but retained only
3/17 complete anchors, only 2/3 of which were within 8 px; the remaining error
was 12.603 px. There were no accepted anchors in one development recording.
The result fixes the observed false-truncation symptom but fails the unchanged
conditional reliability gate and cannot support session-spanning dynamics.

**EXP-SMC-000 (CATALOG ONLY)** retained six development-only natural hard-bout
candidates from a bounded raw visual screen. Five are bracketed only by sparse
legacy proxy anchors, generally hundreds of frames away. Event types are
unadjudicated hypotheses, not truth or recovery outcomes.

Under the SMC plan's stopping rules, latent/width/dynamics/renderer/SMC
experiments were not authorized. The core upstream assumption failed for the
available foreground method: it does not preserve anatomical terminals well
enough to supply reliable full-body anchors. The protected 2025 holdout stayed
closed. Reopening requires a learned foreground method trained/evaluated with
fresh mask/terminal truth, followed by all frozen anchor-reliability gates and
adequate contiguous anchor density across sessions.

### Expert-authorized continuation and downstream gates

**EXP-SMC-002C (COMPLETE DEVELOPMENT CONTINUATION AUTHORIZED)** preserves the
frozen EXP-SMC-002B result while recording explicit expert adjudication of all
30 bound visual rows. Twenty-eight rows were judged development-adequate, row 2
(`2023-09-19-01-f017959`) was identified as a mistaken annotation, and row 22
(`2023-10-11-01-f013785`) was designated an expected SMC-hard case rather than
an easy anchor. This superseded D-0011 only as development governance; it did
not turn EXP-SMC-001B/002B into quantitative passes or alter their artifacts.

**EXP-SMC-002D (STATIC POSTURE EXTRACTION FEASIBLE; SESSION-GENERAL DYNAMICS
NOT FEASIBLE)** scanned three contiguous 101-frame development windows. Strict
acceptance was 22/101 (21.78%), 58/101 (57.43%), and 7/101 (6.93%), for 87/303
(28.71%) overall. The recordings contributed 7, 45, and 0 adjacent accepted
pairs; longest accepted runs were 5, 14, and 1 frames and longest rejected gaps
were 15, 29, and 39 frames. Every session had a static anchor, but one had no
adjacent pair and 45/52 pairs came from a single session. Static posture
extraction could continue; session-general empirical dynamics fitting could
not. Runtime was 765.33 s (0.396 frames/s).

**EXP-SMC-003 (SUPPORTED ORACLE ONLY)** retained the fixed 16-coefficient cubic
tangent spline plus translation, rotation, and length. On 17 complete traces,
its median/p95 per-frame point error was 0.697/1.074 px and median/p95 mean
tangent error was 3.305/4.307 degrees. The K=16 cosine basis also passed, but
the preregistered tie-break selected cubic; leave-one-recording-out PCA remained
diagnostic. This is known-trace representation capacity, not image inference or
natural dynamics evidence.

**EXP-SMC-004 (COMPLETED SUPPORTED PROXY ONLY)** found no need for particle-wise
width flexibility. The bounded-scale model passed its scripted proxy gate at
0.8663 median cleaned-mask IoU, while PCA-2 improved it by only 0.00038; fitted
scale SD was 0.0245 and a 10 px translated centerline still reduced median IoU
by 0.2041 after scale refitting. The least-complexity decision retained the
fixed recording-level mean width profile and left scale out of the initial
particle state.

**EXP-SMC-005 (COMPLETED SUPPORTED ENERGY SHAPE ONLY)** retained soft Dice as
the simplest passing observation energy. All 64 case/perturbation curves had a
minimum at zero or an adjacent grid point, overall outward-step monotonicity
was 0.9714, and all 20-value latent gradient groups were finite and nonzero.
At 183x242 on the RTX 6000 Ada, render plus energy took 7.914 ms for 32
particles. These are local proxy-basin and throughput results, not global
capture, probability calibration, or natural hard-case validation.

**EXP-SMC-006 (COMPLETED NOT SUPPORTED SESSION GENERAL)** confirmed the density
blocker and rejected H5. Adjacent-pair counts were 7/45/0 by recording;
five-frame predictions were 0/16/0 and no 20-frame case existed. On 40 paired
one-frame starts, persistence achieved 3.572 px median error versus 4.484 px for
the best non-persistence diagnostic, global translation/orientation velocity
with shape hold, which was 25.6% worse. A zero-drift block-diagonal random walk
was retained only as a declared synthetic control prior, never as an empirical
natural-motion estimate.

**EXP-SMC-007 (SUPPORTED CONTROLLED SYNTHETIC ONLY)** demonstrated bounded
algorithm execution on renderer-matched zero-drift synthetic random walks. At
128 particles and temperature 0.03, held-out nominal median trajectory error
was 2.19 px for forward SMC and 1.95 px after terminal reweighting, with 1.00
truth survival and median ESS fraction 0.89. All declared stress gates passed,
but exact two-anchor interpolation was best in all six scenarios and terminal
reweighting improved forward SMC in only 2/6. Thus neither general terminal-
anchor smoothing benefit, SMC superiority, nor H8 is supported, and no natural
SMC claim is authorized.

**EXP-SMC-008A (NO BOUNDED STRICT ANCHOR BRACKET)** applied the unchanged final
segmentation and strict-anchor configs to `2023-10-11-01` frames 13685-13885,
centered on expert hard row 22 at frame 13785. It accepted 0/201 frames: there
was no strict anchor before or after the hard frame, no accepted run, and one
201-frame rejected gap. All 201 frames triggered `branch_pixels` and `cycle`;
183 also triggered abrupt/implausible width checks and 150 low render IoU.
Runtime was 564.93 s (0.356 frames/s). No pose was inferred. The selected
natural case therefore cannot test a <=20-frame two-anchor bout under the
current pipeline: the branch's nearby-reliable-anchor assumption fails at its
designated hard case before natural SMC can be evaluated.

All EXP-SMC-002C through EXP-SMC-008A work remained development-only. The
protected 2025 holdout was not opened.
