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
