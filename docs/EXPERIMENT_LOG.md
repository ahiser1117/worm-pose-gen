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

**EXP-0001 (ACCEPT, limited)** generated 90 conservative Tier B proxy
centerlines from 144 uniformly sampled development frames (62.5%). Per-session
yield was 68.8%, 54.2%, and 64.6%; the 24-case deterministic visual audit found
no gross midline failures. These labels are training/QC scaffolding rather than
ground truth, and anatomical head/tail identity remains unvalidated. See
`experiments/exp_0001_classical_proxy/notes.md` and its accepted/rejected
overlay montages.

**EXP-0002 (ACCEPT)** validated the Tier C intrinsic generator, differentiable
tube renderer, and exact static/temporal FOV crops. All 6,400 crop cases passed;
the maximum coordinate round-trip residual was `3.41e-13 px`, and renderer
gradients were finite and nonzero. The synthetic appearance is deliberately
simple and cannot support a real-image accuracy claim. See
`experiments/exp_0002_synthetic_crop/notes.md`.

Together these experiments permit a small learned proposal comparison while
keeping evidence roles separate: real texture with imperfect proxy geometry
versus exact geometry with simplified appearance.
