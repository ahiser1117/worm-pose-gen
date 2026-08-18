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
freezes whole-session train/validation/final-test groups; the 2025 final-test
session remains unopened for all model-selection work.
