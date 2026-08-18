# EXP-0006 — Balanced real-texture crop benchmark

## Hypothesis

The valid EXP-0005 conditions can support a balanced, deterministic benchmark
covering every recording, anatomical end, and 5--40% fraction without changing
crop geometry, silently dropping a condition, or claiming the rejected
same-source-frame series succeeded.

## Difference from baseline

EXP-0005's experimental unit was a source frame with all ten valid conditions;
only 14/90 passed. EXP-0006 prospectively changes the unit to one valid crop
condition. From the frozen 720-case valid pool, it selects exactly 10 cases per
recording for each end/fraction cell: 3 recordings x 2 ends x 5 fractions x 10
= 300 cases. Selection rank is SHA-256 of the frozen seed, recording, frame, end,
and fraction, independent of model performance or proxy quality score.

## Data/split

Use only EXP-0005's immutable 900-case manifest and the immutable `proxy_v1`
accepted-image artifact. Verify both SHA-256 values. Never open source recordings
or the audited holdout. Regenerate only the selected direct source windows and
frozen isotropic resize, retaining exact original/source-window/output mappings.

## Training/resource budget

- maximum steps/epochs: not applicable; deterministic selection/materialization
- wall-time limit: 10 CPU minutes
- seed/repeat policy: seed 20260818; SHA-ranked deterministic selection
- checkpoint cadence: not applicable
- expected GPU time: none
- expected external-storage use: <=250 MiB in `real_crop_balanced_v1`
- early termination conditions: fewer than 10 valid cases in any recording/cell,
  input hash mismatch, output/source collision, generated pixels, transform or
  support mismatch, partial publication, or overwrite attempt

## Success criterion

- primary metric: balanced artifact contract
- numeric practical-effect threshold: exactly 300 completed cases, exactly 10
  per recording/end/fraction cell, zero duplicate condition identities, 100%
  hash/provenance/support/transform/interpolation checks, <=1e-4 px transform
  round trip, complete atomic HDF5 marker, and <=250 MiB output
- variability/confidence rule: contract is exhaustive over all selected cases;
  report source-frame reuse and inspect deterministic random, all 40%-hidden,
  per-recording, and min/max-scale examples
- pass/fail interpretation: ACCEPT establishes a candidate-proxy-referenced
  static real-texture crop benchmark only. It does not restore the rejected
  same-frame claim, establish anatomical accuracy, or supply real temporal truth

## Results

All frozen inputs matched their preregistered identities: EXP-0005 metrics
SHA-256 `b1d87dc020932fe6059ff52ffc5ca2e435baa63e502c77ad6b147ee9cdd0178c`,
EXP-0005 case-manifest SHA-256
`db097249b25182864fa6e6dbd38b494a97c42846f96ccf4f4145948ceba06bcc`,
and proxy SHA-256
`1f891a38c4175e5d6e38d776fbb73c96fe2e2a796e4aecfd386e6ab9da34a11c`.

- SHA ranking selected exactly 300 valid crop-condition identities: exactly 10
  in every one of the 30 recording/end/fraction cells and zero duplicates. The
  tightest source pool contained 11 valid cases, so no cell or threshold was
  relaxed. Selection-manifest SHA-256 is
  `575e0ac2e6d276651ee8fd3887b4e3116f2aaa06c4668e3f86da91976e6d1d35`.
- All 300 accepted-image hashes, direct-window hashes, frozen interpolation
  hashes, source/output half-open supports, and forward/inverse transforms
  passed regeneration checks. Maximum resized-to-source round-trip error was
  5.684e-14 px (limit 1e-4 px).
- The atomically published HDF5 has schema version 1, `complete=true`, exactly
  300 rows, all required images/centerlines/support/identity/window/transform/
  hash datasets, and no remaining partial. It is 26,206,788 bytes, below the
  262,144,000-byte gate. Artifact SHA-256 is
  `57f104cc3a77ad0833257fdedadf153a03b73ac731656dca84de593319e0f849`.
- The selection covers 87 unique source frames. A frame supplies between one
  and seven selected conditions; reuse counts for 1--7 conditions are 7, 16,
  28, 15, 11, 8, and 2 frames. The selected k range is 96--132 with median 96.

## Figures

`figures/balance.png` confirms 10 cases in every cell.
`figures/source_reuse.png` shows the reuse distribution and unique source
coverage (32, 24, and 31 frames by recording).
`figures/random_and_scale_evidence.png` shows deterministic random cases and
the maximum/minimum resize-scale extremes. `figures/all_40_percent_cases.png`
shows all 60 selected 40%-hidden cases across recordings and ends.

Visual inspection confirmed real NIR texture, consistent bilinear resizing,
clean crop truncation, varied poses/backgrounds, and coherent min/max-scale
geometry. All 60 maximum-hidden panels were present and legible. The balance
heatmap exactly matches the recorded cell counts, and the reuse plots match the
87-frame distribution. Overlays are figure-only diagnostics and are not part
of the stored images.

## Runtime

CPU only. The complete selection, one-handle proxy regeneration, atomic HDF5
publication, exhaustive published-artifact validation, hashing, and figure
generation took 9.80 seconds, below the 10-minute limit. No source recording or
audited holdout was opened.

## Interpretation

The condition-level hypothesis is supported: the valid EXP-0005 pool can form a
fully balanced and reproducible static real-texture crop artifact without
altering geometry or omitting a declared cell. This changes the experimental
unit to individual crop conditions and does not reverse EXP-0005's rejection of
the same-source-frame ten-condition series. Source reuse means the 300 rows are
not 300 independent biological frames. Endpoint order and centerlines remain
candidate proxies, so the artifact is neither anatomical accuracy evidence nor
real temporal truth.

## Decision

ACCEPT — every preregistered balanced-artifact gate passed. Freeze the published
artifact for candidate-proxy-referenced static crop diagnostics only; retain
Tier C truth for quantitative hidden-anatomy claims.

## Next experiment

Use this artifact as a frozen candidate-proxy crop stratum for model diagnostics,
while all quantitative hidden-anatomy claims continue to rely on Tier C truth.
