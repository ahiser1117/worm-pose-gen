# EXP-0001 — Classical high-confidence proxy baseline

## Hypothesis

Robust local-background subtraction followed by conservative component and
centerline quality checks can produce a useful set of easy, fully visible real
proxy labels despite cross-recording intensity shifts.

## Difference from baseline

This is the first scientific pose baseline. It uses no learned parameters and
must not be described as ground truth.

## Data/split

Uniformly sample at most 48 frames from each of the three development recordings
in `configs/split_manifest.json`. Do not open the audited-holdout recording.
Open one HDF5 reader at a time and read frames individually.

## Training/resource budget

- maximum steps/epochs: not applicable; deterministic classical processing
- wall-time limit: 20 CPU minutes
- seed/repeat policy: base seed 20260818; deterministic indices
- checkpoint cadence: not applicable
- expected GPU time: none
- expected external-storage use: <= 250 MiB under `datasets/worm_pose_gen/proxy_v1`
- early termination conditions: source write attempt, >1 reader, or a grossly invalid centerline convention

## Success criterion

- primary metric: conservative accepted-proxy yield per recording
- numeric practical-effect threshold: >=50% of sampled frames accepted in each
  development recording; every accepted pose has 100 uniformly spaced points,
  stays at least 3 px from the image boundary, has plausible 250–750 px length,
  has two stable endpoints, and has >=95% point support inside the foreground
  tube/dark-ridge agreement check
- variability/confidence rule: report per-recording Wilson intervals and all
  rejection reasons; inspect 24 deterministic random accepted cases plus every
  accepted case with the worst quality score
- pass/fail interpretation: ACCEPT only if no more than 2/24 random overlays
  are grossly off the anatomical midline and no worst case reveals a systematic
  topology defect; otherwise REVISE or REJECT

Foreground-tube and dark-ridge checks share preprocessing with the extractor;
they are internal QC, not independent accuracy evidence. The deterministic
human overlay audit is an independent qualitative check only. These proxies may
be training targets and engineering-consistency metrics, but never the sole
arbiter of scientific accuracy; Tier C controlled truth and clearly qualified
qualitative evidence remain separate.

## Results

The deterministic CPU run sampled exactly 48 uniformly spaced frames from each
development recording (144 total) and accepted 90 candidate proxy labels. The
audited holdout was not opened.

| Development recording | Accepted / sampled | Yield | Wilson 95% interval | Nonexclusive rejection reasons |
|---|---:|---:|---:|---|
| 2023-09-19-01 | 33 / 48 | 68.8% | [54.7%, 80.1%] | boundary 11; length 2; topology/endpoints 3 |
| 2023-09-27-01 | 26 / 48 | 54.2% | [40.3%, 67.4%] | boundary 10; length 1; ridge support 12; topology/endpoints 1 |
| 2023-10-11-01 | 31 / 48 | 64.6% | [50.4%, 76.6%] | boundary 16; ridge support 2 |

All 90 accepted candidates passed the executable contract check: 100 points,
two endpoints on the exported ordered backbone, 513.3--744.7 px measured
length, at least 13 px centerline boundary clearance (stricter than the frozen
3 px minimum), at least 99% foreground-tube support, and at least 95% local
dark-ridge support. Foreground-tube and ridge support are correlated internal
checks, not independent evidence of anatomical correctness.

Separate qualitative review of the 24 deterministically selected accepted
overlays found 0/24 grossly off the visible anatomical midline. The single
lowest-quality accepted case from each recording (three cases total) showed no
systematic branch, loop, or shortcut defect. This is a visual audit without
manual coordinates, not an accuracy measurement.

## Figures

- `figures/random_accepted_overlays.png`: the frozen 24-case random accepted
  overlay audit (red/blue denote export endpoints, not validated head/tail).
- `figures/worst_quality_overlays.png`: every per-recording minimum-quality
  accepted case.
- `figures/rejected_cases.png`: 24 deterministic rejected cases, including
  boundary, length, topology, and ridge-support rejection modes.
- `figures/angle_profiles.png`: unwrapped image-coordinate tangent-angle
  profiles for 12 of the random accepted cases; export orientation remains
  anatomically uncertain.

## Runtime

The full generation run took 99.176 s on CPU with Python 3.13.15, NumPy 2.5.2,
and h5py 3.16.0 at Git commit `19412beeb2b56bfc95f7d79f15820144c7ebd246`.
Per-recording extraction times were 21.897 s, 29.877 s, and 33.335 s; remaining
time includes figure creation, compressed HDF5 writing, validation metadata,
and hashing. No GPU was used.

Commands run from the repository root:

```bash
scripts/project_env.sh uv run --no-sync --frozen python -m unittest tests.test_classical -v
scripts/project_env.sh uv run --no-sync --frozen python scripts/generate_proxy_labels.py --samples-per-recording 48
scripts/project_env.sh uv run --no-sync --frozen python -m unittest discover -s tests -v
```

The accumulating proxy dataset is
`/temp_data4/alex/external_artifacts/datasets/worm_pose_gen/proxy_v1/proxy_labels.h5`
(external root resolves to `/storage/fs/temp_data4/alex/external_artifacts`)
(50,656,504 bytes; SHA-256
`1f891a38c4175e5d6e38d776fbb73c96fe2e2a796e4aecfd386e6ab9da34a11c`). It
contains only the three development-recording groups and stores accepted source
images, poses, sample indices, all acceptance flags/reasons, per-frame QC, and
source identity. Validation confirmed no audited-holdout group, finite accepted
poses, and every accepted contract check.

## Interpretation

The hypothesis is supported for conservative development-set proxy generation:
the point estimate exceeds 50% in every recording and the visual gate passes.
Evidence is weakest for 2023-09-27: its Wilson interval crosses 50%, and 12
frames fail the correlated local-ridge check. This is useful rejection behavior,
but it also shows that yield is sensitive to session appearance.

The proxy labels do not establish real-image accuracy. Static endpoint
appearance is unvalidated, so the stored head/tail confidence is deliberately
capped at 0.65 and must not be treated as calibrated. Angle profiles expose
some pixel-skeleton-scale roughness. Tight overlaps, severely cropped bodies,
and independent-background generalization remain unevaluated.

## Decision

ACCEPT (limited): retain these 90 cases as high-confidence Tier B candidate
training labels and engineering-consistency scaffolding. Do not use this
generator as the sole evaluator of a learned method, do not claim anatomical
head/tail truth, and do not generalize beyond the three development sessions.

## Next experiment

Use only accepted candidates as Tier B proxy labels; independently validate
geometry with the Tier C synthetic generator before any learned proposal.
