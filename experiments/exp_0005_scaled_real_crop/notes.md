# EXP-0005 — Scaled real-texture camera windows

## Hypothesis

A larger, variable-size 4:3 source camera window followed by an isotropic resize
to the fixed `256 x 192` network input can preserve real NIR texture, exact
support, and a complete 5--40% head/tail crop series for at least 60 candidate
proxy frames.

## Difference from baseline

EXP-0003 incorrectly treated `256 x 192` network pixels as the physical source
camera-window size and retained only 65/900 conditions. EXP-0005 searches direct
source subwindows of size `(4k, 3k)` with `96 <= k <= 240`, chooses the smallest
valid window deterministically, and then resizes it isotropically. The output
contains only resampled pixels from that direct source window: no padding,
painting, compositing, or synthetic background.

## Data/split

Use only the immutable accepted candidate images/centerlines in `proxy_v1` with
the frozen SHA-256. Never open source recordings or the audited holdout. Preserve
recording/frame identity and attempt both ends at exactly 5%, 10%, 20%, 30%, and
40% hidden. Record every valid/rejected transform and support mask.

## Training/resource budget

- maximum steps/epochs: not applicable; deterministic transformation
- wall-time limit: 15 CPU minutes
- seed/repeat policy: base seed 20260818; deterministic source/request order
- checkpoint cadence: not applicable
- expected GPU time: none
- expected external-storage use: zero unless a <=2 GiB atomic cache is needed
- early termination conditions: immutable proxy mismatch, source-recording open,
  non-isotropic mapping, generated/padded output pixels, or support mismatch

## Success criterion

- primary metric: complete valid real-texture crop frames
- numeric practical-effect threshold: at least 60/90 frames permit all ten
  conditions; all 900 requests are reported, every emitted support mask exactly
  matches independent half-open recomputation, source->crop->source point error
  is <=1e-5 px before resize and <=1e-4 px after resize, and output pixels arise
  only from the declared direct source window under the frozen interpolation
- variability/confidence rule: report complete/valid yield by recording and
  condition, source-window scale distribution, deterministic-repeat checks, and
  inspect random, maximum-hidden, smallest/largest-scale, and rejected examples
- pass/fail interpretation: ACCEPT only as candidate-proxy-referenced controlled
  crop geometry. It is not anatomical accuracy evidence

## Results

Pending.

## Figures

Pending crop evidence, yield, and scale-distribution plots.

## Runtime

Pending.

## Interpretation

Pending.

## Decision

INCONCLUSIVE

## Next experiment

If accepted, use the frozen case manifest for proposal and temporal cropped-FOV
comparisons; otherwise retain Tier C crop evidence and report real crop evidence
as unavailable rather than weakening the gate.
