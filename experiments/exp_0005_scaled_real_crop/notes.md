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

The immutable proxy SHA-256 matched the frozen configuration. All 90 accepted
candidate-proxy frames and all 900 preregistered end/fraction requests were
evaluated in deterministic recording/frame/request order.

- 720/900 requests (80.0%) yielded valid exact-support scaled crops, a large
  improvement over EXP-0003's 65/900 direct 256 x 192 source crops.
- Only 14/90 frames yielded all ten conditions, with recording counts 4/33,
  7/26, and 3/31. This fails the unchanged >=60-frame practical-effect gate.
- Valid head/tail counts at 5%, 10%, 20%, 30%, and 40% hidden were 80/78,
  81/77, 76/75, 54/81, and 37/81 respectively. All 180 rejected requests had
  `no_exact_support_window_any_scale`: the visible complement fit within the
  maximum window, but no integer window at any allowed scale excluded exactly
  the requested contiguous end samples without extra support changes.
- The smallest valid source-window `k` ranged from 96 to 132, with median 96;
  428/720 valid requests used the minimum `k=96`. Each request chose its first
  valid k and a deterministic centered integer origin.
- All 720 emitted crops passed direct source-window equality, frozen bilinear
  `align_corners=False` recomputation, independent half-open support checks in
  source and resized coordinates, and deterministic search/interpolation
  repeats. Maximum source-window round-trip error was 0.0 px (limit 1e-5) and
  maximum resized round-trip error was 5.684e-14 px (limit 1e-4).
- `metrics.json` embeds all 900 valid/rejected cases with source identity and
  accepted-image hash, requested/actual support masks, source bounds, k,
  isotropic scale, forward/inverse transforms, source-window/output hashes, or
  rejection reason. Its case-manifest SHA-256 is
  `db097249b25182864fa6e6dbd38b494a97c42846f96ccf4f4145948ceba06bcc`.
- No external cache or crop dataset was retained; regeneration is deterministic.

## Figures

`figures/scaled_real_crop_evidence.png` contains seeded random valid outputs,
40%-hidden cases, minimum/maximum resize-scale cases, and rejected stored-frame
examples spanning all three recordings. `figures/contract_yield.png` reports
per-recording condition counts and yields. `figures/source_window_scales.png`
reports the k distribution and its condition dependence.

Visual inspection confirmed preserved real NIR texture with expected bilinear
resampling, clean window truncation, and no padding, painting, border
compositing, or synthetic background. Minimum- and maximum-scale panels remain
legible and spatially consistent. Rejected panels show the requested contiguous
support over the full stored image and make geometric re-entry/exclusion
failures visible. Figure overlays are diagnostic only and are not stored crop
pixels. The yield and scale plots agree with the recorded counts and k range.

## Runtime

CPU only. The timed benchmark main routine, including proxy hashing, 900-case
search with deterministic repeats, manifest construction, and three figures,
took 17.47 seconds, below the 15-minute limit. Exactly one read-only proxy HDF5
handle was used and each accepted image was read individually. No source
recording or audited holdout was opened.

## Interpretation

Variable-scale windows fix most of EXP-0003's physical-window mismatch and
validate exact candidate-proxy-referenced crop geometry for 720 individual
conditions. They do not produce the preregistered complete series for enough
frames. The shortfall is an exact-support geometry limitation rather than a
pixel provenance, interpolation, invertibility, or resource failure. The
head/tail asymmetry must not be read anatomically because proxy endpoint order
is uncertain. These results are not anatomical accuracy evidence.

## Decision

REJECT — 14 complete frames is below the unchanged >=60-frame gate. Do not use
EXP-0005 as the required complete real-texture downstream benchmark or silently
drop failed conditions. The 720 valid cases may be retained only as explicitly
incomplete candidate-proxy crop diagnostics.

## Next experiment

Do not retry the failed complete-frame gate. Predeclare a balanced per-condition
benchmark drawn deterministically from the 720 valid cases, with the benchmark
unit explicitly changed from complete source frame to crop condition. This can
answer the required fraction/end comparisons without silently dropping failed
conditions or pretending that the rejected same-frame hypothesis passed.
