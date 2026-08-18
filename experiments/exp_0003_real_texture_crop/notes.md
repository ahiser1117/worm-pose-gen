# EXP-0003 — Real-texture controlled crop benchmark

## Hypothesis

Accepted, fully visible real proxy frames can be converted into reproducible
camera-window crops that preserve observed NIR texture/background and provide
exact coordinate transforms and independently defined anatomical support.

## Difference from baseline

EXP-0002 validated analytic geometry and simple rendered appearance. This
experiment applies the same evidence contract to the 90 accepted candidate
proxy frames; their centerlines remain training/QC references rather than
ground truth. Only 24 overlays received the independent qualitative review
needed for even limited Tier B engineering evidence.

## Data/split

Read only `accepted_image`, accepted source indices, and corresponding accepted
centerlines from the immutable external `proxy_v1/proxy_labels.h5`. Do not open
any source recording or the audited holdout. Preserve recording identity so
downstream evaluation follows the three frozen whole-session folds. For each
usable frame, attempt head/tail crops at 5%, 10%, 20%, 30%, and 40% hidden.

## Training/resource budget

- maximum steps/epochs: not applicable; deterministic transformation
- wall-time limit: 10 CPU minutes
- seed/repeat policy: base seed 20260818; deterministic source order
- checkpoint cadence: not applicable
- expected GPU time: none
- expected external-storage use: <=2 GiB under `datasets/worm_pose_gen/real_crop_v1`
- early termination conditions: source-recording open, source/proxy overwrite,
  non-invertible transform, inconsistent support, or padded/painted crop pixels

## Success criterion

- primary metric: valid real-texture crop contract yield
- numeric practical-effect threshold: at least 60 accepted proxy frames permit
  all ten end/fraction crop conditions; every emitted crop is a direct
  axis-aligned subwindow of its stored real image, has no generated pixels,
  has max point transform round-trip error <=1e-5 px, and its stored support
  exactly equals independently recomputed half-open FOV membership
- variability/confidence rule: report usable count by recording and condition;
  inspect deterministic random, maximum-hidden, and rejected cases
- pass/fail interpretation: ACCEPT only as candidate-proxy-referenced crop
  evidence; a pass does not establish anatomical accuracy of the proxy
  centerline

## Results

The immutable proxy artifact hash matched the frozen configuration. All 90
accepted proxy frames were attempted in deterministic recording/frame order,
for 900 total requests. A request hid `ceil(N * fraction)` contiguous samples
at the named end and required the exact complement to be inside the 256 x 192
half-open window. This independently defined support was not relaxed when the
remaining centerline did not fit.

- 65/900 requests (7.22%) yielded a valid direct crop.
- 0/90 frames yielded all ten requested conditions (recording counts: 0/33,
  0/26, and 0/31), failing the predeclared minimum of 60 complete frames.
- Valid counts for head/tail at 5%, 10%, 20%, 30%, and 40% were respectively
  1/1, 1/1, 2/2, 9/6, and 23/19.
- Rejections comprised 818 cases where the declared visible complement could
  not fit and 17 where it fit dimensionally but no window produced the exact
  support mask.
- All 65 emitted crops passed direct source-subwindow pixel equality, exact
  independently recomputed half-open support, and deterministic repeat checks.
  Maximum source-to-crop-to-source point error was 0.0 px (limit 1e-5 px).
- No external crop dataset was retained; the small result is deterministically
  regenerated from the immutable proxy artifact.
- `metrics.json` embeds a 900-entry deterministic case manifest covering every
  valid and rejected request. Each entry records proxy group/frame identity,
  accepted-image/sample positions, half-open crop bounds, unit scale and exact
  translation matrix where valid, requested/actual support bitmasks, and the
  rejection reason otherwise. The manifest SHA-256 is
  `5cd4a56026aa5bbc59104c8d6e8788e0ae0d9421449c94cb6c11b6bdce27214a`.

## Figures

`figures/real_texture_crop_evidence.png` shows seed-selected valid real-texture
crops, maximum-hidden valid crops, and rejected stored-frame examples spanning
all three recordings and both rejection modes. `figures/contract_yield.png`
shows valid count and percentage by recording for every condition.

Visual inspection confirmed that valid panels contain unpainted NIR texture
and hard camera-window truncation without padding or border compositing. The
40% cases visibly retain the declared centerline complement. Rejected examples
make the dominant failure clear: the long visible complement cannot fit within
the fixed axis-aligned window. No plotting artifact was observed that changes
the underlying crop pixels; overlays exist only in the evidence figure.

## Runtime

CPU only. The timed benchmark main routine, including proxy hashing and figure
writing, took 7.34 s; the observed combined focused-test and benchmark command
took 9.6 s, below the 10-minute limit. One read-only HDF5 handle was used and
each accepted image was read individually. No source recording or audited
holdout was opened.

## Interpretation

The pixel and geometry portions of the hypothesis are supported for the 65
conditions that are geometrically usable: every crop is a literal stored-image
slice with a translation-only invertible transform and exact support. The full
hypothesis is rejected for the configured 256 x 192 window because most accepted
proxy centerlines are too spatially extended to keep 60 frames complete across
all ten end/fraction requests. This is candidate proxy-referenced crop evidence
only and does not establish anatomical centerline accuracy.

## Decision

REJECT — 0 complete frames is below the predeclared >=60-frame gate. Retain the
strict crop primitive and valid-condition diagnostics, but do not promote this
configuration as the complete real-texture crop benchmark. It cannot serve as
the required downstream benchmark under the frozen 256 x 192 specification.

## Next experiment

Use accepted crop cases only for exploratory stratification while testing a
new predeclared window specification for the required representation and
temporal proposal benchmark; do not confuse proxy agreement with manual-label
accuracy.
