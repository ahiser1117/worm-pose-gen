# Stage alignment: the substrate in image coordinates

The flv-c rigs keep the worm in view by moving the stage, so the agar moves
through the camera image and the body-field network's difference channels
`frame[t+lag] - frame[t-lag]` mix the worm's own motion with the stage's.
`src/worm_pose_gen/stage_alignment.py` calibrates the recorded stage
position against the image and gives each frame's offset to substrate
coordinates (`substrate_shift_px`), so frames can be aligned on the agar
(`align_frame`) and positions put in a frame fixed to the agar. Where the
stage readings are missing or miss fast motion, the offsets come from
registering consecutive frames instead (see *Filling with registration*).

## The stage track

`/pos_stage` has one row per saved frame, in stage units of 0.1 µm
(`export_table.STAGE_UM_PER_UNIT`). Failed serial reads are NaN rows: 12% of
2023-03-30-01, 7-20% of the recordings below, in runs of 1-4 frames and
occasionally longer. `read_stage` interpolates them linearly in frame index
and marks a frame unreliable inside a run longer than `MAX_GAP_FRAMES` (4)
or outside the first and last reading (2.5% of 2023-03-30-01's frames,
1.1-8.7% of the others below);
the stage-only offset (`stage_shift_px`) is NaN there, and
`substrate_shift_px` fills it by registration.

## Calibration

`StageCalibration`: a 2x2 matrix, pixels of background motion per stage
unit, and a time offset: the reading for frame t is taken at fractional
frame `t + offset_frames`. `estimate_calibration` (cached per recording
under `<data dir>/cache/stage_calibrations/<stem>.json` by
`recording_calibration`):

1. 400 anchor frames spread over the recording (160 by default), each
   registered against frames t+1, t+4 and t+16 (flat-fielded).
2. Registration (`register_shift`): band-pass (Gaussian σ 1 px minus σ 6 px),
   worm left out (workspace masks, or `dark_mask`: 15 grey levels under the
   frame median after a 2 px blur, which covers 98-100% of the workspace's
   worm pixels), the mask edge tapered, Hann window, cross-correlation with
   the cross-power divided by the square root of its magnitude, parabolic
   sub-pixel peak. Plain phase correlation of high-passed frames locks onto
   zero shift: the agar has little texture (a band-passed background is
   mostly pixel noise) and the camera adds a fixed pattern (interference
   rings, dust). The pattern is removed as the median of 64 band-passed
   frames spread over the recording, estimated twice from alternate frames and one estimate
   subtracted from each frame of a pair; one estimate subtracted from both
   frames adds a common noise image and every fit collapsed to a zero
   matrix. With the two estimates 7 of 1,058 pairs of 2023-03-30-01 that
   should move by more than 2 px register at zero (96 without removal).
   `measure_shifts` registers in 8 threads (the FFTs and filters release
   the GIL) and prepares each frame once per run of anchors: a 400-anchor
   calibration takes 24 s, and reproduces the cached matrices to 0.0001
   px/unit and the offsets exactly (2023-03-30-01, 2023-08-22-01,
   2024-01-31-02; the pattern was formerly the median of the anchor frames).
3. Fit (`fit_calibration`): pairs scoring under 15 (peak over the
   correlation's standard deviation) are dropped; for each offset in
   -2..2 frames by 0.05 the matrix is fit by least squares and refit on
   pairs within 3 px; the offset with the lowest truncated squared error wins.

Measured on five recordings (400 anchors, dark-pixel masks except
2023-03-30-01, which used its workspace's masks; the dark-pixel fit of
2023-03-30-01 gives the same matrix to 0.3%):

| Recording | x px/unit | y px/unit | off-diagonal | µm/px | offset (frames) | RMS residual (px) | outliers |
|---|---|---|---|---|---|---|---|
| 2023-03-30-01 | 0.0779 | 0.0783 | -0.0003, -0.0001 | 1.276-1.286 | -0.80 | 1.12 | 78 / 1060 |
| 2023-06-23-01 | 0.0777 | 0.0778 | -0.0002, -0.0002 | 1.282-1.289 | -0.80 | 1.09 | 77 / 1088 |
| 2023-08-22-01 | 0.0775 | 0.0780 | 0.0001, 0.0002 | 1.281-1.292 | -0.70 | 1.11 | 94 / 748 |
| 2024-01-31-02 | 0.0778 | 0.0773 | -0.0000, -0.0000 | 1.285-1.294 | -0.80 | 1.20 | 106 / 933 |
| 2024-05-28-02 | 0.0783 | 0.0786 | -0.0003, 0.0001 | 1.271-1.278 | -0.75 | 1.35 | 227 / 1088 |

- **Axes and signs**: stage x is image x and stage y image y, both with
  positive sign (the background moves by `+matrix @ d` when the stage moves
  by `+d`); rotation under 0.13°. This confirms `export_table`'s convention
  (found there on fitted centroids).
- **Scale**: 0.0773-0.0786 px per unit, 1.27-1.29 µm/px, 1-2% larger than
  the nominal 1.264 µm/px of the library model metadata.
- **Time offset**: -0.7 to -0.8 frames: the reading stored with frame t
  was taken about 40 ms after its exposure, so the position at the exposure
  is best estimated by the readings of t and t-1 weighted 0.2/0.8.
  `export_table` uses their mean (offset -0.5).
- **Consistency**: per quarter of 2023-03-30-01 the diagonal varies by
  ±1% (0.0775-0.0790) and the offset by ±0.08 frames. The quarters of the
  other recordings agree within ±1.5%, except one quarter each of
  2024-01-31-02 (y 0.0756) and 2023-08-22-01 (offset -1.15, its second
  half has the least texture).
- **Residuals** (2023-03-30-01, median / 90th percentile): lag 1 0.59 /
  2.0 px, lag 4 0.83 / 3.1 px, lag 16 1.04 / 3.3 px. The large residuals
  are stage motion the readings miss: during fast stage oscillations
  (2023-03-30-01 frames 4691-4702: the background moves ±23 px per frame
  in y for 10 frames) the readings, half of them missing, show a fraction
  of the excursion. On 2023-08-22-01 and 2024-01-31-02 the lag-16 90th
  percentile is 12 px and 35-38% of lag-16 pairs miss by more than 3 px.
  There the lag-16 registrations themselves are doubtful (next section):
  at shifts of 55-110 px they score a median 13-14 and disagree with
  chained lag-1 registrations as often as with the stage.

## Filling with registration

`substrate_shift_px` uses the stage where it is trusted and registers
consecutive frames where it is not: where readings are missing (runs of
more than four failed reads, before the first and after the last reading)
or miss fast motion.

### How the stage fails

Every frame of 13 windows of 1,000 frames (five recordings, two or three
windows each; 0.11 s per frame on one core) was registered against the next
one, and the lag-1 steps chained. Against these chains, with the slowly
varying difference (running median over 33 frames) removed, the stage's
per-frame offset is off by more than 3 px on 4.2% of 12,344 frames, more
than 5 px on 1.0%, more than 8 px on 0.3% (per window 1.7-11.4% over 3 px;
the worst is 2024-01-31-02 frames 5000-6000). Two kinds:

- **Reading time jitter.** A reading is taken a little early or late, so it
  is off by the speed times the jitter, and the next reading is right again:
  at 10 px per frame, errors of 3-6 px at single frames (2024-01-31-02
  frames 5037-5039: the stage says -14.5, -8.0, -6.7 px per frame, the image
  -8.6, -9.7, -10.3).
- **Missed oscillations.** 2023-03-30-01 frames 4691-4703: the image moves
  +7, +23, +2, -1, -18, -24, -24, +5, +23, +23, +2, 0, -9 px in y, over a
  range of 67 px; the stage has 6 readings in those 13 frames, spanning
  31 px, and its offset is off by up to 31 px (frame 4699).

The lag-1 registration agrees with direct lag-4 registration to 0.16 px
(median; 90th percentile 0.33 px, none over 3 px) on 2023-03-30-01 frames
4400-5400 and to 0.58 / 1.8 px on 2024-01-31-02 frames 5000-6000, and lag-1
chains with direct lag 16 to 0.52 / 1.0 px on 2023-03-30-01. On the other
2023-08-22-01 and 2024-01-31-02 it is biased towards zero by 2-6%
(regression slope of the lag-1 shift on the stage 0.94-0.98 per axis,
against 0.99-1.02 at lag 16), which over a long chain is drift. A lag-1 registration scoring under 12 is often wrong:
of pairs the stage moves by over 2 px, 31% of those scoring 8-10 and 11% of
those scoring 10-12 miss the stage by more than 8 px, 4.7% at 12-20, 1.9% at
20-30 and 0.3% above 30 (the last partly stage error).

### The method

1. **Untrusted frames** (`untrusted_frames`): the stage-only offset is NaN,
   or the stage track (failed reads interpolated, in image pixels) has a
   second difference over 8 px (`UNTRUSTED_ACCELERATION_PX`: a reading off
   by e shows as second differences of e and 2e) or a centred speed over
   10 px per frame (`UNTRUSTED_SPEED_PX`: jitter errors grow with speed).
   Costless, from the stage alone.
2. **Anchors** (`anchor_frames`): frames whose offset is interpolated from
   two real readings (not filled-in failed reads) and that are at least two
   frames (`ANCHOR_GUARD`) from an untrusted frame. Inside an oscillation
   some readings happen to look smooth: 2023-03-30-01 frames 4699-4700 have
   second differences of 4.6 and 6.5 px but are 31 and 15 px off, so the
   frames next to a jump, and interpolated frames, do not anchor.
3. **Stretches** (`fill_stretches`): each run of non-anchor frames holding
   an untrusted frame, reaching out to 4 anchors on each side
   (`FILL_MARGIN`); stretches that overlap merge. Only these frames are
   registered.
4. **Registration** (`recording_steps`): each frame of a stretch against the
   next (`measure_shifts` at lag 1, dark-pixel worm masks, fixed pattern
   removed), cached per recording under
   `<data dir>/cache/stage_steps/<stem>.npz` and extended when a later call
   needs more steps.
5. **Fill** (`fill_shifts`): the steps scoring 12 or more (`STEP_MIN_SCORE`)
   are chained, the chain broken at the others. A piece is anchored to each
   run of anchor frames it holds (the median of stage minus chain, placed
   at the run's middle frame) and its offset interpolated linearly between
   anchors, so the drift accumulated between two anchors is spread evenly
   over the steps between them. A piece without anchors takes the median of
   whatever stage offsets it has (2023-08-22-01's second half, little
   texture, has steps scoring under 12 inside long read gaps); one without
   any stays NaN.

The thresholds were chosen on the dense windows, filling them with their
own lag-1 steps and comparing the offsets with direct registration at lags
4 and 16 (score at least 15), which the fill does not use. Fractions of
pairs off by more than 3 or 8 px, a NaN offset counted as off (lag 4: 1,965
pairs of 2023-03-30-01 frames 4400-5400 and 2024-01-31-02 frames 5000-6000;
lag 16: 2,658 pairs of five windows):

| Variant | frames filled | lag 4 > 3 px | lag 4 > 8 px | lag 16 > 3 px | lag 16 > 8 px |
|---|---|---|---|---|---|
| stage only (9.1% / 11.7% of pairs NaN) | - | 22.8% | 10.5% | 31.1% | 13.4% |
| **acceleration 8, speed 10, guard 2, margin 4** | 18.4% | 8.0% | 0.1% | 17.8% | 1.8% |
| no guard | 15.4% | 8.9% | 0.3% | 18.2% | 1.8% |
| margin 8 | 24.0% | 7.2% | 0.1% | 17.4% | 1.8% |
| acceleration 6, speed 8 | 29.3% | 5.1% | 0.0% | 16.5% | 1.7% |
| acceleration 10, speed 12 | 14.0% | 10.2% | 0.3% | 18.6% | 1.8% |
| acceleration 8, no speed limit | 15.6% | 9.8% | 0.4% | 18.5% | 1.8% |

A first version took every trusted frame as an anchor, interpolated ones
included: the 2023-03-30-01 oscillation was anchored on offsets up to
27 px off, and its filled lag-4 pairs were worse than the stage's (10.8%
against 8.3% over 8 px). Lag-16 pairs stay off on
2023-08-22-01 and 2024-01-31-02 whatever the variant, the lag-16
registrations there being doubtful (above).

### Results

`scripts/stage_alignment_report.py --fill` on the five recordings (cached
calibrations; 2023-03-30-01 evaluated with its workspace masks), 150 frames
sampled from each recording's stretches. *Held out*: offsets of the pairs
(t, t+4) against their direct registration (score at least 15), all pairs
(a NaN offset counted as over 3 px) and, in brackets, the pairs where the
stage-only offset exists. *Worm*: lag-1 difference energy on the worm
(grey levels², as in the next section), stage / filled / registered.

| Recording | frames filled | first use | NaN stage → filled | lag 4 > 3 px, stage → filled | lag 4 > 8 px, stage → filled | worm, lag 1 |
|---|---|---|---|---|---|---|
| 2023-03-30-01 | 16.7% (154 stretches) | 49 s | 2.80% → 0.03% | 39.2% → 0.7% (14.7% → 1.0%) | 30.1% → 0 (2.0% → 0) | 139 / 127 / 128 |
| 2023-06-23-01 | 16.9% (137) | 67 s | 3.51% → 0 | 46.9% → 2.1% (18.1% → 3.2%) | 40.7% → 0 (8.5% → 0) | 179 / 155 / 162 |
| 2023-08-22-01 | 26.6% (126) | 89 s | 9.71% → 2.58% | 49.4% → 3.9% (18.8% → 6.2%) | 40.3% → 0 (4.2% → 0) | 143 / 127 / 142 |
| 2024-01-31-02 | 17.3% (142) | 59 s | 1.61% → 0 | 46.4% → 5.1% (35.7% → 6.1%) | 29.7% → 0 (15.7% → 0) | 179 / 162 / 162 |
| 2024-05-28-02 | 21.5% (202) | 74 s | 1.26% → 0 | 42.7% → 4.7% (37.2% → 5.1%) | 16.0% → 0 (8.0% → 0) | 245 / 218 / 216 |

- **Coverage**: the NaN offsets fall from 1.3-9.7% of frames to none,
  except 0.03% of 2023-03-30-01 and 2.6% of 2023-08-22-01 (64 runs of up
  to 49 frames, inside read gaps of the low-texture second half where the
  steps score under 12).
- **Accuracy**: on the stretches the filled offsets match direct lag-4
  registration to a median 0.3-1.0 px (stage 1.1-2.1 px), and no pair is
  off by 8 px. At lag 16 the filled offsets are better where the
  registration is trustworthy (2023-03-30-01: 33.1% → 1.4% over 3 px;
  2023-06-23-01: 40.7% → 3.4%) and the comparison is inconclusive where it
  is not (2024-01-31-02 56% → 38%, 2024-05-28-02 51% → 40%; where the
  stage exists 44% → 37% and 46% → 41%). On 2023-08-22-01, 2024-01-31-02
  and 2024-05-28-02 aligning by direct lag-16 registration leaves more
  worm difference energy than aligning by the stage (554-724 against
  355-483), so it is not a reference there.
- **Difference energy**: on the worm the filled offsets reduce the lag-1
  energy by 9-13% against the stage, to that of registering each pair; at
  lag 4 by 1-6%; at lag 16 not at all (within 1%). The band-passed
  background changes by at most 0.6 grey levels².
- **Cost**: 17-27% of the frames are registered, 0.1 s per frame on one
  core; with 8 threads the first call took 49-89 s per recording (five
  recordings at once on flv-c4) and later calls read the cache (under a
  second). A 2023-03-30-01 cache is 126 kB.
- **Held out, dense windows** (the cached fill against the dense windows'
  direct registrations): 2023-03-30-01 frames 4400-5400, lag 4 over 3 px
  22.6% → 3.1% of all pairs, on the pairs the fill changes 52.6% → 0;
  2024-01-31-02 frames 5000-6000 23.1% → 13.2%, over 8 px 3.2% → 0. In the
  oscillation of 2023-03-30-01 the filled offset follows the chained
  registration to 1.3 px (frame 4693: stage -1040.9, filled -1068.3, chain
  -1067.8).

Measured with `--fill` runs under
`<data dir>/runs/stage_alignment/fill/<recording>/fill_report.json`.

## Does alignment matter? Difference energy

`scripts/stage_alignment_report.py --workspace .../2023-03-30-01`: 150
frames spread over the recording, `frame[t+lag] - frame[t-lag]` with both
frames moved (whole pixels) to frame t's camera position, median over
frames of the mean square difference (grey levels²) on the same pixels for
every variant. *Background*: valid pixels off the worm in all three frames;
*worm*: pixels on the worm in any of them. *Registered* aligns each frame
by registering it to frame t (the image-based reference).

| Lag | Region | unaligned | stage, offset 0 | stage, offset -0.5 | stage, fitted -0.8 | registered |
|---|---|---|---|---|---|---|
| 1 | worm | 264 | 148 | 137 | 138 | 132 |
| 4 | worm | 586 | 219 | 209 | 209 | 209 |
| 16 | worm | 716 | 306 | 300 | 300 | 308 |
| 1 | background, band-passed | 6.17 | 6.02 | 5.96 | 5.96 | 5.92 |
| 4 | background, band-passed | 6.20 | 6.12 | 5.99 | 5.97 | 5.92 |
| 16 | background, band-passed | 6.13 | 6.22 | 6.06 | 6.01 | 5.91 |
| 1 | background, raw | 78.0 | 79.3 | 79.1 | 79.1 | 78.8 |
| 16 | background, raw | 77.0 | 80.7 | 80.4 | 80.3 | 79.7 |

- **On the worm** alignment removes 48-64% of the difference: most of what
  the network's difference channels show of the worm is the stage moving it
  through the frame, not its motion over the agar. The stage reaches the
  registered alignment at every lag.
- **The background** is almost all pixel noise (σ ≈ 6.2 grey levels, mean
  square 2σ² ≈ 78 unaligned at every lag): there is too little agar texture
  for alignment to show in raw pixels, and aligned frames bring the camera's
  fixed pattern out of cancellation (+1-4 grey levels²). Band-passed, the
  agar texture is 4% of the difference; stage alignment removes 55-85% of
  what registration removes.
- **Time offset**: the fitted offset beats offset 0 by 7% on the worm at
  lag 1 and on the band-passed background at lag 16, where offset 0 is
  worse than no alignment; -0.5 is nearly as good as the fit.
- **Interpolation**: shifted with linear interpolation (`shift_image`'s
  default) the raw background difference falls from 78 to 35 grey levels²
  at every lag, entirely from averaging pixel noise at fractional shifts.
  A difference channel should round the shift (or interpolate both frames)
  so its noise level does not depend on the offset.

The figure of this comparison is written to
`<data dir>/runs/stage_alignment/2023-03-30-01_lag16_aligned_vs_unaligned.png`
(frames t = 4700, 4705, 4717; raw and band-passed rows). Unaligned, the
whole worm appears as a bright and a dark copy displaced by the stage
motion; aligned, only the parts that moved over the agar differ, and the
camera's ring pattern appears faintly in the raw background.

## Coverage

Every recording of the lab library's recording folder
(`/storage/fs/data2/prj_aversion/data_raw`, 50 files) has `/pos_stage` and
`/pos_feature` (`[n, 3, 3]`), with 3-21% of stage rows missing. Of the two
registered recordings, 2025-09-10-01 has both (24% missing) but its frames
need an HDF5 filter plugin that is not installed here, so it cannot be
calibrated; the starved-patch video
(`date-20251223_..._worm004.h5`) has neither. All seven clips of
`eval_sets/sequence_eval_set.json` (four recordings under `/store1/shared`,
copies of the data2 files) have both.

## Use

```python
from worm_pose_gen.stage_alignment import recording_calibration, substrate_shift_px, align_frame

calibration = recording_calibration(recording)              # cached, estimated on first use (~25 s)
offsets = substrate_shift_px(recording, frames, calibration)  # [n, 2]; image xy + offset = substrate xy, NaN where unknown
                                                              # (registration steps cached, ~1-1.5 min on first use)
aligned, valid = align_frame(image_s, offsets[i_s].round(), offsets[i_t].round())  # frame s in frame t's camera position
```
