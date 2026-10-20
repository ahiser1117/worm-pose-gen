# Stage alignment: the substrate in image coordinates

The flv-c rigs keep the worm in view by moving the stage, so the agar moves
through the camera image and the body-field network's difference channels
`frame[t+lag] - frame[t-lag]` mix the worm's own motion with the stage's.
`src/worm_pose_gen/stage_alignment.py` calibrates the recorded stage
position against the image and gives each frame's offset to substrate
coordinates (`substrate_shift_px`), so frames can be aligned on the agar
(`align_frame`) and positions put in a frame fixed to the agar.

## The stage track

`/pos_stage` has one row per saved frame, in stage units of 0.1 µm
(`export_table.STAGE_UM_PER_UNIT`). Failed serial reads are NaN rows: 12% of
2023-03-30-01, 7-20% of the recordings below, in runs of 1-4 frames and
occasionally longer. `read_stage` interpolates them linearly in frame index
and marks a frame unreliable inside a run longer than `MAX_GAP_FRAMES` (4)
or outside the first and last reading (2.5% of 2023-03-30-01's frames,
1.1-8.7% of the others below);
`substrate_shift_px` is NaN there.

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
   rings, dust). The pattern is removed as the median of band-passed anchor
   frames, estimated twice from alternate frames and one estimate
   subtracted from each frame of a pair; one estimate subtracted from both
   frames adds a common noise image and every fit collapsed to a zero
   matrix. With the two estimates 7 of 1,058 pairs of 2023-03-30-01 that
   should move by more than 2 px register at zero (96 without removal).
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

calibration = recording_calibration(recording)              # cached, estimated on first use (~2 min)
offsets = substrate_shift_px(recording, frames, calibration)  # [n, 2]; image xy + offset = substrate xy
aligned, valid = align_frame(image_s, offsets[i_s].round(), offsets[i_t].round())  # frame s in frame t's camera position
```
