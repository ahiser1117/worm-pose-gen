# EXP-SMC-008A — row-22 natural hard-case anchor bracket

## Outcome

The unchanged final EXP-SMC-001B/002B pipeline accepted **0/201** frames in the authorized development window `2023-10-11-01` frames 13685–13885, centered on expert-adjudicated hard frame 13785.

- Nearest strict anchor before 13785 within the scan: none.
- Nearest strict anchor after 13785 within the scan: none.
- Accepted runs: none.
- Rejected gaps: one, frames 13685–13885 inclusive, length 201.
- Two-anchor natural bout with at most 20 intervening frames: **false**.

This result is bounded to ±100 frames. It does not claim that no strict anchor exists elsewhere in the recording.

## Evidence

The local overlays show the expected compact loop/contact geometry around row 22. Orange contours are the EXP-SMC-001B cleaned masks. No cyan accepted centerline appears because every displayed frame is rejected by the strict EXP-SMC-002B detector.

All 201 frames triggered `branch_pixels` and `cycle`; 183 also triggered abrupt/implausible width checks, and 150 triggered low render IoU. These are nonexclusive detector reasons, not anatomical outcome labels.

The bracket overlay contains only frame 13785 because neither bracket anchor exists within the authorized scan.

## Runtime and storage

- Runtime: 564.93 seconds for 201 frames.
- Throughput: 0.356 frames/s, serial CPU.
- Bulky per-frame output: `/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/exp_smc_008a_row22_anchor_bracket/per_frame.h5`.
- HDF5 SHA-256: `57f5d6b9ba3c0a8aa7eec3847744a3de74f6f44a59e559a946120af1db25c23d`.
- Output was created as an exclusive partial, marked complete, flushed, and atomically renamed.

## Evidence boundary

Only `2023-10-11-01` frames 13685–13885 were read. The protected 2025 holdout was not opened. No SMC pose was inferred. The final segmentation/anchor config was unchanged and has SHA-256 `cdd876a5012e1193cacaf52cc078e79d06a3ef292813b5c13b182b88e6ed19dd`.

## Decision

`NO_BOUNDED_STRICT_ANCHOR_BRACKET`

The expert-adjudicated row-22 hard case is not a viable short natural two-anchor SMC pilot under the current strict-anchor pipeline. Reopening it would require a wider preregistered search or improved anchor continuity, not hidden pose inference inside this experiment.
