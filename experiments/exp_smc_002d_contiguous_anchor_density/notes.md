# EXP-SMC-002D — contiguous strict-anchor density

## Outcome

The three authorized 101-frame development windows contain 87 strict EXP-SMC-001B/002B anchors out of 303 frames (28.71%). Every session has at least one anchor, but density and temporal continuity are strongly session-dependent:

| Recording | Inclusive window | Accepted | Density | Accepted runs | Longest run | Longest rejected gap | Adjacent accepted pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023-09-19-01 | 2515–2615 | 22/101 | 21.78% | 15 | 5 | 15 | 7 |
| 2023-09-27-01 | 15360–15460 | 58/101 | 57.43% | 13 | 14 | 29 | 45 |
| 2023-10-11-01 | 7704–7804 | 7/101 | 6.93% | 7 | 1 | 39 | 0 |

Static posture-sample extraction is feasible in this bounded sample. Session-general empirical dynamics fitting is **not supported by these windows**: 2023-10-11 contributes no adjacent accepted pair, and the other sessions are fragmented. The 52 total adjacent accepted pairs are dominated by 2023-09-27 (45/52), so fitting a shared dynamics model would confound session imbalance with temporal behavior.

This is a development-only feasibility result, not anchor-accuracy evidence. The generated selected overlays were inspected for gross consistency, but no new manual truth or outcome labels were created.

## Runtime and storage

- Serial CPU processing: 765.33 seconds for 303 frames (0.396 frames/s).
- Session runtimes: 260.62 s, 251.12 s, and 253.20 s.
- Bulky masks and anchor arrays: `/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/exp_smc_002d_contiguous_anchor_density/per_frame.h5`.
- HDF5 SHA-256: `399bcb9859a22d54d648bc29612b757a45db6ce2f83615d3063e2fdb92333e5d` (changed only because the internal experiment attribute was renamed from EXP-SMC-002C to EXP-SMC-002D; per-frame datasets were not recomputed).
- The output was written as a same-filesystem partial, marked complete, flushed, and atomically renamed.

## Evidence boundary

Only the three specified 2023 recordings and exact 101-frame windows were opened. The 2025 protected holdout was not opened. The final revised configuration is pinned by SHA-256 `cdd876a5012e1193cacaf52cc078e79d06a3ef292813b5c13b182b88e6ed19dd`.

The dominant nonexclusive rejection causes were strict topology checks (`branch_pixels`, `endpoint_count`, and `cycle`), with boundary-clearance rejection additionally important in 2023-09-27 and 2023-10-11. This explains fragmentation but does not establish whether rejected frames are anatomically difficult or merely segmentation failures.

## Decision

`STATIC_POSTURE_EXTRACTION_FEASIBLE__SESSION_GENERAL_DYNAMICS_NOT_FEASIBLE`

Do not fit session-general empirical dynamics from these windows. A future dynamics experiment needs either improved anchor continuity or an explicit sparse/irregular-time formulation with a preregistered minimum transitions-per-session criterion.
