# EXP-SMC-000 — Natural hard-bout candidate catalog

## Result

Six development-only candidate windows were retained: three partial-FOV/boundary windows, two strong coil or possible self-contact/intersection windows, and one tight-bend stress window. Five have known conservative classical proxy anchors on both sides. The frame-zero partial-FOV case is intentionally one-sided because no preceding frame exists.

These are **candidate windows only**. The type hypotheses in `catalog.json` are selection rationales, not adjudicated event labels, ground truth, prevalence measurements, or tracking outcomes. No model was run or scored on the catalog.

## Method and evidence boundary

- Only `2023-09-19-01`, `2023-09-27-01`, and `2023-10-11-01` were opened.
- Twelve seed neighborhoods came from existing development selection/proxy strata.
- The coarse screen read 245 raw `/img_nir` frames at stride 4 within bounded ±40-frame neighborhoods (truncated at recording start).
- The screen figures contain raw images only. No pose/classical overlay was used for visual retention.
- Anchor status comes from the immutable `accepted_frame_index` arrays in `proxy_v1/proxy_labels.h5`. Because that source sampled only 48 frames per recording, anchor distances are conservative upper bounds on the distance to an available easy frame, not claims of nearest possible anchors.
- Raw timestamp values use `img_metadata/q_iter_save == 1`; their physical unit remains inferred.

Bulky screening evidence is stored outside the repository:

```text
/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/
  exp_smc_000_natural_hard_bout_catalog/coarse_raw/
```

The 2025-03-06 protected holdout HDF5 and holdout-derived images were not opened.

## Reproduction

The bounded raw screen can be reproduced with:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  experiments/exp_smc_000_natural_hard_bout_catalog/screen_candidates.py \
  --output-dir <new-empty-output-directory> --radius 40 --stride 4
```

Do not add `--classical` when reproducing the exact raw visual screen. The option exists only for a future explicitly preregistered classical diagnostic and was not used here.

## Recommended use

Use the catalog to define pilot interval loading, bidirectional-anchor interfaces, and qualitative trajectory plots. Before reporting scientific recovery rates, freeze exact evaluation rules and obtain independent centerline/topology adjudication; do not promote these selection hypotheses into truth labels.
