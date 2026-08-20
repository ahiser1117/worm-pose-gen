# Single-annotator Tier-A workflow

This is the active annotation protocol for the stated constraint that one
person, not multiple people, will label images. It uses the frozen 256-frame
development manifest as a candidate pool but does **not** require labeling all
256 frames.

## Bare minimum decision tranche

Label **30 unique frames**: 10 from each readable 2023 development session.
The tool balances each session across proxy-difficult, proxy-easy, uniformly
spaced, and temporal-window strata without showing any proxy or model pose.

After seven days, label **10 of those frames again**. The repeats are selected
deterministically across sessions, remain blind to the first trace, and measure
intra-annotator repeatability. Total initial effort is therefore **40 traces**.

This tranche is sufficient to:

- test whether a method is grossly wrong on real images;
- compare large baseline/architecture improvements on a paired development set;
- replace arbitrary pixel gates with a provisional single-person repeatability
  scale; and
- discover whether the annotation protocol itself is usable.

It is not sufficient to estimate inter-annotator variability, support a claim
of multi-person consensus ground truth, or finely rank methods whose confidence
intervals overlap. If a promising method lands near the repeatability scale,
expand the same frozen worklist to 60 or 90 unique frames before making a close
model-selection decision. Do not open the 2025 holdout for this work.

## What the annotator actually does

For an ordinary frame, drag once along the anatomical centerline from either
end to the other, adjust any generated control points that need correction,
and save. Head/tail identity, width, and difficulty tags are optional.

Only use the exception controls when the body leaves the image, crosses itself,
or cannot be traced reliably. A not-identifiable frame is saved with no invented
coordinates. Temporal neighbors are available for visual context, but the tool
never displays a model, proxy, or previous annotation.

## Launch and resume

Use a stable pseudonym and keep the output outside the repository:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/annotate_tier_a.py \
  --annotator-id alex \
  --output /temp_data4/alex/external_artifacts/annotations/worm_pose_tier_a_alex.json \
  --open
```

The output is created immediately and updated atomically after every saved
trace. Stop with Ctrl-C at any time. Run the same command later to resume. Once
the 30 primary labels are complete, the application reports when the 10 blind
repeats unlock.

For a short usability pilot, use `--primary-count 3 --repeat-count 0` with a
different output path. That pilot does not count as scientific evidence unless
it was itself selected and completed under the intended protocol.

## Evaluate the repeat pass

After all repeats are complete:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/evaluate_annotation_agreement.py \
  --manifest experiments/scientific_exp_001_annotation/selection_manifest.json \
  --annotations /temp_data4/alex/external_artifacts/annotations/worm_pose_tier_a_alex.json \
  --output-dir /temp_data4/alex/external_artifacts/annotations/worm_pose_tier_a_repeatability \
  --comparison-mode intra-annotator \
  --minimum-pairs 10
```

Report the result as **intra-annotator repeatability**, never inter-annotator
agreement or a complete human noise floor.
