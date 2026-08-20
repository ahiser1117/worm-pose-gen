# EXP-001 — Inter-annotator precision and metric calibration

> **Protocol amendment (2026-08-19):** The project will have one annotator.
> The active minimum is 30 balanced unique frames plus 10 delayed blind repeats
> from the same person, selected from this frozen 256-frame pool by
> `scripts/annotate_tier_a.py`. This measures intra-annotator repeatability, not
> the inter-annotator hypothesis originally preregistered below.

## Hypothesis

The meaningful accuracy target for these recordings can be calibrated from
independent human disagreement instead of inherited 4 px / 8 degree gates.

This is motivated by
[DeepTangleCrawl](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013345)'s
direct comparison of model RMSD with manual-annotator RMSD and by the existing project's inability to separate
proxy-label bias from model error without Tier-A evidence.

## Controlled change

This follow-on experiment creates a new development-only Tier-A tranche. It
does not modify the earlier proxy labels, model checkpoint, split groups, or
protected 2025 holdout policy.

The checked-in selection contains 256 targets allocated 86/85/85 across the
three complete 2023 development sessions. Six blind, nonoverlapping 11-frame
windows (two per session; 66 target frames total) are marked for two independent
annotations. Remaining targets combine model-blind uniform coverage with
classical accept/reject outcome strata. Classical centerlines and learned-model
overlays are never shown in the primary pass.

## Result

- Selection manifest: `selection_manifest.json`
- Manifest record-content SHA-256:
  `11d93b68e8c0433e32a0b5b192693704f004fc09eb3240aa319506c1b62dc6b9`
- Manifest file SHA-256:
  `ba0a9ccc6f08844ecfcc52167271ba0cc13a47dafc8ba54bb2f25d1aec2619b3`
- Blind preview: `selection_preview.png`
- Protected holdout opened: **false**
- Active single-annotator labels received: **30/30 primary; 0/10 repeat**
- Primary trace states: **17 complete; 12 naturally truncated; 1 not identifiable**
- Primary-session validation: **all 30 pass schema, frozen-worklist, source-identity,
  development-split, and no-overlay checks**
- Original stronger target received: **0/64 independent pairs**

The label contract is `configs/tier_a_annotation_schema.json`. Runtime
validation and symmetric agreement metrics are implemented in
`worm_pose_gen.annotation`; the evaluator produces the three required figures
and bootstrapped numerical table.

Build/reproduce the immutable selection (a new output directory is required):

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/build_tier_a_annotation_manifest.py --output-dir <new-output-dir>
```

Launch or resume the active annotation tranche:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/annotate_tier_a.py \
  --annotator-id <stable-pseudonym> \
  --output /temp_data4/alex/external_artifacts/annotations/tier_a.json \
  --open
```

After the delayed repeat pass:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/evaluate_annotation_agreement.py \
  --manifest experiments/scientific_exp_001_annotation/selection_manifest.json \
  --annotations /temp_data4/alex/external_artifacts/annotations/tier_a.json \
  --output-dir <new-repeatability-output-dir> \
  --comparison-mode intra-annotator --minimum-pairs 10
```

## Visual evidence

The inspected selection preview shows the raw NIR images only and spans
ordinary locomotion, boundary contact/truncation, scale/background variation,
high curvature, and several tight turns. Automated difficulty hints do not
guarantee every requested rare category; annotators must retain the explicit
difficulty fields and may mark a trace not identifiable.

## Conclusion

`PRIMARY_COMPLETE_REPEATABILITY_PENDING`

The 30-frame primary pass is complete and supports directional model triage.
The human-repeatability measurement remains inconclusive until the 10 delayed
blind repeats unlock on 2026-08-26. Primary labels must not be described as a
human noise floor.

## Consequence

Complete the 10 delayed repeats after their enforced seven-day wait, then run
the checked-in intra-annotator evaluator. The primary pass already authorizes
directional EXP-002 and representation-floor triage, but not close ranking near
human precision. Expand the unique-frame tranche before such a comparison. The
independent Tier-C EXP-008 capture-basin result remains explicitly provisional.
