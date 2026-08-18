# Experiment Flow

The project-level decision flow will be updated as evidence accumulates. The
current branch is deliberately short: environment validation precedes data
forensics, and no model claim exists before the split and evaluation gates are
frozen.

```text
Reproducible CUDA + uv bootstrap [complete]
                    |
                    v
Bounded source audit + leakage-safe split [complete: 4/12 readable]
                    |
                    v
Classical proxy-label baseline + synthetic crop benchmark [accepted, limited]
                    |
                    v
Representation -> temporal context -> optional refinement/uncertainty
                    |
                    v
Frozen selection -> audited holdout -> packaged final system
```

The final version will reference the canonical project-level decision-path
figure and decisive visuals for every supported, rejected, revised, or
inconclusive major hypothesis.

The audit forced one important revision: cross-project/background validation
is unavailable because all readable inputs are in the starvation family.
Development therefore uses whole-session cross-validation. The distinct 2025
Hamamatsu-condition session is an audited holdout with its 32 pre-split audit
indices excluded, not a pristine final test.

## H0 — Can inexpensive labels and controlled truth support model research?

```text
Conservative real extraction                  Analytic synthetic geometry
90 / 144 accepted                             6,400 / 6,400 crops valid
0 / 24 gross visual failures                  <=3.41e-13 px round-trip error
             \                                /
              \                              /
               v                            v
 candidate real texture/proxy shape + Tier C exact geometry
                              |
                              v
             coordinate-versus-intrinsic proposal test
```

The hypothesis is **partially supported**. EXP-0001 supplies useful real-image
training candidates but not independent truth; 24 reviewed overlays form the
limited qualitative Tier B subset. Its most informative visual is
[`random_accepted_overlays.png`](../experiments/exp_0001_classical_proxy/figures/random_accepted_overlays.png).
EXP-0002 supplies exact geometry/crop contracts but simplified appearance; see
[`crop_sequence_montage.png`](../experiments/exp_0002_synthetic_crop/figures/crop_sequence_montage.png).
The consequence is a deliberately small learned proposal experiment that must
report candidate-proxy, reviewed Tier B qualitative, and Tier C evidence
separately.

## H0b — Can a literal training-size window create the real crop benchmark?

```text
256 x 192 direct source window
             |
             v
65 / 900 valid conditions; 0 / 90 complete frames
             |
             v
REJECT: physical window too small for the visible complement
             |
             v
EXP-0005: larger 4:3 source window -> isotropic resize
```

The branch is **not supported**. The decisive yield plot is
[`contract_yield.png`](../experiments/exp_0003_real_texture_crop/figures/contract_yield.png),
and the preserved-pixel successes/rejections are visible in
[`real_texture_crop_evidence.png`](../experiments/exp_0003_real_texture_crop/figures/real_texture_crop_evidence.png).
The exact support logic is retained; only the camera-window scale hypothesis is
revised.

The scale revision itself was also **not supported** at the complete-frame gate:
EXP-0005 produced 720/900 valid crops but only 14/90 complete source frames.
Its [`scaled crop evidence`](../experiments/exp_0005_scaled_real_crop/figures/scaled_real_crop_evidence.png)
shows that curved bodies can re-enter an axis-aligned window. The next branch is
declared as a balanced 300-condition benchmark (10 cases per recording, end,
and fraction), with this change of experimental unit explicit in EXP-0006.

## H0c — Can valid conditions form a balanced static real-texture benchmark?

```text
Frozen 720-condition valid pool
             |
             v
SHA-ranked 10/cell across 3 recordings x 2 ends x 5 fractions
             |
             v
300/300 exact cases; 87 unique source frames; all contracts pass
             |
             v
ACCEPT: static candidate-proxy engineering benchmark only
```

This revised hypothesis is **supported**. The decisive
[`balance figure`](../experiments/exp_0006_balanced_real_crop/figures/balance.png)
shows exact cell coverage, while
[`all 40%-hidden cases`](../experiments/exp_0006_balanced_real_crop/figures/all_40_percent_cases.png)
exposes the hardest selected condition rather than hiding failures. The change
of experimental unit and 87-frame reuse are explicit limitations; quantitative
hidden-anatomy claims continue to rely on Tier C truth.
