# Scientific Experiment Plan for High-Throughput 2D *C. elegans* Pose Estimation

**Purpose:** Replace the model-search portion of the previous `worm-pose-gen` research plan with a literature-grounded sequence of experiments.

**Target application:** 2D pose estimation from ~20 Hz, 10× NIR *C. elegans* behavior recordings. The primary biological output is body tangent angle/curvature as a function of normalized arc length and time. The method must remain useful when the head or tail leaves the field of view (FOV), and throughput must be high enough for full-recording analysis.

**Implementation constraint:** final learned components should remain compatible with the existing PyTorch Lightning pipeline. Reuse the existing HDF5, experiment logging, split, geometry, output-schema, external-storage, and visualization infrastructure unless an experiment demonstrates a reason to change it.

---

## 1. What should change from the first research pass

The first pass established useful infrastructure, but its scientific search was too narrow. It mostly tested global single-frame regression from an aggressively compressed spatial representation and then prevented temporal, refinement, and probabilistic experiments after that proposal failed.

This new plan changes the research logic in six ways:

1. **Calibrate accuracy against human annotation precision.**  
   Do not treat the previous 4 px / 8° gates as fundamental. Measure inter-annotator disagreement and derive meaningful targets from it.

2. **Preserve spatial localization.**  
   Do not ask a globally pooled feature vector to recover the absolute location of a thin worm unless that architecture earns its place against spatially anchored or dense alternatives.

3. **Test temporal information early.**  
   Worm-specific literature strongly supports temporal context. A failed single-frame model must not block a 1/5/11-frame comparison.

4. **Use realistic synthetic data, not only analytic tubes.**  
   Synthetic examples should retain real NIR worm texture/background statistics through image warping or compositing, following the central idea in WormPose.

5. **Measure the capture basin of differentiable refinement explicitly.**  
   Do not decide qualitatively that a proposal is "too poor for refinement." Perturb known poses and measure exactly how far refinement can recover.

6. **Introduce probabilistic machinery only where uncertainty is real and useful.**  
   The main probabilistic target is partial observability: hidden tail/head, head-tail ambiguity, self-overlap, and failure detection. Do not add generic MCMC to ordinary easy frames.

---

# 2. Primary scientific questions

The project should answer these questions in approximately this order:

1. What is the annotation noise floor for this imaging setup?
2. How well do strong existing worm-specific baselines perform?
3. Is the major remaining error caused by poor spatial localization, inadequate pose representation, or insufficient/incorrect training data?
4. How much does temporal context improve visible-body pose and difficult frames?
5. How well can partial-FOV pose be reconstructed from temporal information?
6. What initialization accuracy is required for differentiable image-space refinement?
7. Does refinement improve local angle accuracy enough to justify its cost?
8. What temporal prior best predicts genuinely hidden anatomy?
9. Can uncertainty identify when pose is poorly constrained?
10. Can adaptive compute preserve the best accuracy while maintaining high throughput?

---

# 3. Evidence hierarchy

Maintain the distinction between evidence types, but change how they are used.

## Tier A — manually annotated real images

This should become the **primary model-selection evidence**.

A modest amount is sufficient if carefully selected. Start with approximately **256 dense centerline annotations**, with at least **64 independently double-annotated** frames.

Stratify the 256 frames approximately across:

- ordinary fully visible locomotion;
- high-curvature bends;
- tight turns / self-contact;
- head close to the boundary;
- tail close to the boundary;
- naturally truncated worms;
- low contrast;
- motion blur;
- unusual illumination/background;
- slow or nearly stationary posture.

Include examples from every usable recording/session rather than concentrating annotation in one movie.

For temporal experiments, prioritize annotations that occur inside short continuous clips so the image context is available. The dense annotation itself can remain focused on the central frame.

### Why this comes first

The first study could not distinguish:

- real localization error;
- classical pseudo-label bias;
- synthetic-to-real domain mismatch;
- representation error;
- architecture error.

Tier A labels resolve this ambiguity cheaply compared with running many GPU experiments.

---

## Tier B — high-confidence classical or model-assisted real labels

Use these for:

- expanding training data;
- learning the distribution of normal postures;
- building texture templates;
- generating synthetic examples;
- weak supervision;
- identifying easy versus hard strata.

Do **not** use Tier B as the sole basis for claims of anatomical accuracy.

---

## Tier C — synthetic/controlled truth

Use Tier C for experiments that require exact latent truth:

- crop/FOV censoring;
- refinement capture basin;
- perturbation recovery;
- calibration under known synthetic noise;
- hidden-body reconstruction;
- stress testing.

Synthetic accuracy does not substitute for Tier A accuracy.

---

# 4. Redefine the accuracy target using human precision

## EXP-001 — Inter-annotator precision and metric calibration

> **Operational single-annotator fallback (2026-08-19):** The project owner
> will not use multiple annotators. Start with 30 balanced unique development
> frames plus 10 delayed, prior-trace-blind repeats by the same person. Treat
> the result as intra-annotator repeatability and directional model-triage
> evidence, not inter-annotator agreement. Expand to 60–90 unique frames if a
> close model comparison approaches that repeatability scale. The stronger
> 64-pair test below remains the preferred protocol if the constraint changes.

### Hypothesis

A realistic scientific accuracy target can be defined relative to disagreement between trained human annotators rather than an arbitrary absolute pixel threshold.

### Test

Double-label at least 64 frames, enriched for both easy and difficult postures.

For each pair of annotations:

- resample both centerlines to 100 uniform arc-length positions;
- resolve forward/reverse orientation symmetrically unless anatomical orientation is independently known;
- calculate:
  - median point distance;
  - p95 point distance;
  - tangent-angle error;
  - endpoint disagreement;
  - body-length disagreement;
  - error versus body position;
  - error versus posture difficulty.

Also make an overlay montage showing the actual magnitude of disagreements.

### Decision

Use the observed annotation distribution to define the project's primary target.

A reasonable default formulation is:

> The final method should approach the human annotation noise floor on ordinary visible frames, with median error no worse than approximately 1.5× inter-annotator median error and no systematic body-region failure.

Keep normalized error in units of estimated worm width as a second primary measure.

### Deliverables

- `human_annotation_agreement.png`
- `human_error_by_body_position.png`
- a table of absolute-pixel and body-width-normalized disagreement;
- revised accuracy targets in the evaluation protocol.

### Why this experiment matters

DeepTangleCrawl explicitly compares model error to disagreement between manual annotators. That is a much stronger way to interpret a numerical RMSD than selecting a pixel threshold without knowing how precisely the centerline can be labeled.

---

# 5. Establish serious external baselines before inventing a new model

## EXP-002 — Existing-method baseline suite

### Hypothesis

At least one worm-specific existing method will substantially outperform the previous global-regression model and will reveal which failure modes actually remain unsolved in this imaging regime.

### Baselines

Run, adapt, or faithfully reproduce as many of the following as practical:

### A. Current classical baseline

Retain the existing high-confidence segmentation/skeletonization method.

Measure it on all Tier A strata, not just frames selected because the classical method succeeded.

This provides:

- an easy-frame accuracy baseline;
- a failure-rate baseline;
- pseudo-label candidates.

### B. WormTracer

Test WormTracer because it is specifically designed to estimate centerlines using **temporal continuity** rather than independent frames.

If binarization is reliable on these NIR videos, it may be surprisingly competitive and gives a valuable non-neural temporal baseline.

Measure:

- centerline accuracy;
- angle accuracy;
- failure rate;
- natural boundary behavior;
- runtime.

### C. WormPose-style model

Reproduce the essential WormPose idea:

- intrinsic tangent-angle output;
- realistic synthetic worm imagery based on real texture;
- head-tail symmetric training loss;
- image-based prediction quality check.

A full compatibility port of the package is not mandatory. The scientific idea is the baseline.

### D. Spatially anchored / DeepTangle-style predictor

Implement or adapt the single-worm equivalent of DeepTangle's central idea:

- maintain a spatial feature grid;
- make pose predictions relative to spatial anchors rather than one globally pooled descriptor;
- use temporal input when appropriate.

Because the present data contain a single large worm rather than thousands of small worms, the implementation can be considerably simpler than DeepTangle.

### Output

Create one baseline comparison figure with:

- Tier A median/p95 centerline error;
- angle error;
- failure rate;
- throughput;
- difficult-frame performance.

Do not choose a winner from aggregate error alone. Include failure rate and visual failure type.

---

# 6. Determine whether spatial localization is the dominant neural failure

The previous model collapsed a 192×256 frame into a 2×2 or 4×4 pooled spatial representation before global pose prediction. The existing audit suggests the worm itself is only on the order of tens of pixels wide in the native image; therefore localization information should be preserved deliberately.

## EXP-003 — Localization-preserving architecture comparison

### Hypothesis

The dominant failure of the previous neural model is not the intrinsic angle representation itself; it is loss of spatial localization caused by the global-regression architecture.

### Keep fixed

- same train/validation grouping;
- same manual validation set;
- same approximate optimizer budget;
- same output centerline resolution;
- same preprocessing;
- same training-data mixture.

### Compare three architectures

#### Variant A — global intrinsic regression

Retain the best prior intrinsic model as the negative/reference baseline.

#### Variant B — dense centerline field

Use a lightweight U-Net/FPN-style network that predicts some combination of:

- centerline probability / distance field;
- tangent orientation as `sin(theta), cos(theta)`;
- endpoint/head-tail heatmaps;
- foreground/support probability.

Recover a smooth spline from the dense output.

The exact decoder may evolve, but the network must retain high spatial resolution.

#### Variant C — anchored centerline prediction

Use a DeepTangle-like spatial grid. Grid cells predict:

- confidence;
- center/anchor offset;
- intrinsic shape coefficients or relative centerline coordinates;
- optional length/orientation.

For the single-worm case, only the best candidate needs to survive.

### Resolution ablation

For the leading spatial architecture, compare approximately:

- 256×192;
- 384×288;
- 512×384.

Do not scale higher unless the accuracy curve is still improving enough to justify it.

### Key plots

- point error vs input resolution;
- angle error vs resolution;
- error vs body position;
- overlays at identical frames;
- accuracy vs GPU throughput.

### Decision

If a spatially explicit model dramatically improves localization, retain it even if its initial angle parameterization is not yet optimal.

If no localization-preserving architecture improves Tier A error, investigate supervision/domain mismatch before scaling model size.

---

# 7. Test training-data realism and scale

The first study used a very small synthetic set generated from simple analytic tubes. WormPose and DeepTangle demonstrate that synthetic training can work, but only when the generative process produces enough diversity and realistic appearance.

## EXP-004 — Synthetic data realism

### Hypothesis

Real-texture synthetic augmentation substantially reduces the sim-to-real error relative to simple rendered tubes.

### Build three matched training sources

#### Dataset A — analytic tubes

Use the existing renderer.

This is the control.

#### Dataset B — real-texture warping

From high-confidence full-body real frames:

1. extract the worm texture along its centerline;
2. estimate width profile;
3. warp short cross-sectional/rectangular patches onto a new target centerline;
4. blend overlaps;
5. composite onto real NIR backgrounds or background estimates;
6. apply realistic:
   - blur;
   - intensity shift;
   - local background gradients;
   - sensor noise;
   - scale variation;
   - translations;
   - modest defocus.

This follows the core strategy used by WormPose.

#### Dataset C — mixed real + real-texture synthetic

Combine:

- manually labeled Tier A;
- high-confidence Tier B;
- Dataset B.

### Data-scale ablation

For the best synthetic process, compare roughly:

- 5k examples;
- 25k;
- 100k;
- 250k or until validation performance saturates.

Generation should be streaming/on-the-fly or cached under `/temp_data4/alex/...`; do not store a huge redundant image corpus in the repository.

### Metrics

Most important:

- Tier A error;
- failure rate;
- real/synthetic feature or intensity mismatch diagnostics;
- error on rare curvature strata.

### Decision

Do not continue scaling synthetic data if Tier A performance saturates while synthetic validation continues improving. That pattern indicates domain mismatch rather than insufficient quantity.

---

# 8. Representation experiment: spline angles vs learned posture basis

Do this after spatial localization is working.

## EXP-005 — Intrinsic pose representation

### Hypothesis

A compact intrinsic shape representation produces smoother, more data-efficient predictions without sacrificing local body-angle fidelity.

### Compare

1. direct 100-point centerline output from a spatially anchored model;
2. 16–24 tangent-angle spline coefficients;
3. PCA/eigenworm-style latent representation;
4. optionally a larger learned PCA basis (e.g. 32–72 dimensions) if tight/coiled cases require it.

### Important caution

Four classical eigenworms capture much ordinary posture variance, but this does **not** imply that four coefficients are sufficient for accurate pixel-level reconstruction in unusual, tightly curved, or partially observed frames.

DeepTangleCrawl used a substantially larger PCA representation for crawling worms.

### Metrics

- centerline error;
- tangent-angle error;
- local curvature error;
- error specifically on tight turns;
- reconstruction error when fitting the representation directly to ground-truth centerlines;
- parameter count;
- throughput.

### First perform an oracle test

Before training a network, fit each representation directly to Tier A centerlines and determine its **representation floor**.

If 16 spline coefficients cannot reconstruct the labels to near human precision, do not ask a neural network to overcome that representational limit.

This cheap experiment should precede training.

---

# 9. Test temporal context early, not after a perfect single-frame model

## EXP-006 — 1-frame vs 5-frame vs 11-frame temporal context

### Hypothesis

Temporal context materially improves pose accuracy and failure rate, particularly for tight bends, ambiguous head-tail orientation, low-contrast frames, and FOV truncation.

### Architecture

Use the best localization-preserving spatial architecture.

Start with the simplest temporal formulation:

- encode frames with a shared 2D encoder;
- fuse local features across time with:
  - temporal convolutions, or
  - a small attention/fusion block.

Avoid a large video transformer unless simpler fusion saturates.

### Compare

- 1 frame;
- centered 5-frame window;
- centered 11-frame window.

Because the target workflow is offline analysis of recordings, centered/acausal context is allowed. Record the look-ahead explicitly.

### Also test temporal sampling interval

At 20 Hz, adjacent frames may be very similar during slow behavior.

For the 11-frame model, compare at least:

- consecutive frames;
- stride 2;
- optionally a mixed temporal-scale training scheme.

DeepTangleCrawl intentionally samples clips over different time scales so that temporal input remains informative during both fast and slow behavior.

### Metrics

Stratify by:

- ordinary locomotion;
- high curvature;
- stationary/slow behavior;
- head/tail ambiguity;
- low contrast;
- natural FOV boundary;
- synthetic FOV truncation.

Report:

- centerline error;
- angle error;
- frame failure rate;
- temporal jitter;
- head-tail flip rate;
- throughput.

### Acceptance logic

Temporal modeling does **not** require the 1-frame model to pass a fixed absolute gate.

Retain temporal context when it provides a reproducible improvement on Tier A or crop benchmarks and does not create unacceptable smoothing of fast posture changes.

A reasonable practical threshold is approximately:

- ≥10% relative improvement in the primary difficult-frame metric, or
- a large reduction in catastrophic failures/head-tail flips,

with no important degradation of ordinary-frame accuracy.

---

# 10. Build the partial-FOV benchmark correctly

## EXP-007 — Controlled head/tail censoring

### Hypothesis

Temporal models and explicit FOV handling can recover the hidden part of the worm better than naive independent-frame extrapolation.

### Construct two benchmark families

## A. Analytic/synthetic truth

Generate sequences with exact latent centerlines and move the camera boundary smoothly so that:

- 0%;
- 5%;
- 10%;
- 20%;
- 30%;
- 40%

of either head or tail becomes hidden.

This gives exact hidden truth.

## B. Real-appearance controlled censoring

Start from fully visible real sequences with trustworthy centerlines.

Create a virtual camera transformation that moves a real worm toward/across an image boundary while preserving real worm texture/background wherever possible.

The goal is not photorealistic generation of the unseen region—the unseen region is deliberately absent. The goal is realistic **visible evidence** plus known original full-body pose.

### Compare simple baselines

1. final visible tangent extrapolation;
2. last known full-body shape transformed rigidly;
3. autoregressive latent prediction from previous frames;
4. 5-frame temporal neural model;
5. 11-frame temporal neural model;
6. offline temporal smoother using past + future context.

### Metrics

Separately report:

- visible-body error;
- boundary-band error;
- hidden-body error;
- error vs hidden fraction;
- angle error vs normalized body position;
- uncertainty vs hidden fraction.

### Key figure

Make a heatmap:

- x-axis: time;
- y-axis: normalized body position;
- color: angle error or uncertainty;

with the visible/hidden boundary overlaid.

This should become one of the central project figures.

---

# 11. Measure the differentiable-refinement capture basin

This experiment should happen **before** deciding whether refinement is useful.

## EXP-008 — Refinement perturbation recovery

### Hypothesis

Differentiable rendering can recover subpixel/local-angle accuracy from proposals within a measurable neighborhood of the correct centerline.

### Ground-truth starting points

Use:

- Tier A centerlines where a good image model can be evaluated;
- realistic synthetic images with exact centerlines.

### Generate controlled initialization errors

Independently perturb:

#### Translation

- 1 px
- 2 px
- 4 px
- 8 px
- 16 px
- 32 px
- 64 px

in original-image coordinates.

#### Rotation

For example:

- 1°
- 2°
- 5°
- 10°
- 20°
- 40°

#### Length/scale

For example:

- ±2%;
- ±5%;
- ±10%;
- ±20%.

#### Shape

Perturb one or more spline/PCA coefficients at increasing standardized magnitudes.

#### Combined perturbations

After the one-dimensional tests, test realistic mixtures.

### Compare image objectives

At minimum:

1. raw robust pixel residual;
2. pixel + image gradient/edge residual;
3. local cross-sectional/tube likelihood around the centerline;
4. optional learned feature likelihood only if the simple likelihoods fail on real images.

### Optimize

Compare:

- 1;
- 3;
- 5;
- 10

refinement steps.

### Main output

Plot **probability of successful recovery vs initialization error**.

Define successful recovery relative to Tier A/human-scale accuracy.

This plot defines the actual refinement capture basin.

### Why this is important

The MICCAI 2025 Splender work provides direct evidence that differentiable rendering can improve C. elegans spline localization and reach subpixel accuracy, but initialization remains important. The correct experimental response is therefore to measure that dependency quantitatively.

---

# 12. Apply refinement to the learned proposals

## EXP-009 — Proposal + refinement

### Hypothesis

A small number of differentiable refinement steps improves visible-body centerline and angle accuracy enough to justify the added runtime.

### Prerequisite

Do this for any proposal model whose error distribution places a meaningful fraction of frames inside the capture basin measured in EXP-008.

There is no need for the proposal to meet an arbitrary final-accuracy gate first.

### Compare

- proposal only;
- +1 step;
- +3 steps;
- +5 steps.

Use identical frames.

### Metrics

- Tier A centerline error;
- tangent-angle error;
- endpoint accuracy;
- boundary accuracy;
- failure/catastrophic drift rate;
- milliseconds per frame;
- GPU memory.

### Key plot

Accuracy improvement versus added milliseconds.

### Failure criterion

Reject a refinement configuration if it:

- frequently converges to the wrong structure;
- improves synthetic images but not real Tier A images;
- adds substantial latency for negligible biological-angle improvement.

---

# 13. Add explicit FOV censoring to the renderer

## EXP-010 — FOV-aware image likelihood

### Hypothesis

An image-space objective that correctly censors off-screen anatomy avoids shortening or bending the estimated worm toward the image boundary.

### Compare

1. naive image reconstruction that implicitly expects the whole worm to be visible;
2. FOV-censored likelihood;
3. FOV-censored likelihood + temporal shape prior.

### Test

Use the controlled crop benchmark.

Pay special attention to:

- the last 10–20% of visible anatomy;
- predicted body length;
- tangent direction at the boundary;
- whether the model collapses the hidden segment.

### Expected outcome

This experiment tests one of the core benefits of the probabilistic/inverse-graphics formulation: the latent worm is allowed to continue beyond what the camera observes.

---

# 14. Temporal prior for hidden anatomy

Once visible-body localization is working, infer genuinely unobserved posture.

## EXP-011 — Dynamics prior

### Hypothesis

A learned or fitted low-dimensional dynamics prior predicts hidden tail/head pose better than simple geometric extrapolation.

### Start simple

Represent posture with the best spline/PCA latent from EXP-005.

Compare:

1. constant-shape prior;
2. constant-velocity latent dynamics;
3. AR(1);
4. AR(2);
5. small MLP/GRU dynamics model.

For offline inference, also test a smoother that uses future observations.

### Do not start with a large recurrent model

C. elegans posture is strongly low-dimensional and temporally structured. A compact prior may be sufficient and will be easier to interpret/calibrate.

### Metrics

On controlled crop sequences:

- hidden-body point error;
- hidden-body angle error;
- error vs hidden fraction;
- recovery after the worm re-enters the FOV;
- oversmoothing of genuine fast turns.

---

# 15. Ambiguity-aware probabilistic inference

## EXP-012 — Multiple hypotheses only when needed

### Hypothesis

Maintaining a small number of competing pose hypotheses on ambiguous frames improves difficult-case accuracy without paying that cost on every frame.

### Candidate ambiguity triggers

- low proposal confidence;
- poor image reconstruction after refinement;
- strong disagreement with temporal prior;
- head-tail ambiguity;
- tight self-overlap;
- significant fraction outside FOV;
- unusually large refinement update.

### Hypotheses to maintain

Examples:

- normal orientation;
- head-tail flipped orientation;
- 4–8 perturbed spline hypotheses;
- multiple hidden-tail continuations.

### Compare

- deterministic best pose;
- always-on 8 hypotheses;
- adaptive 8 hypotheses only on flagged frames.

### Evaluation

Measure:

- difficult-frame accuracy;
- catastrophic failure rate;
- percentage of frames requiring extra compute;
- mean throughput;
- p95/worst-case latency.

This is the point where the project becomes meaningfully similar to the 3DNEL/3DP3 philosophy: fast bottom-up proposals plus structured probabilistic inference only where image evidence is ambiguous.

---

# 16. Uncertainty calibration

## EXP-013 — Does uncertainty mean anything?

### Hypothesis

The system's uncertainty increases in body regions/frames where true pose error is larger.

### Candidate uncertainty mechanisms

Test in increasing complexity:

1. deterministic residual/image-likelihood quality score;
2. heteroscedastic output variance;
3. small deep ensemble;
4. local particle/hypothesis spread.

### Evaluate

On Tier A and controlled crop truth:

- error vs predicted uncertainty;
- reliability/coverage plots;
- calibration stratified by:
  - visible vs hidden body;
  - FOV boundary proximity;
  - posture curvature;
  - low contrast;
  - self-overlap.

### Important semantic rule

Keep these concepts separate:

- geometric `in_fov`;
- image support/evidence;
- pose uncertainty.

A point may be inside the FOV yet poorly constrained because of blur or overlap.

---

# 17. Adaptive compute

## EXP-014 — Accuracy/throughput Pareto optimization

### Hypothesis

Most ordinary frames require only the neural proposal, while difficult frames benefit from refinement or multiple hypotheses.

### Compare

1. proposal only;
2. proposal + fixed refinement on every frame;
3. proposal + adaptive refinement;
4. proposal + adaptive refinement + particles.

### Report

- average frames/s;
- batch-1 latency;
- p95 latency;
- worst difficult-frame latency;
- GPU memory;
- fraction of frames taking each path;
- accuracy on ordinary vs difficult strata.

### Throughput goals

Use these as engineering goals rather than scientific truth:

- **hard floor:** >20 fps storage-inclusive on the target GPU;
- **preferred final offline throughput:** substantially above acquisition rate;
- **proposal-only target:** ideally >100 fps so later pipeline stages have headroom.

Do not trade a large biological-angle improvement away merely to maximize a microbenchmark.

---

# 18. Final validation experiment

## EXP-015 — Frozen full-system test

Run this only after architecture, training recipe, refinement, uncertainty, and thresholds are frozen.

### Evaluate

1. all development sessions separately;
2. the protected independent/shifted holdout;
3. complete ~16-minute recordings;
4. natural partial-FOV events;
5. natural tight-turn events;
6. storage-inclusive throughput.

### Produce full-recording diagnostics

For representative recordings:

- centerline-overlay video;
- body-angle heatmap over the full recording;
- uncertainty heatmap;
- flagged failure frames;
- distribution of visible fraction;
- temporal discontinuity/flip diagnostics.

### Biological plausibility checks

Inspect:

- body length over time;
- angle velocity;
- curvature distribution;
- continuity through reversals;
- continuity as tail exits and re-enters FOV.

These are not substitutes for labels, but they reveal silent pipeline failures.

---

# 19. Recommended experiment order

Use the following order unless evidence strongly redirects the project.

```text
EXP-001  Human annotation / noise floor
   |
EXP-002  Existing worm-specific baselines
   |
EXP-003  Localization-preserving architecture
   |
EXP-004  Synthetic realism + data scale
   |
EXP-005  Pose representation floor + learned comparison
   |
   +----------------------+
   |                      |
EXP-006 Temporal       EXP-008 Refinement capture basin
   |                      |
EXP-007 Crop/FOV          |
   |                      |
   +----------+-----------+
              |
        EXP-009 Proposal + refinement
              |
        EXP-010 FOV-aware likelihood
              |
        EXP-011 Temporal dynamics prior
              |
        EXP-012 Multiple hypotheses
              |
        EXP-013 Uncertainty calibration
              |
        EXP-014 Adaptive compute
              |
        EXP-015 Frozen holdout / full recordings
```

The two central branches—**temporal modeling** and **refinement capture-basin measurement**—can proceed independently once a reasonable data/evaluation framework exists.

---

# 20. Decision rules that should replace the previous hard gate

## Do not use this rule

> "Single-frame proposal misses the final accuracy target, therefore temporal/refinement/probabilistic work is forbidden."

That rule throws away scientifically plausible branches.

## Use branch-specific rules instead

### Temporal branch

Temporal modeling may proceed whenever a valid single-frame baseline exists, even if it is not final-quality.

Question:

> Does context improve the same architecture on paired cases?

### Refinement branch

Refinement depends on **capture basin**, not an arbitrary proposal score.

Question:

> What fraction of proposal frames lie inside the empirically measured basin of attraction?

### Probabilistic hidden-body branch

Proceed when visible-body pose is good enough that hidden-body performance can be meaningfully isolated.

Question:

> Does the prior improve genuinely hidden anatomy without degrading observed anatomy?

### Final deployment

Only the final integrated model needs to meet the full scientific acceptance criteria.

---

# 21. Experiment reporting requirements

Every experiment should produce a compact scientific record:

## Hypothesis

One or two sentences.

## Why this is plausible

Cite:

- a prior paper;
- an observed failure from this project;
- or both.

An agent should not invent an architecture experiment without writing down the prior reason it might work.

## Controlled change

State what changed and what remained fixed.

## Result

Include:

- main numerical metrics;
- uncertainty/paired comparison where useful;
- throughput;
- random examples;
- worst examples.

## Visual evidence

At least one figure must be capable of answering the hypothesis by itself.

## Conclusion

Use:

- `SUPPORTED`
- `NOT SUPPORTED`
- `PARTIALLY SUPPORTED`
- `INCONCLUSIVE`

## Consequence

Explain exactly what experiment this result motivates next.

---

# 22. Core final figures

The final research report should be heavily visual.

At minimum produce:

1. **Inter-annotator accuracy floor**
2. **Baseline method comparison**
3. **Input resolution / localization architecture curve**
4. **Synthetic-data realism comparison**
5. **Data-scaling curve**
6. **1/5/11-frame temporal ablation**
7. **Error vs body position**
8. **Error vs hidden fraction**
9. **Body-angle error heatmap during controlled FOV exit**
10. **Refinement capture-basin plot**
11. **Before/after refinement overlays**
12. **Accuracy vs refinement milliseconds**
13. **Uncertainty calibration**
14. **Adaptive-compute accuracy/throughput Pareto curve**
15. **Natural failure montage**
16. **Full-recording body-angle heatmap**
17. **High-level hypothesis → experiment → result → decision flow diagram**

---

# 23. Suggested implementation priorities

The research code should remain simple enough that conclusions are about the algorithms rather than framework complexity.

For the learned path, prefer:

- PyTorch;
- PyTorch Lightning;
- NumPy;
- h5py;
- Matplotlib.

Use PyTorch operations (`grid_sample`, interpolation, convolution, etc.) for image warping/rendering when practical.

A research-only dependency is acceptable if it enables a strong published baseline, but avoid making that dependency part of the final runtime unless it wins experimentally.

Large generated data/checkpoints/videos continue to belong under:

```text
/temp_data4/alex/...
```

with repository symlinks where useful.

---

# 24. Further reading

The agent should read these before making major architecture decisions.

## A. Worm-specific pose estimation

### WormPose — synthetic realistic imagery + tangent-angle regression

**Hebert et al. (2021), "WormPose: Image synthesis and convolutional networks for pose estimation in C. elegans"**

Paper:  
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008914

Code:  
https://github.com/iteal/wormpose

Why it matters:

- directly models worm posture with tangent angles;
- uses real worm texture warped into synthetic postures;
- uses synthetic data at large scale;
- treats head-tail ambiguity explicitly;
- performs an image-based post-prediction quality check.

Read this before implementing EXP-004 or EXP-005.

---

### DeepTangle — spatially anchored centerline prediction + temporal input

**Alonso & Kirkegaard (2023), "Fast detection of slender bodies in high density microscopy data"**

Paper:  
https://www.nature.com/articles/s42003-023-05098-1

Code:  
https://github.com/kirkegaardlab/deeptangle

Why it matters:

- uses an 11-frame stack;
- retains a spatially anchored output grid;
- predicts centerlines directly;
- demonstrates high-throughput inference;
- provides a strong counterexample to globally pooling away spatial location.

Read this before EXP-003 and EXP-006.

---

### DeepTangleCrawl — crawling worms, temporal clips, manual hard cases

**Weheliye et al. (2025), "A neural network model enables worm tracking in challenging conditions and increases signal-to-noise ratio in phenotypic screens"**

Paper:  
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013345

Code / training implementation:  
https://github.com/WeheliyeHashi/tierpsy_tracker_2.0

Annotation tool:  
https://github.com/WeheliyeHashi/Worm_annotation

Dataset / trained models:  
https://zenodo.org/records/15526615

Why it matters:

- specifically addresses crawling worms;
- trains on 11-frame clips;
- combines easy tracker-derived examples with manually annotated difficult cases;
- compares DeepTangleCrawl against Omnipose and a PAF/landmark method;
- explicitly reports human annotation disagreement;
- uses a much richer latent posture representation for crawling behavior.

Read before EXP-001, EXP-002, EXP-003, and EXP-006.

---

### WormTracer — explicit temporal-continuity baseline

**Kuze et al. (2026), "WormTracer: A precise method for worm posture analysis using temporal continuity"**

Paper:  
https://www.sciencedirect.com/science/article/pii/S0165027025002882

Code:  
https://github.com/yuichiiino1/WormTracer

Why it matters:

- estimates centerlines across image sequences rather than independently;
- requires no learned training data;
- is directly relevant to whether temporal continuity alone solves much of this problem;
- reported comparisons include WormPose, EigenWormTracker, and DeepTangleCrawl.

Read and benchmark before investing heavily in a new temporal neural model.

---

## B. Differentiable rendering / refinement

### Splender — C. elegans spline refinement with differentiable rendering

**Zdyb, Alonso & Kirkegaard (MICCAI 2025), "Spline refinement with differentiable rendering"**

Paper:  
https://papers.miccai.org/miccai-2025/0860-Paper1793.html

PDF:  
https://papers.miccai.org/miccai-2025/paper/1793_paper.pdf

Code:  
https://github.com/kirkegaardlab/splender

Why it matters:

- directly refines predicted C. elegans splines in image space;
- is training-free;
- demonstrates subpixel refinement;
- makes initialization sensitivity an empirical question;
- is the closest existing work to the proposed 2D inverse-graphics refinement stage.

Read before EXP-008, EXP-009, and EXP-010.

---

## C. Probabilistic inverse graphics

### 3DNEL

**Zhou et al. (ICCV 2023), "3D Neural Embedding Likelihood: Probabilistic Inverse Graphics for Robust 6D Pose Estimation"**

Paper:  
https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_3D_Neural_Embedding_Likelihood_Probabilistic_Inverse_Graphics_for_Robust_6D_ICCV_2023_paper.html

Why it matters:

- combines a learned proposal/embedding with a generative probabilistic likelihood;
- represents ambiguity rather than forcing a single estimate;
- performs pose tracking under occlusion;
- motivates using expensive probabilistic inference selectively rather than replacing amortized inference.

Read before EXP-012 and EXP-013.

---

### 3DP3

**Gothoskar et al. (NeurIPS 2021), "3DP3: 3D Scene Perception via Probabilistic Programming"**

Paper:  
https://proceedings.neurips.cc/paper/2021/hash/4fc66104f8ada6257fa55f29a2a567c7-Abstract.html

Why it matters:

- illustrates the proposal + generative model + structured inference pattern;
- handles partial observability and occlusion explicitly;
- is useful conceptually even though the present problem is much simpler and 2D.

Read for overall system design, not as an implementation template.

---

## D. Worm posture representation and dynamics

### Eigenworms / low-dimensional posture

**Stephens et al. (2008), "Dimensionality and Dynamics in the Behavior of C. elegans"**

Paper:  
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028

Why it matters:

- establishes that ordinary worm posture occupies a strongly low-dimensional space;
- motivates compact temporal dynamics models;
- provides a useful prior, but should not be interpreted as proof that four coefficients are enough for pixel-accurate difficult-pose reconstruction.

Read before EXP-005 and EXP-011.

---

### Coiled shapes and posture-space tracking

**"Resolving coiled shapes reveals new reorientation behaviors in C. elegans"**

eLife article:  
https://elifesciences.org/articles/17227

Why it matters:

- demonstrates that self-overlapping configurations require explicit reasoning beyond naive skeletonization;
- connects difficult shapes to low-dimensional posture representations.

Useful for representation and difficult-case evaluation.

---

### Markovian posture dynamics

**Costa et al. (2024), "A Markovian dynamics for Caenorhabditis elegans behavior across scales"**

Open-access article:  
https://pmc.ncbi.nlm.nih.gov/articles/PMC11317559/

Why it matters:

- supports the idea that surprisingly simple stochastic dynamics can capture meaningful posture evolution;
- useful background before replacing a simple AR prior with a large learned recurrent model.

Read before EXP-011.

---

# 25. What the next autonomous run should do first

The next agent should **not** immediately train another variant of the existing model.

Its first actions should be:

1. preserve the existing repository and negative results;
2. create the Tier A annotation set and measure inter-annotator precision;
3. benchmark the existing classical method and WormTracer on those labels;
4. implement one localization-preserving neural baseline;
5. build a real-texture synthetic generator;
6. compare 1/5/11-frame input on the same architecture;
7. independently measure the differentiable-refinement capture basin;
8. use those results to decide whether the final system should emphasize:
   - temporal neural inference;
   - differentiable refinement;
   - or both.

Only after those experiments should the project invest in calibrated probabilistic hidden-tail inference.

---

# 26. Expected likely architecture if the hypotheses are supported

This is a **working hypothesis**, not a mandated final design:

```text
full-resolution / moderately downsampled NIR clip
                  |
                  v
localization-preserving spatial encoder
                  |
          5–11 frame fusion
                  |
                  v
spatially anchored intrinsic pose proposal
(centerline + tangent representation + support/confidence)
                  |
          +-------+-------+
          |               |
       easy frame       ambiguous frame
          |               |
          |        differentiable refinement
          |        + temporal prior
          |        + optional particles
          |               |
          +-------+-------+
                  |
                  v
100-point centerline
tangent angle / curvature
image support
head-tail probability
calibrated uncertainty
                  |
                  v
streamed HDF5 output
```

The purpose of the experiment program is to determine which boxes in this diagram actually earn their complexity.

---

# 27. Final success criterion

The project succeeds when it produces a method that:

1. approaches the Tier A human annotation noise floor on ordinary visible worms;
2. materially improves difficult/failure cases relative to strong worm-specific baselines;
3. degrades gracefully as anatomy leaves the FOV;
4. clearly distinguishes observed pose from inferred hidden pose;
5. provides uncertainty that correlates with actual error;
6. processes recordings comfortably faster than their 20 Hz acquisition rate on the target GPU;
7. has a visual and quantitative experiment trail showing why every major component was retained.

A negative result is acceptable for any individual hypothesis. The research process should stop only when the remaining plausible branches have been tested or when the evidence shows that additional complexity is unlikely to improve the biological measurement.
