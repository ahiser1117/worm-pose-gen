# Autonomous Experiment Plan: Segmentation-Anchored Probabilistic *C. elegans* Pose Tracking

## 0. Purpose and relationship to the previous plans

This document defines the next research program for `worm-pose-gen`.

It replaces the **direct neural pose-estimation search** as the primary research direction with a simpler and more structured approach:

```text
NIR video
   ↓
pretrained foreground segmentation
   ↓
classical mask cleanup
   ↓
easy-frame skeletonization
   ↓
high-confidence pose anchors
   ↓
learn a compact generative worm model and temporal dynamics
   ↓
sequential Monte Carlo / particle smoothing
   ↓
infer pose through self-intersections, coils, truncation, and other ambiguous intervals
```

The central hypothesis is:

> Most frames do not require a learned pose estimator at all. If the worm can be segmented reliably, classical skeletonization can recover accurate centerlines on easy frames. Those easy frames can be used to learn the worm's posture distribution, width profile, and short-timescale dynamics. Difficult frames can then be treated as a low-dimensional probabilistic trajectory-inference problem, with the segmentation mask as the observation and nearby easy frames as strong temporal anchors.

This document should be read together with:

- `worm_pose_agent_orchestrator.md`
- `worm_pose_scientific_experiment_plan.md`

### Requirements inherited from `worm_pose_agent_orchestrator.md`

Unless explicitly superseded here, retain its requirements for:

- `uv`-managed Python environment;
- PyTorch / PyTorch Lightning integration;
- HDF5 streaming and read-only source handling;
- CUDA usage and reproducibility;
- leakage-safe session/recording splits;
- experiment IDs and immutable experiment records;
- subagent orchestration;
- experiment logging and decision logging;
- visual-first `EXPERIMENT_FLOW.md`;
- failure-mode tracking;
- performance benchmarking;
- final HDF5 output schema;
- large-file storage beneath `/temp_data4/alex/...`;
- final documentation and packaging.

Do **not** discard the previous repository or its negative results. Preserve them as evidence that the earlier global pose-regression architecture was unsuccessful. This document begins a new scientific branch.

### Requirements inherited from `worm_pose_scientific_experiment_plan.md`

Retain especially:

- the evidence hierarchy separating manual truth, proxy labels, and controlled synthetic truth;
- inter-annotator measurement as the basis for meaningful final accuracy targets;
- controlled head/tail FOV-censoring benchmarks;
- accuracy-versus-throughput reporting;
- uncertainty semantics;
- strong visual reporting;
- the final requirement that every retained component must earn its complexity experimentally.

The prior document's direct learned-pose architecture search is now **secondary/fallback work**, not the default next step.

---

# 1. Mission

Build a high-throughput probabilistic tracker for 2D *C. elegans* centerline and body-angle estimation from approximately 20 Hz NIR recordings.

The final scientific outputs should include, for every frame:

- `centerline_xy`, nominally 100 uniformly spaced body points;
- tangent angle as a function of normalized body position;
- curvature as a function of body position;
- head/tail orientation probability if identifiable;
- geometric in-FOV status;
- image-support or observation-support status;
- pose uncertainty;
- a frame-level quality/confidence value;
- enough posterior information to identify genuinely ambiguous frames rather than forcing false certainty.

The method should perform especially well when:

- the worm is ordinary and non-overlapping;
- the worm becomes tightly curved;
- the worm self-intersects and the segmentation becomes a blob;
- the head or tail leaves the FOV;
- classical skeletonization temporarily fails;
- the worm later returns to an easy configuration.

The target recordings are offline. Therefore, the final method **may and should use future frames** when doing so materially improves difficult-interval inference. Online causal inference can be added later as a separate mode if useful.

---

# 2. Core scientific model

The system should explicitly separate four problems:

```text
1. APPEARANCE
   "Which pixels are worm?"
        ↓
   pretrained segmentation network

2. EASY GEOMETRY
   "What is the centerline when topology is unambiguous?"
        ↓
   classical cleanup + skeletonization

3. WORM PRIOR / DYNAMICS
   "What shapes can this worm take, and how quickly can they change?"
        ↓
   generative latent pose model learned from easy frames

4. AMBIGUOUS INFERENCE
   "Which latent trajectory best explains a difficult mask between known poses?"
        ↓
   SMC / resample-move / smoothing
```

Do not conflate these tasks.

The segmentation network should not be required to infer centerline topology.

The generative pose model should not be required to explain raw NIR appearance initially.

Classical skeletonization should not be forced to solve self-intersections.

SMC should not be used on every easy frame if a deterministic high-confidence measurement already exists.

---

# 3. Primary hypotheses

The autonomous research program should test these hypotheses.

## H1 — Segmentation is sufficiently reliable

A pretrained segmentation network plus simple cleanup can produce a stable worm foreground probability/mask on most frames, including many frames where centerline topology is ambiguous.

## H2 — Easy masks provide accurate centerline anchors

When the cleaned segmentation has simple topology, classical skeletonization can recover a centerline close to manual annotation precision.

## H3 — Worm posture is low dimensional enough for efficient inference

A compact latent posture representation learned from easy frames can reconstruct easy centerlines to approximately the classical/manual noise floor.

## H4 — Width is mostly stable

A recording/animal-specific mean width profile plus very low-dimensional width variation can explain mask geometry without allowing width to absorb pose errors.

## H5 — Short-timescale posture dynamics are strongly predictive

At 20 Hz, the next-frame latent pose distribution conditioned on recent easy frames is sufficiently narrow to guide inference through short ambiguous intervals.

## H6 — A rendered latent pose can explain the segmentation observation

A differentiable or efficiently rasterized worm-tube model can reproduce cleaned masks/probability maps accurately enough to define a useful observation likelihood.

## H7 — Sequential Monte Carlo can survive ambiguous intervals

Starting from an accurate easy-frame anchor, SMC can maintain plausible pose hypotheses through self-intersection or truncation until the pose becomes observable again.

## H8 — Future easy frames disambiguate difficult bouts

Bidirectional or smoothing inference using easy anchors on both sides of a difficult interval materially outperforms forward-only filtering.

## H9 — Multiple posterior modes are scientifically useful

Some segmentation blobs admit multiple plausible centerline topologies. Retaining several hypotheses until future frames resolve them is more accurate and better calibrated than committing immediately.

## H10 — Raw NIR appearance is only needed as a tie-breaker

Mask-space inference should solve most difficult intervals. Raw image evidence should be added only if controlled experiments show persistent mask-level ambiguity.

---

# 4. Keep the segmentation probability map, not only the binary mask

For every frame, retain both:

```text
P_t(x, y) = predicted probability that pixel (x, y) belongs to the worm
```

and a cleaned binary mask:

```text
M_t(x, y) = cleaned thresholded foreground mask
```

Use them differently.

## Binary mask uses

Use `M_t` for:

- connected-component cleanup;
- hole filling;
- removal of small objects;
- topology measurements;
- skeletonization;
- branch/end-point counting;
- easy/difficult frame classification;
- geometric diagnostics.

## Soft probability uses

Use `P_t` for:

- probabilistic observation likelihoods;
- subpixel boundary information;
- distinguishing uncertain boundary pixels from confident background;
- optional gradient-based pose refinement.

Do not throw away segmentation logits/probabilities after thresholding.

The output of the segmentation stage should therefore be cached in a compact streamed representation when doing so saves repeated inference. Large caches must live beneath `/temp_data4/alex/...`.

---

# 5. Phase A — Validate segmentation before building the probabilistic tracker

## EXP-SMC-001 — Segmentation quality audit

### Question

Is the segmentation network good enough that downstream inference should operate primarily on segmentation rather than raw NIR images?

### Data

Use representative frames from every usable recording and all major difficulty strata:

- ordinary;
- high curvature;
- self-overlap;
- tight coil;
- head at boundary;
- tail at boundary;
- natural partial FOV;
- low contrast;
- motion blur.

Use available manual mask annotations if present. If none exist, manually annotate a modest stratified set or derive mask truth from centerline/width annotation where appropriate.

### Evaluate

At minimum:

- foreground IoU/Dice;
- boundary distance;
- false foreground components;
- holes;
- area error;
- centroid error;
- stability across adjacent frames;
- performance at self-intersections;
- performance at FOV boundaries.

### Important qualitative review

Create montages containing:

```text
raw NIR
segmentation probability
binary mask
cleaned mask
mask boundary overlay
```

for random and worst cases.

### Decision

If segmentation fails grossly on difficult frames, do **not** immediately abandon the probabilistic approach.

First determine whether the failure is:

- threshold/cleanup related;
- model confidence but correct shape;
- genuine foreground/background error.

A segmentation model need not know centerline topology. A blob is an acceptable observation if it contains the correct worm material.

### Deliverables

- segmentation audit report;
- probability/mask montages;
- per-recording metrics;
- runtime benchmark;
- explicit threshold/cleanup configuration.

---

# 6. Phase B — Define classical easy-frame anchors

## EXP-SMC-002 — Easy-frame detector and skeleton accuracy

### Objective

Create a conservative system that answers:

> Is this segmentation topologically simple enough that its skeleton can be trusted as a pose observation?

A frame rejected as difficult is not a failure. Precision of anchor selection matters more than recall.

## Recommended cleanup

Begin with the simplest cleanup that works:

- retain the primary worm component using area/proximity logic;
- fill small internal holes;
- remove tiny foreground islands;
- optionally apply a very small closing/opening operation;
- avoid morphology aggressive enough to alter body shape.

Record every operation.

## Skeleton extraction

From the cleaned mask:

1. skeletonize;
2. identify endpoints and branchpoints;
3. identify any cycles;
4. extract the longest valid path between the two anatomical ends when topology is simple;
5. smooth/resample to approximately 100 arc-length positions;
6. estimate local width from the foreground distance transform;
7. calculate tangent angles.

## Candidate anchor criteria

A high-confidence easy frame should initially satisfy most or all of:

- one dominant connected foreground component;
- exactly two meaningful skeleton endpoints;
- zero meaningful branchpoints;
- no cycles;
- plausible foreground area;
- plausible centerline length;
- plausible width distribution;
- no implausible abrupt width jumps;
- low mask-to-render residual after reconstructing the mask from extracted centerline/width;
- temporal consistency with nearby high-confidence frames;
- no large unmodeled boundary truncation unless boundary anchors are handled explicitly.

Do not hard-code biological thresholds without first examining their empirical distributions.

## Manual validation

Randomly sample:

- accepted easy frames;
- rejected frames;
- accepted frames nearest each threshold.

Manually determine whether accepted anchors have correct centerline topology.

### Primary quantity

Estimate:

```text
P(correct centerline | frame accepted as easy)
```

This should be extremely high.

A smaller number of trusted anchors is preferable to contaminating the generative model with wrong topology.

### Deliverables

- easy/difficult frame timeline for each recording;
- distributions of topology/quality metrics;
- random accepted overlays;
- worst accepted overlays;
- rejected-frame montage;
- anchor precision estimate;
- fraction of frames classified easy.

---

# 7. Phase C — Build an empirical posture dataset from anchors

Once the easy-frame detector is validated, process the recordings and build an anchor-pose dataset.

Each anchor should contain:

```text
recording_id
frame_index
timestamp
centerline_xy[100, 2]
tangent_angle[100]
estimated_width[100]
body_length
centroid
global_orientation
mask quality metrics
distance to previous/next anchor
```

Keep raw coordinates in original-image units.

## Avoid temporal oversampling bias

At 20 Hz, adjacent easy frames can be nearly identical.

For distribution-fitting experiments:

- either subsample temporally;
- or weight frames so long stationary periods do not dominate the posture prior.

For dynamics fitting, retain contiguous sequences.

---

# 8. Phase D — Determine the correct latent pose representation

Do this before implementing SMC.

The particle filter must not operate over 100 independent `(x,y)` points or 100 independent angles.

A proposed latent state is:

```text
z_t =
    translation x_t, y_t
    global orientation phi_t
    body length L_t
    posture coefficients a_t[1:K]
    width scale alpha_t
    optional discrete orientation h_t
```

## EXP-SMC-003 — Representation oracle test

### Question

How low-dimensional can posture be while still reconstructing anchor centerlines below the measurement noise floor?

### Compare

At minimum:

1. PCA/eigenworm-style basis fit to tangent-angle profiles;
2. cubic B-spline tangent-angle basis;
3. optionally a learned linear basis if distinct from PCA;
4. a high-dimensional reference representation.

### Test latent dimensions

For example:

```text
K = 4, 6, 8, 12, 16, 24, 32
```

Do not train an image model.

Directly project/final-fit known anchor poses into each representation and reconstruct them.

### Measure

- point reconstruction error;
- tangent-angle reconstruction error;
- curvature error;
- endpoint error;
- error versus body position;
- error versus bend severity;
- errors on rare tight poses;
- compute cost.

### Decision

Choose the smallest representation whose reconstruction floor is comfortably below the desired final measurement precision.

Do not choose four eigenworms merely because four components explain most variance. Variance explained is not the same as pixel-accurate reconstruction.

### Expected starting point

A reasonable initial hypothesis is `K ≈ 8–16`, but measurements decide.

---

# 9. Phase E — Learn width without allowing width to absorb pose mistakes

Width is necessary for rendering a segmentation mask, but it should be highly constrained.

From easy anchors, estimate normalized width as a function of body position:

```text
w_bar(s)
```

where `s ∈ [0,1]`.

## EXP-SMC-004 — Width model

Compare:

1. fixed recording-level mean width profile;
2. mean width profile × one frame-specific scale;
3. mean profile + one or two width PCA coefficients;
4. more flexible width only if required.

### Key test

Given the **true anchor centerline**, how much does each width model improve reconstruction of the observed segmentation?

### Constraint

Prefer the least flexible model that explains genuine width variation.

If width freedom allows an incorrectly located or incorrectly shaped centerline to achieve a good mask likelihood, the width model is too flexible.

### Default initial state

Prefer:

```text
w_t(s) = alpha_t * w_bar(s)
```

with a strong prior on `alpha_t`.

Body length should also have a strong recording-level prior.

---

# 10. Phase F — Build and validate the generative segmentation renderer

Given latent state `z_t`:

1. reconstruct tangent-angle profile;
2. integrate to a centerline;
3. apply global translation/orientation/length;
4. apply the width profile;
5. render a soft worm tube;
6. clip rendering to the camera FOV.

Let:

```text
R(z_t) ∈ [0,1]^(H×W)
```

be the rendered foreground probability/soft occupancy.

## Renderer requirements

- differentiable with respect to continuous pose latents if practical;
- batched across particles;
- GPU-friendly;
- explicit original↔model coordinate transforms;
- exact handling of body points outside the FOV;
- no penalty for anatomy outside the camera;
- stable gradients near mask boundaries;
- ability to render self-intersections naturally as the union of body occupancy.

Use PyTorch operations where practical.

## EXP-SMC-005 — Generative fidelity on anchors

### Question

Can the latent pose + width model reproduce easy-frame segmentation accurately enough that mask likelihood is meaningful?

For known anchor poses, render masks and compare to observed segmentation.

### Compare likelihood/energy candidates

At minimum:

#### A. Soft-mask BCE / Bernoulli likelihood

Compare `R(z)` to segmentation probability `P_t`.

#### B. Dice/IoU-like energy

Useful but potentially less informative for gradients.

#### C. Boundary or signed-distance energy

Compare rendered and observed distance transforms/boundaries.

#### D. Hybrid

For example:

```text
E_obs =
    lambda_mask * BCE(R, P)
  + lambda_boundary * E_boundary
```

Do not add many terms initially.

### Diagnostics

For each observation model, perturb a known anchor by controlled amounts and plot observation energy versus:

- translation;
- rotation;
- shape perturbation;
- length error.

A useful likelihood should have a smooth basin around the correct solution.

### Deliverables

- render overlays;
- mask residual maps;
- energy-vs-perturbation curves;
- runtime per rendered particle;
- gradients sanity tests.

---

# 11. Phase G — Learn temporal dynamics from easy sequences

This is central to the method.

At 20 Hz, posture should usually change only modestly over one frame.

Fit the transition model using contiguous stretches where anchors are available.

## State decomposition

Do not necessarily use one dynamics model for every latent.

A sensible initial factorization is:

```text
translation:
    constant velocity + noise

global orientation:
    angular velocity + noise

body length:
    nearly constant / slowly varying

width scale:
    nearly constant / slowly varying

posture coefficients:
    low-order autoregressive dynamics
```

## EXP-SMC-006 — Dynamics predictability

### Compare posture dynamics

At minimum:

1. random walk;
2. constant latent velocity;
3. AR(1);
4. AR(2);
5. optional small nonlinear model only if linear models leave substantial predictable error.

Fit parameters on training recordings only.

### Evaluate one-step and multi-step prediction

Measure prediction at horizons such as:

```text
1 frame   = 50 ms
2 frames  = 100 ms
5 frames  = 250 ms
10 frames = 500 ms
20 frames = 1 s
```

depending on the verified acquisition cadence.

### Plot

- coefficient prediction RMSE vs horizon;
- centerline prediction error vs horizon;
- angle prediction error vs horizon;
- uncertainty growth vs horizon.

### Important output

Estimate how long the prior remains informative **without image evidence**.

That tells you what lengths of difficult intervals are realistically recoverable.

---

# 12. Phase H — Establish simple non-SMC temporal baselines

Before attributing success to SMC, create cheap baselines.

For a difficult interval starting after an easy anchor, compare:

1. hold previous shape;
2. constant-velocity pose extrapolation;
3. AR dynamics posterior mean;
4. optimize one pose per frame independently against the segmentation;
5. deterministic MAP trajectory optimization over the full interval.

These provide important reference points.

The probabilistic method must beat them on difficult cases or provide calibrated multimodality that they cannot.

---

# 13. Phase I — Synthetic recovery experiments before natural coils

Do not debug SMC for the first time on uncontrolled self-intersections.

## EXP-SMC-007 — Known-pose reacquisition

### Construct controlled sequences

Start from real easy-frame anchor poses and create known latent trajectories or perturbations.

Generate observations with the validated renderer or controlled transformations.

Test:

- translation errors;
- orientation errors;
- latent shape errors;
- width errors;
- partial FOV;
- temporary observation degradation.

### Goal

Determine how many particles and what proposal mechanism are needed to recover the true trajectory.

### Particle counts

Benchmark a logarithmic range, e.g.:

```text
32
64
128
256
512
1024
```

Do not assume 512 is optimal.

Report effective sample size and failure probability, not only mean error.

---

# 14. SMC algorithm hierarchy

Implement the simplest method first, then add sophistication only when failure analysis justifies it.

## Level 1 — Bootstrap particle filter

For particle `i`:

```text
z_t^(i) ~ p(z_t | z_(t-1)^(i))
w_t^(i) ∝ w_(t-1)^(i) p(P_t | z_t^(i))
```

Normalize weights.

Compute:

```text
ESS = 1 / sum_i w_i^2
```

Resample when ESS falls below a declared fraction of the particle count.

Use systematic or stratified resampling.

### Purpose

This is the baseline probabilistic tracker.

### Likely failure

The segmentation observation may be sufficiently sharp that most transition-prior particles receive near-zero weight.

If so, do not merely increase particle count indefinitely.

---

## Level 2 — Guided proposal

Use cheap observation information to propose particles closer to likely current states.

Possible guidance signals:

- foreground centroid;
- principal orientation of the mask;
- area;
- bounding box;
- skeleton fragments;
- previous pose transformed toward current centroid;
- gradient of observation likelihood.

The proposal should remain probabilistically accounted for when necessary.

Test whether guided proposals materially increase ESS and reduce particle count.

---

## Level 3 — Resample-move / local rejuvenation

After resampling promising particles:

1. calculate differentiable observation energy;
2. perform 1–3 small optimization/MALA-like local moves in continuous latent space;
3. retain particle diversity.

The first implementation may use deterministic gradient refinement as a pragmatic resample-move approximation. If posterior correctness/calibration becomes important, implement a properly invariant MCMC move.

### Hypothesis

The transition prior gets particles into the right basin; a few local updates align them precisely with the observed segmentation.

This is likely to be far more efficient than blind sampling.

---

## Level 4 — Auxiliary or look-ahead particle filtering

If bootstrap/guided SMC repeatedly wastes particles on implausible ancestors, test an auxiliary particle filter that uses the upcoming observation to choose ancestors/proposals.

Do this only if evidence shows it is necessary.

---

# 15. Phase J — Use easy frames as hard or near-hard anchors

This is one of the key advantages of the method.

Suppose the easy/difficult classification is:

```text
frame:  240 241 242 243 244 245 246 247 248 249
state:   E   E   E   H   H   H   H   H   E   E
```

At frame 242, the classical centerline should create a very narrow posterior over `z_242`.

The probabilistic tracker only needs to solve frames 243–247.

At frame 248, another high-confidence classical pose becomes available.

Do not run unconstrained SMC through thousands of easy frames.

## Anchor update

When a new easy anchor is reached:

- compare the particle posterior to the classical pose;
- measure whether the correct mode survived;
- update/reset the posterior around the anchor;
- record any disagreement as a failure diagnostic.

Anchor disagreement is valuable evidence.

If the particle posterior strongly prefers a different pose than the supposedly high-confidence skeleton, investigate whether:

- anchor detection is wrong;
- the generative observation model is wrong;
- head/tail ordering differs;
- the SMC trajectory lost the correct mode.

---

# 16. Phase K — Detect and catalog difficult bouts

Automatically transform the easy/difficult timeline into **difficult intervals**:

```text
left easy anchor
difficult frames
right easy anchor
```

For every interval record:

- interval length;
- mask area;
- branch/cycle statistics;
- maximum curvature estimate if available;
- whether FOV contact occurs;
- whether head or tail is hidden;
- whether the interval contains self-intersection;
- segmentation confidence;
- start/end anchor poses.

Create a distribution of difficult-bout durations.

This is crucial.

If most hard bouts last only 3–10 frames, the temporal inference problem is much easier than if they routinely persist for hundreds of frames.

---

# 17. Phase L — Forward filtering on natural difficult intervals

## EXP-SMC-008 — Forward SMC through natural hard bouts

### Hypothesis

Starting from the left easy anchor, the particle filter can preserve the correct trajectory through a significant fraction of naturally difficult intervals.

### Evaluation without manual labels on every hard frame

Use several evidence sources:

1. **right-anchor recovery**
   - does the posterior at the end match the next trusted skeleton?

2. **render likelihood**
   - do inferred poses explain intermediate segmentation?

3. **trajectory smoothness**
   - are shape changes plausible under the learned dynamics?

4. **manual annotations**
   - densely annotate a small stratified subset of hard intervals for actual accuracy.

5. **synthetic controlled intervals**
   - retain exact-truth benchmarks.

### Primary endpoint metric

For intervals with a reliable right anchor:

```text
distance between predicted posterior at right boundary
and observed right-anchor pose
```

This is not sufficient by itself—a wrong trajectory could return to the correct final pose—so pair it with manual intermediate checks and trajectory likelihood.

### Stratify by

- interval length;
- self-intersection severity;
- FOV truncation;
- segmentation confidence.

---

# 18. Phase M — Bidirectional inference / particle smoothing

This should be a high-priority experiment because the data are offline.

Forward filtering estimates:

```text
p(z_t | observations up to t)
```

The desired offline target is closer to:

```text
p(z_t | all observations in the difficult bout,
        left anchor,
        right anchor)
```

## EXP-SMC-009 — Two-anchor smoothing

### Hypothesis

Using the next easy pose as a future constraint significantly improves centerline topology and reduces uncertainty through difficult intervals.

### Start with the simplest useful implementation

Possible first methods:

#### A. Forward trajectories + terminal reweighting

- propagate trajectory particles from the left anchor;
- at the right anchor, strongly weight trajectories by compatibility with the known right pose;
- resample/rerank full trajectories.

#### B. Forward/backward particle populations

- run forward from the left anchor;
- run backward from the right anchor using an approximate reversed dynamics model;
- combine compatible states/trajectories in the middle.

#### C. MAP trajectory optimization initialized by particles

- use high-weight particle trajectories as candidate initializations;
- optimize the entire latent sequence jointly with:
  - observation likelihood;
  - transition prior;
  - both anchor constraints.

This may be a very effective pragmatic smoother.

### Only later consider

- FFBSi;
- particle Gibbs;
- PGAS;
- more formal particle MCMC.

Do not implement sophisticated particle MCMC merely because it is elegant. First establish whether the simpler bridge/smoothing methods solve the scientific problem.

### Key comparison

For the same difficult intervals:

```text
forward-only SMC
vs
two-anchor smoothing
```

Report:

- manually annotated intermediate error;
- final-anchor consistency;
- posterior uncertainty;
- compute cost.

---

# 19. Phase N — Self-intersection topology experiment

## EXP-SMC-010 — Can the model resolve crossings?

Select natural difficult bouts where the segmentation mask contains a clear self-intersection or blob.

Manually annotate the true centerline for a manageable subset of intervals.

### Question

Does the posterior maintain the correct branch connectivity through the crossing?

### Compare

1. classical skeleton repair heuristic;
2. deterministic MAP;
3. forward SMC;
4. two-anchor smoothing;
5. optional mask + raw-image likelihood.

### Important metric

Do not evaluate only point distance.

Also evaluate **topology/connectivity correctness**.

A centerline can have modest point error while connecting the wrong arms at a crossing.

Define a crossing/topology success label for annotated cases.

### Posterior multimodality

For frames where two centerlines explain the mask similarly, visualize the top 2–4 posterior hypotheses.

This is a required figure.

The model should be rewarded for retaining multiple plausible topologies when the observation is genuinely ambiguous.

---

# 20. Phase O — Partial-FOV inference

Reuse the controlled crop framework from `worm_pose_scientific_experiment_plan.md`.

## EXP-SMC-011 — Head/tail leaves the camera

### Core generative rule

The latent worm exists beyond the image.

Only the in-camera portion contributes to the segmentation observation likelihood.

Do **not** shorten the latent body to fit the image.

### Compare

1. visible-tangent extrapolation;
2. hold previous full-body shape;
3. dynamics-only prediction;
4. forward SMC;
5. two-anchor smoothing after the body re-enters the FOV.

### Report separately

- visible-body error;
- boundary-band error;
- hidden-body error;
- body-length error;
- uncertainty vs hidden fraction.

### Expected behavior

Uncertainty on the hidden portion should grow as:

- more body leaves the FOV;
- the hidden interval becomes longer.

It should shrink when future observations constrain the body again.

This is a major calibration target.

---

# 21. Phase P — Head/tail identity

Self-intersections and reversals can make orientation ambiguous.

Treat anatomical orientation as a discrete latent variable when necessary:

```text
h_t ∈ {forward ordering, reversed ordering}
```

Do not silently canonicalize ambiguity away during inference.

### Strategies

- propagate both orientation hypotheses;
- use known anchors to constrain orientation;
- exploit any reliable static head/tail appearance only if independently validated;
- use temporal continuity.

### Evaluate

- head/tail flip frequency;
- duration of unresolved ambiguity;
- calibration of orientation probability.

---

# 22. Phase Q — Observation model escalation

Start mask-only.

Only add raw image evidence if experiments demonstrate a limitation.

## Trigger for escalation

Examples:

- two topologically distinct centerlines render nearly identical segmentation masks;
- future anchors do not resolve the ambiguity soon enough;
- segmentation boundaries are too coarse for desired angle precision;
- inferred mask fit is good but manual centerline error remains high.

## EXP-SMC-012 — Does raw NIR evidence break mask ties?

Compare:

1. segmentation-only likelihood;
2. segmentation + raw local edge likelihood;
3. segmentation + local intensity/cross-sectional likelihood;
4. only if needed, a compact learned feature likelihood.

Use the same posterior proposals.

Do not rebuild the entire generative model around raw appearance unless evidence requires it.

### Local likelihood preferred

Instead of rendering the whole raw NIR image, sample narrow cross-sections normal to the proposed centerline and compare expected versus observed local image structure.

This may preserve useful subpixel information at much lower cost.

---

# 23. Phase R — Calibration and posterior diagnostics

A probabilistic tracker should be evaluated as a probabilistic tracker.

## EXP-SMC-013 — Posterior calibration

On manually labeled and controlled synthetic cases, evaluate:

- posterior mean/median pose error;
- best-particle error;
- posterior spread vs actual error;
- credible-interval coverage for tangent angles;
- posterior entropy vs difficulty;
- probability assigned to the correct crossing topology;
- probability assigned to the correct head/tail orientation.

### Particle-health diagnostics

Track:

- ESS over time;
- resampling frequency;
- unique ancestor count;
- trajectory diversity;
- weight entropy;
- maximum particle weight;
- number of distinct topology modes.

Plot these over representative difficult bouts.

A filter that returns one confident trajectory after complete particle collapse is not automatically well calibrated.

---

# 24. Phase S — Adaptive compute

Easy frames should be cheap.

Hard frames should receive more computation.

Define a routing policy using quantities such as:

- easy-frame detector;
- segmentation confidence;
- skeleton topology;
- particle ESS;
- posterior entropy;
- mask residual;
- FOV truncation;
- disagreement with dynamics.

A possible runtime hierarchy:

```text
EASY FRAME
    classical anchor
    minimal/no SMC work

MILDLY AMBIGUOUS
    64–128 guided particles

DIFFICULT
    256–512 particles
    + resample-move

SEVERE COIL / LONG OCCLUSION
    more particles
    + full two-anchor smoothing
    + optional raw-image tie-breaker
```

## EXP-SMC-014 — Accuracy/compute Pareto

Compare fixed compute and adaptive compute.

Report:

- mean frames/s over full recordings;
- mean compute per easy frame;
- mean compute per hard frame;
- p95 difficult-frame latency;
- GPU memory;
- fraction of frames/intervals routed to each level;
- accuracy by route.

The final system must exceed the ~20 Hz acquisition rate **storage-inclusive** on the target GPU for offline processing.

Because the method is offline, temporary bursts of expensive computation are acceptable if average full-recording throughput remains strong.

---

# 25. Critical experiment: can easy anchors actually bridge natural hard bouts?

Before spending substantial effort on sophisticated inference, the orchestrator should treat this as the decisive proof-of-concept.

## EXP-SMC-015 — Anchor-to-anchor coil bridge benchmark

### Build benchmark

Identify approximately 50–200 naturally difficult intervals, depending on availability, each with:

```text
trusted left anchor
1+ difficult frames
trusted right anchor
```

Stratify by:

- duration;
- self-intersection;
- FOV truncation;
- tight bend without true crossing;
- low segmentation confidence.

Manually annotate a representative subset throughout the hard interval.

### Compare

1. classical skeletonization/repair;
2. deterministic dynamics + mask optimization;
3. bootstrap SMC;
4. guided/resample-move SMC;
5. bidirectional/two-anchor smoother.

### Primary questions

- Does the true mode survive from one easy anchor to the next?
- Does future-anchor smoothing select the correct trajectory?
- How does success rate depend on difficult-bout duration?
- At what interval duration does uncertainty become too large?
- Are failures caused by segmentation, transition priors, particle degeneracy, or mask non-identifiability?

### Decision

If this experiment succeeds, continue toward a polished full-recording tracker.

If it fails, diagnose the failure before returning to direct neural pose prediction.

---

# 26. Recommended experiment order

Use this sequence unless results provide a strong reason to change it.

```text
EXP-SMC-001  segmentation quality
      |
EXP-SMC-002  easy-frame detector + skeleton accuracy
      |
EXP-SMC-003  latent posture representation
      |
EXP-SMC-004  width model
      |
EXP-SMC-005  generative mask likelihood
      |
EXP-SMC-006  temporal dynamics
      |
      +---------------------------+
      |                           |
EXP-SMC-007                 build natural
synthetic SMC recovery      difficult-bout catalog
      |                           |
      +-------------+-------------+
                    |
EXP-SMC-008  forward SMC
                    |
EXP-SMC-009  two-anchor smoothing
                    |
        +-----------+-----------+
        |                       |
EXP-SMC-010 crossing       EXP-SMC-011 FOV
        |                       |
        +-----------+-----------+
                    |
EXP-SMC-015 anchor-to-anchor benchmark
                    |
          only if needed:
                    |
EXP-SMC-012 raw-image tie-breaker
                    |
EXP-SMC-013 calibration
                    |
EXP-SMC-014 adaptive compute
                    |
final full-recording validation
```

`EXP-SMC-015` is intentionally listed after the core algorithm experiments but should be prepared early. It is the most important integrated scientific benchmark.

---

# 27. Subagent organization

The orchestrator may reuse the subagent framework from `worm_pose_agent_orchestrator.md`.

Recommended roles for this branch:

## Segmentation / anchor subagent

Own:

- segmentation audit;
- mask cleanup;
- easy-frame criteria;
- skeletonization;
- width extraction;
- anchor validation.

## Latent-model subagent

Own:

- tangent-angle representation;
- PCA/spline fitting;
- width model;
- generative renderer;
- geometry tests.

## Dynamics / SMC subagent

Own:

- transition-model fitting;
- bootstrap filter;
- ESS/resampling;
- guided proposals;
- resample-move;
- smoothing.

## Evaluation / visualization subagent

Own:

- hard-bout catalog;
- manual-annotation sampling;
- synthetic recovery benchmark;
- FOV benchmark;
- crossing benchmark;
- calibration plots;
- performance plots.

The orchestrator remains responsible for interfaces and scientific decisions.

Do not let different subagents silently use different:

- coordinate conventions;
- head/tail ordering;
- latent bases;
- segmentation thresholds;
- frame splits;
- mask likelihood definitions.

---

# 28. Experiment record requirements

Use the same experiment-record conventions established in the prior documents.

For every `EXP-SMC-XXX`, record:

```markdown
# EXP-SMC-XXX — Name

## Hypothesis

## Scientific rationale
What prior observation or literature makes this worth testing?

## Inputs
Recordings, frame identities, evidence tier.

## Changed variable

## Fixed variables

## Method

## Predeclared success / failure criteria

## Results

## Visual evidence

## Particle / inference diagnostics
When relevant.

## Runtime

## Failure analysis

## Decision
SUPPORTED / NOT SUPPORTED / PARTIALLY SUPPORTED / INCONCLUSIVE

## Consequence
What exactly should happen next?
```

Every important SMC experiment must include visual trajectory examples, not only scalar metrics.

---

# 29. Required visualizations

The final experiment flow should heavily rely on visuals.

At minimum create:

## Segmentation / anchors

1. segmentation probability and cleaned-mask montage;
2. accepted easy-frame skeleton overlays;
3. rejected difficult-frame montage;
4. timeline of easy vs difficult frames;
5. histogram of difficult-bout duration.

## Generative model

6. true mask vs rendered mask on anchors;
7. width profile distribution;
8. latent reconstruction error vs latent dimension;
9. likelihood energy vs controlled pose perturbation.

## Dynamics

10. prediction error vs time horizon;
11. example latent trajectories and AR predictions.

## SMC

12. particles overlaid on a difficult frame;
13. ESS and posterior entropy over a difficult interval;
14. top posterior pose hypotheses at a self-intersection;
15. forward-filter trajectory visualization;
16. forward vs smoothed trajectory visualization;
17. particle genealogy or diversity diagnostic for representative failure cases.

## FOV

18. hidden-fraction vs pose error;
19. body-position × time heatmap of error/uncertainty as the tail exits/re-enters the FOV.

## Integrated system

20. accuracy vs particle count;
21. accuracy vs compute;
22. success rate vs difficult-bout duration;
23. classical vs MAP vs SMC vs smoother benchmark;
24. full-recording body-angle heatmap;
25. full-recording uncertainty heatmap;
26. final high-level hypothesis → experiment → conclusion flow figure.

---

# 30. Failure modes to actively search for

In addition to the previous failure-mode list, specifically inspect:

- wrong branch pairing at a self-intersection;
- particle collapse to one wrong topology;
- posterior retaining the correct mode but giving it negligible weight;
- anchor detector accepting a wrong skeleton;
- left and right anchors having inconsistent head/tail order;
- width inflation compensating for wrong pose;
- body shortening to explain a cropped mask;
- renderer preferring a geometrically wrong pose with similar mask area;
- transition model oversmoothing rapid turns;
- transition noise too narrow to follow real motion;
- transition noise too broad, causing particle inefficiency;
- resampling destroying rare but correct modes;
- local refinement pulling particles into the same wrong basin;
- future-anchor smoothing creating implausible intermediate motion;
- segmentation errors mistaken for pose ambiguity;
- long hard intervals where dynamics genuinely lose identifiability;
- ambiguous masks that cannot be solved without raw-image information;
- posterior uncertainty remaining low despite incorrect topology.

For every failure, classify the dominant cause as one of:

```text
SEGMENTATION
ANCHOR
REPRESENTATION
WIDTH MODEL
RENDERER / LIKELIHOOD
DYNAMICS PRIOR
SMC DEGENERACY
SMOOTHER
FUNDAMENTAL MASK AMBIGUITY
RAW-IMAGE INFORMATION NEEDED
```

This classification should drive the next experiment.

---

# 31. Implementation guidance

## PyTorch Lightning

The final package should still integrate with PyTorch Lightning as required by the previous project.

However, not every inference component needs to inherit from `LightningModule`.

A sensible division is:

- segmentation model wrapper/fine-tuning: Lightning-compatible;
- any learned dynamics or learned likelihood: Lightning-compatible;
- generative renderer: plain `torch.nn.Module` / functions;
- SMC/smoothing inference: plain PyTorch inference module;
- final pipeline: importable package that composes these components.

Do not distort SMC into a training framework abstraction merely to say it uses Lightning.

## Minimal dependencies

Reuse existing classical segmentation/skeleton code where possible.

Add a dependency such as `scikit-image` or `scipy` only if it materially improves:

- skeletonization;
- distance transforms;
- connected-component analysis;

and record it as a justified research/runtime dependency.

## GPU vectorization

Vectorize over particles.

Prefer tensor shape conventions such as:

```text
particles:
    [N_particles, latent_dim]

centerlines:
    [N_particles, N_body, 2]

rendered masks:
    [N_particles, H, W]
```

Avoid Python loops over particles.

If full-frame rendering is too expensive, test local/narrow-band observation scoring before considering custom CUDA.

---

# 32. Storage and repository policy

Retain the storage rules from the prior orchestrator.

Large files must live beneath:

```text
/temp_data4/alex/...
```

and not in the Git working directory.

This includes:

- segmentation-probability caches;
- cleaned-mask datasets;
- extracted anchor datasets;
- large latent-pose arrays;
- particle-trajectory dumps;
- rendered sequences;
- checkpoints;
- videos;
- profiler traces;
- full-recording output HDF5 files.

Repository symlinks to these locations are allowed.

Small:

- configs;
- CSV summaries;
- JSON metadata;
- Markdown;
- final figures;

may remain in the repository.

Do not overwrite source HDF5 recordings.

---

# 33. Final system architecture if the hypothesis is supported

The likely final architecture should look approximately like:

```text
                        NIR frame t
                            |
                            v
                  pretrained segmentation
                            |
                 +----------+----------+
                 |                     |
          probability map         cleaned mask
                 |                     |
                 |             topology/skeleton test
                 |                     |
                 |              +------+------+
                 |              |             |
                 |            EASY          HARD
                 |              |             |
                 |       classical pose       |
                 |          anchor            |
                 |              |             |
                 +--------------+-------------+
                                |
                                v
                       temporal latent prior
                                |
                                v
                 particle proposals / hypotheses
                                |
                                v
                  generative mask rendering
                                |
                                v
                   observation likelihood
                                |
                                v
                   resample + local move
                                |
                                v
                     posterior over pose
                                |
                     next easy anchor exists?
                         /             \
                       yes             no
                        |               |
                 offline smoothing   continue filter
                        |
                        v
                  final posterior pose
                        |
                        v
         centerline / angles / curvature / uncertainty
                        |
                        v
                 streamed output HDF5
```

Easy frames should function as regular high-confidence resets/constraints.

Hard intervals should be the only regions requiring substantial probabilistic inference.

---

# 34. Final scientific deliverables

In addition to the inherited final deliverables, this branch must produce:

## A. Anchor characterization

- fraction of frames accepted as easy;
- anchor precision;
- difficult-bout length distribution.

## B. Generative model characterization

- chosen latent dimension;
- reconstruction floor;
- width model;
- mask-rendering fidelity.

## C. Dynamics characterization

- fitted transition model;
- prediction error vs horizon;
- uncertainty growth vs horizon.

## D. SMC characterization

- particle-count curve;
- ESS behavior;
- resampling strategy;
- guided-proposal/resample-move benefit;
- posterior mode examples.

## E. Smoothing characterization

- forward-only vs two-anchor result;
- improvement vs difficult-bout duration;
- crossing-resolution accuracy.

## F. FOV characterization

- hidden-body accuracy;
- uncertainty growth;
- re-entry recovery.

## G. Full-recording performance

- storage-inclusive throughput;
- fraction of frames solved classically;
- fraction requiring SMC;
- fraction requiring smoothing;
- fraction remaining unresolved/ambiguous.

## H. Final experiment-flow narrative

Update `docs/EXPERIMENT_FLOW.md` so that a reader can visually follow:

```text
segmentation works?
    ↓
anchors reliable?
    ↓
latent representation sufficient?
    ↓
mask likelihood informative?
    ↓
dynamics predictive?
    ↓
SMC survives hard intervals?
    ↓
future anchor improves result?
    ↓
mask alone sufficient?
    ↓
final tracker
```

Every branch should show the decisive graph/visual and conclusion.

---

# 35. Further reading

Use the reading list in `worm_pose_scientific_experiment_plan.md` as the canonical bibliography.

The most relevant items for this specific branch are:

## WormPose

Hebert et al., *WormPose: Image synthesis and convolutional networks for pose estimation in C. elegans*

https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008914

Relevant because it explicitly builds on classically resolved non-coiled worm poses, angle-based posture representations, width profiles, and generative synthetic worms.

## WormTracer

Kuze et al., *WormTracer: A precise method for worm posture analysis using temporal continuity*

https://www.sciencedirect.com/science/article/pii/S0165027025002882

Relevant as a direct temporal-continuity baseline and evidence that sequential information can materially improve centerline reconstruction.

## Splender

Zdyb, Alonso & Kirkegaard, *Spline refinement with differentiable rendering*

https://papers.miccai.org/miccai-2025/0860-Paper1793.html

Code:
https://github.com/kirkegaardlab/splender

Relevant for differentiable rendering and local optimization of worm spline geometry against image evidence.

## 3DNEL

Zhou et al., *3D Neural Embedding Likelihood: Probabilistic Inverse Graphics for Robust 6D Pose Estimation*

https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_3D_Neural_Embedding_Likelihood_Probabilistic_Inverse_Graphics_for_Robust_6D_ICCV_2023_paper.html

Relevant conceptually for combining amortized observations/proposals with a structured generative model and retaining uncertainty under occlusion.

## 3DP3

Gothoskar et al., *3DP3: 3D Scene Perception via Probabilistic Programming*

https://proceedings.neurips.cc/paper/2021/hash/4fc66104f8ada6257fa55f29a2a567c7-Abstract.html

Relevant conceptually for proposal-driven probabilistic inference over structured latent geometry.

## Eigenworms / low-dimensional posture

Stephens et al., *Dimensionality and Dynamics in the Behavior of C. elegans*

https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028

Relevant for choosing and interpreting a low-dimensional posture state.

## Markovian worm dynamics

Costa et al., *A Markovian dynamics for Caenorhabditis elegans behavior across scales*

https://pmc.ncbi.nlm.nih.gov/articles/PMC11317559/

Relevant before replacing simple AR dynamics with a large learned temporal model.

---

# 36. What the autonomous agent should do first

When instructed to execute this plan, the orchestrator should **not** start by implementing a sophisticated particle filter.

The first concrete sequence is:

1. preserve the current repository state and previous negative results;
2. inspect/reuse the existing segmentation and classical baseline code;
3. run `EXP-SMC-001` segmentation audit;
4. implement and validate the conservative easy-frame detector;
5. establish anchor precision on manual/random review;
6. build the anchor pose dataset;
7. run the latent representation oracle test;
8. estimate the width profile;
9. implement the generative mask renderer;
10. verify mask likelihood around known anchor poses;
11. fit and evaluate simple temporal dynamics;
12. catalog natural hard bouts and their duration;
13. only then implement bootstrap SMC;
14. run controlled recovery tests;
15. move to natural forward SMC;
16. add two-anchor smoothing;
17. run the anchor-to-anchor hard-bout benchmark;
18. add raw NIR evidence only if mask-space inference leaves unresolved ambiguity;
19. optimize adaptive compute only after scientific accuracy is established;
20. package and validate the full-recording pipeline.

---

# 37. Stopping and redirection rules

Do not continue adding SMC sophistication if an earlier assumption fails.

## If segmentation is the bottleneck

Improve/fine-tune segmentation first.

## If easy anchors are unreliable

Fix the anchor detector or classical geometry before fitting the prior.

## If the latent representation cannot reconstruct anchors

Increase/change the representation before particle filtering.

## If the renderer cannot reproduce anchor masks

Fix width/renderer/likelihood before inference.

## If dynamics lose predictive power within 1–2 frames

The central temporal-prior assumption is weak; investigate whether pose coefficients or orientation conventions are wrong before abandoning it.

## If bootstrap SMC collapses

Try guided proposals and resample-move before increasing particle count dramatically.

## If forward SMC fails but the correct mode survives weakly

Prioritize smoothing/terminal-anchor reweighting.

## If mask-level posterior remains genuinely multimodal

This is not necessarily a failure. Preserve modes and test raw-image tie-breaking.

## If even raw-image evidence cannot disambiguate

Report the pose as unresolved with appropriate posterior uncertainty.

Do not force a deterministic answer where the data are non-identifying.

---

# 38. Final instruction to the orchestrator

The goal of this research branch is not to demonstrate that SMC is sophisticated.

The goal is to exploit the structure of the dataset as efficiently as possible:

- segmentation solves appearance;
- classical skeletonization solves easy geometry;
- easy geometry supplies abundant empirical training data;
- a compact generative model constrains valid worm shapes;
- 20 Hz sampling supplies a strong temporal prior;
- SMC preserves alternative explanations during ambiguous observations;
- future easy frames resolve many of those ambiguities;
- uncertainty remains explicit when they cannot be resolved.

Prefer the simplest method that successfully bridges real difficult intervals while maintaining accurate easy-frame measurements and high full-recording throughput.

The decisive scientific result is whether **temporally anchored generative inference can correctly reconstruct natural hard bouts that classical skeletonization cannot**.
