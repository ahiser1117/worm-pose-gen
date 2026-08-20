# Segmentation-anchored generative branch — final development report

## User expert adjudication and governing scope

The stop recorded below remains the frozen quantitative outcome of
EXP-SMC-001B/002B, but it is no longer the current development-governance
decision. EXP-SMC-002C records an explicit expert review of all 30 visual rows:
row 2 (`2023-09-19-01-f017959`) is judged to have a genuinely mistaken
annotation; row 22 (`2023-10-11-01-f013785`) is judged an expected hard case
for SMC rather than a required easy anchor; and the remaining rows are judged
visually adequate to move forward.

D-0012 therefore superseded D-0011 for development continuation only. The
original annotation and all frozen metrics remain unchanged, and the prior
`NOT_SUPPORTED` results are not converted into numeric passes. Development of
the anchor dataset, latent representation, width/renderer, dynamics, and
controlled SMC was authorized. This expert review is not a quantitative noise
floor, independent validation, protected-holdout evidence, or deployment
authorization. The 2025 holdout remains closed.

## Final outcome

The authorized development branch separated successful generative components
from a failed empirical temporal assumption. A fixed K=16 representation,
recording-mean width proxy, differentiable soft-Dice observation basin, and
controlled synthetic SMC implementation all passed their bounded tests. Those
results establish useful geometry, rendering, and algorithm infrastructure.

The central natural-data premise did not pass. EXP-SMC-002D found 87/303 strict
anchors but only 7/45/0 adjacent pairs across recordings. EXP-SMC-006 therefore
could not fit session-general dynamics, and its best non-persistence h=1
diagnostic was worse than persistence. EXP-SMC-007 passed every absolute
synthetic recovery gate, yet exact two-anchor interpolation had the lowest
median trajectory error in all six scenarios and terminal reranking improved
forward SMC in only 2/6. Finally, EXP-SMC-008A found 0/201 strict anchors around
row 22 and no natural bracket with at most 20 intervening frames.

Consequently, controlled synthetic execution is supported, but a general
terminal-smoothing benefit, SMC superiority, a natural motion prior, and
natural hard-bout recovery are not supported. This is not evidence that SMC is
intrinsically ineffective; it shows that the present anchor continuity and
temporal evidence do not justify the natural inference claim.

## Reproducibility and data boundary

- Python is managed exclusively through `scripts/project_env.sh` and `uv`.
- The final full `unittest` discovery run passed all 190 tests after branch
  implementation.
- Physical CUDA preflight passed on exactly one visible NVIDIA RTX 6000 Ada;
  EXP-SMC-005 used CUDA, while mask audits and bounded EXP-SMC-007 were serial
  CPU measurements.
- Source HDF5 remained read-only and streaming. Three usable 2023 development
  recordings supplied 60,468 frames; the 2025 recording remained protected.
- Large hard-bout screening artifacts live below
  `/temp_data4/alex/external_artifacts/experiments/worm_pose_gen/smc/`.
- Every metrics record states `protected_2025_holdout_opened: false`; per-case
  records contain only the three 2023 sessions.
- The 10 delayed blind repeat annotations remain pending, so no human
  repeatability/noise-floor conclusion is available.

## Experiment chain

| Experiment | Hypothesis and method | Result | Decision |
|---|---|---|---|
| EXP-SMC-001 | Classical robust-dark-ridge soft map, conservative cleanup, 30 central development frames plus adjacent stability | Whole-trace median/p10 0.980/0.958; terminal median 0.800; soft trace score 0.738 | `NOT_SUPPORTED` |
| EXP-SMC-002 | Strict simple-topology skeleton, unpruned longest path, width/render QC | 10/30 accepted; seven complete at 6.756 px median and 4.634°; only 6/7 within 8 px; 3/12 truncated falsely accepted | `NOT_SUPPORTED` |
| EXP-SMC-001B | Prospective 0.25/0.5 connected hysteresis with growth/stability caps | Growth median/p95 0.97%/2.20%, but terminal median stayed 0.800 | `NOT_SUPPORTED` |
| EXP-SMC-002B | One-median-width boundary clearance on revised masks | 12/12 truncated rejected, but only 3/17 complete accepted, only 2/3 within 8 px, none accepted in one session | `NOT_SUPPORTED` |
| [EXP-SMC-002C](../experiments/exp_smc_002c_expert_visual_adjudication/notes.md) | Explicit user expert review bound to all 30 frozen visual rows | Row 2 annotation mistaken; row 22 expected SMC-hard; remaining rows adequate for development | `DEVELOPMENT_CONTINUATION_AUTHORIZED` |
| [EXP-SMC-002D](../experiments/exp_smc_002d_contiguous_anchor_density/notes.md) | Three contiguous 101-frame strict-anchor scans | 87/303 accepted; 52 adjacent pairs distributed 7/45/0 | `SESSION_GENERAL_DYNAMICS_NOT_FEASIBLE` |
| [EXP-SMC-003](../experiments/exp_smc_003_latent_representation/notes.md) | Fixed tangent bases and recording-held-out PCA on complete traces | Cubic K=16 selected at 0.697/1.074 px median/p95 and 3.305°/4.307° | `SUPPORTED_ORACLE_ONLY` |
| [EXP-SMC-004](../experiments/exp_smc_004_width_model/notes.md) | Leave-one-frame-out recording width models against cleaned masks | Fixed mean IoU 0.8676; scale/PCA add negligible reconstruction value | `SUPPORTED_PROXY_ONLY` |
| [EXP-SMC-005](../experiments/exp_smc_005_observation_energy/notes.md) | Differentiable controlled translation/rotation/shape/length energy basins | Soft Dice selected; 64/64 minima within one grid step; 0.9714 monotonicity | `SUPPORTED_ENERGY_SHAPE_ONLY` |
| [EXP-SMC-006](../experiments/exp_smc_006_dynamics_predictability/notes.md) | Paired persistence, velocity, and AR forecast diagnostics | Paired h=1 persistence 3.57 px; best non-persistence 4.48 px; one session has no transition | `NOT_SUPPORTED_SESSION_GENERAL` |
| [EXP-SMC-007](../experiments/exp_smc_007_controlled_smc/notes.md) | Calibrated synthetic zero-drift walks, five stresses, four trajectory methods | Absolute gates pass; interpolation wins 6/6; terminal reranking helps 2/6 | `SUPPORTED_CONTROLLED_SYNTHETIC_ONLY` |
| [EXP-SMC-008A](../experiments/exp_smc_008a_row22_anchor_bracket/notes.md) | Bounded ±100-frame strict-anchor scan around row 22 | 0/201 accepted; no ≤20-frame two-anchor bracket; no pose inferred | `NO_BOUNDED_STRICT_ANCHOR_BRACKET` |
| EXP-SMC-000 | Raw-only bounded natural hard-bout screen | Six candidate windows; event types unadjudicated; sparse proxy brackets usually hundreds of frames away | `CATALOG_ONLY` |

The fixed 16-term tangent basis reconstructs the 17 complete manual traces below
1 px median error, so representation capacity is not the observed blocker. The
width and observation results likewise show that a simple tube can provide a
smooth local mask-space score. The downstream failure is specifically the lack
of balanced contiguous anchors and supported natural transition dynamics, not
a failure of every component.

## Failure mechanism

The baseline segmentation usually contains the body interior but clips thin
ends. Three truly truncated worms therefore appeared as simple interior
components whose skeletons cleared the fixed 13 px margin. Their clearances
(20–23 px) were smaller than their measured widths (38–49 px). A width-relative
guard correctly rejected this failure class.

The cleanup repair did not recover the missing anatomy. It added little area
and stayed temporally stable, yet terminal containment did not move. It also
made many masks contact the FOV and changed one retained complete skeleton from
8.73 to 12.60 px error. The limitation is therefore not solved by another
threshold pass over the same labels.

## What is and is not supported

Supported, narrowly:

- visible midbody trace containment is high for the classical masks;
- accepted complete classical anchors can be geometrically accurate;
- a width-relative FOV guard is useful for conservative truncation rejection on
  this development sample;
- low-dimensional intrinsic posture has ample oracle capacity on the small
  manual set.
- a fixed recording-level width proxy reconstructs the classical cleaned masks
  well on known complete traces;
- soft Dice provides a smooth differentiable local controlled energy basin;
- bootstrap SMC can preserve and recover truth under short renderer-matched
  synthetic random walks and the frozen controlled stresses.

Not tested or unsupported:

- learned foreground segmentation accuracy or calibration;
- manual-mask Dice/IoU (manual masks do not exist);
- a human annotation noise floor (delayed repeats are not yet available);
- session-general anchor-derived temporal dynamics or an empirical process
  scale;
- SMC superiority over exact two-anchor interpolation or a general benefit from
  terminal-anchor reranking;
- crossing resolution, natural partial-FOV reconstruction, posterior
  calibration, or natural hard-bout recovery;
- full-recording throughput or deployment readiness.

## Final prerequisite for reopening natural SMC

First obtain prospectively improved strict-anchor continuity or independently
annotated short temporal bouts in every development session. Freeze the natural
bout selection and scoring before inference, require real anchors on both sides
within the intended horizon, and re-establish a recording-balanced transition
model or explicitly justify a non-empirical prior. Row 22 cannot serve as that
pilot under the current detector: the bounded scan contains no accepted anchor
at all. A wider search would be a new experiment and would not change the
0/201 result.

Only after a natural controlled benchmark beats interpolation and demonstrates
a reproducible terminal-anchor benefit should the branch claim natural SMC
recovery or approach the protected holdout. The delayed 10-trace repeatability
measurement must also be completed before invoking a human precision/noise
floor.

## Original minimum prerequisite before EXP-SMC-002C

Freeze a new foreground method—most plausibly a small learned segmenter—using
fresh mask annotations that explicitly label thin terminals and natural FOV
contact. Do not tune another classical threshold on the same primary 30. The
new pipeline must prospectively pass terminal structure and every unchanged
anchor-reliability gate, then demonstrate enough validated contiguous anchors
in every development session to estimate short-horizon dynamics and bracket
short hard bouts. Only after that should EXP-SMC-003 through EXP-SMC-015 be
authorized. Keep the protected 2025 holdout closed until development selection
and all upstream gates are frozen.

EXP-SMC-002C and D-0012 supersede that absolute development stop on bounded
expert visual evidence. The paragraph above remains as the historical
quantitative recommendation and still applies before any independent/final or
holdout claim; it no longer prohibits development-only downstream experiments.
