# Follow-on scientific experiment status

This page tracks execution of `worm_pose_scientific_experiment_plan.md` without
rewriting the evidence from the completed first study. The old checkpoint
remains rejected and exploratory-only. The 2025 audited holdout remains
unopened beyond its 32 previously disclosed audit indices.

## Current outcome

User expert adjudication is the governing starting point for the later
segmentation-anchored branch. EXP-SMC-002C records that row 2
(`2023-09-19-01-f017959`) has a genuinely mistaken annotation, row 22
(`2023-10-11-01-f013785`) is an expected downstream SMC-hard case, and the
remaining rows are visually adequate for development. D-0012 therefore
superseded the earlier absolute stop for development continuation only; it did
not erase the frozen `NOT_SUPPORTED` segmentation/anchor gates or create a
quantitative noise floor.

That authorization allowed the downstream assumptions to be tested. Compact
pose representation, a simple width proxy, a differentiable mask energy, and
controlled synthetic SMC execution all passed their bounded gates. The central
empirical temporal premise did not: strict anchors were sparse and
session-imbalanced, one session supplied no adjacent pair, fitted velocity/AR
diagnostics did not beat persistence, and the expert-designated row-22 hard
case had no strict anchor anywhere in its bounded 201-frame scan. Synthetic SMC
passed its absolute recovery gates, but exact two-anchor interpolation was best
in every scenario and terminal reranking helped only 2/6 scenarios. Therefore
there is no natural hard-bout recovery claim. The protected 2025 holdout remains
closed, and the 10 delayed blind repeats remain pending.

| Experiment | State | Evidence | Consequence |
|---|---|---|---|
| EXP-001 human precision | `PRIMARY_COMPLETE_REPEATABILITY_PENDING` | 30/30 primary traces validated: 17 complete, 12 truncated, 1 not identifiable; 0/10 delayed repeats | Directional model triage is authorized; wait until 2026-08-26 for blinded repeatability and do not call it inter-annotator agreement |
| EXP-002 baseline suite | `PARTIAL_BASELINE_COMPLETE` | On primary Tier A, accepted classical: 14/30 coverage and 12.12 px complete-trace median; rejected global model: 30/30 output but 80.87 px | Keep global checkpoint rejected; classical is selective reference only; external/anchored baselines remain |
| EXP-003 spatial architecture | `COMPLETE_REJECT_IMPLEMENTATIONS` | Nine paired CUDA runs; dense/global/anchored complete-trace seed-median medians 83.10/72.36/113.22 px; both spatial variants pass 0/3 seeds | Do not run resolution ablation; repair ordered decoding and hard-cell supervision on Tier C before another Tier-A evaluation |
| EXP-003B topology-safe rescue | `PRIMARY_CONTROLLED_GATE_FAIL` | Soft anchor + single intrinsic curve: fully visible Tier-C 61.99 px, 25.77°, 0.34 length error versus 16 px/15°/0.15 gates | Stop repeat seeds and keep Tier A closed; proceed to training diversity/scale before real-texture EXP-004 |
| EXP-004 synthetic realism | `ANALYTIC_5K_PRIMARY_CONTROLLED_GATE_FAIL` | Topology-safe model on 5,000 analytic cases: fully visible Tier-C 46.64 px, 25.17°, 0.261 length error versus 16 px/15°/0.15 gates | Stop repeat seeds and keep Tier A closed; real-texture synthesis is not authorized while analytic geometry still fails |
| EXP-005 representation floor | `SUPPORTED_ORACLE_ONLY` | On 17 complete traces, spline-16: 0.70 px/3.31 deg; cosine-16: 0.67 px/3.36 deg; PCA-8 leave-one-recording-out: 3.10 px/7.59 deg | Representation is not the dominant bottleneck; use fixed 16–24 coefficient tangent head initially |
| EXP-006 temporal context | not started | No 1/5/11-frame paired model comparison | Requires CUDA and the localization-preserving architecture, but not a final-quality single-frame gate |
| EXP-007 controlled FOV | partial infrastructure | Analytic/static and real-texture controlled crops exist; temporal model comparison does not | Continue after temporal proposals exist |
| EXP-008 refinement basin | `PARTIALLY SUPPORTED` on Tier C | Eight poses, 29 conditions, 3 objectives, 0/1/3/5/10 steps | Use raw robust pixels as the reference; repeat on real Tier A; test a new proposal only if it enters the basin |
| EXP-009 proposal + refinement | not authorized for old model | Old proposal has 0/31 Tier-B and 0/43 Tier-C cases at even <=16 px median error | Do not spend refinement compute on the rejected global checkpoint |
| EXP-010–015 | not started | Prerequisites not met | Preserve protected holdout and avoid unsupported final-system claims |
| EXP-SMC-001/001B foreground | `NOT_SUPPORTED` | Visible-trace median 0.98 in both; terminal median 0.80 in both; stable hysteresis growth did not repair ends | Frozen quantitative failure preserved; D-0012 separately authorizes development continuation on expert visual evidence |
| EXP-SMC-002/002B anchors | `NOT_SUPPORTED` | Baseline complete median 6.76 px but 3/12 truncated accepted; revision rejects 12/12 truncated but keeps 3/17 complete and only 2/3 within 8 px | Frozen quantitative failure preserved; see EXP-SMC-002C for the bounded current governance decision |
| [EXP-SMC-002C expert visual adjudication](../experiments/exp_smc_002c_expert_visual_adjudication/notes.md) | `DEVELOPMENT_CONTINUATION_AUTHORIZED` | Row 2 annotation judged mistaken; row 22 judged an expected SMC-hard case; remaining reviewed rows judged adequate | Supersedes the prior stop for development only; preserves frozen metrics and keeps holdout closed |
| [EXP-SMC-002D contiguous anchor density](../experiments/exp_smc_002d_contiguous_anchor_density/notes.md) | `STATIC_POSTURE_EXTRACTION_FEASIBLE__SESSION_GENERAL_DYNAMICS_NOT_FEASIBLE` | 87/303 strict anchors; adjacent pairs 7/45/0 by recording; one session has no adjacent pair | Static anchor poses are usable, but these windows cannot support session-general empirical dynamics |
| [EXP-SMC-003 latent representation](../experiments/exp_smc_003_latent_representation/notes.md) | `SUPPORTED_ORACLE_ONLY` | Cubic K=16 reconstructs complete traces at 0.697/1.074 px median/p95 point error and 3.305°/4.307° tangent error | Compact geometry is sufficient on known traces; this is not image inference or temporal evidence |
| [EXP-SMC-004 width model](../experiments/exp_smc_004_width_model/notes.md) | `COMPLETED_SUPPORTED_PROXY_ONLY` | Fixed recording mean cleaned-mask IoU 0.8676; extra scale/PCA flexibility earns negligible gain | Use fixed recording mean for controlled rendering; no manual-mask or biological-width claim |
| [EXP-SMC-005 observation energy](../experiments/exp_smc_005_observation_energy/notes.md) | `COMPLETED_SUPPORTED_ENERGY_SHAPE_ONLY` | Soft Dice selected; 64/64 controlled curves minimize within one grid step; monotonicity 0.9714 | Local differentiable observation basin supported against the classical segmentation proxy, not calibrated likelihood or natural ambiguity |
| [EXP-SMC-006 dynamics](../experiments/exp_smc_006_dynamics_predictability/notes.md) | `COMPLETED_NOT_SUPPORTED_SESSION_GENERAL` | Paired h=1 persistence 3.57 px versus best non-persistence 4.48 px; evidence counts 7/45/0 adjacent pairs | Natural velocity/AR dynamics unsupported; zero-drift random walk retained only as a synthetic control |
| [EXP-SMC-007 controlled SMC](../experiments/exp_smc_007_controlled_smc/notes.md) | `SUPPORTED_CONTROLLED_SYNTHETIC_ONLY` | All absolute synthetic gates pass; interpolation wins all six scenarios; terminal reranking improves forward SMC in 2/6 | Algorithm execution/truth survival supported only on renderer-matched synthetic walks; H8, smoothing benefit, and SMC superiority are not supported |
| [EXP-SMC-008A row-22 bracket](../experiments/exp_smc_008a_row22_anchor_bracket/notes.md) | `NO_BOUNDED_STRICT_ANCHOR_BRACKET` | 0/201 strict anchors in frames 13685–13885; no bracket with at most 20 intervening frames | Row 22 is not a viable short natural two-anchor pilot under the current strict-anchor pipeline; no pose was inferred |
| EXP-SMC-000 hard-bout catalog | `CATALOG_ONLY` | Six unadjudicated development candidates; sparse legacy brackets are generally hundreds of frames away | Retain for future protocol design, not outcome claims |

## EXP-001 evidence boundary

The selection manifest is development-only and contains exact source identity,
frame/timestamp mapping, blind-view policy, context windows, and assignment
flags. It allocates 86/85/85 frames across the three 2023 sessions. Two
nonoverlapping 11-frame windows per session are completely double-annotated in
the assignment design, giving 66 pairs rather than the minimum 64.

The selection preview was visually inspected without pose overlays. It shows
ordinary motion, substantial curvature, boundary contact/truncation, varied
scale/background, and tight turns. Automated selection cannot certify all rare
biological strata; the annotation schema therefore retains explicit difficulty,
support, truncation, and not-identifiable states.

No metric called human accuracy, manual ground truth, or annotation noise floor
exists yet. The single-person primary pass is complete: all 30 records validate
against the frozen worklist and source identities, with 17 complete, 12
naturally truncated, and one not-identifiable trace. Ten blind repeats remain
locked until seven days after their corresponding primary traces. The checked-in
evaluator will compute the provisional intra-annotator distribution only after
those independent repeat records exist.

## Primary-30 baseline and representation result

The real-label overlay audit confirms the earlier negative result. The rejected
global checkpoint draws a short, displaced, low-curvature mean pose: its median
per-frame error is 80.87 px with 51.04 degrees mean tangent error on the 17
complete traces. Its one-way visible-trace distance on 12 truncated cases is
49.32 px. It remains far outside the Tier-C refinement basin.

The conservative classical method accepts 14/30 frames, including 0/9 of the
proxy-difficult stratum. On its 11 accepted complete traces, median per-frame
error is 12.12 px. When its rejection gate is ignored for diagnosis, a curve is
available on all 30 frames and remains at 12.12 px across all 17 complete
traces, with 6.75 px visible-trace distance on the truncated cases. This is
useful image-aligned evidence, not a validated all-frame output policy.

Direct representation fits rule out intrinsic compression as the main failure.
A fixed 16-coefficient cubic tangent spline reconstructs complete labels at
0.70 px median and 3.31 degrees; a 16-term cosine basis gives 0.67 px and 3.36
degrees. Eight-component PCA reaches 0.83 px in-sample but degrades to 3.10 px
and 7.59 degrees when a full recording is held out, so the small tranche does
not justify a learned posture basis.

## EXP-008 result

The tuned eight-pose Tier-C sweep found a useful but narrow basin with the
provisional 4 px success definition. Raw robust pixels recovered all cases
through 4 px translation, 2 degrees rotation, 2% length error, and 2 degrees
RMS shape error. At the next tested levels, recovery was 87.5%, 87.5%, 93.75%,
and 87.5%, respectively. The mild combined perturbation recovered only 37.5%,
and moderate/severe mixtures recovered none.

Across the complete grid, raw pixels reached 48.71% success and 4.16 px median
final error. Pixel+gradient reached 46.98% / 4.48 px; tube weighting reached
46.12% / 4.59 px. The added objectives therefore did not earn their complexity
on analytic Tier C. This does not predict real NIR refinement performance.

The existing rejected proposal is far outside the measured range: its best
reported median per-case errors were 33.69 px on Tier-B candidates and 43.92 px
on fully visible Tier C. Zero cases in either stratum were at or below 16 px.

## Concrete next action

The segmentation-anchored branch should not advance to a natural recovery
claim with the present strict anchors. Its successful components remain useful
infrastructure, but reopening natural SMC requires prospectively improved
anchor continuity or independently annotated short temporal bouts. A wider
search around row 22 would be a new preregistered experiment, not evidence that
the current bounded bracket succeeded.

Separately, do not reopen or revise the primary traces. Starting 2026-08-26,
resume the local tool and complete the 10 delayed tasks, which never display the
prior trace:

```bash
scripts/project_env.sh uv run --no-sync --frozen python \
  scripts/annotate_tier_a.py \
  --annotator-id <stable-pseudonym> \
  --output /temp_data4/alex/external_artifacts/annotations/tier_a.json \
  --open
```

After the repeat pass, run the agreement evaluator with
`--comparison-mode intra-annotator --minimum-pairs 10`. EXP-003 is now complete:
the exact paired 3-architecture by 3-seed CUDA matrix rejected both spatial
implementations. The diagnostic overlays and held-out Tier-C truth show that
the dense decoder covers worm pixels while violating curve order, and the
anchored decoder has a hard-cell selection mismatch. The next compute task is a
small Tier-C-first decoder experiment with an explicit topology gate. EXP-003B
performed that test with a soft anchor and guaranteed ordered intrinsic curve,
but failed all three controlled gates; repeat seeds and Tier-A evaluation were
therefore stopped. EXP-004 then completed the preregistered 5k analytic
training-scale control with the unchanged topology-safe model, materialized
tensors, leakage exclusions, and physical GPU 0. It improved fully-visible
Tier-C point and length errors to 46.64 px and 0.261, but tangent error remained
25.17 degrees and all frozen gates failed. The two repeat seeds, Tier A,
delayed repeats, and protected holdout remain closed. Do not add real texture
to a model that still fails its analytic geometry control, scale resolution, or
reuse the 30 Tier-A values for tuning. Any next proposal requires a new
Tier-C-only preregistration and must earn the same controlled gate before real
or protected evaluation.
