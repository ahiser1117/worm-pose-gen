Step 6 of `docs/POSE_PIPELINE_PLAN.md` (parts 6a, 6c, 6b), stacked on `pose-pipeline-step5`. Results and the tuning history are in the plan's step 6 section; sequence-set results in `docs/pose_pipeline_step6/`.

## 6a: autoregressive chain proposals with a temporal prior
- Inside a propagation chain each frame is started from a first-order prediction of the pose (shape, rotation and centroid carried on with damping 0.6, length held) as well as the neighbour's copy, and the batched fitter pulls the in-view centerline toward a per-frame reference (`temporal_prior_weight` 0.01, sigma half a width). Weight sweep: 0.0025 inert, 0.025 held long chains on stale predictions, 0.01 removed the raw spiral's 14 failures with nothing else regressing; sequence set 12 -> 8 below IoU 0.9.

## 6c: beam chains, independent refit, one path per stretch
- Up to three distinct chain states per direction (ranked by the fit's full energy), the stored independent pose refit under the chain schedule with the anchors' length prior, and `select_path`: one candidate per frame along the stretch by dynamic programming over all candidates and their mirrors, anchored on the frames outside the stretch (energy over a temperature plus pose distance in widths, in-view change, and log length change). Every candidate is stored (`hypotheses_*`, `path_*`) and the viewer draws them ranked by energy with the path's choice.
- Edge minute pose jumps 23 -> 1, flips 96 -> 68; sequence set 8 -> 7. The balanced refit schedule was slower and worse and is not the default.

## 6b: bend limit, coverage, track length pass, jump seeds, anchor diversity
- Bends tighter than a radius of half a body width are penalised in both fitters (`--min-bend-radius`); tightest bends across the minutes fall from 14--17 to 3--5 widths per radius.
- `tube_coverage` (fraction of the tube on mask) beside the IoU, with the viewer tag "mask has extra body (segmentation)" for frames where a plate streak merged into the mask.
- After propagation, clipped frames outside stretches whose length departs from the track median are refit with the track's length prior; stretches are also seeded by pose jumps and border entries; each chain starts from two anchor states, which resolved two coil clips whose stretch flipped between a 0.97 and a 0.88 winding on one-frame changes of its boundary.
- Sequence set 7 -> 5 below IoU 0.9; spiral_0528's median 0.970 -> 0.957 under the bend limit; propagation about twice 6c.

Tests: `python -m unittest discover -s tests` passes at the head of this branch.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011UHA1vypJ4WSrfmz39vj66
