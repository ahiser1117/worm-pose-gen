Steps 5, 5b and 5c of `docs/POSE_PIPELINE_PLAN.md`, plus the browser diagnostic viewer.

## Step 5: temporal propagation across ambiguous stretches
- Stretches of frames with ambiguity score >= 2 (padded, merged) are refit by forward and backward chains warm-started from the good frames on either side, all stretches in lockstep; per frame the lowest total energy among independent, forward and backward wins (`source`, `iou_independent`, `score_independent` in `poses.npz`).
- Sequence set: frames below IoU 0.9 fall 611 -> 62; overlay videos are rendered from the final arrays.

## Step 5b: tuning on the sequence set
- Anchor-centred 2% length prior in the chains (spiral over-length fixed: 62 -> 11 failures), `mask_on_border` and the `edge_inside` flag, off-camera redirected starts, labeling round 2 manifest (393 frames, held-out animals).

## Step 5c: segmenter round 2 and raw masks
- 91 bootstrap labels retired; `r2-hand165` trained from scratch on 165 hand labels and promoted (val 0.978 / test 0.981); model name registry `docs/segmenter_model_names.json`; seven reworked comparison plots.
- `fit_recording.py --raw-mask` (hole fill and largest-component rule optional; the ambiguity statistics stay defined on raw masks). Five minute-long before/after videos: coil gaps are real background, edge fragments are gone with the new model, and the largest-component rule still guards the unseen plate `2024-06-18-12`, so cleaned masks stay the default.

## Pose viewer (`worm_pose_gen.pose_viewer`, `worm-pose-viewer`)
- Localhost app that scrubs a stored run with every pipeline layer composited on the frame (probability, mask stages, tube, residual, independent fit, starts), every per-frame statistic, the ambiguity flags with thresholds, a frame classification, width and curvature profiles, synced timelines, a compare run, and review notes; asynchronous scrubbing with a light tier, resizable panels, HiDPI-correct charts.
- The fitter stores the independent pose so the viewer can show what propagation replaced.

Tests: `python -m unittest discover -s tests` passes (245 tests at the head of this branch).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011UHA1vypJ4WSrfmz39vj66
