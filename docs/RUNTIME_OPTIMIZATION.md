# Pose pipeline runtime optimization

Measured on physical **GPU 3 (NVIDIA RTX 6000 Ada)**, with the existing fast
schedule, starts, and raster resolution retained. The default changes reduced
elapsed time by **9–19% in two paired learned-pipeline measurements** (about
**1.10–1.23× throughput**), with identical final poses on the test clip. Absolute
times varied substantially between rounds. This is a small development
benchmark, not a whole-recording throughput estimate.

## Measurements

Times are medians after excluding the first repetition in each process.
Independent fitting uses four saved development masks; the learned pipeline
uses six consecutive source frames, including segmentation and temporal fitting.

| Workload | Baseline | Optimized default | Optional whole-energy compilation |
| --- | ---: | ---: | ---: |
| Independent pipeline, four masks | 3.120 s | 2.710 s (1.15×) | 2.102 s (1.48×) |
| Learned pipeline, six frames, first pair | 12.970 s | 10.568 s (1.23×) | — |
| Learned pipeline, six frames, confirmation | 8.634 s | 7.882 s (1.10×) | 7.889 s (1.09×) |

The optional mode compiles only independent fits. Comparing its 7.889 seconds
against the earlier 12.970-second baseline would overstate the improvement:
the adjacent repeat baseline also became faster. It provided no additional
full-pipeline gain over the default in the confirmation round. Its 1.48×
independent result is from a separate four-mask comparison and is not a general
speed guarantee. GPU/host load and compiler cache history were not controlled;
there are only two warm samples per learned run, and export latency also varied.

Default learned-pipeline stage times from the first pair (raw confirmation
timings are also retained):

| Stage | Baseline | Optimized |
| --- | ---: | ---: |
| Segmentation, flat field, cleanup, workspace writes | 1.171 s | 0.683 s |
| Initialization and independent fitting | 2.736 s | 2.317 s |
| Bidirectional propagation and path selection | 8.997 s | 7.519 s |
| Prior loading, ambiguity, tracking, export | 0.066 s | 0.049 s |

Peak PyTorch allocated GPU memory for the default learned run decreased from
4.146 GB to 3.761 GB (9.3%). This excludes allocator reservations and other
processes. Temporal fitting remains the largest runtime cost.

## Changes

- Replace quadratic pairwise-distance work with SciPy's exact Euclidean
  distance transform. Preserve the signed-distance convention and prepare
  masks/distances on CPU before transferring complete grouped arrays to CUDA.
- Replace repeated image-wide flood-fill passes and Python component labeling
  with SciPy frontier propagation and connected components. Use separable
  square morphology, retaining connectivity, border, and finite-iteration rules.
- Cache fixed loss targets, weights, target mass, and point indices. Keep
  optimizer history and winning snapshots on the device until the result is
  needed; retain the nonfinite-loss check.
- Bound final hard-mask rendering to the region that can contain the tube,
  while preserving coordinates and returning full-size masks.
- Use scalar coordinate arithmetic for the CPU renderer. CUDA retains the
  original vector arithmetic: testing the scalar rewrite with the installed
  compiler exposed NaN gradients on real geometry, so that rewrite is CPU-only.
- Vectorize candidate-transition costs and dynamic programming for propagation.
  Cache immutable spline bases while returning independent writable arrays to
  callers.
- Preallocate segmentation output, use inference mode, and avoid redundant
  slab copies. The recording CLI retains up to 128 MiB of packed cleaned masks
  so propagation/tracking can reuse segmentation; eviction triggers normal
  segmentation. The staged benchmark already stores masks, so its measured
  speedup does not include this CLI cache benefit.
- Add opt-in whole-energy compilation for independent fitting. Temporal,
  head-constrained, and slow-refit paths retain their existing renderer route.
  A compiler specialization-limit error falls back for the affected group
  without restarting optimization; unrelated compiler failures remain visible.

CPU microbenchmarks show 9.67× faster path selection (100 frames, seven
candidates plus mirrors), 15.06× faster repeated spline decoding, and large
preprocessing gains (hole cleanup about 98×, distances about 14× on four real
masks). These are component measurements, not additive pipeline speedups.
The isolated segmentation network throughput was unchanged within noise.

## Numerical validation

All six learned-pipeline frames fit successfully. With default optimizations,
final centerlines, width profiles, body lengths, IoUs, selected starts, sources,
and path choices were bitwise identical to the baseline. Median IoU was
0.9696093946. On the four independent masks, IoUs and choices were identical;
the maximum centerline coordinate difference was 0.000244 pixels.

Whole-energy compilation changes floating-point evaluation order. Maximum
centerline coordinate differences were 0.006424 pixels for independent masks
and 0.008484 pixels after the learned temporal pipeline. Maximum IoU differences
were 0.0000871 and 0.0000303 respectively; starts, orientations, and path/source
choices were unchanged. These checks assess numerical regression on the sampled
frames, not anatomical accuracy across the dataset. The confirmation pair also
produced bitwise identical default outputs.

Targeted tests cover preprocessing against the previous implementations,
component labels, render outputs and gradients, cropped-render equivalence,
batched fitting, compiler dispatch/fallback, path selection, spline caching,
segmentation, and recording mask reuse. The CUDA renderer regression includes
real latent geometry. Both CPU and GPU checks passed; the entire repository
test suite was not run.

## Reproduce and enable

The benchmark script saves per-stage timings, fit counts, output arrays,
configuration, source hash, device, and peak GPU allocations. Baseline source
was copied before optimization, including existing workspace edits. JSON
reports and numerical comparisons are in [runtime_optimization](runtime_optimization/).

```bash
scripts/project_env.sh env CUDA_VISIBLE_DEVICES=3 .venv/bin/python \
  scripts/benchmark_pose_runtime.py \
  --output /tmp/pose-runtime-new --repeats 3 --frames 6 \
  --recording /store1/shared/all_data_raw/prj_aversion/2024-05-22/2024-05-22-15.h5 \
  --checkpoint checkpoints/segmenter/best.ckpt \
  --prior-file /temp_data4/alex/external_artifacts/workspaces/2024-05-22-15_f0-1199/recording_prior.json \
  --dataset-root /temp_data4/alex/external_artifacts/datasets/worm_pose_gen/segmentation_v1 \
  --compile --device cuda --propagate
```

Use a new output directory for each invocation. To compare another source
snapshot, add `--source-root /path/to/snapshot/src`. For the optional mode, add
`--fit-overrides '{"compile_energy":true}'`. In `scripts/fit_recording.py`, use
`--compile-energy`; workspace pipeline callers can set
`FitParams.overrides={"compile_energy": True}`.

Whole-energy compilation stays disabled by default because it adds compiler
startup and its benefit depends on group shapes and reuse. The first
four-mask run took 50.1 seconds with it versus 26.8 seconds for the optimized
default in these processes. Compiler disk caches were not cleared, so these
are startup observations, not controlled cold-cache comparisons. All first-run
times remain in the reports.

The environment used Python 3.13.15, PyTorch 2.13.0+cu130, and SciPy 1.18.1.
PyTorch, OpenMP, MKL, and OpenBLAS were limited to one CPU thread for the
pipeline benchmark; preprocessing microbenchmarks used four PyTorch threads.
The GPU benchmarks ran sequentially on GPU 3. No holdout recording was used.

The learned sample is source frames 0–5 at 732×968. Segmentation uses the
existing checkpoint and cached flat field; the prior is loaded from an existing
development workspace. Propagation forces an interior stretch (rows 1–4),
anchors both endpoints, and uses `min_score=0`, `pad=0`, `beam=1`,
`anchor_diversity=False`, and `jump_seeds=False` in both revisions. Tracking runs
but performs zero refits on these frames. Parquet export is included. Prior
bootstrap, Python imports, video encoding, and optional diagnostics are excluded.
The independent sample contains workspace rows 0, 100, 300, and 500, replayed
through threshold segmentation. It has four repetitions; learned runs have
three. Longer recordings, varied crop shapes, difficult coils, and default
multi-candidate propagation can have different speedups.
