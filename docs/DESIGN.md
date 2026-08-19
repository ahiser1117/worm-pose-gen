# Design and final engineering state

## Status

No proposal was accepted. The frozen EXP-0007 decision is
`PRIMARY_FOLD_FAIL`; `final_accept_geometry_rescue=false` and
`authorize_temporal_modeling=false`. Consequently this repository does **not**
claim a final deployable pose estimator. It provides two distinct things:

1. accepted research/evaluation infrastructure; and
2. a fail-closed exploratory adapter around a rejected diagnostic checkpoint.

`configs/final.yaml` records that negative state. The checkpoint is retained
only so another researcher can reproduce the failures and exercise the output
pipeline without confusing it with an accepted model.

## Accepted infrastructure

```text
read-only frame source
        |
        v
deterministic grayscale normalization (192 x 256)
        |
        v
Lightning proposal interface + explicit geometry
        |
        v
original-image coordinate mapping
        |
        v
versioned streamed HDF5 writer
  .partial -> validate -> atomic publish
```

The stable separation is:

- `data.py`: bounded, read-only HDF5 frame access;
- `training_data.py`: deterministic normalization and research datasets;
- `model.py`: importable PyTorch Lightning proposal module;
- `geometry.py`: coordinate, tangent, curvature, FOV, orientation, and
  resampling operations;
- `inference.py`: explicitly exploratory frame/batch/stream adapter;
- `io.py`: model-independent output persistence and validation.

HDF5 assumptions do not live inside the Lightning module.

## Geometry convention

Pixel `(0, 0)` is the center of the upper-left pixel. `x` increases right and
`y` increases down. A centerline point is in the field of view exactly when
`0 <= x < image_width` and `0 <= y < image_height`. Tangent angle is
`atan2(dy, dx)` wrapped to `[-pi, pi)`; increasing angle therefore turns
clockwise in a displayed image. Curvature is the wrapped tangent-angle
difference divided by original-image arc length, with one-sided endpoints and
centered interiors.

The research models predict 100 body positions in a 256×192 raster. Inference
maps these to the actual input raster before computing tangents, curvature, or
FOV membership:

```text
x_original = x_model * original_width / 256
y_original = y_model * original_height / 192
```

The training/evaluation metric is symmetric to forward/reverse ordering because
true anatomical head/tail labels are absent. The exploratory export keeps model
order and sets head/tail probability to exactly `0.5`; it does not invent a
canonical anatomical orientation.

## Investigated proposal representations

### Direct coordinates

The shared convolutional encoder predicts 200 unconstrained normalized
coordinates plus 100 support logits. EXP-0004 showed catastrophic zigzag
topology and implausible body length. This representation was rejected.

### Intrinsic tangent basis

The intrinsic head predicts anchor `(x,y)`, positive length, global orientation,
16 bounded cosine-basis coefficients, and 100 support logits. The centerline is
reconstructed by integrating unit tangents around anchor index 50. This enforces
a smooth connected curve and prevented the coordinate topology explosion.

EXP-0004 compressed the final 12×16 feature map to 2×2. EXP-0007 changed only
that retained grid to 4×4, increasing the model to 722,137 parameters. The
rescue improved primary-fold median localization but still collapsed toward
short, straight mean poses and failed every ordinary-frame geometry gate.
Intrinsic structure is therefore an informative research direction, not an
accepted final model.

## Visibility and uncertainty semantics

Three concepts remain separate:

- `in_fov_mask` is deterministic half-open geometric membership;
- `image_support_probability` is the learned estimate of usable local image
  evidence;
- pose/angle uncertainty concerns uncertainty in the anatomical prediction.

The rejected checkpoint has only the support head. It has no calibrated
head/tail, pose uncertainty, or quality heads. The exploratory adapter therefore
exports conservative, machine-readable sentinels:

| field | exploratory value | interpretation |
|---|---:|---|
| `head_tail_probability` | `0.5` | exact unknown-orientation tie; model order retained |
| `angle_uncertainty` | `pi` radians | uncalibrated maximum-width sentinel, not an interval claim |
| `quality_score` | `0` | rejected/not quality-qualified, not a probability |

Support calibration evidence must not be described as pose uncertainty
calibration. The final support reliability plot says this explicitly.

## Temporal model and refinement

Neither component was implemented as a scientific model because the canonical
Phase-3 gate requires a reliable ordinary-frame proposal first. The public
exploratory API exposes `temporal_window_status()` and raises
`TemporalInferenceUnsupported` for temporal prediction. It never silently runs
independent frames and labels that temporal inference.

The differentiable Tier C renderer was accepted as a geometry/gradient test
fixture, but image-space refinement was not attempted: the proposal was far
outside a plausible refinement basin. FOV-censored refinement, temporal priors,
particles, and calibrated pose uncertainty remain blocked research branches.

## Inference interfaces

`ExploratoryPoseInference` supports:

- one grayscale frame `[H,W]`;
- an independent-frame batch `[B,H,W]` on a selected CPU/CUDA device; and
- streamed full/partial HDF5 ranges through `infer_hdf5`.

Every entry point requires `allow_exploratory=True` or
`--allow-exploratory`. It also binds the checkpoint SHA, variant, encoder pool,
body-point count, and input dimensions to an explicit final config before it
accesses the source or creates output. Uint8 input
uses `[0,255]`; floating input must already be in `[0,1]`; ambiguous integer
formats are rejected. The source dataset path is explicit rather than guessed.

## HDF5 persistence

The implemented version `1.0.0` schema is documented in
`docs/OUTPUT_SCHEMA.md`. `PoseHDF5Writer`:

1. rejects lexical and resolved aliases of the source for both destination and
   partial paths;
2. creates `<output>.partial` on the destination filesystem;
3. appends frame-major compressed chunks;
4. validates shapes, dtypes, finite values, probability ranges, monotonic
   frame/timestamp mappings, and recomputed FOV masks;
5. marks the group complete and atomically publishes only after validation.

Interrupted partials are preserved and resume is explicitly unsupported.

## Performance scope

The reported EXP-0007 batch-32 rate of 2,461 samples/s includes synthetic NumPy
input creation, deterministic preprocessing, transfer, and model forward with
CUDA synchronization. It excludes HDF5 reads and output serialization. It is a
proposal-only in-memory benchmark, not a final storage-inclusive system
throughput claim. No accepted model exists for the latter benchmark.

## Revisit conditions

Model development should resume only after manually annotated Tier A evidence
can distinguish localization, scale, shape, and orientation errors on real
images. A new architecture must be preregistered, pass the unchanged ordinary
frame gates on the primary development fold, and then earn authorization for
additional folds, cropped-FOV testing, temporal context, and the untouched
holdout in that order.
