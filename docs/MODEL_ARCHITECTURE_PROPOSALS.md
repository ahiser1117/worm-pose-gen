# Model architecture proposals: generalization and temporal consistency

Written 2026-10-16 on branch `model-architecture-survey`. A survey of current
architectures, pretrained models and training schemes that could replace or
extend the body-field network (`src/worm_pose_gen/body_net.py`), with the
reason each might beat it. Each proposal ends with the measurement that would decide it; P1, P2, P3, P5 and
P9 were tested on 2026-10-16/17, see [`ARCH_SURVEY_RESULTS.md`](ARCH_SURVEY_RESULTS.md).

## 1. What the current model is, and where it is weak

- **Architecture.** ResNet-18 U-Net (ImageNet weights, 11 M encoder
  parameters), one flat-fielded NIR frame plus three symmetric difference
  channels `frame[t+lag] - frame[t-lag]` at lags 1, 4 and 16 (0.05, 0.2,
  0.8 s), fused at the first convolution. Five outputs: mask, A-P field,
  head and tail heatmaps, overlap. About 145 fps at batch 16 on an A6000.
- **Data.** 250-330 hand-labeled frames per model (`nir-flv-labels-279`,
  `on-food-finetune-329`), of which only ~13 show self-contact, all in train.
  The non-mask targets come from tube fits to the hand masks, and the tail
  end is noisy because people cannot place it consistently. Against that,
  every recording holds tens of thousands of unlabeled frames at 20 Hz.
- **Temporal behaviour.** The network sees ±0.8 s of motion only as three
  pixel differences squeezed through one 7×7 stem convolution, and every
  frame is predicted independently. Nothing in training asks frame *t* and
  frame *t+1* to agree, so the A-P field, the ends and (at contacts) the
  mask can flicker; the pipeline then spends propagation chains, beams and
  Viterbi linking (plan steps 5-6) repairing what the network leaves
  inconsistent. The lag ablation found every lag set equal on held-out
  frames, which says less about the value of motion than about how weakly
  early fusion can use it and that the held-out set has no contact frames.
- **Generalization.** A new setup (on-food) needed its own 50 labels and a
  separate model. An ImageNet ResNet-18 trained on ~300 frames has little
  reason to transfer across illumination, magnification or substrate.

The proposals below attack three things in turn: **what the network is
trained on** (unlabeled video, synthetic coils), **what it sees in time**
(feature-level temporal modelling instead of input differences), and **what
it starts from** (a stronger pretrained backbone). They are ordered by
expected gain per unit of work; the first two need no new architecture.

## 2. Proposals

### P1. Semi-supervised training on unlabeled video with temporal losses (highest priority)

Lightning Pose (Biderman et al., Nature Methods 2024) showed, for animal
keypoints with few labels, that adding unsupervised penalties on unlabeled
video (temporal continuity, plausible posture) to the supervised loss cuts
both error and jitter, and that the gain is largest on out-of-distribution
animals. Our situation is the same: few labels, unlimited video, and the
failure we care about is temporal.

Add a second loader that draws short unlabeled clips (e.g. 3-5 frames at
stride 1-4) from every registered recording, and train on the sum of the
current supervised loss and:

- **Flow-warped field consistency.** Warp the A-P and mask predictions of
  frame *t* to *t+k* with an optical-flow estimate (P4) or, more simply and
  more robustly here, with the pipeline's own fitted centerlines (a body
  point at arc position *s* in frame *t* should be at arc position *s* in
  frame *t+1*). Penalize disagreement on pixels where the warp is
  trustworthy (non-overlap, both frames confidently masked).
- **End continuity.** The head peak cannot move more than the measured
  motion scale between frames (`project-fixed-body-smoother`), so penalize
  head displacement beyond it, and penalize a head/tail swap between
  neighbouring frames (the expected A-P value at the predicted head should
  stay near 0). Weight the tail low or not at all (it is the unreliable end).
- **Augmentation equivariance.** Predictions of a flipped or rotated clip,
  un-flipped, should match the original (FixMatch/mean-teacher style), which
  regularizes the backbone toward features that do not depend on the
  particular 300 frames.

A mean teacher (EMA of the student) produces the targets for the
consistency terms, so the student is not chasing its own noise.

*Why it might win:* it is the only proposal that uses the >10⁶ unlabeled
frames we already have, it is aimed directly at flicker, and it changes
nothing at inference (same network, same throughput, same field store).
*Risk:* a consistency loss can be satisfied by smoothing through real fast
events (reversals, omega entry); keep the weight small and gate it by the
pipeline's ambiguity flags.
*Decide by:* head flips and length jumps per minute on the sequence set and
the held-out minutes (`scripts/evaluate_sequence_set.py`), plus a new
flicker measure (mean |ΔA-P| along tracked body points between frames).

### P2. Self-training from the pipeline's own confident fits

The pipeline already turns network output into whole-recording poses with
propagation and linking, and its confident stretches (IoU > 0.95, no
ambiguity flags, consistent head over seconds) are far better targets than
a per-frame network. Render A-P, head and overlap targets from those poses
for thousands of unlabeled frames, including frames just inside coils that
propagation bridged from clear anchors, and retrain on them with the hand
labels. This is the CoTracker3 recipe (teachers pseudo-label real video for
a student, beating 1000× more synthetic data) with the pose pipeline as the
teacher. Two or three rounds (train → analyse → re-select confident frames)
is the usual pattern.

*Why it might win:* contact and coil frames are exactly where hand labels are
scarce (13) and where the pipeline's temporal reasoning already does better
than the network; distilling it back makes the network itself temporally
informed. It also solves the on-food problem without new labels: analyse
on-food recordings with the current model, keep the confident stretches,
fine-tune.
*Risk:* confirmation bias (the network learns the pipeline's systematic
errors, e.g. a wrong tail). Keep the tail out of the pseudo-targets as with
P1, and audit a sample of pseudo-labels in the Labeling page.
*Decide by:* held-out hand-label metrics must not drop; sequence-set flips,
jumps and frames below IoU 0.9 should fall.

### P3. Synthetic coils and contacts from the tube renderer

WormPose (Hebert et al., PLOS Comp Biol 2021) and DeepTangle (Alonso &
Kirkegaard 2023; DeepTangleCrawl, PLOS Comp Biol 2025) both train on
synthetic worms because coiled and overlapping postures are rare and hard to
label, and both generalize to real video. We have the parts already: the
tube renderer (`renderer.py`, `tube_kernels.py`), per-recording width
profiles, and real pose sequences from analysed recordings. Generate short
synthetic clips by taking real pose trajectories (or perturbing them into
tighter coils), rendering them with a texture model fit to real NIR frames
(darker where the body overlaps, as noted in plan §4), and compositing onto
real empty backgrounds. Every target, including overlap and the A-P value on
both layers of a crossing, is exact.

*Why it might win:* it is the only cheap way to get many held-out-quality
contact examples; it also gives exact targets through crossings, where tube
fits to hand masks are least reliable.
*Risk:* the sim-to-real gap in texture. Mix synthetic and real batches and
check that real held-out metrics do not drop.
*Decide by:* a new held-out set of hand-labeled contact frames (needed
anyway; today val/test contain none).

### P4. Feature-level temporal modelling instead of input differences

Replace early fusion with a shared per-frame encoder and a temporal module
over features. Two designs, in increasing cost:

- **Temporal shift / temporal convolution in the decoder.** TSM (Lin et al.
  2019) shifts a fraction of channels to the neighbouring frames at zero
  FLOPs; a 1-D temporal convolution at the 1/16 and 1/32 levels is similar.
  The encoder stays ImageNet- or DINO-pretrained and per-frame.
- **Temporal attention head, as in Video Depth Anything** (Chen et al.,
  CVPR 2025). VDA turned a per-frame model into a temporally consistent one
  by keeping the pretrained encoder per-frame and adding four temporal
  attention layers in the decoder only, attending across frames at each
  spatial position, trained with a temporal-gradient matching loss; keeping
  temporal layers out of the backbone protected it from scarce video data.
  Our analogue: a bidirectional window of ±8-16 frames at a few strides,
  temporal attention at 1/16 and 1/8 resolution, and the P1 losses as the
  temporal-gradient term.

*Why it might win:* motion becomes a learned, high-level cue (which part is
leading, which body segment is which through a crossing) instead of a raw
intensity difference a stem must decode; the window can be ±1-2 s instead
of the ±0.8 s the lag channels allow. Analysis is offline, so the window can
look both ways.
*Cost:* the encoder still runs once per frame. Encoder features can be cached
per frame (as `field_store` caches outputs), so the window adds only the
temporal head; expect 60-80% of today's throughput, not a factor of the
window length.
*Decide by:* the same lag/no-lag comparison as the October ablation, but on
held-out contact frames and on sequence-set flips. Without P1/P2 this has
the same data problem as today (the temporal layers would see ~300 labeled
windows), so build it after P1.

### P5. A stronger pretrained backbone: DINOv3 ConvNeXt or ViT-S

DINOv3 (Meta, Aug 2025) gives dense features that are state of the art for
segmentation and depth with a frozen backbone and a thin head, and ships
ConvNeXt-T/S/B/L distilled from the 7B model alongside ViT-S/B/L. ConvNeXt-T
(28 M) or -S is a drop-in for the U-Net encoder: hierarchical 1/4-1/32
features, so the decoder and full-resolution skip stay as they are. A ViT-S/16
needs a DPT-style head plus a high-resolution convolutional skip, since a
worm is ~40 px wide and 16 px patches blur its edges.

*Why it might win:* self-supervised features from 1.7 B images are far less
tied to the training illumination than ImageNet-supervised ResNet-18
features, which is the generalization gap seen on on-food. Frozen or
low-learning-rate encoders also overfit 300 labels less.
*Risks:* NIR grayscale microscopy is far from web photos (replicate the
channel, as today); the DINOv3 license is a custom gated Meta license, so
check it suits redistribution of trained weights in the lab library; ViT
throughput at 732×968 is several times lower than ResNet-18.
*Decide by:* cross-setup generalization: train on nir-flv labels only and
score on-food held-out labels without fine-tuning, against the ResNet-18
baseline trained the same way. Within-setup median IoU is already 0.97 and
will not separate them.

### P6. Optical flow or point tracks as input and teacher

SEA-RAFT (ECCV 2024) is a fast, well-generalizing optical-flow network.
Flow channels instead of (or beside) difference channels give the network
motion direction, not just where intensity changed, which is the cue for the
leading end. Flow is also what P1's warping needs if the pipeline's poses
are not used. Point trackers such as CoTracker3 could carry A-P values from
a clear frame through a coil as pseudo-labels.

*Why it might win:* explicit motion is a much cleaner signal than a lag-16
difference, which at fast crawling is two overlapping worms.
*Risk:* the worm body is nearly textureless in NIR; flow and point trackers
trained on natural video may track the background or slide along the body.
Test on a few clips by eye before building on it. Flow adds a network pass
per frame pair.
*Decide by:* qualitative check first (does the flow follow the body
through a reversal and a crossing?), then the P4 comparison with flow
inputs.

### P7. Predict the body curve, not only fields (DeepTangle-style)

DeepTangle regresses each worm directly as a spline in a learned latent
space, and predicts the past and future splines as well, which is what makes
its tracks consistent. Our tube fit already uses a 20-value latent (16
tangent-angle coefficients, rotation, length, centroid). A query-based head
(DETR-style) on the encoder could regress that latent for frames *t-k..t+k*
with the head end fixed by order, trained on hand-traced poses plus P2/P3
targets. Its output would be a start for `mask_fit` that is already
temporally coherent, and its disagreement with the fit is an ambiguity cue.

*Why it might win:* a curve with an ordered head cannot flip locally or
split A-P values across a crossing, which fields can; predicting a short
trajectory forces temporal coherence into the representation.
*Risk:* regression of a whole curve from ~300 labels is data-hungry; this
only makes sense after P2/P3 supply thousands of targets. Keep the fields
(the fitter scores against them) and add the curve as an extra head.
*Decide by:* start quality (IoU after zero fit steps, steps to converge) and
flips on the sequence set.

### P8. Video foundation models as teachers, not as the deployed model

- **SAM 2.1 / SAM 3 / SAM 3.1** (Meta; SAM 3 Nov 2025, SAM 3.1 Mar 2026)
  segment and track objects through video with a memory bank, and fine-tuning
  code is public. They know nothing of A-P or heads, but as a mask teacher
  they could propagate a mask through contacts and give a second opinion for
  P2's confident-frame selection. Too heavy to run on every frame of an hour.
- **V-JEPA 2.1** (Meta, Mar 2026) is a video encoder whose recipe targets
  temporally consistent dense features. As a frozen teacher, its features
  could be distilled into the ResNet/ConvNeXt student over unlabeled clips
  (a feature-matching loss beside P1), passing on temporal structure without
  its ViT-L cost at inference.

*Why it might win:* both bring temporal knowledge learned from far more video
than we will ever label. *Risk:* domain gap, compute for distillation, and
for V-JEPA the 16-frame, low-resolution clip format. Lower priority than
P1-P3.

### P9. Ensembles and test-time augmentation (cheap baseline)

Lightning Pose's ensemble Kalman smoother and plain deep ensembles improve
accuracy and give a calibrated uncertainty. Three seeds or a flip TTA of the
current model, averaged before the fitter, costs 2-4× network time (the
network is now a small share of the pipeline) and the ensemble variance is a
per-pixel ambiguity cue the linking stage could use. This is the floor every
other proposal should beat.

## 3. Suggested order

1. **Measurement first.** Label 30-50 held-out contact/coil frames from
   recordings not in train, and add a per-frame flicker metric to the
   sequence evaluation. Without them none of the proposals can show what
   they are for.
2. **P9** as the baseline to beat (a day).
3. **P1 + P2** with the current ResNet-18 U-Net: the largest expected gain
   in temporal consistency and on-food generalization, no inference change.
4. **P3** to cover contacts.
5. **P5** (ConvNeXt-T DINOv3 encoder) and **P4** (temporal head), each on top
   of the P1/P2 training; they need the extra data to pay off.
6. **P6, P7, P8** as research bets once the above have settled.

## Sources

- Lightning Pose: [Biderman et al. (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10168383/)
- Video Depth Anything: [arXiv 2501.12375](https://arxiv.org/abs/2501.12375v1); online variant [arXiv 2510.09182](https://ar5iv.arxiv.org/html/2510.09182)
- CoTracker3: [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.html)
- DeepTangle: [arXiv 2301.04460](https://arxiv.org/abs/2301.04460v2); DeepTangleCrawl: [PMC12360645](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12360645/)
- WormPose: [PLOS Comp Biol 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008914)
- Deep-Pose-Tracker: [Sci Rep 2026](https://www.nature.com/articles/s41598-026-63642-5); unsupervised worm pose embeddings: [eLife reviewed preprint](https://elifesciences.org/reviewed-preprints/106593)
- DINOv3: [Meta](https://ai.meta.com/research/dinov3/), [model card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/blob/main/README.md)
- SAM 3 / 3.1: [Ultralytics docs](https://docs.ultralytics.com/models/sam-3), [SAM 3.1 news](https://gigazine.net/gsc_news/en/20260331-meta-releasing-sam-3-1); SAM 2.1 fine-tuning: [SurgiSAM2](https://arxiv.org/pdf/2503.03942)
- V-JEPA 2 / 2.1: [repository README](https://cdn.jsdelivr.net/gh/facebookresearch/vjepa2@main/README.md), [Transformers docs](https://huggingface.co/docs/transformers/model_doc/vjepa2)
- SEA-RAFT: [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1065_ECCV_2024_paper.php)
- Flow-warped consistency without video labels: [arXiv 2110.08893](https://ar5iv.arxiv.org/html/2110.08893)
- TSM: Lin, Gan & Han, "TSM: Temporal Shift Module for Efficient Video Understanding", ICCV 2019
