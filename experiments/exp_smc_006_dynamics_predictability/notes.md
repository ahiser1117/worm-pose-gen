# EXP-SMC-006 — temporal dynamics predictability

## Hypothesis

Strict classical-anchor runs contain enough recording-balanced adjacent poses
to show that a simple factored latent transition predicts one and multiple
frames better than persistence.

## Rationale and method

Encode every contiguous strict-anchor run with the EXP-SMC-003 cubic K=16
representation. Resolve endpoint ordering deterministically at each run start,
then by minimum point distance within the run, and unwrap global rotation only
within a run. Compare persistence, full constant latent velocity, global
translation/rotation velocity with shape hold, and diagonal shape AR(1)/AR(2)
under leave-one-recording-out fitting. Evaluate 1, 2, 5, 10, and 20 frame
horizons when actual contiguous evidence exists.

The prerequisite density audit already indicates a likely coverage failure;
the frozen gate nevertheless quantifies exactly what evidence is absent rather
than silently pooling one dominant recording. If it fails, use only a declared
conservative prior for synthetic/controlled SMC and make no natural-dynamics
claim. The protected holdout remains closed.

## Status

`COMPLETED_NOT_SUPPORTED_SESSION_GENERAL`.

## Quantitative results

The evidence gate failed exactly where the density audit predicted: adjacent
pairs were 7/45/0 by recording, five-frame predictions were 0/16/0, and no
20-frame prediction existed. Persistence achieved 3.77 px median mean-point
error at one frame (52 cases), 5.36 px at two (40), 8.75 px at five (16), and
10.00 px at ten (only four cases from one recording).

On the 40 starts shared by models needing two history frames, paired
persistence error was 3.57 px. The best non-persistence diagnostic—global
translation/orientation velocity with shape hold—was 4.48 px, 25.6% worse.
It was worse in both contributing recordings. Full latent velocity, AR(1), and
AR(2) also degraded rapidly with horizon. The initial unpaired comparison is
preserved as `metrics_unpaired_invalid.json` and is not evidence.

## Visual evidence

[`dynamics_horizon.png`](figures/dynamics_horizon.png) shows persistence
dominating every fitted or velocity diagnostic on common forecast keys. Lines
beyond two frames are based on increasingly sparse evidence; the paired counts
are 40/30/13/3 at horizons 1/2/5/10, with no horizon-20 case. The earlier
unpaired figure is preserved with the invalid metrics and is not evidence.

## Failure analysis and decision

H5, as stated, is unsupported by the available strict-anchor runs. The failure
has two parts: transition coverage is absent in one session and heavily
imbalanced in the others, and the observed one-step changes do not support a
velocity mean. The frozen config named a constant-velocity fallback, but the
paired result falsified that choice. The immutable `decision_addendum.json`
therefore revises the controlled-experiment prior to a zero-drift block-
diagonal random walk. Its scale is a synthetic control parameter, not an
empirical natural-motion estimate.

Multi-horizon scoring takes the better forward/reversed target ordering, so it
does not use hidden intervening orientation as evaluation truth. AR fitting
still pools lexicographically initialized run orientations and is therefore an
instability diagnostic, not a general falsification of unoriented AR dynamics.
Likewise, “block-diagonal”
describes only the intended synthetic factorization: no empirical covariance
or process-noise scale has been validated.

## Consequence

Proceed only to EXP-SMC-007 implementation and controlled known-pose recovery.
Passing synthetic SMC can establish algorithm correctness and characterize
misspecification tolerance; it cannot establish natural difficult-bout
reconstruction. Natural SMC experiments require either denser trustworthy
anchors or prospectively annotated temporal bouts. The protected holdout stays
closed.
