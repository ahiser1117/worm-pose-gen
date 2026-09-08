"""Temporal propagation across ambiguous stretches (plan step 5).

Independent per-frame fits are fast and, where the frame is unambiguous,
consistent (step 4: frames with ambiguity score 0 fit at median IoU 0.97).
Coils, spirals and self-contact defeat them, because the start built from
the mask is wrong there and no schedule recovers.  The frames on either side
of such a stretch are good, and at 20 fps consecutive poses differ little,
so the good pose is carried through the stretch: forward from the last good
frame before it and backward from the first good frame after it, each frame
warm-started from its neighbour's fit.  Every stretch of a recording is
propagated at the same time, one lockstep batch per step, so wall-clock is
sequential only over the longest stretch.  Per frame the candidate with the
lowest total energy (overlap plus priors) wins: independent, forward, or
backward.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .batch_fit import BatchFitConfig, fit_masks
from .latent import decode_centerline, unwrap_latent_rotation
from .mask_fit import Initialization, MaskFitConfig, MaskFitResult, redirect_start_through_exit
from .pose_run import touches_border


FloatArray = NDArray[np.float64]


SOURCES = ("independent", "forward", "backward")


@dataclass(frozen=True)
class PropagationConfig:
    # Frames with at least this ambiguity score seed a stretch.
    min_score: int = 2
    # Frames added on each side of a seed, so the chain starts on a frame
    # whose independent fit is trusted and re-fits the marginal ones.
    pad: int = 2
    # Seeds closer than this many frames are one stretch.
    max_gap: int = 3
    forward: bool = True
    backward: bool = True
    # Log-sigma of the length prior inside a chain.  A carried pose already
    # has the right length; a coil lets a looser tube wind further than the
    # worm (spiral_0131: 5% gave 787 px median against a 730 px worm and 53
    # failures, 2% gave 740 px and 1).  With the prior centred on the
    # recording, 2% dragged whole worms of `edge_0528` from 715--742 px
    # toward the 778 px prior; centred on the anchor frame's length it does
    # not, and 3% let a forward chain drift to 750 px inside the spiral.
    # ``None`` keeps the fit's value.  Candidates are compared under the
    # fit's own prior.
    chain_length_sigma: float | None = 0.02
    # Center the chain's length prior on the anchor frame's fitted length
    # rather than the recording's: the worm does not change length between
    # neighbouring frames, but its fitted length drifts by several percent
    # over a recording (715--790 px within one minute of 2024-05-28-02), so a
    # tight prior on the recording value pulls whole worms off their length.
    chain_length_from_anchor: bool = True
    # Chains whose anchor lengths fall in the same log bucket share a batch.
    anchor_length_bucket: float = 0.02
    # When the mask reaches the image border and the carried start does not
    # leave the image, also try the start redirected off camera.
    redirect_at_border: bool = True
    # A body cut by the camera edge fits the visible mask equally well whether
    # the tube stops at the edge or leaves it, so energy cannot prefer the
    # continuation.  On a frame flagged ``edge_inside`` a chain candidate that
    # leaves the image wins if its energy is within this tolerance of the
    # independent fit.  Off by default: at 0.005 it did nothing on the frames
    # it was meant for (163 and 193 of the 2024-05-28-02 minute, where no
    # chain candidate came that close) and accepted a lower-overlap candidate
    # on frame 64 (0.943 -> 0.918).  A proper temporal smoothness term is
    # the step 6 answer.
    edge_tolerance: float = 0.0
    # Plan step 6a.  A chain frame is started not only from a copy of the
    # neighbour's pose but also from a first-order prediction, the pose
    # extrapolated with damping from the chain's last two frames (shape
    # coefficients, rotation and centroid; the length is held), and the
    # fit is pulled toward that prediction by a Gaussian temporal prior
    # with a sigma of ``temporal_prior_sigma_widths`` body widths.  Damping
    # 0 makes the prediction the copy and turns the step off; weight 0
    # turns the prior off.  Weight: on three minutes (plan step 6a) 0.0025
    # changed nothing, 0.025 held the chains on stale predictions through a
    # long coil (23 failures where there were none), 0.01 removed the raw
    # spiral's 14 failures and regressed nothing.
    prediction_damping: float = 0.6
    temporal_prior_weight: float = 0.01
    temporal_prior_sigma_widths: float = 0.5
    # Plan step 6c.  Distinct chain states kept per direction (a single state
    # is fragile: on the coil minute the backward chain landed on a 0.94 or a
    # 0.88 configuration depending on any perturbation), the pose distance
    # below which two states count as the same, and whether the stored
    # independent pose is refit under the chain schedule as a candidate.
    beam: int = 3
    beam_distinct_widths: float = 0.25
    refit_independent: bool = True
    # Per-stretch path selection by dynamic programming instead of the lowest
    # energy per frame: energy over the temperature is the node cost, the
    # squared oriented pose distance in widths and the change of the in-view
    # fraction are the edge costs, and exact mirrors of every candidate are
    # nodes too, so orientation follows the stretch's neighbours.
    path: bool = True
    path_temperature: float = 0.01
    path_distance_weight: float = 1.0
    path_inview_weight: float = 2.0
    # Squared change of log body length between consecutive frames, in units
    # of this sigma: the body does not change length between frames.
    path_length_weight: float = 1.0
    path_length_sigma: float = 0.02
    path_mirrors: bool = True


def warm_schedule(
    config: BatchFitConfig, fraction: float = 0.7, minimum_steps: int = 10, length_sigma: float | None = None
) -> BatchFitConfig:
    """The schedule for a start carried over from a neighbouring frame.

    The same stages and rates as the independent fits, so the total energies
    are measured on the same raster and can be compared, with the step
    counts scaled by ``fraction``.  A much shorter schedule (20 and 40 steps
    against 60 and 100) could not keep up with the change between two frames
    of a forming coil and lost to the poor independent fit on energy.
    ``length_sigma`` tightens the length prior when one is set.
    """

    warm = replace(
        config, stage_steps=tuple(max(minimum_steps, int(round(fraction * steps))) for steps in config.stage_steps)
    )
    if length_sigma is not None and config.length_prior_px is not None:
        warm = replace(warm, length_prior_log_sigma=min(config.length_prior_log_sigma, length_sigma))
    return warm


def ambiguous_stretches(
    score: NDArray[np.generic], fitted: NDArray[np.generic], config: PropagationConfig = PropagationConfig()
) -> list[tuple[int, int]]:
    """Inclusive row ranges around frames whose score reaches ``min_score``."""

    score = np.asarray(score)
    fitted = np.asarray(fitted, dtype=bool)
    n = len(score)
    seeds = np.nonzero(fitted & (score >= config.min_score))[0]
    if not len(seeds):
        return []
    marked = np.zeros(n, dtype=bool)
    for index in seeds:
        marked[max(0, index - config.pad) : min(n, index + config.pad + 1)] = True
    rows = np.nonzero(marked)[0]
    stretches: list[tuple[int, int]] = []
    start = last = int(rows[0])
    for index in rows[1:]:
        if int(index) - last <= config.max_gap + 1:
            last = int(index)
        else:
            stretches.append((start, last))
            start = last = int(index)
    stretches.append((start, last))
    return stretches


def warm_initialization(
    latent: NDArray[np.generic], width_px: float, width_shape: NDArray[np.generic] | None, name: str
) -> Initialization:
    shape = None if width_shape is None or np.size(width_shape) == 0 else np.asarray(width_shape, dtype=np.float64)
    return Initialization(name, np.asarray(latent, dtype=np.float64).copy(), float(width_px), shape)


def prior_penalty(config: MaskFitConfig, body_length_px: float, width_px: float, width_shape: NDArray[np.generic]) -> float:
    """The size and width-profile penalties of a pose, as the fitter adds them to the overlap energy."""

    total = 0.0
    if config.length_bounds_px is not None:
        low, high = config.length_bounds_px
        total += config.bound_weight * (max(low - body_length_px, 0.0) ** 2 + max(body_length_px - high, 0.0) ** 2)
    if config.width_bounds_px is not None:
        low, high = config.width_bounds_px
        total += config.bound_weight * (max(low - width_px, 0.0) ** 2 + max(width_px - high, 0.0) ** 2)
    if config.length_prior_px is not None:
        total += config.prior_weight * ((math.log(body_length_px) - math.log(config.length_prior_px)) / config.length_prior_log_sigma) ** 2
    if config.width_prior_px is not None:
        total += config.prior_weight * ((math.log(width_px) - math.log(config.width_prior_px)) / config.width_prior_log_sigma) ** 2
    shape = np.asarray(width_shape, dtype=np.float64)
    if shape.size:
        mean = np.zeros_like(shape) if config.width_shape_prior_mean is None else np.asarray(config.width_shape_prior_mean, dtype=np.float64)
        total += config.width_shape_prior * float(np.sum((shape - mean) ** 2))
    return float(total)


@dataclass
class Candidate:
    source: str
    result: MaskFitResult
    total_energy: float
    # The chain's prediction of this frame (``None`` on a chain's first frame
    # without velocity), the start that won inside the candidate's fit, and
    # the mean in-view distance of the result to the prediction.
    prediction_xy: np.ndarray | None = None
    start_name: str = ""
    distance_to_prediction_px: float = float("nan")
    # Rank of the chain state that produced this candidate within its direction's beam.
    beam: int = 0


def predict_latent(
    current: NDArray[np.generic], previous: NDArray[np.generic] | None, damping: float, coefficients: int = 16
) -> FloatArray:
    """First-order prediction of the next latent from the last two.

    Shape coefficients, rotation (on the branch nearest the current frame) and
    centroid move on by ``damping`` times their last change; the length is
    held, since the body does not change length between frames.  Without a
    previous pose, or at damping 0, the prediction is the current pose.
    """

    now = np.asarray(current, dtype=np.float64).copy()
    if previous is None or damping <= 0:
        return now
    before = unwrap_latent_rotation(np.asarray(previous, dtype=np.float64), now)
    step = now - before
    step[coefficients + 1] = 0.0  # length
    return now + damping * step


def pose_distance_px(curve_a: NDArray[np.generic], curve_b: NDArray[np.generic], image_shape: tuple[int, int] | None) -> float:
    """Mean distance between two oriented centerlines over the points inside the camera on both."""

    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    if image_shape is None:
        inside = np.ones(len(a), dtype=bool)
    else:
        height, width = image_shape
        inside = (a[:, 0] >= 0) & (a[:, 0] < width) & (a[:, 1] >= 0) & (a[:, 1] < height)
        inside &= (b[:, 0] >= 0) & (b[:, 0] < width) & (b[:, 1] >= 0) & (b[:, 1] < height)
    if inside.sum() < 2:
        return float("nan")
    return float(np.linalg.norm(a[inside] - b[inside], axis=1).mean())


def _pose_of(arrays: dict[str, np.ndarray], row: int) -> tuple[np.ndarray, float, np.ndarray]:
    return arrays["latent"][row], float(arrays["width_px"][row]), arrays["width_shape"][row]


def _chain_config(
    warm: BatchFitConfig, anchor_length: float | None, temporal_sigma_px: float | None = None, temporal_weight: float = 0.0
) -> BatchFitConfig:
    overrides: dict[str, Any] = {}
    if anchor_length is not None and warm.length_prior_px is not None:
        overrides["length_prior_px"] = float(anchor_length)
    if temporal_sigma_px is not None and temporal_weight > 0:
        overrides["temporal_prior_weight"] = float(temporal_weight)
        overrides["temporal_prior_sigma_px"] = float(temporal_sigma_px)
    return replace(warm, **overrides) if overrides else warm


def slow_schedule(config: BatchFitConfig, preset: BatchFitConfig, length_sigma: float | None = None) -> BatchFitConfig:
    """The refit schedule of the second pass: the preset's steps on the fit's own rasters.

    Only the step counts, learning-rate scales and decay are taken from
    ``preset``; the stage rasters and point strides stay the fit's, so the
    candidates' energies are measured where the independent fits' were.
    """

    if preset.stage_downsample != config.stage_downsample or preset.stage_point_stride != config.stage_point_stride:
        raise ValueError("the refit preset must share the fit's stage rasters and point strides")
    slow = replace(
        config, stage_steps=preset.stage_steps, stage_lr_scale=preset.stage_lr_scale, within_stage_decay=preset.within_stage_decay
    )
    if length_sigma is not None and config.length_prior_px is not None:
        slow = replace(slow, length_prior_log_sigma=min(config.length_prior_log_sigma, length_sigma))
    return slow


def _distinct_states(
    results: Sequence[tuple[MaskFitResult, float, Any]], keep: int, min_distance_px: float
) -> list[tuple[MaskFitResult, float, Any]]:
    """The ``keep`` lowest-energy results whose centerlines differ by at least ``min_distance_px``."""

    ordered = sorted(results, key=lambda item: item[1])
    kept: list[tuple[MaskFitResult, float, Any]] = []
    for item in ordered:
        if all(pose_distance_px(item[0].centerline_xy, other[0].centerline_xy, None) > min_distance_px for other in kept):
            kept.append(item)
        if len(kept) >= keep:
            break
    return kept


def propagate(
    arrays: dict[str, np.ndarray],
    stretches: Sequence[tuple[int, int]],
    masks: dict[int, NDArray[np.generic]],
    *,
    config: BatchFitConfig,
    device: Any = None,
    width_template: NDArray[np.generic] | None = None,
    propagation: PropagationConfig = PropagationConfig(),
    warm_config: BatchFitConfig | None = None,
) -> tuple[dict[int, list[Candidate]], dict[str, Any]]:
    """Carry the anchor poses through every stretch in lockstep; returns candidates per row and diagnostics.

    Each direction of each stretch keeps up to ``propagation.beam`` distinct
    chain states (plan step 6c): every state offers the copied pose, the
    first-order prediction and, at the border, the redirected start; the
    results are pooled and the lowest-energy distinct poses become the
    states of the next frame.  With ``refit_independent`` the stored
    independent pose of every stretch frame is also refit under the same
    schedule, so all candidates of a frame are comparable.  ``masks`` maps a
    row to its cleaned mask; rows without a mask are skipped and the chain
    keeps its pose.  A stretch without a good frame on a side gets no chain
    from that side.
    """

    fitted = np.asarray(arrays["fitted"], dtype=bool)
    n = len(fitted)
    in_stretch = np.zeros(n, dtype=bool)
    for a, b in stretches:
        in_stretch[a : b + 1] = True
    warm = warm_schedule(config, length_sigma=propagation.chain_length_sigma) if warm_config is None else warm_config
    beam = max(1, int(propagation.beam))

    def anchor_length(row: int) -> float | None:
        if not propagation.chain_length_from_anchor or warm.length_prior_px is None:
            return None
        return float(arrays["body_length_px"][row])

    # The frame before the anchor (on the anchor's side) gives the chain an
    # initial velocity when it is a fitted frame outside any stretch.
    def previous_pose(anchor: int, direction: int) -> np.ndarray | None:
        before = anchor + direction
        if 0 <= before < n and fitted[before] and not in_stretch[before]:
            return arrays["latent"][before]
        return None

    chains: list[dict[str, Any]] = []
    for a, b in stretches:
        if propagation.forward and a - 1 >= 0 and fitted[a - 1] and not in_stretch[a - 1]:
            chains.append({
                "source": "forward", "rows": list(range(a, b + 1)), "anchor": a - 1, "length": anchor_length(a - 1),
                "states": [{"pose": _pose_of(arrays, a - 1), "previous": previous_pose(a - 1, -1)}],
            })
        if propagation.backward and b + 1 < n and fitted[b + 1] and not in_stretch[b + 1]:
            chains.append({
                "source": "backward", "rows": list(range(b, a - 1, -1)), "anchor": b + 1, "length": anchor_length(b + 1),
                "states": [{"pose": _pose_of(arrays, b + 1), "previous": previous_pose(b + 1, +1)}],
            })
    image_shape: tuple[int, int] | None = None
    for mask in masks.values():
        image_shape = (int(np.asarray(mask).shape[0]), int(np.asarray(mask).shape[1]))
        break
    candidates: dict[int, list[Candidate]] = defaultdict(list)
    steps = 0
    rows_fit = 0
    redirects = 0
    predictions_offered = 0
    predictions_won = 0
    independent_refits = 0

    # The independent poses of the stretch frames, refit under the chain
    # schedule with the stretch's anchor length prior (the anchors know the
    # body's length; the recording prior at 5% let these refits step the
    # length where the path switched to them).  One batch per stretch.
    if propagation.refit_independent:
        for a, b in stretches:
            rows = [r for r in range(a, b + 1) if fitted[r] and r in masks and np.asarray(masks[r]).any()]
            if not rows:
                continue
            anchors = [anchor_length(r) for r in (a - 1, b + 1) if 0 <= r < n and fitted[r] and not in_stretch[r]]
            anchors = [v for v in anchors if v is not None]
            length = float(np.exp(np.mean(np.log(anchors)))) if anchors else None
            independent_config = _chain_config(warm, length)
            for chunk_start in range(0, len(rows), max(1, config.max_rows)):
                chunk = rows[chunk_start : chunk_start + max(1, config.max_rows)]
                starts = [[warm_initialization(*_pose_of(arrays, r), "independent_refit")] for r in chunk]
                results = fit_masks([np.asarray(masks[r], dtype=bool) for r in chunk], starts, width_template=width_template, config=independent_config, device=device)
                for r, result in zip(chunk, results, strict=True):
                    candidates[int(r)].append(Candidate("independent", result, comparable_energy(config, result), None, "independent_refit", float("nan")))
                    independent_refits += 1

    longest = max((len(c["rows"]) for c in chains), default=0)
    for k in range(longest):
        # Chains are batched by anchor-length bucket, since the length prior
        # is part of the configuration of a fit_masks call.
        groups: dict[int | None, list[tuple[dict[str, Any], int]]] = defaultdict(list)
        for chain in chains:
            if k >= len(chain["rows"]):
                continue
            row = chain["rows"][k]
            mask = masks.get(row)
            if mask is None or not np.asarray(mask).any():
                continue
            key = None if chain["length"] is None else int(round(math.log(chain["length"]) / propagation.anchor_length_bucket))
            groups[key].append((chain, row))
        if not groups:
            continue
        steps += 1
        for key, members in groups.items():
            sigma_px = propagation.temporal_prior_sigma_widths * float(np.mean([s["pose"][1] for c, _ in members for s in c["states"]]))
            group_config = _chain_config(
                warm, None if key is None else float(np.exp(np.mean([math.log(c["length"]) for c, _ in members]))),
                sigma_px, propagation.temporal_prior_weight,
            )
            # One fit row per (chain, state, start), so every start's result
            # survives to the beam selection.
            batch_masks = []
            batch_starts = []
            batch_references: list[np.ndarray | None] = []
            owners: list[tuple[dict[str, Any], int, dict[str, Any], np.ndarray | None]] = []
            for chain, row in members:
                binary = np.asarray(masks[row], dtype=bool)
                at_border = propagation.redirect_at_border and touches_border(binary, 2)
                for state in chain["states"]:
                    latent, width_px, shape = state["pose"]
                    starts = [warm_initialization(latent, width_px, shape, f"warm_{chain['source']}")]
                    prediction_xy: np.ndarray | None = None
                    if state["previous"] is not None and propagation.prediction_damping > 0:
                        predicted = predict_latent(latent, state["previous"], propagation.prediction_damping, config.coefficients)
                        prediction_xy = decode_centerline(predicted, config.coefficients)
                        if pose_distance_px(prediction_xy, decode_centerline(latent, config.coefficients), None) > 0.5:
                            starts.append(warm_initialization(predicted, width_px, shape, f"predicted_{chain['source']}"))
                            predictions_offered += 1
                    if at_border:
                        redirected = redirect_start_through_exit(starts[0], binary, config=group_config)
                        if redirected is not None:
                            starts.append(redirected)
                            redirects += 1
                    # The temporal prior pulls toward the prediction, or toward
                    # the copied pose when the state has no velocity yet.
                    reference = prediction_xy if prediction_xy is not None else decode_centerline(latent, config.coefficients)
                    for start in starts:
                        batch_masks.append(binary)
                        batch_starts.append([start])
                        batch_references.append(reference if propagation.temporal_prior_weight > 0 else None)
                        owners.append((chain, row, state, prediction_xy))
            results = fit_masks(
                batch_masks, batch_starts, width_template=width_template, config=group_config, device=device, references=batch_references
            )
            rows_fit += len(results)
            # Pool each chain's results and keep the beam.  States are ranked
            # by the fit's full energy, temporal prior included, so the chain
            # keeps the states consistent with its own motion; the candidate
            # handed to the path carries the comparable energy (overlap plus
            # the fit's priors), which every candidate of a frame shares.
            pooled: dict[int, list[tuple[MaskFitResult, float, Any]]] = defaultdict(list)
            for (chain, row, state, prediction_xy), result in zip(owners, results, strict=True):
                pooled[id(chain)].append((result, float(result.records[result.best_index]["final_energy"]), (chain, row, state, prediction_xy)))
            for chain, row in members:
                items = pooled.get(id(chain), [])
                if not items:
                    continue
                width_px = float(np.mean([s["pose"][1] for s in chain["states"]]))
                kept = _distinct_states(items, beam, propagation.beam_distinct_widths * width_px)
                new_states = []
                for index, (result, _, (_, _, state, prediction_xy)) in enumerate(kept):
                    start_name = str(result.initializations[result.best_index].name)
                    if start_name.startswith("predicted_"):
                        predictions_won += 1
                    distance = float("nan") if prediction_xy is None else pose_distance_px(result.centerline_xy, prediction_xy, image_shape)
                    candidates[row].append(
                        Candidate(chain["source"], result, comparable_energy(config, result), prediction_xy, start_name, distance, beam=index)
                    )
                    new_states.append({"pose": (result.latent, result.width_px, result.width_shape), "previous": state["pose"][0]})
                chain["states"] = new_states
    info = {
        "stretches": [[int(a), int(b)] for a, b in stretches],
        "frames_in_stretches": int(in_stretch.sum()),
        "chains": len(chains),
        "chains_forward": sum(c["source"] == "forward" for c in chains),
        "chains_backward": sum(c["source"] == "backward" for c in chains),
        "lockstep_steps": steps,
        "warm_fits": rows_fit,
        "independent_refits": independent_refits,
        "redirected_starts": redirects,
        "longest_stretch": longest,
        "chain_length_sigma": warm.length_prior_log_sigma if warm.length_prior_px is not None else None,
        "chain_length_from_anchor": bool(propagation.chain_length_from_anchor and warm.length_prior_px is not None),
        "anchor_lengths_px": sorted(round(c["length"]) for c in chains if c["length"] is not None),
        "prediction_damping": propagation.prediction_damping,
        "temporal_prior_weight": propagation.temporal_prior_weight,
        "temporal_prior_sigma_widths": propagation.temporal_prior_sigma_widths,
        "predicted_starts_offered": predictions_offered,
        "predicted_starts_won": predictions_won,
        "beam": beam,
        "schedule_steps": list(warm.stage_steps),
    }
    return candidates, info


OVERRIDE_ENERGY = 1e-3


@dataclass
class PathChoice:
    candidate: Candidate
    mirrored: bool
    # Whether the path took a candidate other than the lowest-energy one of its frame, and by how much energy.
    override: bool
    energy_gap: float
    cost: float


def _oriented_curve(candidate: Candidate, mirrored: bool) -> np.ndarray:
    curve = np.asarray(candidate.result.centerline_xy, dtype=np.float64)
    return curve[::-1] if mirrored else curve


def select_path(
    candidates: dict[int, list[Candidate]],
    arrays: dict[str, np.ndarray],
    stretches: Sequence[tuple[int, int]],
    config: MaskFitConfig,
    propagation: PropagationConfig = PropagationConfig(),
    image_shape: tuple[int, int] | None = None,
) -> dict[int, PathChoice]:
    """One candidate per frame along each stretch, chosen by dynamic programming (plan step 6c).

    Nodes are a frame's candidates and, with ``path_mirrors``, their exact
    mirrors; a frame without candidates keeps its stored pose as its only
    node.  Node cost is the comparable energy over ``path_temperature``; the
    edge cost between consecutive frames is ``path_distance_weight`` times
    the squared oriented pose distance in widths plus
    ``path_inview_weight`` times the change of the in-view fraction.  The
    fitted frames just outside the stretch anchor the path on both sides,
    so a stretch becomes one continuous track and its orientation follows
    its neighbours.  Rows without a stored fit and without candidates break
    the stretch into independently solved pieces.
    """

    fitted = np.asarray(arrays["fitted"], dtype=bool)
    n = len(fitted)
    n_points = int(arrays["centerline_xy"].shape[1])
    chosen: dict[int, PathChoice] = {}

    Geometry = tuple[np.ndarray, int, float, float]  # centerline, points in view, width, body length

    def stored_node(row: int) -> Geometry | None:
        if not fitted[row]:
            return None
        return (
            np.asarray(arrays["centerline_xy"][row], dtype=np.float64), int(arrays["points_in_fov"][row]),
            float(arrays["width_px"][row]), float(arrays["body_length_px"][row]),
        )

    def edge_cost(prev: Geometry, cur: Geometry) -> float:
        width = 0.5 * (prev[2] + cur[2])
        distance = pose_distance_px(prev[0], cur[0], image_shape)
        if not np.isfinite(distance):
            distance = 0.0
        length_change = math.log(max(cur[3], 1.0) / max(prev[3], 1.0)) / propagation.path_length_sigma
        return (
            propagation.path_distance_weight * (distance / max(width, 1.0)) ** 2
            + propagation.path_inview_weight * abs(prev[1] - cur[1]) / n_points
            + propagation.path_length_weight * length_change**2
        )

    for a, b in stretches:
        rows = list(range(a, b + 1))
        # Nodes per row: (geometry, node cost, PathChoice-or-None for a stored pose).
        nodes: list[list[tuple[Geometry, float, tuple[Candidate, bool] | None]]] = []
        for row in rows:
            options = candidates.get(row, [])
            row_nodes = []
            if options:
                best_energy = min(c.total_energy for c in options)
                for candidate in options:
                    geometry = (
                        np.asarray(candidate.result.centerline_xy, dtype=np.float64), int(candidate.result.points_in_fov),
                        float(candidate.result.width_px), float(candidate.result.body_length_px),
                    )
                    row_nodes.append((geometry, candidate.total_energy / propagation.path_temperature, (candidate, False)))
                    if propagation.path_mirrors:
                        mirrored = (_oriented_curve(candidate, True), geometry[1], geometry[2], geometry[3])
                        row_nodes.append((mirrored, candidate.total_energy / propagation.path_temperature, (candidate, True)))
            else:
                stored = stored_node(row)
                if stored is not None:
                    row_nodes.append((stored, float(arrays["total_energy"][row]) / propagation.path_temperature, None))
            nodes.append(row_nodes)
        # Anchors: the fitted frames outside the stretch, fixed and free.
        before = stored_node(a - 1) if a - 1 >= 0 and fitted[a - 1] else None
        after = stored_node(b + 1) if b + 1 < n and fitted[b + 1] else None
        # Solve piecewise between rows without nodes.
        start = 0
        while start < len(rows):
            if not nodes[start]:
                start += 1
                continue
            end = start
            while end + 1 < len(rows) and nodes[end + 1]:
                end += 1
            piece = nodes[start : end + 1]
            anchor_before = before if start == 0 else None
            anchor_after = after if end == len(rows) - 1 else None
            costs = []
            back: list[list[int]] = []
            first = [node[1] + (edge_cost(anchor_before, node[0]) if anchor_before is not None else 0.0) for node in piece[0]]
            costs.append(first)
            back.append([-1] * len(piece[0]))
            for k in range(1, len(piece)):
                current = []
                pointers = []
                for node in piece[k]:
                    best_cost = math.inf
                    best_j = -1
                    for j, prev_node in enumerate(piece[k - 1]):
                        total = costs[k - 1][j] + edge_cost(prev_node[0], node[0])
                        if total < best_cost:
                            best_cost, best_j = total, j
                    current.append(best_cost + node[1])
                    pointers.append(best_j)
                costs.append(current)
                back.append(pointers)
            final = [c + (edge_cost(node[0], anchor_after) if anchor_after is not None else 0.0) for c, node in zip(costs[-1], piece[-1], strict=True)]
            j = int(np.argmin(final))
            for k in range(len(piece) - 1, -1, -1):
                row = rows[start + k]
                geometry, _, choice = piece[k][j]
                if choice is not None:
                    candidate, mirrored = choice
                    options = candidates[row]
                    best_energy = min(c.total_energy for c in options)
                    # An override is a choice measurably above the frame's lowest energy, not a tie broken the other way.
                    chosen[row] = PathChoice(
                        candidate, mirrored, candidate.total_energy > best_energy + OVERRIDE_ENERGY, candidate.total_energy - best_energy, float(costs[k][j])
                    )
                j = back[k][j]
            start = end + 1
    return chosen


def continuity_summary(arrays: dict[str, np.ndarray], *, length_jump_fraction: float = 0.03) -> dict[str, Any]:
    """How continuous the stored track is: length jumps, pose jumps, in-view changes, prediction distances.

    Counts are over consecutive fitted frames.  A length jump is a change of
    more than ``length_jump_fraction`` of the recording prior's length (the
    frame's own length when no prior is stored); a pose jump is the stored
    ``pose_jump_px`` above the fitted width; an in-view change is the body
    entering or leaving the camera (points in view crossing the full count).
    """

    fitted = np.asarray(arrays["fitted"], dtype=bool)
    rows = np.nonzero(fitted)[0]
    pairs = [(a, b) for a, b in zip(rows[:-1], rows[1:], strict=False) if b == a + 1]
    length = np.asarray(arrays["body_length_px"], dtype=np.float64)
    width = np.asarray(arrays["width_px"], dtype=np.float64)
    reference = float(np.nanmedian(length[fitted])) if fitted.any() else float("nan")
    length_jumps = sum(abs(length[b] - length[a]) > length_jump_fraction * reference for a, b in pairs)
    jump = np.asarray(arrays.get("pose_jump_px", np.full(len(fitted), np.nan)), dtype=np.float64)
    with np.errstate(invalid="ignore"):
        pose_jumps = int(np.sum(fitted & (jump > width)))
    n_points = arrays["centerline_xy"].shape[1]
    in_view = np.asarray(arrays["points_in_fov"])
    in_view_changes = sum((in_view[a] >= n_points) != (in_view[b] >= n_points) for a, b in pairs)
    out: dict[str, Any] = {
        "consecutive_pairs": len(pairs),
        "length_jumps_over_fraction": int(length_jumps),
        "length_jump_fraction": length_jump_fraction,
        "length_reference_px": reference,
        "pose_jumps_over_width": pose_jumps,
        "in_view_changes": int(in_view_changes),
    }
    if "prediction_distance_px" in arrays:
        distance = np.asarray(arrays["prediction_distance_px"], dtype=np.float64)
        finite = distance[np.isfinite(distance)]
        out["prediction_distance_px_p50_p90_max"] = [float(v) for v in np.percentile(finite, [50, 90, 100])] if len(finite) else None
        out["frames_with_prediction"] = int(len(finite))
    return out


def comparable_energy(config: MaskFitConfig, result: MaskFitResult) -> float:
    """Overlap energy plus the size and profile priors of ``config``, without the crop-escape term."""

    dice = float(result.records[result.best_index]["final_soft_dice_energy"])
    return dice + prior_penalty(config, result.body_length_px, result.width_px, result.width_shape)


def independent_energy(config: MaskFitConfig, arrays: dict[str, np.ndarray], row: int) -> float:
    """The stored independent fit's energy on the same footing as ``comparable_energy``."""

    if not arrays["fitted"][row]:
        return float("inf")
    return float(arrays["energy"][row]) + prior_penalty(
        config, float(arrays["body_length_px"][row]), float(arrays["width_px"][row]), arrays["width_shape"][row]
    )


def select_candidates(
    candidates: dict[int, list[Candidate]],
    arrays: dict[str, np.ndarray],
    config: MaskFitConfig | None = None,
    propagation: PropagationConfig = PropagationConfig(),
) -> dict[int, Candidate]:
    """Per row, the propagated candidate with the lowest energy, if it beats the independent fit.

    With ``config`` the comparison uses overlap plus that config's priors for
    every candidate; without it the stored total energies are compared.  On a
    frame flagged ``edge_inside`` a candidate that leaves the image also wins
    within ``propagation.edge_tolerance`` of the independent energy.
    """

    edge_inside = arrays.get("flag_edge_inside")
    n_points = arrays["centerline_xy"].shape[1] if "centerline_xy" in arrays else None
    chosen: dict[int, Candidate] = {}
    for row, options in candidates.items():
        best = min(options, key=lambda c: c.total_energy)
        if config is not None:
            independent = independent_energy(config, arrays, row)
        else:
            independent = float(arrays["total_energy"][row]) if arrays["fitted"][row] else float("inf")
        tolerance = 0.0
        if edge_inside is not None and bool(edge_inside[row]) and n_points is not None:
            leaving = [c for c in options if c.result.points_in_fov < n_points]
            if leaving:
                best = min(leaving, key=lambda c: c.total_energy)
                tolerance = propagation.edge_tolerance
        if not np.isfinite(independent) or best.total_energy < independent + tolerance:
            chosen[row] = best
    return chosen
