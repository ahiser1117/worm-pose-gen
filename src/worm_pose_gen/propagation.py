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
from .mask_fit import Initialization, MaskFitConfig, MaskFitResult, redirect_start_through_exit
from .pose_run import touches_border


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


def _pose_of(arrays: dict[str, np.ndarray], row: int) -> tuple[np.ndarray, float, np.ndarray]:
    return arrays["latent"][row], float(arrays["width_px"][row]), arrays["width_shape"][row]


def _chain_config(warm: BatchFitConfig, anchor_length: float | None) -> BatchFitConfig:
    if anchor_length is None or warm.length_prior_px is None:
        return warm
    return replace(warm, length_prior_px=float(anchor_length))


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

    ``masks`` maps a row to its cleaned mask; rows without a mask are skipped
    and the chain keeps its pose.  A stretch without a good frame on a side
    gets no chain from that side.
    """

    fitted = np.asarray(arrays["fitted"], dtype=bool)
    n = len(fitted)
    in_stretch = np.zeros(n, dtype=bool)
    for a, b in stretches:
        in_stretch[a : b + 1] = True
    warm = warm_schedule(config, length_sigma=propagation.chain_length_sigma) if warm_config is None else warm_config
    chains: list[dict[str, Any]] = []
    def anchor_length(row: int) -> float | None:
        if not propagation.chain_length_from_anchor or warm.length_prior_px is None:
            return None
        return float(arrays["body_length_px"][row])

    for a, b in stretches:
        if propagation.forward and a - 1 >= 0 and fitted[a - 1] and not in_stretch[a - 1]:
            chains.append({"source": "forward", "rows": list(range(a, b + 1)), "pose": _pose_of(arrays, a - 1), "anchor": a - 1, "length": anchor_length(a - 1)})
        if propagation.backward and b + 1 < n and fitted[b + 1] and not in_stretch[b + 1]:
            chains.append({"source": "backward", "rows": list(range(b, a - 1, -1)), "pose": _pose_of(arrays, b + 1), "anchor": b + 1, "length": anchor_length(b + 1)})
    candidates: dict[int, list[Candidate]] = defaultdict(list)
    steps = 0
    rows_fit = 0
    redirects = 0
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
            group_config = _chain_config(
                warm, None if key is None else float(np.exp(np.mean([math.log(c["length"]) for c, _ in members])))
            )
            batch_masks = []
            batch_starts = []
            for chain, row in members:
                latent, width_px, shape = chain["pose"]
                binary = np.asarray(masks[row], dtype=bool)
                starts = [warm_initialization(latent, width_px, shape, f"warm_{chain['source']}")]
                if propagation.redirect_at_border and touches_border(binary, 2):
                    redirected = redirect_start_through_exit(starts[0], binary, config=group_config)
                    if redirected is not None:
                        starts.append(redirected)
                        redirects += 1
                batch_masks.append(binary)
                batch_starts.append(starts)
            results = fit_masks(batch_masks, batch_starts, width_template=width_template, config=group_config, device=device)
            for (chain, row), result in zip(members, results, strict=True):
                # Energy under the fit's own prior, so a tighter chain prior
                # shapes the optimization but not the comparison with the
                # independent fit.
                total = comparable_energy(config, result)
                candidates[row].append(Candidate(chain["source"], result, total))
                chain["pose"] = (result.latent, result.width_px, result.width_shape)
                rows_fit += 1
    info = {
        "stretches": [[int(a), int(b)] for a, b in stretches],
        "frames_in_stretches": int(in_stretch.sum()),
        "chains": len(chains),
        "chains_forward": sum(c["source"] == "forward" for c in chains),
        "chains_backward": sum(c["source"] == "backward" for c in chains),
        "lockstep_steps": steps,
        "warm_fits": rows_fit,
        "redirected_starts": redirects,
        "longest_stretch": longest,
        "chain_length_sigma": warm.length_prior_log_sigma if warm.length_prior_px is not None else None,
        "chain_length_from_anchor": bool(propagation.chain_length_from_anchor and warm.length_prior_px is not None),
        "anchor_lengths_px": sorted(round(c["length"]) for c in chains if c["length"] is not None),
    }
    return candidates, info


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
