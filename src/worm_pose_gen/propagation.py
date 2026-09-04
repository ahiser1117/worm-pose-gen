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
from .mask_fit import Initialization, MaskFitConfig, MaskFitResult


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


def warm_schedule(config: BatchFitConfig, fraction: float = 0.7, minimum_steps: int = 10) -> BatchFitConfig:
    """The schedule for a start carried over from a neighbouring frame.

    The same stages and rates as the independent fits, so the total energies
    are measured on the same raster and can be compared, with the step
    counts scaled by ``fraction``.  A much shorter schedule (20 and 40 steps
    against 60 and 100) could not keep up with the change between two frames
    of a forming coil and lost to the poor independent fit on energy.
    """

    return replace(
        config, stage_steps=tuple(max(minimum_steps, int(round(fraction * steps))) for steps in config.stage_steps)
    )


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
    warm = warm_schedule(config) if warm_config is None else warm_config
    chains: list[dict[str, Any]] = []
    for a, b in stretches:
        if propagation.forward and a - 1 >= 0 and fitted[a - 1] and not in_stretch[a - 1]:
            chains.append({"source": "forward", "rows": list(range(a, b + 1)), "pose": _pose_of(arrays, a - 1), "anchor": a - 1})
        if propagation.backward and b + 1 < n and fitted[b + 1] and not in_stretch[b + 1]:
            chains.append({"source": "backward", "rows": list(range(b, a - 1, -1)), "pose": _pose_of(arrays, b + 1), "anchor": b + 1})
    candidates: dict[int, list[Candidate]] = defaultdict(list)
    steps = 0
    rows_fit = 0
    longest = max((len(c["rows"]) for c in chains), default=0)
    for k in range(longest):
        batch_masks = []
        batch_starts = []
        batch_owner = []
        for chain in chains:
            if k >= len(chain["rows"]):
                continue
            row = chain["rows"][k]
            mask = masks.get(row)
            if mask is None or not np.asarray(mask).any():
                continue
            latent, width_px, shape = chain["pose"]
            batch_masks.append(np.asarray(mask, dtype=bool))
            batch_starts.append([warm_initialization(latent, width_px, shape, f"warm_{chain['source']}")])
            batch_owner.append((chain, row))
        if not batch_masks:
            continue
        results = fit_masks(batch_masks, batch_starts, width_template=width_template, config=warm, device=device)
        steps += 1
        for (chain, row), result in zip(batch_owner, results, strict=True):
            total = float(result.records[result.best_index]["final_energy"])
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
        "longest_stretch": longest,
    }
    return candidates, info


def select_candidates(
    candidates: dict[int, list[Candidate]], arrays: dict[str, np.ndarray]
) -> dict[int, Candidate]:
    """Per row, the propagated candidate with the lowest total energy, if it beats the independent fit."""

    chosen: dict[int, Candidate] = {}
    for row, options in candidates.items():
        best = min(options, key=lambda c: c.total_energy)
        independent = float(arrays["total_energy"][row]) if arrays["fitted"][row] else float("inf")
        if not np.isfinite(independent) or best.total_energy < independent:
            chosen[row] = best
    return chosen
