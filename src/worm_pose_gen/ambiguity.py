"""Per-frame ambiguity signals for fitted poses (plan step 4).

A fit can agree with the mask and still be wrong, or disagree with it for
good reason.  These signals, all cheap and computed from a run's stored
arrays, flag the frames where the single-frame answer should not be trusted
without its neighbours:

- ``area_ratio``: mask area over the fitted tube's area.  Below one the tube
  overlaps itself (a coil or self-contact hides body); above one the tube
  misses body (the midbody optimum of a cold start).
- ``self_contact_px``: smallest distance between centerline points that are
  at least ``min_separation`` points apart along the body.  Below the body
  width the fitted body touches itself.
- ``pose_jump_px``: mean point distance to the previous fitted frame's
  centerline, taken over both orientations.  A jump of a body width in one
  frame at 20 fps is a change of answer, not of pose.
- ``length_deviation``: log of fitted length over the recording prior's
  length; large only when the mask and the prior disagree.
- Mask cleanup statistics already stored: filled hole pixels (a tight omega
  turn encloses background), pixels outside the largest component (a
  dropped tail), and the overlap of the fit itself.
- ``edge_inside``: the mask reaches the image border but every centerline
  point is inside the image, so the body continues off camera and the tube
  stopped or folded at the edge instead of leaving.

``ambiguity_score`` counts the flags that fire; ``summarize_ambiguity``
reports how often each fires and how overlap degrades with the score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


FLAG_NAMES = (
    "low_iou",
    "area_deficit",
    "area_excess",
    "self_contact",
    "holes",
    "fragments",
    "length_deviation",
    "pose_jump",
    "edge_inside",
)


@dataclass(frozen=True)
class AmbiguityThresholds:
    low_iou: float = 0.9
    area_deficit: float = 0.9
    area_excess: float = 1.1
    # Fraction of the fitted width below which two body points touch.
    self_contact_width_fraction: float = 0.8
    holes_px: int = 200
    fragments_px: int = 500
    # Length deviation in units of the prior's log sigma.
    length_sigmas: float = 2.0
    # Jump between consecutive frames, as a multiple of the fitted width.
    jump_width_fraction: float = 1.0
    min_separation: int = 15


def self_contact_px(centerline_xy: NDArray[np.generic], min_separation: int = 15) -> float:
    """Smallest distance between centerline points at least ``min_separation`` apart."""

    points = np.asarray(centerline_xy, dtype=np.float64)
    n = len(points)
    if n <= min_separation:
        return float("inf")
    best = float("inf")
    for offset in range(min_separation, n):
        distance = np.linalg.norm(points[offset:] - points[: n - offset], axis=1).min()
        if distance < best:
            best = float(distance)
    return best


def inside_camera(centerline_xy: NDArray[np.generic], image_shape: tuple[int, int] | None) -> NDArray[np.bool_]:
    """Which centerline points lie inside the image; all of them when the shape is unknown."""

    points = np.asarray(centerline_xy, dtype=np.float64)
    if image_shape is None:
        return np.ones(len(points), dtype=bool)
    height, width = image_shape
    return (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)


def visible_tube_area_px(
    centerline_xy: NDArray[np.generic], width_profile: NDArray[np.generic], image_shape: tuple[int, int] | None
) -> float:
    """Area of the tube along the part of the centerline inside the image.

    A body completed off camera (step 3) has tube area the mask cannot show;
    comparing the mask with the visible part only keeps the area ratio a
    self-overlap signal rather than a clipping signal.
    """

    points = np.asarray(centerline_xy, dtype=np.float64)
    widths = np.asarray(width_profile, dtype=np.float64)
    inside = inside_camera(points, image_shape)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    both = inside[:-1] & inside[1:]
    return float(np.sum(0.5 * (widths[:-1] + widths[1:])[both] * segment[both]))


def pose_jump_px(
    curve_a: NDArray[np.generic], curve_b: NDArray[np.generic], image_shape: tuple[int, int] | None = None
) -> float:
    """Mean point distance between two centerlines, whichever orientation is closer.

    Only points inside the image on both curves count: the off-camera part of
    a completed body is unconstrained and may move freely between frames.
    """

    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    inside_a = inside_camera(a, image_shape)
    best = float("inf")
    for candidate in (b, b[::-1]):
        both = inside_a & inside_camera(candidate, image_shape)
        if both.sum() >= 10:
            best = min(best, float(np.linalg.norm(a[both] - candidate[both], axis=1).mean()))
    return best if np.isfinite(best) else float("nan")


def compute_ambiguity(
    arrays: dict[str, np.ndarray],
    *,
    prior: dict[str, Any] | None = None,
    image_shape: tuple[int, int] | None = None,
    thresholds: AmbiguityThresholds = AmbiguityThresholds(),
) -> dict[str, np.ndarray]:
    """Signals, flags, and score per frame from a run's arrays; unfitted frames stay NaN/False.

    ``image_shape`` (height, width) restricts the area ratio and the pose jump
    to the part of the body inside the camera.
    """

    n = len(arrays["frame_index"])
    fitted = np.asarray(arrays["fitted"], dtype=bool)
    nan = np.full(n, np.nan)
    out: dict[str, np.ndarray] = {
        "area_ratio": nan.copy(),
        "self_contact_px": nan.copy(),
        "pose_jump_px": nan.copy(),
        "length_deviation": nan.copy(),
    }
    previous: int | None = None
    for row in range(n):
        if not fitted[row]:
            continue
        tube = visible_tube_area_px(arrays["centerline_xy"][row], arrays["width_profile"][row], image_shape)
        out["area_ratio"][row] = float(arrays["worm_pixels"][row]) / max(tube, 1.0)
        out["self_contact_px"][row] = self_contact_px(arrays["centerline_xy"][row], thresholds.min_separation)
        if previous is not None and int(arrays["frame_index"][row]) - int(arrays["frame_index"][previous]) == 1:
            out["pose_jump_px"][row] = pose_jump_px(arrays["centerline_xy"][row], arrays["centerline_xy"][previous], image_shape)
        if prior is not None:
            out["length_deviation"][row] = float(np.log(arrays["body_length_px"][row] / prior["length_px"]))
        previous = row
    width = np.asarray(arrays["width_px"], dtype=np.float64)
    n_points = arrays["centerline_xy"].shape[1]
    on_border = np.asarray(arrays.get("mask_on_border", np.zeros(n, dtype=bool)), dtype=bool)
    with np.errstate(invalid="ignore"):
        flags = {
            "low_iou": fitted & (np.asarray(arrays["iou"], dtype=np.float64) < thresholds.low_iou),
            "area_deficit": fitted & (out["area_ratio"] < thresholds.area_deficit),
            "area_excess": fitted & (out["area_ratio"] > thresholds.area_excess),
            "self_contact": fitted & (out["self_contact_px"] < thresholds.self_contact_width_fraction * width),
            "holes": fitted & (np.asarray(arrays["pixels_filled"]) > thresholds.holes_px),
            "fragments": fitted & (np.asarray(arrays["pixels_outside_largest"]) > thresholds.fragments_px),
            "length_deviation": fitted
            & (
                np.abs(out["length_deviation"]) > thresholds.length_sigmas * float(prior["log_length_sigma"])
                if prior is not None
                else np.zeros(n, dtype=bool)
            ),
            "pose_jump": fitted & (out["pose_jump_px"] > thresholds.jump_width_fraction * width),
            "edge_inside": fitted & on_border & (np.asarray(arrays["points_in_fov"]) >= n_points),
        }
    for name in FLAG_NAMES:
        out[f"flag_{name}"] = np.asarray(flags[name], dtype=bool)
    out["ambiguity_score"] = np.sum([out[f"flag_{name}"] for name in FLAG_NAMES], axis=0).astype(np.int64)
    return out


def summarize_ambiguity(arrays: dict[str, np.ndarray], thresholds: AmbiguityThresholds = AmbiguityThresholds()) -> dict[str, Any]:
    """Flag counts and overlap by score, for a run summary."""

    fitted = np.asarray(arrays["fitted"], dtype=bool)
    score = np.asarray(arrays["ambiguity_score"])
    iou = np.asarray(arrays["iou"], dtype=np.float64)
    by_score = {}
    for value in sorted(set(int(v) for v in score[fitted])):
        members = fitted & (score == value)
        by_score[str(value)] = {"frames": int(members.sum()), "iou_median": float(np.median(iou[members]))}
    return {
        "thresholds": asdict(thresholds),
        "flag_counts": {name: int(arrays[f"flag_{name}"].sum()) for name in FLAG_NAMES},
        "frames_with_score_at_least_1": int((fitted & (score >= 1)).sum()),
        "frames_with_score_at_least_2": int((fitted & (score >= 2)).sum()),
        "iou_by_score": by_score,
        "area_ratio_p10_p50_p90": [float(v) for v in np.nanpercentile(arrays["area_ratio"][fitted], [10, 50, 90])] if fitted.any() else None,
        "self_contact_px_p10": float(np.nanpercentile(arrays["self_contact_px"][fitted], 10)) if fitted.any() else None,
        "pose_jump_px_p50_p90_max": [float(v) for v in np.nanpercentile(arrays["pose_jump_px"][fitted], [50, 90, 100])]
        if fitted.any() and np.isfinite(arrays["pose_jump_px"][fitted]).any() else None,
    }
