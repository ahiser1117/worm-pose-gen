"""Mask-native extraction of conservative easy-frame pose anchors."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .classical import _skeleton_longest_path, _thin, resample_centerline, tangent_angles


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class AnchorConfig:
    n_points: int = 100
    min_area: int = 100
    max_area: int = 100_000
    min_length: float = 20.0
    max_length: float = 2_000.0
    boundary_margin: float = 1.0
    required_endpoints: int = 2
    max_branch_pixels: int = 0
    min_width: float = 1.0
    max_width: float = 100.0
    max_width_jump: float = 20.0
    min_render_iou: float = 0.65
    width_step: float = 0.25
    max_topology_spur_length: int = 4
    min_boundary_clearance_widths: float = 0.0


@dataclass(frozen=True)
class AnchorResult:
    accepted: bool
    centerline_xy: FloatArray | None
    tangent_angle: FloatArray | None
    estimated_width: FloatArray | None
    body_length: float | None
    quality_score: float
    head_tail_probability: float
    rejection_reasons: tuple[str, ...]
    qc: dict[str, float | int | bool]
    skeleton: BoolArray
    rendered_mask: BoolArray | None = None

    @property
    def width_profile(self) -> FloatArray | None:
        """Stable downstream alias for the mask-derived width estimate."""

        return self.estimated_width


def _neighbors(node: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    y, x = node
    result: list[tuple[int, int]] = []
    for ny in range(y - 1, y + 2):
        for nx in range(x - 1, x + 2):
            if (ny, nx) == node or (ny, nx) not in pixels:
                continue
            if ny != y and nx != x and ((y, nx) in pixels or (ny, x) in pixels):
                continue
            result.append((ny, nx))
    return result


def skeleton_topology(skeleton: NDArray[np.generic]) -> dict[str, int | bool]:
    """Return strict graph diagnostics for a one-pixel skeleton."""

    values = np.asarray(skeleton, dtype=bool)
    if values.ndim != 2:
        raise ValueError("skeleton must be two-dimensional")
    pixels = {tuple(map(int, point)) for point in np.argwhere(values)}
    if not pixels:
        return {
            "skeleton_pixels": 0,
            "skeleton_components": 0,
            "endpoint_count": 0,
            "branch_pixels": 0,
            "edge_count": 0,
            "has_cycle": False,
            "has_path": False,
        }
    degrees = {point: len(_neighbors(point, pixels)) for point in pixels}
    remaining = set(pixels)
    components = 0
    while remaining:
        components += 1
        start = remaining.pop()
        queue = deque([start])
        while queue:
            point = queue.popleft()
            for adjacent in _neighbors(point, pixels):
                if adjacent in remaining:
                    remaining.remove(adjacent)
                    queue.append(adjacent)
    endpoints = sum(degree == 1 for degree in degrees.values())
    branches = sum(degree > 2 for degree in degrees.values())
    edges = sum(degrees.values()) // 2
    has_cycle = edges > len(pixels) - components
    return {
        "skeleton_pixels": len(pixels),
        "skeleton_components": components,
        "endpoint_count": endpoints,
        "branch_pixels": branches,
        "edge_count": edges,
        "has_cycle": bool(has_cycle),
        "has_path": bool(components == 1 and endpoints >= 2),
    }


def _prune_short_side_spurs(
    skeleton: BoolArray, max_spur_length: int
) -> tuple[BoolArray, int, int]:
    """Remove only short endpoint-to-junction side branches for assessment.

    A simple path has no junction, so its two anatomical endpoints are never
    removed. The returned skeleton is used only for topology assessment; the
    final centerline path is recovered from the original unpruned skeleton.
    """

    if max_spur_length < 0:
        raise ValueError("max_topology_spur_length must be non-negative")
    assessment = np.asarray(skeleton, dtype=bool).copy()
    if max_spur_length == 0:
        return assessment, 0, 0
    pruned_branches = 0
    pruned_pixels = 0
    while True:
        pixels = {tuple(map(int, point)) for point in np.argwhere(assessment)}
        degrees = {point: len(_neighbors(point, pixels)) for point in pixels}
        candidates: list[list[tuple[int, int]]] = []
        for endpoint, degree in degrees.items():
            if degree != 1:
                continue
            chain = [endpoint]
            previous: tuple[int, int] | None = None
            current = endpoint
            terminus_degree = degree
            while len(chain) <= max_spur_length:
                onward = [point for point in _neighbors(current, pixels) if point != previous]
                if not onward:
                    terminus_degree = 0
                    break
                next_point = onward[0]
                terminus_degree = degrees[next_point]
                if terminus_degree != 2:
                    break
                chain.append(next_point)
                previous, current = current, next_point
            if terminus_degree > 2 and len(chain) <= max_spur_length:
                candidates.append(chain)
        if not candidates:
            break
        unique = set().union(*(set(chain) for chain in candidates))
        for y, x in unique:
            assessment[y, x] = False
        pruned_branches += len(candidates)
        pruned_pixels += len(unique)
    return assessment, pruned_branches, pruned_pixels


def _inside(mask: BoolArray, x: float, y: float) -> bool:
    xi, yi = int(round(x)), int(round(y))
    return 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and bool(mask[yi, xi])


def _terminal_curve(points_xy: FloatArray, context_points: int) -> tuple[float, float]:
    """Fit terminal tangent angle and constant curvature from an oriented path."""

    local = points_xy[-min(context_points, len(points_xy)) :]
    difference = np.diff(local, axis=0)
    segment_length = np.linalg.norm(difference, axis=1)
    if np.any(segment_length <= 1e-9):
        raise ValueError("consecutive centerline points must be distinct")
    arc = np.concatenate(([0.0], np.cumsum(segment_length)))
    midpoint = 0.5 * (arc[:-1] + arc[1:]) - arc[-1]
    angle = np.unwrap(np.arctan2(difference[:, 1], difference[:, 0]))
    design = np.column_stack((np.ones_like(midpoint), midpoint))
    terminal_angle, fitted_curvature = np.linalg.lstsq(design, angle, rcond=None)[0]
    return float(terminal_angle), float(fitted_curvature)


def _advance_curve(
    point_xy: FloatArray, tangent_angle: float, curvature: float, distance: float
) -> tuple[FloatArray, float]:
    """Advance exactly along a constant-curvature arc in image coordinates."""

    next_angle = tangent_angle + curvature * distance
    if abs(curvature) < 1e-10:
        offset = distance * np.asarray(
            [math.cos(tangent_angle), math.sin(tangent_angle)], dtype=np.float64
        )
    else:
        offset = np.asarray(
            [
                (math.sin(next_angle) - math.sin(tangent_angle)) / curvature,
                (math.cos(tangent_angle) - math.cos(next_angle)) / curvature,
            ],
            dtype=np.float64,
        )
    return point_xy + offset, next_angle


def _extend_oriented_endpoint(
    points_xy: FloatArray,
    mask: BoolArray,
    *,
    context_points: int,
    step: float,
    max_extension: float,
) -> list[FloatArray]:
    """Extend the final endpoint of a path already oriented toward that end."""

    tangent_angle, fitted_curvature = _terminal_curve(points_xy, context_points)
    current = points_xy[-1].copy()
    travelled = 0.0
    extension: list[FloatArray] = []
    while travelled < max_extension:
        distance = min(step, max_extension - travelled)
        candidate, candidate_angle = _advance_curve(
            current, tangent_angle, fitted_curvature, distance
        )
        if _inside(mask, *candidate):
            extension.append(candidate)
            current, tangent_angle = candidate, candidate_angle
            travelled += distance
            continue

        # Locate the first exit along the curved step. ``low`` remains inside,
        # so the returned endpoint lies on the foreground side of the boundary.
        low, high = 0.0, distance
        for _ in range(32):
            middle = 0.5 * (low + high)
            trial, _ = _advance_curve(current, tangent_angle, fitted_curvature, middle)
            if _inside(mask, *trial):
                low = middle
            else:
                high = middle
        boundary, _ = _advance_curve(current, tangent_angle, fitted_curvature, low)
        if np.linalg.norm(boundary - current) > 1e-7:
            extension.append(boundary)
        return extension
    raise RuntimeError("curve continuation did not reach the mask boundary within max_extension")


def extend_centerline_to_mask_boundary(
    centerline_xy: NDArray[np.generic],
    mask: NDArray[np.generic],
    *,
    context_points: int = 8,
    step: float = 0.25,
    max_extension: float | None = None,
) -> FloatArray:
    """Continue both ends of an ordered centerline to a binary-mask boundary.

    Each end uses a least-squares fit of tangent angle against arc length over
    the local terminal segments.  The fitted angle and signed curvature define
    a circular-arc continuation, rather than a straight terminal-tangent ray.
    Subpixel integration stops at the first foreground exit and bisection places
    the final point on the foreground side of that boundary.

    The input samples are retained exactly and newly integrated samples are
    prepended/appended.  Both input endpoints must lie in ``mask``.
    """

    points = np.asarray(centerline_xy, dtype=np.float64)
    binary = np.asarray(mask, dtype=bool)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        raise ValueError("centerline_xy must have shape [N>=3,2]")
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if not np.all(np.isfinite(points)):
        raise ValueError("centerline_xy must be finite")
    if context_points < 3:
        raise ValueError("context_points must be at least 3")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    if not _inside(binary, *points[0]) or not _inside(binary, *points[-1]):
        raise ValueError("both centerline endpoints must lie inside mask")
    limit = float(math.hypot(*binary.shape)) if max_extension is None else float(max_extension)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("max_extension must be finite and positive")

    tail = _extend_oriented_endpoint(
        points, binary, context_points=context_points, step=step, max_extension=limit
    )
    head_outward = _extend_oriented_endpoint(
        points[::-1], binary, context_points=context_points, step=step, max_extension=limit
    )
    pieces = [np.asarray(head_outward[::-1]), points, np.asarray(tail)]
    return np.concatenate([piece.reshape(-1, 2) for piece in pieces], axis=0)


def estimate_width_along_normals(
    mask: NDArray[np.generic], centerline_xy: NDArray[np.generic], *, step: float = 0.25
) -> FloatArray:
    """Estimate full body width by walking both normals until foreground exit."""

    binary = np.asarray(mask, dtype=bool)
    points = np.asarray(centerline_xy, dtype=np.float64)
    if binary.ndim != 2 or points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("mask must be 2-D and centerline_xy must be [N>=2,2]")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    derivative = np.gradient(points, axis=0)
    magnitude = np.linalg.norm(derivative, axis=1)
    if np.any(magnitude <= 1e-9):
        raise ValueError("centerline tangent is degenerate")
    normal = np.column_stack((-derivative[:, 1], derivative[:, 0])) / magnitude[:, None]
    max_distance = float(math.hypot(*binary.shape))
    widths = np.zeros(len(points), dtype=np.float64)
    for index, (point, direction) in enumerate(zip(points, normal, strict=True)):
        sides: list[float] = []
        for sign in (-1.0, 1.0):
            distance = 0.0
            while distance <= max_distance and _inside(
                binary, *(point + sign * direction * distance)
            ):
                distance += step
            sides.append(max(0.0, distance - step))
        # One pixel accounts for the center pixel shared by both walks.
        widths[index] = sides[0] + sides[1] + 1.0
    return widths


def render_centerline_mask(
    centerline_xy: NDArray[np.generic],
    width: NDArray[np.generic] | float,
    image_shape: tuple[int, int],
) -> BoolArray:
    """Rasterize a nearest-centerline-sample tube using only NumPy."""

    points = np.asarray(centerline_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("centerline_xy must have shape [N>=2,2]")
    height, image_width = map(int, image_shape)
    if height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    diameter = np.asarray(width, dtype=np.float64)
    if diameter.ndim == 0:
        diameter = np.full(len(points), float(diameter))
    else:
        diameter = np.broadcast_to(diameter, (len(points),))
    if not np.all(np.isfinite(diameter)) or np.any(diameter <= 0):
        raise ValueError("width must be finite and positive")
    yy, xx = np.mgrid[:height, :image_width]
    best_distance = np.full((height, image_width), np.inf, dtype=np.float64)
    nearest = np.zeros((height, image_width), dtype=np.int64)
    for index, (x, y) in enumerate(points):
        distance = (xx - x) ** 2 + (yy - y) ** 2
        update = distance < best_distance
        best_distance[update] = distance[update]
        nearest[update] = index
    radius = 0.5 * diameter[nearest]
    return best_distance <= np.square(radius)


def extract_mask_anchor(
    cleaned_mask: NDArray[np.generic],
    probability: NDArray[np.generic] | None = None,
    config: AnchorConfig | None = None,
) -> AnchorResult:
    """Extract a pose anchor only when cleaned-mask topology is unambiguous."""

    cfg = config or AnchorConfig()
    mask = np.asarray(cleaned_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("cleaned_mask must have shape [height, width]")
    soft: FloatArray | None = None
    if probability is not None:
        soft = np.asarray(probability, dtype=np.float64)
        if soft.shape != mask.shape:
            raise ValueError("probability must match cleaned_mask shape")
        if not np.all(np.isfinite(soft)) or np.any((soft < 0) | (soft > 1)):
            raise ValueError("probability must be finite and lie in [0,1]")
    if cfg.n_points < 2 or cfg.width_step <= 0 or cfg.max_topology_spur_length < 0:
        raise ValueError("n_points and width_step must be positive")
    if (
        not np.isfinite(cfg.min_boundary_clearance_widths)
        or cfg.min_boundary_clearance_widths < 0
    ):
        raise ValueError("min_boundary_clearance_widths must be finite and non-negative")
    area = int(mask.sum())
    skeleton = _thin(mask)
    raw_topology = skeleton_topology(skeleton)
    assessment_skeleton, pruned_spur_count, pruned_spur_pixels = _prune_short_side_spurs(
        skeleton, cfg.max_topology_spur_length
    )
    topology = skeleton_topology(assessment_skeleton)
    path, _, _ = _skeleton_longest_path(skeleton)
    boundary_contact = bool(
        area and (np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))
    )
    qc: dict[str, float | int | bool] = {
        "mask_area": area,
        "mask_touches_boundary": boundary_contact,
        **{f"raw_{name}": value for name, value in raw_topology.items()},
        **topology,
        "topology_pruned_spur_count": pruned_spur_count,
        "topology_pruned_spur_pixels": pruned_spur_pixels,
        "cycle_or_no_path": bool(topology["has_cycle"] or not topology["has_path"]),
    }
    if soft is not None:
        qc["mean_probability_in_mask"] = float(soft[mask].mean()) if area else 0.0
        qc["mean_probability_outside_mask"] = float(soft[~mask].mean()) if np.any(~mask) else 0.0
    reasons: list[str] = []
    if not cfg.min_area <= area <= cfg.max_area:
        reasons.append("implausible_area")
    if boundary_contact:
        reasons.append("boundary_contact")
    if topology["skeleton_components"] != 1:
        reasons.append("disconnected_skeleton")
    if topology["endpoint_count"] != cfg.required_endpoints:
        reasons.append("endpoint_count")
    if topology["branch_pixels"] > cfg.max_branch_pixels:
        reasons.append("branch_pixels")
    if topology["has_cycle"]:
        reasons.append("cycle")
    if not topology["has_path"] or path is None:
        reasons.append("no_path")

    centerline: FloatArray | None = None
    angles: FloatArray | None = None
    widths: FloatArray | None = None
    body_length: float | None = None
    rendered: BoolArray | None = None
    if path is not None:
        centerline = resample_centerline(path, cfg.n_points)
        body_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
        angles = tangent_angles(centerline)
        boundary_distance = float(
            np.min(
                np.column_stack(
                    (
                        centerline[:, 0],
                        centerline[:, 1],
                        (mask.shape[1] - 1) - centerline[:, 0],
                        (mask.shape[0] - 1) - centerline[:, 1],
                    )
                )
            )
        )
        qc["body_length_px"] = body_length
        qc["centerline_boundary_distance_px"] = boundary_distance
        if not cfg.min_length <= body_length <= cfg.max_length:
            reasons.append("implausible_length")
        if boundary_distance < cfg.boundary_margin:
            reasons.append("boundary_margin")
        widths = estimate_width_along_normals(mask, centerline, step=cfg.width_step)
        width_jump = float(np.max(np.abs(np.diff(widths)))) if len(widths) > 1 else 0.0
        qc.update(
            {
                "width_min_px": float(widths.min()),
                "width_median_px": float(np.median(widths)),
                "width_max_px": float(widths.max()),
                "width_max_adjacent_jump_px": width_jump,
            }
        )
        median_width = float(np.median(widths))
        required_boundary_clearance = (
            cfg.min_boundary_clearance_widths * median_width
        )
        qc.update(
            {
                "min_boundary_clearance_widths": float(
                    cfg.min_boundary_clearance_widths
                ),
                "required_boundary_clearance_px": float(
                    required_boundary_clearance
                ),
                "boundary_clearance_width_ratio": float(
                    boundary_distance / max(median_width, np.finfo(float).eps)
                ),
            }
        )
        if boundary_distance < required_boundary_clearance:
            reasons.append("width_relative_boundary_clearance")
        if float(widths.min()) < cfg.min_width or float(widths.max()) > cfg.max_width:
            reasons.append("implausible_width")
        if width_jump > cfg.max_width_jump:
            reasons.append("abrupt_width_jump")
        rendered = render_centerline_mask(centerline, widths, mask.shape)
        intersection = int(np.logical_and(mask, rendered).sum())
        union = int(np.logical_or(mask, rendered).sum())
        iou = intersection / union if union else 0.0
        residual = int(np.logical_xor(mask, rendered).sum()) / max(union, 1)
        qc["mask_render_iou"] = float(iou)
        qc["mask_render_residual"] = float(residual)
        if iou < cfg.min_render_iou:
            reasons.append("low_render_iou")

    rejection_reasons = tuple(dict.fromkeys(reasons))
    accepted = not rejection_reasons
    iou_value = float(qc.get("mask_render_iou", 0.0))
    topology_quality = 1.0 if (
        topology["endpoint_count"] == cfg.required_endpoints
        and topology["branch_pixels"] <= cfg.max_branch_pixels
        and not topology["has_cycle"]
    ) else 0.0
    quality = float(np.clip(0.7 * iou_value + 0.3 * topology_quality, 0.0, 1.0))
    return AnchorResult(
        accepted=accepted,
        centerline_xy=centerline if accepted else None,
        tangent_angle=angles if accepted else None,
        estimated_width=widths if accepted else None,
        body_length=body_length if accepted else None,
        quality_score=quality,
        head_tail_probability=0.5,
        rejection_reasons=rejection_reasons,
        qc=qc,
        skeleton=skeleton,
        rendered_mask=rendered,
    )


# Descriptive aliases for early internal callers.
render_numpy_tube = render_centerline_mask


def extract_anchor_from_mask(
    cleaned_mask: NDArray[np.generic], config: AnchorConfig | None = None
) -> AnchorResult:
    return extract_mask_anchor(cleaned_mask, probability=None, config=config)
