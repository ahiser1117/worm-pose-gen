"""Field-of-view aware utilities for truncated worm observations.

The camera boundary is an observation boundary, not an anatomical endpoint.
This module therefore operates on the *raw* threshold mask, before zero-padded
morphology can move a genuine boundary contact inward.  It also provides a
small virtual-mask construction for stable skeletonization and an explicitly
censored centerline completion primitive.

All coordinates are ``(x, y)`` pixel coordinates.  In-FOV support follows the
half-open convention ``0 <= x < width, 0 <= y < height``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from .classical import _skeleton_longest_path, _thin


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
BoundaryState = Literal[
    "fully_visible", "one_end_truncated", "two_sided_truncated", "boundary_uncertain"
]
EndName = Literal["start", "end"]
SideName = Literal["top", "right", "bottom", "left"]


@dataclass(frozen=True)
class BoundaryContact:
    """One connected raw-mask contact with the conservative edge band."""

    center_xy: tuple[float, float]
    sides: tuple[SideName, ...]
    endpoint: EndName | None
    pixel_count: int
    endpoint_distance_px: float


@dataclass(frozen=True)
class BoundaryTruncationResult:
    """Classification made from the largest raw foreground component."""

    state: BoundaryState
    contacts: tuple[BoundaryContact, ...]
    contact_ends: tuple[EndName, ...]
    edge_band_px: int
    component_mask: BoolArray
    diagnostics: dict[str, float | int | bool | str]


@dataclass(frozen=True)
class BoundarySkeletonResult:
    """Skeleton produced after adding virtual tubes beyond contact boundaries."""

    centerline_xy: FloatArray | None
    visible_centerline_xy: FloatArray | None
    skeleton: BoolArray
    padded_mask: BoolArray
    offset_xy: tuple[float, float]
    diagnostics: dict[str, float | int | bool | str]


@dataclass(frozen=True)
class CenterlineCompletionResult:
    """A visible centerline plus any explicitly inferred off-camera stations."""

    centerline_xy: FloatArray
    observed_support: BoolArray
    in_fov: BoolArray
    complete: bool
    ambiguous: bool
    diagnostics: dict[str, float | int | bool | str]


def _largest_component(mask: BoolArray) -> BoolArray:
    remaining = np.asarray(mask, dtype=bool).copy()
    output = np.zeros_like(remaining)
    largest: list[tuple[int, int]] = []
    height, width = remaining.shape
    for y, x in np.argwhere(remaining):
        yi, xi = int(y), int(x)
        if not remaining[yi, xi]:
            continue
        remaining[yi, xi] = False
        queue = deque([(yi, xi)])
        component: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if remaining[ny, nx]:
                        remaining[ny, nx] = False
                        queue.append((ny, nx))
        if len(component) > len(largest):
            largest = component
    if largest:
        yy, xx = np.asarray(largest).T
        output[yy, xx] = True
    return output


def _components(mask: BoolArray) -> list[NDArray[np.int64]]:
    remaining = mask.copy()
    height, width = remaining.shape
    result: list[NDArray[np.int64]] = []
    for y, x in np.argwhere(remaining):
        yi, xi = int(y), int(x)
        if not remaining[yi, xi]:
            continue
        remaining[yi, xi] = False
        queue = deque([(yi, xi)])
        points: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            points.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if remaining[ny, nx]:
                        remaining[ny, nx] = False
                        queue.append((ny, nx))
        result.append(np.asarray(points, dtype=np.int64))
    return result


def classify_boundary_truncation(
    raw_mask: NDArray[np.generic],
    *,
    close_radius: int = 2,
    worm_radius_px: float = 7.0,
) -> BoundaryTruncationResult:
    """Classify boundary censoring from a pre-morphology binary mask.

    The edge band is ``ceil(close_radius + worm_radius_px)``.  A contact is
    considered end-like only when it is close to an endpoint of the raw
    component's longest skeleton path.  Contacts not explained by one or both
    path endpoints are deliberately marked ``boundary_uncertain`` rather than
    being interpreted as missing anatomy.
    """

    values = np.asarray(raw_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("raw_mask must be two-dimensional")
    if close_radius < 0:
        raise ValueError("close_radius must be non-negative")
    if not np.isfinite(worm_radius_px) or worm_radius_px < 0:
        raise ValueError("worm_radius_px must be finite and non-negative")
    band = max(1, int(math.ceil(close_radius + worm_radius_px)))
    component = _largest_component(values)
    height, width = component.shape
    diagnostics: dict[str, float | int | bool | str] = {
        "raw_foreground_area": int(values.sum()),
        "largest_component_area": int(component.sum()),
        "edge_band_px": band,
        "classification_basis": "raw_pre_morphology_mask",
    }
    if not np.any(component):
        diagnostics.update(contact_count=0, reason="no_foreground")
        return BoundaryTruncationResult(
            "boundary_uncertain", (), (), band, component, diagnostics
        )

    yy, xx = np.mgrid[:height, :width]
    distance_to_edge = np.minimum.reduce((yy, xx, height - 1 - yy, width - 1 - xx))
    contact_regions = _components(component & (distance_to_edge < band))
    path, endpoint_count, branch_pixels = _skeleton_longest_path(_thin(component))
    diagnostics.update(
        raw_skeleton_endpoint_count=int(endpoint_count),
        raw_skeleton_branch_pixels=int(branch_pixels),
        contact_count=len(contact_regions),
    )
    if not contact_regions:
        diagnostics["reason"] = "no_edge_band_contact"
        return BoundaryTruncationResult("fully_visible", (), (), band, component, diagnostics)
    if path is None:
        diagnostics["reason"] = "raw_skeleton_has_no_ordered_path"
        return BoundaryTruncationResult(
            "boundary_uncertain", (), (), band, component, diagnostics
        )

    endpoints = np.asarray((path[0], path[-1]))
    association_limit = float(band + worm_radius_px + 2.0)
    contacts: list[BoundaryContact] = []
    for region_yx in contact_regions:
        region_xy = region_yx[:, ::-1].astype(np.float64)
        center = np.median(region_xy, axis=0)
        distances = np.linalg.norm(region_xy[:, None, :] - endpoints[None, :, :], axis=2)
        nearest_index = int(np.unravel_index(np.argmin(distances), distances.shape)[1])
        endpoint_distance = float(distances[:, nearest_index].min())
        endpoint: EndName | None = (
            ("start" if nearest_index == 0 else "end")
            if endpoint_distance <= association_limit
            else None
        )
        ry, rx = region_yx.T
        sides: list[SideName] = []
        if np.any(ry < band):
            sides.append("top")
        if np.any(rx >= width - band):
            sides.append("right")
        if np.any(ry >= height - band):
            sides.append("bottom")
        if np.any(rx < band):
            sides.append("left")
        contacts.append(
            BoundaryContact(
                (float(center[0]), float(center[1])),
                tuple(sides),
                endpoint,
                int(len(region_yx)),
                endpoint_distance,
            )
        )

    associated = tuple(contact.endpoint for contact in contacts if contact.endpoint is not None)
    unique_ends = tuple(end for end in ("start", "end") if end in associated)
    if endpoint_count != 2 or branch_pixels > 0:
        state: BoundaryState = "boundary_uncertain"
        diagnostics["reason"] = "raw_skeleton_topology_is_not_a_simple_path"
    elif len(contacts) > 2 or len(associated) != len(contacts) or len(unique_ends) != len(contacts):
        state: BoundaryState = "boundary_uncertain"
        diagnostics["reason"] = "contact_not_uniquely_associated_with_path_endpoint"
    elif len(unique_ends) == 1:
        state = "one_end_truncated"
        diagnostics["reason"] = "one_endpoint_meets_conservative_edge_band"
    elif len(unique_ends) == 2:
        state = "two_sided_truncated"
        diagnostics["reason"] = "both_endpoints_meet_conservative_edge_band"
    else:
        state = "boundary_uncertain"
        diagnostics["reason"] = "unresolved_edge_contact"
    diagnostics["associated_endpoint_count"] = len(unique_ends)
    return BoundaryTruncationResult(
        state, tuple(contacts), unique_ends, band, component, diagnostics
    )


def _outward_direction(sides: tuple[SideName, ...]) -> FloatArray:
    direction = np.zeros(2, dtype=np.float64)
    for side in sides:
        direction += {
            "top": np.asarray((0.0, -1.0)),
            "right": np.asarray((1.0, 0.0)),
            "bottom": np.asarray((0.0, 1.0)),
            "left": np.asarray((-1.0, 0.0)),
        }[side]
    magnitude = float(np.linalg.norm(direction))
    if magnitude == 0:
        raise ValueError("boundary contact has no side")
    return direction / magnitude


def build_boundary_stable_skeleton(
    raw_mask: NDArray[np.generic],
    truncation: BoundaryTruncationResult,
    *,
    extension_px: float | None = None,
    tube_radius_px: float | None = None,
) -> BoundarySkeletonResult:
    """Skeletonize after attaching virtual tubes across classified contacts.

    The returned coordinates use the original image coordinate system, so
    virtual points have negative or beyond-FOV coordinates.  Only confidently
    endpoint-associated contacts are extended.
    """

    values = np.asarray(raw_mask, dtype=bool)
    if values.ndim != 2 or values.shape != truncation.component_mask.shape:
        raise ValueError("raw_mask must match truncation.component_mask shape")
    radius = (
        max(1.0, 0.5 * truncation.edge_band_px)
        if tube_radius_px is None
        else float(tube_radius_px)
    )
    extension = (
        float(2 * truncation.edge_band_px)
        if extension_px is None
        else float(extension_px)
    )
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("tube_radius_px must be finite and positive")
    if not np.isfinite(extension) or extension <= 0:
        raise ValueError("extension_px must be finite and positive")
    padding = int(math.ceil(extension + radius + truncation.edge_band_px + 2))
    component = truncation.component_mask
    padded = np.zeros(
        (component.shape[0] + 2 * padding, component.shape[1] + 2 * padding), dtype=bool
    )
    padded[padding : padding + component.shape[0], padding : padding + component.shape[1]] = component

    base_path, _, _ = _skeleton_longest_path(_thin(component))
    extended_contacts = 0
    yy, xx = np.mgrid[: padded.shape[0], : padded.shape[1]]
    if base_path is not None:
        endpoint_by_name = {"start": base_path[0], "end": base_path[-1]}
        for contact in truncation.contacts:
            if contact.endpoint is None or not contact.sides:
                continue
            start = endpoint_by_name[contact.endpoint]
            edge_normal = _outward_direction(contact.sides)
            context = min(8, len(base_path) - 1)
            if contact.endpoint == "start":
                tangent = base_path[0] - base_path[context]
            else:
                tangent = base_path[-1] - base_path[-1 - context]
            tangent_magnitude = float(np.linalg.norm(tangent))
            direction = edge_normal
            if tangent_magnitude > 1e-9:
                tangent = tangent / tangent_magnitude
                outward_component = float(np.dot(tangent, edge_normal))
                if outward_component > 0.2:
                    direction = tangent
                elif outward_component > 0.0:
                    # Keep the observed obliqueness, but prevent a nearly
                    # tangential collar from running along the camera edge.
                    direction = tangent + (0.2 - outward_component) * edge_normal
                    direction /= np.linalg.norm(direction)
            # Ray distance to the first crossed half-pixel camera boundary.
            exit_distances: list[float] = []
            if direction[0] < -1e-9:
                exit_distances.append((start[0] + 0.5) / -direction[0])
            elif direction[0] > 1e-9:
                exit_distances.append((component.shape[1] - 0.5 - start[0]) / direction[0])
            if direction[1] < -1e-9:
                exit_distances.append((start[1] + 0.5) / -direction[1])
            elif direction[1] > 1e-9:
                exit_distances.append((component.shape[0] - 0.5 - start[1]) / direction[1])
            positive_exit = [value for value in exit_distances if value >= 0.0]
            distance_to_boundary = min(positive_exit) if positive_exit else 0.0
            length = max(0.0, distance_to_boundary) + extension
            samples = max(2, int(math.ceil(length * 2.0)) + 1)
            for center in start[None, :] + np.linspace(0.0, length, samples)[:, None] * direction:
                cx, cy = center + padding
                padded |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
            extended_contacts += 1

    skeleton = _thin(padded)
    path, endpoint_count, branch_pixels = _skeleton_longest_path(skeleton)
    centerline: FloatArray | None = None
    visible: FloatArray | None = None
    if path is not None:
        centerline = path - np.asarray((padding, padding), dtype=np.float64)
        in_fov = _in_fov(centerline, component.shape)
        visible = centerline[in_fov]
    diagnostics: dict[str, float | int | bool | str] = {
        "virtual_contact_count": extended_contacts,
        "padding_px": padding,
        "tube_radius_px": radius,
        "extension_px": extension,
        "skeleton_endpoint_count": int(endpoint_count),
        "skeleton_branch_pixels": int(branch_pixels),
        "has_ordered_path": path is not None,
    }
    return BoundarySkeletonResult(
        centerline, visible, skeleton, padded, (float(padding), float(padding)), diagnostics
    )


def _polyline_length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _in_fov(points: FloatArray, image_shape: tuple[int, int]) -> BoolArray:
    height, width = map(int, image_shape)
    return (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < height)
    )


def _robust_terminal_fit(
    oriented_points: FloatArray, context_points: int, max_abs_curvature: float
) -> tuple[float, float]:
    local = oriented_points[-min(context_points, len(oriented_points)) :]
    delta = np.diff(local, axis=0)
    length = np.linalg.norm(delta, axis=1)
    if np.any(length <= 1e-9):
        raise ValueError("consecutive centerline points must be distinct")
    arc = np.concatenate(([0.0], np.cumsum(length)))
    position = 0.5 * (arc[:-1] + arc[1:]) - arc[-1]
    angle = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    keep = np.ones(len(angle), dtype=bool)
    coefficients = np.asarray((angle[-1], 0.0))
    # Two iterations reject an occasional segmentation-hook segment without a
    # SciPy dependency.  Small samples remain ordinary least squares.
    for _ in range(2):
        design = np.column_stack((np.ones(keep.sum()), position[keep]))
        coefficients = np.linalg.lstsq(design, angle[keep], rcond=None)[0]
        residual = angle - (coefficients[0] + coefficients[1] * position)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        if len(angle) < 5 or mad <= 1e-8:
            break
        candidate = np.abs(residual - np.median(residual)) <= 3.5 * 1.4826 * mad
        if candidate.sum() < 3 or np.array_equal(candidate, keep):
            break
        keep = candidate
    curvature = float(np.clip(coefficients[1], -max_abs_curvature, max_abs_curvature))
    return float(coefficients[0]), curvature


def _continue_endpoint(
    oriented_points: FloatArray,
    missing_length: float,
    *,
    context_points: int,
    step_px: float,
    max_abs_curvature: float,
) -> FloatArray:
    if missing_length <= 1e-9:
        return np.empty((0, 2), dtype=np.float64)
    angle, curvature = _robust_terminal_fit(
        oriented_points, context_points, max_abs_curvature
    )
    current = oriented_points[-1].copy()
    travelled = 0.0
    output: list[FloatArray] = []
    while travelled < missing_length - 1e-10:
        chord = min(step_px, missing_length - travelled)
        # Midpoint-tangent integration, normalized to an exact chord length,
        # ensures the returned polyline has the requested total length.
        midpoint_angle = angle + 0.5 * curvature * chord
        current = current + chord * np.asarray(
            (math.cos(midpoint_angle), math.sin(midpoint_angle)), dtype=np.float64
        )
        angle += curvature * chord
        output.append(current.copy())
        travelled += chord
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def complete_centerline_to_length(
    centerline_xy: NDArray[np.generic],
    image_shape: tuple[int, int],
    target_length_px: float,
    truncation: BoundaryTruncationResult,
    *,
    missing_length_by_end: Mapping[EndName, float] | None = None,
    context_points: int = 8,
    step_px: float = 1.0,
    max_abs_curvature: float = 0.1,
) -> CenterlineCompletionResult:
    """Complete a censored ordered centerline to a known total body length.

    Contact geometry is re-associated with the supplied curve's endpoints, so
    this function does not assume its orientation matches the raw skeleton.
    With two censored ends, the missing-length split is not identifiable from a
    single frame: callers must provide both ``start`` and ``end`` allocations.
    Without them, the unchanged visible curve is returned with
    ``ambiguous=True`` and ``complete=False``.
    """

    points = np.asarray(centerline_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        raise ValueError("centerline_xy must have shape [N>=3,2]")
    if not np.all(np.isfinite(points)):
        raise ValueError("centerline_xy must be finite")
    height, width = map(int, image_shape)
    if height <= 0 or width <= 0:
        raise ValueError("image_shape dimensions must be positive")
    if not np.isfinite(target_length_px) or target_length_px <= 0:
        raise ValueError("target_length_px must be finite and positive")
    if context_points < 3:
        raise ValueError("context_points must be at least 3")
    if not np.isfinite(step_px) or step_px <= 0:
        raise ValueError("step_px must be finite and positive")
    if not np.isfinite(max_abs_curvature) or max_abs_curvature < 0:
        raise ValueError("max_abs_curvature must be finite and non-negative")
    visible_length = _polyline_length(points)
    tolerance = 1e-6 * max(target_length_px, 1.0)
    if visible_length > target_length_px + tolerance:
        raise ValueError("target_length_px is shorter than the visible centerline")
    missing = max(0.0, target_length_px - visible_length)
    observed = np.ones(len(points), dtype=bool)

    diagnostics: dict[str, float | int | bool | str] = {
        "truncation_state": truncation.state,
        "visible_length_px": visible_length,
        "target_length_px": float(target_length_px),
        "missing_length_px": missing,
        "completion_method": "robust_terminal_tangent_curvature",
    }
    if missing <= tolerance:
        diagnostics.update(start_extension_px=0.0, end_extension_px=0.0, reason="already_full_length")
        return CenterlineCompletionResult(
            points.copy(), observed, _in_fov(points, image_shape), True, False, diagnostics
        )

    # Re-associate physical contacts with this curve's orientation.
    endpoint_coordinates = np.asarray((points[0], points[-1]))
    assigned: list[EndName] = []
    for contact in truncation.contacts:
        distance = np.linalg.norm(
            endpoint_coordinates - np.asarray(contact.center_xy)[None, :], axis=1
        )
        assigned.append("start" if int(np.argmin(distance)) == 0 else "end")
    unique_assigned = tuple(end for end in ("start", "end") if end in assigned)
    diagnostics["curve_contact_end_count"] = len(unique_assigned)

    allocation: dict[EndName, float]
    if len(unique_assigned) == 1:
        allocation = {unique_assigned[0]: missing}
    elif len(unique_assigned) == 2:
        if missing_length_by_end is None:
            diagnostics.update(
                start_extension_px=0.0,
                end_extension_px=0.0,
                reason="two_contact_missing_length_split_required",
            )
            return CenterlineCompletionResult(
                points.copy(), observed, _in_fov(points, image_shape), False, True, diagnostics
            )
        if set(missing_length_by_end) != {"start", "end"}:
            raise ValueError("two-contact missing_length_by_end must contain start and end")
        allocation = {
            "start": float(missing_length_by_end["start"]),
            "end": float(missing_length_by_end["end"]),
        }
        if not all(np.isfinite(value) and value >= 0 for value in allocation.values()):
            raise ValueError("missing-length allocations must be finite and non-negative")
        if not math.isclose(sum(allocation.values()), missing, rel_tol=1e-6, abs_tol=tolerance):
            raise ValueError("missing-length allocations must sum to missing body length")
    else:
        diagnostics.update(
            start_extension_px=0.0,
            end_extension_px=0.0,
            reason="no_unique_censored_endpoint",
        )
        return CenterlineCompletionResult(
            points.copy(), observed, _in_fov(points, image_shape), False, True, diagnostics
        )

    start_length = allocation.get("start", 0.0)
    end_length = allocation.get("end", 0.0)
    start_extension_outward = _continue_endpoint(
        points[::-1],
        start_length,
        context_points=context_points,
        step_px=step_px,
        max_abs_curvature=max_abs_curvature,
    )
    end_extension = _continue_endpoint(
        points,
        end_length,
        context_points=context_points,
        step_px=step_px,
        max_abs_curvature=max_abs_curvature,
    )
    start_extension = start_extension_outward[::-1]
    completed = np.concatenate((start_extension, points, end_extension), axis=0)
    support = np.concatenate(
        (
            np.zeros(len(start_extension), dtype=bool),
            observed,
            np.zeros(len(end_extension), dtype=bool),
        )
    )
    completed_length = _polyline_length(completed)
    diagnostics.update(
        start_extension_px=start_length,
        end_extension_px=end_length,
        completed_length_px=completed_length,
        inferred_station_count=int((~support).sum()),
        reason="completed_to_known_length",
    )
    return CenterlineCompletionResult(
        completed,
        support,
        _in_fov(completed, image_shape),
        bool(math.isclose(completed_length, target_length_px, rel_tol=1e-6, abs_tol=tolerance)),
        False,
        diagnostics,
    )
