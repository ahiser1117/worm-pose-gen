"""Visible-only repair of centerlines censored by the camera boundary.

The image boundary is not an anatomical end.  Skeleton samples within roughly
one worm radius of a boundary contact are consequently treated as unreliable:
only the remaining interior core is smoothed, and the omitted *in-image*
portion is reconstructed to a crossing measured from the raw (pre-morphology)
mask.  No hidden, off-camera anatomy is produced.

Coordinates are ``(x, y)`` pixel centres.  The closed camera rectangle is
``[0, width - 1] x [0, height - 1]``; a censored endpoint lies exactly on that
rectangle's boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .fov_completion import BoundaryContact, BoundaryTruncationResult


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
EndName = Literal["start", "end"]


@dataclass(frozen=True)
class EdgeCensoredResult:
    """Result of fitting a reliable core and restoring only its visible ends.

    ``reliable_core_mask`` distinguishes samples used by the fit from the
    reconstructed edge-band samples. ``observed_support`` reports whether each
    returned sample falls on the supplied raw component; it does not turn the
    reconstructed samples into fitting observations. ``censored_endpoint_mask``
    marks only terminal samples that represent camera crossings.
    """

    success: bool
    centerline_xy: FloatArray | None
    reliable_core_mask: BoolArray | None
    observed_support: BoolArray | None
    censored_endpoint_mask: BoolArray | None
    censored_ends: tuple[EndName, ...]
    core_slice: tuple[int, int] | None
    boundary_crossings_xy: dict[EndName, tuple[float, float]]
    diagnostics: dict[str, float | int | bool | str]
    failure_reason: str | None = None


def _failure(
    reason: str,
    ends: tuple[EndName, ...],
    diagnostics: dict[str, float | int | bool | str],
) -> EdgeCensoredResult:
    diagnostics = {**diagnostics, "failure_reason": reason}
    return EdgeCensoredResult(
        False, None, None, None, None, ends, None, {}, diagnostics, reason
    )


def _smooth_core(points: FloatArray, strength: float) -> FloatArray:
    """Penalized least-squares fit with a second-difference roughness term."""

    if strength == 0 or len(points) < 3:
        return points.copy()
    rows = len(points) - 2
    difference = np.zeros((rows, len(points)), dtype=np.float64)
    index = np.arange(rows)
    difference[index, index] = 1.0
    difference[index, index + 1] = -2.0
    difference[index, index + 2] = 1.0
    system = np.eye(len(points)) + strength * difference.T @ difference
    return np.linalg.solve(system, points)


def _endpoint_geometry(
    core: FloatArray, end: EndName, context_points: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return anchor, outward unit tangent, and d2(x,y)/ds2 at an end."""

    local = core[:context_points] if end == "start" else core[::-1][:context_points]
    segment = np.linalg.norm(np.diff(local, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    degree = min(2, len(local) - 1)
    coefficient_x = np.polyfit(arc, local[:, 0], degree)
    coefficient_y = np.polyfit(arc, local[:, 1], degree)
    derivative = np.asarray(
        (
            np.polyval(np.polyder(coefficient_x), 0.0),
            np.polyval(np.polyder(coefficient_y), 0.0),
        ),
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(derivative))
    if norm <= 1e-8:
        raise ValueError("core endpoint tangent is degenerate")
    tangent_out = -derivative / norm
    if degree == 2:
        second = np.asarray((2 * coefficient_x[0], 2 * coefficient_y[0]))
    else:
        second = np.zeros(2, dtype=np.float64)
    # Only the normal component bends the curve; tangential acceleration is a
    # parameterization artifact and destabilizes extrapolation.
    second = second - np.dot(second, tangent_out) * tangent_out
    return local[0].copy(), tangent_out, second


def _side_crossing(
    mask: BoolArray,
    contact: BoundaryContact,
    reference_xy: FloatArray,
) -> FloatArray | None:
    """Measure a border-interval midpoint, falling back to contact projection."""

    height, width = mask.shape
    candidates: list[FloatArray] = []
    for side in contact.sides:
        if side in ("top", "bottom"):
            y = 0 if side == "top" else height - 1
            positions = np.flatnonzero(mask[y])
            target_value = contact.center_xy[0]
            if len(positions):
                # Use the contiguous border run closest to this contact.
                breaks = np.flatnonzero(np.diff(positions) > 1) + 1
                runs = np.split(positions, breaks)
                run = min(runs, key=lambda value: np.min(np.abs(value - target_value)))
                x = float(0.5 * (run[0] + run[-1]))
            else:
                x = float(np.clip(contact.center_xy[0], 0, width - 1))
            candidates.append(np.asarray((x, float(y))))
        elif side in ("left", "right"):
            x = 0 if side == "left" else width - 1
            positions = np.flatnonzero(mask[:, x])
            target_value = contact.center_xy[1]
            if len(positions):
                breaks = np.flatnonzero(np.diff(positions) > 1) + 1
                runs = np.split(positions, breaks)
                run = min(runs, key=lambda value: np.min(np.abs(value - target_value)))
                y = float(0.5 * (run[0] + run[-1]))
            else:
                y = float(np.clip(contact.center_xy[1], 0, height - 1))
            candidates.append(np.asarray((float(x), y)))
    if not candidates:
        return None
    return min(candidates, key=lambda value: float(np.linalg.norm(value - reference_xy)))


def _inside(points: FloatArray, image_shape: tuple[int, int], tolerance: float = 1e-7) -> bool:
    height, width = image_shape
    return bool(
        np.all(points[:, 0] >= -tolerance)
        and np.all(points[:, 0] <= width - 1 + tolerance)
        and np.all(points[:, 1] >= -tolerance)
        and np.all(points[:, 1] <= height - 1 + tolerance)
    )


def _reconstruct(
    anchor: FloatArray,
    tangent: FloatArray,
    second: FloatArray,
    crossing: FloatArray,
    count: int,
    end: EndName,
    image_shape: tuple[int, int],
) -> tuple[FloatArray, float]:
    """Cubic extrapolation matching core tangent/curvature and the crossing."""

    length = max(float(np.linalg.norm(crossing - anchor)), 1e-6)
    sample_t = np.linspace(0.0, length, count + 1)
    # Camera corners and noisy contact centres can make full curvature leave the
    # rectangle. Damping curvature is preferable to clipping coordinates.
    for curvature_scale in (1.0, 0.5, 0.25, 0.0):
        acceleration = second * curvature_scale
        cubic = (
            crossing - anchor - tangent * length - 0.5 * acceleration * length**2
        ) / length**3
        values = (
            anchor[None]
            + sample_t[:, None] * tangent[None]
            + 0.5 * sample_t[:, None] ** 2 * acceleration[None]
            + sample_t[:, None] ** 3 * cubic[None]
        )
        values[-1] = crossing
        if _inside(values, image_shape):
            if end == "start":
                return values[::-1][:-1], curvature_scale
            return values[1:], curvature_scale
    raise ValueError("visible reconstruction leaves the camera rectangle")


def repair_edge_censored_centerline(
    centerline_xy: NDArray[np.generic],
    image_shape: tuple[int, int],
    truncation: BoundaryTruncationResult,
    *,
    raw_mask: NDArray[np.generic] | None = None,
    edge_band_px: int | None = None,
    smoothness: float = 2.0,
    context_points: int = 8,
    min_core_points: int = 8,
    min_core_length_px: float = 20.0,
) -> EdgeCensoredResult:
    """Repair camera-censored A3 geometry without inventing off-FOV anatomy.

    Contact-to-end association is recomputed geometrically.  This is important:
    the raw-mask skeleton used for classification may have the opposite ordering
    to the selected A3 path.
    """

    points = np.asarray(centerline_xy, dtype=np.float64)
    diagnostics: dict[str, float | int | bool | str] = {
        "input_points": int(len(points)) if points.ndim else 0,
        "boundary_state": truncation.state,
    }
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        raise ValueError("centerline_xy must have shape [N>=3,2]")
    if not np.isfinite(points).all():
        raise ValueError("centerline_xy must be finite")
    height, width = image_shape
    if height < 2 or width < 2:
        raise ValueError("image_shape dimensions must be >=2")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
        raise ValueError("consecutive centerline samples must be distinct")
    band = truncation.edge_band_px if edge_band_px is None else edge_band_px
    if band < 1:
        raise ValueError("edge_band_px must be >=1")
    if smoothness < 0 or not np.isfinite(smoothness):
        raise ValueError("smoothness must be finite and non-negative")
    if context_points < 3 or min_core_points < 3:
        raise ValueError("context_points and min_core_points must be >=3")
    if min_core_length_px <= 0 or not np.isfinite(min_core_length_px):
        raise ValueError("min_core_length_px must be finite and positive")
    component = np.asarray(
        truncation.component_mask if raw_mask is None else raw_mask, dtype=bool
    )
    if component.shape != image_shape:
        raise ValueError("raw_mask/component_mask shape must match image_shape")

    # Associate each contact with the nearest endpoint in this path orientation.
    assigned: dict[EndName, BoundaryContact] = {}
    for contact in truncation.contacts:
        # ``endpoint=None`` denotes a side/body contact that the classifier
        # could not safely interpret as missing terminal anatomy.
        if contact.endpoint is None or not contact.sides:
            continue
        center = np.asarray(contact.center_xy)
        distances = (float(np.linalg.norm(center - points[0])), float(np.linalg.norm(center - points[-1])))
        end: EndName = "start" if distances[0] <= distances[1] else "end"
        previous = assigned.get(end)
        if previous is None or min(distances) < np.linalg.norm(np.asarray(previous.center_xy) - (points[0] if end == "start" else points[-1])):
            assigned[end] = contact
    ends = tuple(end for end in ("start", "end") if end in assigned)
    diagnostics.update(edge_band_px=int(band), censored_end_count=len(ends))
    if not ends:
        return _failure("no_endpoint_boundary_contact", ends, diagnostics)

    distance = np.minimum.reduce(
        (points[:, 0], points[:, 1], width - 1 - points[:, 0], height - 1 - points[:, 1])
    )
    start_index = 0
    stop_index = len(points)
    if "start" in assigned:
        while start_index < stop_index and distance[start_index] < band:
            start_index += 1
    if "end" in assigned:
        while stop_index > start_index and distance[stop_index - 1] < band:
            stop_index -= 1
    core = points[start_index:stop_index]
    core_length = float(np.linalg.norm(np.diff(core, axis=0), axis=1).sum()) if len(core) >= 2 else 0.0
    diagnostics.update(
        core_start_index=int(start_index),
        core_stop_index=int(stop_index),
        reliable_core_points=int(len(core)),
        reliable_core_length_px=core_length,
    )
    if ("start" in assigned and start_index == 0) or ("end" in assigned and stop_index == len(points)):
        return _failure("contacted_endpoint_has_no_edge_band_samples", ends, diagnostics)
    if len(core) < min_core_points:
        return _failure("insufficient_core_points", ends, diagnostics)
    if core_length < min_core_length_px:
        return _failure("insufficient_core_length", ends, diagnostics)

    smooth = _smooth_core(core, smoothness)
    output_parts: list[FloatArray] = []
    reliable_parts: list[BoolArray] = []
    crossings: dict[EndName, tuple[float, float]] = {}
    curvature_scales: dict[EndName, float] = {}
    try:
        if "start" in assigned:
            crossing = _side_crossing(component, assigned["start"], points[0])
            if crossing is None:
                return _failure("start_contact_has_no_boundary_side", ends, diagnostics)
            anchor, tangent, second = _endpoint_geometry(smooth, "start", min(context_points, len(smooth)))
            rebuilt, scale = _reconstruct(anchor, tangent, second, crossing, start_index, "start", image_shape)
            output_parts.append(rebuilt)
            reliable_parts.append(np.zeros(len(rebuilt), dtype=bool))
            crossings["start"] = (float(crossing[0]), float(crossing[1]))
            curvature_scales["start"] = scale
            diagnostics["start_outward_tangent_angle_rad"] = float(
                np.arctan2(tangent[1], tangent[0])
            )
            diagnostics["start_core_curvature_per_px"] = float(np.linalg.norm(second))
        output_parts.append(smooth)
        reliable_parts.append(np.ones(len(smooth), dtype=bool))
        if "end" in assigned:
            crossing = _side_crossing(component, assigned["end"], points[-1])
            if crossing is None:
                return _failure("end_contact_has_no_boundary_side", ends, diagnostics)
            anchor, tangent, second = _endpoint_geometry(smooth, "end", min(context_points, len(smooth)))
            rebuilt, scale = _reconstruct(anchor, tangent, second, crossing, len(points) - stop_index, "end", image_shape)
            output_parts.append(rebuilt)
            reliable_parts.append(np.zeros(len(rebuilt), dtype=bool))
            crossings["end"] = (float(crossing[0]), float(crossing[1]))
            curvature_scales["end"] = scale
            diagnostics["end_outward_tangent_angle_rad"] = float(
                np.arctan2(tangent[1], tangent[0])
            )
            diagnostics["end_core_curvature_per_px"] = float(np.linalg.norm(second))
    except (ValueError, np.linalg.LinAlgError) as error:
        diagnostics["geometry_error"] = str(error)
        return _failure("edge_reconstruction_failed", ends, diagnostics)

    output = np.concatenate(output_parts)
    reliable = np.concatenate(reliable_parts)
    if len(output) != len(points) or not _inside(output, image_shape):
        return _failure("invalid_reconstructed_geometry", ends, diagnostics)
    # Remove only floating roundoff at exact rectangle boundaries.
    output[:, 0] = np.clip(output[:, 0], 0.0, width - 1.0)
    output[:, 1] = np.clip(output[:, 1], 0.0, height - 1.0)
    rounded = np.rint(output).astype(int)
    support = component[rounded[:, 1], rounded[:, 0]]
    censored = np.zeros(len(output), dtype=bool)
    if "start" in assigned:
        censored[0] = True
    if "end" in assigned:
        censored[-1] = True
    diagnostics.update(
        output_points=int(len(output)),
        reconstructed_points=int((~reliable).sum()),
        observed_support_fraction=float(support.mean()),
        start_curvature_scale=float(curvature_scales.get("start", -1.0)),
        end_curvature_scale=float(curvature_scales.get("end", -1.0)),
    )
    return EdgeCensoredResult(
        True,
        output,
        reliable,
        support,
        censored,
        ends,
        (start_index, stop_index),
        crossings,
        diagnostics,
    )
