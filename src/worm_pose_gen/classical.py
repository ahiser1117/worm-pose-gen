"""Conservative, dependency-free classical worm centerline extraction.

This module intentionally targets easy, fully visible frames.  Rejection is a
valid result: output from this code is a proxy label, never ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ClassicalConfig:
    local_radius: int = 31
    smooth_radius: int = 2
    foreground_z: float = 2.6
    # Optional hysteresis threshold. Pixels below ``foreground_z`` are admitted
    # only when they connect to the high-confidence largest component. This can
    # recover faint worm sections without retaining disconnected dark debris.
    # Disabled by default: the interactively tuned 61/3/4.25/2.05/8 setting cut
    # 30-frame stress acceptance from 11 to 3 frames (see
    # docs/final_algorithm_tuned_local_darkness_unannotated30) and was reverted.
    connected_foreground_z: float | None = None
    close_radius: int = 2
    min_area: int = 2_500
    max_area: int = 30_000
    min_length: float = 250.0
    max_length: float = 750.0
    # The audited width is roughly 14--25 px.  Requiring a centerline clearance
    # of 13 px conservatively rejects a tube that is truncated at an image edge
    # even when local normalization makes its detected ridge end slightly inboard.
    boundary_margin: float = 13.0
    # Geometric check only: the smoothed centerline must remain inside the
    # segmented body. Darkness is used to find the border, not to score the
    # centerline itself.
    min_tube_support: float = 0.95
    n_points: int = 100
    max_branch_pixels: int = 50
    max_raw_endpoints: int = 16


@dataclass(frozen=True)
class ClassicalResult:
    accepted: bool
    centerline_xy: FloatArray | None
    tangent_angle: FloatArray | None
    quality_score: float
    head_tail_probability: float
    rejection_reasons: tuple[str, ...]
    qc: dict[str, float | int | bool]
    mask: BoolArray | None = None


@dataclass(frozen=True)
class DarkRidgeSegmentation:
    """Inspectable output of the local-darkness segmentation stages."""

    score: FloatArray
    high_threshold_mask: BoolArray
    connected_threshold_mask: BoolArray | None
    closed_high_mask: BoolArray
    closed_connected_mask: BoolArray | None
    high_component: BoolArray
    component: BoolArray
    component_count: int
    recovered_area: int
    disconnected_connected_area: int


def box_blur(image: NDArray[np.generic], radius: int) -> FloatArray:
    """Edge-padded square box blur using only NumPy."""

    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return values.copy()
    padded = np.pad(values, radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    width = 2 * radius + 1
    total = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return total / float(width * width)


def robust_dark_ridge(image: NDArray[np.generic], config: ClassicalConfig) -> FloatArray:
    """Return robust local darkness scores; larger values are darker ridges."""

    image_f = np.asarray(image, dtype=np.float64)
    local_background = box_blur(image_f, config.local_radius)
    smoothed = box_blur(image_f, config.smooth_radius)
    residual = local_background - smoothed
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    scale = max(1.4826 * mad, 1.0)
    return (residual - median) / scale


def _dilate(mask: BoolArray, radius: int) -> BoolArray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant")
    out = np.zeros_like(mask)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _erode(mask: BoolArray, radius: int) -> BoolArray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant")
    out = np.ones_like(mask)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _largest_component(mask: BoolArray) -> tuple[BoolArray, int, int]:
    """Return largest 8-connected component, its area, and component count."""

    h, w = mask.shape
    visited = np.zeros_like(mask)
    largest: list[tuple[int, int]] = []
    count = 0
    for y, x in np.argwhere(mask):
        yi, xi = int(y), int(x)
        if visited[yi, xi]:
            continue
        count += 1
        queue = deque([(yi, xi)])
        visited[yi, xi] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            pixels.append((cy, cx))
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        if len(pixels) > len(largest):
            largest = pixels
    result = np.zeros_like(mask)
    if largest:
        yy, xx = np.asarray(largest).T
        result[yy, xx] = True
    return result, len(largest), count


def _connected_extension(seed: BoolArray, eligible: BoolArray) -> BoolArray:
    """Grow an 8-connected seed only through eligible pixels."""

    if seed.shape != eligible.shape or seed.ndim != 2:
        raise ValueError("seed and eligible masks must share a two-dimensional shape")
    result = seed.copy()
    queue = deque((int(y), int(x)) for y, x in np.argwhere(seed))
    height, width = result.shape
    while queue:
        y, x = queue.popleft()
        for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
            for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                if eligible[neighbor_y, neighbor_x] and not result[neighbor_y, neighbor_x]:
                    result[neighbor_y, neighbor_x] = True
                    queue.append((neighbor_y, neighbor_x))
    return result


def segment_dark_ridge(
    image: NDArray[np.generic], config: ClassicalConfig | None = None
) -> DarkRidgeSegmentation:
    """Run the local-darkness threshold, cleanup, and optional hysteresis.

    The high threshold remains the seed. When ``connected_foreground_z`` is
    enabled, lower-confidence pixels survive only if they are connected to the
    high-confidence largest component after the same morphological closing.
    """

    cfg = config or ClassicalConfig()
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("image must have shape [height, width]")
    if cfg.local_radius < 0 or cfg.smooth_radius < 0 or cfg.close_radius < 0:
        raise ValueError("segmentation radii must be non-negative")
    if not np.isfinite(cfg.foreground_z):
        raise ValueError("foreground_z must be finite")
    if cfg.connected_foreground_z is not None and (
        not np.isfinite(cfg.connected_foreground_z)
        or cfg.connected_foreground_z >= cfg.foreground_z
    ):
        raise ValueError("connected_foreground_z must be finite and below foreground_z")

    score = robust_dark_ridge(values, cfg)
    high_mask = score >= cfg.foreground_z
    high_closed = _erode(_dilate(high_mask, cfg.close_radius), cfg.close_radius)
    high_component, _, component_count = _largest_component(high_closed)

    connected_mask: BoolArray | None = None
    connected_closed: BoolArray | None = None
    component = high_component
    recovered_area = 0
    disconnected_connected_area = 0
    if cfg.connected_foreground_z is not None:
        connected_mask = score >= cfg.connected_foreground_z
        connected_closed = _erode(
            _dilate(connected_mask, cfg.close_radius), cfg.close_radius
        )
        component = _connected_extension(high_component, connected_closed)
        recovered_area = int(np.logical_and(component, ~high_component).sum())
        disconnected_connected_area = int(
            np.logical_and(connected_closed, ~component).sum()
        )

    return DarkRidgeSegmentation(
        score=score,
        high_threshold_mask=high_mask,
        connected_threshold_mask=connected_mask,
        closed_high_mask=high_closed,
        closed_connected_mask=connected_closed,
        high_component=high_component,
        component=component,
        component_count=component_count,
        recovered_area=recovered_area,
        disconnected_connected_area=disconnected_connected_area,
    )


def _border_mask(shape: tuple[int, int]) -> BoolArray:
    result = np.zeros(shape, dtype=bool)
    result[[0, -1], :] = True
    result[:, [0, -1]] = True
    return result


def _thin(mask: BoolArray, max_iterations: int = 256) -> BoolArray:
    """Zhang-Suen thinning on a tight crop."""

    image = mask.copy()
    for _ in range(max_iterations):
        changed = False
        for first in (True, False):
            p = np.pad(image, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            neighbors = sum(value.astype(np.uint8) for value in (p2, p3, p4, p5, p6, p7, p8, p9))
            transitions = (
                sum(value.astype(np.uint8) for value in (
                    ~p2 & p3, ~p3 & p4, ~p4 & p5, ~p5 & p6,
                    ~p6 & p7, ~p7 & p8, ~p8 & p9, ~p9 & p2,
                ))
            )
            common = image & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
            if first:
                remove = common & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                remove = common & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            if np.any(remove):
                image[remove] = False
                changed = True
        if not changed:
            break
    return image


def _skeleton_longest_path(skeleton: BoolArray) -> tuple[FloatArray | None, int, int]:
    pixels = [tuple(map(int, value)) for value in np.argwhere(skeleton)]
    if not pixels:
        return None, 0, 0
    pixel_set = set(pixels)

    def neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = node
        result: list[tuple[int, int]] = []
        for ny in range(y - 1, y + 2):
            for nx in range(x - 1, x + 2):
                if (ny, nx) == node or (ny, nx) not in pixel_set:
                    continue
                # Suppress redundant corner edges when an orthogonal bridge is
                # present; otherwise a one-pixel curve acquires false branches.
                if ny != y and nx != x and ((y, nx) in pixel_set or (ny, x) in pixel_set):
                    continue
                result.append((ny, nx))
        return result

    degrees = {node: len(neighbors(node)) for node in pixels}
    endpoints = [node for node, degree in degrees.items() if degree == 1]
    branch_pixels = sum(degree > 2 for degree in degrees.values())
    if len(endpoints) < 2:
        return None, len(endpoints), branch_pixels

    best_path: list[tuple[int, int]] = []
    for start in endpoints:
        queue = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        distance = {start: 0.0}
        while queue:
            node = queue.popleft()
            for nxt in neighbors(node):
                step = math.sqrt(2.0) if nxt[0] != node[0] and nxt[1] != node[1] else 1.0
                candidate = distance[node] + step
                if nxt not in distance or candidate < distance[nxt]:
                    distance[nxt] = candidate
                    parent[nxt] = node
                    queue.append(nxt)
        reachable = [node for node in endpoints if node in distance]
        if not reachable:
            continue
        end = max(reachable, key=distance.__getitem__)
        path: list[tuple[int, int]] = []
        node: tuple[int, int] | None = end
        while node is not None:
            path.append(node)
            node = parent[node]
        if len(path) > len(best_path):
            best_path = path[::-1]
    if not best_path:
        return None, len(endpoints), branch_pixels
    return np.asarray([(x, y) for y, x in best_path], dtype=np.float64), len(endpoints), branch_pixels


def _prune_skeleton_endpoints(skeleton: BoolArray, iterations: int = 8) -> BoolArray:
    """Peel short terminal spurs before topology assessment.

    The same small peel occurs at the two anatomical ends, so it cannot create
    false endpoint confidence and is negligible relative to a 250--750 px body.
    """

    result = skeleton.copy()
    kernel_offsets = tuple(
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
    )
    for _ in range(iterations):
        padded = np.pad(result, 1)
        degree = np.zeros_like(result, dtype=np.uint8)
        for dy, dx in kernel_offsets:
            degree += padded[1 + dy : 1 + dy + result.shape[0],
                             1 + dx : 1 + dx + result.shape[1]]
        endpoints = result & (degree <= 1)
        if not np.any(endpoints):
            break
        result[endpoints] = False
    return result


def resample_centerline(points_xy: NDArray[np.generic], n_points: int = 100) -> FloatArray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("points_xy must have shape [N>=2, 2]")
    delta = np.diff(points, axis=0)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(delta, axis=1))))
    keep = np.concatenate(([True], np.diff(arc) > 1e-9))
    points, arc = points[keep], arc[keep]
    if len(points) < 2 or arc[-1] <= 0:
        raise ValueError("centerline has zero length")
    target = np.linspace(0.0, arc[-1], n_points)
    result = np.column_stack(
        (np.interp(target, arc, points[:, 0]), np.interp(target, arc, points[:, 1]))
    )
    # A light intrinsic smoothing suppresses pixel-grid staircase artifacts.
    if n_points >= 5:
        smooth = result.copy()
        smooth[1:-1] = (result[:-2] + 2.0 * result[1:-1] + result[2:]) / 4.0
        result = resample_centerline(smooth, n_points) if n_points != len(points) else smooth
    return result


def tangent_angles(centerline_xy: NDArray[np.generic]) -> FloatArray:
    points = np.asarray(centerline_xy, dtype=np.float64)
    derivative = np.gradient(points, axis=0)
    return np.arctan2(derivative[:, 1], derivative[:, 0])


def extract_centerline(
    image: NDArray[np.generic], config: ClassicalConfig | None = None, *, keep_mask: bool = False
) -> ClassicalResult:
    """Extract a high-confidence proxy centerline or conservatively reject."""

    cfg = config or ClassicalConfig()
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("image must have shape [height, width]")
    segmentation = segment_dark_ridge(values, cfg)
    component = segmentation.component
    component_count = segmentation.component_count
    # Closing repairs small cross-sectional gaps.  Do not indiscriminately fill
    # holes: a tightly bent worm can enclose a large background region whose
    # fill would create a false shortcut through the anatomy.
    area = int(component.sum())
    reasons: list[str] = []
    if area < cfg.min_area or area > cfg.max_area:
        reasons.append("implausible_area")
    if not area:
        qc = {
            "area": 0,
            "component_count": component_count,
            "connected_threshold_enabled": cfg.connected_foreground_z is not None,
            "connected_threshold_recovered_area": segmentation.recovered_area,
            "connected_threshold_disconnected_area": (
                segmentation.disconnected_connected_area
            ),
        }
        return ClassicalResult(
            False,
            None,
            None,
            0.0,
            0.5,
            tuple(reasons or ["no_component"]),
            qc,
            component if keep_mask else None,
        )

    yy, xx = np.nonzero(component)
    pad = 3
    y0, y1 = max(0, int(yy.min()) - pad), min(values.shape[0], int(yy.max()) + pad + 1)
    x0, x1 = max(0, int(xx.min()) - pad), min(values.shape[1], int(xx.max()) + pad + 1)
    skeleton_crop = _prune_skeleton_endpoints(_thin(component[y0:y1, x0:x1]))
    path, endpoint_count, branch_pixels = _skeleton_longest_path(skeleton_crop)
    if path is None:
        reasons.append("unstable_endpoints")
        qc = {
            "area": area,
            "component_count": component_count,
            "endpoint_count": endpoint_count,
            "branch_pixels": branch_pixels,
            "connected_threshold_enabled": cfg.connected_foreground_z is not None,
            "connected_threshold_recovered_area": segmentation.recovered_area,
            "connected_threshold_disconnected_area": (
                segmentation.disconnected_connected_area
            ),
        }
        return ClassicalResult(False, None, None, 0.0, 0.5, tuple(dict.fromkeys(reasons)), qc,
                               component if keep_mask else None)
    path[:, 0] += x0
    path[:, 1] += y0
    centerline = resample_centerline(path, cfg.n_points)
    length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    h, w = values.shape
    boundary_distance = float(np.min(np.column_stack((centerline[:, 0], centerline[:, 1],
        (w - 1) - centerline[:, 0], (h - 1) - centerline[:, 1]))))
    component_boundary_contact = bool(np.any(component & _border_mask(component.shape)))
    sample_x = np.clip(np.rint(centerline[:, 0]).astype(int), 0, w - 1)
    sample_y = np.clip(np.rint(centerline[:, 1]).astype(int), 0, h - 1)
    tube_support = float(np.mean(_dilate(component, 1)[sample_y, sample_x]))
    if not cfg.min_length <= length <= cfg.max_length:
        reasons.append("implausible_length")
    if boundary_distance < cfg.boundary_margin or component_boundary_contact:
        reasons.append("boundary_contact")
    # The exported longest ordered backbone has exactly two endpoints.  Small
    # skeleton spurs are tolerated only within explicit raw-topology bounds.
    if endpoint_count < 2 or endpoint_count > cfg.max_raw_endpoints or branch_pixels > cfg.max_branch_pixels:
        reasons.append("unstable_endpoints")
    if tube_support < cfg.min_tube_support:
        reasons.append("low_tube_support")

    # Static appearance is weak evidence.  Use it only to choose export order,
    # and cap confidence far below a claim of anatomical certainty.
    endpoint_radius = max(3, int(round(math.sqrt(area / max(length, 1.0) / math.pi))))
    endpoint_stats: list[float] = []
    for x, y in (centerline[0], centerline[-1]):
        xi, yi = int(round(x)), int(round(y))
        patch = component[max(0, yi - endpoint_radius): yi + endpoint_radius + 1,
                          max(0, xi - endpoint_radius): xi + endpoint_radius + 1]
        endpoint_stats.append(float(patch.mean()) if patch.size else 0.0)
    appearance_delta = abs(endpoint_stats[0] - endpoint_stats[1])
    head_tail_probability = float(0.5 + min(0.15, 0.5 * appearance_delta))
    if endpoint_stats[0] < endpoint_stats[1]:
        centerline = centerline[::-1].copy()

    length_score = max(0.0, 1.0 - abs(length - 450.0) / 300.0)
    topology_score = max(0.0, 1.0 - branch_pixels / max(cfg.max_branch_pixels, 1))
    boundary_score = min(1.0, boundary_distance / 20.0)
    quality = float(np.clip(0.4 * tube_support + 0.2 * length_score + 0.2 * topology_score + 0.2 * boundary_score, 0, 1))
    qc = {
        "area": area,
        "component_count": component_count,
        "connected_threshold_enabled": cfg.connected_foreground_z is not None,
        "connected_threshold_recovered_area": segmentation.recovered_area,
        "connected_threshold_disconnected_area": segmentation.disconnected_connected_area,
        "length_px": length,
        "boundary_distance_px": boundary_distance,
        "component_boundary_contact": component_boundary_contact,
        "backbone_endpoint_count": 2,
        "raw_skeleton_endpoint_count": endpoint_count,
        "branch_pixels": branch_pixels,
        "tube_support_fraction": tube_support,
        "endpoint_appearance_delta": appearance_delta,
    }
    accepted = not reasons
    return ClassicalResult(accepted, centerline, tangent_angles(centerline), quality,
                           head_tail_probability, tuple(dict.fromkeys(reasons)), qc,
                           component if keep_mask else None)
