"""Fit the intrinsic tube model directly to a binary worm mask.

Instead of thinning a mask into a skeleton and requiring that skeleton to have
simple topology, this module renders the low-dimensional body model (16 cubic
tangent-angle coefficients, global rotation, length, centroid, and a width
scale over a fixed width template) as a soft tube and moves it by gradient
descent until the rendered mask agrees with the observed one.  Several
initializations are optimized as one batch and the best final overlap wins.

The width template is symmetric; a smooth log-space correction (a few cubic
B-spline coefficients over body position, pulled toward zero by a Gaussian
prior) lets the two ends taper differently.  That asymmetry is what tells the
tail from the head: ``orient_tail_last`` reverses a fit whose thinner end came
first.  A reversed start is an exact mirror image of the forward one under a
zero-centered prior, so orientation is decided after the fit rather than by
doubling the starts.

Rendering happens on a padded crop around the observed mask, coarse to fine.
Pixels outside the camera are never instantiated, so anatomy that leaves the
field of view is censored rather than penalized as missing foreground.  The
observed mask is the only evidence; nothing here uses manual annotations.

Coordinates are ``(x, y)`` pixel centers of the full source image.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .anchors import estimate_width_along_normals
from .classical import (
    _prune_skeleton_endpoints,
    _skeleton_longest_path,
    _thin,
    resample_centerline,
)
from .latent import cubic_bspline_basis, decode_centerline, decode_centerline_torch, encode_centerline
from .observation import soft_dice_energy


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MaskFitConfig:
    coefficients: int = 16
    n_points: int = 100
    edge_softness: float = 0.8
    crop_padding: int = 64
    # Coarse-to-fine schedule: block-averaged target at each downsample factor.
    stage_downsample: tuple[int, ...] = (4, 2, 1, 1)
    stage_steps: tuple[int, ...] = (150, 100, 150, 150)
    # Adam moves roughly one learning rate per step regardless of gradient
    # size, so near an optimum it random-walks at the step scale.  The rates
    # are scaled down per stage and decayed within a stage, and the best
    # full-resolution iterate is kept rather than the last one.
    stage_lr_scale: tuple[float, ...] = (1.0, 0.5, 0.25, 0.05)
    within_stage_decay: float = 0.1
    translation_lr: float = 1.0
    rotation_lr: float = 0.01
    log_length_lr: float = 0.005
    shape_lr: float = 0.02
    log_width_lr: float = 0.01
    # Asymmetric width: the template is multiplied by exp of a mean-centered
    # cubic B-spline over body position (0 disables the correction).  The
    # Gaussian prior toward zero decides the profile where the mask does not.
    width_coefficients: int = 6
    width_shape_lr: float = 0.01
    width_shape_prior: float = 1e-3
    # Center of the width correction (tail last); ``None`` means zero, the
    # symmetric template.  A recording prior (``recording_prior.py``) sets it.
    width_shape_prior_mean: tuple[float, ...] | None = None
    # Hard bounds, ``None`` to remove them.  Recording priors replace them.
    length_bounds_px: tuple[float, float] | None = (250.0, 750.0)
    width_bounds_px: tuple[float, float] | None = (15.0, 90.0)
    # Gaussian priors on log body length and log width scale, centered on a
    # recording's values; ``None`` disables them.  ``prior_weight`` converts a
    # squared deviation in sigmas into dice units: 0.0025 makes a two-sigma
    # deviation cost 0.01, about the energy gap between a good and a poor fit.
    length_prior_px: float | None = None
    length_prior_log_sigma: float = 0.05
    width_prior_px: float | None = None
    width_prior_log_sigma: float = 0.05
    prior_weight: float = 0.0025
    # Zero by default: under Adam even a tiny consistent regularizer gradient
    # becomes a full-size step once the data gradient vanishes.
    shape_smoothness: float = 0.0
    bound_weight: float = 1e-4
    crop_escape_weight: float = 1e-3
    default_length_px: float = 600.0
    default_width_px: float = 45.0
    # Constant-curvature arcs (rad/px) used by the moment-based starts.
    moment_arc_curvatures: tuple[float, ...] = (0.0, 0.003, -0.003)
    hard_threshold: float = 0.5


@dataclass(frozen=True)
class Initialization:
    """One starting state: a 20-value latent plus a full-body width in px."""

    name: str
    latent: FloatArray
    width_px: float
    # Log-space width correction coefficients; ``None`` starts symmetric.
    width_shape: FloatArray | None = None


@dataclass(frozen=True)
class CropWindow:
    x0: int
    x1: int
    y0: int
    y1: int
    image_height: int
    image_width: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


@dataclass
class MaskFitResult:
    best_index: int
    initializations: list[Initialization]
    records: list[dict[str, float | str | int]]
    latent: FloatArray
    width_px: float
    width_profile: FloatArray
    centerline_xy: FloatArray
    crop: CropWindow
    rendered_hard_mask: BoolArray
    energy_history: FloatArray
    points_in_fov: int
    body_length_px: float
    extra_iou: dict[str, float] = field(default_factory=dict)
    width_shape: FloatArray = field(default_factory=lambda: np.zeros(0))


def default_width_template(n_points: int = 100, cap_fraction: float = 0.12) -> FloatArray:
    """Unit-peak full-width template: flat midbody with elliptical end caps."""

    if n_points < 2 or not 0 < cap_fraction < 0.5:
        raise ValueError("n_points >= 2 and 0 < cap_fraction < 0.5 are required")
    position = np.linspace(0.0, 1.0, n_points)
    distance_to_end = np.minimum(position, 1.0 - position)
    inside_cap = distance_to_end < cap_fraction
    template = np.ones(n_points, dtype=np.float64)
    ratio = (cap_fraction - distance_to_end[inside_cap]) / cap_fraction
    template[inside_cap] = np.sqrt(np.clip(1.0 - ratio**2, 0.0, 1.0))
    return np.maximum(template, 0.02)


def measure_width_template(
    masks: Sequence[NDArray[np.generic]],
    centerlines: Sequence[NDArray[np.generic]],
    *,
    n_points: int = 100,
    smoothing_passes: int = 3,
) -> tuple[FloatArray, FloatArray]:
    """Estimate a unit-peak width template from masks with known centerlines.

    Each measured profile is normalized by its own midbody median so recordings
    with different magnification contribute shape only.  Returns the template
    and the midbody widths used for normalization.
    """

    if len(masks) != len(centerlines) or not masks:
        raise ValueError("masks and centerlines must be non-empty and equal length")
    profiles = []
    scales = []
    central = slice(n_points // 5, n_points - n_points // 5)
    for mask, curve in zip(masks, centerlines, strict=True):
        points = resample_centerline(np.asarray(curve, dtype=np.float64), n_points)
        profile = estimate_width_along_normals(np.asarray(mask, dtype=bool), points)
        scale = float(np.median(profile[central]))
        if not np.isfinite(scale) or scale <= 1.0:
            continue
        profiles.append(profile / scale)
        scales.append(scale)
    if not profiles:
        raise ValueError("no measurable width profiles")
    stacked = np.asarray(profiles)
    # Orientation is arbitrary, so symmetrize before taking the median.
    stacked = np.concatenate((stacked, stacked[:, ::-1]), axis=0)
    template = np.median(stacked, axis=0)
    for _ in range(smoothing_passes):
        template[1:-1] = (template[:-2] + 2.0 * template[1:-1] + template[2:]) / 4.0
    template = template / max(float(np.median(template[central])), 1e-6)
    return np.maximum(template, 0.02), np.asarray(scales, dtype=np.float64)


def _measured_width_px(mask: BoolArray, centerline_xy: FloatArray, default: float) -> float:
    n = len(centerline_xy)
    central = slice(n // 5, n - n // 5)
    try:
        profile = estimate_width_along_normals(mask, centerline_xy)
    except ValueError:
        return default
    value = float(np.median(profile[central]))
    return value if np.isfinite(value) and value > 3.0 else default


def init_from_centerline(
    centerline_xy: NDArray[np.generic],
    mask: NDArray[np.generic],
    *,
    name: str = "centerline",
    config: MaskFitConfig = MaskFitConfig(),
) -> Initialization:
    """Encode an existing centerline (any point count) as a starting state."""

    points = resample_centerline(np.asarray(centerline_xy, dtype=np.float64), config.n_points)
    latent = encode_centerline(points, config.coefficients)
    width = _measured_width_px(np.asarray(mask, dtype=bool), points, config.default_width_px)
    return Initialization(name, latent, width)


def init_from_skeleton(
    mask: NDArray[np.generic],
    *,
    minimum_path_px: float = 60.0,
    config: MaskFitConfig = MaskFitConfig(),
) -> Initialization | None:
    """Start from the longest skeleton path even when the skeleton branches."""

    binary = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(binary)
    if not len(yy):
        return None
    pad = 3
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(binary.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(binary.shape[1], int(xx.max()) + pad + 1)
    skeleton = _prune_skeleton_endpoints(_thin(binary[y0:y1, x0:x1]))
    path, _, _ = _skeleton_longest_path(skeleton)
    if path is None or len(path) < 2:
        return None
    path = path + np.array([x0, y0], dtype=np.float64)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    if length < minimum_path_px:
        return None
    return init_from_centerline(path, binary, name="skeleton_longest_path", config=config)


def init_from_moments(
    mask: NDArray[np.generic],
    *,
    curvature: float = 0.0,
    length_px: float | None = None,
    config: MaskFitConfig = MaskFitConfig(),
) -> Initialization | None:
    """Start from the mask centroid and principal axis as a constant-curvature arc."""

    binary = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(binary)
    if len(yy) < 3:
        return None
    centroid = np.array([xx.mean(), yy.mean()], dtype=np.float64)
    covariance = np.cov(np.stack((xx - centroid[0], yy - centroid[1])))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    rotation = float(math.atan2(axis[1], axis[0]))
    length = float(config.default_length_px if length_px is None else length_px)
    position = np.linspace(0.0, 1.0, config.n_points - 1)
    angle = curvature * length * (position - 0.5)
    basis = cubic_bspline_basis(config.n_points - 1, config.coefficients)
    shape = np.linalg.lstsq(basis, angle, rcond=None)[0]
    latent = np.concatenate((shape, [rotation, length], centroid))
    width = float(config.default_width_px)
    # The mask area gives a crude width for a body of the prior length.
    area_width = float(binary.sum()) / max(length, 1.0)
    wlow, whigh = config.width_bounds_px or (3.0, float("inf"))
    if np.isfinite(area_width) and wlow < area_width < whigh:
        width = area_width
    name = "moments_straight" if curvature == 0 else f"moments_arc_{curvature:+.4f}"
    return Initialization(name, latent, width)


def standard_initializations(
    mask: NDArray[np.generic],
    *,
    reference_centerline_xy: NDArray[np.generic] | None = None,
    config: MaskFitConfig = MaskFitConfig(),
) -> list[Initialization]:
    """Reference curve (if any), longest skeleton path, and moment-based arcs."""

    starts: list[Initialization] = []
    if reference_centerline_xy is not None:
        starts.append(
            init_from_centerline(reference_centerline_xy, mask, name="reference", config=config)
        )
    skeleton = init_from_skeleton(mask, config=config)
    if skeleton is not None:
        starts.append(skeleton)
    for curvature in config.moment_arc_curvatures:
        start = init_from_moments(mask, curvature=curvature, config=config)
        if start is not None:
            starts.append(start)
    return starts


def reverse_initialization(
    start: Initialization, *, config: MaskFitConfig = MaskFitConfig()
) -> Initialization:
    """The same starting body traversed from the other end.

    The clamped uniform B-spline bases are mirror-symmetric, so the width
    correction reverses coefficient-wise; the centerline is re-encoded.
    """

    curve = decode_centerline(start.latent, config.coefficients)[::-1]
    latent = encode_centerline(curve, config.coefficients)
    shape = None if start.width_shape is None else np.asarray(start.width_shape, dtype=np.float64)[::-1].copy()
    return Initialization(f"{start.name}_reversed", latent, start.width_px, shape)


def extend_start_to_length(
    start: Initialization,
    mask: NDArray[np.generic],
    target_length_px: float,
    *,
    border_margin: float = 80.0,
    config: MaskFitConfig = MaskFitConfig(),
) -> Initialization:
    """Lengthen a start to the prior length, straight out past the camera edge.

    A start built from the mask is as long as the visible body.  Mask pixels
    on the image border mark where the body leaves the camera; an end of the
    start within ``border_margin`` pixels of such pixels is cut off, so the
    missing length is added there along the end tangent (split between both
    ends if both are cut).  Skeleton ends stop well inside a body that exits
    through a wide cross-section, hence the generous margin.  A start whose
    ends are both inside the image is returned unchanged: its length is the
    mask's, and the fit decides.
    """

    binary = np.asarray(mask, dtype=bool)
    curve = decode_centerline(start.latent, config.coefficients)
    height, width = binary.shape[:2]
    length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
    missing = float(target_length_px) - length
    if missing <= 1.0:
        return start
    edge = np.zeros_like(binary)
    edge[:2] = edge[-2:] = True
    edge[:, :2] = edge[:, -2:] = True
    yy, xx = np.nonzero(binary & edge)
    if not len(yy):
        return start
    border_xy = np.column_stack((xx, yy)).astype(np.float64)

    def at_border(point: NDArray[np.float64]) -> bool:
        return bool(np.linalg.norm(border_xy - point, axis=1).min() <= border_margin)

    ends = [at_border(curve[0]), at_border(curve[-1])]
    if all(ends):
        # Both ends near border pixels: give the length to the nearer one
        # unless the border pixels split into two far-apart groups.
        d0 = np.linalg.norm(border_xy - curve[0], axis=1)
        d1 = np.linalg.norm(border_xy - curve[-1], axis=1)
        near0, near1 = d0 <= border_margin, d1 <= border_margin
        if not (near0 & ~near1).any() or not (near1 & ~near0).any():
            ends = [bool(d0.min() <= d1.min()), bool(d1.min() < d0.min())]
    if not any(ends):
        return start
    share = missing / sum(ends)

    def continuation(end: NDArray[np.float64], inner: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        # Head for the border pixels nearest this end (where the body leaves
        # the camera), then keep going straight; a skeleton's last segment
        # inside a wide exit blob is not a reliable direction.
        distance = np.linalg.norm(border_xy - end, axis=1)
        exit_xy = border_xy[distance <= distance.min() + 0.5 * border_margin].mean(0)
        to_exit = exit_xy - end
        reach = float(np.linalg.norm(to_exit))
        if reach < 2.0:
            direction = end - inner
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            return [(end + direction * share)[None, :]]
        direction = to_exit / reach
        if reach >= share:
            return [(end + direction * share)[None, :]]
        return [exit_xy[None, :], (exit_xy + direction * (share - reach))[None, :]]

    pieces = [curve]
    if ends[0]:
        pieces = continuation(curve[0], curve[1])[::-1] + pieces
    if ends[1]:
        pieces = pieces + continuation(curve[-1], curve[-2])
    extended = resample_centerline(np.vstack(pieces), config.n_points)
    return replace(start, latent=encode_centerline(extended, config.coefficients))


def redirect_start_through_exit(
    start: Initialization,
    mask: NDArray[np.generic],
    *,
    config: MaskFitConfig = MaskFitConfig(),
) -> Initialization | None:
    """Send the end of a start off camera through the point where the mask meets the border.

    For a start whose every point lies inside the image while the mask
    reaches the border, the body continues off camera but the start does
    not; the fitter then folds the tube back from the edge instead of
    leaving.  The start is cut at its point nearest the border contact and
    that end is replaced by a straight run through the contact and beyond,
    keeping the total length.  Returns ``None`` when the mask does not reach
    the border or the start already leaves the image.
    """

    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape[:2]
    edge = np.zeros_like(binary)
    edge[:2] = edge[-2:] = True
    edge[:, :2] = edge[:, -2:] = True
    yy, xx = np.nonzero(binary & edge)
    if not len(yy):
        return None
    curve = decode_centerline(start.latent, config.coefficients)
    inside = (curve[:, 0] >= 0) & (curve[:, 0] < width) & (curve[:, 1] >= 0) & (curve[:, 1] < height)
    if not inside.all():
        return None
    border_xy = np.column_stack((xx, yy)).astype(np.float64)
    exit_xy = border_xy.mean(0)
    distance = np.linalg.norm(curve - exit_xy, axis=1)
    cut = int(np.argmin(distance))
    n = len(curve)
    if cut < n // 2:
        curve = curve[::-1]
        cut = n - 1 - cut
    kept = curve[: cut + 1]
    total = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
    kept_length = float(np.linalg.norm(np.diff(kept, axis=0), axis=1).sum())
    direction = exit_xy - kept[-1]
    reach = float(np.linalg.norm(direction))
    if reach < 1.0:
        direction = kept[-1] - kept[-2]
        reach = 0.0
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    remaining = max(total - kept_length, 1.0)
    pieces = [kept]
    if reach >= remaining:
        # Not enough length left to reach the border: head straight for it.
        pieces.append((kept[-1] + direction * remaining)[None, :])
    else:
        pieces.append(exit_xy[None, :])
        pieces.append((exit_xy + direction * (remaining - reach))[None, :])
    extended = resample_centerline(np.vstack(pieces), config.n_points)
    return replace(start, name=f"{start.name}_exit", latent=encode_centerline(extended, config.coefficients))


def orientation_pair(start: Initialization, *, config: MaskFitConfig = MaskFitConfig()) -> list[Initialization]:
    """The start and its reversal with the *same* width correction.

    Under an asymmetric width prior the model's tail is always at the end of
    the body, so both orientations begin at the prior's profile and the
    energy decides which physical end is the tail.
    """

    reversed_start = reverse_initialization(start, config=config)
    return [start, replace(reversed_start, width_shape=start.width_shape)]


def taper_asymmetry(width_profile: NDArray[np.generic], fraction: float = 0.3) -> float:
    """Mean log width over the last ``fraction`` of the body minus the first.

    Negative means the last end is the thinner one.  The tail of *C. elegans*
    tapers over a longer stretch than the head, so the sign labels the ends.
    """

    profile = np.asarray(width_profile, dtype=np.float64)
    if profile.ndim != 1 or len(profile) < 2 or not 0 < fraction <= 0.5:
        raise ValueError("width_profile must be 1-D and 0 < fraction <= 0.5")
    n = max(1, int(round(fraction * len(profile))))
    log_profile = np.log(np.clip(profile, 1e-6, None))
    return float(log_profile[-n:].mean() - log_profile[:n].mean())


def reverse_result(result: MaskFitResult, *, config: MaskFitConfig = MaskFitConfig()) -> MaskFitResult:
    """The same fit with the body traversed from the other end."""

    curve = result.centerline_xy[::-1].copy()
    return replace(
        result,
        latent=encode_centerline(curve, config.coefficients),
        centerline_xy=curve,
        width_profile=result.width_profile[::-1].copy(),
        width_shape=result.width_shape[::-1].copy(),
    )


def orient_tail_last(
    result: MaskFitResult,
    *,
    config: MaskFitConfig = MaskFitConfig(),
    fraction: float = 0.3,
    minimum_asymmetry: float = 1e-6,
) -> tuple[MaskFitResult, bool]:
    """Reverse the fit when its first end is the thinner one; returns whether it was reversed.

    A symmetric profile (correction disabled) is left as fitted: its asymmetry
    is float noise, below ``minimum_asymmetry``.
    """

    if taper_asymmetry(result.width_profile, fraction) > minimum_asymmetry:
        return reverse_result(result, config=config), True
    return result, False


def fill_narrow_holes(
    mask: NDArray[np.generic],
    radius: int,
    *,
    device: torch.device | str | None = None,
    max_iterations: int = 4096,
) -> tuple[BoolArray, int]:
    """Fill enclosed background that cannot contain a ``(2r+1)`` square.

    Segmentation texture holes inside the body are narrow; the interior of a
    coiled worm is not.  Background connected to the image border is never
    filled.  Returns the filled mask and the number of pixels added.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0 or not binary.any():
        return binary.copy(), 0
    resolved = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    yy, xx = np.nonzero(binary)
    height, width = binary.shape
    y0, y1 = max(0, int(yy.min()) - 1), min(height, int(yy.max()) + 2)
    x0, x1 = max(0, int(xx.min()) - 1), min(width, int(xx.max()) + 2)
    local = torch.as_tensor(binary[y0:y1, x0:x1], device=resolved)
    background = ~local
    reached = torch.zeros_like(background)
    reached[0, :] = background[0, :]
    reached[-1, :] = background[-1, :]
    reached[:, 0] = background[:, 0]
    reached[:, -1] = background[:, -1]

    def dilate(values: Tensor, size: int) -> Tensor:
        pooled = F.max_pool2d(values.to(torch.float32)[None, None], size, stride=1, padding=size // 2)
        return pooled[0, 0] > 0

    for _ in range(max_iterations):
        grown = dilate(reached, 3) & background
        if bool(torch.equal(grown, reached)):
            break
        reached = grown
    enclosed = background & ~reached
    if not bool(enclosed.any()):
        return binary.copy(), 0
    size = 2 * radius + 1
    eroded = ~dilate(~enclosed, size)
    survivors = dilate(eroded, size) & enclosed
    addition = enclosed & ~survivors
    filled = binary.copy()
    filled[y0:y1, x0:x1] |= addition.cpu().numpy()
    return filled, int(addition.sum())


def crop_window(mask: NDArray[np.generic], padding: int, multiple: int) -> CropWindow:
    """Padded bounding box whose size is a multiple of the coarsest factor."""

    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    yy, xx = np.nonzero(binary)
    if not len(yy):
        raise ValueError("mask is empty")
    x0 = max(0, int(xx.min()) - padding)
    y0 = max(0, int(yy.min()) - padding)
    x1 = min(width, int(xx.max()) + padding + 1)
    y1 = min(height, int(yy.max()) + padding + 1)
    # Grow toward the origin if the far side is clipped, then trim to a multiple.
    span_x = x1 - x0
    span_y = y1 - y0
    span_x -= span_x % multiple
    span_y -= span_y % multiple
    if span_x < multiple or span_y < multiple:
        raise ValueError("mask crop is smaller than the coarsest downsample factor")
    return CropWindow(x0, x0 + span_x, y0, y0 + span_y, height, width)


def signed_edge_distance(mask: Tensor, *, chunk_pixels: int = 4096) -> Tensor:
    """Signed distance to the edge halfway between inside and outside pixels.

    Positive inside.  A pixel adjacent to the edge reads ``0.5``, so a soft
    target ``sigmoid(distance / softness)`` crosses one half exactly where the
    renderer's tube boundary does.  Distances are measured to the opposite
    side's boundary pixels, which keeps the computation small.
    """

    values = mask.to(dtype=torch.bool)
    if values.ndim != 2:
        raise ValueError("mask must have shape [H,W]")
    if not bool(values.any()) or bool(values.all()):
        raise ValueError("mask must contain foreground and background")
    floating = values.to(dtype=torch.float32)
    dilated = F.max_pool2d(floating[None, None], 3, stride=1, padding=1)[0, 0] > 0
    eroded = -F.max_pool2d(-floating[None, None], 3, stride=1, padding=1)[0, 0] > 0
    outer_boundary = dilated & ~values
    inner_boundary = values & ~eroded
    height, width = values.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=values.device),
        torch.arange(width, device=values.device),
        indexing="ij",
    )
    pixels = torch.stack((yy, xx), -1).reshape(-1, 2).to(dtype=torch.float32)
    result = torch.empty(height * width, dtype=torch.float32, device=values.device)
    flat_inside = values.reshape(-1)
    for side, targets in ((True, outer_boundary), (False, inner_boundary)):
        target_yx = torch.nonzero(targets, as_tuple=False).to(dtype=torch.float32)
        rows = torch.nonzero(flat_inside == side, as_tuple=False).squeeze(1)
        for start in range(0, len(rows), chunk_pixels):
            index = rows[start : start + chunk_pixels]
            distance = torch.cdist(pixels[index], target_yx).min(1).values - 0.5
            result[index] = distance if side else -distance
    return result.reshape(height, width)


def _downsample(values: Tensor, factor: int) -> Tensor:
    if factor == 1:
        return values
    height, width = values.shape[-2:]
    return values.reshape(height // factor, factor, width // factor, factor).mean((1, 3))


class _MaskFitState(nn.Module):
    def __init__(self, starts: Sequence[Initialization], config: MaskFitConfig, device: torch.device):
        super().__init__()
        latents = torch.as_tensor(np.stack([s.latent for s in starts]), dtype=torch.float32, device=device)
        widths = torch.as_tensor([s.width_px for s in starts], dtype=torch.float32, device=device)
        k = config.coefficients
        self.shape = nn.Parameter(latents[:, :k].clone())
        self.rotation = nn.Parameter(latents[:, k].clone())
        self.log_length = nn.Parameter(latents[:, k + 1].clamp_min(1.0).log())
        self.centroid = nn.Parameter(latents[:, k + 2 :].clone())
        self.log_width = nn.Parameter(widths.clamp_min(1.0).log())
        kw = config.width_coefficients
        if kw < 0 or 0 < kw < 4:
            raise ValueError("width_coefficients must be 0 or at least 4")
        shapes = np.zeros((len(starts), kw), dtype=np.float64)
        for index, start in enumerate(starts):
            if start.width_shape is not None:
                values = np.asarray(start.width_shape, dtype=np.float64)
                if values.shape != (kw,):
                    raise ValueError(f"width_shape of start {index} must have shape [{kw}]")
                shapes[index] = values
        self.width_shape = nn.Parameter(torch.as_tensor(shapes, dtype=torch.float32, device=device))
        self.register_buffer(
            "basis",
            torch.as_tensor(cubic_bspline_basis(config.n_points - 1, k), dtype=torch.float32, device=device),
        )
        width_basis = cubic_bspline_basis(config.n_points, kw) if kw else np.zeros((config.n_points, 0))
        self.register_buffer("width_basis", torch.as_tensor(width_basis, dtype=torch.float32, device=device))
        mean = np.zeros(kw, dtype=np.float64)
        if config.width_shape_prior_mean is not None:
            mean = np.asarray(config.width_shape_prior_mean, dtype=np.float64)
            if mean.shape != (kw,):
                raise ValueError(f"width_shape_prior_mean must have shape [{kw}]")
        self.register_buffer("width_shape_mean", torch.as_tensor(mean, dtype=torch.float32, device=device))
        self.config = config

    def parameter_snapshot(self) -> list[Tensor]:
        return [parameter.detach().clone() for parameter in self.parameters()]

    def restore_rows(self, snapshot: Sequence[Tensor], rows: Tensor) -> None:
        with torch.no_grad():
            for parameter, saved in zip(self.parameters(), snapshot, strict=True):
                parameter[rows] = saved[rows]

    def latent(self) -> Tensor:
        return torch.cat(
            (self.shape, self.rotation[:, None], self.log_length.exp()[:, None], self.centroid), dim=1
        )

    def centerline(self) -> Tensor:
        return decode_centerline_torch(self.latent(), self.basis, self.config.coefficients)

    def log_width_correction(self) -> Tensor:
        """Mean-centered log correction per body position, ``[B, n_points]``."""

        if self.width_shape.shape[1] == 0:
            return torch.zeros(
                (len(self.width_shape), self.config.n_points), dtype=torch.float32, device=self.width_shape.device
            )
        correction = self.width_shape @ self.width_basis.T
        return correction - correction.mean(1, keepdim=True)

    def diameter(self, template: Tensor) -> Tensor:
        """Full width along the body: scale times template times the correction."""

        return (self.log_width[:, None] + self.log_width_correction()).exp() * template[None, :]

    def optimizer(self) -> torch.optim.Optimizer:
        c = self.config
        groups = [
            {"params": [self.centroid], "lr": c.translation_lr},
            {"params": [self.rotation], "lr": c.rotation_lr},
            {"params": [self.log_length], "lr": c.log_length_lr},
            {"params": [self.shape], "lr": c.shape_lr},
            {"params": [self.log_width], "lr": c.log_width_lr},
        ]
        if self.width_shape.shape[1]:
            groups.append({"params": [self.width_shape], "lr": c.width_shape_lr})
        return torch.optim.Adam(groups)

    def width_prior(self) -> Tensor:
        """Gaussian pull of the width correction toward its prior mean."""

        if self.width_shape.shape[1] == 0:
            return torch.zeros(len(self.width_shape), dtype=torch.float32, device=self.width_shape.device)
        return self.config.width_shape_prior * (self.width_shape - self.width_shape_mean[None, :]).square().sum(1)

    def size_regularization(self) -> Tensor:
        """Hard bounds (when set) and Gaussian priors (when set) on length and width scale."""

        c = self.config
        length = self.log_length.exp()
        width = self.log_width.exp()
        total = torch.zeros_like(length)
        if c.length_bounds_px is not None:
            low, high = c.length_bounds_px
            total = total + c.bound_weight * ((low - length).clamp_min(0).square() + (length - high).clamp_min(0).square())
        if c.width_bounds_px is not None:
            wlow, whigh = c.width_bounds_px
            total = total + c.bound_weight * ((wlow - width).clamp_min(0).square() + (width - whigh).clamp_min(0).square())
        if c.length_prior_px is not None:
            deviation = (self.log_length - math.log(c.length_prior_px)) / c.length_prior_log_sigma
            total = total + c.prior_weight * deviation.square()
        if c.width_prior_px is not None:
            deviation = (self.log_width - math.log(c.width_prior_px)) / c.width_prior_log_sigma
            total = total + c.prior_weight * deviation.square()
        return total

    def regularization(self, centerline: Tensor, crop: CropWindow) -> Tensor:
        c = self.config
        smooth = c.shape_smoothness * (self.shape[:, 1:] - self.shape[:, :-1]).square().mean(1)
        length = self.log_length.exp()
        x, y = centerline.unbind(-1)
        escape = torch.zeros_like(x)
        # Only crop edges strictly inside the image constrain the body; a crop
        # edge that coincides with the camera edge is a censoring boundary,
        # and a point past the camera edge is censored altogether.
        if crop.x0 > 0:
            escape = escape + (crop.x0 - x).clamp_min(0).square()
        if crop.x1 < crop.image_width:
            escape = escape + (x - (crop.x1 - 1)).clamp_min(0).square()
        if crop.y0 > 0:
            escape = escape + (crop.y0 - y).clamp_min(0).square()
        if crop.y1 < crop.image_height:
            escape = escape + (y - (crop.y1 - 1)).clamp_min(0).square()
        inside_camera = ((x >= 0) & (x < crop.image_width) & (y >= 0) & (y < crop.image_height)).to(x.dtype)
        escape = (escape * inside_camera).mean(1)
        return smooth + self.size_regularization() + c.crop_escape_weight * escape + self.width_prior()


def render_tube_segments(
    centerline_xy: Tensor,
    diameter: Tensor,
    image_height: int,
    image_width: int,
    *,
    edge_softness: float = 0.8,
) -> Tensor:
    """Soft tube occupancy from distance to the centerline *polyline*.

    ``worm_pose_gen.renderer.render_worm`` measures distance to the nearest
    centerline sample.  That makes occupancy between samples depend on sample
    spacing, so lengthening the body shrinks the rendered tube everywhere and
    the length gradient carries a consistent shortening bias.  Distance to the
    polyline segments removes that bias.  Diameter is interpolated along the
    nearest segment.  Shapes: ``[B,N,2]`` points, ``[B,N]`` diameters.
    """

    if centerline_xy.ndim != 3 or centerline_xy.shape[-1] != 2 or centerline_xy.shape[1] < 2:
        raise ValueError("centerline_xy must have shape [B,N>=2,2]")
    if diameter.shape != centerline_xy.shape[:2]:
        raise ValueError("diameter must have shape [B,N]")
    if image_height <= 0 or image_width <= 0 or edge_softness <= 0:
        raise ValueError("positive image dimensions and edge_softness are required")
    dtype, device = centerline_xy.dtype, centerline_xy.device
    yy, xx = torch.meshgrid(
        torch.arange(image_height, dtype=dtype, device=device),
        torch.arange(image_width, dtype=dtype, device=device),
        indexing="ij",
    )
    pixels = torch.stack((xx, yy), -1).reshape(1, -1, 1, 2)
    start = centerline_xy[:, None, :-1, :]
    segment = (centerline_xy[:, 1:, :] - centerline_xy[:, :-1, :])[:, None, :, :]
    segment_length_sq = segment.square().sum(-1).clamp_min(1e-6)
    to_pixel = pixels - start
    t = ((to_pixel * segment).sum(-1) / segment_length_sq).clamp(0.0, 1.0)
    closest = start + t[..., None] * segment
    distance_sq = (pixels - closest).square().sum(-1)
    min_distance_sq, nearest = distance_sq.min(-1)
    t_nearest = torch.gather(t, 2, nearest[..., None]).squeeze(-1)
    diameter_start = torch.gather(diameter[:, :-1], 1, nearest)
    diameter_end = torch.gather(diameter[:, 1:], 1, nearest)
    local_diameter = (1.0 - t_nearest) * diameter_start + t_nearest * diameter_end
    distance = torch.sqrt(min_distance_sq + torch.finfo(dtype).eps)
    mask = torch.sigmoid((0.5 * local_diameter - distance) / edge_softness)
    return mask.reshape(centerline_xy.shape[0], image_height, image_width)


def _render_in_crop(
    centerline: Tensor, width_profile: Tensor, crop: CropWindow, factor: int, softness: float
) -> Tensor:
    offset = torch.tensor([crop.x0, crop.y0], dtype=centerline.dtype, device=centerline.device)
    local = (centerline - offset + 0.5) / factor - 0.5
    return render_tube_segments(
        local,
        width_profile / factor,
        crop.height // factor,
        crop.width // factor,
        edge_softness=softness,
    )


def hard_iou(prediction: NDArray[np.generic], target: NDArray[np.generic]) -> float:
    a = np.asarray(prediction, dtype=bool)
    b = np.asarray(target, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / union if union else 0.0


def fit_mask(
    mask: NDArray[np.generic],
    initializations: Sequence[Initialization],
    *,
    width_template: NDArray[np.generic] | None = None,
    config: MaskFitConfig = MaskFitConfig(),
    device: torch.device | str | None = None,
    extra_masks: dict[str, NDArray[np.generic]] | None = None,
) -> MaskFitResult:
    """Optimize every initialization jointly and return the best final fit."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("mask must be a non-empty 2-D boolean image")
    if not initializations:
        raise ValueError("at least one initialization is required")
    if not (
        len(config.stage_downsample) == len(config.stage_steps) == len(config.stage_lr_scale)
    ):
        raise ValueError("stage_downsample, stage_steps, and stage_lr_scale must align")
    if not 0 < config.within_stage_decay <= 1:
        raise ValueError("within_stage_decay must lie in (0, 1]")
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    template = (
        default_width_template(config.n_points)
        if width_template is None
        else np.asarray(width_template, dtype=np.float64)
    )
    if template.shape != (config.n_points,) or np.any(template <= 0):
        raise ValueError("width_template must be positive with shape [n_points]")
    template_t = torch.as_tensor(template, dtype=torch.float32, device=resolved_device)

    crop = crop_window(binary, config.crop_padding, max(config.stage_downsample))
    target_full = torch.as_tensor(
        binary[crop.y0 : crop.y1, crop.x0 : crop.x1], dtype=torch.float32, device=resolved_device
    )
    # The renderer produces a sigmoid of distance at ``edge_softness`` pixels of
    # the stage raster.  Blurring the observed mask with the same sigmoid of
    # its signed distance keeps the optimum consistent across stages; a hard
    # target would bias a coarse stage toward a shorter, wider tube.
    if bool(target_full.all()):
        signed_distance = torch.full_like(target_full, float("inf"))
    else:
        signed_distance = signed_edge_distance(target_full >= 0.5)

    def stage_target(factor: int) -> Tensor:
        soft = torch.sigmoid(signed_distance / (config.edge_softness * factor))
        return _downsample(soft, factor)
    state = _MaskFitState(initializations, config, resolved_device)
    optimizer = state.optimizer()
    base_rates = [group["lr"] for group in optimizer.param_groups]
    history: list[list[float]] = []

    def energy(factor: int) -> tuple[Tensor, Tensor]:
        centerline = state.centerline()
        width = state.diameter(template_t)
        rendered = _render_in_crop(centerline, width, crop, factor, config.edge_softness)
        dice = soft_dice_energy(rendered, stage_target(factor))
        return dice, dice + state.regularization(centerline, crop)

    with torch.no_grad():
        initial_dice, _ = energy(1)
    best_loss = torch.full((len(initializations),), float("inf"), device=resolved_device)
    best_snapshot = state.parameter_snapshot()
    for factor, steps, scale in zip(
        config.stage_downsample, config.stage_steps, config.stage_lr_scale, strict=True
    ):
        for step in range(steps):
            progress = step / max(steps - 1, 1)
            decay = 1.0 + (config.within_stage_decay - 1.0) * progress
            for group, base in zip(optimizer.param_groups, base_rates, strict=True):
                group["lr"] = base * scale * decay
            optimizer.zero_grad(set_to_none=True)
            dice, loss = energy(factor)
            total = loss.sum()
            if not bool(torch.isfinite(total)):
                raise RuntimeError("non-finite mask-fit energy")
            if factor == 1:
                improved = loss.detach() < best_loss
                if bool(improved.any()):
                    rows = torch.nonzero(improved, as_tuple=False).squeeze(1)
                    current = state.parameter_snapshot()
                    for kept, now in zip(best_snapshot, current, strict=True):
                        kept[rows] = now[rows]
                    best_loss = torch.where(improved, loss.detach(), best_loss)
            total.backward()
            optimizer.step()
            history.append([float(v) for v in dice.detach().cpu()])
    if bool(torch.isfinite(best_loss).any()):
        rows = torch.nonzero(torch.isfinite(best_loss), as_tuple=False).squeeze(1)
        state.restore_rows(best_snapshot, rows)

    with torch.no_grad():
        final_dice, final_loss = energy(1)
        centerline = state.centerline()
        width_scale = state.log_width.exp()
        width = state.diameter(template_t)
        rendered = _render_in_crop(centerline, width, crop, 1, config.edge_softness)
        hard = (rendered >= config.hard_threshold).cpu().numpy()
        target_np = binary[crop.y0 : crop.y1, crop.x0 : crop.x1]
        # Initial hard IoU for the record: render the starting states once more.
        start_state = _MaskFitState(initializations, config, resolved_device)
        start_width = start_state.diameter(template_t)
        start_hard = (
            _render_in_crop(start_state.centerline(), start_width, crop, 1, config.edge_softness)
            >= config.hard_threshold
        ).cpu().numpy()

    records = []
    for index, start in enumerate(initializations):
        records.append(
            {
                "name": start.name,
                "initial_soft_dice_energy": float(initial_dice[index]),
                "final_soft_dice_energy": float(final_dice[index]),
                "final_energy": float(final_loss[index]),
                "initial_iou": hard_iou(start_hard[index], target_np),
                "final_iou": hard_iou(hard[index], target_np),
            }
        )
    # Winner by total energy: overlap plus priors.
    best = int(torch.argmin(final_loss))
    best_centerline = centerline[best].cpu().numpy().astype(np.float64)
    in_fov = int(
        np.sum(
            (best_centerline[:, 0] >= 0)
            & (best_centerline[:, 0] < crop.image_width)
            & (best_centerline[:, 1] >= 0)
            & (best_centerline[:, 1] < crop.image_height)
        )
    )
    extra: dict[str, float] = {}
    for name, other in (extra_masks or {}).items():
        other_crop = np.asarray(other, dtype=bool)[crop.y0 : crop.y1, crop.x0 : crop.x1]
        extra[name] = hard_iou(hard[best], other_crop)
    return MaskFitResult(
        best_index=best,
        initializations=list(initializations),
        records=records,
        latent=state.latent()[best].detach().cpu().numpy().astype(np.float64),
        width_px=float(width_scale[best]),
        width_profile=(width[best]).cpu().numpy().astype(np.float64),
        centerline_xy=best_centerline,
        crop=crop,
        rendered_hard_mask=hard[best],
        energy_history=np.asarray(history, dtype=np.float64),
        points_in_fov=in_fov,
        body_length_px=float(np.linalg.norm(np.diff(best_centerline, axis=0), axis=1).sum()),
        extra_iou=extra,
        width_shape=state.width_shape[best].detach().cpu().numpy().astype(np.float64),
    )
