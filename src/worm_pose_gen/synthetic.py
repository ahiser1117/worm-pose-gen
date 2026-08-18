"""Analytic Tier-C worm geometry and controlled anatomical censoring.

Synthetic poses live in an audited, original-image-equivalent 732 x 968
pixel canvas. Training rasters use 192 x 256 pixels. Coordinates map as
``u = x * (256 / 968)`` (and analogously for y), so half-open geometric FOV
membership is preserved exactly and the inverse is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .geometry import curvature_from_angles, in_fov_mask, reconstruct_centerline, wrap_angle
from .metrics import anatomical_support_mask


@dataclass(frozen=True)
class SyntheticConfig:
    num_points: int = 100
    original_height: int = 732
    original_width: int = 968
    render_height: int = 192
    render_width: int = 256
    min_length: float = 250.0
    max_length: float = 700.0


@dataclass(frozen=True)
class ParameterProfile:
    """Frozen split-specific geometry distribution."""

    name: str
    length_bands: tuple[tuple[float, float], ...]
    bend_amplitude_range: tuple[float, float]


PARAMETER_PROFILES = {
    "development": ParameterProfile(
        name="development",
        length_bands=((300.0, 600.0),),
        bend_amplitude_range=(0.25, 0.55),
    ),
    "held_out": ParameterProfile(
        name="held_out",
        length_bands=((250.0, 299.0), (601.0, 700.0)),
        bend_amplitude_range=(0.65, 0.90),
    ),
}


@dataclass(frozen=True)
class CameraTransform:
    """Invertible source-to-camera rigid transform in original pixel units."""

    matrix: Tensor  # [2, 2]
    offset: Tensor  # [2]
    width: int
    height: int

    def to_camera(self, xy: Tensor) -> Tensor:
        return xy @ self.matrix.transpose(-1, -2) + self.offset

    def to_source(self, xy: Tensor) -> Tensor:
        return (xy - self.offset) @ self.matrix


def original_to_render(xy: Tensor, config: SyntheticConfig = SyntheticConfig()) -> Tensor:
    """Map original-equivalent pixel centers to training-raster centers."""

    scale = xy.new_tensor(
        (config.render_width / config.original_width, config.render_height / config.original_height)
    )
    return xy * scale


def render_to_original(xy: Tensor, config: SyntheticConfig = SyntheticConfig()) -> Tensor:
    """Inverse of :func:`original_to_render`."""

    scale = xy.new_tensor(
        (config.render_width / config.original_width, config.render_height / config.original_height)
    )
    return xy / scale


def _uniform(generator: torch.Generator, low: float, high: float, shape: tuple[int, ...] = ()) -> Tensor:
    return low + (high - low) * torch.rand(shape, generator=generator, dtype=torch.float64)


def _profile(name: str) -> ParameterProfile:
    try:
        return PARAMETER_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown parameter profile {name!r}") from error


def generate_synthetic_pose(
    seed: int,
    config: SyntheticConfig = SyntheticConfig(),
    *,
    profile: str = "development",
) -> dict[str, Tensor | int | str]:
    """Generate one deterministic, smooth intrinsic-angle worm pose.

    The coefficients are bounded so the curve progresses monotonically along
    its global axis.  This makes exact head/tail camera censoring possible
    without changing the known anatomy.
    """

    if config.num_points < 2 or config.min_length < 250 or config.max_length > 700:
        raise ValueError("the Tier-C contract requires N>=2 and lengths within 250..700 px")
    distribution = _profile(profile)
    for low, high in distribution.length_bands:
        if low < config.min_length or high > config.max_length or low > high:
            raise ValueError("profile length band is outside the configured contract")
    generator = torch.Generator().manual_seed(seed)
    s = torch.linspace(0, 1, config.num_points, dtype=torch.float64)
    phase = _uniform(generator, -math.pi, math.pi, (3,))
    coefficient_direction = _uniform(generator, -1.0, 1.0, (3,)) * torch.tensor(
        [1.0, 0.55, 0.25], dtype=torch.float64
    )
    basis = torch.stack(
        [torch.sin(2 * math.pi * (index + 1) * s + phase[index]) for index in range(3)], dim=-1
    )
    intrinsic = basis @ coefficient_direction
    # Remove the mean orientation: phi alone controls global rotation.
    intrinsic = intrinsic - intrinsic.mean()
    bend_amplitude = _uniform(generator, *distribution.bend_amplitude_range)
    intrinsic = intrinsic * (bend_amplitude / intrinsic.abs().max().clamp_min(1e-12))
    phi = _uniform(generator, -math.pi, math.pi)
    tangent = phi + intrinsic
    if len(distribution.length_bands) == 1:
        band_index = 0
    else:
        band_index = int(torch.randint(len(distribution.length_bands), (), generator=generator))
    length_band = distribution.length_bands[band_index]
    length = _uniform(generator, *length_band)
    centered = reconstruct_centerline(
        torch.zeros(2, dtype=torch.float64), tangent, length, anchor_index=config.num_points // 2
    )

    margin = 18.0
    minimum, maximum = centered.amin(0), centered.amax(0)
    low = -minimum + margin
    high = centered.new_tensor((config.original_width, config.original_height)) - maximum - margin
    if bool(torch.any(low > high)):
        # Bounded curvature should make this unreachable under the declared dimensions.
        raise RuntimeError("generated centerline cannot fit the audited canvas")
    anchor = low + torch.rand(2, generator=generator, dtype=torch.float64) * (high - low)
    centerline = centered + anchor
    width = length.new_tensor(12.0) + _uniform(generator, -2.0, 3.0)
    width_profile = width * (0.35 + 0.65 * torch.sin(math.pi * s).clamp_min(0).pow(0.45))
    return {
        "seed": seed,
        "parameter_profile": profile,
        "length_band_index": band_index,
        "bend_amplitude": bend_amplitude,
        "global_orientation": phi,
        "centerline_xy": centerline,
        "centerline_render_xy": original_to_render(centerline, config),
        "tangent_angle": tangent,
        "curvature": curvature_from_angles(tangent, length),
        "body_length": length,
        "width_profile": width_profile,
        "width_profile_render": width_profile * (config.render_width / config.original_width),
        "in_fov_mask": in_fov_mask(centerline, config.original_height, config.original_width),
    }


def _longitudinal_rotation(centerline_xy: Tensor) -> Tensor:
    """Rotate a forward-progressing centerline onto increasing camera x."""

    segment = centerline_xy[1:] - centerline_xy[:-1]
    angle = torch.atan2(segment[:, 1], segment[:, 0])
    relative = wrap_angle(angle - angle[0])
    longitudinal_angle = angle[0] + 0.5 * (relative.amin() + relative.amax())
    cosine, sine = torch.cos(longitudinal_angle), torch.sin(longitudinal_angle)
    rotation = torch.stack((torch.stack((cosine, sine)), torch.stack((-sine, cosine))))
    projected_step = (segment @ rotation.transpose(-1, -2))[:, 0]
    if not bool(torch.all(projected_step > 0)):
        raise RuntimeError("pose does not progress monotonically along its longitudinal axis")
    return rotation


def anatomical_crop_transform(
    centerline_xy: Tensor,
    hidden_fraction: float,
    hidden_end: str,
    config: SyntheticConfig = SyntheticConfig(),
) -> tuple[CameraTransform, Tensor, Tensor]:
    """Place exactly the requested anatomical points outside a fixed camera.

    The latent pose is rigidly re-expressed in camera coordinates and rendered
    directly there; no border, padding, or composited crop edge enters the
    input. Returns transform, camera coordinates, and exact geometric support.
    """

    target = anatomical_support_mask(config.num_points, hidden_fraction, hidden_end=hidden_end)
    rotation = _longitudinal_rotation(centerline_xy)
    rotated = centerline_xy @ rotation.transpose(-1, -2)
    hidden = int((~target).sum())
    if hidden == 0:
        x_offset = config.original_width / 2 - rotated[:, 0].mean()
    elif hidden_end == "head":
        boundary = 0.5 * (rotated[hidden - 1, 0] + rotated[hidden, 0])
        x_offset = -boundary
    else:
        boundary = 0.5 * (rotated[-hidden - 1, 0] + rotated[-hidden, 0])
        x_offset = config.original_width - boundary
    y_offset = config.original_height / 2 - rotated[:, 1].mean()
    transform = CameraTransform(
        matrix=rotation,
        offset=centerline_xy.new_tensor((x_offset, y_offset)),
        width=config.original_width,
        height=config.original_height,
    )
    camera = transform.to_camera(centerline_xy)
    support = in_fov_mask(camera, config.original_height, config.original_width)
    if not torch.equal(support.cpu(), target.cpu()):
        raise RuntimeError("crop geometry did not produce the requested exact anatomical support")
    return transform, camera, support


def moving_crop_sequence(
    centerline_xy: Tensor,
    *,
    hidden_end: str,
    start_hidden_fraction: float = 0.05,
    end_hidden_fraction: float = 0.40,
    num_frames: int = 21,
    config: SyntheticConfig = SyntheticConfig(),
) -> dict[str, Tensor | list[CameraTransform]]:
    """Create a temporally coherent camera sequence with a smooth boundary.

    The latent pose remains fixed while the camera offset moves linearly from
    the exact start-fraction boundary to the exact end-fraction boundary.
    Support changes only when that continuous boundary crosses an anatomical
    point. Every frame retains an exact invertible rigid transform.
    """

    if hidden_end not in {"head", "tail"}:
        raise ValueError("hidden_end must be 'head' or 'tail'")
    if num_frames < 2:
        raise ValueError("num_frames must be at least two")
    start_hidden = round(start_hidden_fraction * config.num_points)
    end_hidden = round(end_hidden_fraction * config.num_points)
    if not 1 <= start_hidden < end_hidden < config.num_points:
        raise ValueError("sequence requires increasing nonempty hidden fractions")

    rotation = _longitudinal_rotation(centerline_xy)
    rotated = centerline_xy @ rotation.transpose(-1, -2)

    def boundary_for_count(hidden: int) -> Tensor:
        if hidden_end == "head":
            return 0.5 * (rotated[hidden - 1, 0] + rotated[hidden, 0])
        return 0.5 * (rotated[-hidden - 1, 0] + rotated[-hidden, 0])

    boundary = torch.linspace(
        float(boundary_for_count(start_hidden)),
        float(boundary_for_count(end_hidden)),
        num_frames,
        dtype=centerline_xy.dtype,
        device=centerline_xy.device,
    )
    y_offset = config.original_height / 2 - rotated[:, 1].mean()
    transforms: list[CameraTransform] = []
    camera_frames: list[Tensor] = []
    support_frames: list[Tensor] = []
    for value in boundary:
        x_offset = -value if hidden_end == "head" else config.original_width - value
        transform = CameraTransform(
            matrix=rotation,
            offset=torch.stack((x_offset, y_offset)),
            width=config.original_width,
            height=config.original_height,
        )
        camera = transform.to_camera(centerline_xy)
        transforms.append(transform)
        camera_frames.append(camera)
        support_frames.append(in_fov_mask(camera, config.original_height, config.original_width))
    camera_xy = torch.stack(camera_frames)
    support = torch.stack(support_frames)
    hidden_count = (~support).sum(-1)
    if int(hidden_count[0]) != start_hidden or int(hidden_count[-1]) != end_hidden:
        raise RuntimeError("sequence endpoint support does not match requested fractions")
    if not bool(torch.all(hidden_count[1:] >= hidden_count[:-1])):
        raise RuntimeError("moving boundary produced non-monotonic anatomical support")
    return {
        "transforms": transforms,
        "centerline_camera_xy": camera_xy,
        "support_mask": support,
        "hidden_count": hidden_count,
        "boundary_source_x": boundary,
    }
