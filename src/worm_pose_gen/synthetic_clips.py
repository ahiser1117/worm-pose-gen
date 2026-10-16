"""Synthetic self-contact clips for body-field training (proposal P3 of docs/MODEL_ARCHITECTURE_PROPOSALS.md).

Coils and self-contacts are rare in the hand labels and their targets (tube
fits to hand masks) are least reliable exactly there.  A synthetic item has
exact targets through every crossing and keeps the real motion statistics the
lagged difference channels depend on:

**Pose.**  A stretch of an analysed workspace (``state.npz``: fitted
head-first centerline and width profile per frame) supplies one real pose
per frame the item needs (the centre and ``t ± lag``).  Every pose of the
stretch is transformed the same way in its tangent-angle shape (the latent
of :mod:`latent`): the undulation is amplified, a constant curvature curls
the body, a bend near the head turns it back onto the body, and the worm is
scaled; centroid and mean orientation stay the real ones, so the worm moves
with the stage like the real one did.  Time is warped by reading the real
frames ``t ± round(speed * lag)``, background and pose alike.

**Appearance.**  The real worm is removed from each real frame (its
workspace mask, dilated, filled by normalized convolution plus camera
noise).  The synthetic worm is drawn with a transmission map ``T(s, u)``
(image / background at arc position ``s`` and lateral offset ``u`` in local
radii, :func:`sample_texture`) taken from a clean, contact-free frame of the
same recording, so its edges, internal texture and head/tail taper are those
of a real worm.  Where two parts of the body cross, the two layers'
transmissions combine as ``min(t) * max(t) ** OVERLAP_GAMMA`` (:func:`composite`:
on real crossings the overlap is about as dark as the darker part, not the
product of the two).

**Targets.**  The mask is the tube's coverage; A-P, overlap, head and tail
come from :func:`body_targets.render_body_targets` on that mask with the
exact centerline, i.e. the same definitions as the hand-label targets.
Pixels of the removed real worm that the synthetic worm does not cover are
excluded from the losses (the fill is not a real background).

Items are rendered once (:func:`generate`, CPU, many processes) into
``item_NNNNNN.npz`` files that :class:`SyntheticClipDataset` serves to
:mod:`model_training` beside the hand labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .body_targets import OVERLAP_ARC_DIAMETERS, render_body_targets, self_contact
from .latent import cubic_bspline_basis
from .segmenter import IGNORE_LABEL


FloatArray = NDArray[np.float64]

# Half-width of the texture band in local radii: the real edge falls off a
# little outside the fitted tube, and the band carries that fall-off.
U_MAX = 1.5
TEXTURE_SHAPE = (256, 61)  # samples along (head to tail) and across the body
# The lighter layer of a crossing contributes ``t ** OVERLAP_GAMMA`` (:func:`composite`).
OVERLAP_GAMMA = 0.5
# Real worm pixels are removed with this margin (px) around the workspace mask.
REMOVAL_MARGIN_PX = 6
FILL_SIGMA_PX = 10.0
# Pose stretches: every frame fitted with at least this IoU, and the head never
# jumps more than this many body widths between consecutive frames.
MIN_SOURCE_IOU = 0.9
MAX_HEAD_STEP_WIDTHS = 0.75
# A synthetic body is bent no tighter than this many diameters of radius.
MIN_BEND_RADIUS_DIAMETERS = 0.5
SEPARATION_DIAMETERS = OVERLAP_ARC_DIAMETERS


# --------------------------------------------------------------------------- body coordinates


@dataclass(frozen=True)
class Layers:
    """Body coordinates of the pixels in a box: up to two layers (parts of the body far apart along it).

    ``s`` is the normalized arc position (0 head), ``u`` the signed lateral
    offset in local radii, ``coverage`` the tube coverage ``radius - distance``
    in pixels (positive inside the tube).  Layer 1 is the part whose surface
    is nearest; layer 2 the nearest part at least ``SEPARATION_DIAMETERS``
    away along the body, NaN where there is none within the texture band.
    """

    y0: int
    x0: int
    s: NDArray[np.float32]  # [2,h,w]
    u: NDArray[np.float32]  # [2,h,w]
    coverage: NDArray[np.float32]  # [2,h,w]; -inf where the layer is absent

    @property
    def mask(self) -> NDArray[np.bool_]:
        """The tube's coverage within the box."""

        return (self.coverage > 0).any(0)

    def mask_in(self, shape: tuple[int, int]) -> NDArray[np.bool_]:
        """The tube's coverage in the whole image."""

        full = np.zeros(shape, dtype=bool)
        h, w = self.s.shape[1:]
        full[self.y0 : self.y0 + h, self.x0 : self.x0 + w] = self.mask
        return full


def _frame_geometry(centerline: FloatArray, diameter: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    step = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(step)))
    return step, arc, arc / max(float(arc[-1]), 1e-9), 0.5 * diameter


def body_layers(
    centerline_xy: NDArray[np.generic], width_profile: NDArray[np.generic], shape: tuple[int, int], *, u_max: float = U_MAX,
    chunk_pixels: int = 16384,
) -> Layers:
    """:class:`Layers` for every pixel of the box around the body (clipped to ``shape``)."""

    centerline = np.asarray(centerline_xy, dtype=np.float64)
    diameter = np.asarray(width_profile, dtype=np.float64)
    step, arc, position, radius = _frame_geometry(centerline, diameter)
    separation = SEPARATION_DIAMETERS * float(np.median(diameter))
    margin = u_max * float(radius.max()) + 2
    x0 = int(max(0, np.floor(centerline[:, 0].min() - margin)))
    x1 = int(min(shape[1], np.ceil(centerline[:, 0].max() + margin) + 1))
    y0 = int(max(0, np.floor(centerline[:, 1].min() - margin)))
    y1 = int(min(shape[0], np.ceil(centerline[:, 1].max() + margin) + 1))
    h, w = max(0, y1 - y0), max(0, x1 - x0)
    s_out = np.full((2, h, w), np.nan, dtype=np.float32)
    u_out = np.full((2, h, w), np.nan, dtype=np.float32)
    c_out = np.full((2, h, w), -np.inf, dtype=np.float32)
    if h == 0 or w == 0:
        return Layers(y0, x0, s_out, u_out, c_out)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    # Float64 like body_targets.render_body_targets, so masks and overlaps agree pixel for pixel.
    points = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float64)
    start = centerline[:-1]
    segment = np.diff(centerline, axis=0)
    length_sq = np.maximum((segment**2).sum(-1), 1e-12)
    r0, r1 = radius[:-1], radius[1:]
    p0, dp = position[:-1], position[1:] - position[:-1]
    flat = [a.reshape(2, -1) for a in (s_out, u_out, c_out)]
    for begin in range(0, len(points), chunk_pixels):
        chunk = points[begin : begin + chunk_pixels]
        offset = chunk[:, None, :] - start[None]
        t = np.clip((offset * segment[None]).sum(-1) / length_sq[None], 0.0, 1.0)
        rel = offset - t[..., None] * segment[None]
        distance = np.sqrt((rel**2).sum(-1))
        cross = segment[None, :, 0] * rel[..., 1] - segment[None, :, 1] * rel[..., 0]
        r = (1 - t) * r0[None] + t * r1[None]
        u = np.where(cross >= 0, 1.0, -1.0) * distance / np.maximum(r, 0.25)
        s = p0[None] + t * dp[None]
        coverage = r - distance
        band = np.abs(u) < u_max
        score = np.where(band, coverage, -np.inf)
        rows = np.arange(len(chunk))
        first = np.argmax(score, axis=1)
        has_first = np.isfinite(score[rows, first])
        far = np.abs(s - s[rows, first][:, None]) * float(arc[-1]) > separation
        score2 = np.where(far, score, -np.inf)
        second = np.argmax(score2, axis=1)
        has_second = np.isfinite(score2[rows, second]) & has_first
        sl = slice(begin, begin + len(chunk))
        for layer, index, present in ((0, first, has_first), (1, second, has_second)):
            flat[0][layer, sl] = np.where(present, s[rows, index], np.nan)
            flat[1][layer, sl] = np.where(present, u[rows, index], np.nan)
            flat[2][layer, sl] = np.where(present, coverage[rows, index], -np.inf)
    return Layers(y0, x0, s_out, u_out, c_out)


# --------------------------------------------------------------------------- appearance


@dataclass(frozen=True)
class Texture:
    """Transmission (image / background) of a real worm on a ``[along, across]`` grid of body coordinates."""

    transmission: NDArray[np.float32]
    source: str = ""

    def sample(self, s: NDArray[np.floating], u: NDArray[np.floating]) -> NDArray[np.float32]:
        along, across = self.transmission.shape
        rows = np.clip(np.nan_to_num(s, nan=0.0), 0.0, 1.0) * (along - 1)
        cols = (np.clip(np.nan_to_num(u, nan=U_MAX), -U_MAX, U_MAX) + U_MAX) / (2 * U_MAX) * (across - 1)
        values = ndimage.map_coordinates(self.transmission, [rows.ravel(), cols.ravel()], order=1, mode="nearest")
        values = values.reshape(np.shape(s)).astype(np.float32)
        return np.where(np.isnan(s) | np.isnan(u), np.float32(1.0), values)


def sample_texture(
    frame: NDArray[np.generic], background: NDArray[np.generic], centerline_xy: NDArray[np.generic],
    width_profile: NDArray[np.generic], source: str = "",
) -> Texture:
    """The transmission map of the worm in ``frame`` along its fitted head-first tube.

    ``background`` is the frame without the worm (:func:`remove_worm`).  The
    map fades to 1 at the band's edge so a drawn worm never leaves a seam.
    """

    centerline = np.asarray(centerline_xy, dtype=np.float64)
    radius = 0.5 * np.asarray(width_profile, dtype=np.float64)
    along, across = TEXTURE_SHAPE
    s = np.linspace(0.0, 1.0, along)
    u = np.linspace(-U_MAX, U_MAX, across)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))))
    position = arc / arc[-1]
    x = np.interp(s, position, centerline[:, 0])
    y = np.interp(s, position, centerline[:, 1])
    r = np.interp(s, position, radius)
    tangent = np.gradient(np.stack((x, y), 1), axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    # The side with positive cross product (segment x offset) is +u, as in body_layers.
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    px = x[:, None] + u[None, :] * r[:, None] * normal[:, None, 0]
    py = y[:, None] + u[None, :] * r[:, None] * normal[:, None, 1]
    image = ndimage.map_coordinates(np.asarray(frame, dtype=np.float32), [py.ravel(), px.ravel()], order=1, mode="nearest")
    base = ndimage.map_coordinates(np.asarray(background, dtype=np.float32), [py.ravel(), px.ravel()], order=1, mode="nearest")
    transmission = (image / np.maximum(base, 1.0)).reshape(along, across)
    fade = np.clip((U_MAX - np.abs(u)) / 0.25, 0.0, 1.0)[None, :]
    transmission = 1.0 + (transmission - 1.0) * fade
    return Texture(np.clip(transmission, 0.2, 1.6).astype(np.float32), source)


def remove_worm(
    frame: NDArray[np.generic], worm: NDArray[np.bool_], rng: np.random.Generator, noise_sigma: float,
    margin_px: int = REMOVAL_MARGIN_PX,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """The frame with the worm filled in from its surroundings, and the filled pixels.

    The fill is a normalized convolution of the known background (repeated
    at wider scales where a pixel is far from it) plus white noise of the
    camera's level.
    """

    image = np.asarray(frame, dtype=np.float32)
    filled = ndimage.binary_dilation(worm, iterations=margin_px) if margin_px else np.array(worm, dtype=bool)
    result = image.copy()
    if not filled.any():
        return result, filled
    # Work in the box around the worm, with room for the widest fill kernel.
    ys, xs = np.nonzero(filled)
    pad = int(4 * FILL_SIGMA_PX * 4)
    box = (slice(max(0, ys.min() - pad), ys.max() + pad + 1), slice(max(0, xs.min() - pad), xs.max() + pad + 1))
    region, hole = image[box], filled[box]
    known = (~hole).astype(np.float32)
    fill = region.copy()
    remaining = hole.copy()
    sigma = FILL_SIGMA_PX
    while remaining.any() and sigma <= 4 * FILL_SIGMA_PX:
        weight = ndimage.gaussian_filter(known, sigma)
        value = ndimage.gaussian_filter(region * known, sigma)
        ready = remaining & (weight > 0.05)
        fill[ready] = value[ready] / weight[ready]
        remaining &= ~ready
        sigma *= 2
    if remaining.any():
        fill[remaining] = float(np.median(region[~hole])) if (~hole).any() else float(region.mean())
    fill[hole] += rng.normal(0.0, noise_sigma, int(hole.sum())).astype(np.float32)
    result[box] = fill
    return result, filled


def camera_noise(frame: NDArray[np.generic], worm: NDArray[np.bool_]) -> float:
    """White-noise level of a frame's background (std of a 2-px high-pass, corrected for the filter)."""

    image = np.asarray(frame, dtype=np.float32)
    high = image - ndimage.gaussian_filter(image, 2.0)
    away = ~ndimage.binary_dilation(worm, iterations=15)
    # A unit white noise keeps about 0.96 of its std through (1 - G_2).
    return float(np.std(high[away]) / 0.96) if away.any() else 5.0


def composite(first: NDArray[np.float32], second: NDArray[np.float32]) -> NDArray[np.float32]:
    """Transmission where two layers cover a pixel: the darker one fully, the lighter one at ``OVERLAP_GAMMA``.

    Real crossings are about as dark as the darker body part, with the other
    part's outline still visible; a plain product (additive absorbance) makes
    them much darker than they look.  Symmetric, so no layer order is needed.
    """

    darker, lighter = np.minimum(first, second), np.maximum(first, second)
    return (darker * np.power(np.maximum(lighter, 1e-3), OVERLAP_GAMMA)).astype(np.float32)


def render_worm(
    background: NDArray[np.float32], centerline_xy: NDArray[np.generic], width_profile: NDArray[np.generic], texture: Texture,
    rng: np.random.Generator, noise_sigma: float,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """``background`` with the worm drawn on it (8-bit) and the worm's mask (tube coverage)."""

    image = np.array(background, dtype=np.float32, copy=True)
    layers = body_layers(centerline_xy, width_profile, image.shape)
    mask = np.zeros(image.shape, dtype=bool)
    h, w = layers.s.shape[1:]
    if h and w:
        top = texture.sample(layers.s[0], layers.u[0])
        bottom = texture.sample(layers.s[1], layers.u[1])
        transmission = composite(top, bottom)
        box = (slice(layers.y0, layers.y0 + h), slice(layers.x0, layers.x0 + w))
        region = image[box]
        # Replace the texture's frozen noise by fresh camera noise inside the band.
        band = np.isfinite(layers.coverage[0])
        drawn = region * transmission
        drawn = np.where(band, drawn + rng.normal(0.0, 0.5 * noise_sigma, drawn.shape).astype(np.float32), drawn)
        image[box] = drawn
        mask[box] = layers.mask
    return np.clip(np.rint(image), 0, 255).astype(np.uint8), mask


# --------------------------------------------------------------------------- poses


@dataclass(frozen=True)
class PoseTransform:
    """One item's change of a real pose stretch (applied identically to every frame, plus a slow curl drift).

    ``gain`` multiplies the undulation (the tangent angle about its mean);
    ``curl`` adds a constant curvature of that total turning (rad) along the
    body, drifting by ``curl_drift`` per ``DRIFT_FRAMES`` real frames;
    ``head_bend`` turns the anterior ``HEAD_BEND_EXTENT`` of the body by that
    angle (rad) toward one side; ``scale`` scales length and width.
    """

    gain: float = 1.0
    curl: float = 0.0
    curl_drift: float = 0.0
    head_bend: float = 0.0
    scale: float = 1.0


DRIFT_FRAMES = 16
HEAD_BEND_EXTENT = 0.3


def random_transform(rng: np.random.Generator) -> PoseTransform:
    return PoseTransform(
        gain=float(rng.uniform(1.0, 2.0)),
        curl=float(rng.choice((-1, 1)) * rng.uniform(0.0, 2.5 * np.pi) * (rng.random() < 0.7)),
        curl_drift=float(rng.normal(0.0, 0.3 * np.pi)),
        head_bend=float(rng.normal(0.0, 1.2) * (rng.random() < 0.5)),
        scale=float(rng.uniform(0.9, 1.15)),
    )


def tangent_angles(centerline_xy: NDArray[np.generic]) -> FloatArray:
    difference = np.diff(np.asarray(centerline_xy, dtype=np.float64), axis=0)
    return np.unwrap(np.arctan2(difference[:, 1], difference[:, 0]))


def transform_pose(
    centerline_xy: NDArray[np.generic], length: float, transform: PoseTransform, frame_offset: int = 0
) -> FloatArray:
    """A transformed copy of a head-first centerline: same centroid and mean orientation, ``length * scale`` long.

    The shape is smoothed through the 16-coefficient B-spline of
    :mod:`latent` first, so the gain amplifies the body wave, not the fit's
    noise.
    """

    angle = tangent_angles(centerline_xy)
    count = len(angle)
    basis = cubic_bspline_basis(count, 16)
    rotation = float(angle.mean())
    shape = basis @ np.linalg.lstsq(basis, angle - rotation, rcond=None)[0]
    s = (np.arange(count) + 0.5) / count
    curl = transform.curl + transform.curl_drift * frame_offset / DRIFT_FRAMES
    extra = curl * (s - 0.5) + transform.head_bend * np.clip(1.0 - s / HEAD_BEND_EXTENT, 0.0, 1.0) ** 2
    new = transform.gain * shape + extra
    new = new - new.mean() + rotation
    step = transform.scale * length / count
    curve = np.vstack((np.zeros(2), np.cumsum(step * np.column_stack((np.cos(new), np.sin(new))), axis=0)))
    centroid = np.asarray(centerline_xy, dtype=np.float64).mean(0)
    return curve - curve.mean(0) + centroid


def min_bend_radius(centerline_xy: NDArray[np.generic]) -> float:
    """Smallest radius of curvature (px) along a centerline."""

    angle = tangent_angles(centerline_xy)
    step = np.linalg.norm(np.diff(np.asarray(centerline_xy, dtype=np.float64), axis=0), axis=1)
    curvature = np.abs(np.diff(angle)) / np.maximum(0.5 * (step[1:] + step[:-1]), 1e-9)
    return float(1.0 / max(curvature.max(), 1e-9))


# --------------------------------------------------------------------------- sources


def _clip_windows(manifest: dict[str, Any], recording: str, margin: int) -> list[tuple[int, int]]:
    name = Path(recording).stem
    return [(c["start"] - margin, c["start"] + c["frames"] + margin)
            for c in manifest["clips"] if Path(c["recording"]).stem == name]


class Source:
    """An analysed workspace: its recording's frames, cleaned masks and fitted poses."""

    def __init__(self, workspace: Path) -> None:
        from .temporal_eval import load_manifest

        self.path = Path(workspace)
        info = json.loads((self.path / "workspace.json").read_text())
        self.recording = str(info["recording"])
        self.name = Path(self.recording).stem
        self.shape = tuple(int(v) for v in info["image_shape"])
        with np.load(self.path / "state.npz") as state:
            self.frame_index = state["frame_index"].astype(np.int64)
            self.centerline = state["centerline_xy"].astype(np.float64)
            self.width = state["width_profile"].astype(np.float64)
            fitted, iou = state["fitted"], state["iou"]
            self.contact = state["flag_self_contact"].astype(bool)
            self.border = state["mask_on_border"].astype(bool)
            self.iou = iou
        head = self.centerline[:, 0]
        width = np.median(self.width, axis=1)
        step = np.linalg.norm(np.diff(head, axis=0), axis=1)
        continuous = np.concatenate(([True], step < MAX_HEAD_STEP_WIDTHS * width[1:])) & np.concatenate(
            (step < MAX_HEAD_STEP_WIDTHS * width[:-1], [True]))
        manifest = load_manifest()
        self.excluded_recording = self.name in manifest["excluded_recordings"]
        allowed = np.ones(len(self.frame_index), dtype=bool)
        for start, stop in _clip_windows(manifest, self.recording, int(manifest["exclusion_margin"]) + 32):
            allowed &= ~((self.frame_index >= start) & (self.frame_index < stop))
        self.good = fitted & (iou >= MIN_SOURCE_IOU) & continuous & allowed & ~self.excluded_recording
        self._chunks: dict[int, dict[str, NDArray[Any]]] = {}
        self._frames: Any = None

    def windows(self, half: int) -> NDArray[np.int64]:
        """Rows whose whole ``±half`` stretch is good and consecutive in frame index."""

        good = self.good.astype(np.int64)
        total = np.convolve(good, np.ones(2 * half + 1, dtype=np.int64), mode="same")
        rows = np.flatnonzero(total == 2 * half + 1)
        rows = rows[(rows >= half) & (rows < len(good) - half)]
        return rows[self.frame_index[rows + half] - self.frame_index[rows - half] == 2 * half]

    def texture_rows(self) -> NDArray[np.int64]:
        """Clean frames to take a worm's appearance from: well fitted, no contact, fully in view, not tightly bent."""

        rows = np.flatnonzero(self.good & (self.iou >= 0.95) & ~self.contact & ~self.border)
        keep = [r for r in rows[:: max(1, len(rows) // 400)]
                if min_bend_radius(self.centerline[r]) > 1.2 * np.median(self.width[r])]
        return np.asarray(keep, dtype=np.int64)

    def mask(self, row: int) -> NDArray[np.bool_]:
        chunk = row // 1024
        if chunk not in self._chunks:
            with np.load(self.path / "masks" / f"chunk_{chunk:05d}.npz") as archive:
                self._chunks = {chunk: {k: archive[k] for k in archive.files}}
        data = self._chunks[chunk]
        position = int(np.searchsorted(data["rows"], row))
        if position >= len(data["rows"]) or data["rows"][position] != row:
            raise KeyError(f"{self.path.name}: no mask for row {row}")
        height, width = self.shape
        return np.unpackbits(data["packed"][position])[: height * width].reshape(height, width).astype(bool)

    def frames(self, rows: Sequence[int]) -> NDArray[np.uint8]:
        from .pipeline import Frames

        if self._frames is None:
            self._frames = Frames(Path(self.recording))
        return self._frames.corrected([int(self.frame_index[r]) for r in rows])[0]


def build_textures(source: Source, count: int, rng: np.random.Generator) -> list[Texture]:
    rows = source.texture_rows()
    chosen = np.sort(rng.choice(rows, size=min(count, len(rows)), replace=False))
    frames = source.frames(chosen)
    textures = []
    for row, frame in zip(chosen, frames):
        worm = source.mask(int(row))
        background, _ = remove_worm(frame, worm, rng, 0.0)
        textures.append(sample_texture(frame, background, source.centerline[row], source.width[row], f"{source.name}:{source.frame_index[row]}"))
    return textures


# --------------------------------------------------------------------------- items


SPEEDS = (0.5, 0.75, 1.0, 1.0, 1.25, 1.5, 2.0)
CONTACT_SHARE = 0.8


def real_offsets(lags: Sequence[int], speed: float) -> list[int]:
    """Real frame offsets standing in for the model's lags at a playback speed (strictly increasing, at least 1)."""

    offsets: list[int] = []
    for lag in lags:
        offset = max(1, int(round(speed * lag)))
        offsets.append(max(offset, offsets[-1] + 1) if offsets else offset)
    return offsets


def make_item(
    source: Source, textures: Sequence[Texture], centre: int, lags: Sequence[int], rng: np.random.Generator,
    *, want_contact: bool | None = None, tries: int = 40,
) -> dict[str, Any] | None:
    """One rendered item around ``centre`` (a row of ``source.windows``), or ``None`` when no transform qualifies."""

    if want_contact is None:
        want_contact = bool(rng.random() < CONTACT_SHARE)
    speed = float(rng.choice(SPEEDS))
    offsets = real_offsets(lags, speed)
    if centre - offsets[-1] < 0 or centre + offsets[-1] >= len(source.frame_index):
        return None
    signed = [-o for o in offsets[::-1]] + [0] + offsets
    rows = [centre + o for o in signed]
    if not source.good[rows].all():
        return None
    width = source.width[centre]
    length = float(np.linalg.norm(np.diff(source.centerline[centre], axis=0), axis=1).sum())
    chosen = None
    for _ in range(tries):
        transform = random_transform(rng)
        diameter = transform.scale * width
        poses = [transform_pose(source.centerline[row], length, transform, offset) for row, offset in zip(rows, signed)]
        bend = min(min_bend_radius(p) for p in poses)
        if bend < MIN_BEND_RADIUS_DIAMETERS * float(np.median(diameter)):
            continue
        if self_contact(poses[len(offsets)], diameter) == want_contact:
            chosen = (transform, diameter, poses)
            break
    if chosen is None:
        return None
    transform, diameter, poses = chosen
    texture = textures[int(rng.integers(len(textures)))]
    real = source.frames(rows)
    centre_position = len(offsets)
    noise = camera_noise(real[centre_position], source.mask(rows[centre_position]))
    frames = np.empty_like(real)
    for k, (row, frame, pose) in enumerate(zip(rows, real, poses)):
        background, filled = remove_worm(frame, source.mask(row), rng, noise)
        frames[k], mask = render_worm(background, pose, diameter, texture, rng, noise)
        if k == centre_position:
            centre_mask, centre_filled = mask, filled
    targets = render_body_targets(centre_mask, poses[centre_position], diameter)
    label = centre_mask.astype(np.uint8)
    label[centre_filled & ~ndimage.binary_dilation(centre_mask, iterations=3)] = IGNORE_LABEL
    return {
        "frames": frames, "lags": np.asarray(lags, dtype=np.int64), "offsets": np.asarray(offsets, dtype=np.int64),
        "mask": label, "ap": targets.ap.astype(np.float16), "overlap": targets.overlap,
        "head_xy": targets.head_xy, "tail_xy": targets.tail_xy, "diameter_px": np.float64(targets.diameter_px),
        "centerline_xy": poses[centre_position].astype(np.float32), "width_profile": diameter.astype(np.float32),
        "meta": json.dumps({
            "source": str(source.path), "recording": source.name, "frame": int(source.frame_index[centre]),
            "speed": speed, "transform": transform.__dict__, "texture": texture.source, "contact": want_contact,
            "overlap_px": int(targets.overlap.sum()),
        }),
    }


def save_item(path: Path, item: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **item)
    temporary.replace(path)


# --------------------------------------------------------------------------- generation


TEXTURES_PER_WORKER = 12


def _render_batch(job: tuple[str, list[int], tuple[int, ...], int, str]) -> list[dict[str, Any]]:
    workspace, ids, lags, seed, output = job
    rng = np.random.default_rng((seed, ids[0]))
    source = Source(Path(workspace))
    textures = build_textures(source, TEXTURES_PER_WORKER, rng)
    centres = source.windows(int(round(max(SPEEDS) * max(lags))) + 1)
    written = []
    for item_id in ids:
        item = None
        while item is None:
            item = make_item(source, textures, int(rng.choice(centres)), lags, rng)
        save_item(Path(output) / f"item_{item_id:06d}.npz", item)
        written.append({"id": item_id, **json.loads(str(item["meta"]))})
    return written


def generate(
    workspaces: Sequence[Path], output: Path, items: int, lags: Sequence[int], *, processes: int = 16, seed: int = 0,
    batch: int = 25,
) -> dict[str, Any]:
    """Render ``items`` items, split evenly over ``workspaces``, into ``output`` (with ``manifest.json``)."""

    from multiprocessing import get_context

    from .model_training import _git_commit

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for item_id in range(0, items, batch):
        ids = list(range(item_id, min(items, item_id + batch)))
        jobs.append((str(workspaces[(item_id // batch) % len(workspaces)]), ids, tuple(int(l) for l in lags), seed, str(output)))
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    with get_context("spawn").Pool(processes) as pool:
        for done in pool.imap_unordered(_render_batch, jobs):
            records.extend(done)
            print(f"{len(records)}/{items} items, {time.monotonic() - started:.0f} s", flush=True)
    records.sort(key=lambda r: r["id"])
    manifest = {
        "items": len(records), "lags": list(lags), "seed": seed, "workspaces": [str(w) for w in workspaces],
        "git": _git_commit(), "seconds": round(time.monotonic() - started, 1),
        "settings": {"U_MAX": U_MAX, "OVERLAP_GAMMA": OVERLAP_GAMMA, "SPEEDS": SPEEDS, "CONTACT_SHARE": CONTACT_SHARE,
                     "MIN_SOURCE_IOU": MIN_SOURCE_IOU, "MIN_BEND_RADIUS_DIAMETERS": MIN_BEND_RADIUS_DIAMETERS},
        "contact_items": sum(r["contact"] for r in records), "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# --------------------------------------------------------------------------- training data


class SyntheticClipDataset:
    """Rendered items as body-field training items, in the layout of :class:`model_training.LabelDataset`."""

    def __init__(
        self, directory: str | Path, lags: Sequence[int], *, augment: bool = True, crop_size: int | None = None, seed: int = 0,
    ) -> None:
        self.directory = Path(directory)
        self.paths = sorted(self.directory.glob("item_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no synthetic items in {self.directory}")
        self.lags = tuple(int(lag) for lag in lags)
        self.augment = augment
        self.crop_size = crop_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        from .body_net import TARGET_ROWS, _augment, body_target_rows
        from .segmenter import INPUT_MEAN, INPUT_STD
        from .temporal_context import difference_channels

        path = self.paths[index]
        with np.load(path) as archive:
            data = {k: archive[k] for k in archive.files}
        stored = tuple(int(lag) for lag in data["lags"])
        if not set(self.lags) <= set(stored):
            raise ValueError(f"{path.name} holds lags {stored}; the model needs {self.lags}")
        targets = body_target_rows(data["mask"], {
            "ap": data["ap"].astype(np.float32), "overlap": data["overlap"], "head_xy": data["head_xy"],
            "tail_xy": data["tail_xy"], "diameter_px": float(data["diameter_px"]),
        })
        frames = data["frames"].astype(np.float32)
        if self.augment:
            rng = np.random.default_rng((self.seed, index, int(torch.initial_seed()) % (1 << 31)))
            frames, targets = _augment(frames, targets, rng, self.crop_size)
        centre = len(stored)
        max_lag = max(stored)
        stack = np.zeros((2 * max_lag + 1, *frames.shape[1:]), dtype=np.float32)
        valid = np.zeros(2 * max_lag + 1, dtype=bool)
        for k, lag in enumerate(stored):
            for sign, position in ((1, centre + 1 + k), (-1, centre - 1 - k)):
                stack[max_lag + sign * lag] = frames[position]
                valid[max_lag + sign * lag] = True
        stack[max_lag], valid[max_lag] = frames[centre], True
        image = ((frames[centre] / 255.0 - INPUT_MEAN) / INPUT_STD)[None]
        inputs = np.concatenate((image, difference_channels(stack, valid, self.lags)), axis=0)
        assert targets.shape[0] == TARGET_ROWS
        return {"key": f"synthetic:{path.stem}", "image": torch.as_tensor(inputs), "targets": torch.as_tensor(targets)}


class MixedSampler:
    """Every real item once per epoch plus synthetic items making up ``fraction`` of the epoch, shuffled together.

    Indices refer to ``ConcatDataset([real, synthetic])``; synthetic items are
    drawn without replacement until all have been seen, then the cycle repeats.
    """

    def __init__(self, real: int, synthetic: int, fraction: float, seed: int = 0) -> None:
        if not 0.0 <= fraction < 1.0:
            raise ValueError("the synthetic fraction must lie in [0, 1)")
        self.real, self.synthetic, self.fraction = int(real), int(synthetic), float(fraction)
        self.per_epoch = int(round(self.real * self.fraction / (1.0 - self.fraction))) if self.synthetic else 0
        self.rng = np.random.default_rng(seed)
        self._pool: list[int] = []

    def __len__(self) -> int:
        return self.real + self.per_epoch

    def __iter__(self):
        chosen = []
        while len(chosen) < self.per_epoch:
            if not self._pool:
                self._pool = list(self.rng.permutation(self.synthetic))
            chosen.append(self._pool.pop())
        order = np.concatenate((np.arange(self.real), self.real + np.asarray(chosen, dtype=np.int64)))
        return iter(self.rng.permutation(order).tolist())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", action="append", required=True, help="analysed workspace to take poses and backgrounds from")
    parser.add_argument("--output", required=True)
    parser.add_argument("--items", type=int, default=4000)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = generate([Path(w) for w in args.workspace], Path(args.output), args.items, args.lags,
                        processes=args.processes, seed=args.seed)
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}), flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
