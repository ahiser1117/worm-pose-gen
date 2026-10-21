"""Stage position to image coordinates: where the substrate is in each frame.

The flv-c rigs track the worm by moving the stage, so the agar moves
through the camera image.  ``/pos_stage`` holds one stage reading per saved
frame (units of 0.1 µm, NaN where the serial read failed, 8-24% of rows).
A ``StageCalibration`` maps a stage displacement to the image displacement
of the background, ``matrix @ (stage[t1] - stage[t0])`` pixels, with the
reading for frame t taken at fractional frame ``t + offset_frames``.  It is
estimated per recording by registering the background of frame pairs at
lags 1, 4 and 16 (worm masked out) and fitting the matrix and the offset to
the stage displacements (docs/STAGE_ALIGNMENT.md has the method and the
measurements: about 0.078 px per unit on both axes, axes and signs as the
image's, no rotation, offset -0.7 to -0.8 frames, 1.1-1.35 px RMS residual).

``substrate_shift_px`` gives each frame's image-to-substrate offset;
``align_frame`` moves one frame's content into another's camera position.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from .recordings import RecordingSource
from .settings import CACHE_DIR


STAGE_DATASET = "/pos_stage"
DEFAULT_CACHE = CACHE_DIR / "stage_calibrations"
MAX_GAP_FRAMES = 4
BANDPASS_SIGMA_PX = 1.0
HIGHPASS_SIGMA_PX = 6.0
MASK_DILATION_PX = 12
MASK_TAPER_PX = 4.0
DARK_MARGIN = 15.0
CALIBRATION_LAGS = (1, 4, 16)
CALIBRATION_ANCHORS = 160
CALIBRATION_OFFSETS = tuple(np.round(np.arange(-2.0, 2.0001, 0.05), 2))
OUTLIER_PX = 3.0
MIN_SCORE = 15.0
PATTERN_FRAMES = 64

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class StageTrack:
    """A recording's stage positions (stage units), one row per saved frame.

    ``raw`` holds NaN where the serial read failed; ``filled`` interpolates
    those linearly in frame index and holds the nearest reading before the
    first and after the last one.  ``gap`` is the length of the run of failed
    reads a frame lies in (0 for a reading); ``reliable`` is false inside runs
    longer than ``MAX_GAP_FRAMES`` and outside the first and last reading.
    """

    raw: FloatArray
    filled: FloatArray
    gap: NDArray[np.int64]
    reliable: BoolArray

    def at(self, frames: ArrayLike) -> FloatArray:
        """Stage at fractional frame indices, linear between frames; NaN where unreliable or outside."""

        frames = np.asarray(frames, dtype=np.float64)
        n = len(self.filled)
        index = np.arange(n, dtype=np.float64)
        out = np.stack([np.interp(frames, index, self.filled[:, axis]) for axis in range(2)], axis=-1)
        lo = np.clip(np.floor(frames), 0, n - 1).astype(np.int64)
        hi = np.clip(np.ceil(frames), 0, n - 1).astype(np.int64)
        bad = (frames < 0) | (frames > n - 1) | ~self.reliable[lo] | ~self.reliable[hi]
        out[bad] = np.nan
        return out


def clean_stage(raw: ArrayLike, max_gap: int = MAX_GAP_FRAMES) -> StageTrack:
    """``StageTrack`` of ``[n, 2]`` stage readings with NaN rows for failed reads."""

    raw = np.asarray(raw, dtype=np.float64)
    valid = np.isfinite(raw).all(axis=1)
    if valid.sum() < 2:
        raise ValueError("fewer than two stage readings")
    n = len(raw)
    index = np.arange(n)
    filled = np.stack([np.interp(index, index[valid], raw[valid, axis]) for axis in range(2)], axis=1)
    gap = np.zeros(n, dtype=np.int64)
    edges = np.flatnonzero(np.diff(np.concatenate([[1], valid.astype(np.int8), [1]])))
    for start, stop in zip(edges[::2], edges[1::2]):
        gap[start:stop] = stop - start
    first, last = np.flatnonzero(valid)[[0, -1]]
    reliable = gap <= max_gap
    reliable[:first] = False
    reliable[last + 1 :] = False
    return StageTrack(raw=raw, filled=filled, gap=gap, reliable=reliable)


def read_stage(recording: Path | str, max_gap: int = MAX_GAP_FRAMES) -> StageTrack:
    with h5py.File(recording, "r") as handle:
        if STAGE_DATASET not in handle:
            raise KeyError(f"{recording}: no {STAGE_DATASET}")
        return clean_stage(np.asarray(handle[STAGE_DATASET]), max_gap)


@dataclass(frozen=True)
class StageCalibration:
    """Stage displacement (units) to image displacement of the substrate (pixels, x right, y down).

    ``matrix @ (stage[t1] - stage[t0])`` is how far the background moves in
    the image from frame t0 to t1; the stage reading for frame t is taken at
    fractional frame ``t + offset_frames``.  ``residual_px`` is the RMS of
    measured minus predicted background shift over ``pairs`` frame pairs.
    """

    matrix: tuple[tuple[float, float], tuple[float, float]]
    offset_frames: float
    residual_px: float
    pairs: int
    outliers: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageCalibration":
        values = dict(data)
        values["matrix"] = tuple(tuple(float(v) for v in row) for row in values["matrix"])
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StageCalibration":
        return cls.from_dict(json.loads(Path(path).read_text()))


def bandpass(image: NDArray[Any]) -> FloatArray:
    """The image smoothed by ``BANDPASS_SIGMA_PX`` minus its ``HIGHPASS_SIGMA_PX`` blur: specks and grain of the agar, no pixel noise or illumination."""

    image = np.asarray(image, dtype=np.float64)
    return ndimage.gaussian_filter(image, BANDPASS_SIGMA_PX) - ndimage.gaussian_filter(image, HIGHPASS_SIGMA_PX)


def register_shift(moving: FloatArray, reference: FloatArray, invalid: BoolArray | None = None) -> tuple[FloatArray, float]:
    """``(shift_xy, score)`` with ``moving(p) ≈ reference(p - shift)``, from two band-passed images.

    Cross-correlation with the cross-power spectrum divided by the square root
    of its magnitude (half way to phase correlation), under a Hann window
    times the valid pixels; ``invalid`` (worm in either frame) is tapered over
    ``MASK_TAPER_PX`` so its edge does not correlate with itself at zero shift.
    ``score`` is the peak over the correlation surface's standard deviation.
    The parabola through the peak and its neighbours gives the sub-pixel part.
    """

    h, w = moving.shape
    keep = np.outer(np.hanning(h), np.hanning(w))
    if invalid is not None:
        keep = keep * np.clip(ndimage.gaussian_filter((~invalid).astype(np.float64), MASK_TAPER_PX), 0.0, 1.0) * ~invalid
    cross = np.fft.rfft2(moving * keep) * np.conj(np.fft.rfft2(reference * keep))
    cross /= np.sqrt(np.abs(cross)) + 1e-12
    corr = np.fft.fftshift(np.fft.irfft2(cross, s=(h, w)))
    corr /= corr.std()
    cy, cx = h // 2, w // 2
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[py, px])

    def refine(c_minus: float, c0: float, c_plus: float) -> float:
        denominator = c_minus - 2 * c0 + c_plus
        return 0.0 if denominator >= 0 else 0.5 * (c_minus - c_plus) / denominator

    dy = refine(corr[py - 1, px], peak, corr[py + 1, px]) if 0 < py < h - 1 else 0.0
    dx = refine(corr[py, px - 1], peak, corr[py, px + 1]) if 0 < px < w - 1 else 0.0
    return np.array([px - cx + dx, py - cy + dy]), peak


def dark_mask(image: NDArray[Any]) -> BoolArray:
    """Pixels darker than the frame's median by ``DARK_MARGIN`` after a 2 px blur: the worm, without a segmenter.

    On 12 frames of 2023-03-30-01 this mask, dilated like the workspace
    masks, covers 98-100% of the workspace's worm pixels and 2% of the rest.
    """

    smooth = ndimage.gaussian_filter(np.asarray(image, dtype=np.float64), 2.0)
    return smooth < np.median(smooth) - DARK_MARGIN


def measure_shifts(
    recording: Path | str,
    anchors: Sequence[int],
    lags: Sequence[int] = CALIBRATION_LAGS,
    mask_of: Callable[[int], BoolArray | None] | None = None,
    flat_field_dir: Path = CACHE_DIR / "flat_fields",
) -> FloatArray:
    """``[n, 5]`` rows ``(t, lag, dx, dy, score)``: the background shift from frame t to t + lag of flat-fielded frames.

    ``mask_of(frame)`` gives the worm's pixels (``dark_mask`` of the frame
    when ``None``; dilated by ``MASK_DILATION_PX`` and left out of both
    frames).  The camera adds a fixed pattern (interference rings, dust) that
    correlates at zero shift as strongly as the agar's sparse texture at the
    true one: unremoved, 8-12% of the pairs of 2023-03-30-01 and most of
    the second half of 2023-08-22-01 register at zero.  The pattern is the
    median of band-passed anchor frames (worm left out), estimated twice from
    alternate frames, one estimate subtracted from each frame of a pair;
    one estimate subtracted from both adds the same noise to both, and then
    most pairs register at zero.
    """

    source = RecordingSource(Path(recording), Path(flat_field_dir))
    cache: dict[int, tuple[FloatArray, BoolArray]] = {}

    def frame(index: int) -> tuple[FloatArray, BoolArray]:
        if index not in cache:
            image = source.corrected(index)[1]
            mask = None if mask_of is None else mask_of(index)
            mask = dark_mask(image) if mask is None else mask
            cache[index] = (bandpass(image), ndimage.binary_dilation(mask, iterations=MASK_DILATION_PX))
        return cache[index]

    rows = []
    try:
        samples = np.asarray(anchors)[np.unique(np.linspace(0, len(anchors) - 1, PATTERN_FRAMES).astype(np.int64))]
        stack = np.empty((len(samples), *source.shape), dtype=np.float32)
        for row, index in enumerate(samples):
            image, mask = frame(int(index))
            stack[row] = np.where(mask, np.nan, image)
            cache.clear()
        # Two estimates from alternate samples: subtracting one estimate from both frames would add its noise to both,
        # a common component that correlates at zero shift like the pattern itself.
        pattern_a = np.nan_to_num(np.nanmedian(stack[0::2], axis=0))
        pattern_b = np.nan_to_num(np.nanmedian(stack[1::2], axis=0))
        del stack
        for t in anchors:
            cache.clear()
            a, mask_a = frame(int(t))
            for lag in lags:
                if t + lag >= source.frame_count:
                    continue
                b, mask_b = frame(int(t + lag))
                shift, score = register_shift(b - pattern_b, a - pattern_a, mask_a | mask_b)
                rows.append([t, lag, shift[0], shift[1], score])
    finally:
        source.close()
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def fit_calibration(
    track: StageTrack, shifts: FloatArray, offsets: Sequence[float] = CALIBRATION_OFFSETS, outlier_px: float = OUTLIER_PX
) -> StageCalibration:
    """The matrix and stage time offset that best predict measured background shifts (``measure_shifts`` rows).

    Pairs whose registration scores under ``MIN_SCORE`` are dropped (a third
    of 2023-08-22-01's pairs, on a background with almost no texture, and
    44% of those miss the stage prediction by over 3 px; with them its
    scale comes out 4% low).  For each offset the 2x2 matrix is fit by least
    squares to the stage displacement over each pair, refit three times on
    the pairs whose residual is under ``outlier_px`` (the worm or a texture-
    less background matched, or stage reads that miss a fast jump); the
    offset with the lowest truncated squared error wins.
    """

    shifts = shifts[shifts[:, 4] >= MIN_SCORE]
    t, lag, measured = shifts[:, 0], shifts[:, 1], shifts[:, 2:4]
    best: tuple[float, StageCalibration] | None = None
    for offset in offsets:
        displacement = track.at(t + lag + offset) - track.at(t + offset)
        usable = np.isfinite(displacement).all(axis=1)
        keep = usable.copy()
        for _ in range(4):
            if keep.sum() < 4:
                break
            solution, *_ = np.linalg.lstsq(displacement[keep], measured[keep], rcond=None)
            residual = np.linalg.norm(measured - displacement @ solution, axis=1)
            keep = usable & (residual < outlier_px)
        if keep.sum() < 4:
            continue
        loss = float(np.mean(np.minimum(residual[usable], outlier_px) ** 2))
        calibration = StageCalibration(
            matrix=tuple(tuple(float(v) for v in row) for row in solution.T),
            offset_frames=float(offset),
            residual_px=float(np.sqrt(np.mean(residual[keep] ** 2))),
            pairs=int(keep.sum()),
            outliers=int(usable.sum() - keep.sum()),
        )
        if best is None or loss < best[0]:
            best = (loss, calibration)
    if best is None:
        raise ValueError("too few frame pairs with stage readings to calibrate")
    return best[1]


def estimate_calibration(
    recording: Path | str,
    mask_of: Callable[[int], BoolArray | None] | None = None,
    anchor_count: int = CALIBRATION_ANCHORS,
    flat_field_dir: Path = CACHE_DIR / "flat_fields",
) -> StageCalibration:
    """Calibrate from ``anchor_count`` frames spread over the recording, each registered at ``CALIBRATION_LAGS``."""

    track = read_stage(recording)
    reliable = np.flatnonzero(track.reliable)
    anchors = np.unique(np.linspace(reliable[0], reliable[-1] - max(CALIBRATION_LAGS), anchor_count).astype(np.int64))
    return fit_calibration(track, measure_shifts(recording, anchors, CALIBRATION_LAGS, mask_of, flat_field_dir))


def recording_calibration(
    recording: Path | str, mask_of: Callable[[int], BoolArray | None] | None = None, cache_dir: Path = DEFAULT_CACHE
) -> StageCalibration:
    """The recording's calibration from ``cache_dir/<stem>.json``, estimated and written there on first use."""

    path = Path(cache_dir) / f"{Path(recording).stem}.json"
    if path.exists():
        return StageCalibration.load(path)
    calibration = estimate_calibration(recording, mask_of)
    calibration.save(path)
    return calibration


def shift_image(image: NDArray[Any], shift_xy: Sequence[float], order: int = 1) -> tuple[FloatArray, BoolArray]:
    """``(shifted, valid)``: image content moved by ``shift_xy`` (sub-pixel); ``valid`` is false where it came from outside.

    ``order`` 1 interpolates linearly, which averages pixel noise at
    fractional shifts: on 2023-03-30-01 it alone halves the mean square of a
    background difference (78 to 35 grey levels²).  Round the shift for a
    difference channel whose noise level must not depend on the offset.
    """

    image = np.asarray(image, dtype=np.float64)
    dx, dy = float(shift_xy[0]), float(shift_xy[1])
    shifted = ndimage.shift(image, (dy, dx), order=order, mode="constant", cval=0.0, prefilter=order > 1)
    h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    valid = (yy - dy >= 0) & (yy - dy <= h - 1) & (xx - dx >= 0) & (xx - dx <= w - 1)
    return shifted, valid


def align_frame(image: NDArray[Any], source_shift: Sequence[float], target_shift: Sequence[float]) -> tuple[FloatArray, BoolArray]:
    """Frame ``image`` (substrate shift ``source_shift``) moved into the camera position of a frame with ``target_shift``."""

    return shift_image(image, np.asarray(source_shift, dtype=np.float64) - np.asarray(target_shift, dtype=np.float64))


def substrate_shift_px(recording: Path | str, frame_indices: Sequence[int], calibration: StageCalibration) -> FloatArray:
    """``[n, 2]`` offsets (x, y) in pixels: image position + offset = substrate position, NaN where the stage is unknown.

    Substrate coordinates are image pixels of the recording's first reliable
    frame, so a point fixed in the agar keeps its substrate position while
    the stage moves.  The offset of frame t is ``-matrix @ (stage(t +
    offset_frames) - stage(first))``; it is NaN where that touches a frame in
    a run of failed reads longer than ``MAX_GAP_FRAMES`` or lies outside the
    readings.  Its error is the calibration's residual (about 1 px) except
    during fast stage oscillations that the readings miss (docs/STAGE_ALIGNMENT.md).
    """

    track = read_stage(recording)
    reference = track.filled[int(np.flatnonzero(track.reliable)[0])]
    stage = track.at(np.asarray(frame_indices, dtype=np.float64) + calibration.offset_frames)
    return -(stage - reference) @ np.asarray(calibration.matrix).T
