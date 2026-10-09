"""The export: one documented table per recording, one row per frame.

An export is what an analyst takes out of the app (``docs/APP_SIMPLIFICATION.md``,
section 2, "Export task").  It is a directory ``<workspace>/exports/<name>/``,
named ``<recording>_<UTC time>``, holding ``<name>.parquet`` and
``export.json`` (model, setup, units, conventions); the Parquet file carries the
same JSON under its schema metadata key ``export.json`` and each column's
unit and description in the field metadata, so the table alone is
self-describing.  Masks, provenance and review records stay in the workspace.

Columns:

- ``frame`` (recording frame index), ``time_s`` and ``status`` (``auto``,
  ``reviewed``, ``fixed`` or ``unresolved``, see ``frame_status``),
- then the derived features: ``FEATURES`` is a list of functions
  ``(midline, width, time, meta) -> {column: Column}``, so a new feature is one
  new function appended to it.  ``midline`` is ``[n, MIDLINE_POINTS, 2]``
  head first, ``width`` the full body width at those points, both already in
  the export's length unit (µm with a pixel size, else pixels) and NaN on
  frames without a pose; ``time`` is seconds (NaN when unknown).

Coordinates are image coordinates: ``(0, 0)`` is the centre of the
upper-left pixel, x points right and y points down, scaled to µm when the
pixel size is known.  The world frame of the motion columns has the same
orientation (see ``read_recording_motion`` for how the stage enters it).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable

import h5py
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .geometry import curvature as signed_curvature, resample_centerline
from .run_records import timestamp_slug


# Points along the body, head (index 0) to tail, at uniform arc length.  The
# fitter's centerlines have 100 points (``MaskFitConfig.n_points``), so the
# default resampling loses nothing.
MIDLINE_POINTS = 100
STATUS_DEFINITIONS = {
    "auto": "the pipeline's pose, no quality flag, not looked at by a person",
    "reviewed": "the pipeline's pose, accepted by a person in review",
    "fixed": "a pose a person corrected (flip, pick, refit, or an edited mask)",
    "unresolved": "no pose, a quality flag nobody reviewed, or an edited mask not yet refit",
}
METADATA_FILE = "export.json"

# ConfocalTrackerControl.jl acquisition files (the flv-c NIR rigs).  Saved
# frames are the camera rows with ``q_iter_save`` and ``q_recording`` set
# (the camera runs at twice the saved rate); ``img_timestamp`` is the camera
# clock in nanoseconds.  ``pos_stage`` has one row per saved frame in stage
# units of 0.1 µm (BehaviorDataNIR.jl: ``STAGE_UNIT = 10000`` per mm), NaN
# where the serial read failed.
CAMERA_GROUP = "/img_metadata"
STAGE_DATASET = "/pos_stage"
TIMESTAMP_SECONDS = 1e-9
STAGE_UM_PER_UNIT = 0.1

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Column:
    """One exported column: per-frame ``values`` (``[n]`` or ``[n, k]``, NaN = no value), its unit and meaning."""

    values: np.ndarray
    unit: str | None
    description: str


@dataclass(frozen=True)
class Meta:
    """What the feature functions need besides the body: the length unit and the stage, if any.

    ``stage`` is ``[n, 2]`` in the length unit: subtracting it from an image
    position gives the position in the world frame (axes parallel to the
    image's).  ``None`` means positions and velocities stay in the image frame.
    """

    length_unit: str
    stage: FloatArray | None


Feature = Callable[[FloatArray, FloatArray, FloatArray, Meta], dict[str, Column]]


# ---------------------------------------------------------------------------
# Features


def midline_points(midline: FloatArray, width: FloatArray, time: FloatArray, meta: Meta) -> dict[str, Column]:
    return {
        "midline_x": Column(midline[..., 0], meta.length_unit, f"x of {MIDLINE_POINTS} midline points, head to tail, uniform in arc length"),
        "midline_y": Column(midline[..., 1], meta.length_unit, f"y of {MIDLINE_POINTS} midline points, head to tail, uniform in arc length"),
    }


def curvature(midline: FloatArray, width: FloatArray, time: FloatArray, meta: Meta) -> dict[str, Column]:
    values = np.full(midline.shape[:2], np.nan)
    steps = np.linalg.norm(np.diff(midline, axis=1), axis=2)
    rows = np.flatnonzero(np.isfinite(steps).all(axis=1) & (steps > 0).all(axis=1))
    if len(rows):
        values[rows] = signed_curvature(torch.from_numpy(midline[rows])).numpy()
    return {
        "curvature": Column(
            values, f"rad/{meta.length_unit}",
            "signed curvature at each midline point; positive where the body turns clockwise in the image walking head to tail",
        ),
    }


def width_profile(midline: FloatArray, width: FloatArray, time: FloatArray, meta: Meta) -> dict[str, Column]:
    return {"width": Column(width, meta.length_unit, "full body width at each midline point, head to tail")}


def head_tail(midline: FloatArray, width: FloatArray, time: FloatArray, meta: Meta) -> dict[str, Column]:
    return {
        "head_x": Column(midline[:, 0, 0], meta.length_unit, "x of the head end of the midline"),
        "head_y": Column(midline[:, 0, 1], meta.length_unit, "y of the head end of the midline"),
        "tail_x": Column(midline[:, -1, 0], meta.length_unit, "x of the tail end of the midline"),
        "tail_y": Column(midline[:, -1, 1], meta.length_unit, "y of the tail end of the midline"),
    }


def centroid(midline: FloatArray, width: FloatArray, time: FloatArray, meta: Meta) -> dict[str, Column]:
    center = midline.mean(axis=1)
    return {
        "centroid_x": Column(center[:, 0], meta.length_unit, "x of the mean of the midline points (image frame)"),
        "centroid_y": Column(center[:, 1], meta.length_unit, "y of the mean of the midline points (image frame)"),
    }


def motion(midline: FloatArray, width: FloatArray, time: FloatArray, meta: Meta) -> dict[str, Column]:
    """The centroid in the world frame and its velocity (central differences over the neighbouring frames)."""

    world = midline.mean(axis=1)
    frame = "image"
    if meta.stage is not None:
        world = world - meta.stage
        frame = "world"
    velocity = np.full_like(world, np.nan)
    if len(world) >= 2 and np.isfinite(time).all():
        velocity = np.gradient(world, time, axis=0)
    rate = f"{meta.length_unit}/s"
    return {
        "centroid_world_x": Column(world[:, 0], meta.length_unit, f"x of the centroid in the {frame} frame"),
        "centroid_world_y": Column(world[:, 1], meta.length_unit, f"y of the centroid in the {frame} frame"),
        "velocity_x": Column(velocity[:, 0], rate, f"x velocity of the centroid in the {frame} frame"),
        "velocity_y": Column(velocity[:, 1], rate, f"y velocity of the centroid in the {frame} frame"),
        "speed": Column(np.hypot(velocity[:, 0], velocity[:, 1]), rate, f"speed of the centroid in the {frame} frame"),
    }


FEATURES: tuple[Feature, ...] = (midline_points, curvature, width_profile, head_tail, centroid, motion)


# ---------------------------------------------------------------------------
# Inputs


def body_geometry(centerline_xy: np.ndarray, width_profile_px: np.ndarray, fitted: np.ndarray, scale: float) -> tuple[FloatArray, FloatArray]:
    """The stored centerlines and widths resampled to ``MIDLINE_POINTS`` and scaled by ``scale`` (length units per pixel).

    Rows without a pose (not fitted, non-finite, or with coincident points)
    are NaN.  Widths are interpolated at the resampled points' arc-length
    fractions; the stored ones sit at uniform fractions too.
    """

    curves = np.asarray(centerline_xy, dtype=np.float64)
    profile = np.asarray(width_profile_px, dtype=np.float64)
    n = len(curves)
    midline = np.full((n, MIDLINE_POINTS, 2), np.nan)
    width = np.full((n, MIDLINE_POINTS), np.nan)
    if n == 0:
        return midline, width
    steps = np.linalg.norm(np.diff(curves, axis=1), axis=2)
    rows = np.flatnonzero(np.asarray(fitted, dtype=bool) & np.isfinite(curves).all(axis=(1, 2)) & (steps > 0).all(axis=1))
    if len(rows):
        midline[rows] = resample_centerline(torch.from_numpy(curves[rows]), MIDLINE_POINTS).numpy() * scale
        stored = np.linspace(0.0, 1.0, profile.shape[1])
        wanted = np.linspace(0.0, 1.0, MIDLINE_POINTS)
        width[rows] = np.stack([np.interp(wanted, stored, profile[row]) for row in rows]) * scale
    return midline, width


def frame_status(fitted: np.ndarray, flagged: np.ndarray, stale: np.ndarray, fixed: np.ndarray, reviewed: np.ndarray) -> np.ndarray:
    """Per frame: ``unresolved``, ``fixed``, ``reviewed`` or ``auto``, in that order of precedence.

    A frame whose mask was edited but not refit (``stale``) is unresolved: its
    pose does not reflect the edit.  Otherwise a frame a person changed is
    ``fixed``, one a person looked at and accepted is ``reviewed``, and of the
    rest those without a pose or with a quality flag are ``unresolved`` and
    the others ``auto``.
    """

    status = np.full(len(fitted), "auto", dtype=object)
    status[~np.asarray(fitted, dtype=bool) | np.asarray(flagged, dtype=bool)] = "unresolved"
    status[np.asarray(reviewed, dtype=bool)] = "reviewed"
    status[np.asarray(fixed, dtype=bool)] = "fixed"
    status[np.asarray(stale, dtype=bool)] = "unresolved"
    return status


def read_recording_motion(recording: Path, dataset: str, frames: np.ndarray) -> tuple[FloatArray | None, FloatArray | None, dict[str, Any]]:
    """``(time_s, stage_um, notes)`` of ``frames`` from the recording's acquisition metadata; either is ``None`` when absent.

    ``time_s`` is the camera timestamp relative to the recording's first saved
    frame.  It is used only when the saved-frame rows of the camera metadata
    match the frame count of ``dataset`` one to one.

    ``stage_um`` is the stage position at each frame's exposure, in µm, such
    that ``image position - stage_um`` is the world position, with axes
    parallel to the image's.  On the flv-c rigs stage x and y run along image
    x and y, and a worm standing still in the world moves by ``+d`` in the
    image when the stage moves by ``+d`` (checked on the fitted centroids of
    2024-01-31-02: of the eight assignments of stage axes and signs to image
    axes, only this one makes the world centroid smooth).  Stage
    samples are read about half a saved frame after the exposure
    (ConfocalTrackerControl's stage loop queries after the image grab and
    sleeps 25 ms), so the position at the exposure is the mean of the frame's
    sample and the one before.  Failed reads (NaN, typically 1–4 frames in a
    row) are linearly interpolated in time, as BehaviorDataNIR.jl does.
    """

    notes: dict[str, Any] = {}
    frames = np.asarray(frames, dtype=np.int64)
    try:
        with h5py.File(recording, "r") as handle:
            total = int(handle[dataset].shape[0])
            times = _saved_frame_times(handle, total, notes)
            stage = _stage(handle, total, times, notes)
    except (OSError, KeyError) as error:
        notes["error"] = f"recording unreadable: {type(error).__name__}: {error}"
        return None, None, notes
    if len(frames) and (frames.min() < 0 or frames.max() >= total):
        raise ValueError(f"frames outside the {total} frames of {recording}")
    return (None if times is None else times[frames]), (None if stage is None else stage[frames]), notes


def _saved_frame_times(handle: h5py.File, total: int, notes: dict[str, Any]) -> FloatArray | None:
    group = handle.get(CAMERA_GROUP)
    if group is None or not all(key in group for key in ("q_iter_save", "q_recording", "img_timestamp")):
        notes["time"] = f"no {CAMERA_GROUP} timestamps"
        return None
    saved = np.flatnonzero((np.asarray(group["q_iter_save"]) == 1) & (np.asarray(group["q_recording"]) == 1))
    if len(saved) != total:
        notes["time"] = f"{len(saved)} saved camera rows for {total} frames"
        return None
    stamps = np.asarray(group["img_timestamp"])[saved].astype(np.int64)
    seconds = (stamps - stamps[0]).astype(np.float64) * TIMESTAMP_SECONDS
    if total > 1 and not (np.diff(seconds) > 0).all():
        notes["time"] = "camera timestamps are not increasing"
        return None
    return seconds


def _stage(handle: h5py.File, total: int, times: FloatArray | None, notes: dict[str, Any]) -> FloatArray | None:
    if STAGE_DATASET not in handle:
        notes["stage"] = f"no {STAGE_DATASET}"
        return None
    raw = np.asarray(handle[STAGE_DATASET], dtype=np.float64)
    if raw.shape != (total, 2):
        notes["stage"] = f"{STAGE_DATASET} has shape {raw.shape}, expected ({total}, 2)"
        return None
    valid = np.isfinite(raw).all(axis=1)
    if valid.sum() < 2:
        notes["stage"] = f"{STAGE_DATASET} has fewer than two readings"
        return None
    clock = times if times is not None else np.arange(total, dtype=np.float64)
    filled = np.stack([np.interp(clock, clock[valid], raw[valid, axis]) for axis in range(2)], axis=1)
    notes["stage_interpolated_samples"] = int((~valid).sum())
    at_exposure = filled.copy()
    at_exposure[1:] = 0.5 * (filled[1:] + filled[:-1])
    return at_exposure * STAGE_UM_PER_UNIT


# ---------------------------------------------------------------------------
# The table


def build_table(frames: np.ndarray, time_s: FloatArray, status: np.ndarray, midline: FloatArray, width: FloatArray, meta: Meta) -> tuple[pa.Table, dict[str, dict[str, Any]]]:
    """The export table and the documentation of its columns (unit, dtype, description), in column order."""

    columns: dict[str, Column] = {
        "frame": Column(np.asarray(frames, dtype=np.int64), None, "recording frame index (0-based)"),
        "time_s": Column(np.asarray(time_s, dtype=np.float64), "s", "time since the recording's first frame"),
        "status": Column(np.asarray(status, dtype=object), None, "auto | reviewed | fixed | unresolved (export.json: status)"),
    }
    for feature in FEATURES:
        for name, column in feature(midline, width, time_s, meta).items():
            if name in columns:
                raise ValueError(f"feature {feature.__name__} redefines column {name!r}")
            columns[name] = column
    has_pose = np.isfinite(midline).all(axis=(1, 2))
    fields, arrays, documentation = [], [], {}
    for name, column in columns.items():
        array = _arrow(column.values, has_pose)
        metadata = {"description": column.description, **({"unit": column.unit} if column.unit else {})}
        fields.append(pa.field(name, array.type, metadata=metadata))
        arrays.append(array)
        documentation[name] = {"unit": column.unit, "dtype": str(array.type), "description": column.description}
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields)), documentation


def _arrow(values: np.ndarray, has_pose: np.ndarray) -> pa.Array:
    if values.dtype == object:
        return pa.array(values.tolist(), pa.string()).dictionary_encode()
    if values.dtype.kind in "iub":
        return pa.array(values)
    if values.ndim == 2:
        # Per-point values: float32 lists (sub-nanometre at worm scales), null on frames without a pose.
        # Variable-size lists: Parquet does not read back null fixed-size lists (pyarrow 25).
        flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).ravel())
        offsets = pa.array(np.arange(len(values) + 1, dtype=np.int32) * values.shape[1])
        return pa.ListArray.from_arrays(offsets, flat, mask=pa.array(~has_pose))
    values = np.asarray(values, dtype=np.float64)
    # NaN means "no value"; Parquet readers expect null for that.
    return pa.array(values, mask=np.isnan(values))


def export(
    exports_dir: Path,
    *,
    recording: Path,
    dataset: str,
    frames: np.ndarray,
    centerline_xy: np.ndarray,
    width_profile_px: np.ndarray,
    fitted: np.ndarray,
    status: np.ndarray,
    pixel_size_um: float | None,
    fps: float | None,
    about: dict[str, Any],
) -> dict[str, Any]:
    """Build and write one export named ``<recording>_<UTC time>``; returns its ``export.json`` contents and ``path`` (the table).

    ``about`` is the caller's part of ``export.json`` (workspace, setup,
    models, app revision, creation time).  Lengths are in µm when
    ``pixel_size_um`` is given, else in pixels.  Time comes from the camera
    timestamps, else from ``fps``, else it is unknown (null).  Velocity is in
    the world frame when the recording has stage positions and the pixel size
    is known, else in the image frame; ``export.json``'s ``velocity`` says
    which and why.
    """

    frames = np.asarray(frames, dtype=np.int64)
    unit, scale = ("um", float(pixel_size_um)) if pixel_size_um else ("px", 1.0)
    midline, width = body_geometry(centerline_xy, width_profile_px, fitted, scale)
    stamps, stage, notes = read_recording_motion(recording, dataset, frames)
    if stamps is not None:
        time_s = stamps
        time_info: dict[str, Any] = {"source": "camera timestamps", "dataset": f"{CAMERA_GROUP}/img_timestamp", "origin": "the recording's first saved frame"}
        if len(stamps) > 1:
            time_info["measured_fps"] = float((len(stamps) - 1) / (stamps[-1] - stamps[0]))
    elif fps:
        time_s = frames / float(fps)
        time_info = {"source": "frame rate", "fps": float(fps), "reason": notes.get("time") or notes.get("error")}
    else:
        time_s = np.full(len(frames), np.nan)
        time_info = {"source": "none", "reason": f"{notes.get('time') or notes.get('error')}, and no frame rate was given"}
    if stage is not None and pixel_size_um:
        meta = Meta(unit, stage)
        velocity = {
            "frame": "world",
            "stage_dataset": STAGE_DATASET, "stage_um_per_unit": STAGE_UM_PER_UNIT,
            "stage_interpolated_samples": notes.get("stage_interpolated_samples", 0),
            "convention": "world = image position - stage position; axes parallel to the image's (x right, y down)",
        }
    else:
        meta = Meta(unit, None)
        reason = notes.get("stage") or notes.get("error") or "no pixel size, so the stage (µm) and the image (pixels) cannot be combined"
        velocity = {"frame": "image", "reason": reason}
    velocity["method"] = "central differences of centroid_world over the neighbouring frames (one-sided at the ends)"
    table, columns = build_table(frames, time_s, status, midline, width, meta)
    values, counts = np.unique(np.asarray(status, dtype=str), return_counts=True)
    metadata = {
        **about,
        "recording": str(recording),
        "frames": {"first": int(frames[0]), "last": int(frames[-1]), "count": int(len(frames))} if len(frames) else None,
        "rows": table.num_rows,
        "pixel_size_um": pixel_size_um,
        "fps": fps,
        "length_unit": unit,
        "coordinates": "image coordinates: (0, 0) is the centre of the upper-left pixel, x right, y down",
        "midline_points": MIDLINE_POINTS,
        "time": time_info,
        "velocity": velocity,
        "status": {"counts": {str(v): int(c) for v, c in zip(values, counts)}, "definitions": STATUS_DEFINITIONS},
        "columns": columns,
    }
    directory = write_export(Path(exports_dir), f"{Path(recording).stem}_{timestamp_slug()}", table, metadata)
    written = json.loads((directory / METADATA_FILE).read_text())
    return {**written, "path": str(directory / written["table"])}


def write_export(exports_dir: Path, name: str, table: pa.Table, metadata: dict[str, Any]) -> Path:
    """Write ``<exports_dir>/<name>/`` (``<name>.parquet`` and ``export.json``); returns the directory.

    The files are written into a hidden temporary directory that is renamed
    into place, so a listed export is always complete.  A name already taken
    gets a ``-2``, ``-3`` ... suffix (two exports in the same second).
    """

    exports_dir.mkdir(parents=True, exist_ok=True)
    final, suffix = exports_dir / name, 1
    while final.exists():
        suffix += 1
        final = exports_dir / f"{name}-{suffix}"
    metadata = {**metadata, "name": final.name, "table": f"{final.name}.parquet"}
    staging = exports_dir / f".{final.name}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        text = json.dumps(metadata, indent=1)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), METADATA_FILE.encode(): text.encode()})
        pq.write_table(table, staging / metadata["table"])
        (staging / METADATA_FILE).write_text(text)
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final
