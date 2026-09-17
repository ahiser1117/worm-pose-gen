"""A frozen recording body fitted to reviewed poses, stored independently.

Only segment angles vary per frame. Distances are measured from the reviewed
head without normalizing each frame's length. Missing tails are straight
continuations; missing middle sections are explicitly unresolved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from .workspace import _load_npz, _write_npz_atomic, utc_now

FILENAME = "fixed_body.npz"


def input_revision(path: Path) -> str:
    """Cheap revision of poses and masks, including atomic file replacements."""
    files = [path / name for name in ("state.npz", "workspace.json", "edits.jsonl")]
    files += sorted((path / "masks").glob("*.npz"))
    files += sorted((path / "overrides" / "masks").glob("*.npz"))
    digest = hashlib.sha256()
    for file in files:
        if file.exists():
            stat = file.stat()
            digest.update(f"{file.relative_to(path)}:{stat.st_mtime_ns}:{stat.st_size}:{stat.st_ino}".encode())
    return digest.hexdigest()


def inside(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return ((points >= 0) & (points <= np.array(shape[::-1]) - 1)).all(axis=-1)


def arc_samples(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-9]
    return np.column_stack([np.interp(distances, arc[keep], points[keep, axis]) for axis in (0, 1)])


def chain_targets(points: np.ndarray, length: float, segments: int, shape: tuple[int, int], *, stop_at_exit: bool = False) -> dict[str, Any]:
    """Equal arc-distance targets of a fixed-length chain along an ordered midline, cut at the camera rectangle.

    Distances are absolute from the head, so a midline longer than the model
    is truncated and a shorter one supports fewer segments.  The result has
    a ``status`` (``head_outside``, ``reentry_unresolved``,
    ``insufficient_visible_body``) or ``targets`` ``[count + 1, 2]`` with
    ``count`` supported segments, ``clipped`` and the visible ``observed_length``.
    A midline that leaves and re-enters the image is unresolved unless
    ``stop_at_exit`` cuts it at its first exit.
    """

    visible = inside(points, shape)
    if not visible[0]:
        return {"status": "head_outside"}
    outside = np.flatnonzero(~visible)
    clipped = bool(len(outside))
    if clipped and not stop_at_exit and visible[outside[0]:].any():
        return {"status": "reentry_unresolved"}
    observed = points.copy()
    if clipped:
        # Include the exact first intersection with the camera rectangle.
        stop = int(outside[0])
        a, b = points[stop - 1], points[stop]
        direction = b - a
        limits = np.array(shape[::-1], dtype=float) - 1
        fractions = [(limits[j] - a[j]) / direction[j] if direction[j] > 0 else -a[j] / direction[j]
                     for j in (0, 1) if abs(direction[j]) > 1e-12]
        boundary = a + min(1., min(fractions)) * direction
        observed = np.vstack((points[:stop], boundary))
    observed_length = float(np.linalg.norm(np.diff(observed, axis=0), axis=1).sum())
    step = length / segments
    count = min(segments, int(np.floor((observed_length + 1e-8) / step)))
    if count < 1:
        return {"status": "insufficient_visible_body"}
    return {"targets": arc_samples(observed, np.arange(count + 1) * step), "count": count, "clipped": clipped, "observed_length": observed_length}


def fit_chain(points: np.ndarray, length: float, segments: int, shape: tuple[int, int]) -> dict[str, Any]:
    """Fit equal links to ordered, absolute arc-distance targets with a fixed head."""
    sampled = chain_targets(points, length, segments, shape)
    if "status" in sampled:
        return sampled
    targets, count, clipped, observed_length = sampled["targets"], sampled["count"], sampled["clipped"], sampled["observed_length"]
    step = length / segments
    delta = np.diff(targets, axis=0)
    angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))

    def decode(theta: np.ndarray) -> np.ndarray:
        links = step * np.column_stack((np.cos(theta), np.sin(theta)))
        return np.vstack((points[0], points[0] + np.cumsum(links, axis=0)))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        residual = (decode(theta)[1:] - targets[1:]) / step
        downstream = np.cumsum(residual[::-1], axis=0)[::-1]
        derivative = np.column_stack((-np.sin(theta), np.cos(theta)))
        gradient = 2 * (downstream * derivative).sum(axis=1)
        # A small bend penalty suppresses numerical zigzags without temporal smoothing.
        bends = np.diff(theta)
        gradient[:-1] -= .002 * bends
        gradient[1:] += .002 * bends
        return float((residual ** 2).sum() + .001 * (bends ** 2).sum()), gradient

    result = minimize(objective, angles, jac=True, method="L-BFGS-B", options={"maxiter": 120, "ftol": 1e-10})
    theta = result.x if np.isfinite(result.fun) and result.fun <= objective(angles)[0] else angles
    fitted = decode(theta)
    # Never let refinement invent a bend outside the image.
    escaped = np.flatnonzero(~inside(fitted, shape))
    if len(escaped):
        count = int(escaped[0]) - 1
        fitted = fitted[:count + 1]
    if count < 1:
        return {"status": "insufficient_visible_body"}
    rms = float(np.sqrt(np.mean(np.sum((fitted - targets[:count + 1]) ** 2, axis=1))))
    chain = np.empty((segments + 1, 2))
    chain[:count + 1] = fitted
    direction = fitted[-1] - fitted[-2]
    for i in range(count + 1, segments + 1):
        chain[i] = chain[i - 1] + direction
    extrapolated = np.arange(segments + 1) > count
    return {
        "status": "extrapolated" if extrapolated.any() else "aligned",
        "centerline_xy": chain, "extrapolated": extrapolated,
        "in_fov": inside(chain, shape), "rms_px": rms,
        "length_error_px": np.nan if clipped else observed_length - length,
    }


def trusted_rows(state: dict[str, np.ndarray], min_iou: float) -> np.ndarray:
    """Rows whose current pose is fitted, current, unambiguous and overlaps its mask at least ``min_iou``."""

    n = len(state["frame_index"])
    trusted = np.asarray(state["fitted"], dtype=bool) & np.isfinite(np.asarray(state["centerline_xy"], dtype=float)).all(axis=(1, 2))
    trusted &= ~np.asarray(state.get("mask_stale", np.zeros(n)), dtype=bool)
    trusted &= np.asarray(state.get("iou", np.full(n, np.nan)), dtype=float) >= min_iou
    trusted &= np.asarray(state.get("ambiguity_score", np.full(n, np.nan)), dtype=float) < 2
    return trusted


def whole_body_rows(state: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Rows whose pose, including its width, lies inside the image with finite positive widths."""

    curves = np.asarray(state["centerline_xy"], dtype=float)
    widths = np.asarray(state["width_profile"], dtype=float)
    radius = widths / 2
    limits = np.array(shape[::-1]) - 1
    finite = np.isfinite(curves).all(axis=(1, 2)) & np.isfinite(widths).all(axis=1) & (widths > 0).all(axis=1)
    whole = np.zeros(len(curves), dtype=bool)
    whole[finite] = ((curves[finite] - radius[finite, :, None] > 0) & (curves[finite] + radius[finite, :, None] < limits)).all(axis=(1, 2))
    return whole


def calibrate_body(
    workspace: Any, state: dict[str, np.ndarray], *, samples: int, min_iou: float, min_anchors: int,
    anchor_frames: str = "", near: int | None = None, limit: int | None = None, progress: Any = None,
) -> tuple[list[int], float, np.ndarray]:
    """The recording's body: calibration rows, median length and smoothed median width profile at ``samples`` positions.

    Calibration rows are the trusted, fully visible poses whose stored mask
    does not touch the image border (``anchor_frames`` names them instead,
    and every one must qualify).  With ``near``, rows are visited from that
    row outward and ``limit`` stops after that many qualifying rows.
    """

    shape = workspace.image_shape
    if shape is None:
        raise ValueError("Fixed body needs the recording image dimensions")
    n = len(state["frame_index"])
    fitted = np.asarray(state["fitted"], dtype=bool) & np.isfinite(np.asarray(state["centerline_xy"], dtype=float)).all(axis=(1, 2))
    fitted &= ~np.asarray(state.get("mask_stale", np.zeros(n)), dtype=bool)
    curves = np.asarray(state["centerline_xy"], dtype=float)
    lengths = np.linalg.norm(np.diff(curves, axis=1), axis=2).sum(axis=1)
    eligible = fitted & (lengths > 0) & whole_body_rows(state, shape)
    requested = str(anchor_frames or "").strip()
    if requested:
        try:
            anchors = sorted({workspace.row_of(int(f.strip())) for f in requested.split(",")})
        except ValueError as error:
            raise ValueError("Anchor frames must be comma-separated frame IDs in this workspace") from error
        if not eligible[anchors].all():
            raise ValueError("Every anchor must have a current, fully visible pose and positive widths")
    else:
        eligible &= trusted_rows(state, min_iou)
        anchors = np.flatnonzero(eligible).tolist()
        if near is not None:
            anchors.sort(key=lambda row: (abs(row - near), row))
    # A model inside the image is insufficient if the actual mask is clipped.
    workspace.clear_mask_cache()
    unclipped: list[int] = []
    if progress is not None:
        progress(0., f"Fixed body: checking {len(anchors)} calibration candidates")
    for index, row in enumerate(anchors):
        if limit is not None and len(unclipped) >= limit:
            break
        if progress is not None and index % 100 == 0:
            progress(.15 * index / len(anchors), f"Fixed body: calibration {index + 1}/{len(anchors)}")
        mask = workspace.effective_mask(row)
        if mask is not None and (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
            if requested:
                raise ValueError(f"Anchor frame {workspace.frame_index[row]} has a mask touching the image border")
            continue
        unclipped.append(row)
    anchors = sorted(unclipped)
    if len(anchors) < min_anchors:
        raise ValueError(f"Fixed body needs at least {min_anchors} unambiguous, fully visible frames; found {len(anchors)}. Review poses or specify trusted anchor frame IDs.")
    length = float(np.median(lengths[anchors]))
    widths = np.asarray(state["width_profile"], dtype=float)
    profiles = []
    for row in anchors:
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curves[row], axis=0), axis=1))] / lengths[row]
        profiles.append(np.interp(np.linspace(0, 1, samples), arc, widths[row]))
    profile = gaussian_filter1d(np.median(profiles, axis=0), sigma=1., mode="nearest")
    return anchors, length, profile


def run_fixed_body(workspace: Any, params: Any, *, progress: Any = None, job: str = "") -> dict[str, Any]:
    """Called under the pipeline workspace lock; publish one atomic derived file."""
    if not isinstance(params.segments, int) or not 4 <= params.segments <= 200:
        raise ValueError("segments must be between 4 and 200")
    if (not isinstance(params.min_iou, (int, float)) or not 0 <= params.min_iou <= 1
            or not isinstance(params.min_anchors, int) or params.min_anchors < 1):
        raise ValueError("min_iou must be between 0 and 1 and min_anchors must be positive")
    shape = workspace.image_shape
    if shape is None:
        raise ValueError("Fixed body needs the recording image dimensions")
    revision = input_revision(workspace.path)
    state = workspace.load_state()
    if "centerline_xy" not in state or "width_profile" not in state:
        raise ValueError("Run the main pipeline and review its poses before building a fixed body")
    curves = np.asarray(state["centerline_xy"], dtype=float)
    valid = np.asarray(state["fitted"], dtype=bool) & np.isfinite(curves).all(axis=(1, 2))
    valid &= ~np.asarray(state.get("mask_stale", np.zeros(workspace.n)), dtype=bool)
    valid &= np.linalg.norm(np.diff(curves, axis=1), axis=2).sum(axis=1) > 0
    anchors, length, profile = calibrate_body(
        workspace, state, samples=params.segments + 1, min_iou=params.min_iou, min_anchors=params.min_anchors,
        anchor_frames=params.anchor_frames, progress=progress,
    )
    n, p = workspace.n, params.segments + 1
    arrays = {
        "frame_index": workspace.frame_index,
        "centerline_xy": np.full((n, p, 2), np.nan),
        "extrapolated": np.zeros((n, p), dtype=bool), "in_fov": np.zeros((n, p), dtype=bool),
        "rms_px": np.full(n, np.nan), "length_error_px": np.full(n, np.nan),
        "status": np.full(n, "unfitted_or_stale", dtype="<U32"),
        "width_profile": profile,
    }
    for row in range(n):
        if valid[row]:
            result = fit_chain(curves[row], length, params.segments, shape)
            for key, value in result.items():
                arrays[key][row] = value
        if progress is not None and (row % 25 == 0 or row == n - 1):
            progress(.15 + .85 * (row + 1) / n, f"Fixed body: {row + 1}/{n} frames")
    statuses, counts = np.unique(arrays["status"], return_counts=True)
    metadata = {
        "version": 1, "created_at": utc_now(), "job": job, "input_revision": revision,
        "length_px": length, "segments": params.segments, "segment_length_px": length / params.segments,
        "anchor_frames": workspace.frame_index[anchors].tolist(), "min_iou": params.min_iou,
        "frame_count": n, "status_counts": dict(zip(statuses.tolist(), counts.tolist())),
    }
    arrays["metadata"] = np.array(json.dumps(metadata))
    if revision != input_revision(workspace.path):
        raise ValueError("Workspace changed while building the fixed body; rerun the stage")
    _write_npz_atomic(workspace.path / FILENAME, arrays)
    return metadata


class FixedBodyResult:
    def __init__(self, path: Path) -> None:
        self.arrays = _load_npz(path / FILENAME)
        self.metadata = json.loads(str(self.arrays["metadata"])) if self.arrays else None
        self.stale = bool(self.metadata and self.metadata["input_revision"] != input_revision(path))

    def frame(self, row: int) -> dict[str, Any] | None:
        if self.metadata is None:
            return None
        out = {key: self.metadata[key] for key in ("created_at", "length_px", "segments", "segment_length_px")}
        out.update(stale=self.stale, status=str(self.arrays["status"][row]), anchor_count=len(self.metadata["anchor_frames"]))
        if self.stale:
            return out
        curve = self.arrays["centerline_xy"][row]
        if np.isfinite(curve).all():
            for key in ("centerline_xy", "extrapolated", "in_fov"):
                out[key] = self.arrays[key][row].tolist()
            out["width_profile"] = self.arrays["width_profile"].tolist()
            for key in ("rms_px", "length_error_px"):
                value = float(self.arrays[key][row])
                out[key] = value if np.isfinite(value) else None
        return out
