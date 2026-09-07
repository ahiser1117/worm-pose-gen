"""Localhost diagnostic viewer for stored pose runs.

Given run directories written by ``scripts/fit_recording.py`` (``summary.json``
and ``poses.npz``), the viewer lets you scrub through the frames of a run
and see every layer the pipeline produced for a frame: the flat-fielded
image, the segmenter's probability map, the thresholded mask, the mask after
hole filling and after keeping the largest component, the mask the fit was
scored against, the fitted tube and its centerline, the independent fit
before propagation replaced it (when the run stored it), and the skeleton
and moment starts the fitter would begin from.  Alongside the images it
reports every per-frame statistic the run tracks, the ambiguity flags with
the values and thresholds behind them, a frame classification, the width
profile along the body against its symmetric and prior-shaped references,
and the body's curvature; over the whole run it serves the time series
(length, area, IoU, ambiguity, jump, width, energy) for synced charts.

The server binds to localhost, opens recordings read-only, re-runs the
segmenter on the frame being viewed (cached), and writes nothing except
review notes to the file named by ``--notes``.  Masks exchanged with the
browser are PNGs with 0 = background and 255 = inside.
"""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from dataclasses import fields as dataclass_fields
import datetime as dt
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.resources
import io
import json
import math
from pathlib import Path
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch

from .ambiguity import FLAG_NAMES, AmbiguityThresholds
from .heuristic_tuner import encode_png
from .label_app import RecordingSource
from .latent import cubic_bspline_basis, decode_centerline
from .mask_fit import MaskFitConfig, fill_narrow_holes, standard_initializations
from .connected_components import largest_component
from .pose_run import cleanup_options, render_tube, tube_area_px
from .segmentation_dataset import DEFAULT_DATASET_ROOT
from .segmenter import load_segmenter


DEFAULT_RUNS_ROOT = Path("/temp_data4/alex/external_artifacts/poses")
DEFAULT_CHECKPOINT = Path("checkpoints/segmenter/best.ckpt")
DEFAULT_NOTES = Path("docs/pose_review/notes.json")
FRAME_CACHE_SIZE = 96
SOURCE_NAMES = {0: "independent", 1: "forward", 2: "backward"}

# Per-frame scalar arrays served as time series (those present in the run).
SERIES_KEYS = (
    "iou", "iou_independent", "energy", "total_energy", "body_length_px", "width_px", "points_in_fov",
    "taper_asymmetry", "orientation_gap", "worm_pixels", "raw_worm_pixels", "pixels_filled", "components",
    "pixels_outside_largest", "area_ratio", "self_contact_px", "pose_jump_px", "length_deviation",
    "ambiguity_score", "score_independent", "source", "n_starts", "mask_on_border", "reversed",
)

# Flag semantics after propagation (plan step 5b): some flags describe a
# well-fit coil or a body leaving the camera, others a fit that went wrong.
FAILURE_FLAGS = ("low_iou", "area_excess", "pose_jump", "length_deviation")
COIL_FLAGS = ("self_contact", "holes", "area_deficit")
EDGE_FLAGS = ("edge_inside",)
MASK_FLAGS = ("fragments",)

FLAG_DESCRIPTIONS = {
    "low_iou": "tube and mask overlap below the threshold",
    "area_deficit": "mask area below the visible tube area (tube overlaps itself)",
    "area_excess": "mask area above the visible tube area (tube misses body)",
    "self_contact": "two body points at least 15 samples apart closer than 0.8 widths",
    "holes": "pixels a narrow-hole fill adds (a closed turn encloses background)",
    "fragments": "mask pixels outside the largest component (dropped tail, debris)",
    "length_deviation": "fitted length far from the recording prior (in log sigmas)",
    "pose_jump": "mean centerline move from the previous frame, in widths",
    "edge_inside": "mask reaches the border but every centerline point is inside",
}


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not math.isfinite(v) else round(v, digits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_round(v, digits) for v in value.tolist()]
    return value


def _series(values: np.ndarray, digits: int = 4) -> list[Any]:
    array = np.asarray(values)
    if array.dtype.kind == "f":
        rounded = np.round(array.astype(np.float64), digits)
        return [None if not math.isfinite(v) else float(v) for v in rounded.tolist()]
    if array.dtype.kind == "b":
        return [int(v) for v in array.tolist()]
    if array.dtype.kind in "iu":
        return [int(v) for v in array.tolist()]
    return [str(v) for v in array.tolist()]


def data_url(values: NDArray[np.uint8]) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(values)).decode("ascii")


def jpeg_data_url(gray: NDArray[np.uint8], quality: int = 88) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(gray, dtype=np.uint8)).save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def mask_data_url(mask: NDArray[np.generic]) -> str:
    return data_url(np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8))


def probability_data_url(probability: NDArray[np.floating]) -> str:
    return data_url(np.clip(np.rint(np.asarray(probability, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8))


def signed_curvature(centerline_xy: NDArray[np.generic]) -> NDArray[np.float64]:
    """Signed curvature (1/px) at each centerline point from central differences."""

    points = np.asarray(centerline_xy, dtype=np.float64)
    if len(points) < 3:
        return np.zeros(len(points))
    d1 = np.gradient(points, axis=0)
    d2 = np.gradient(d1, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = cross / np.where(speed > 0, speed**3, np.nan)
    return np.nan_to_num(curvature)


def prior_width_profile(
    width_px: float, template: NDArray[np.generic], width_shape: NDArray[np.generic] | None
) -> NDArray[np.float64]:
    """Full width along the body for a scale, the symmetric template, and log-space shape coefficients.

    Mirrors ``_MaskFitState.diameter``: the correction is a clamped cubic
    B-spline through the coefficients, mean-centred, applied in log space.
    """

    profile = np.asarray(template, dtype=np.float64)
    if width_shape is None or len(np.asarray(width_shape)) == 0:
        return float(width_px) * profile
    shape = np.asarray(width_shape, dtype=np.float64)
    correction = cubic_bspline_basis(len(profile), len(shape)) @ shape
    correction -= correction.mean()
    return float(width_px) * np.exp(correction) * profile


def classify_frame(
    fitted: bool,
    flags: dict[str, bool],
    score: int,
    *,
    points_in_fov: int,
    n_points: int,
    mask_on_border: bool,
    source: int = 0,
) -> dict[str, Any]:
    """Name what a frame's flags say about it.

    ``kind`` follows the score (0 clean, 1 watch, >= 2 ambiguous); ``tags``
    separate a coil (self-contact, holes, area deficit) and a body leaving the
    camera (edge flag, border mask, off-camera points) from a suspected fit
    failure (low overlap, area excess, jump, length deviation) and a
    fragmented mask, and record whether propagation replaced the fit.
    """

    if not fitted:
        return {"kind": "unfitted", "label": "no fit", "tags": ["no fit"]}
    tags: list[str] = []
    if any(flags.get(name, False) for name in COIL_FLAGS):
        tags.append("coil / self-contact")
    if any(flags.get(name, False) for name in EDGE_FLAGS) or mask_on_border or points_in_fov < n_points:
        tags.append("body at camera edge")
    if any(flags.get(name, False) for name in MASK_FLAGS):
        tags.append("fragmented mask")
    if any(flags.get(name, False) for name in FAILURE_FLAGS):
        tags.append("fit failure suspected")
    if source in (1, 2):
        tags.append(f"propagated {SOURCE_NAMES[source]}")
    kind = "clean" if score == 0 else ("watch" if score == 1 else "ambiguous")
    descriptive = [t for t in tags if not t.startswith("propagated")]
    label = kind if not descriptive else f"{kind}: " + ", ".join(descriptive)
    return {"kind": kind, "label": label, "tags": tags}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run_entry(path: Path) -> dict[str, Any]:
    """Catalog row for one run directory, from its summary only."""

    summary = _load_json(path / "summary.json")
    cleanup = cleanup_options(summary)
    iou = summary.get("iou") or {}
    frames = summary.get("frames") or [None, None]
    checkpoint = (summary.get("checkpoint") or {}).get("sha256") or ""
    return {
        "name": path.name,
        "path": str(path),
        "recording": Path(summary["recording"]).stem,
        "recording_path": summary["recording"],
        "frames": [None if frames[0] is None else int(frames[0]), None if frames[1] is None else int(frames[1])],
        "step": int(summary.get("step", 1)),
        "frame_count": int(summary.get("frame_count", 0)),
        "started_at": summary.get("started_at"),
        "preset": summary.get("preset"),
        "iou_median": iou.get("median"),
        "iou_min": iou.get("min"),
        "frames_below_0.9": None if iou.get("fraction_at_least_0.9") is None or not summary.get("frames_fitted")
        else int(round((1.0 - float(iou["fraction_at_least_0.9"])) * int(summary["frames_fitted"]))),
        "mask_cleanup": ("fill" if cleanup["fill_holes"] else "no fill") + " + " + ("largest" if cleanup["largest_only"] else "all components"),
        "propagated": bool(summary.get("propagation")),
        "checkpoint_sha": checkpoint[:8],
        "checkpoint_path": (summary.get("checkpoint") or {}).get("path"),
        "git_commit": (summary.get("git") or {}).get("commit", "")[:8],
    }


class Segmenters:
    """Segmenter modules by checkpoint path, loaded once and shared."""

    def __init__(self, device: torch.device, fallback: Path | None) -> None:
        self.device = device
        self.fallback = fallback
        self._modules: dict[str, Any] = {}
        self._lock = threading.Lock()

    def resolve(self, checkpoint: str | None) -> Path | None:
        candidates = [Path(checkpoint)] if checkpoint else []
        if self.fallback is not None:
            candidates.append(self.fallback)
        for path in candidates:
            if path.exists():
                return path
        return None

    def probability(self, checkpoint: str | None, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32] | None, str | None]:
        path = self.resolve(checkpoint)
        if path is None:
            return None, None
        key = str(path.resolve())
        with self._lock:
            module = self._modules.get(key)
            if module is None:
                module = load_segmenter(path, self.device)
                self._modules[key] = module
            return module.predict_probability_batch(image[None], batch_size=1)[0], key


class LoadedRun:
    """One run directory with its arrays, prior, recording, and a frame cache."""

    def __init__(self, path: Path, source: RecordingSource | None, source_error: str | None) -> None:
        self.path = Path(path)
        self.summary = _load_json(self.path / "summary.json")
        with np.load(self.path / "poses.npz", allow_pickle=False) as archive:
            self.arrays = {name: archive[name] for name in archive.files}
        self.source = source
        self.source_error = source_error
        prior_path = self.path / "recording_prior.json"
        self.prior = _load_json(prior_path) if prior_path.exists() else self.summary.get("prior")
        thresholds = (self.summary.get("ambiguity") or {}).get("thresholds") or {}
        known = {f.name for f in dataclass_fields(AmbiguityThresholds)}
        self.thresholds = AmbiguityThresholds(**{k: v for k, v in thresholds.items() if k in known})
        self.cleanup = cleanup_options(self.summary)
        self.threshold = float(self.summary.get("threshold", 0.5))
        self.frame_index = np.asarray(self.arrays["frame_index"], dtype=np.int64)
        self._rows = {int(f): r for r, f in enumerate(self.frame_index.tolist())}
        self.n_points = int(self.arrays["centerline_xy"].shape[1])
        self.stretches = [(int(a), int(b)) for a, b in ((self.summary.get("propagation") or {}).get("stretches") or [])]
        self._cache: OrderedDict[tuple[int, float], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def image_shape(self) -> tuple[int, int] | None:
        return None if self.source is None else self.source.shape

    def row_of(self, frame: int) -> int:
        try:
            return self._rows[int(frame)]
        except KeyError as error:
            raise ValueError(f"frame {frame} is not in this run") from error

    def fit_config(self) -> MaskFitConfig:
        known = {f.name for f in dataclass_fields(MaskFitConfig)}
        values = {}
        for key, value in (self.summary.get("fit_config") or {}).items():
            if key in known:
                values[key] = tuple(value) if isinstance(value, list) else value
        return MaskFitConfig(**values)

    def stretch_of(self, row: int) -> dict[str, Any] | None:
        for index, (a, b) in enumerate(self.stretches):
            if a <= row <= b:
                return {"index": index, "rows": [a, b], "frames": [int(self.frame_index[a]), int(self.frame_index[b])], "length": b - a + 1}
        return None

    def series(self) -> dict[str, Any]:
        arrays = self.arrays
        fitted = np.asarray(arrays["fitted"], dtype=bool)
        out: dict[str, Any] = {"frame_index": _series(self.frame_index), "fitted": _series(fitted)}
        for key in SERIES_KEYS:
            if key in arrays:
                out[key] = _series(arrays[key])
        out["flags"] = {name: _series(arrays[f"flag_{name}"]) for name in FLAG_NAMES if f"flag_{name}" in arrays}
        if "best_start" in arrays:
            out["best_start"] = _series(arrays["best_start"])
        # Tube area over the visible body, the denominator of the area ratio.
        if "area_ratio" in arrays:
            with np.errstate(divide="ignore", invalid="ignore"):
                visible = np.asarray(arrays["worm_pixels"], dtype=np.float64) / np.asarray(arrays["area_ratio"], dtype=np.float64)
            out["tube_area_visible_px"] = _series(np.where(fitted, visible, np.nan), 1)
        out["tube_area_px"] = _series(
            np.array([tube_area_px(arrays["width_profile"][r], float(arrays["body_length_px"][r])) if fitted[r] else np.nan for r in range(len(fitted))]), 1
        )
        out["classification"] = [self._classification(r)["kind"] for r in range(len(fitted))]
        return out

    def _flags(self, row: int) -> dict[str, bool]:
        return {name: bool(self.arrays[f"flag_{name}"][row]) for name in FLAG_NAMES if f"flag_{name}" in self.arrays}

    def _classification(self, row: int) -> dict[str, Any]:
        arrays = self.arrays
        return classify_frame(
            bool(arrays["fitted"][row]), self._flags(row), int(arrays["ambiguity_score"][row]) if "ambiguity_score" in arrays else 0,
            points_in_fov=int(arrays["points_in_fov"][row]), n_points=self.n_points,
            mask_on_border=bool(arrays["mask_on_border"][row]) if "mask_on_border" in arrays else False,
            source=int(arrays["source"][row]) if "source" in arrays else 0,
        )

    def flag_details(self, row: int) -> list[dict[str, Any]]:
        """Each flag with the value it tested and the threshold, for the stats panel."""

        arrays, t = self.arrays, self.thresholds
        width = float(arrays["width_px"][row])
        sigma = None if self.prior is None else float(self.prior["log_length_sigma"])
        values: dict[str, tuple[Any, Any, str]] = {
            "low_iou": (arrays["iou"][row], t.low_iou, "<"),
            "area_deficit": (arrays["area_ratio"][row] if "area_ratio" in arrays else None, t.area_deficit, "<"),
            "area_excess": (arrays["area_ratio"][row] if "area_ratio" in arrays else None, t.area_excess, ">"),
            "self_contact": (arrays["self_contact_px"][row] if "self_contact_px" in arrays else None, t.self_contact_width_fraction * width, "<"),
            "holes": (arrays["pixels_filled"][row], t.holes_px, ">"),
            "fragments": (arrays["pixels_outside_largest"][row], t.fragments_px, ">"),
            "length_deviation": (
                arrays["length_deviation"][row] if "length_deviation" in arrays else None,
                None if sigma is None else t.length_sigmas * sigma, "|x| >",
            ),
            "pose_jump": (arrays["pose_jump_px"][row] if "pose_jump_px" in arrays else None, t.jump_width_fraction * width, ">"),
            "edge_inside": (
                f"border={bool(arrays['mask_on_border'][row]) if 'mask_on_border' in arrays else False}, in_view={int(arrays['points_in_fov'][row])}/{self.n_points}",
                None, "",
            ),
        }
        flags = self._flags(row)
        return [
            {
                "name": name, "fired": flags.get(name, False), "value": _round(values[name][0]), "threshold": _round(values[name][1]),
                "test": values[name][2], "description": FLAG_DESCRIPTIONS[name],
                "group": "failure" if name in FAILURE_FLAGS else "coil" if name in COIL_FLAGS else "edge" if name in EDGE_FLAGS else "mask",
            }
            for name in FLAG_NAMES
        ]

    def stats(self, row: int) -> dict[str, Any]:
        arrays = self.arrays
        out: dict[str, Any] = {"row": row, "frame_index": int(self.frame_index[row])}
        for key in ("fitted", *SERIES_KEYS, "best_start", "taper_asymmetry"):
            if key in arrays:
                out[key] = _round(arrays[key][row])
        if "source" in arrays:
            out["source_name"] = SOURCE_NAMES.get(int(arrays["source"][row]), str(int(arrays["source"][row])))
        out["in_view_fraction"] = _round(float(arrays["points_in_fov"][row]) / self.n_points)
        if arrays["fitted"][row]:
            out["tube_area_px"] = _round(tube_area_px(arrays["width_profile"][row], float(arrays["body_length_px"][row])), 1)
            out["crop"] = _round(arrays["crop"][row])
            if self.prior is not None:
                out["length_vs_prior_sigmas"] = _round(
                    math.log(float(arrays["body_length_px"][row]) / float(self.prior["length_px"])) / float(self.prior["log_length_sigma"]), 2
                )
                out["width_vs_prior_sigmas"] = _round(
                    math.log(float(arrays["width_px"][row]) / float(self.prior["width_px"])) / float(self.prior["log_width_sigma"]), 2
                )
        out["classification"] = self._classification(row)
        out["flags"] = self.flag_details(row)
        out["stretch"] = self.stretch_of(row)
        return out

    def pose(self, row: int) -> dict[str, Any] | None:
        """Centerline, width profile, references and curvature of the stored fit."""

        arrays = self.arrays
        if not arrays["fitted"][row]:
            return None
        centerline = np.asarray(arrays["centerline_xy"][row], dtype=np.float64)
        profile = np.asarray(arrays["width_profile"][row], dtype=np.float64)
        template = arrays["width_template"] if "width_template" in arrays else None
        width = float(arrays["width_px"][row])
        out: dict[str, Any] = {
            "centerline_xy": _round(np.round(centerline, 2), 2),
            "width_profile": _round(np.round(profile, 2), 2),
            "curvature": _round(np.round(signed_curvature(centerline), 5), 5),
            "body_length_px": _round(arrays["body_length_px"][row], 1),
            "width_px": _round(width, 2),
            "crop": _round(arrays["crop"][row]),
        }
        if template is not None:
            out["width_template_profile"] = _round(np.round(width * np.asarray(template, dtype=np.float64), 2), 2)
            if self.prior is not None and self.prior.get("width_shape") is not None:
                out["width_prior_profile"] = _round(np.round(prior_width_profile(width, template, self.prior["width_shape"]), 2), 2)
        if "centerline_xy_independent" in arrays and "source" in arrays and int(arrays["source"][row]) != 0:
            out["independent"] = {
                "centerline_xy": _round(np.round(arrays["centerline_xy_independent"][row], 2), 2),
                "width_profile": _round(np.round(arrays["width_profile_independent"][row], 2), 2) if "width_profile_independent" in arrays else None,
                "iou": _round(arrays["iou_independent"][row]) if "iou_independent" in arrays else None,
            }
        return out

    def frame(
        self, row: int, segmenters: Segmenters, threshold: float | None, device: torch.device, *, raw: bool = False, detail: str = "full"
    ) -> dict[str, Any]:
        """Layers and statistics of one frame, cached per threshold.

        ``detail="light"`` skips the segmenter and the mask layers (image,
        tube, pose and statistics only), which is what the browser asks for
        while the user scrubs; the full layers follow once the cursor rests.
        """

        threshold = self.threshold if threshold is None else float(threshold)
        light = detail == "light"
        key = (row, threshold, "full")
        with self._lock:
            cached = self._cache.get(key)
            if cached is None and light:
                key = (row, threshold, "light")
                cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
        if cached is None:
            cached = self._layers(row, segmenters, threshold, device, light=light)
            with self._lock:
                self._cache[key] = cached
                while len(self._cache) > FRAME_CACHE_SIZE:
                    self._cache.popitem(last=False)
        payload = dict(cached)
        if raw and self.source is not None:
            payload["image_raw"] = jpeg_data_url(self.source.read(int(self.frame_index[row])))
        payload["stats"] = self.stats(row)
        payload["pose"] = self.pose(row)
        return payload

    def _layers(self, row: int, segmenters: Segmenters, threshold: float, device: torch.device, *, light: bool = False) -> dict[str, Any]:
        arrays = self.arrays
        frame_index = int(self.frame_index[row])
        payload: dict[str, Any] = {
            "row": row, "frame_index": frame_index, "threshold": threshold, "detail": "light" if light else "full",
            "layers": {}, "mask_stats": None, "errors": [],
        }
        if self.source is None:
            payload["errors"].append(f"recording not readable: {self.source_error}")
            height, width = (int(v) for v in (self.summary.get("image_shape") or (0, 0)))
        else:
            _, image = self.source.corrected(frame_index)
            height, width = image.shape
            payload["layers"]["image"] = jpeg_data_url(image)
            probability, checkpoint = (None, None) if light else segmenters.probability((self.summary.get("checkpoint") or {}).get("path"), image)
            if probability is None:
                if not light:
                    payload["errors"].append("no segmenter checkpoint available; mask layers skipped")
            else:
                payload["checkpoint"] = checkpoint
                payload["layers"]["probability"] = probability_data_url(probability)
                raw_mask = probability >= threshold
                stats = {"raw_worm_pixels": int(raw_mask.sum()), "pixels_filled": 0, "components": 0, "pixels_outside_largest": 0, "worm_pixels": 0}
                payload["layers"]["mask_raw"] = mask_data_url(raw_mask)
                if raw_mask.any():
                    filled, added = fill_narrow_holes(raw_mask, self.cleanup["hole_radius"], device=device)
                    largest, area, count = largest_component(filled)
                    final = filled if self.cleanup["fill_holes"] else raw_mask
                    if self.cleanup["largest_only"]:
                        final = largest if self.cleanup["fill_holes"] else (raw_mask & largest)
                    stats.update(pixels_filled=int(added), components=int(count), pixels_outside_largest=int(filled.sum()) - int(area), worm_pixels=int(final.sum()))
                    payload["layers"]["mask_filled"] = mask_data_url(filled)
                    payload["layers"]["mask_largest"] = mask_data_url(largest)
                    payload["layers"]["mask_final"] = mask_data_url(final)
                payload["mask_stats"] = stats
                stored = {k: int(arrays[k][row]) for k in ("raw_worm_pixels", "pixels_filled", "components", "pixels_outside_largest", "worm_pixels") if k in arrays}
                payload["mask_stats_stored"] = stored
        payload["height"], payload["width"] = height, width
        if arrays["fitted"][row] and height and width:
            tube = render_tube(arrays["centerline_xy"][row], arrays["width_profile"][row], height, width, window=tuple(arrays["crop"][row]), device=device)
            payload["layers"]["tube"] = mask_data_url(tube)
            if "centerline_xy_independent" in arrays and "source" in arrays and int(arrays["source"][row]) != 0:
                independent = render_tube(
                    arrays["centerline_xy_independent"][row], arrays["width_profile_independent"][row], height, width, device=device
                )
                payload["layers"]["tube_independent"] = mask_data_url(independent)
        return payload

    def starts(self, row: int, segmenters: Segmenters, threshold: float | None, device: torch.device) -> dict[str, Any]:
        """The fitter's standard starting centerlines on this frame's final mask."""

        threshold = self.threshold if threshold is None else float(threshold)
        if self.source is None:
            raise ValueError(f"recording not readable: {self.source_error}")
        _, image = self.source.corrected(int(self.frame_index[row]))
        probability, _ = segmenters.probability((self.summary.get("checkpoint") or {}).get("path"), image)
        if probability is None:
            raise ValueError("no segmenter checkpoint available")
        raw_mask = probability >= threshold
        if not raw_mask.any():
            return {"starts": []}
        filled, _ = fill_narrow_holes(raw_mask, self.cleanup["hole_radius"], device=device)
        largest, _, _ = largest_component(filled)
        final = filled if self.cleanup["fill_holes"] else raw_mask
        if self.cleanup["largest_only"]:
            final = largest if self.cleanup["fill_holes"] else (raw_mask & largest)
        config = self.fit_config()
        starts = standard_initializations(final, config=config)
        return {
            "starts": [
                {"name": s.name, "width_px": _round(s.width_px, 2), "centerline_xy": _round(np.round(decode_centerline(s.latent, config.coefficients), 2), 2)}
                for s in starts
            ]
        }


class Notes:
    """Append-only review notes: frames marked by eye with tags and a comment."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text())
        return list(payload.get("notes", [])) if isinstance(payload, dict) else list(payload)

    def add(self, note: dict[str, Any]) -> list[dict[str, Any]]:
        with self._lock:
            notes = self.load()
            notes.append(note)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"notes": notes}, indent=1) + "\n")
        return notes

    def delete(self, index: int) -> list[dict[str, Any]]:
        with self._lock:
            notes = self.load()
            if 0 <= index < len(notes):
                notes.pop(index)
                self.path.write_text(json.dumps({"notes": notes}, indent=1) + "\n")
        return notes


class ViewerState:
    """Run catalog, lazily loaded runs, shared recordings and segmenters."""

    def __init__(
        self,
        runs: list[Path],
        *,
        dataset_root: Path = DEFAULT_DATASET_ROOT,
        checkpoint: Path | None = DEFAULT_CHECKPOINT,
        device: str | None = None,
        notes: Path = DEFAULT_NOTES,
    ) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataset_root = Path(dataset_root)
        self.segmenters = Segmenters(self.device, checkpoint)
        self.notes = Notes(notes)
        self.catalog: dict[str, dict[str, Any]] = {}
        self.catalog_errors: dict[str, str] = {}
        for path in runs:
            try:
                entry = run_entry(path)
            except (OSError, ValueError, KeyError) as error:
                self.catalog_errors[str(path)] = f"{type(error).__name__}: {error}"
                continue
            self.catalog[entry["name"]] = entry
        self._runs: dict[str, LoadedRun] = {}
        self._sources: dict[str, tuple[RecordingSource | None, str | None]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def discover(root: Path) -> list[Path]:
        root = Path(root)
        if not root.is_dir():
            return []
        return sorted((p for p in root.iterdir() if (p / "summary.json").exists() and (p / "poses.npz").exists()), key=lambda p: p.name)

    def _source(self, recording: str) -> tuple[RecordingSource | None, str | None]:
        with self._lock:
            if recording not in self._sources:
                try:
                    self._sources[recording] = (RecordingSource(Path(recording), self.dataset_root / "flat_fields"), None)
                except (OSError, ValueError, KeyError) as error:
                    self._sources[recording] = (None, f"{type(error).__name__}: {error}")
            return self._sources[recording]

    def run(self, name: str) -> LoadedRun:
        if name not in self.catalog:
            raise ValueError(f"unknown run {name!r}")
        with self._lock:
            loaded = self._runs.get(name)
        if loaded is None:
            entry = self.catalog[name]
            source, error = self._source(entry["recording_path"])
            loaded = LoadedRun(Path(entry["path"]), source, error)
            with self._lock:
                self._runs[name] = loaded
        return loaded

    def state(self) -> dict[str, Any]:
        runs = sorted(self.catalog.values(), key=lambda e: e["started_at"] or "", reverse=True)
        return {
            "runs": runs,
            "errors": self.catalog_errors,
            "device": str(self.device),
            "fallback_checkpoint": None if self.segmenters.fallback is None or not self.segmenters.fallback.exists() else str(self.segmenters.fallback),
            "notes_path": str(self.notes.path),
            "flag_names": list(FLAG_NAMES),
            "flag_groups": {"failure": list(FAILURE_FLAGS), "coil": list(COIL_FLAGS), "edge": list(EDGE_FLAGS), "mask": list(MASK_FLAGS)},
        }

    def run_payload(self, name: str) -> dict[str, Any]:
        run = self.run(name)
        summary = run.summary
        return {
            "entry": self.catalog[name],
            "recording_readable": run.source is not None,
            "recording_error": run.source_error,
            "image_shape": None if run.image_shape is None else list(run.image_shape),
            "recording_frame_count": None if run.source is None else run.source.frame_count,
            "n_points": run.n_points,
            "threshold": run.threshold,
            "cleanup": run.cleanup,
            "prior": run.prior,
            "thresholds": run.thresholds.__dict__,
            "stretches": [[a, b] for a, b in run.stretches],
            "propagation": summary.get("propagation"),
            "ambiguity": summary.get("ambiguity"),
            "fit_config": summary.get("fit_config"),
            "summary_iou": summary.get("iou"),
            "summary_length": summary.get("body_length_px"),
            "has_independent_pose": "centerline_xy_independent" in run.arrays,
            "series": run.series(),
            "compatible_runs": self.compatible_runs(name),
        }

    def compatible_runs(self, name: str) -> list[dict[str, Any]]:
        """Other runs of the same recording, those overlapping this run's frames first."""

        entry = self.catalog[name]
        first, last = entry["frames"]
        rows = []
        for other in self.catalog.values():
            if other["name"] == name or other["recording_path"] != entry["recording_path"]:
                continue
            a, b = other["frames"]
            overlap = 0 if None in (first, last, a, b) else max(0, min(last, b) - max(first, a) + 1)
            rows.append({"name": other["name"], "frames": other["frames"], "overlap": overlap, "iou_median": other["iou_median"], "mask_cleanup": other["mask_cleanup"]})
        rows.sort(key=lambda r: (-r["overlap"], r["name"]))
        return rows

    def frame_payload(self, name: str, frame: int, threshold: float | None, raw: bool, detail: str = "full") -> dict[str, Any]:
        if detail not in ("full", "light"):
            raise ValueError("detail must be 'full' or 'light'")
        run = self.run(name)
        return run.frame(run.row_of(frame), self.segmenters, threshold, self.device, raw=raw, detail=detail)

    def pose_payload(self, name: str, frame: int) -> dict[str, Any]:
        run = self.run(name)
        try:
            row = run.row_of(frame)
        except ValueError:
            return {"present": False}
        return {"present": True, "pose": run.pose(row), "stats": run.stats(row)}

    def starts_payload(self, name: str, frame: int, threshold: float | None) -> dict[str, Any]:
        run = self.run(name)
        return run.starts(run.row_of(frame), self.segmenters, threshold, self.device)

    def add_note(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        name = str(payload["run"])
        entry = self.catalog[name]
        note = {
            "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "run": name,
            "recording": entry["recording"],
            "frame_index": int(payload["frame_index"]),
            "tags": [str(t) for t in payload.get("tags", [])],
            "comment": str(payload.get("comment", "")),
        }
        return self.notes.add(note)

    def close(self) -> None:
        for source, _ in self._sources.values():
            if source is not None:
                source.close()


def _static_bytes(name: str) -> bytes:
    return importlib.resources.files("worm_pose_gen.pose_viewer_ui").joinpath(name).read_bytes()


class ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: ViewerState) -> None:
        super().__init__(address, ViewerRequestHandler)
        self.state = state

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The browser aborts frame requests it no longer needs while the
        # user scrubs; a closed connection is not worth a traceback.
        import sys

        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class ViewerRequestHandler(BaseHTTPRequestHandler):
    server: ViewerHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, name: str, content_type: str) -> None:
        body = _static_bytes(name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 4 * 1024 * 1024:
            raise ValueError("request body missing or too large")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        state = self.server.state
        try:
            if parsed.path in ("/", "/index.html"):
                self._send_static("index.html", "text/html; charset=utf-8")
            elif parsed.path == "/app.js":
                self._send_static("app.js", "text/javascript; charset=utf-8")
            elif parsed.path == "/style.css":
                self._send_static("style.css", "text/css; charset=utf-8")
            elif parsed.path == "/api/state":
                self._send_json(state.state())
            elif parsed.path == "/api/run":
                self._send_json(state.run_payload(query["name"]))
            elif parsed.path == "/api/frame":
                threshold = query.get("threshold")
                self._send_json(
                    state.frame_payload(
                        query["run"], int(query["frame"]), None if threshold in (None, "") else float(threshold), query.get("raw", "0") == "1",
                        query.get("detail", "full"),
                    )
                )
            elif parsed.path == "/api/pose":
                self._send_json(state.pose_payload(query["run"], int(query["frame"])))
            elif parsed.path == "/api/starts":
                threshold = query.get("threshold")
                self._send_json(state.starts_payload(query["run"], int(query["frame"]), None if threshold in (None, "") else float(threshold)))
            elif parsed.path == "/api/notes":
                self._send_json({"notes": state.notes.load(), "path": str(state.notes.path)})
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, IndexError, KeyError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # noqa: BLE001 - report to the browser instead of dropping the socket
            self._send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        state = self.server.state
        try:
            payload = self._read_json()
            if self.path == "/api/note":
                self._send_json({"notes": state.add_note(payload)})
            elif self.path == "/api/note/delete":
                self._send_json({"notes": state.notes.delete(int(payload["index"]))})
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, IndexError, KeyError, RuntimeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # noqa: BLE001 - report to the browser instead of dropping the socket
            self._send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def create_server(state: ViewerState, host: str = "127.0.0.1", port: int = 8768) -> ViewerHTTPServer:
    return ViewerHTTPServer((host, port), state)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", type=Path, dest="runs", help="run directory to serve (repeatable)")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="directory whose run subdirectories are all served")
    parser.add_argument("--only-runs", action="store_true", help="serve only the --run directories, not the root's")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="where the flat field cache lives")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="segmenter used when a run's own checkpoint is gone")
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES, help="JSON file review notes are appended to")
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs = list(args.runs or [])
    if not args.only_runs:
        runs += [p for p in ViewerState.discover(args.runs_root) if p not in runs]
    state = ViewerState(runs, dataset_root=args.dataset_root, checkpoint=args.checkpoint, device=args.device, notes=args.notes)
    server = create_server(state, args.host, args.port)
    print(f"pose viewer at http://{args.host}:{args.port}/", flush=True)
    print(f"{len(state.catalog)} runs, device {state.device}, notes in {state.notes.path}", flush=True)
    for path, error in state.catalog_errors.items():
        print(f"skipped {path}: {error}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
