"""What the Workspace page shows of one frame of a workspace: its layers, statistics, pose and the per-row series.

``LoadedRun`` holds a workspace's arrays (its state merged with its
hypotheses), its synthesised run summary and a lookup of the masks the fit
was scored against, and answers per frame with

- the flat-fielded image (JPEG) and, from ``--dev`` only, the network's
  probability with the cleanup steps behind the mask (thresholded, holes
  filled, largest component): running the segmenter on every rested frame
  only feeds those developer layers, so an analyst's frame shows the stored
  mask and costs no network pass;
- the stored final mask, the fitted tube, and the independent fit's tube
  where propagation replaced it;
- every per-frame statistic the pipeline tracks, the ambiguity flags with
  the values and thresholds behind them, a classification, and the pose
  (centerline, width profile against its symmetric and prior-shaped
  references, curvature).

Over the whole workspace it serves the time series for the timeline.  Frames
are cached per detail level and model; ``detail="light"`` (playback and
scrubbing) skips the mask and tube layers.  ``WorkspaceView``
(``app/workspace_view.py``) rebuilds the ``LoadedRun`` whenever a job or an
edit changes the workspace's files.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields as dataclass_fields
import json
import math
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import torch

from ..ambiguity import FLAG_NAMES, AmbiguityThresholds
from ..connected_components import largest_component
from ..latent import cubic_bspline_basis
from ..mask_fit import fill_narrow_holes
from ..pose_run import cleanup_options, render_tube, tube_area_px
from ..recordings import RecordingSource
from ..segmenter import load_segmenter
from .images import jpeg_data_url, mask_data_url, probability_data_url


FRAME_CACHE_SIZE = 96
SOURCE_NAMES = {0: "independent", 1: "forward", 2: "backward"}

# Per-frame scalar arrays served as time series (those present in the run).
SERIES_KEYS = (
    "iou", "iou_independent", "energy", "total_energy", "body_length_px", "width_px", "points_in_fov",
    "taper_asymmetry", "orientation_gap", "worm_pixels", "raw_worm_pixels", "pixels_filled", "components",
    "pixels_outside_largest", "area_ratio", "self_contact_px", "pose_jump_px", "length_deviation",
    "ambiguity_score", "score_independent", "source", "n_starts", "mask_on_border", "reversed", "prediction_distance_px",
    "hypotheses_count", "path_override", "path_mirrored", "path_energy_gap",
    "tube_coverage", "max_bend_widths", "track_length_px", "length_refit",
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
    path_override: bool = False,
    iou: float | None = None,
    coverage: float | None = None,
    components: int = 1,
    pixels_outside_largest: int = 0,
    area_ratio: float | None = None,
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
    if path_override:
        tags.append("path overrode lowest energy")
    # Two readings of a low overlap with the tube (almost) entirely on mask.
    # Mask far larger than the tube: the segmenter painted something else as
    # worm, attached (a plate streak merging with the body) or separate, and
    # the IoU says little about the pose.  Mask about the tube's size: the
    # tube fits where it is but leaves body uncovered (a short tube inside a
    # coil).
    if iou is not None and coverage is not None and iou < 0.9 and coverage >= 0.85:
        extra_mask = (area_ratio is not None and area_ratio > 1.2) or (components > 1 and pixels_outside_largest > 500)
        tags.append("mask has extra body (segmentation)" if extra_mask else "tube on mask, mask not covered")
    kind = "clean" if score == 0 else ("watch" if score == 1 else "ambiguous")
    descriptive = [t for t in tags if not t.startswith("propagated")]
    label = kind if not descriptive else f"{kind}: " + ", ".join(descriptive)
    return {"kind": kind, "label": label, "tags": tags}


class Segmenters:
    """Segmenters by checkpoint path, loaded on the app's device at first use and reloaded when the file changes."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._modules: dict[str, Any] = {}
        self._stamps: dict[str, tuple[int, int, int]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def resolve(checkpoint: str | None) -> Path | None:
        path = Path(checkpoint) if checkpoint else None
        return path if path is not None and path.exists() else None

    def signature(self, checkpoint: str | None) -> tuple[Any, ...]:
        path = self.resolve(checkpoint)
        if path is None:
            return (None,)
        stat = path.stat()
        return (str(path.resolve()), stat.st_mtime_ns, stat.st_size, stat.st_ino)

    def probability(self, checkpoint: str | None, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32] | None, str | None]:
        path = self.resolve(checkpoint)
        if path is None:
            return None, None
        key = str(path.resolve())
        with self._lock:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            module = self._modules.get(key)
            if module is None or self._stamps.get(key) != stamp:
                module = load_segmenter(path, self.device)
                self._modules[key] = module
                self._stamps[key] = stamp
            return module.predict_probability_batch(image[None], batch_size=1)[0], key


MaskLookup = Callable[[int], NDArray[np.bool_] | None]


class LoadedRun:
    """A workspace's arrays, summary, prior and recording, with a frame cache.

    ``arrays`` is the workspace's state merged with its hypotheses,
    ``summary`` its synthesised run summary (``pipeline.read_summary``), and
    ``masks`` a lookup from row to the stored final mask; ``path`` is where
    ``recording_prior.json`` is looked for.
    """

    def __init__(
        self,
        path: Path,
        source: RecordingSource | None,
        source_error: str | None,
        *,
        arrays: dict[str, np.ndarray],
        summary: dict[str, Any],
        masks: MaskLookup,
    ) -> None:
        self.path = Path(path)
        self.summary = summary
        self.arrays = arrays
        self.masks = masks
        self.source = source
        self.source_error = source_error
        prior_path = self.path / "recording_prior.json"
        self.prior = json.loads(prior_path.read_text()) if prior_path.exists() else self.summary.get("prior")
        thresholds = (self.summary.get("ambiguity") or {}).get("thresholds") or {}
        known = {f.name for f in dataclass_fields(AmbiguityThresholds)}
        self.thresholds = AmbiguityThresholds(**{k: v for k, v in thresholds.items() if k in known})
        self.cleanup = cleanup_options(self.summary)
        self.threshold = float(self.summary.get("threshold", 0.5))
        self.frame_index = np.asarray(self.arrays["frame_index"], dtype=np.int64)
        self._rows = {int(f): r for r, f in enumerate(self.frame_index.tolist())}
        self.n_points = int(self.arrays["centerline_xy"].shape[1])
        self.stretches = [(int(a), int(b)) for a, b in ((self.summary.get("propagation") or {}).get("stretches") or [])]
        self._cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def image_shape(self) -> tuple[int, int] | None:
        return None if self.source is None else self.source.shape

    @property
    def segmentation_checkpoint(self) -> str | None:
        """Selected model for previews; the stored mask's checkpoint stays in the summary."""

        return self.summary.get("selected_checkpoint") or (self.summary.get("checkpoint") or {}).get("path")

    def row_of(self, frame: int) -> int:
        try:
            return self._rows[int(frame)]
        except KeyError as error:
            raise ValueError(f"frame {frame} is not in this workspace") from error

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
            path_override=bool(arrays["path_override"][row]) if "path_override" in arrays else False,
            iou=float(arrays["iou"][row]) if bool(arrays["fitted"][row]) else None,
            coverage=float(arrays["tube_coverage"][row]) if "tube_coverage" in arrays and bool(arrays["fitted"][row]) else None,
            components=int(arrays["components"][row]) if "components" in arrays else 1,
            pixels_outside_largest=int(arrays["pixels_outside_largest"][row]) if "pixels_outside_largest" in arrays else 0,
            area_ratio=float(arrays["area_ratio"][row]) if "area_ratio" in arrays and bool(arrays["fitted"][row]) else None,
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
        self, row: int, segmenters: Segmenters, device: torch.device, *, raw: bool = False, detail: str = "full", segment: bool = False
    ) -> dict[str, Any]:
        """Layers and statistics of one frame, cached per detail level and model.

        ``detail="light"`` skips the mask and tube layers: only the image, the
        saved pose and the statistics follow the cursor, and the full layers
        load when playback or scrubbing stops.  ``segment`` (developer mode)
        also runs the segmenter for the probability and cleanup layers.
        """

        light = detail == "light"
        segment = segment and not light
        model_signature = segmenters.signature(self.segmentation_checkpoint) if segment else None
        key = (row, "light" if light else "full", segment, model_signature)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
        if cached is None:
            cached = self._layers(row, segmenters, device, light=light, segment=segment)
            with self._lock:
                self._cache[key] = cached
                while len(self._cache) > FRAME_CACHE_SIZE:
                    self._cache.popitem(last=False)
        payload = {**cached, "layers": dict(cached["layers"])}
        if raw and self.source is not None:
            payload["image_raw"] = jpeg_data_url(self.source.read(int(self.frame_index[row])))
        payload["stats"] = self.stats(row)
        payload["pose"] = self.pose(row)
        return payload

    def stored_mask(self, row: int) -> NDArray[np.bool_] | None:
        """The final mask the workspace stored for ``row`` (its edited mask when there is one)."""

        return self.masks(row)

    def _cleaned(self, raw_mask: NDArray[np.bool_], device: torch.device) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], dict[str, int]]:
        """The fitter's cleanup of a thresholded mask: filled, largest component, final, and its statistics."""

        filled, added = fill_narrow_holes(raw_mask, self.cleanup["hole_radius"], device=device)
        largest, area, count = largest_component(filled)
        final = filled if self.cleanup["fill_holes"] else raw_mask
        if self.cleanup["largest_only"]:
            final = largest if self.cleanup["fill_holes"] else (raw_mask & largest)
        stats = {"pixels_filled": int(added), "components": int(count), "pixels_outside_largest": int(filled.sum()) - int(area), "worm_pixels": int(final.sum())}
        return filled, largest, final, stats

    def _layers(self, row: int, segmenters: Segmenters, device: torch.device, *, light: bool, segment: bool) -> dict[str, Any]:
        arrays = self.arrays
        frame_index = int(self.frame_index[row])
        payload: dict[str, Any] = {
            "row": row, "frame_index": frame_index, "threshold": self.threshold, "detail": "light" if light else "full",
            "layers": {}, "mask_stats": None, "errors": [],
        }
        if self.source is None:
            payload["errors"].append(f"recording not readable: {self.source_error}")
            height, width = (int(v) for v in (self.summary.get("image_shape") or (0, 0)))
        else:
            _, image = self.source.corrected(frame_index)
            height, width = image.shape
            payload["layers"]["image"] = jpeg_data_url(image)
            probability, checkpoint = segmenters.probability(self.segmentation_checkpoint, image) if segment else (None, None)
            if segment and probability is None:
                payload["errors"].append("the workspace's segmenter checkpoint is missing; probability layers skipped")
            if probability is not None:
                payload["checkpoint"] = checkpoint
                payload["layers"]["probability"] = probability_data_url(probability)
                raw_mask = probability >= self.threshold
                stats = {"raw_worm_pixels": int(raw_mask.sum()), "pixels_filled": 0, "components": 0, "pixels_outside_largest": 0, "worm_pixels": 0}
                payload["layers"]["mask_raw"] = mask_data_url(raw_mask)
                if raw_mask.any():
                    filled, largest, final, cleaned = self._cleaned(raw_mask, device)
                    stats.update(cleaned)
                    payload["layers"]["mask_filled"] = mask_data_url(filled)
                    payload["layers"]["mask_largest"] = mask_data_url(largest)
                    payload["layers"]["mask_final"] = mask_data_url(final)
                payload["mask_stats"] = stats
            stored_mask = None if light else self.stored_mask(row)
            if stored_mask is not None:
                # The mask the fit was scored against, as the workspace stored it.
                payload["layers"]["mask_final"] = mask_data_url(stored_mask)
                payload["mask_final_source"] = "stored"
            elif "mask_final" in payload["layers"]:
                payload["mask_final_source"] = "recomputed"
            if not light:
                stored = {k: int(arrays[k][row]) for k in ("raw_worm_pixels", "pixels_filled", "components", "pixels_outside_largest", "worm_pixels") if k in arrays}
                payload["mask_stats_stored"] = stored
        payload["height"], payload["width"] = height, width
        if not light and arrays["fitted"][row] and height and width:
            tube = render_tube(arrays["centerline_xy"][row], arrays["width_profile"][row], height, width, window=tuple(arrays["crop"][row]), device=device)
            payload["layers"]["tube"] = mask_data_url(tube)
            if "centerline_xy_independent" in arrays and "source" in arrays and int(arrays["source"][row]) != 0:
                independent = render_tube(
                    arrays["centerline_xy_independent"][row], arrays["width_profile_independent"][row], height, width, device=device
                )
                payload["layers"]["tube_independent"] = mask_data_url(independent)
        return payload
