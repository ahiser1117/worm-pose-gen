"""Local web app for tuning the real local-darkness segmentation heuristic."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.resources
import json
from pathlib import Path
import struct
import threading
from typing import Any, Mapping
from urllib.parse import urlparse
import zlib

import h5py
import numpy as np
from numpy.typing import NDArray

from .classical import ClassicalConfig, segment_dark_ridge


DEFAULT_PROXY_HDF5 = Path(
    "/temp_data4/alex/external_artifacts/datasets/"
    "worm_pose_gen/proxy_v1/proxy_labels.h5"
)
DEFAULT_SAMPLE_ID = "2023-09-19-01:3420"

PARAMETER_LIMITS: dict[str, tuple[float, float]] = {
    "local_radius": (3, 81),
    "smooth_radius": (0, 8),
    "foreground_z": (0.5, 6.0),
    "connected_foreground_z": (0.25, 5.95),
    "close_radius": (0, 8),
}


class ProxyFrameRepository:
    """Read the exact cached uint8 frames from the proxy-label artifact."""

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)
        self._lock = threading.Lock()
        self._handle: h5py.File | None = None
        self._lookup: dict[str, tuple[str, int]] = {}
        self.samples: list[dict[str, str | int]] = []
        with h5py.File(self.source_path, "r") as handle:
            for recording in sorted(handle.keys()):
                group = handle[recording]
                if "accepted_frame_index" not in group or "accepted_image" not in group:
                    continue
                frame_indices = np.asarray(group["accepted_frame_index"], dtype=np.int64)
                images = group["accepted_image"]
                if len(frame_indices) != len(images):
                    raise ValueError(f"cached frame index/image mismatch in {recording}")
                for position, frame_index in enumerate(frame_indices):
                    sample_id = f"{recording}:{int(frame_index)}"
                    self._lookup[sample_id] = (recording, position)
                    self.samples.append({
                        "sample_id": sample_id,
                        "recording": recording,
                        "frame_index": int(frame_index),
                        "label": f"{recording} · frame {int(frame_index)}",
                    })
        if not self.samples:
            raise ValueError("proxy HDF5 contains no cached accepted images")

    def read(self, sample_id: str) -> NDArray[np.uint8]:
        if sample_id not in self._lookup:
            raise ValueError("unknown sample_id")
        recording, position = self._lookup[sample_id]
        with self._lock:
            if self._handle is None:
                self._handle = h5py.File(self.source_path, "r")
            image = np.asarray(
                self._handle[recording]["accepted_image"][position], dtype=np.uint8
            )
        if image.ndim != 2:
            raise ValueError("tuner requires two-dimensional grayscale frames")
        return image

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


def parse_config(payload: Mapping[str, Any]) -> ClassicalConfig:
    """Validate a browser request and return the executable pipeline config."""

    defaults = ClassicalConfig()

    def number(name: str, *, integer: bool = False) -> int | float:
        raw = payload.get(name, getattr(defaults, name))
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
        low, high = PARAMETER_LIMITS[name]
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must lie between {low:g} and {high:g}")
        if integer and not value.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(value) if integer else value

    connected_enabled = payload.get("connected_enabled", True)
    if not isinstance(connected_enabled, bool):
        raise ValueError("connected_enabled must be boolean")
    foreground_z = float(number("foreground_z"))
    connected_z = (
        float(number("connected_foreground_z")) if connected_enabled else None
    )
    if connected_z is not None and connected_z >= foreground_z:
        raise ValueError("faint continuation cutoff must be below the primary cutoff")
    return ClassicalConfig(
        local_radius=int(number("local_radius", integer=True)),
        smooth_radius=int(number("smooth_radius", integer=True)),
        foreground_z=foreground_z,
        connected_foreground_z=connected_z,
        close_radius=int(number("close_radius", integer=True)),
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def encode_png(values: NDArray[np.uint8]) -> bytes:
    """Encode a uint8 gray, RGB, or RGBA image without another dependency."""

    image = np.ascontiguousarray(values, dtype=np.uint8)
    if image.ndim == 2:
        height, width = image.shape
        color_type = 0
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        height, width, channels = image.shape
        color_type = 2 if channels == 3 else 6
    else:
        raise ValueError("PNG input must be a uint8 gray, RGB, or RGBA image")
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )


def _data_url(image: NDArray[np.uint8]) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(image)).decode("ascii")


def _score_rgb(score: NDArray[np.float64], high_cutoff: float) -> NDArray[np.uint8]:
    upper = max(4.5, high_cutoff + 0.75, float(np.percentile(score, 99.5)))
    normalized = np.clip((score + 2.0) / (upper + 2.0), 0.0, 1.0)
    positions = np.asarray([0.0, 0.22, 0.48, 0.72, 1.0])
    colors = np.asarray(
        [
            [5, 8, 18],
            [50, 18, 91],
            [146, 35, 99],
            [238, 91, 43],
            [252, 244, 194],
        ],
        dtype=np.float64,
    )
    output = np.empty((*score.shape, 3), dtype=np.float64)
    for channel in range(3):
        output[..., channel] = np.interp(normalized, positions, colors[:, channel])
    return np.rint(output).astype(np.uint8)


def analyze_image(image: NDArray[np.uint8], config: ClassicalConfig) -> dict[str, Any]:
    """Return visual layers and concise metrics for one parameter setting."""

    result = segment_dark_ridge(image, config)
    high = result.high_threshold_mask
    faint = (
        np.logical_and(result.connected_threshold_mask, ~high)
        if result.connected_threshold_mask is not None
        else np.zeros_like(high)
    )
    recovered = np.logical_and(result.component, ~result.high_component)
    discarded_high = np.logical_and(result.closed_high_mask, ~result.high_component)

    candidate = np.zeros((*image.shape, 4), dtype=np.uint8)
    candidate[faint] = (255, 177, 66, 168)
    candidate[high] = (241, 65, 150, 205)

    kept = np.zeros((*image.shape, 4), dtype=np.uint8)
    kept[discarded_high] = (242, 89, 77, 135)
    kept[result.high_component] = (52, 215, 196, 188)
    kept[recovered] = (173, 232, 109, 205)

    high_area = int(high.sum())
    component_area = int(result.component.sum())
    return {
        "images": {
            "frame": _data_url(image),
            "score": _data_url(_score_rgb(result.score, config.foreground_z)),
            "candidate": _data_url(candidate),
            "kept": _data_url(kept),
        },
        "metrics": {
            "high_candidate_area": high_area,
            "faint_candidate_area": int(faint.sum()),
            "retained_component_area": component_area,
            "recovered_area": result.recovered_area,
            "recovered_percent": (
                100.0 * result.recovered_area / max(int(result.high_component.sum()), 1)
            ),
            "high_component_count": result.component_count,
            "disconnected_faint_area": result.disconnected_connected_area,
            "score_min": float(result.score.min()),
            "score_max": float(result.score.max()),
        },
        "config": asdict(config),
    }


def _static_bytes(name: str) -> bytes:
    root = importlib.resources.files("worm_pose_gen.heuristic_tuner_ui")
    return root.joinpath(name).read_bytes()


class TunerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], frames: ProxyFrameRepository
    ) -> None:
        super().__init__(address, TunerRequestHandler)
        self.frames = frames


class TunerRequestHandler(BaseHTTPRequestHandler):
    server: TunerHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_static(self, name: str, content_type: str) -> None:
        content = _static_bytes(name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_static("index.html", "text/html; charset=utf-8")
            elif parsed.path == "/app.js":
                self._send_static("app.js", "text/javascript; charset=utf-8")
            elif parsed.path == "/style.css":
                self._send_static("style.css", "text/css; charset=utf-8")
            elif parsed.path == "/api/state":
                sample_ids = {
                    str(sample["sample_id"]) for sample in self.server.frames.samples
                }
                default_id = (
                    DEFAULT_SAMPLE_ID
                    if DEFAULT_SAMPLE_ID in sample_ids
                    else min(sample_ids)
                )
                defaults = asdict(ClassicalConfig())
                self._send_json(
                    {
                        "source": str(self.server.frames.source_path),
                        "samples": self.server.frames.samples,
                        "default_sample_id": default_id,
                        "defaults": defaults,
                    }
                )
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, OSError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/api/analyze":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 50_000:
                raise ValueError("analysis request size is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("analysis request must be an object")
            sample_id = str(payload.get("sample_id", ""))
            config = parse_config(payload)
            self._send_json(analyze_image(self.server.frames.read(sample_id), config))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def create_server(
    source_path: str | Path = DEFAULT_PROXY_HDF5,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
) -> TunerHTTPServer:
    return TunerHTTPServer((host, port), ProxyFrameRepository(source_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-hdf5", type=Path, default=DEFAULT_PROXY_HDF5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = create_server(args.proxy_hdf5, host=args.host, port=args.port)
    print(f"Local-darkness tuner: http://{args.host}:{args.port}")
    print(f"Frame source: {args.proxy_hdf5}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.frames.close()
        server.server_close()


if __name__ == "__main__":
    main()
