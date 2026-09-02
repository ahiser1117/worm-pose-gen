"""Localhost labeling app for worm segmentation masks.

The loop it serves: pick a frame (random, sequential, or the frame the
current network is least sure about), flat-field it, propose a mask from the
network and from the classical pipeline, refine the proposal with pipeline
elements (hole filling, largest component, grow/shrink, the mask-fit tube),
edit it with a brush in the browser, and save it into the segmentation store
where it is assigned to train, validation, or test.

The server binds to localhost, opens recordings read-only, and never writes
anything except samples and per-recording flat fields under the dataset
root.  Mask PNGs exchanged with the browser use 0 = background, 255 = worm,
and 128 = ignore.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.resources
import io
import json
from pathlib import Path
import threading
import time
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch

from .classical import ClassicalConfig, _dilate, _erode, _largest_component, segment_dark_ridge
from .flat_field import FlatField, apply_flat_field, estimate_flat_field
from .heuristic_tuner import encode_png
from .mask_fit import MaskFitConfig, fill_narrow_holes, fit_mask, standard_initializations
from .segmentation_dataset import DEFAULT_DATASET_ROOT, SegmentationStore, make_sample_id
from .segmenter import IGNORE_LABEL, SegmentationModule, load_segmenter


DEFAULT_RECORDINGS = (
    Path("/store1/shared/all_data_raw/prj_aversion/2024-01-31/2024-01-31-02.h5"),
    Path("/store1/shared/all_data_raw/prj_aversion/2023-08-22/2023-08-22-01.h5"),
    Path("/store1/shared/all_data_raw/prj_aversion/2023-06-23/2023-06-23-01.h5"),
)
DATASET_PATH = "/img_nir"
DEFAULT_CHECKPOINT = Path("checkpoints/segmenter/best.ckpt")
FLAT_FIELD_SAMPLE_COUNT = 64
HOLE_FILL_RADIUS_PX = 8
UNCERTAIN_BAND = (0.2, 0.8)
PNG_WORM = 255
PNG_IGNORE = 128


def mask_to_png_values(mask: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Store labels (0/1/255) to browser PNG values (0/255/128)."""

    values = np.asarray(mask)
    out = np.zeros(values.shape, dtype=np.uint8)
    out[values == 1] = PNG_WORM
    out[values == IGNORE_LABEL] = PNG_IGNORE
    return out


def png_values_to_mask(values: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Browser PNG values back to store labels; anything mid-gray is ignore."""

    gray = np.asarray(values)
    out = np.zeros(gray.shape, dtype=np.uint8)
    out[gray >= 192] = 1
    out[(gray > 64) & (gray < 192)] = IGNORE_LABEL
    return out


def data_url(values: NDArray[np.uint8]) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(values)).decode("ascii")


def decode_mask_data_url(url: str, shape: tuple[int, int]) -> NDArray[np.uint8]:
    if not url.startswith("data:image/png;base64,"):
        raise ValueError("mask must be a base64 PNG data URL")
    raw = base64.b64decode(url.split(",", 1)[1])
    image = Image.open(io.BytesIO(raw)).convert("L")
    values = np.asarray(image, dtype=np.uint8)
    if values.shape != tuple(shape):
        raise ValueError(f"mask shape {values.shape} does not match frame {tuple(shape)}")
    return png_values_to_mask(values)


def probability_to_png(probability: NDArray[np.floating]) -> NDArray[np.uint8]:
    return np.clip(np.rint(np.asarray(probability) * 255.0), 0, 255).astype(np.uint8)


class RecordingSource:
    """One read-only recording with a lazily fitted, disk-cached flat field."""

    def __init__(self, path: Path, cache_dir: Path) -> None:
        self.path = Path(path)
        self.name = self.path.stem
        self.cache_dir = Path(cache_dir)
        self._lock = threading.Lock()
        self._handle: h5py.File | None = None
        self._field: FlatField | None = None
        with h5py.File(self.path, "r") as handle:
            dataset = handle[DATASET_PATH]
            if dataset.ndim != 3:
                raise ValueError(f"{self.path}: expected a [T,H,W] dataset")
            self.frame_count = int(dataset.shape[0])
            self.shape = (int(dataset.shape[1]), int(dataset.shape[2]))

    def _dataset(self) -> h5py.Dataset:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle[DATASET_PATH]

    def read(self, frame_index: int) -> NDArray[np.uint8]:
        if not 0 <= frame_index < self.frame_count:
            raise IndexError("frame index out of range")
        with self._lock:
            return np.asarray(self._dataset()[int(frame_index)], dtype=np.uint8)

    def flat_field(self) -> FlatField:
        if self._field is not None:
            return self._field
        cache = self.cache_dir / f"{self.name}.npz"
        if cache.exists():
            with np.load(cache) as archive:
                self._field = FlatField(
                    illumination=np.asarray(archive["illumination"], dtype=np.float64),
                    dark_level=float(archive["dark_level"]),
                    reference_level=float(archive["reference_level"]),
                    gain=np.asarray(archive["gain"], dtype=np.float64),
                )
            return self._field
        indices = np.linspace(0, self.frame_count - 1, min(FLAT_FIELD_SAMPLE_COUNT, self.frame_count), dtype=np.int64)
        with self._lock:
            dataset = self._dataset()
            calibration = np.stack([np.asarray(dataset[int(i)], dtype=np.uint8) for i in indices])
        field = estimate_flat_field(
            calibration, temporal_quantile=0.8, spatial_radius=31, smoothing_passes=2, min_gain=0.5, max_gain=2.5
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".npz.partial")
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle, illumination=field.illumination, dark_level=field.dark_level,
                reference_level=field.reference_level, gain=field.gain,
            )
        temporary.replace(cache)
        self._field = field
        return field

    def corrected(self, frame_index: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        raw = self.read(frame_index)
        corrected = apply_flat_field(raw, self.flat_field(), clip=(0.0, 255.0))
        return raw, np.clip(np.rint(corrected), 0, 255).astype(np.uint8)

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


class Proposer:
    """Network and classical proposals plus pipeline refinements."""

    def __init__(self, checkpoint: Path | None, device: str | None) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = checkpoint
        self.module: SegmentationModule | None = None
        self._lock = threading.Lock()
        if checkpoint is not None and checkpoint.exists():
            self.module = load_segmenter(checkpoint, self.device)

    def network(self, image: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        if self.module is None:
            return None
        with self._lock:
            return self.module.predict_probability(image)

    @staticmethod
    def classical(image: NDArray[np.uint8]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        segmentation = segment_dark_ridge(image, ClassicalConfig())
        filled, _ = fill_narrow_holes(segmentation.component, HOLE_FILL_RADIUS_PX)
        return filled, segmentation.high_threshold_mask

    @staticmethod
    def refine(mask: NDArray[np.uint8], method: str, device: torch.device) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        worm = mask == 1
        ignore = mask == IGNORE_LABEL
        info: dict[str, Any] = {}
        if method == "fill_holes":
            worm, added = fill_narrow_holes(worm, HOLE_FILL_RADIUS_PX, device=device)
            info["pixels_added"] = int(added)
        elif method == "largest_component":
            if worm.any():
                worm, _, count = _largest_component(worm)
                info["components_removed"] = int(count - 1)
        elif method == "dilate":
            worm = _dilate(worm, 1)
        elif method == "erode":
            worm = _erode(worm, 1)
        elif method == "mask_fit":
            if worm.sum() < 500:
                raise ValueError("mask too small for the tube fit")
            starts = standard_initializations(worm, config=MaskFitConfig())
            result = fit_mask(worm, starts, config=MaskFitConfig(), device=device)
            rendered = np.zeros_like(worm)
            crop = result.crop
            rendered[crop.y0 : crop.y1, crop.x0 : crop.x1] = result.rendered_hard_mask
            worm = rendered
            best = result.records[result.best_index]
            info.update(
                {
                    "iou_with_input": float(best["final_iou"]),
                    "body_length_px": float(result.body_length_px),
                    "width_px": float(result.width_px),
                    "points_in_fov": int(result.points_in_fov),
                }
            )
        else:
            raise ValueError(f"unknown refinement {method!r}")
        out = np.zeros(mask.shape, dtype=np.uint8)
        out[worm] = 1
        out[ignore & ~worm] = IGNORE_LABEL
        return out, info


class LabelState:
    def __init__(self, recordings: list[Path], store: SegmentationStore, proposer: Proposer) -> None:
        self.store = store
        self.proposer = proposer
        self.sources: dict[str, RecordingSource] = {}
        failures: dict[str, str] = {}
        for path in recordings:
            try:
                source = RecordingSource(path, store.root / "flat_fields")
            except (OSError, ValueError) as error:
                failures[str(path)] = str(error)
                continue
            self.sources[source.name] = source
        self.failures = failures
        if not self.sources:
            raise ValueError("no readable recordings")
        self.rng = np.random.default_rng()
        self.history: list[tuple[str, int]] = []

    def state(self) -> dict[str, Any]:
        labeled = {}
        for record in self.store.records():
            labeled[record.recording] = labeled.get(record.recording, 0) + 1
        return {
            "dataset_root": str(self.store.root),
            "counts": self.store.counts(),
            "recordings": [
                {"name": name, "frame_count": source.frame_count, "labeled": labeled.get(name, 0)}
                for name, source in self.sources.items()
            ],
            "unreadable_recordings": self.failures,
            "checkpoint": None if self.proposer.module is None else str(self.proposer.checkpoint),
            "device": str(self.proposer.device),
        }

    def frame(self, recording: str, frame_index: int) -> dict[str, Any]:
        source = self.sources[recording]
        raw, image = source.corrected(frame_index)
        started = time.perf_counter()
        probability = self.proposer.network(image)
        classical, raw_threshold = self.proposer.classical(image)
        sample_id = make_sample_id(recording, frame_index)
        record = self.store.get(sample_id)
        existing = None
        if record is not None:
            _, mask, _ = self.store.load(sample_id)
            existing = data_url(mask_to_png_values(mask))
        proposals: dict[str, str | None] = {
            "classical": data_url(mask_to_png_values(classical.astype(np.uint8))),
            "raw_threshold": data_url(mask_to_png_values(raw_threshold.astype(np.uint8))),
            "network": None,
            "network_probability": None,
        }
        uncertain_fraction = None
        if probability is not None:
            proposals["network"] = data_url(mask_to_png_values((probability >= 0.5).astype(np.uint8)))
            proposals["network_probability"] = data_url(probability_to_png(probability))
            uncertain_fraction = float(((probability > UNCERTAIN_BAND[0]) & (probability < UNCERTAIN_BAND[1])).mean())
        self.history.append((recording, frame_index))
        del self.history[:-200]
        return {
            "recording": recording,
            "frame_index": int(frame_index),
            "frame_count": source.frame_count,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "image": data_url(image),
            "image_raw": data_url(raw),
            "existing_mask": existing,
            "existing_record": None if record is None else asdict(record),
            "proposals": proposals,
            "network_uncertain_fraction": uncertain_fraction,
            "proposal_seconds": time.perf_counter() - started,
        }

    def next_frame(self, mode: str, recording: str | None, current_index: int | None, stride: int, candidates: int) -> dict[str, Any]:
        names = list(self.sources)
        if mode == "sequential":
            if recording is None or recording not in self.sources:
                recording = names[0]
            source = self.sources[recording]
            index = (0 if current_index is None else int(current_index) + max(1, stride)) % source.frame_count
            return {"recording": recording, "frame_index": index}
        pool = [recording] if recording in self.sources else names

        def random_unlabeled() -> tuple[str, int]:
            for _ in range(100):
                name = pool[int(self.rng.integers(len(pool)))]
                index = int(self.rng.integers(self.sources[name].frame_count))
                if not self.store.has(name, index):
                    return name, index
            return name, index

        if mode == "uncertain" and self.proposer.module is not None:
            best: tuple[float, str, int] | None = None
            for _ in range(max(1, candidates)):
                name, index = random_unlabeled()
                _, image = self.sources[name].corrected(index)
                probability = self.proposer.network(image)
                assert probability is not None
                score = float(((probability > UNCERTAIN_BAND[0]) & (probability < UNCERTAIN_BAND[1])).sum())
                if best is None or score > best[0]:
                    best = (score, name, index)
            assert best is not None
            return {"recording": best[1], "frame_index": best[2], "uncertain_pixels": best[0]}
        name, index = random_unlabeled()
        return {"recording": name, "frame_index": index}

    def save(self, recording: str, frame_index: int, mask_url: str, label_source: str) -> dict[str, Any]:
        source = self.sources[recording]
        raw, image = source.corrected(frame_index)
        mask = decode_mask_data_url(mask_url, image.shape)
        if not (mask == 1).any():
            raise ValueError("refusing to save a label with no worm pixels; use ignore for empty frames")
        record = self.store.save(
            recording, frame_index, image, mask, image_raw=raw, source_path=str(source.path),
            label_source=label_source, flat_fielded=True,
        )
        return {"record": asdict(record), "counts": self.store.counts()}

    def refine(self, recording: str, frame_index: int, mask_url: str, method: str) -> dict[str, Any]:
        source = self.sources[recording]
        mask = decode_mask_data_url(mask_url, source.shape)
        refined, info = Proposer.refine(mask, method, self.proposer.device)
        return {"mask": data_url(mask_to_png_values(refined)), "info": info}

    def close(self) -> None:
        for source in self.sources.values():
            source.close()


def _static_bytes(name: str) -> bytes:
    return importlib.resources.files("worm_pose_gen.label_app_ui").joinpath(name).read_bytes()


class LabelHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: LabelState) -> None:
        super().__init__(address, LabelRequestHandler)
        self.state = state


class LabelRequestHandler(BaseHTTPRequestHandler):
    server: LabelHTTPServer

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
        if length <= 0 or length > 64 * 1024 * 1024:
            raise ValueError("request body missing or too large")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
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
            elif parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                recording = query.get("recording", [""])[0]
                if recording not in state.sources:
                    raise ValueError("unknown recording")
                self._send_json(state.frame(recording, int(query.get("index", ["0"])[0])))
            elif parsed.path == "/api/samples":
                self._send_json({"samples": [asdict(r) for r in state.store.records()]})
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
            if self.path == "/api/save":
                self._send_json(
                    state.save(
                        str(payload["recording"]), int(payload["frame_index"]),
                        str(payload["mask"]), str(payload.get("label_source", "manual")),
                    )
                )
            elif self.path == "/api/refine":
                self._send_json(
                    state.refine(
                        str(payload["recording"]), int(payload["frame_index"]),
                        str(payload["mask"]), str(payload["method"]),
                    )
                )
            elif self.path == "/api/next":
                self._send_json(
                    state.next_frame(
                        str(payload.get("mode", "random")), payload.get("recording"),
                        payload.get("current_index"), int(payload.get("stride", 200)),
                        int(payload.get("candidates", 6)),
                    )
                )
            elif self.path == "/api/delete":
                self._send_json({"deleted": state.store.delete(str(payload["sample_id"])), "counts": state.store.counts()})
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, IndexError, KeyError, RuntimeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # noqa: BLE001 - report to the browser instead of dropping the socket
            self._send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def create_server(state: LabelState, host: str = "127.0.0.1", port: int = 8767) -> LabelHTTPServer:
    return LabelHTTPServer((host, port), state)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", action="append", type=Path, dest="recordings")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store = SegmentationStore(args.dataset_root)
    proposer = Proposer(args.checkpoint, args.device)
    state = LabelState(list(args.recordings or DEFAULT_RECORDINGS), store, proposer)
    server = create_server(state, args.host, args.port)
    summary = state.state()
    print(f"worm labeler at http://{args.host}:{args.port}/")
    print(f"dataset root {summary['dataset_root']} counts {summary['counts']}")
    print(f"network checkpoint: {summary['checkpoint'] or 'none (classical proposals only)'}")
    for path, error in summary["unreadable_recordings"].items():
        print(f"skipped unreadable recording {path}: {error}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
