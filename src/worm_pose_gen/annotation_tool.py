"""Small local web application for blind Tier-A centerline annotation.

The server binds to localhost, reads source HDF5 recordings without writing to
them, and atomically stores annotations in a separate JSON file.  It deliberately
does not expose proxy or model overlays.  The default worklist is a staged
single-annotator protocol: 30 primary frames and 10 delayed blind repeats.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.resources
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse
import uuid

import numpy as np

from .annotation import SCHEMA_VERSION, validate_annotation
from .data import HDF5FrameSource


TOOL_NAME = "worm-pose-gen-annotation-web"
TOOL_VERSION = "0.1.0"
SESSION_SCHEMA_VERSION = "1.0.0"
DEFAULT_SELECTION_SEED = 20260819
DEFAULT_PRIMARY_COUNT = 30
DEFAULT_REPEAT_COUNT = 10
DEFAULT_REPEAT_DELAY_HOURS = 168.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a nonempty ISO-8601 string")
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be valid ISO-8601") from error
    if result.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def _stable_order(rows: Sequence[Mapping[str, Any]], seed: int) -> list[Mapping[str, Any]]:
    def key(row: Mapping[str, Any]) -> str:
        value = f"{seed}:{row['sample_id']}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    return sorted(rows, key=key)


def _allocate(total: int, groups: int, *, require_each: bool = True) -> list[int]:
    if groups < 1 or total < 0:
        raise ValueError("allocation counts must be nonnegative with at least one group")
    if require_each and total < groups:
        raise ValueError("the worklist needs at least one primary frame per recording")
    base, extra = divmod(total, groups)
    return [base + (position < extra) for position in range(groups)]


def _balanced_recording_selection(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[Mapping[str, Any]]:
    strata = (
        "proxy_difficult",
        "proxy_easy",
        "uniform_coverage",
        "double_annotation_temporal_window",
    )
    buckets = {
        stratum: list(_stable_order(
            [row for row in rows if row.get("selection_stratum") == stratum], seed
        ))
        for stratum in strata
    }
    remaining = list(_stable_order(
        [row for row in rows if row.get("selection_stratum") not in strata], seed
    ))
    selected: list[Mapping[str, Any]] = []
    position = 0
    while len(selected) < count:
        stratum = strata[position % len(strata)]
        position += 1
        if buckets[stratum]:
            selected.append(buckets[stratum].pop(0))
            continue
        available = next((bucket for bucket in buckets.values() if bucket), None)
        if available:
            selected.append(available.pop(0))
        elif remaining:
            selected.append(remaining.pop(0))
        else:
            raise ValueError("manifest does not contain enough records for the worklist")
    return selected


def build_single_annotator_worklist(
    records: Sequence[Mapping[str, Any]],
    *,
    primary_count: int = DEFAULT_PRIMARY_COUNT,
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
) -> list[dict[str, str]]:
    """Build a deterministic session-balanced primary/repeat worklist."""

    if not records:
        raise ValueError("annotation manifest contains no records")
    if repeat_count < 0 or repeat_count > primary_count:
        raise ValueError("repeat_count must lie between zero and primary_count")
    recordings = list(dict.fromkeys(str(row["recording"]) for row in records))
    primary_allocation = _allocate(primary_count, len(recordings))
    primary: list[Mapping[str, Any]] = []
    for position, (recording, count) in enumerate(zip(recordings, primary_allocation, strict=True)):
        candidates = [row for row in records if row["recording"] == recording]
        primary.extend(_balanced_recording_selection(candidates, count, seed + position))

    repeat: list[Mapping[str, Any]] = []
    if repeat_count:
        repeat_allocation = _allocate(repeat_count, len(recordings), require_each=False)
        for position, (recording, count) in enumerate(
            zip(recordings, repeat_allocation, strict=True)
        ):
            candidates = [row for row in primary if row["recording"] == recording]
            repeat.extend(_balanced_recording_selection(
                candidates, count, seed + 10_000 + position
            ))

    return [
        *({"sample_id": str(row["sample_id"]), "annotation_pass": "primary"}
          for row in primary),
        *({"sample_id": str(row["sample_id"]), "annotation_pass": "repeat"}
          for row in repeat),
    ]


@dataclass(frozen=True)
class AnnotationProtocol:
    primary_count: int = DEFAULT_PRIMARY_COUNT
    repeat_count: int = DEFAULT_REPEAT_COUNT
    repeat_delay_hours: float = DEFAULT_REPEAT_DELAY_HOURS
    selection_seed: int = DEFAULT_SELECTION_SEED

    def __post_init__(self) -> None:
        if self.primary_count < 1:
            raise ValueError("primary_count must be positive")
        if self.repeat_count < 0 or self.repeat_count > self.primary_count:
            raise ValueError("repeat_count must lie between zero and primary_count")
        if not math.isfinite(self.repeat_delay_hours) or self.repeat_delay_hours < 0:
            raise ValueError("repeat_delay_hours must be finite and nonnegative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_count": self.primary_count,
            "repeat_count": self.repeat_count,
            "repeat_delay_hours": self.repeat_delay_hours,
            "selection_seed": self.selection_seed,
            "purpose": (
                "single-annotator development benchmark plus delayed blind "
                "intra-annotator repeatability; not inter-annotator agreement"
            ),
        }


class AnnotationSession:
    """Validated, resumable, append-only annotation session."""

    def __init__(
        self,
        manifest_path: Path,
        output_path: Path,
        annotator_id: str,
        protocol: AnnotationProtocol,
        *,
        now: datetime | None = None,
    ) -> None:
        if not annotator_id.strip():
            raise ValueError("annotator_id must be nonempty")
        self.manifest_path = manifest_path.resolve(strict=True)
        self.output_path = output_path.resolve(strict=False)
        self.annotator_id = annotator_id.strip()
        self.protocol = protocol
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("protected_holdout_opened") is not False:
            raise ValueError("the annotation manifest must keep the protected holdout closed")
        raw_records = manifest.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("manifest records must be a nonempty list")
        self.manifest = manifest
        self.records = {str(row["sample_id"]): row for row in raw_records}
        if len(self.records) != len(raw_records):
            raise ValueError("manifest sample_id values must be unique")
        self.worklist = build_single_annotator_worklist(
            raw_records,
            primary_count=protocol.primary_count,
            repeat_count=protocol.repeat_count,
            seed=protocol.selection_seed,
        )
        self._lock = threading.Lock()
        if self.output_path.exists():
            self.payload = json.loads(self.output_path.read_text())
            self._validate_existing_session()
        else:
            self.payload = {
                "session_schema_version": SESSION_SCHEMA_VERSION,
                "annotation_schema_version": SCHEMA_VERSION,
                "manifest_path": str(self.manifest_path),
                "manifest_records_sha256": manifest.get("records_sha256"),
                "annotator_id": self.annotator_id,
                "tool_name": TOOL_NAME,
                "tool_version": TOOL_VERSION,
                "created_at_utc": utc_text(now or utc_now()),
                "protocol": protocol.as_dict(),
                "worklist": self.worklist,
                "annotations": [],
            }
            self._write()

    def _validate_existing_session(self) -> None:
        expected = {
            "session_schema_version": SESSION_SCHEMA_VERSION,
            "annotation_schema_version": SCHEMA_VERSION,
            "manifest_records_sha256": self.manifest.get("records_sha256"),
            "annotator_id": self.annotator_id,
            "tool_name": TOOL_NAME,
            "protocol": self.protocol.as_dict(),
            "worklist": self.worklist,
        }
        for key, value in expected.items():
            if self.payload.get(key) != value:
                raise ValueError(f"existing annotation session mismatches {key}")
        annotations = self.payload.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError("existing annotation session has invalid annotations")
        seen: set[tuple[str, str]] = set()
        allowed = {(row["sample_id"], row["annotation_pass"]) for row in self.worklist}
        for raw in annotations:
            key = (raw.get("sample_id"), raw.get("annotation_pass"))
            if key not in allowed or key in seen:
                raise ValueError("existing annotation session has an unexpected or duplicate task")
            seen.add(key)
            row = self.records[str(raw["sample_id"])]
            validate_annotation(
                raw,
                image_height=int(row["image_height"]),
                image_width=int(row["image_width"]),
            )

    def _write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(f".{self.output_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(self.payload, indent=2) + "\n")
        os.replace(temporary, self.output_path)

    def _annotation_index(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        return {
            (str(row["sample_id"]), str(row["annotation_pass"])): row
            for row in self.payload["annotations"]
        }

    def _task_status(
        self, task: Mapping[str, str], annotations: Mapping[tuple[str, str], Mapping[str, Any]], now: datetime
    ) -> tuple[str, str | None]:
        key = (task["sample_id"], task["annotation_pass"])
        if key in annotations:
            return "complete", None
        if task["annotation_pass"] == "primary":
            return "ready", None
        primary = annotations.get((task["sample_id"], "primary"))
        if primary is None:
            return "waiting_for_primary", None
        available = parse_utc(str(primary["completed_at_utc"])) + timedelta(
            hours=self.protocol.repeat_delay_hours
        )
        if now < available:
            return "waiting_until", utc_text(available)
        return "ready", utc_text(available)

    def state(self, *, now: datetime | None = None) -> dict[str, Any]:
        current = (now or utc_now()).astimezone(timezone.utc)
        annotations = self._annotation_index()
        tasks: list[dict[str, Any]] = []
        next_times: list[str] = []
        for task in self.worklist:
            status, available_at = self._task_status(task, annotations, current)
            if status == "waiting_until" and available_at:
                next_times.append(available_at)
            row = self.records[task["sample_id"]]
            tasks.append({
                **task,
                "status": status,
                "available_at_utc": available_at,
                "recording": row["recording"],
                "frame_index": row["frame_index"],
                "selection_stratum": row["selection_stratum"],
                "difficulty_hints": row.get("difficulty_hints", []),
                "temporal_window_indices": row["temporal_window_indices"],
                "image_height": row["image_height"],
                "image_width": row["image_width"],
            })
        primary_complete = sum(
            task["annotation_pass"] == "primary" and task["status"] == "complete"
            for task in tasks
        )
        repeat_complete = sum(
            task["annotation_pass"] == "repeat" and task["status"] == "complete"
            for task in tasks
        )
        return {
            "annotator_id": self.annotator_id,
            "output_path": str(self.output_path),
            "protocol": self.protocol.as_dict(),
            "primary_complete": primary_complete,
            "repeat_complete": repeat_complete,
            "total_complete": primary_complete + repeat_complete,
            "total_tasks": len(tasks),
            "next_available_at_utc": min(next_times) if next_times else None,
            "tasks": tasks,
        }

    def save_annotation(
        self, request: Mapping[str, Any], *, now: datetime | None = None
    ) -> dict[str, Any]:
        current = (now or utc_now()).astimezone(timezone.utc)
        sample_id = str(request.get("sample_id", ""))
        annotation_pass = str(request.get("annotation_pass", ""))
        task = next((
            item for item in self.worklist
            if item["sample_id"] == sample_id and item["annotation_pass"] == annotation_pass
        ), None)
        if task is None:
            raise ValueError("sample/pass is not part of this worklist")
        with self._lock:
            annotations = self._annotation_index()
            status, available_at = self._task_status(task, annotations, current)
            if status == "complete":
                raise ValueError("this annotation task is already locked")
            if status != "ready":
                detail = f" until {available_at}" if available_at else ""
                raise ValueError(f"this repeat annotation is not available{detail}")

            row = self.records[sample_id]
            trace_state = str(request.get("trace_state", "complete"))
            if trace_state not in {"complete", "truncated", "not_identifiable"}:
                raise ValueError("invalid trace_state")
            vertices = request.get("vertices", [])
            if not isinstance(vertices, list):
                raise ValueError("vertices must be a list")
            outside_start = bool(request.get("outside_fov_at_start", False))
            outside_end = bool(request.get("outside_fov_at_end", False))
            if trace_state == "not_identifiable":
                vertices = []
                outside_start = outside_end = False
            elif outside_start or outside_end:
                trace_state = "truncated"
                if outside_start:
                    vertices = [{"xy": [None, None], "support_state": "outside_fov"}, *vertices]
                if outside_end:
                    vertices = [*vertices, {"xy": [None, None], "support_state": "outside_fov"}]
            elif trace_state == "truncated":
                raise ValueError("a truncated trace must specify which endpoint leaves the FOV")

            started_at = parse_utc(str(request.get("started_at_utc", "")))
            if started_at > current + timedelta(minutes=1):
                raise ValueError("started_at_utc cannot be in the future")
            width = request.get("worm_width_px")
            if width in ("", None):
                width = None
            else:
                try:
                    width = float(width)
                except (TypeError, ValueError) as error:
                    raise ValueError("worm_width_px must be numeric") from error
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "annotation_id": str(uuid.uuid4()),
                "annotator_id": self.annotator_id,
                "tool_name": TOOL_NAME,
                "tool_version": TOOL_VERSION,
                "started_at_utc": utc_text(started_at),
                "completed_at_utc": utc_text(current),
                "configured_source_path": row["configured_source_path"],
                "resolved_source_path": row["resolved_source_path"],
                "source_size_bytes": row["source_size_bytes"],
                "source_mtime_ns": row["source_mtime_ns"],
                "source_dataset_path": row["source_dataset_path"],
                "frame_index": row["frame_index"],
                "split_role": row["split_role"],
                "selection_stratum": row["selection_stratum"],
                "annotation_view": "temporal_window",
                "temporal_window_indices": row["temporal_window_indices"],
                "annotation_overlays": [],
                "parent_annotation_id": None,
                "timestamp_raw": row["timestamp_raw"],
                "timestamp_mapping": row["timestamp_mapping"],
                "head_tail_state": request.get("head_tail_state", "ambiguous"),
                "trace_state": trace_state,
                "worm_width_px": width,
                "difficulty": request.get("difficulty", []),
                "vertices": vertices,
                "annotation_pass": annotation_pass,
                "repeat_of_annotation_id": (
                    annotations[(sample_id, "primary")]["annotation_id"]
                    if annotation_pass == "repeat" else None
                ),
                "single_annotator_protocol": True,
            }
            validate_annotation(
                record,
                image_height=int(row["image_height"]),
                image_width=int(row["image_width"]),
            )
            self.payload["annotations"].append(record)
            self._write()
            return record


class FrameRepository:
    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        self.records = records
        self.sources: dict[str, HDF5FrameSource] = {}
        self._lock = threading.Lock()

    def read(self, sample_id: str, frame_index: int) -> tuple[np.ndarray, float, float]:
        if sample_id not in self.records:
            raise ValueError("unknown sample_id")
        row = self.records[sample_id]
        if frame_index not in row["temporal_window_indices"]:
            raise ValueError("frame lies outside the declared temporal context")
        recording = str(row["recording"])
        with self._lock:
            if recording not in self.sources:
                self.sources[recording] = HDF5FrameSource(
                    PROJECT_ROOT / row["configured_source_path"],
                    row["source_dataset_path"],
                    expected_ndim=3,
                    max_frames_per_read=11,
                )
            image = np.asarray(self.sources[recording].read_frame(frame_index))
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("annotation UI currently requires grayscale uint8 frames")
        low, high = (float(value) for value in np.percentile(image, (1, 99)))
        return image, low, high

    def close(self) -> None:
        for source in self.sources.values():
            source.close()


def _static_bytes(name: str) -> bytes:
    root = importlib.resources.files("worm_pose_gen.annotation_ui")
    return root.joinpath(name).read_bytes()


class AnnotationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], session: AnnotationSession, frames: FrameRepository
    ) -> None:
        super().__init__(address, AnnotationRequestHandler)
        self.annotation_session = session
        self.frame_repository = frames


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    server: AnnotationHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
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

    def _error(self, error: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": str(error)}, status)

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
                self._send_json(self.server.annotation_session.state())
            elif parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                sample_id = query.get("sample_id", [""])[0]
                frame_text = query.get("frame_index", [""])[0]
                try:
                    frame_index = int(frame_text)
                except ValueError as error:
                    raise ValueError("frame_index must be an integer") from error
                image, low, high = self.server.frame_repository.read(sample_id, frame_index)
                content = image.tobytes(order="C")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Image-Width", str(image.shape[1]))
                self.send_header("X-Image-Height", str(image.shape[0]))
                self.send_header("X-Display-Low", str(low))
                self.send_header("X-Display-High", str(high))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, OSError) as error:
            self._error(error)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotations":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 2_000_000:
                raise ValueError("annotation request size is invalid")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("annotation request must be an object")
            record = self.server.annotation_session.save_annotation(request)
            self._send_json({"saved": record["annotation_id"]}, HTTPStatus.CREATED)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._error(error)


def create_server(
    session: AnnotationSession, *, host: str = "127.0.0.1", port: int = 8765
) -> AnnotationHTTPServer:
    return AnnotationHTTPServer((host, port), session, FrameRepository(session.records))
