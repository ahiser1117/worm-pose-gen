"""Deterministic, leakage-safe selection for the EXP-001 Tier-A tranche."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SELECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SelectionRecord:
    sample_id: str
    recording: str
    frame_index: int
    selection_stratum: str
    difficulty_hints: tuple[str, ...]
    double_annotate: bool
    temporal_window_id: str | None
    temporal_window_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "recording": self.recording,
            "frame_index": self.frame_index,
            "selection_stratum": self.selection_stratum,
            "difficulty_hints": list(self.difficulty_hints),
            "double_annotate": self.double_annotate,
            "temporal_window_id": self.temporal_window_id,
            "temporal_window_indices": list(self.temporal_window_indices),
        }


def _uniform_candidates(frame_count: int, count: int) -> list[int]:
    if count <= 0:
        return []
    return np.linspace(5, frame_count - 6, count, dtype=np.int64).tolist()


def select_recording_frames(
    *,
    recording: str,
    frame_count: int,
    target_count: int,
    proxy_rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> list[SelectionRecord]:
    """Select two complete 11-frame double-label windows plus diverse frames."""

    if frame_count < 33 or target_count < 22:
        raise ValueError("selection requires frame_count>=33 and target_count>=22")
    if any(not 0 <= int(row["frame_index"]) < frame_count for row in proxy_rows):
        raise ValueError("proxy frame index lies outside the recording")
    # The random offset avoids always choosing an exact duration fraction while
    # retaining deterministic, model-independent selection.
    generator = np.random.default_rng(seed)
    jitter = generator.integers(-max(frame_count // 100, 1), max(frame_count // 100, 1) + 1, 2)
    centers = [frame_count // 3 + int(jitter[0]), 2 * frame_count // 3 + int(jitter[1])]
    centers = [min(max(center, 5), frame_count - 6) for center in centers]

    selected: dict[int, SelectionRecord] = {}
    for number, center in enumerate(centers, start=1):
        indices = tuple(range(center - 5, center + 6))
        window_id = f"{recording}-window-{number}"
        for frame_index in indices:
            selected[frame_index] = SelectionRecord(
                sample_id=f"{recording}-f{frame_index:06d}",
                recording=recording,
                frame_index=frame_index,
                selection_stratum="double_annotation_temporal_window",
                difficulty_hints=(),
                double_annotate=True,
                temporal_window_id=window_id,
                temporal_window_indices=indices,
            )

    # Proxy outcome is used only as a difficulty-enrichment hint.  Neither the
    # proxy centerline nor a model overlay is exposed to primary annotators.
    ordered_proxy = sorted(
        proxy_rows,
        key=lambda row: (bool(row["accepted"]), int(row["frame_index"])),
    )
    for row in ordered_proxy:
        frame_index = int(row["frame_index"])
        if frame_index in selected:
            continue
        accepted = bool(row["accepted"])
        reasons = tuple(sorted(str(value) for value in row.get("rejection_reasons", ())))
        selected[frame_index] = SelectionRecord(
            sample_id=f"{recording}-f{frame_index:06d}",
            recording=recording,
            frame_index=frame_index,
            selection_stratum="proxy_easy" if accepted else "proxy_difficult",
            difficulty_hints=() if accepted else reasons,
            double_annotate=False,
            temporal_window_id=None,
            temporal_window_indices=tuple(range(max(0, frame_index - 5), min(frame_count, frame_index + 6))),
        )
        if len(selected) == target_count:
            break

    candidate_count = max(frame_count // 4, target_count * 8)
    for frame_index in _uniform_candidates(frame_count, candidate_count):
        if frame_index in selected:
            continue
        selected[frame_index] = SelectionRecord(
            sample_id=f"{recording}-f{frame_index:06d}",
            recording=recording,
            frame_index=frame_index,
            selection_stratum="uniform_coverage",
            difficulty_hints=(),
            double_annotate=False,
            temporal_window_id=None,
            temporal_window_indices=tuple(range(frame_index - 5, frame_index + 6)),
        )
        if len(selected) == target_count:
            break
    if len(selected) != target_count:
        raise RuntimeError(f"could select only {len(selected)} of {target_count} frames")
    return sorted(selected.values(), key=lambda value: value.frame_index)


def proxy_rows(proxy_path: Path, recording: str) -> list[dict[str, Any]]:
    """Read only screening outcomes/indices, never proxy centerlines."""

    rows: list[dict[str, Any]] = []
    with h5py.File(proxy_path, "r") as handle:
        group = handle[recording]
        frame_index = np.asarray(group["sample_frame_index"])
        accepted = np.asarray(group["accepted"])
        reasons = np.asarray(group["rejection_reasons"]).astype(str)
        for index in range(len(frame_index)):
            decoded = tuple(value for value in reasons[index].split(";") if value)
            rows.append(
                {
                    "frame_index": int(frame_index[index]),
                    "accepted": bool(accepted[index]),
                    "rejection_reasons": decoded,
                }
            )
    return rows


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
